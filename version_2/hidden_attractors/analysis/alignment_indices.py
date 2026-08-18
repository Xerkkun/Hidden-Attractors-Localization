"""Smaller and generalized alignment indices for integer-order dynamics.

Stability: experimental
    The integer q=1 contracts are public experimental APIs; signatures may
    gain optional controls while preserving declared layouts and result types.

The public history convention is ``(n_samples, n_vectors, dimension)``.
Instantaneous deviation matrices, including ``initial_deviations`` and
``final_deviations`` in :class:`AlignmentIndexResult`, use deviation vectors
as columns and therefore have shape ``(dimension, n_vectors)``.

The routines in this module deliberately do not classify an orbit as chaotic.
In particular, alignment indices may also decay for dissipative non-chaotic
motions such as convergence to a stable fixed point or a limit cycle.  They are
finite-time diagnostics that must be interpreted together with boundedness,
convergence and other dynamical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from .._rhs import bind_rhs
from ._system_order import declared_system_order as _declared_system_order

try:  # Numba is a declared dependency, but import failure remains recoverable.
    from numba import njit

    NUMBA_AVAILABLE = True
except (ImportError, OSError):  # pragma: no cover - exercised without Numba
    NUMBA_AVAILABLE = False
    njit = None  # type: ignore[assignment]


_ALIGNMENT_REFERENCES: tuple[str, ...] = (
    "Skokos 2001, Alignment indices, doi:10.1088/0305-4470/34/47/309",
    "Skokos, Bountis & Antonopoulos 2007, GALI, doi:10.1016/j.physd.2007.04.004",
    "Manda, Hillebrand & Skokos 2025, multi-particle GALI, "
    "doi:10.1016/j.cnsns.2025.108635",
    "Rolim Sales, Leonel & Antonopoulos 2026, SVD rates for LDI/GALI/SALI, "
    "doi:10.1016/j.chaos.2026.117884",
)

_BASE_WARNINGS: tuple[str, ...] = (
    "This implementation is restricted to q=1 memoryless integer-order dynamics; "
    "it is not a fractional-memory variational method.",
    "SALI/GALI values are finite-time diagnostics and are not, by themselves, a "
    "universal classifier of chaos or hiddenness.",
    "In dissipative systems, alignment can also occur near stable fixed points, "
    "limit cycles, and other non-chaotic invariant sets; decay alone must not be "
    "reported as proof of chaos.",
    "For chaotic maps the SALI asymptotic rate uses the two largest Lyapunov "
    "exponents even when the second exponent is negative.",
)

_MULTI_PARTICLE_WARNINGS: tuple[str, ...] = (
    "The multi-particle method uses finite neighboring trajectories and is "
    "sensitive to deviation_size, reinjection interval, solver error, and roundoff.",
    "The published double-precision guidance deviation_size approximately "
    "sqrt(machine epsilon) and renormalization_time no greater than one was "
    "validated primarily on Hamiltonian examples; it is not a universal error "
    "guarantee for every dissipative or non-Hamiltonian model.",
)

_EPS = float(np.finfo(np.float64).eps)
_LOG_MIN_SUBNORMAL = float(np.log(np.nextafter(0.0, 1.0)))
_ORDER_TOLERANCE = 1.0e-9


if NUMBA_AVAILABLE:

    @njit(cache=False, fastmath=False, nogil=True)  # type: ignore[misc]
    def _householder_qr_log_volume_numba(matrix: np.ndarray) -> tuple[float, float, bool]:
        """Return volume, log-volume and censoring using temporary Householder QR."""

        work = matrix.copy()
        n_rows, n_columns = work.shape
        log_volume = 0.0

        for pivot in range(n_columns):
            diagonal_magnitude = 0.0
            for row in range(pivot, n_rows):
                diagonal_magnitude = np.hypot(diagonal_magnitude, work[row, pivot])
            if diagonal_magnitude == 0.0:
                return 0.0, -np.inf, True

            alpha = -diagonal_magnitude
            if work[pivot, pivot] < 0.0:
                alpha = diagonal_magnitude
            work[pivot, pivot] -= alpha
            log_volume += np.log(diagonal_magnitude)
            if pivot + 1 < n_columns:
                reflector_norm = 0.0
                for row in range(pivot, n_rows):
                    reflector_norm = np.hypot(reflector_norm, work[row, pivot])
                if reflector_norm == 0.0:
                    return 0.0, -np.inf, True
                for row in range(pivot, n_rows):
                    work[row, pivot] /= reflector_norm
                for column in range(pivot + 1, n_columns):
                    inner = 0.0
                    for row in range(pivot, n_rows):
                        inner += work[row, pivot] * work[row, column]
                    for row in range(pivot, n_rows):
                        work[row, column] -= 2.0 * inner * work[row, pivot]

        if log_volume < _LOG_MIN_SUBNORMAL:
            return 0.0, log_volume, True
        volume = np.exp(log_volume)
        if volume == 0.0:
            return 0.0, log_volume, True
        return volume, log_volume, False


@dataclass(frozen=True)
class AlignmentIndexResult:
    """Finite-time SALI/GALI result without an orbit classification.

    ``gali`` and ``log_gali`` have shape ``(n_samples, n_orders)`` and
    ``censored`` has exactly the same shape.  A censored cell means that the
    linear volume was set to zero because it underflowed or the matrix was
    numerically rank deficient.  Its finite ``log_gali`` value is retained on
    underflow; exact/numerical rank deficiency is represented by ``-inf``.
    """

    coordinates: np.ndarray
    sali: np.ndarray
    log_sali: np.ndarray
    gali_orders: np.ndarray
    gali: np.ndarray
    log_gali: np.ndarray
    censored: np.ndarray
    status: str
    final_state: np.ndarray | None
    sampled_states: np.ndarray | None
    initial_deviations: np.ndarray
    final_deviations: np.ndarray
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    method_id: str = "alignment_indices_from_tangent_history"
    system_kind: str = "precomputed"
    coordinate_kind: str = "sample"
    evolution_method: str = "precomputed"
    backend: str = "numpy"
    volume_method: str = "svd_product"
    derivative_model: str = "integer"
    q: float = 1.0
    finite_time_local: bool = True
    normalization: str = "independent_l2"
    orthonormalization: str = "none_during_evolution"
    reference_ids: tuple[str, ...] = field(default_factory=lambda: _ALIGNMENT_REFERENCES)
    methodological_warnings: tuple[str, ...] = field(default_factory=lambda: _BASE_WARNINGS)


def _require_integer_order(q: float | Sequence[float] | np.ndarray, routine: str) -> None:
    values = np.asarray(q, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)) or not np.all(
        np.abs(values - 1.0) <= _ORDER_TOLERANCE
    ):
        raise ValueError(
            f"{routine} is valid only for q=1 integer-order dynamics; received q={q!r}."
        )


def _normalize_columns(vectors: np.ndarray, *, name: str = "deviations") -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError(f"{name} contains a zero or non-finite deviation vector.")
    return np.ascontiguousarray(matrix / norms)


_NUMBA_OPERATIONAL: bool | None = None


def _numba_is_operational() -> bool:
    global _NUMBA_OPERATIONAL
    if _NUMBA_OPERATIONAL is None:
        if not NUMBA_AVAILABLE:
            _NUMBA_OPERATIONAL = False
        else:
            try:
                _householder_qr_log_volume_numba(np.eye(2, dtype=float))
            except Exception:  # pragma: no cover - depends on local Numba/LAPACK
                _NUMBA_OPERATIONAL = False
            else:
                _NUMBA_OPERATIONAL = True
    return bool(_NUMBA_OPERATIONAL)


def _resolve_backend(backend: str) -> str:
    requested = str(backend).strip().lower()
    if requested not in {"auto", "numpy", "numba"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'numba'.")
    if requested == "numpy":
        return "numpy"
    if requested == "numba":
        if not _numba_is_operational():
            raise RuntimeError("The Numba alignment-volume backend is not available.")
        return "numba"
    return "numba" if _numba_is_operational() else "numpy"


def _svd_volume_metrics(matrix: np.ndarray) -> tuple[float, float, bool]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        raise ValueError("At least one deviation vector is required.")
    if singular_values[-1] == 0.0:
        return 0.0, float("-inf"), True
    log_volume = float(np.sum(np.log(singular_values)))
    if log_volume < _LOG_MIN_SUBNORMAL:
        return 0.0, log_volume, True
    volume = float(np.exp(log_volume))
    return volume, log_volume, bool(volume == 0.0)


def _volume_metrics(matrix: np.ndarray, backend: str) -> tuple[float, float, bool]:
    if backend == "numpy":
        return _svd_volume_metrics(matrix)
    return _householder_qr_log_volume_numba(np.ascontiguousarray(matrix, dtype=float))


def smaller_alignment_index(
    deviations: np.ndarray | Sequence[Sequence[float]],
    second: np.ndarray | Sequence[float] | None = None,
    *,
    normalize: bool = True,
) -> float:
    """Compute SALI from two deviation vectors.

    With one argument, vectors are columns of a ``(dimension, n_vectors)``
    matrix and the first two are used.  Alternatively, pass the two vectors as
    separate arguments.
    """

    if second is None:
        matrix = np.asarray(deviations, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            raise ValueError("deviations must have shape (dimension, at least 2).")
        pair = matrix[:, :2]
    else:
        first_vector = np.asarray(deviations, dtype=float)
        second_vector = np.asarray(second, dtype=float)
        if first_vector.ndim != 1 or second_vector.ndim != 1 or first_vector.shape != second_vector.shape:
            raise ValueError("The two SALI vectors must be one-dimensional with equal shape.")
        pair = np.column_stack((first_vector, second_vector))
    if normalize:
        pair = _normalize_columns(pair, name="SALI deviations")
    elif not np.all(np.isfinite(pair)):
        raise ValueError("SALI deviations must contain only finite values.")
    return float(min(np.linalg.norm(pair[:, 0] + pair[:, 1]), np.linalg.norm(pair[:, 0] - pair[:, 1])))


def generalized_alignment_index(
    deviations: np.ndarray | Sequence[Sequence[float]],
    *,
    order: int | None = None,
    backend: str = "auto",
    normalize: bool = True,
) -> float:
    """Compute one GALI volume from column deviation vectors."""

    matrix = np.asarray(deviations, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("deviations must have shape (dimension, n_vectors).")
    if order is not None and (
        isinstance(order, (bool, np.bool_)) or int(order) != order
    ):
        raise ValueError("order must be an integer.")
    selected_order = matrix.shape[1] if order is None else int(order)
    if selected_order < 2 or selected_order > matrix.shape[1]:
        raise ValueError("order must be between 2 and the number of deviation vectors.")
    if selected_order > matrix.shape[0]:
        raise ValueError("order cannot exceed the state-space dimension.")
    selected = matrix[:, :selected_order]
    if normalize:
        selected = _normalize_columns(selected, name="GALI deviations")
    elif not np.all(np.isfinite(selected)):
        raise ValueError("GALI deviations must contain only finite values.")
    value, _log_value, _censored = _volume_metrics(selected, _resolve_backend(backend))
    return float(value)


def linear_dependence_index(
    vectors: np.ndarray | Sequence[Sequence[float]],
    *,
    order: int | None = None,
    backend: str = "auto",
    normalize: bool = True,
) -> float:
    """Compute the SVD/LDI volume, numerically identical to the same GALI.

    The 2026 SVD derivation (doi:10.1016/j.chaos.2026.117884) establishes
    the equivalence of the LDI singular-value product and GALI volume for the
    same normalized deviation vectors.
    """

    return generalized_alignment_index(
        vectors,
        order=order,
        backend=backend,
        normalize=normalize,
    )


def _validate_orders(gali_orders: Sequence[int] | None, n_vectors: int) -> np.ndarray:
    if gali_orders is None:
        orders = np.arange(2, n_vectors + 1, dtype=int)
    else:
        raw = tuple(gali_orders)
        if not raw:
            raise ValueError("gali_orders must contain at least one order.")
        if any(isinstance(value, (bool, np.bool_)) or int(value) != value for value in raw):
            raise ValueError("gali_orders must contain only integers.")
        orders = np.asarray(raw, dtype=int)
    if np.any(orders < 2) or np.any(orders > n_vectors):
        raise ValueError("Each GALI order must be between 2 and n_vectors.")
    if np.unique(orders).size != orders.size:
        raise ValueError("gali_orders must not contain duplicates.")
    return orders


def alignment_indices_from_tangent_history(
    tangent_history: np.ndarray | Sequence[Sequence[Sequence[float]]],
    *,
    coordinates: np.ndarray | Sequence[float] | None = None,
    states: np.ndarray | Sequence[Sequence[float]] | None = None,
    gali_orders: Sequence[int] | None = None,
    backend: str = "auto",
    system_kind: str = "precomputed",
    coordinate_kind: str = "sample",
    method: str = "precomputed",
    method_id: str = "alignment_indices_from_tangent_history",
    q: float | Sequence[float] | np.ndarray = 1.0,
    metadata: Mapping[str, Any] | None = None,
    methodological_warnings: Sequence[str] | None = None,
) -> AlignmentIndexResult:
    """Evaluate SALI/GALI for a public ``(samples, vectors, dimension)`` history."""

    _require_integer_order(q, "alignment_indices_from_tangent_history")
    history = np.asarray(tangent_history, dtype=float)
    if history.ndim != 3 or history.shape[0] == 0:
        raise ValueError(
            "tangent_history must have shape (n_samples, n_vectors, dimension)."
        )
    n_samples, n_vectors, dimension = map(int, history.shape)
    if n_vectors < 2:
        raise ValueError("tangent_history requires at least two deviation vectors.")
    if dimension < n_vectors:
        raise ValueError("n_vectors cannot exceed the state-space dimension.")
    if not np.all(np.isfinite(history)):
        raise ValueError("tangent_history must contain only finite values.")

    if coordinates is None:
        coordinate_values = np.arange(n_samples, dtype=float)
    else:
        coordinate_values = np.asarray(coordinates, dtype=float)
        if coordinate_values.shape != (n_samples,) or not np.all(np.isfinite(coordinate_values)):
            raise ValueError("coordinates must be a finite vector with one value per sample.")

    sampled_states: np.ndarray | None
    if states is None:
        sampled_states = None
    else:
        sampled_states = np.asarray(states, dtype=float)
        if sampled_states.shape != (n_samples, dimension) or not np.all(np.isfinite(sampled_states)):
            raise ValueError("states must have shape (n_samples, dimension) and be finite.")
        sampled_states = sampled_states.copy()

    orders = _validate_orders(gali_orders, n_vectors)
    resolved_backend = _resolve_backend(backend)
    volume_method = (
        "svd_product" if resolved_backend == "numpy" else "householder_qr_log_volume"
    )
    normalized_history = np.empty((n_samples, dimension, n_vectors), dtype=float)
    sali = np.empty(n_samples, dtype=float)
    log_sali = np.empty(n_samples, dtype=float)
    gali = np.empty((n_samples, orders.size), dtype=float)
    log_gali = np.empty_like(gali)
    censored = np.zeros_like(gali, dtype=bool)

    for sample in range(n_samples):
        columns = _normalize_columns(
            history[sample].T,
            name=f"tangent_history[{sample}]",
        )
        normalized_history[sample] = columns
        sali_value = smaller_alignment_index(columns, normalize=False)
        sali[sample] = sali_value
        log_sali[sample] = np.log(sali_value) if sali_value > 0.0 else -np.inf
        for order_index, order in enumerate(orders):
            value, log_value, was_censored = _volume_metrics(
                columns[:, : int(order)],
                resolved_backend,
            )
            gali[sample, order_index] = value
            log_gali[sample, order_index] = log_value
            censored[sample, order_index] = was_censored

    result_metadata = dict(metadata or {})
    result_metadata.update(
        {
            "dimension": dimension,
            "n_vectors": n_vectors,
            "n_samples": n_samples,
            "history_layout": "samples_vectors_dimension",
            "instantaneous_layout": "dimension_vectors_columns",
            "censored_cells": int(np.count_nonzero(censored)),
        }
    )
    final_state = None if sampled_states is None else sampled_states[-1].copy()
    warnings = tuple(methodological_warnings or _BASE_WARNINGS)
    return AlignmentIndexResult(
        coordinates=coordinate_values.copy(),
        sali=sali,
        log_sali=log_sali,
        gali_orders=orders.copy(),
        gali=gali,
        log_gali=log_gali,
        censored=censored,
        status="ok",
        final_state=final_state,
        sampled_states=sampled_states,
        initial_deviations=normalized_history[0].copy(),
        final_deviations=normalized_history[-1].copy(),
        metadata=result_metadata,
        method_id=str(method_id),
        system_kind=str(system_kind),
        coordinate_kind=str(coordinate_kind),
        evolution_method=str(method),
        backend=resolved_backend,
        volume_method=volume_method,
        methodological_warnings=warnings,
    )


def _prepare_initial_deviations(
    dimension: int,
    gali_orders: Sequence[int],
    initial_deviations: np.ndarray | None,
    n_vectors: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_orders = tuple(gali_orders)
    if not raw_orders:
        raise ValueError("gali_orders must contain at least one order.")
    required = max(2, max(int(value) for value in raw_orders))
    if initial_deviations is not None:
        matrix = np.asarray(initial_deviations, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != dimension:
            raise ValueError("initial_deviations must have shape (dimension, n_vectors).")
        vector_count = int(matrix.shape[1])
        if n_vectors is not None and int(n_vectors) != vector_count:
            raise ValueError("n_vectors conflicts with initial_deviations.shape[1].")
        if vector_count < required:
            raise ValueError("initial_deviations does not contain enough vectors for gali_orders.")
        normalized = _normalize_columns(matrix, name="initial_deviations")
    else:
        vector_count = required if n_vectors is None else int(n_vectors)
        if vector_count < required:
            raise ValueError("n_vectors is smaller than the largest requested GALI order.")
        if vector_count > dimension or vector_count < 2:
            raise ValueError("n_vectors must be between 2 and the state-space dimension.")
        generator = np.random.default_rng(int(seed))
        generic = generator.standard_normal((dimension, vector_count))
        orthogonal, upper = np.linalg.qr(generic, mode="reduced")
        signs = np.where(np.diag(upper) < 0.0, -1.0, 1.0)
        normalized = np.ascontiguousarray(orthogonal * signs)
    if normalized.shape[1] > dimension:
        raise ValueError("n_vectors cannot exceed the state-space dimension.")
    orders = _validate_orders(raw_orders, normalized.shape[1])
    return normalized, orders


def _checked_state(initial_state: np.ndarray) -> np.ndarray:
    state = np.asarray(initial_state, dtype=float).copy()
    if state.ndim != 1 or state.size < 2 or not np.all(np.isfinite(state)):
        raise ValueError("x0 must be a finite one-dimensional state vector of dimension at least 2.")
    return state


def _checked_positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    converted = float(value)
    valid = np.isfinite(converted) and (converted >= 0.0 if allow_zero else converted > 0.0)
    if not valid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return converted


def _finite_difference_jacobian(
    callback: Callable[[float, np.ndarray], Any],
    coordinate: float,
    state: np.ndarray,
    step_size: float,
) -> np.ndarray:
    dimension = state.size
    matrix = np.empty((dimension, dimension), dtype=float)
    for column in range(dimension):
        perturbation = np.zeros(dimension, dtype=float)
        component_step = step_size * max(1.0, abs(float(state[column])))
        perturbation[column] = component_step
        forward = np.asarray(callback(coordinate, state + perturbation), dtype=float)
        backward = np.asarray(callback(coordinate, state - perturbation), dtype=float)
        matrix[:, column] = (forward - backward) / (2.0 * component_step)
    return matrix


def _result_from_column_history(
    deviations: Sequence[np.ndarray],
    coordinates: Sequence[float],
    states: Sequence[np.ndarray],
    *,
    orders: np.ndarray,
    backend: str,
    status: str,
    error_message: str | None,
    method_id: str,
    system_kind: str,
    coordinate_kind: str,
    method: str,
    metadata: Mapping[str, Any],
    warnings: Sequence[str],
) -> AlignmentIndexResult:
    public_history = np.stack([matrix.T for matrix in deviations], axis=0)
    result = alignment_indices_from_tangent_history(
        public_history,
        coordinates=np.asarray(coordinates, dtype=float),
        states=np.stack(states, axis=0),
        gali_orders=orders,
        backend=backend,
        system_kind=system_kind,
        coordinate_kind=coordinate_kind,
        method=method,
        method_id=method_id,
        metadata=metadata,
        methodological_warnings=warnings,
    )
    return replace(result, status=status, error_message=error_message)


def integer_flow_alignment_indices(
    rhs: Callable[..., Any],
    jacobian: Callable[..., Any] | None,
    x0: np.ndarray | Sequence[float],
    *,
    t_final: float,
    t_burn: float = 0.0,
    renormalization_time: float = 0.5,
    gali_orders: Sequence[int] = (2,),
    initial_deviations: np.ndarray | None = None,
    n_vectors: int | None = None,
    seed: int = 0,
    method: str = "variational",
    deviation_size: float | None = None,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-12,
    max_step: float = np.inf,
    jacobian_eps: float = 1.0e-6,
    div_threshold: float | None = None,
    parameters: Any = None,
    backend: str = "auto",
    q: float | Sequence[float] | np.ndarray = 1.0,
) -> AlignmentIndexResult:
    """Compute q=1 flow SALI/GALI with DOP853.

    ``method='variational'`` integrates state and tangent equations jointly.
    ``method='multi_particle'`` integrates the reference and finite neighboring
    trajectories and reinjects each neighbor after every segment.
    """

    _require_integer_order(q, "integer_flow_alignment_indices")
    evolution_method = str(method).strip().lower()
    if evolution_method not in {"variational", "multi_particle"}:
        raise ValueError("method must be 'variational' or 'multi_particle'.")
    state = _checked_state(np.asarray(x0, dtype=float))
    dimension = state.size
    total_time = _checked_positive(t_final, "t_final")
    burn_time = _checked_positive(t_burn, "t_burn", allow_zero=True)
    interval = _checked_positive(renormalization_time, "renormalization_time")
    relative_tolerance = _checked_positive(rtol, "rtol")
    absolute_tolerance = _checked_positive(atol, "atol")
    finite_difference_step = _checked_positive(jacobian_eps, "jacobian_eps")
    maximum_step = float(max_step)
    if np.isnan(maximum_step) or maximum_step <= 0.0:
        raise ValueError("max_step must be positive.")
    threshold = None if div_threshold is None else _checked_positive(div_threshold, "div_threshold")
    separation_size = (
        float(np.sqrt(_EPS))
        if deviation_size is None
        else _checked_positive(deviation_size, "deviation_size")
    )
    initial, orders = _prepare_initial_deviations(
        dimension,
        gali_orders,
        initial_deviations,
        n_vectors,
        seed,
    )
    bound_rhs = bind_rhs(rhs, parameters)
    bound_jacobian = None if jacobian is None else bind_rhs(jacobian, parameters)

    rhs_calls = 0

    def evaluate(time: float, point: np.ndarray) -> np.ndarray:
        nonlocal rhs_calls
        rhs_calls += 1
        value = np.asarray(bound_rhs(time, point), dtype=float)
        if value.shape != (dimension,) or not np.all(np.isfinite(value)):
            raise ValueError(f"rhs must return a finite ({dimension},) vector.")
        return value

    def evaluate_jacobian(time: float, point: np.ndarray) -> np.ndarray:
        if bound_jacobian is None:
            matrix = _finite_difference_jacobian(
                evaluate,
                time,
                point,
                finite_difference_step,
            )
        else:
            matrix = np.asarray(bound_jacobian(time, point), dtype=float)
        if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"jacobian must return a finite ({dimension}, {dimension}) matrix.")
        return matrix

    counters = {"solver_nfev": 0, "burn_solver_nfev": 0, "segments": 0}
    burn_completed = 0.0
    deviations: list[np.ndarray] = [initial.copy()]
    coordinates: list[float] = [0.0]
    sampled_states: list[np.ndarray] = [state.copy()]
    warnings = _BASE_WARNINGS + (
        _MULTI_PARTICLE_WARNINGS if evolution_method == "multi_particle" else ()
    )
    metadata_base: dict[str, Any] = {
        "solver": "scipy.integrate.solve_ivp",
        "solver_method": "DOP853",
        "evolution_method": evolution_method,
        "dimension": dimension,
        "n_vectors": initial.shape[1],
        "t_final_requested": total_time,
        "t_burn_requested": burn_time,
        "renormalization_time": interval,
        "deviation_size": separation_size if evolution_method == "multi_particle" else None,
        "rtol": relative_tolerance,
        "atol": absolute_tolerance,
        "max_step": maximum_step,
        "jacobian_source": (
            "not_used_multi_particle"
            if evolution_method == "multi_particle"
            else ("analytic" if jacobian is not None else "central_relative_componentwise")
        ),
        "jacobian_eps": finite_difference_step if jacobian is None else None,
        "normalization": "independent_l2_no_evolution_qr",
    }

    def finalize(status: str, error: str | None = None) -> AlignmentIndexResult:
        metadata = dict(metadata_base)
        metadata.update(counters)
        metadata["rhs_calls"] = rhs_calls
        metadata["t_burn_completed"] = burn_completed
        metadata["t_accumulate_completed"] = coordinates[-1]
        result = _result_from_column_history(
            deviations,
            coordinates,
            sampled_states,
            orders=orders,
            backend=backend,
            status=status,
            error_message=error,
            method_id=f"integer_flow_sali_gali_{evolution_method}",
            system_kind="flow",
            coordinate_kind="time_after_burn",
            method=evolution_method,
            metadata=metadata,
            warnings=warnings,
        )
        return replace(result, final_state=state.copy())

    if threshold is not None and np.linalg.norm(state) >= threshold:
        return finalize("burn_diverged", "initial state meets or exceeds div_threshold")
    if burn_time > 0.0:
        try:
            burned = solve_ivp(
                evaluate,
                (0.0, burn_time),
                state,
                method="DOP853",
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                max_step=maximum_step,
            )
        except (TypeError, ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            return finalize("burn_solver_exception", str(exc))
        counters["solver_nfev"] += int(burned.nfev)
        counters["burn_solver_nfev"] = int(burned.nfev)
        burn_completed = float(burned.t[-1]) if burned.t.size else 0.0
        state = np.asarray(burned.y[:, -1], dtype=float)
        sampled_states[0] = state.copy()
        if not burned.success:
            return finalize("burn_solver_failure", str(burned.message))
        if not np.all(np.isfinite(state)):
            return finalize("nonfinite_solution", "burn-in produced non-finite state")
        if threshold is not None and np.linalg.norm(state) >= threshold:
            return finalize("burn_diverged", "state exceeded div_threshold during burn-in")
        burn_completed = burn_time

    current_deviations = initial.copy()
    neighbors = state[:, None] + separation_size * current_deviations
    elapsed = 0.0
    time_epsilon = 16.0 * _EPS * total_time
    while elapsed < total_time - time_epsilon:
        duration = min(interval, total_time - elapsed)
        start_time = burn_time + elapsed
        stop_time = start_time + duration
        if evolution_method == "variational":
            augmented0 = np.concatenate((state, current_deviations.ravel()))

            def augmented_rhs(time: float, augmented: np.ndarray) -> np.ndarray:
                current = augmented[:dimension]
                tangent = augmented[dimension:].reshape(dimension, initial.shape[1])
                return np.concatenate(
                    (evaluate(time, current), (evaluate_jacobian(time, current) @ tangent).ravel())
                )

        else:
            augmented0 = np.concatenate((state, neighbors.T.ravel()))

            def augmented_rhs(time: float, augmented: np.ndarray) -> np.ndarray:
                reference = augmented[:dimension]
                nearby = augmented[dimension:].reshape(initial.shape[1], dimension)
                derivatives = np.empty_like(nearby)
                for index in range(initial.shape[1]):
                    derivatives[index] = evaluate(time, nearby[index])
                return np.concatenate((evaluate(time, reference), derivatives.ravel()))

        try:
            solved = solve_ivp(
                augmented_rhs,
                (start_time, stop_time),
                augmented0,
                method="DOP853",
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                max_step=maximum_step,
            )
        except (TypeError, ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            return finalize("solver_exception", str(exc))
        counters["solver_nfev"] += int(solved.nfev)
        if not solved.success:
            state = np.asarray(solved.y[:dimension, -1], dtype=float)
            return finalize("solver_failure", str(solved.message))
        state = np.asarray(solved.y[:dimension, -1], dtype=float)
        if evolution_method == "variational":
            evolved = np.asarray(solved.y[dimension:, -1], dtype=float).reshape(
                dimension,
                initial.shape[1],
            )
        else:
            neighbor_rows = np.asarray(solved.y[dimension:, -1], dtype=float).reshape(
                initial.shape[1],
                dimension,
            )
            evolved = neighbor_rows.T - state[:, None]
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(evolved)):
            return finalize("nonfinite_solution", "evolution produced non-finite values")
        try:
            current_deviations = _normalize_columns(evolved, name="evolved deviations")
        except ValueError as exc:
            return finalize("collapsed_deviation", str(exc))
        neighbors = state[:, None] + separation_size * current_deviations
        elapsed += duration
        if total_time - elapsed <= time_epsilon:
            elapsed = total_time
        counters["segments"] += 1
        coordinates.append(float(elapsed))
        deviations.append(current_deviations.copy())
        sampled_states.append(state.copy())
        if threshold is not None and np.linalg.norm(state) >= threshold:
            return finalize("diverged", "state exceeded div_threshold")
    return finalize("ok")


def integer_map_alignment_indices(
    map_function: Callable[..., Any],
    jacobian: Callable[..., Any] | None,
    x0: np.ndarray | Sequence[float],
    *,
    iterations: int,
    transient_iterations: int = 0,
    sample_every: int = 1,
    gali_orders: Sequence[int] = (2,),
    initial_deviations: np.ndarray | None = None,
    n_vectors: int | None = None,
    seed: int = 0,
    method: str = "variational",
    deviation_size: float | None = None,
    jacobian_eps: float = 1.0e-6,
    div_threshold: float | None = None,
    parameters: Any = None,
    backend: str = "auto",
    q: float | Sequence[float] | np.ndarray = 1.0,
) -> AlignmentIndexResult:
    """Compute q=1 map SALI/GALI by exact Jacobian or neighboring-map propagation."""

    _require_integer_order(q, "integer_map_alignment_indices")
    evolution_method = str(method).strip().lower()
    if evolution_method not in {"variational", "multi_particle"}:
        raise ValueError("method must be 'variational' or 'multi_particle'.")
    state = _checked_state(np.asarray(x0, dtype=float))
    dimension = state.size
    if isinstance(iterations, (bool, np.bool_)) or int(iterations) != iterations or int(iterations) <= 0:
        raise ValueError("iterations must be a positive integer.")
    total_iterations = int(iterations)
    if (
        isinstance(transient_iterations, (bool, np.bool_))
        or int(transient_iterations) != transient_iterations
        or int(transient_iterations) < 0
    ):
        raise ValueError("transient_iterations must be a nonnegative integer.")
    transient_count = int(transient_iterations)
    if isinstance(sample_every, (bool, np.bool_)) or int(sample_every) != sample_every or int(sample_every) <= 0:
        raise ValueError("sample_every must be a positive integer.")
    sampling_interval = int(sample_every)
    finite_difference_step = _checked_positive(jacobian_eps, "jacobian_eps")
    threshold = None if div_threshold is None else _checked_positive(div_threshold, "div_threshold")
    separation_size = (
        float(np.sqrt(_EPS))
        if deviation_size is None
        else _checked_positive(deviation_size, "deviation_size")
    )
    initial, orders = _prepare_initial_deviations(
        dimension,
        gali_orders,
        initial_deviations,
        n_vectors,
        seed,
    )
    bound_map = bind_rhs(map_function, parameters)
    bound_jacobian = None if jacobian is None else bind_rhs(jacobian, parameters)

    def evaluate(iteration: int, point: np.ndarray) -> np.ndarray:
        value = np.asarray(bound_map(float(iteration), point), dtype=float)
        if value.shape != (dimension,) or not np.all(np.isfinite(value)):
            raise ValueError(f"map_function must return a finite ({dimension},) vector.")
        return value

    def evaluate_jacobian(iteration: int, point: np.ndarray) -> np.ndarray:
        if bound_jacobian is None:
            matrix = _finite_difference_jacobian(
                lambda coordinate, current: evaluate(int(coordinate), current),
                float(iteration),
                point,
                finite_difference_step,
            )
        else:
            matrix = np.asarray(bound_jacobian(float(iteration), point), dtype=float)
        if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"jacobian must return a finite ({dimension}, {dimension}) matrix.")
        return matrix

    warnings = _BASE_WARNINGS + (
        _MULTI_PARTICLE_WARNINGS if evolution_method == "multi_particle" else ()
    )
    deviations: list[np.ndarray] = [initial.copy()]
    coordinates: list[float] = [0.0]
    sampled_states: list[np.ndarray] = [state.copy()]
    completed_transient = 0
    completed_iterations = 0
    metadata_base: dict[str, Any] = {
        "evolution_method": evolution_method,
        "dimension": dimension,
        "n_vectors": initial.shape[1],
        "iterations_requested": total_iterations,
        "transient_iterations_requested": transient_count,
        "sample_every": sampling_interval,
        "renormalization_interval_iterations": 1,
        "deviation_size": separation_size if evolution_method == "multi_particle" else None,
        "jacobian_source": (
            "not_used_multi_particle"
            if evolution_method == "multi_particle"
            else ("analytic" if jacobian is not None else "central_relative_componentwise")
        ),
        "jacobian_eps": finite_difference_step if jacobian is None else None,
        "normalization": "independent_l2_no_evolution_qr",
    }

    def finalize(status: str, error: str | None = None) -> AlignmentIndexResult:
        metadata = dict(metadata_base)
        metadata["transient_iterations_completed"] = completed_transient
        metadata["iterations_completed"] = completed_iterations
        result = _result_from_column_history(
            deviations,
            coordinates,
            sampled_states,
            orders=orders,
            backend=backend,
            status=status,
            error_message=error,
            method_id=f"integer_map_sali_gali_{evolution_method}",
            system_kind="map",
            coordinate_kind="iteration_after_transient",
            method=evolution_method,
            metadata=metadata,
            warnings=warnings,
        )
        return replace(result, final_state=state.copy())

    if threshold is not None and np.linalg.norm(state) >= threshold:
        return finalize("burn_diverged", "initial state meets or exceeds div_threshold")
    try:
        for index in range(transient_count):
            state = evaluate(index, state)
            completed_transient = index + 1
            if threshold is not None and np.linalg.norm(state) >= threshold:
                sampled_states[0] = state.copy()
                return finalize("burn_diverged", "state exceeded div_threshold during transient")
    except (TypeError, ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        sampled_states[0] = state.copy()
        return finalize("burn_map_exception", str(exc))
    sampled_states[0] = state.copy()

    current_deviations = initial.copy()
    neighbors = state[:, None] + separation_size * current_deviations
    try:
        for local_iteration in range(total_iterations):
            absolute_iteration = transient_count + local_iteration
            if evolution_method == "variational":
                matrix = evaluate_jacobian(absolute_iteration, state)
                evolved = matrix @ current_deviations
                next_state = evaluate(absolute_iteration, state)
            else:
                next_state = evaluate(absolute_iteration, state)
                next_neighbors = np.empty_like(neighbors)
                for vector_index in range(initial.shape[1]):
                    next_neighbors[:, vector_index] = evaluate(
                        absolute_iteration,
                        neighbors[:, vector_index],
                    )
                evolved = next_neighbors - next_state[:, None]
            state = next_state
            current_deviations = _normalize_columns(evolved, name="evolved deviations")
            neighbors = state[:, None] + separation_size * current_deviations
            completed_iterations = local_iteration + 1
            if (
                completed_iterations % sampling_interval == 0
                or completed_iterations == total_iterations
            ):
                coordinates.append(float(completed_iterations))
                deviations.append(current_deviations.copy())
                sampled_states.append(state.copy())
            if threshold is not None and np.linalg.norm(state) >= threshold:
                if coordinates[-1] != float(completed_iterations):
                    coordinates.append(float(completed_iterations))
                    deviations.append(current_deviations.copy())
                    sampled_states.append(state.copy())
                return finalize("diverged", "state exceeded div_threshold")
    except (TypeError, ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        return finalize("map_exception", str(exc))
    return finalize("ok")


def integer_system_alignment_indices(
    system: object,
    x0: np.ndarray | Sequence[float],
    *,
    q: float | Sequence[float] | np.ndarray = 1.0,
    **kwargs: Any,
) -> AlignmentIndexResult:
    """Dispatch integer-order alignment indices for a system object."""

    _require_integer_order(q, "integer_system_alignment_indices")
    declared_order = _declared_system_order(system)
    if declared_order is not None:
        _require_integer_order(declared_order, "integer_system_alignment_indices")
    kind = str(getattr(system, "kind", "flow")).strip().lower()
    if kind not in {"flow", "map"}:
        raise ValueError("system.kind must be 'flow' or 'map'.")
    evaluate = getattr(system, "evaluate", None)
    if not callable(evaluate):
        raise ValueError("system must expose a callable evaluate(state) method.")
    jacobian_matrix = getattr(system, "jacobian_matrix", None)
    jacobian_declaration = getattr(system, "jacobian", "attribute_not_declared")
    use_analytic = callable(jacobian_matrix) and jacobian_declaration is not None
    options = dict(kwargs)
    parameter_overrides = options.pop("parameters", None)
    if parameter_overrides is None:
        rhs = lambda state: evaluate(state)
        jacobian = (lambda state: jacobian_matrix(state)) if use_analytic else None
    else:
        rhs = lambda state: evaluate(state, parameter_overrides)
        jacobian = (
            (lambda state: jacobian_matrix(state, parameter_overrides))
            if use_analytic
            else None
        )
    if kind == "flow":
        return integer_flow_alignment_indices(
            rhs,
            jacobian,
            x0,
            q=1.0,
            **options,
        )
    return integer_map_alignment_indices(
        rhs,
        jacobian,
        x0,
        q=1.0,
        **options,
    )


__all__ = [
    "AlignmentIndexResult",
    "NUMBA_AVAILABLE",
    "smaller_alignment_index",
    "generalized_alignment_index",
    "linear_dependence_index",
    "alignment_indices_from_tangent_history",
    "integer_flow_alignment_indices",
    "integer_map_alignment_indices",
    "integer_system_alignment_indices",
]
