"""Covariant Lyapunov vectors for memoryless integer-order dynamics.

Stability: experimental
    The q=1 layouts and evidence boundary are public experimental contracts.
    Optional convergence controls may be added without promoting a
    fractional-memory interpretation.

The implementation follows the forward-QR/backward-triangular algorithm of
Ginelli et al.  It is written independently from pynamicalsys and does not use
ChaosTools.jl as an inner-loop backend.  Public CLV histories have shape
``(samples, n_vectors, dimension)``; internal QR matrices keep vectors in
columns.

References
----------
Ginelli et al. (2007), doi:10.1103/PhysRevLett.99.130601.
Kuptsov & Parlitz (2012), doi:10.1007/s00332-012-9126-5.
Ginelli et al. (2013), doi:10.1088/1751-8113/46/25/254005.
du Plessis, Hillebrand & Skokos (2026), doi:10.1016/j.physd.2026.135237.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_triangular, subspace_angles

from .._rhs import bind_rhs

try:  # Numba is declared, but an import/runtime failure remains recoverable.
    from numba import njit

    NUMBA_AVAILABLE = True
except (ImportError, OSError):  # pragma: no cover - exercised without Numba
    NUMBA_AVAILABLE = False
    njit = None  # type: ignore[assignment]


_ORDER_TOLERANCE = 1.0e-9
_DEFAULT_MAX_WORKSPACE_BYTES = 512 * 1024 * 1024
_REFERENCES: tuple[str, ...] = (
    "Ginelli et al. 2007, doi:10.1103/PhysRevLett.99.130601",
    "Kuptsov & Parlitz 2012, doi:10.1007/s00332-012-9126-5",
    "Ginelli et al. 2013, doi:10.1088/1751-8113/46/25/254005",
    "Froyland et al. 2013, doi:10.1016/j.physd.2012.12.005",
    "du Plessis, Hillebrand & Skokos 2026, doi:10.1016/j.physd.2026.135237",
)
_BASE_WARNINGS: tuple[str, ...] = (
    "Valid only for q=1 memoryless integer-order tangent cocycles.",
    "CLVs and their angles are finite-time diagnostics and do not by themselves "
    "certify chaos, hyperbolicity, attraction, or hiddenness.",
    "Forward and backward transient lengths are explicit user controls; this "
    "version does not implement an automatic transient-convergence stopping test.",
    "The sign of each CLV is a gauge choice and must not be compared as an "
    "oriented vector unless the application supplies an orientation convention.",
)
_DEGENERACY_WARNING = (
    "Repeated or nearly degenerate finite-time exponents make individual CLV "
    "columns nonunique or ill-conditioned; compare covariant subspaces instead."
)


if NUMBA_AVAILABLE:

    @njit(cache=False, fastmath=False, nogil=True)  # type: ignore[misc]
    def _backward_sweep_numba(
        q_history: np.ndarray,
        observed_r: np.ndarray,
        future_r: np.ndarray,
        terminal_coefficients: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_observed = observed_r.shape[0]
        dimension = q_history.shape[1]
        n_vectors = q_history.shape[2]
        coefficients = np.empty(
            (n_observed + 1, n_vectors, n_vectors), dtype=np.float64
        )
        vectors = np.empty(
            (n_observed + 1, n_vectors, dimension), dtype=np.float64
        )
        current = terminal_coefficients.copy()

        for future_index in range(future_r.shape[0] - 1, -1, -1):
            upper = future_r[future_index]
            solved = np.empty_like(current)
            for column in range(n_vectors):
                for row in range(n_vectors - 1, -1, -1):
                    value = current[row, column]
                    for inner in range(row + 1, n_vectors):
                        value -= upper[row, inner] * solved[inner, column]
                    solved[row, column] = value / upper[row, row]
            for column in range(n_vectors):
                norm = 0.0
                for row in range(n_vectors):
                    norm = np.hypot(norm, solved[row, column])
                for row in range(n_vectors):
                    solved[row, column] /= norm
            current = solved

        for sample in range(n_observed, -1, -1):
            coefficients[sample] = current
            for vector_index in range(n_vectors):
                norm = 0.0
                for coordinate in range(dimension):
                    value = 0.0
                    for inner in range(n_vectors):
                        value += (
                            q_history[sample, coordinate, inner]
                            * current[inner, vector_index]
                        )
                    vectors[sample, vector_index, coordinate] = value
                    norm = np.hypot(norm, value)
                for coordinate in range(dimension):
                    vectors[sample, vector_index, coordinate] /= norm

            if sample > 0:
                upper = observed_r[sample - 1]
                solved = np.empty_like(current)
                for column in range(n_vectors):
                    for row in range(n_vectors - 1, -1, -1):
                        value = current[row, column]
                        for inner in range(row + 1, n_vectors):
                            value -= upper[row, inner] * solved[inner, column]
                        solved[row, column] = value / upper[row, row]
                for column in range(n_vectors):
                    norm = 0.0
                    for row in range(n_vectors):
                        norm = np.hypot(norm, solved[row, column])
                    for row in range(n_vectors):
                        solved[row, column] /= norm
                current = solved

        return vectors, coefficients


@dataclass(frozen=True)
class CovariantQRHistoryResult:
    """CLVs reconstructed from a validated QR history.

    ``vectors`` has shape ``(samples, n_vectors, dimension)`` and
    ``coefficients`` has shape ``(samples, n_vectors, n_vectors)``.
    """

    vectors: np.ndarray
    coefficients: np.ndarray
    backend: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    method_id: str = "integer_covariant_vectors_from_qr_history"
    derivative_model: str = "integer"
    q: float = 1.0
    reference_ids: tuple[str, ...] = field(default_factory=lambda: _REFERENCES)
    methodological_warnings: tuple[str, ...] = field(
        default_factory=lambda: _BASE_WARNINGS
    )


@dataclass(frozen=True)
class CovariantLyapunovResult:
    """Finite-time q=1 Ginelli CLV result for a flow or map."""

    coordinates: np.ndarray
    sampled_states: np.ndarray
    vectors: np.ndarray
    exponents: np.ndarray
    convergence: np.ndarray
    singular_segments: np.ndarray
    status: str
    final_state: np.ndarray
    future_state: np.ndarray
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    method_id: str = "integer_ginelli_clv"
    system_kind: str = "precomputed"
    coordinate_kind: str = "sample"
    backend: str = "numpy"
    propagation_backend: str = "precomputed"
    derivative_model: str = "integer"
    q: float = 1.0
    finite_time_local: bool = True
    normalization: str = "independent_l2_columns"
    orthonormalization: str = "forward_reduced_qr_positive_diagonal"
    reference_ids: tuple[str, ...] = field(default_factory=lambda: _REFERENCES)
    methodological_warnings: tuple[str, ...] = field(
        default_factory=lambda: _BASE_WARNINGS
    )


@dataclass(frozen=True)
class CovariantAngleResult:
    """Pair and minimum-principal-subspace angles from a CLV history."""

    coordinates: np.ndarray
    pair_indices: np.ndarray
    pair_angles: np.ndarray
    subspace_pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    subspace_angles: np.ndarray
    window_coordinates: np.ndarray
    window_pair_angles: np.ndarray
    window_subspace_angles: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    method_id: str = "covariant_lyapunov_angles"
    angle_unit: str = "radian"
    finite_time_local: bool = True


class _InvalidRhsError(ValueError):
    pass


class _InvalidJacobianError(ValueError):
    pass


class _EvolutionFailure(RuntimeError):
    def __init__(self, status: str, message: str, state: np.ndarray):
        super().__init__(message)
        self.status = status
        self.state = np.asarray(state, dtype=float).copy()


def _require_integer_order(
    q: float | Sequence[float] | np.ndarray,
    routine: str,
) -> None:
    try:
        values = np.asarray(q, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{routine} requires a finite scalar or vector q.") from exc
    if values.size == 0 or not np.all(np.isfinite(values)) or not np.all(
        np.abs(values - 1.0) <= _ORDER_TOLERANCE
    ):
        raise ValueError(
            f"{routine} is valid only for q=1 integer-order dynamics; received q={q!r}."
        )


def _checked_state(x0: np.ndarray | Sequence[float]) -> np.ndarray:
    state = np.asarray(x0, dtype=float).copy()
    if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
        raise ValueError("x0 must be a non-empty finite one-dimensional state vector.")
    return state


def _checked_nonnegative(value: float, name: str) -> float:
    converted = float(value)
    if not np.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return converted


def _checked_positive(value: float, name: str) -> float:
    converted = float(value)
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return converted


def _checked_nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return converted


def _checked_positive_integer(value: int, name: str) -> int:
    converted = _checked_nonnegative_integer(value, name)
    if converted == 0:
        raise ValueError(f"{name} must be positive.")
    return converted


def _checked_workspace_limit(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("max_workspace_bytes must be an integer or None.")
    converted = int(value)
    if converted <= 0:
        raise ValueError("max_workspace_bytes must be positive when supplied.")
    return converted


def _segment_count(total: float, interval: float) -> int:
    if total == 0.0:
        return 0
    return int(np.ceil(total / interval - 16.0 * np.finfo(float).eps))


def _estimate_workspace_bytes(
    n_observed: int,
    n_future: int,
    dimension: int,
    n_vectors: int,
) -> int:
    # Q history, states, R histories, coefficient history, output CLVs,
    # convergence and coordinates.  Integrator temporaries are not included.
    float_count = (
        (n_observed + 1) * (2 * dimension * n_vectors + dimension)
        + (n_observed + n_future) * n_vectors * n_vectors
        + (n_observed + 1) * n_vectors * n_vectors
        + n_observed * n_vectors
        + (n_observed + 1)
    )
    return int(8 * float_count)


def _enforce_workspace_limit(estimated: int, maximum: int | None) -> None:
    if maximum is not None and estimated > maximum:
        raise MemoryError(
            "Estimated CLV workspace exceeds max_workspace_bytes: "
            f"estimated={estimated}, maximum={maximum}."
        )


def _positive_reduced_qr(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        basis, upper = np.linalg.qr(np.asarray(matrix, dtype=float), mode="reduced")
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(f"QR factorization failed: {exc}") from exc
    diagonal = np.diag(upper)
    scale = float(np.max(np.abs(upper))) if upper.size else 0.0
    threshold = np.finfo(float).eps * max(1, upper.shape[0]) * scale * 16.0
    if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(upper)):
        raise np.linalg.LinAlgError("QR factorization produced non-finite values.")
    if scale == 0.0 or np.any(np.abs(diagonal) <= threshold):
        raise np.linalg.LinAlgError("Tangent cocycle lost numerical rank during QR.")
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    basis = basis * signs[None, :]
    upper = signs[:, None] * upper
    return np.ascontiguousarray(basis), np.ascontiguousarray(upper)


def _prepare_basis(
    dimension: int,
    n_vectors: int | None,
    initial_basis: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, int]:
    if initial_basis is None:
        count = dimension if n_vectors is None else _checked_positive_integer(
            n_vectors, "n_vectors"
        )
        if count > dimension:
            raise ValueError("n_vectors cannot exceed the state dimension.")
        generator = np.random.default_rng(int(seed))
        raw = generator.standard_normal((dimension, count))
    else:
        raw = np.asarray(initial_basis, dtype=float)
        if raw.ndim != 2 or raw.shape[0] != dimension or raw.shape[1] == 0:
            raise ValueError(
                "initial_basis must have shape (dimension, n_vectors) with at least one column."
            )
        if not np.all(np.isfinite(raw)):
            raise ValueError("initial_basis must contain only finite values.")
        count = raw.shape[1]
        if n_vectors is not None and _checked_positive_integer(
            n_vectors, "n_vectors"
        ) != count:
            raise ValueError("n_vectors must match initial_basis.shape[1].")
        if count > dimension:
            raise ValueError("n_vectors cannot exceed the state dimension.")
    try:
        basis, _ = _positive_reduced_qr(raw)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"initial_basis must have full column rank: {exc}") from exc
    return basis, count


def _normalize_coefficient_columns(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=float).copy()
    norms = np.linalg.norm(result, axis=0)
    if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError("terminal_coefficients contains a zero or non-finite column.")
    return np.ascontiguousarray(result / norms)


def _prepare_terminal_coefficients(
    count: int,
    terminal_coefficients: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    if terminal_coefficients is None:
        generator = np.random.default_rng(int(seed))
        matrix = np.triu(generator.standard_normal((count, count)))
        diagonal = np.diag(matrix).copy()
        diagonal = np.where(np.abs(diagonal) < 0.5, np.where(diagonal < 0.0, -1.0, 1.0), diagonal)
        np.fill_diagonal(matrix, diagonal)
    else:
        matrix = np.asarray(terminal_coefficients, dtype=float)
        if matrix.shape != (count, count) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"terminal_coefficients must be a finite ({count}, {count}) matrix."
            )
        scale = max(1.0, float(np.max(np.abs(matrix))))
        if not np.allclose(matrix, np.triu(matrix), rtol=0.0, atol=1.0e-13 * scale):
            raise ValueError("terminal_coefficients must be upper triangular.")
        diagonal = np.diag(matrix)
        if np.any(diagonal == 0.0):
            raise ValueError("terminal_coefficients must have a nonzero diagonal.")
    return _normalize_coefficient_columns(matrix)


def _validate_qr_histories(
    orthonormal_bases: np.ndarray,
    r_factors: np.ndarray,
    future_r_factors: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bases = np.asarray(orthonormal_bases, dtype=float)
    observed = np.asarray(r_factors, dtype=float)
    if bases.ndim != 3:
        raise ValueError(
            "orthonormal_bases must have shape (samples, dimension, n_vectors)."
        )
    samples, dimension, count = bases.shape
    if samples == 0 or dimension == 0 or count == 0 or count > dimension:
        raise ValueError("orthonormal_bases has invalid or empty dimensions.")
    if observed.shape != (samples - 1, count, count):
        raise ValueError(
            "r_factors must have shape (samples - 1, n_vectors, n_vectors)."
        )
    if future_r_factors is None:
        future = np.empty((0, count, count), dtype=float)
    else:
        future = np.asarray(future_r_factors, dtype=float)
        if future.ndim != 3 or future.shape[1:] != (count, count):
            raise ValueError(
                "future_r_factors must have shape (future_segments, n_vectors, n_vectors)."
            )
    if not np.all(np.isfinite(bases)) or not np.all(np.isfinite(observed)) or not np.all(
        np.isfinite(future)
    ):
        raise ValueError("QR histories must contain only finite values.")
    identity = np.eye(count)
    residuals = np.linalg.norm(
        np.einsum("sdi,sdj->sij", bases, bases) - identity[None, :, :],
        axis=(1, 2),
    )
    if np.any(residuals > 1.0e-8 * max(1, count)):
        raise ValueError("orthonormal_bases columns must be orthonormal.")
    for name, history in (("r_factors", observed), ("future_r_factors", future)):
        if history.size == 0:
            continue
        scale = np.maximum(1.0, np.max(np.abs(history), axis=(1, 2)))
        lower = np.tril(history, k=-1)
        if np.any(np.max(np.abs(lower), axis=(1, 2)) > 1.0e-12 * scale):
            raise ValueError(f"{name} must contain upper-triangular matrices.")
        if np.any(np.diagonal(history, axis1=1, axis2=2) <= 0.0):
            raise ValueError(f"{name} must have strictly positive diagonals.")
    return (
        np.ascontiguousarray(bases),
        np.ascontiguousarray(observed),
        np.ascontiguousarray(future),
    )


def _singular_segments(
    observed: np.ndarray,
    future: np.ndarray,
    singular_tolerance: float | None,
) -> np.ndarray:
    combined = (
        np.concatenate((observed, future), axis=0) if future.shape[0] else observed
    )
    if combined.shape[0] == 0:
        return np.empty(0, dtype=int)
    if singular_tolerance is not None:
        tolerance = _checked_positive(singular_tolerance, "singular_tolerance")
        thresholds = np.full(combined.shape[0], tolerance, dtype=float)
    else:
        scales = np.max(np.abs(combined), axis=(1, 2))
        thresholds = (
            np.finfo(float).eps * max(1, combined.shape[1]) * scales * 16.0
        )
    diagonals = np.min(np.abs(np.diagonal(combined, axis1=1, axis2=2)), axis=1)
    return np.flatnonzero(diagonals <= thresholds).astype(int)


_NUMBA_OPERATIONAL: bool | None = None


def _numba_is_operational() -> bool:
    global _NUMBA_OPERATIONAL
    if _NUMBA_OPERATIONAL is None:
        if not NUMBA_AVAILABLE:
            _NUMBA_OPERATIONAL = False
        else:
            try:
                q_history = np.repeat(np.eye(2)[None, :, :], 2, axis=0)
                observed = np.eye(2)[None, :, :]
                future = np.empty((0, 2, 2), dtype=float)
                _backward_sweep_numba(q_history, observed, future, np.eye(2))
            except Exception:  # pragma: no cover - environment dependent
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
            raise RuntimeError("The Numba CLV backward backend is not available.")
        return "numba"
    return "numba" if _numba_is_operational() else "numpy"


def _backward_sweep_numpy(
    bases: np.ndarray,
    observed: np.ndarray,
    future: np.ndarray,
    terminal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = bases.shape[2]
    coefficients = np.empty((bases.shape[0], count, count), dtype=float)
    vectors = np.empty((bases.shape[0], count, bases.shape[1]), dtype=float)
    current = terminal.copy()
    for upper in future[::-1]:
        current = solve_triangular(
            upper, current, lower=False, check_finite=False, overwrite_b=False
        )
        current = _normalize_coefficient_columns(current)
    for sample in range(bases.shape[0] - 1, -1, -1):
        coefficients[sample] = current
        reconstructed = bases[sample] @ current
        reconstructed /= np.linalg.norm(reconstructed, axis=0)
        vectors[sample] = reconstructed.T
        if sample > 0:
            current = solve_triangular(
                observed[sample - 1],
                current,
                lower=False,
                check_finite=False,
                overwrite_b=False,
            )
            current = _normalize_coefficient_columns(current)
    return np.ascontiguousarray(vectors), np.ascontiguousarray(coefficients)


def integer_covariant_vectors_from_qr_history(
    orthonormal_bases: np.ndarray,
    r_factors: np.ndarray,
    *,
    future_r_factors: np.ndarray | None = None,
    terminal_coefficients: np.ndarray | None = None,
    seed: int = 0,
    backend: str = "auto",
    singular_tolerance: float | None = None,
    max_workspace_bytes: int | None = _DEFAULT_MAX_WORKSPACE_BYTES,
    q: float | Sequence[float] | np.ndarray = 1.0,
) -> CovariantQRHistoryResult:
    """Reconstruct q=1 CLVs from positive-diagonal reduced-QR histories."""

    _require_integer_order(q, "integer_covariant_vectors_from_qr_history")
    bases, observed, future = _validate_qr_histories(
        orthonormal_bases, r_factors, future_r_factors
    )
    count = bases.shape[2]
    terminal = _prepare_terminal_coefficients(count, terminal_coefficients, seed)
    maximum = _checked_workspace_limit(max_workspace_bytes)
    estimated = int(
        8
        * bases.shape[0]
        * (bases.shape[1] * count + count * count)
    )
    _enforce_workspace_limit(estimated, maximum)
    singular = _singular_segments(observed, future, singular_tolerance)
    if singular.size:
        raise np.linalg.LinAlgError(
            "QR history contains singular or near-singular R factors at combined "
            f"segment indices {singular.tolist()}."
        )
    resolved = _resolve_backend(backend)
    if resolved == "numba":
        vectors, coefficients = _backward_sweep_numba(
            bases, observed, future, terminal
        )
    else:
        vectors, coefficients = _backward_sweep_numpy(
            bases, observed, future, terminal
        )
    return CovariantQRHistoryResult(
        vectors=np.asarray(vectors, dtype=float),
        coefficients=np.asarray(coefficients, dtype=float),
        backend=resolved,
        metadata={
            "samples": int(bases.shape[0]),
            "dimension": int(bases.shape[1]),
            "n_vectors": int(count),
            "observed_segments": int(observed.shape[0]),
            "future_segments": int(future.shape[0]),
            "estimated_output_workspace_bytes": estimated,
            "terminal_coefficients_source": (
                "seeded_random_upper_triangular"
                if terminal_coefficients is None
                else "user_supplied_upper_triangular"
            ),
            "qr_diagonal_convention": "strictly_positive",
        },
    )


def _canonical_pairs(
    pairs: Sequence[Sequence[int]] | None,
    count: int,
) -> np.ndarray:
    if pairs is None:
        values = list(combinations(range(count), 2))
    else:
        values = []
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("Every pair must contain exactly two CLV indices.")
            first, second = pair
            if isinstance(first, (bool, np.bool_)) or isinstance(second, (bool, np.bool_)):
                raise TypeError("CLV pair indices must be integers.")
            if not isinstance(first, (int, np.integer)) or not isinstance(
                second, (int, np.integer)
            ):
                raise TypeError("CLV pair indices must be integers.")
            item = (int(first), int(second))
            if item[0] == item[1] or min(item) < 0 or max(item) >= count:
                raise ValueError("CLV pair indices must be distinct and in range.")
            values.append(item)
    return (
        np.asarray(values, dtype=int).reshape(-1, 2)
        if values
        else np.empty((0, 2), dtype=int)
    )


def _canonical_subspaces(
    subspaces: Sequence[Sequence[Sequence[int]]] | None,
    count: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    if subspaces is None:
        return ()
    canonical: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for pair in subspaces:
        if len(pair) != 2:
            raise ValueError("Each subspace pair must contain two index sequences.")
        sides: list[tuple[int, ...]] = []
        for side in pair:
            if len(side) == 0:
                raise ValueError("Subspace index sequences cannot be empty.")
            indices: list[int] = []
            for value in side:
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, np.integer)
                ):
                    raise TypeError("Subspace indices must be integers.")
                index = int(value)
                if index < 0 or index >= count:
                    raise ValueError("Subspace index is out of range.")
                indices.append(index)
            if len(set(indices)) != len(indices):
                raise ValueError("A subspace index sequence cannot contain duplicates.")
            sides.append(tuple(indices))
        canonical.append((sides[0], sides[1]))
    return tuple(canonical)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.shape[1] == 0:
        return np.empty((values.shape[0] - window + 1, 0), dtype=float)
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
    return (cumulative[window:] - cumulative[:-window]) / float(window)


def covariant_lyapunov_angles(
    vectors: np.ndarray,
    *,
    coordinates: np.ndarray | Sequence[float] | None = None,
    pairs: Sequence[Sequence[int]] | None = None,
    subspaces: Sequence[Sequence[Sequence[int]]] | None = None,
    unoriented: bool = True,
    window: int | None = None,
) -> CovariantAngleResult:
    """Compute stable pair and minimum-principal-subspace angles.

    This is a geometric postprocessor.  It does not assert that an arbitrary
    supplied history is a valid fractional or integer CLV history.
    """

    history = np.asarray(vectors, dtype=float)
    if history.ndim != 3 or min(history.shape) == 0:
        raise ValueError("vectors must have shape (samples, n_vectors, dimension).")
    if not np.all(np.isfinite(history)):
        raise ValueError("vectors must contain only finite values.")
    samples, count, dimension = history.shape
    norms = np.linalg.norm(history, axis=2)
    if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError("vectors contains a zero or non-finite CLV direction.")
    unit = history / norms[:, :, None]
    if coordinates is None:
        axis = np.arange(samples, dtype=float)
    else:
        axis = np.asarray(coordinates, dtype=float)
        if axis.shape != (samples,) or not np.all(np.isfinite(axis)):
            raise ValueError("coordinates must be a finite (samples,) vector.")
        if samples > 1 and np.any(np.diff(axis) <= 0.0):
            raise ValueError("coordinates must be strictly increasing.")
    canonical_pairs = _canonical_pairs(pairs, count)
    canonical_subspaces = _canonical_subspaces(subspaces, count)
    pair_values = np.empty((samples, canonical_pairs.shape[0]), dtype=float)
    for column, (first, second) in enumerate(canonical_pairs):
        dot = np.einsum("ij,ij->i", unit[:, first, :], unit[:, second, :])
        cosine = np.abs(dot) if unoriented else dot
        cosine = np.clip(cosine, -1.0, 1.0)
        sine = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
        pair_values[:, column] = np.arctan2(sine, cosine)
    subspace_values = np.empty((samples, len(canonical_subspaces)), dtype=float)
    for sample in range(samples):
        for column, (first, second) in enumerate(canonical_subspaces):
            angles = subspace_angles(
                unit[sample, np.asarray(first), :].T,
                unit[sample, np.asarray(second), :].T,
            )
            subspace_values[sample, column] = float(np.min(angles))
    if window is None:
        window_size = 1
    else:
        window_size = _checked_positive_integer(window, "window")
        if window_size > samples:
            raise ValueError("window cannot exceed the number of samples.")
    if window_size == 1:
        window_axis = axis.copy()
        window_pairs = pair_values.copy()
        window_subspaces = subspace_values.copy()
    else:
        window_axis = np.convolve(axis, np.ones(window_size) / window_size, mode="valid")
        window_pairs = _rolling_mean(pair_values, window_size)
        window_subspaces = _rolling_mean(subspace_values, window_size)
    return CovariantAngleResult(
        coordinates=axis,
        pair_indices=canonical_pairs,
        pair_angles=pair_values,
        subspace_pairs=canonical_subspaces,
        subspace_angles=subspace_values,
        window_coordinates=np.asarray(window_axis, dtype=float),
        window_pair_angles=np.asarray(window_pairs, dtype=float),
        window_subspace_angles=np.asarray(window_subspaces, dtype=float),
        metadata={
            "input_layout": "samples_n_vectors_dimension",
            "dimension": int(dimension),
            "n_vectors": int(count),
            "unoriented": bool(unoriented),
            "pair_formula": "atan2(sqrt(max(0,1-cosine^2)),cosine)",
            "subspace_method": "scipy.linalg.subspace_angles_minimum",
            "window": int(window_size),
            "window_policy": "centered_coordinate_rolling_mean_valid",
            "evidence_boundary": (
                "geometric postprocessing only; validity of supplied CLVs is not inferred"
            ),
        },
    )


def _finite_difference_jacobian(
    callback: Callable[[float, np.ndarray], np.ndarray],
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
        matrix[:, column] = (
            callback(coordinate, state + perturbation)
            - callback(coordinate, state - perturbation)
        ) / (2.0 * component_step)
    return matrix


def _degeneracy_warning(
    exponents: np.ndarray,
    tolerance: float,
) -> tuple[bool, tuple[str, ...]]:
    if exponents.size < 2 or not np.all(np.isfinite(exponents)):
        return False, _BASE_WARNINGS
    near = bool(np.any(np.abs(np.diff(exponents)) <= tolerance))
    return near, _BASE_WARNINGS + ((_DEGENERACY_WARNING,) if near else ())


def _failure_result(
    *,
    dimension: int,
    count: int,
    state: np.ndarray,
    status: str,
    error_message: str,
    metadata: Mapping[str, Any],
    method_id: str,
    system_kind: str,
    coordinate_kind: str,
    backend: str,
    propagation_backend: str,
    singular_segments: np.ndarray | None = None,
) -> CovariantLyapunovResult:
    return CovariantLyapunovResult(
        coordinates=np.empty(0, dtype=float),
        sampled_states=np.empty((0, dimension), dtype=float),
        vectors=np.empty((0, count, dimension), dtype=float),
        exponents=np.full(count, np.nan),
        convergence=np.empty((0, count), dtype=float),
        singular_segments=(
            np.empty(0, dtype=int)
            if singular_segments is None
            else np.asarray(singular_segments, dtype=int)
        ),
        status=status,
        final_state=np.asarray(state, dtype=float).copy(),
        future_state=np.asarray(state, dtype=float).copy(),
        error_message=error_message,
        metadata=dict(metadata),
        method_id=method_id,
        system_kind=system_kind,
        coordinate_kind=coordinate_kind,
        backend=backend,
        propagation_backend=propagation_backend,
    )


def integer_flow_covariant_lyapunov_vectors(
    rhs: Callable[..., Any],
    jacobian: Callable[..., Any] | None,
    x0: np.ndarray | Sequence[float],
    *,
    t_final: float,
    forward_transient_time: float = 0.0,
    backward_transient_time: float = 0.0,
    t_burn: float = 0.0,
    qr_interval: float = 0.5,
    n_vectors: int | None = None,
    initial_basis: np.ndarray | None = None,
    terminal_coefficients: np.ndarray | None = None,
    seed: int = 0,
    parameters: Any = None,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-12,
    max_step: float = np.inf,
    jacobian_eps: float = 1.0e-6,
    div_threshold: float | None = None,
    backend: str = "auto",
    singular_tolerance: float | None = None,
    degeneracy_tolerance: float = 1.0e-8,
    max_workspace_bytes: int | None = _DEFAULT_MAX_WORKSPACE_BYTES,
    q: float | Sequence[float] | np.ndarray = 1.0,
) -> CovariantLyapunovResult:
    """Compute finite-time q=1 flow CLVs with DOP853 and Ginelli recursion."""

    routine = "integer_flow_covariant_lyapunov_vectors"
    _require_integer_order(q, routine)
    state = _checked_state(x0)
    dimension = state.size
    observation_time = _checked_positive(t_final, "t_final")
    forward_time = _checked_nonnegative(forward_transient_time, "forward_transient_time")
    backward_time = _checked_nonnegative(
        backward_transient_time, "backward_transient_time"
    )
    burn_time = _checked_nonnegative(t_burn, "t_burn")
    interval = _checked_positive(qr_interval, "qr_interval")
    relative_tolerance = _checked_positive(rtol, "rtol")
    absolute_tolerance = _checked_positive(atol, "atol")
    finite_difference_step = _checked_positive(jacobian_eps, "jacobian_eps")
    degeneracy_threshold = _checked_positive(
        degeneracy_tolerance, "degeneracy_tolerance"
    )
    maximum_step = float(max_step)
    if np.isnan(maximum_step) or maximum_step <= 0.0:
        raise ValueError("max_step must be positive.")
    divergence = None if div_threshold is None else _checked_positive(
        div_threshold, "div_threshold"
    )
    basis, count = _prepare_basis(
        dimension, n_vectors, initial_basis, seed
    )
    maximum_workspace = _checked_workspace_limit(max_workspace_bytes)
    observed_segments = _segment_count(observation_time, interval)
    future_segments = _segment_count(backward_time, interval)
    estimated_workspace = _estimate_workspace_bytes(
        observed_segments, future_segments, dimension, count
    )
    _enforce_workspace_limit(estimated_workspace, maximum_workspace)
    resolved_backend = _resolve_backend(backend)
    bound_rhs = bind_rhs(rhs, parameters)
    bound_jacobian = None if jacobian is None else bind_rhs(jacobian, parameters)
    counters = {
        "rhs_calls": 0,
        "jacobian_calls": 0,
        "solver_nfev": 0,
        "solver_njev": 0,
        "solver_nlu": 0,
        "burn_segments": 0,
        "forward_segments": 0,
        "observed_segments": 0,
        "backward_segments": 0,
    }
    metadata_base: dict[str, Any] = {
        "solver": "scipy.integrate.solve_ivp",
        "solver_method": "DOP853",
        "dimension": int(dimension),
        "n_vectors": int(count),
        "t_burn_requested": burn_time,
        "forward_transient_time_requested": forward_time,
        "observation_time_requested": observation_time,
        "backward_transient_time_requested": backward_time,
        "qr_interval": interval,
        "rtol": relative_tolerance,
        "atol": absolute_tolerance,
        "max_step": maximum_step,
        "jacobian_source": (
            "analytic" if jacobian is not None else "central_relative_componentwise"
        ),
        "jacobian_eps": finite_difference_step if jacobian is None else None,
        "div_threshold": divergence,
        "backward_backend": resolved_backend,
        "propagation_backend": "scipy_dop853",
        "estimated_workspace_bytes": estimated_workspace,
        "max_workspace_bytes": maximum_workspace,
        "auto_transient_stopping": False,
        "transient_policy": "explicit_forward_and_backward_horizons",
        "transient_reference_doi": "10.1016/j.physd.2026.135237",
        "evidence_boundary": (
            "finite q=1 tangent-cocycle diagnostic; no fractional/history-space claim"
        ),
    }

    def evaluate(time: float, point: np.ndarray) -> np.ndarray:
        counters["rhs_calls"] += 1
        try:
            value = np.asarray(bound_rhs(time, point), dtype=float)
        except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
            raise _InvalidRhsError(f"rhs evaluation failed: {exc}") from exc
        if value.shape != (dimension,) or not np.all(np.isfinite(value)):
            raise _InvalidRhsError(
                f"rhs must return a finite ({dimension},) vector; got {value.shape}."
            )
        return value

    def evaluate_jacobian(time: float, point: np.ndarray) -> np.ndarray:
        counters["jacobian_calls"] += 1
        try:
            if bound_jacobian is None:
                matrix = _finite_difference_jacobian(
                    evaluate, time, point, finite_difference_step
                )
            else:
                matrix = np.asarray(bound_jacobian(time, point), dtype=float)
        except _InvalidRhsError:
            raise
        except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
            raise _InvalidJacobianError(f"jacobian evaluation failed: {exc}") from exc
        if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
            raise _InvalidJacobianError(
                "jacobian must return a finite "
                f"({dimension}, {dimension}) matrix; got {matrix.shape}."
            )
        return matrix

    def event(_time: float, current: np.ndarray) -> float:
        assert divergence is not None
        return divergence - float(np.linalg.norm(current[:dimension]))

    event.direction = -1.0  # type: ignore[attr-defined]
    event.terminal = True  # type: ignore[attr-defined]
    events = event if divergence is not None else None

    def integrate_state(duration: float, time_value: float, phase: str) -> tuple[np.ndarray, float]:
        if duration == 0.0:
            return state.copy(), time_value
        solved = solve_ivp(
            evaluate,
            (time_value, time_value + duration),
            state,
            method="DOP853",
            events=events,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            max_step=maximum_step,
        )
        counters["solver_nfev"] += int(solved.nfev)
        counters["solver_njev"] += int(solved.njev)
        counters["solver_nlu"] += int(solved.nlu)
        if divergence is not None and solved.t_events and solved.t_events[0].size:
            crossed = np.asarray(solved.y_events[0][0][:dimension], dtype=float)
            raise _EvolutionFailure(f"{phase}_diverged", "state crossed div_threshold", crossed)
        if not solved.success:
            last = np.asarray(solved.y[:dimension, -1], dtype=float)
            raise _EvolutionFailure(f"{phase}_solver_failure", str(solved.message), last)
        last = np.asarray(solved.y[:dimension, -1], dtype=float)
        if not np.all(np.isfinite(last)):
            raise _EvolutionFailure(
                f"{phase}_nonfinite", "integration produced a non-finite state", last
            )
        return last, time_value + duration

    def integrate_tangent_segment(
        current_state: np.ndarray,
        current_basis: np.ndarray,
        duration: float,
        time_value: float,
        phase: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        def augmented_rhs(local_time: float, augmented: np.ndarray) -> np.ndarray:
            point = augmented[:dimension]
            tangent = augmented[dimension:].reshape(dimension, count)
            return np.concatenate(
                (evaluate(local_time, point), (evaluate_jacobian(local_time, point) @ tangent).ravel())
            )

        augmented0 = np.concatenate((current_state, current_basis.ravel()))
        solved = solve_ivp(
            augmented_rhs,
            (time_value, time_value + duration),
            augmented0,
            method="DOP853",
            events=events,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            max_step=maximum_step,
        )
        counters["solver_nfev"] += int(solved.nfev)
        counters["solver_njev"] += int(solved.njev)
        counters["solver_nlu"] += int(solved.nlu)
        if divergence is not None and solved.t_events and solved.t_events[0].size:
            crossed = np.asarray(solved.y_events[0][0][:dimension], dtype=float)
            raise _EvolutionFailure(f"{phase}_diverged", "state crossed div_threshold", crossed)
        if not solved.success:
            last = np.asarray(solved.y[:dimension, -1], dtype=float)
            raise _EvolutionFailure(f"{phase}_solver_failure", str(solved.message), last)
        last_state = np.asarray(solved.y[:dimension, -1], dtype=float)
        tangent = np.asarray(solved.y[dimension:, -1], dtype=float).reshape(
            dimension, count
        )
        if not np.all(np.isfinite(last_state)) or not np.all(np.isfinite(tangent)):
            raise _EvolutionFailure(
                f"{phase}_nonfinite", "augmented integration produced non-finite values", last_state
            )
        try:
            next_basis, upper = _positive_reduced_qr(tangent)
        except np.linalg.LinAlgError as exc:
            raise _EvolutionFailure("singular_cocycle", str(exc), last_state) from exc
        return last_state, next_basis, upper, time_value + duration

    absolute_time = 0.0
    if divergence is not None and np.linalg.norm(state) >= divergence:
        metadata = dict(metadata_base)
        metadata.update(counters)
        return _failure_result(
            dimension=dimension,
            count=count,
            state=state,
            status="burn_diverged",
            error_message="initial state meets or exceeds div_threshold",
            metadata=metadata,
            method_id="integer_flow_ginelli_clv",
            system_kind="flow",
            coordinate_kind="time_after_forward_transient",
            backend=resolved_backend,
            propagation_backend="scipy_dop853",
        )
    try:
        if burn_time:
            state, absolute_time = integrate_state(burn_time, absolute_time, "burn")
            counters["burn_segments"] = 1
        elapsed = 0.0
        while elapsed < forward_time - 16.0 * np.finfo(float).eps * max(1.0, forward_time):
            duration = min(interval, forward_time - elapsed)
            state, basis, _upper, absolute_time = integrate_tangent_segment(
                state, basis, duration, absolute_time, "forward_transient"
            )
            elapsed += duration
            counters["forward_segments"] += 1

        bases = [basis.copy()]
        states = [state.copy()]
        coordinates = [0.0]
        observed_r_values: list[np.ndarray] = []
        convergence: list[np.ndarray] = []
        log_sums = np.zeros(count, dtype=float)
        elapsed_observed = 0.0
        while elapsed_observed < observation_time - 16.0 * np.finfo(float).eps * observation_time:
            duration = min(interval, observation_time - elapsed_observed)
            state, basis, upper, absolute_time = integrate_tangent_segment(
                state, basis, duration, absolute_time, "observation"
            )
            observed_r_values.append(upper)
            log_sums += np.log(np.diag(upper))
            elapsed_observed += duration
            if observation_time - elapsed_observed <= 16.0 * np.finfo(float).eps * observation_time:
                elapsed_observed = observation_time
            bases.append(basis.copy())
            states.append(state.copy())
            coordinates.append(float(elapsed_observed))
            convergence.append(log_sums / elapsed_observed)
            counters["observed_segments"] += 1
        final_state = state.copy()

        future_r_values: list[np.ndarray] = []
        elapsed_future = 0.0
        while elapsed_future < backward_time - 16.0 * np.finfo(float).eps * max(1.0, backward_time):
            duration = min(interval, backward_time - elapsed_future)
            state, basis, upper, absolute_time = integrate_tangent_segment(
                state, basis, duration, absolute_time, "backward_transient"
            )
            future_r_values.append(upper)
            elapsed_future += duration
            counters["backward_segments"] += 1
        future_state = state.copy()
    except (_InvalidRhsError, _InvalidJacobianError) as exc:
        metadata = dict(metadata_base)
        metadata.update(counters)
        return _failure_result(
            dimension=dimension,
            count=count,
            state=state,
            status="invalid_callback",
            error_message=str(exc),
            metadata=metadata,
            method_id="integer_flow_ginelli_clv",
            system_kind="flow",
            coordinate_kind="time_after_forward_transient",
            backend=resolved_backend,
            propagation_backend="scipy_dop853",
        )
    except _EvolutionFailure as exc:
        metadata = dict(metadata_base)
        metadata.update(counters)
        return _failure_result(
            dimension=dimension,
            count=count,
            state=exc.state,
            status=exc.status,
            error_message=str(exc),
            metadata=metadata,
            method_id="integer_flow_ginelli_clv",
            system_kind="flow",
            coordinate_kind="time_after_forward_transient",
            backend=resolved_backend,
            propagation_backend="scipy_dop853",
        )
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        metadata = dict(metadata_base)
        metadata.update(counters)
        return _failure_result(
            dimension=dimension,
            count=count,
            state=state,
            status="solver_exception",
            error_message=str(exc),
            metadata=metadata,
            method_id="integer_flow_ginelli_clv",
            system_kind="flow",
            coordinate_kind="time_after_forward_transient",
            backend=resolved_backend,
            propagation_backend="scipy_dop853",
        )

    bases_array = np.ascontiguousarray(np.stack(bases))
    observed_array = np.ascontiguousarray(np.stack(observed_r_values))
    future_array = (
        np.ascontiguousarray(np.stack(future_r_values))
        if future_r_values
        else np.empty((0, count, count), dtype=float)
    )
    singular = _singular_segments(observed_array, future_array, singular_tolerance)
    metadata = dict(metadata_base)
    metadata.update(counters)
    metadata.update(
        {
            "t_burn_completed": burn_time,
            "forward_transient_time_completed": forward_time,
            "observation_time_completed": observation_time,
            "backward_transient_time_completed": backward_time,
            "absolute_final_time": absolute_time,
        }
    )
    if singular.size:
        return _failure_result(
            dimension=dimension,
            count=count,
            state=future_state,
            status="singular_cocycle",
            error_message=f"near-singular R factors at combined indices {singular.tolist()}",
            metadata=metadata,
            method_id="integer_flow_ginelli_clv",
            system_kind="flow",
            coordinate_kind="time_after_forward_transient",
            backend=resolved_backend,
            propagation_backend="scipy_dop853",
            singular_segments=singular,
        )
    history = integer_covariant_vectors_from_qr_history(
        bases_array,
        observed_array,
        future_r_factors=future_array,
        terminal_coefficients=terminal_coefficients,
        seed=seed + 1,
        backend=resolved_backend,
        singular_tolerance=singular_tolerance,
        max_workspace_bytes=maximum_workspace,
        q=1.0,
    )
    exponents = log_sums / observation_time
    near_degenerate, warnings = _degeneracy_warning(
        exponents, degeneracy_threshold
    )
    metadata["near_degenerate_finite_time_spectrum"] = near_degenerate
    metadata["degeneracy_tolerance"] = degeneracy_threshold
    return CovariantLyapunovResult(
        coordinates=np.asarray(coordinates, dtype=float),
        sampled_states=np.stack(states),
        vectors=history.vectors,
        exponents=np.asarray(exponents, dtype=float),
        convergence=np.asarray(convergence, dtype=float),
        singular_segments=np.empty(0, dtype=int),
        status="ok",
        final_state=final_state,
        future_state=future_state,
        metadata=metadata,
        method_id="integer_flow_ginelli_clv",
        system_kind="flow",
        coordinate_kind="time_after_forward_transient",
        backend=history.backend,
        propagation_backend="scipy_dop853",
        methodological_warnings=warnings,
    )


def integer_map_covariant_lyapunov_vectors(
    map_function: Callable[..., Any],
    jacobian: Callable[..., Any] | None,
    x0: np.ndarray | Sequence[float],
    *,
    iterations: int,
    forward_transient_iterations: int = 0,
    backward_transient_iterations: int = 0,
    transient_iterations: int = 0,
    qr_interval_iterations: int = 1,
    n_vectors: int | None = None,
    initial_basis: np.ndarray | None = None,
    terminal_coefficients: np.ndarray | None = None,
    seed: int = 0,
    parameters: Any = None,
    jacobian_eps: float = 1.0e-6,
    div_threshold: float | None = None,
    backend: str = "auto",
    singular_tolerance: float | None = None,
    degeneracy_tolerance: float = 1.0e-8,
    max_workspace_bytes: int | None = _DEFAULT_MAX_WORKSPACE_BYTES,
    q: float | Sequence[float] | np.ndarray = 1.0,
) -> CovariantLyapunovResult:
    """Compute finite-time q=1 map CLVs with exact Jacobian recurrence."""

    routine = "integer_map_covariant_lyapunov_vectors"
    _require_integer_order(q, routine)
    state = _checked_state(x0)
    dimension = state.size
    observed_iterations = _checked_positive_integer(iterations, "iterations")
    forward_iterations = _checked_nonnegative_integer(
        forward_transient_iterations, "forward_transient_iterations"
    )
    backward_iterations = _checked_nonnegative_integer(
        backward_transient_iterations, "backward_transient_iterations"
    )
    state_transient = _checked_nonnegative_integer(
        transient_iterations, "transient_iterations"
    )
    qr_interval = _checked_positive_integer(
        qr_interval_iterations, "qr_interval_iterations"
    )
    finite_difference_step = _checked_positive(jacobian_eps, "jacobian_eps")
    degeneracy_threshold = _checked_positive(
        degeneracy_tolerance, "degeneracy_tolerance"
    )
    divergence = None if div_threshold is None else _checked_positive(
        div_threshold, "div_threshold"
    )
    basis, count = _prepare_basis(dimension, n_vectors, initial_basis, seed)
    observed_segments = int(np.ceil(observed_iterations / qr_interval))
    future_segments = int(np.ceil(backward_iterations / qr_interval))
    maximum_workspace = _checked_workspace_limit(max_workspace_bytes)
    estimated_workspace = _estimate_workspace_bytes(
        observed_segments, future_segments, dimension, count
    )
    _enforce_workspace_limit(estimated_workspace, maximum_workspace)
    resolved_backend = _resolve_backend(backend)
    bound_map = bind_rhs(map_function, parameters)
    bound_jacobian = None if jacobian is None else bind_rhs(jacobian, parameters)
    counters = {"map_calls": 0, "jacobian_calls": 0, "map_iterations": 0}
    metadata_base: dict[str, Any] = {
        "dimension": int(dimension),
        "n_vectors": int(count),
        "transient_iterations_requested": state_transient,
        "forward_transient_iterations_requested": forward_iterations,
        "observation_iterations_requested": observed_iterations,
        "backward_transient_iterations_requested": backward_iterations,
        "qr_interval_iterations": qr_interval,
        "jacobian_source": (
            "analytic" if jacobian is not None else "central_relative_componentwise"
        ),
        "jacobian_eps": finite_difference_step if jacobian is None else None,
        "div_threshold": divergence,
        "backward_backend": resolved_backend,
        "propagation_backend": "exact_map_jacobian_recurrence",
        "estimated_workspace_bytes": estimated_workspace,
        "max_workspace_bytes": maximum_workspace,
        "auto_transient_stopping": False,
        "transient_policy": "explicit_forward_and_backward_horizons",
        "transient_reference_doi": "10.1016/j.physd.2026.135237",
        "evidence_boundary": (
            "finite q=1 tangent-cocycle diagnostic; no fractional-difference/history claim"
        ),
    }

    def evaluate(iteration: float, point: np.ndarray) -> np.ndarray:
        counters["map_calls"] += 1
        try:
            value = np.asarray(bound_map(iteration, point), dtype=float)
        except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
            raise _InvalidRhsError(f"map evaluation failed: {exc}") from exc
        if value.shape != (dimension,) or not np.all(np.isfinite(value)):
            raise _InvalidRhsError(
                f"map_function must return a finite ({dimension},) vector; got {value.shape}."
            )
        return value

    def evaluate_jacobian(iteration: float, point: np.ndarray) -> np.ndarray:
        counters["jacobian_calls"] += 1
        try:
            if bound_jacobian is None:
                matrix = _finite_difference_jacobian(
                    evaluate, iteration, point, finite_difference_step
                )
            else:
                matrix = np.asarray(bound_jacobian(iteration, point), dtype=float)
        except _InvalidRhsError:
            raise
        except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
            raise _InvalidJacobianError(f"jacobian evaluation failed: {exc}") from exc
        if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
            raise _InvalidJacobianError(
                "jacobian must return a finite "
                f"({dimension}, {dimension}) matrix; got {matrix.shape}."
            )
        return matrix

    coordinate = 0

    def state_step(current: np.ndarray, phase: str) -> np.ndarray:
        nonlocal coordinate
        updated = evaluate(float(coordinate), current)
        coordinate += 1
        counters["map_iterations"] += 1
        if divergence is not None and np.linalg.norm(updated) >= divergence:
            raise _EvolutionFailure(f"{phase}_diverged", "state meets div_threshold", updated)
        return updated

    def tangent_chunk(
        current: np.ndarray,
        current_basis: np.ndarray,
        steps: int,
        phase: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal coordinate
        tangent = current_basis.copy()
        point = current.copy()
        for _ in range(steps):
            matrix = evaluate_jacobian(float(coordinate), point)
            tangent = matrix @ tangent
            point = evaluate(float(coordinate), point)
            coordinate += 1
            counters["map_iterations"] += 1
            if divergence is not None and np.linalg.norm(point) >= divergence:
                raise _EvolutionFailure(f"{phase}_diverged", "state meets div_threshold", point)
            if not np.all(np.isfinite(tangent)):
                raise _EvolutionFailure(
                    f"{phase}_nonfinite", "tangent recurrence produced non-finite values", point
                )
        try:
            next_basis, upper = _positive_reduced_qr(tangent)
        except np.linalg.LinAlgError as exc:
            raise _EvolutionFailure("singular_cocycle", str(exc), point) from exc
        return point, next_basis, upper

    if divergence is not None and np.linalg.norm(state) >= divergence:
        metadata = dict(metadata_base)
        metadata.update(counters)
        return _failure_result(
            dimension=dimension,
            count=count,
            state=state,
            status="transient_diverged",
            error_message="initial state meets or exceeds div_threshold",
            metadata=metadata,
            method_id="integer_map_ginelli_clv",
            system_kind="map",
            coordinate_kind="iteration_after_forward_transient",
            backend=resolved_backend,
            propagation_backend="exact_map_jacobian_recurrence",
        )
    try:
        for _ in range(state_transient):
            state = state_step(state, "transient")
        completed = 0
        while completed < forward_iterations:
            chunk = min(qr_interval, forward_iterations - completed)
            state, basis, _upper = tangent_chunk(
                state, basis, chunk, "forward_transient"
            )
            completed += chunk

        bases = [basis.copy()]
        states = [state.copy()]
        coordinates = [0.0]
        observed_r_values: list[np.ndarray] = []
        convergence: list[np.ndarray] = []
        log_sums = np.zeros(count, dtype=float)
        completed = 0
        while completed < observed_iterations:
            chunk = min(qr_interval, observed_iterations - completed)
            state, basis, upper = tangent_chunk(state, basis, chunk, "observation")
            observed_r_values.append(upper)
            log_sums += np.log(np.diag(upper))
            completed += chunk
            bases.append(basis.copy())
            states.append(state.copy())
            coordinates.append(float(completed))
            convergence.append(log_sums / completed)
        final_state = state.copy()

        future_r_values: list[np.ndarray] = []
        completed_future = 0
        while completed_future < backward_iterations:
            chunk = min(qr_interval, backward_iterations - completed_future)
            state, basis, upper = tangent_chunk(
                state, basis, chunk, "backward_transient"
            )
            future_r_values.append(upper)
            completed_future += chunk
        future_state = state.copy()
    except (_InvalidRhsError, _InvalidJacobianError) as exc:
        metadata = dict(metadata_base)
        metadata.update(counters)
        return _failure_result(
            dimension=dimension,
            count=count,
            state=state,
            status="invalid_callback",
            error_message=str(exc),
            metadata=metadata,
            method_id="integer_map_ginelli_clv",
            system_kind="map",
            coordinate_kind="iteration_after_forward_transient",
            backend=resolved_backend,
            propagation_backend="exact_map_jacobian_recurrence",
        )
    except _EvolutionFailure as exc:
        metadata = dict(metadata_base)
        metadata.update(counters)
        return _failure_result(
            dimension=dimension,
            count=count,
            state=exc.state,
            status=exc.status,
            error_message=str(exc),
            metadata=metadata,
            method_id="integer_map_ginelli_clv",
            system_kind="map",
            coordinate_kind="iteration_after_forward_transient",
            backend=resolved_backend,
            propagation_backend="exact_map_jacobian_recurrence",
        )
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        metadata = dict(metadata_base)
        metadata.update(counters)
        return _failure_result(
            dimension=dimension,
            count=count,
            state=state,
            status="map_exception",
            error_message=str(exc),
            metadata=metadata,
            method_id="integer_map_ginelli_clv",
            system_kind="map",
            coordinate_kind="iteration_after_forward_transient",
            backend=resolved_backend,
            propagation_backend="exact_map_jacobian_recurrence",
        )

    bases_array = np.ascontiguousarray(np.stack(bases))
    observed_array = np.ascontiguousarray(np.stack(observed_r_values))
    future_array = (
        np.ascontiguousarray(np.stack(future_r_values))
        if future_r_values
        else np.empty((0, count, count), dtype=float)
    )
    singular = _singular_segments(observed_array, future_array, singular_tolerance)
    metadata = dict(metadata_base)
    metadata.update(counters)
    metadata["total_coordinate_iterations"] = int(coordinate)
    if singular.size:
        return _failure_result(
            dimension=dimension,
            count=count,
            state=future_state,
            status="singular_cocycle",
            error_message=f"near-singular R factors at combined indices {singular.tolist()}",
            metadata=metadata,
            method_id="integer_map_ginelli_clv",
            system_kind="map",
            coordinate_kind="iteration_after_forward_transient",
            backend=resolved_backend,
            propagation_backend="exact_map_jacobian_recurrence",
            singular_segments=singular,
        )
    history = integer_covariant_vectors_from_qr_history(
        bases_array,
        observed_array,
        future_r_factors=future_array,
        terminal_coefficients=terminal_coefficients,
        seed=seed + 1,
        backend=resolved_backend,
        singular_tolerance=singular_tolerance,
        max_workspace_bytes=maximum_workspace,
        q=1.0,
    )
    exponents = log_sums / observed_iterations
    near_degenerate, warnings = _degeneracy_warning(exponents, degeneracy_threshold)
    metadata["near_degenerate_finite_time_spectrum"] = near_degenerate
    metadata["degeneracy_tolerance"] = degeneracy_threshold
    return CovariantLyapunovResult(
        coordinates=np.asarray(coordinates, dtype=float),
        sampled_states=np.stack(states),
        vectors=history.vectors,
        exponents=np.asarray(exponents, dtype=float),
        convergence=np.asarray(convergence, dtype=float),
        singular_segments=np.empty(0, dtype=int),
        status="ok",
        final_state=final_state,
        future_state=future_state,
        metadata=metadata,
        method_id="integer_map_ginelli_clv",
        system_kind="map",
        coordinate_kind="iteration_after_forward_transient",
        backend=history.backend,
        propagation_backend="exact_map_jacobian_recurrence",
        methodological_warnings=warnings,
    )


def _declared_system_order(system: object) -> Any:
    for attribute in ("q", "order", "fractional_order"):
        value = getattr(system, attribute, None)
        if value is not None:
            return value
    for attribute in ("metadata", "parameters", "params"):
        mapping = getattr(system, attribute, None)
        if isinstance(mapping, Mapping) and mapping.get("q") is not None:
            return mapping["q"]
    return None


def integer_system_covariant_lyapunov_vectors(
    system: object,
    x0: np.ndarray | Sequence[float],
    *,
    q: float | Sequence[float] | np.ndarray = 1.0,
    **kwargs: Any,
) -> CovariantLyapunovResult:
    """Dispatch q=1 CLVs for a HAFO-compatible flow or map object."""

    _require_integer_order(q, "integer_system_covariant_lyapunov_vectors")
    declared = _declared_system_order(system)
    if declared is not None:
        _require_integer_order(declared, "integer_system_covariant_lyapunov_vectors")
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
        callback = lambda state: evaluate(state)
        jacobian = (lambda state: jacobian_matrix(state)) if use_analytic else None
    else:
        callback = lambda state: evaluate(state, parameter_overrides)
        jacobian = (
            (lambda state: jacobian_matrix(state, parameter_overrides))
            if use_analytic
            else None
        )
    if kind == "flow":
        return integer_flow_covariant_lyapunov_vectors(
            callback, jacobian, x0, q=1.0, **options
        )
    return integer_map_covariant_lyapunov_vectors(
        callback, jacobian, x0, q=1.0, **options
    )


__all__ = [
    "CovariantAngleResult",
    "CovariantLyapunovResult",
    "CovariantQRHistoryResult",
    "NUMBA_AVAILABLE",
    "covariant_lyapunov_angles",
    "integer_covariant_vectors_from_qr_history",
    "integer_flow_covariant_lyapunov_vectors",
    "integer_map_covariant_lyapunov_vectors",
    "integer_system_covariant_lyapunov_vectors",
]

