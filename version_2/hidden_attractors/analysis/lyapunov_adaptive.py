"""Adaptive integer-order variational Lyapunov estimates.

This module provides an independent DOP853 implementation of the classical
Benettin/QR algorithm.  It is intentionally separate from the maintained
fixed-step EFORK q=1 estimator in :mod:`hidden_attractors.analysis.lyapunov`.
Neither implementation is a fractional-memory method and neither certifies
chaos or hiddenness by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from scipy.integrate import solve_ivp

from .lyapunov import finite_difference_jacobian


_ADAPTIVE_QR_REFERENCES: tuple[str, ...] = (
    "Benettin et al. 1980 - Lyapunov characteristic exponents (Meccanica 15)",
    "Hairer, Norsett & Wanner 1993 - Solving Ordinary Differential Equations I",
)

_ADAPTIVE_QR_WARNINGS: tuple[str, ...] = (
    "Valid only for q=1 memoryless ordinary differential equations.",
    "The exponents are finite-time local estimates and require convergence checks.",
    "A positive estimate alone does not certify chaos or hiddenness.",
)


class _InvalidRhsError(ValueError):
    """Internal marker for a malformed vector-field evaluation."""


class _InvalidJacobianError(ValueError):
    """Internal marker for a malformed Jacobian evaluation."""


@dataclass(frozen=True)
class AdaptiveLyapunovResult:
    """Finite-time q=1 DOP853 variational QR result.

    ``times`` and ``convergence`` contain one row per completed QR segment;
    their time origin is the end of burn-in.  Runtime failures are recorded in
    ``status`` and ``error_message``.  Invalid user configuration raises
    :class:`ValueError` before integration starts.
    """

    exponents: np.ndarray
    times: np.ndarray
    convergence: np.ndarray
    status: str
    final_state: np.ndarray
    accumulated_time: float
    error_message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    method_id: str = "integer_dop853_variational_qr"
    derivative_model: str = "integer"
    q: float = 1.0
    finite_time_local: bool = True
    orthonormalization: str = "qr"
    reference_ids: tuple[str, ...] = field(default_factory=lambda: _ADAPTIVE_QR_REFERENCES)
    methodological_warnings: tuple[str, ...] = field(
        default_factory=lambda: _ADAPTIVE_QR_WARNINGS
    )

    @property
    def sum_exponents(self) -> float:
        """Return the finite-time sum, or NaN when no estimate is available."""

        if np.all(np.isfinite(self.exponents)):
            return float(np.sum(self.exponents))
        return float("nan")


def _infer_system_order(system: object) -> float | None:
    """Best-effort extraction of an order declaration from a system object."""

    for attribute in ("q", "order", "fractional_order"):
        value = getattr(system, attribute, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    for attribute in ("metadata", "parameters", "params"):
        mapping = getattr(system, attribute, None)
        if isinstance(mapping, Mapping) and mapping.get("q") is not None:
            try:
                return float(mapping["q"])
            except (TypeError, ValueError):
                pass
    return None


def integer_dop853_variational_qr(
    rhs: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray] | None,
    x0: np.ndarray,
    *,
    t_accumulate: float,
    t_burn: float = 0.0,
    qr_interval: float = 0.5,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-12,
    max_step: float = np.inf,
    jacobian_eps: float = 1.0e-6,
    div_threshold: float | None = None,
    q: float = 1.0,
) -> AdaptiveLyapunovResult:
    """Estimate all q=1 Lyapunov exponents with DOP853 and QR.

    Burn-in is integrated as a state-only interval.  The tangent basis is
    initialized *after* that interval, so ``t_burn`` need not be aligned with
    ``qr_interval`` and no partial-burn tangent growth is accidentally counted.

    Parameters
    ----------
    rhs, jacobian
        Autonomous vector field and analytic Jacobian.  Passing ``None`` as
        ``jacobian`` selects central finite differences.
    x0
        Finite initial state of arbitrary positive dimension.
    t_accumulate
        Positive duration over which logarithmic QR growth is accumulated.
    t_burn
        Non-negative discarded state-only transient duration.
    qr_interval
        Positive time between QR factorizations.  A shorter final segment is
        included exactly.
    rtol, atol, max_step
        DOP853 controls used for burn-in and every augmented segment.
    jacobian_eps
        Central-difference step when no analytic Jacobian is supplied.
    div_threshold
        Optional positive norm threshold for early termination.
    q
        Must equal 1.0; fractional Caputo dynamics require a memory-aware
        variational method.
    """

    q_value = float(q)
    burn_time = float(t_burn)
    accumulation_time = float(t_accumulate)
    interval = float(qr_interval)
    relative_tolerance = float(rtol)
    absolute_tolerance = float(atol)
    maximum_step = float(max_step)
    finite_difference_step = float(jacobian_eps)
    if not np.isfinite(q_value) or abs(q_value - 1.0) > 1.0e-9:
        raise ValueError(
            "integer_dop853_variational_qr is valid only for q=1; "
            f"received q={q}."
        )
    if not np.isfinite(burn_time) or burn_time < 0.0:
        raise ValueError("t_burn must be finite and nonnegative.")
    if not np.isfinite(accumulation_time) or accumulation_time <= 0.0:
        raise ValueError("t_accumulate must be finite and positive.")
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("qr_interval must be finite and positive.")
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("rtol must be finite and positive.")
    if not np.isfinite(absolute_tolerance) or absolute_tolerance <= 0.0:
        raise ValueError("atol must be finite and positive.")
    if maximum_step <= 0.0 or np.isnan(maximum_step):
        raise ValueError("max_step must be positive.")
    if not np.isfinite(finite_difference_step) or finite_difference_step <= 0.0:
        raise ValueError("jacobian_eps must be finite and positive.")
    threshold: float | None = None
    if div_threshold is not None:
        threshold = float(div_threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("div_threshold must be finite and positive.")

    state = np.asarray(x0, dtype=float).copy()
    if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
        raise ValueError("x0 must be a non-empty finite one-dimensional state vector.")
    dimension = int(state.size)
    jacobian_source = "analytic" if jacobian is not None else "central_finite_difference"

    def checked_rhs(_time: float, current: np.ndarray) -> np.ndarray:
        try:
            derivative = np.asarray(rhs(current), dtype=float)
        except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
            raise _InvalidRhsError(f"rhs evaluation failed: {exc}") from exc
        if derivative.shape != (dimension,) or not np.all(np.isfinite(derivative)):
            raise _InvalidRhsError(
                f"rhs returned {derivative.shape} with finite={np.all(np.isfinite(derivative))}; "
                f"expected a finite ({dimension},) vector."
            )
        return derivative

    def checked_jacobian(current: np.ndarray) -> np.ndarray:
        if jacobian is None:
            matrix = finite_difference_jacobian(
                lambda point: checked_rhs(0.0, point),
                current,
                eps=finite_difference_step,
            )
        else:
            try:
                matrix = np.asarray(jacobian(current), dtype=float)
            except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
                raise _InvalidJacobianError(f"jacobian evaluation failed: {exc}") from exc
        if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
            raise _InvalidJacobianError(
                f"jacobian returned {matrix.shape} with finite={np.all(np.isfinite(matrix))}; "
                f"expected a finite ({dimension}, {dimension}) matrix."
            )
        return matrix

    def divergence_event(_time: float, current: np.ndarray) -> float:
        assert threshold is not None
        return threshold - float(np.linalg.norm(current[:dimension]))

    divergence_event.direction = -1.0  # type: ignore[attr-defined]
    divergence_event.terminal = True  # type: ignore[attr-defined]
    events = divergence_event if threshold is not None else None

    counters = {"nfev": 0, "njev": 0, "nlu": 0, "burn_nfev": 0, "qr_segments": 0}
    burn_completed = 0.0
    metadata_base: dict[str, Any] = {
        "solver": "scipy.integrate.solve_ivp",
        "solver_method": "DOP853",
        "dimension": dimension,
        "t_burn_requested": burn_time,
        "t_accumulate_requested": accumulation_time,
        "qr_interval": interval,
        "rtol": relative_tolerance,
        "atol": absolute_tolerance,
        "max_step": maximum_step,
        "jacobian_source": jacobian_source,
        "jacobian_eps": finite_difference_step if jacobian is None else None,
        "div_threshold": threshold,
    }

    times: list[float] = []
    convergence: list[np.ndarray] = []
    log_sums = np.zeros(dimension, dtype=float)
    elapsed = 0.0

    def result(status: str, error_message: str | None = None) -> AdaptiveLyapunovResult:
        estimates = log_sums / elapsed if elapsed > 0.0 else np.full(dimension, np.nan)
        metadata = dict(metadata_base)
        metadata.update(counters)
        metadata["t_burn_completed"] = burn_completed
        metadata["t_accumulate_completed"] = elapsed
        return AdaptiveLyapunovResult(
            exponents=np.asarray(estimates, dtype=float),
            times=np.asarray(times, dtype=float),
            convergence=(
                np.asarray(convergence, dtype=float)
                if convergence
                else np.empty((0, dimension), dtype=float)
            ),
            status=status,
            final_state=np.asarray(state, dtype=float).copy(),
            accumulated_time=float(elapsed),
            error_message=error_message,
            metadata=metadata,
        )

    if threshold is not None and float(np.linalg.norm(state)) >= threshold:
        return result("burn_diverged", "initial state meets or exceeds div_threshold")

    if burn_time > 0.0:
        try:
            burned = solve_ivp(
                checked_rhs,
                (0.0, burn_time),
                state,
                method="DOP853",
                events=events,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                max_step=maximum_step,
            )
        except _InvalidRhsError as exc:
            return result("invalid_rhs", str(exc))
        except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
            return result("solver_exception", str(exc))
        counters["nfev"] += int(burned.nfev)
        counters["njev"] += int(burned.njev)
        counters["nlu"] += int(burned.nlu)
        counters["burn_nfev"] = int(burned.nfev)
        burn_completed = float(burned.t[-1]) if burned.t.size else 0.0
        if threshold is not None and burned.t_events and burned.t_events[0].size:
            state = np.asarray(burned.y_events[0][0], dtype=float)
            return result("burn_diverged", "state crossed div_threshold during burn-in")
        if not burned.success:
            state = np.asarray(burned.y[:, -1], dtype=float)
            return result("burn_solver_failure", str(burned.message))
        state = np.asarray(burned.y[:, -1], dtype=float)
        if not np.all(np.isfinite(state)):
            return result("nonfinite_solution", "burn-in produced a non-finite state")
        burn_completed = burn_time

    basis = np.eye(dimension, dtype=float)

    def augmented_rhs(time_value: float, augmented: np.ndarray) -> np.ndarray:
        current = augmented[:dimension]
        tangent = augmented[dimension:].reshape(dimension, dimension)
        return np.concatenate(
            (checked_rhs(time_value, current), (checked_jacobian(current) @ tangent).ravel())
        )

    while elapsed < accumulation_time - 16.0 * np.finfo(float).eps * accumulation_time:
        duration = min(interval, accumulation_time - elapsed)
        start_time = burn_time + elapsed
        stop_time = start_time + duration
        augmented0 = np.concatenate((state, basis.ravel()))
        try:
            solved = solve_ivp(
                augmented_rhs,
                (start_time, stop_time),
                augmented0,
                method="DOP853",
                events=events,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                max_step=maximum_step,
            )
        except _InvalidRhsError as exc:
            return result("invalid_rhs", str(exc))
        except _InvalidJacobianError as exc:
            return result("invalid_jacobian", str(exc))
        except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
            return result("solver_exception", str(exc))
        counters["nfev"] += int(solved.nfev)
        counters["njev"] += int(solved.njev)
        counters["nlu"] += int(solved.nlu)
        if threshold is not None and solved.t_events and solved.t_events[0].size:
            state = np.asarray(solved.y_events[0][0][:dimension], dtype=float)
            return result("diverged", "state crossed div_threshold during accumulation")
        if not solved.success:
            state = np.asarray(solved.y[:dimension, -1], dtype=float)
            return result("solver_failure", str(solved.message))

        state = np.asarray(solved.y[:dimension, -1], dtype=float)
        tangent = np.asarray(solved.y[dimension:, -1], dtype=float).reshape(
            dimension, dimension
        )
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(tangent)):
            return result("nonfinite_solution", "augmented integration produced non-finite values")
        try:
            basis, upper = np.linalg.qr(tangent)
        except np.linalg.LinAlgError as exc:
            return result("qr_failure", str(exc))
        diagonal = np.diag(upper)
        signs = np.where(diagonal < 0.0, -1.0, 1.0)
        basis = basis * signs
        log_sums += np.log(np.maximum(np.abs(diagonal), np.finfo(float).tiny))
        elapsed += duration
        if accumulation_time - elapsed <= 16.0 * np.finfo(float).eps * accumulation_time:
            elapsed = accumulation_time
        counters["qr_segments"] += 1
        times.append(float(elapsed))
        convergence.append(log_sums / elapsed)

    return result("ok")


def integer_system_dop853_variational_qr(
    system: object,
    x0: np.ndarray,
    **kwargs: Any,
) -> AdaptiveLyapunovResult:
    """System-object wrapper for :func:`integer_dop853_variational_qr`.

    The object must expose ``evaluate(state)``.  An analytic
    ``jacobian_matrix(state)`` is used when the declaration indicates that it
    exists; otherwise the core routine uses finite differences.  Discrete maps
    and systems explicitly declaring ``q != 1`` are rejected.
    """

    if getattr(system, "kind", "flow") != "flow":
        raise ValueError("integer_system_dop853_variational_qr requires a continuous-time flow.")
    system_order = _infer_system_order(system)
    if system_order is not None and abs(system_order - 1.0) > 1.0e-9:
        raise ValueError(
            "integer_system_dop853_variational_qr is valid only for q=1; "
            f"the supplied system declares q={system_order}."
        )
    evaluate = getattr(system, "evaluate", None)
    if not callable(evaluate):
        raise ValueError("system must expose a callable evaluate(state) method.")
    jacobian_matrix = getattr(system, "jacobian_matrix", None)
    jacobian_declaration = getattr(system, "jacobian", "attribute_not_declared")
    use_analytic = callable(jacobian_matrix) and jacobian_declaration is not None
    rhs = lambda state: system.evaluate(state)
    jacobian = (lambda state: system.jacobian_matrix(state)) if use_analytic else None
    options = dict(kwargs)
    requested_q = float(options.pop("q", 1.0))
    if not np.isfinite(requested_q) or abs(requested_q - 1.0) > 1.0e-9:
        raise ValueError(
            "integer_system_dop853_variational_qr is valid only for q=1; "
            f"received q={requested_q}."
        )
    return integer_dop853_variational_qr(
        rhs,
        jacobian,
        np.asarray(x0, dtype=float),
        q=1.0,
        **options,
    )


__all__ = [
    "AdaptiveLyapunovResult",
    "integer_dop853_variational_qr",
    "integer_system_dop853_variational_qr",
]
