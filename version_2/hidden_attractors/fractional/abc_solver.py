r"""Conventional ABC predictor--corrector with an audited startup contract.

Stability: experimental

This module implements the conventional, full-history method in equations
(9)--(14) of Lee, Kim, and Jang (2024) for a commensurate order
``0 < alpha < 1``.  It does not implement their sum-of-exponentials fast
history algorithm.

The ABC FDE is represented by the Volterra equation

.. math::

   x(t)=x_0+c_\alpha f(t,x(t))+
   \frac{d_\alpha}{\Gamma(\alpha)}
   \int_a^t(t-s)^{\alpha-1}f(s,x(s))\,ds,

where ``c_alpha=(1-alpha)/B(alpha)`` and
``d_alpha=alpha/B(alpha)``.  Consequently, a regular classical initial value
must satisfy ``f(a,x0)=0``.  HAFO checks that compatibility instead of
silently integrating an inconsistent problem.

The published recurrence assumes a starting value with error ``O(h**2)`` but
does not prescribe how to obtain it.  HAFO computes the first value by fixed-
point solution of the product-trapezoid equation on the first interval and
reports the iteration count.  Failure to converge is explicit.

References
----------
S. Lee, H. Kim, B. Jang, "A Novel Numerical Method for Solving Nonlinear
Fractional-Order Differential Equations and Its Applications", Fractal and
Fractional 8 (2024), 65, https://doi.org/10.3390/fractalfract8010065.

A. Atangana, D. Baleanu, "New fractional derivatives with nonlocal and
non-singular kernel", Thermal Science 20 (2016), 763--769,
https://doi.org/10.2298/TSCI160111018A.

K. Diethelm et al., "Why fractional derivatives with nonsingular kernels
should not be used", FCAA 23 (2020), 610--634,
https://doi.org/10.1515/fca-2020-0032.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
from numba import njit
from numba.core.registry import CPUDispatcher

from .._rhs import bind_rhs


ABC_PREDICTOR_CORRECTOR_REFERENCES = (
    "https://doi.org/10.3390/fractalfract8010065",
    "https://doi.org/10.2298/TSCI160111018A",
    "https://doi.org/10.1515/fca-2020-0032",
)


@dataclass(frozen=True, slots=True)
class ABCPredictorCorrectorResult:
    """Finite trajectory and numerical/evidence contract for ABC PCM."""

    times: np.ndarray
    states: np.ndarray
    order: float
    lower_terminal: float
    step: float
    method: str
    backend: str
    status: str
    normalization_value: float
    normalization_description: str
    compatibility_residual: float
    compatibility_tolerance: float
    startup_iterations: int
    startup_tolerance: float
    startup_max_iterations: int
    memory_policy: str
    solver_info: Mapping[str, Any]
    references: tuple[str, ...] = ABC_PREDICTOR_CORRECTOR_REFERENCES
    scope: str = "finite_numerical_trajectory_only"
    evidence_warning: str = (
        "nonsingular-kernel compatibility remains contested; no chaos, "
        "attraction, hiddenness, or fast-SOE claim"
    )

    @property
    def trajectory(self) -> np.ndarray:
        return np.column_stack((self.times, self.states))


def _normalization_value(
    normalization: float | Callable[[float], float],
    alpha: float,
    normalization_name: str | None,
) -> tuple[float, str]:
    if callable(normalization):
        value = float(normalization(alpha))
        name = getattr(normalization, "__name__", type(normalization).__name__)
        description = normalization_name or f"callable:{name}"
    else:
        if isinstance(normalization, (bool, np.bool_)):
            raise ValueError("normalization must be a positive scalar or callable.")
        value = float(normalization)
        description = normalization_name or f"constant B(alpha)={value:.17g}"
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("B(alpha) must evaluate to a finite positive number.")
    return value, description


def _linear_product_weights(
    order: float,
    step: float,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two exact interval weights indexed by one-based lag."""

    theta0 = np.zeros(count + 1, dtype=np.float64)
    theta1 = np.zeros(count + 1, dtype=np.float64)
    q = np.longdouble(order)
    hq = np.longdouble(step) ** q
    for lag in range(1, count + 1):
        m = np.longdouble(lag)
        previous = m - np.longdouble(1.0)
        if lag == 1:
            order_difference = np.longdouble(1.0)
            next_difference = np.longdouble(1.0)
        else:
            order_difference = previous**q * np.expm1(
                q * np.log1p(np.longdouble(1.0) / previous)
            )
            next_order = q + np.longdouble(1.0)
            next_difference = previous**next_order * np.expm1(
                next_order * np.log1p(np.longdouble(1.0) / previous)
            )
        left = hq * (
            next_difference / (q + 1.0)
            - previous * order_difference / q
        )
        right = hq * (
            m * order_difference / q
            - next_difference / (q + 1.0)
        )
        theta0[lag] = float(left)
        theta1[lag] = float(right)
        expected_sum = float(hq * order_difference / q)
        actual_sum = theta0[lag] + theta1[lag]
        tolerance = 4096.0 * np.finfo(np.float64).eps * max(
            1.0, abs(expected_sum)
        )
        if (
            theta0[lag] < -tolerance
            or theta1[lag] < -tolerance
            or abs(actual_sum - expected_sum) > tolerance
        ):
            raise ArithmeticError(
                f"ABC product-integration weights lost consistency at lag {lag}."
            )
        theta0[lag] = max(0.0, theta0[lag])
        theta1[lag] = max(0.0, theta1[lag])
    return theta0, theta1


@njit(cache=True, nogil=True)
def _abc_pcm_numba_core(
    rhs,
    initial_state,
    parameters,
    order,
    lower_terminal,
    step,
    n_steps,
    local_scale,
    integral_scale,
    local_predictor_scale,
    theta0,
    theta1,
    startup_tolerance,
    startup_max_iterations,
    divergence_limit,
):
    dimension = initial_state.size
    times = np.empty(n_steps + 1, dtype=np.float64)
    states = np.empty((n_steps + 1, dimension), dtype=np.float64)
    values = np.empty((n_steps + 1, dimension), dtype=np.float64)
    times[0] = lower_terminal
    states[0] = initial_state
    values[0] = rhs(lower_terminal, initial_state, parameters)
    completed = 0
    status_code = 0
    startup_iterations = 0

    t1 = lower_terminal + step
    startup_state = initial_state.copy()
    startup_coefficient = local_scale + local_predictor_scale
    for iteration in range(startup_max_iterations):
        startup_value = rhs(t1, startup_state, parameters)
        next_state = (
            initial_state
            + local_predictor_scale * order * values[0]
            + startup_coefficient * startup_value
        )
        difference_squared = 0.0
        scale_squared = 0.0
        finite_state = True
        for component in range(dimension):
            difference = next_state[component] - startup_state[component]
            difference_squared += difference * difference
            scale_squared += next_state[component] * next_state[component]
            if not np.isfinite(next_state[component]):
                finite_state = False
        startup_state = next_state
        startup_iterations = iteration + 1
        if not finite_state:
            status_code = 2
            break
        if np.sqrt(difference_squared) <= startup_tolerance * (
            1.0 + np.sqrt(scale_squared)
        ):
            break
    else:
        status_code = 3

    if status_code != 0:
        return (
            times[:1],
            states[:1],
            status_code,
            startup_iterations,
        )

    times[1] = t1
    states[1] = startup_state
    values[1] = rhs(t1, startup_state, parameters)
    completed = 1
    norm_squared = 0.0
    finite_value = True
    for component in range(dimension):
        norm_squared += startup_state[component] * startup_state[component]
        if not np.isfinite(values[1, component]):
            finite_value = False
    if not finite_value:
        return times[:2], states[:2], 2, startup_iterations
    if np.sqrt(norm_squared) > divergence_limit:
        return times[:2], states[:2], 1, startup_iterations

    for n in range(1, n_steps):
        target_time = lower_terminal + (n + 1) * step
        memory = np.zeros(dimension, dtype=np.float64)
        for history_index in range(n):
            lag = n + 1 - history_index
            for component in range(dimension):
                memory[component] += integral_scale * (
                    theta0[lag] * values[history_index, component]
                    + theta1[lag] * values[history_index + 1, component]
                )
        predictor = (
            initial_state
            + local_scale * (-values[n - 1] + 2.0 * values[n])
            + memory
            + local_predictor_scale
            * (-values[n - 1] + (order + 2.0) * values[n])
        )
        predicted_value = rhs(target_time, predictor, parameters)
        corrected = (
            initial_state
            + local_scale * predicted_value
            + memory
            + local_predictor_scale
            * (order * values[n] + predicted_value)
        )
        times[n + 1] = target_time
        states[n + 1] = corrected
        completed = n + 1
        finite_state = True
        norm_squared = 0.0
        for component in range(dimension):
            value = corrected[component]
            if not np.isfinite(value):
                finite_state = False
            norm_squared += value * value
        if not finite_state:
            status_code = 2
            break
        if np.sqrt(norm_squared) > divergence_limit:
            status_code = 1
            break
        values[n + 1] = rhs(target_time, corrected, parameters)
        for component in range(dimension):
            if not np.isfinite(values[n + 1, component]):
                status_code = 2
                break
        if status_code != 0:
            break
    return (
        times[: completed + 1],
        states[: completed + 1],
        status_code,
        startup_iterations,
    )


def _validate_scalar(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise TypeError(f"{name} must be a real scalar.")
    normalized = float(value)
    if not np.isfinite(normalized) or (positive and normalized <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return normalized


def _validate_count(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= 1.")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= 1.") from exc
    if normalized < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return int(normalized)


def abc_linear_product_weights(
    order: float,
    step: float,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Lee--Kim--Jang linear product weights by one-based lag.

    Index zero is intentionally unused.  The returned arrays therefore have
    length ``count + 1`` and contain the exact interval weights
    :math:`\\Theta^0_m` and :math:`\\Theta^1_m` for lags ``m=1,...,count``.
    """

    alpha = _validate_scalar(order, name="order")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("ABC product weights require 0 < order < 1.")
    h = _validate_scalar(step, name="step", positive=True)
    n_weights = _validate_count(count, name="count")
    return _linear_product_weights(alpha, h, n_weights)


def integrate_abc_predictor_corrector(
    rhs: Callable,
    initial_state: Any,
    order: float,
    parameters: Any = None,
    *,
    step: float,
    n_steps: int,
    lower_terminal: float = 0.0,
    normalization: float | Callable[[float], float] = 1.0,
    normalization_name: str | None = None,
    compatibility_tolerance: float = 1.0e-12,
    startup_tolerance: float = 1.0e-12,
    startup_max_iterations: int = 100,
    use_acceleration: bool = True,
    allow_python_fallback: bool = True,
    divergence_norm: float | None = 120.0,
) -> ABCPredictorCorrectorResult:
    """Integrate the conventional full-history ABC predictor--corrector."""

    if not callable(rhs):
        raise TypeError("rhs must be callable.")
    state = np.asarray(initial_state, dtype=np.float64).reshape(-1)
    if state.size < 1 or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain at least one finite value.")
    alpha = _validate_scalar(order, name="order")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("The ABC predictor-corrector requires 0 < order < 1.")
    h = _validate_scalar(step, name="step", positive=True)
    terminal = _validate_scalar(lower_terminal, name="lower_terminal")
    n_steps = _validate_count(n_steps, name="n_steps")
    compatibility_tolerance = _validate_scalar(
        compatibility_tolerance,
        name="compatibility_tolerance",
        positive=True,
    )
    startup_tolerance = _validate_scalar(
        startup_tolerance,
        name="startup_tolerance",
        positive=True,
    )
    startup_max_iterations = _validate_count(
        startup_max_iterations,
        name="startup_max_iterations",
    )
    normalization_value, normalization_description = _normalization_value(
        normalization,
        alpha,
        normalization_name,
    )
    if divergence_norm is None:
        divergence_limit = float("inf")
        recorded_divergence = None
    else:
        divergence_limit = _validate_scalar(
            divergence_norm,
            name="divergence_norm",
            positive=True,
        )
        recorded_divergence = divergence_limit

    if isinstance(rhs, CPUDispatcher):
        if parameters is None:
            parameter_vector = np.empty(0, dtype=np.float64)
        elif isinstance(parameters, (list, tuple, np.ndarray)):
            parameter_vector = np.asarray(parameters, dtype=np.float64).reshape(-1)
        else:
            raise TypeError("The Numba ABC ABI requires a numeric parameter vector.")
        if not np.all(np.isfinite(parameter_vector)):
            raise ValueError("parameters must contain only finite values.")
        initial_rhs = np.asarray(
            rhs(terminal, state, parameter_vector),
            dtype=np.float64,
        ).reshape(-1)
    else:
        parameter_vector = np.empty(0, dtype=np.float64)
        bound_rhs = bind_rhs(rhs, parameters)
        initial_rhs = np.asarray(bound_rhs(terminal, state), dtype=np.float64).reshape(-1)
    if initial_rhs.shape != state.shape or not np.all(np.isfinite(initial_rhs)):
        raise ValueError("rhs at the lower terminal must be finite and match initial_state.")
    compatibility_residual = float(np.linalg.norm(initial_rhs))
    if compatibility_residual > compatibility_tolerance:
        raise ValueError(
            "ABC classical initial compatibility requires f(lower_terminal, x0)=0; "
            f"residual {compatibility_residual:.6g} exceeds "
            f"compatibility_tolerance={compatibility_tolerance:.6g}."
        )

    theta0, theta1 = abc_linear_product_weights(alpha, h, n_steps)
    local_scale = (1.0 - alpha) / normalization_value
    d_alpha = alpha / normalization_value
    integral_scale = d_alpha / math.gamma(alpha)
    local_predictor_scale = d_alpha * h**alpha / math.gamma(alpha + 2.0)

    accelerated = bool(use_acceleration) and isinstance(rhs, CPUDispatcher)
    if accelerated:
        times, states, status_code, startup_iterations = _abc_pcm_numba_core(
            rhs,
            np.ascontiguousarray(state),
            np.ascontiguousarray(parameter_vector),
            alpha,
            terminal,
            h,
            n_steps,
            local_scale,
            integral_scale,
            local_predictor_scale,
            np.ascontiguousarray(theta0),
            np.ascontiguousarray(theta1),
            startup_tolerance,
            startup_max_iterations,
            divergence_limit,
        )
        status = {
            0: "ok",
            1: "diverged",
            2: "nonfinite_solution",
            3: "startup_no_convergence",
        }[int(status_code)]
        backend = "numba_abc_pcm_full_history"
    else:
        if bool(use_acceleration) and not bool(allow_python_fallback):
            raise RuntimeError(
                "The requested RHS has no Numba ABC backend and "
                "allow_python_fallback=False."
            )
        bound_rhs = bind_rhs(rhs, parameters)

        def evaluate(time: float, current: np.ndarray) -> np.ndarray:
            value = np.asarray(bound_rhs(time, current), dtype=np.float64).reshape(-1)
            if value.shape != state.shape:
                raise ValueError("rhs output shape must match initial_state.")
            if not np.all(np.isfinite(value)):
                raise ValueError("rhs must return finite values.")
            return value

        times = terminal + np.arange(n_steps + 1, dtype=np.float64) * h
        states = np.empty((n_steps + 1, state.size), dtype=np.float64)
        values = np.empty_like(states)
        states[0] = state
        values[0] = initial_rhs
        startup = state.copy()
        startup_coefficient = local_scale + local_predictor_scale
        startup_iterations = 0
        converged = False
        for iteration in range(startup_max_iterations):
            next_state = (
                state
                + local_predictor_scale * alpha * values[0]
                + startup_coefficient * evaluate(times[1], startup)
            )
            startup_iterations = iteration + 1
            if float(np.linalg.norm(next_state - startup)) <= startup_tolerance * (
                1.0 + float(np.linalg.norm(next_state))
            ):
                startup = next_state
                converged = True
                break
            startup = next_state
        if not converged:
            times = times[:1]
            states = states[:1]
            status = "startup_no_convergence"
        else:
            states[1] = startup
            values[1] = evaluate(times[1], startup)
            status = "ok"
            completed = 1
            if float(np.linalg.norm(startup)) > divergence_limit:
                status = "diverged"
                times = times[:2]
                states = states[:2]
            else:
                for n in range(1, n_steps):
                    memory = np.zeros(state.size, dtype=np.float64)
                    for history_index in range(n):
                        lag = n + 1 - history_index
                        memory += integral_scale * (
                            theta0[lag] * values[history_index]
                            + theta1[lag] * values[history_index + 1]
                        )
                    predictor = (
                        state
                        + local_scale * (-values[n - 1] + 2.0 * values[n])
                        + memory
                        + local_predictor_scale
                        * (-values[n - 1] + (alpha + 2.0) * values[n])
                    )
                    predicted_value = evaluate(times[n + 1], predictor)
                    corrected = (
                        state
                        + local_scale * predicted_value
                        + memory
                        + local_predictor_scale
                        * (alpha * values[n] + predicted_value)
                    )
                    states[n + 1] = corrected
                    completed = n + 1
                    if not np.all(np.isfinite(corrected)):
                        status = "nonfinite_solution"
                        break
                    if float(np.linalg.norm(corrected)) > divergence_limit:
                        status = "diverged"
                        break
                    values[n + 1] = evaluate(times[n + 1], corrected)
                times = times[: completed + 1]
                states = states[: completed + 1]
        backend = "python_abc_pcm_full_history"

    solver_info = MappingProxyType(
        {
            "used_numba_backend": accelerated,
            "acceleration_requested": bool(use_acceleration),
            "allow_python_fallback": bool(allow_python_fallback),
            "n_steps_requested": n_steps,
            "n_steps_completed": max(0, len(times) - 1),
            "divergence_norm": recorded_divergence,
            "history_complexity": "O(N^2)",
            "fast_soe_used": False,
            "published_recurrence": "Lee-Kim-Jang-2024-equations-9-14",
            "startup_contract": "implicit_product_trapezoid_fixed_point",
            "initial_compatibility": "f(lower_terminal,x0)=0",
        }
    )
    return ABCPredictorCorrectorResult(
        times=np.asarray(times, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64),
        order=alpha,
        lower_terminal=terminal,
        step=h,
        method="abc_predictor_corrector",
        backend=backend,
        status=status,
        normalization_value=normalization_value,
        normalization_description=normalization_description,
        compatibility_residual=compatibility_residual,
        compatibility_tolerance=compatibility_tolerance,
        startup_iterations=int(startup_iterations),
        startup_tolerance=startup_tolerance,
        startup_max_iterations=startup_max_iterations,
        memory_policy="full_history",
        solver_info=solver_info,
    )


__all__ = [
    "ABC_PREDICTOR_CORRECTOR_REFERENCES",
    "ABCPredictorCorrectorResult",
    "abc_linear_product_weights",
    "integrate_abc_predictor_corrector",
]
