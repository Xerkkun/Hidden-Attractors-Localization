"""Experimental Numba solver based on an explicit GL recurrence.

Stability: experimental

Two initialization contracts are supported and intentionally labelled:

``caputo_shifted``
    applies the GL recurrence to ``x-x0``;
``discrete_gl``
    applies the raw discrete GL history with ``x0`` as the first sample.

The second option is not presented as a classical Riemann-Liouville initial
value problem.  Users must choose it explicitly because its initialization
semantics differ from Caputo's classical initial values.

References
----------
I. Podlubny, *Fractional Differential Equations*, Academic Press, 1999,
ISBN 978-0-12-558840-9. The explicit recurrence is an experimental HAFO
discretization and is not attributed to a stronger convergence theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit
from numba.core.registry import CPUDispatcher

from .._rhs import bind_rhs
from .contracts import normalize_fractional_orders


@dataclass(frozen=True, slots=True)
class FractionalSimulationResult:
    """Structured result from an experimental fractional recurrence."""

    times: np.ndarray
    states: np.ndarray
    orders: np.ndarray
    derivative: str
    method: str
    initialization: str
    memory_policy: str
    history_window: int | None
    backend: str = "numba"
    status: str = "ok"
    divergence_norm: float | None = None


@njit(cache=True, nogil=True)
def _gl_solver_core(
    rhs,
    x0,
    parameters,
    orders,
    t0,
    step,
    n_steps,
    shift_initial,
    history_window,
    divergence_limit,
):
    dimension = x0.size
    times = np.empty(n_steps + 1, dtype=np.float64)
    states = np.empty((n_steps + 1, dimension), dtype=np.float64)
    weights = np.empty((dimension, n_steps + 1), dtype=np.float64)
    step_powers = np.empty(dimension, dtype=np.float64)
    for component in range(dimension):
        weights[component, 0] = 1.0
        for lag in range(1, n_steps + 1):
            weights[component, lag] = weights[component, lag - 1] * (
                1.0 - (orders[component] + 1.0) / lag
            )
        step_powers[component] = step ** orders[component]

    times[0] = t0
    states[0] = x0
    status_code = 0
    completed = 0
    for n in range(1, n_steps + 1):
        previous_time = t0 + (n - 1) * step
        forcing = rhs(previous_time, states[n - 1], parameters)
        max_lag = n
        if history_window > 0 and max_lag >= history_window:
            max_lag = history_window - 1
        for component in range(dimension):
            anchor = x0[component] if shift_initial else 0.0
            history = 0.0
            for lag in range(1, max_lag + 1):
                history += weights[component, lag] * (
                    states[n - lag, component] - anchor
                )
            base = anchor if shift_initial else 0.0
            states[n, component] = (
                base + step_powers[component] * forcing[component] - history
            )
        times[n] = t0 + n * step
        completed = n
        norm_squared = 0.0
        finite_state = True
        for component in range(dimension):
            value = states[n, component]
            if not np.isfinite(value):
                finite_state = False
            norm_squared += value * value
        if not finite_state:
            status_code = 2
            break
        if np.sqrt(norm_squared) > divergence_limit:
            status_code = 1
            break
    return times[: completed + 1], states[: completed + 1], status_code


def _validated_divergence_limit(value: float | None) -> tuple[float, float | None]:
    if value is None:
        return float("inf"), None
    threshold = float(value)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("divergence_norm must be finite and positive, or None.")
    return threshold, threshold


def integrate_gl_explicit_numba(
    rhs: Any,
    x0: np.ndarray,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    parameters: np.ndarray | tuple[float, ...] | list[float] = (),
    *,
    step: float,
    n_steps: int,
    t0: float = 0.0,
    initialization: str = "caputo_shifted",
    history_window: int | None = None,
    divergence_norm: float | None = None,
) -> FractionalSimulationResult:
    """Integrate an njit-compatible RHS with an explicit GL recurrence.

    The callable signature is ``rhs(t, state, parameter_vector)``.  This first
    solver is intended for reproducible comparison and method development; its
    explicit stability region must be checked for each model and step size.
    """

    if not isinstance(rhs, CPUDispatcher):
        raise TypeError(
            "rhs must be compiled with numba.njit and use the signature "
            "(time, state, parameter_vector)."
        )
    state = np.asarray(x0, dtype=float).reshape(-1)
    parameters_array = np.asarray(parameters, dtype=float).reshape(-1)
    if state.size < 1 or not np.all(np.isfinite(state)):
        raise ValueError("x0 must contain at least one finite state value.")
    if not np.all(np.isfinite(parameters_array)):
        raise ValueError("parameters must contain only finite values.")
    normalized_orders = normalize_fractional_orders(orders, state.size)
    step = float(step)
    n_steps = int(n_steps)
    t0 = float(t0)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be a finite positive number.")
    if n_steps < 1:
        raise ValueError("n_steps must be a positive integer.")
    if not np.isfinite(t0):
        raise ValueError("t0 must be finite.")
    divergence_limit, recorded_divergence_norm = _validated_divergence_limit(
        divergence_norm
    )
    initialization = str(initialization).strip().lower()
    if initialization not in {"caputo_shifted", "discrete_gl"}:
        raise ValueError("initialization must be 'caputo_shifted' or 'discrete_gl'.")
    if history_window is None:
        window_value = 0
        memory_policy = "full_history"
    else:
        window_value = int(history_window)
        if window_value < 2:
            raise ValueError("history_window must contain at least two samples.")
        memory_policy = "finite_window"

    times, states, status_code = _gl_solver_core(
        rhs,
        np.ascontiguousarray(state),
        np.ascontiguousarray(parameters_array),
        np.ascontiguousarray(normalized_orders),
        t0,
        step,
        n_steps,
        initialization == "caputo_shifted",
        window_value,
        divergence_limit,
    )
    status = {0: "ok", 1: "diverged", 2: "nonfinite_solution"}[int(status_code)]
    derivative = "caputo" if initialization == "caputo_shifted" else "grunwald_letnikov"
    return FractionalSimulationResult(
        times=times,
        states=states,
        orders=normalized_orders,
        derivative=derivative,
        method="gl_explicit_discrete",
        initialization=initialization,
        memory_policy=memory_policy,
        history_window=history_window,
        status=status,
        divergence_norm=recorded_divergence_norm,
    )


def _integrate_gl_explicit_python(
    rhs: Any,
    state: np.ndarray,
    normalized_orders: np.ndarray,
    parameters: Any,
    *,
    step: float,
    n_steps: int,
    t0: float,
    initialization: str,
    history_window: int | None,
    divergence_norm: float | None,
) -> FractionalSimulationResult:
    """Reference NumPy implementation used for arbitrary Python callables."""

    dimension = state.size
    times = t0 + np.arange(n_steps + 1, dtype=float) * step
    states = np.empty((n_steps + 1, dimension), dtype=float)
    states[0] = state
    weights = np.empty((dimension, n_steps + 1), dtype=float)
    weights[:, 0] = 1.0
    for lag in range(1, n_steps + 1):
        weights[:, lag] = weights[:, lag - 1] * (
            1.0 - (normalized_orders + 1.0) / lag
        )
    step_powers = step**normalized_orders
    shift_initial = initialization == "caputo_shifted"
    anchor = state if shift_initial else np.zeros_like(state)
    bound_rhs = bind_rhs(rhs, parameters)
    divergence_limit, recorded_divergence_norm = _validated_divergence_limit(
        divergence_norm
    )

    status = "ok"
    completed = 0
    for n in range(1, n_steps + 1):
        try:
            forcing = np.asarray(
                bound_rhs(times[n - 1], states[n - 1]),
                dtype=float,
            )
        except Exception as exc:
            status = f"solver_exception:{type(exc).__name__}:{exc}"
            break
        if forcing.shape != state.shape:
            raise ValueError("rhs output shape must match initial_state.")
        max_lag = n
        if history_window is not None:
            max_lag = min(max_lag, history_window - 1)
        history = np.zeros(dimension, dtype=float)
        for component in range(dimension):
            history[component] = np.dot(
                weights[component, 1 : max_lag + 1],
                states[n - np.arange(1, max_lag + 1), component] - anchor[component],
            )
        states[n] = anchor + step_powers * forcing - history
        completed = n
        if not np.all(np.isfinite(states[n])):
            status = "nonfinite_solution"
            break
        if float(np.linalg.norm(states[n])) > divergence_limit:
            status = "diverged"
            break

    derivative = "caputo" if shift_initial else "grunwald_letnikov"
    return FractionalSimulationResult(
        times=times[: completed + 1],
        states=states[: completed + 1],
        orders=normalized_orders,
        derivative=derivative,
        method="gl_explicit_discrete",
        initialization=initialization,
        memory_policy="full_history" if history_window is None else "finite_window",
        history_window=history_window,
        backend="python_numpy",
        status=status,
        divergence_norm=recorded_divergence_norm,
    )


def integrate_gl_explicit(
    rhs: Any,
    x0: np.ndarray,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    parameters: Any = None,
    *,
    step: float,
    n_steps: int,
    t0: float = 0.0,
    initialization: str = "caputo_shifted",
    history_window: int | None = None,
    use_acceleration: bool = True,
    divergence_norm: float | None = None,
) -> FractionalSimulationResult:
    """Integrate with automatic Numba acceleration or a NumPy fallback.

    The fallback makes declarative/GUI-defined vector fields executable.  Both
    paths implement the same finite recurrence and are cross-checked in tests.
    """

    state = np.asarray(x0, dtype=float).reshape(-1)
    if state.size < 1 or not np.all(np.isfinite(state)):
        raise ValueError("x0 must contain at least one finite state value.")
    normalized_orders = normalize_fractional_orders(orders, state.size)
    step = float(step)
    n_steps = int(n_steps)
    t0 = float(t0)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be a finite positive number.")
    if n_steps < 1:
        raise ValueError("n_steps must be a positive integer.")
    if not np.isfinite(t0):
        raise ValueError("t0 must be finite.")
    initialization = str(initialization).strip().lower()
    if initialization not in {"caputo_shifted", "discrete_gl"}:
        raise ValueError("initialization must be 'caputo_shifted' or 'discrete_gl'.")
    if history_window is not None and int(history_window) < 2:
        raise ValueError("history_window must contain at least two samples.")
    history_window = None if history_window is None else int(history_window)

    if use_acceleration and isinstance(rhs, CPUDispatcher):
        if parameters is None:
            parameter_vector = ()
        elif isinstance(parameters, (list, tuple, np.ndarray)):
            parameter_vector = parameters
        else:
            raise TypeError("The Numba GL ABI requires a numeric parameter vector.")
        return integrate_gl_explicit_numba(
            rhs,
            state,
            normalized_orders,
            parameter_vector,
            step=step,
            n_steps=n_steps,
            t0=t0,
            initialization=initialization,
            history_window=history_window,
            divergence_norm=divergence_norm,
        )
    return _integrate_gl_explicit_python(
        rhs,
        state,
        normalized_orders,
        parameters,
        step=step,
        n_steps=n_steps,
        t0=t0,
        initialization=initialization,
        history_window=history_window,
        divergence_norm=divergence_norm,
    )


__all__ = [
    "FractionalSimulationResult",
    "integrate_gl_explicit",
    "integrate_gl_explicit_numba",
]
