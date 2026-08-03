"""Generic Numba kernels for integer-order flows and maps.

Stability: experimental

Compiled callables use the common signature ``f(t, state, parameter_vector)``.
This small numerical ABI is independent of HAFO's dictionary-based system
objects and can therefore be reused by Toolbox Chaos without copying solver
code.  Callables must already be decorated with :func:`numba.njit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit
from numba.core.registry import CPUDispatcher


@dataclass(frozen=True, slots=True)
class NumbaSimulationResult:
    """Structured output from a compiled integer flow or map."""

    times: np.ndarray
    states: np.ndarray
    backend: str
    method: str
    kind: str
    status: str = "ok"


@njit(cache=True, nogil=True)
def _rk4_core(rhs, x0, parameters, t0, step, n_steps):
    dimension = x0.size
    times = np.empty(n_steps + 1, dtype=np.float64)
    states = np.empty((n_steps + 1, dimension), dtype=np.float64)
    times[0] = t0
    states[0] = x0
    for n in range(n_steps):
        t = t0 + n * step
        state = states[n]
        k1 = rhs(t, state, parameters)
        k2 = rhs(t + 0.5 * step, state + 0.5 * step * k1, parameters)
        k3 = rhs(t + 0.5 * step, state + 0.5 * step * k2, parameters)
        k4 = rhs(t + step, state + step * k3, parameters)
        states[n + 1] = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        times[n + 1] = t + step
    return times, states


@njit(cache=True, nogil=True)
def _map_core(map_function, x0, parameters, n_steps, discard):
    dimension = x0.size
    states = np.empty((n_steps + 1, dimension), dtype=np.float64)
    times = np.empty(n_steps + 1, dtype=np.float64)
    state = x0.copy()
    for n in range(discard):
        state = map_function(n, state, parameters)
    states[0] = state
    times[0] = discard
    for n in range(n_steps):
        iteration = discard + n
        state = map_function(iteration, state, parameters)
        states[n + 1] = state
        times[n + 1] = iteration + 1
    return times, states


def _require_compiled(function: Any, label: str) -> None:
    if not isinstance(function, CPUDispatcher):
        raise TypeError(
            f"{label} must be compiled with numba.njit and use the signature "
            "(time_or_iteration, state, parameter_vector)."
        )


def integrate_rk4_numba(
    rhs: Any,
    x0: np.ndarray,
    parameters: np.ndarray | tuple[float, ...] | list[float] = (),
    *,
    step: float,
    n_steps: int,
    t0: float = 0.0,
) -> NumbaSimulationResult:
    """Integrate an njit-compiled integer-order flow with classical RK4."""

    _require_compiled(rhs, "rhs")
    state = np.asarray(x0, dtype=float).reshape(-1)
    parameter_vector = np.asarray(parameters, dtype=float).reshape(-1)
    step = float(step)
    n_steps = int(n_steps)
    t0 = float(t0)
    if state.size < 1 or not np.all(np.isfinite(state)):
        raise ValueError("x0 must contain at least one finite state value.")
    if not np.all(np.isfinite(parameter_vector)):
        raise ValueError("parameters must contain only finite values.")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be a finite positive number.")
    if n_steps < 1:
        raise ValueError("n_steps must be a positive integer.")
    if not np.isfinite(t0):
        raise ValueError("t0 must be finite.")
    times, states = _rk4_core(
        rhs,
        np.ascontiguousarray(state),
        np.ascontiguousarray(parameter_vector),
        t0,
        step,
        n_steps,
    )
    status = "ok" if np.all(np.isfinite(states)) else "nonfinite_solution"
    return NumbaSimulationResult(times, states, "numba", "rk4", "flow", status)


def iterate_map_numba(
    map_function: Any,
    x0: np.ndarray,
    parameters: np.ndarray | tuple[float, ...] | list[float] = (),
    *,
    n_steps: int,
    discard: int = 0,
) -> NumbaSimulationResult:
    """Iterate an njit-compiled discrete map after an optional transient."""

    _require_compiled(map_function, "map_function")
    state = np.asarray(x0, dtype=float).reshape(-1)
    parameter_vector = np.asarray(parameters, dtype=float).reshape(-1)
    n_steps = int(n_steps)
    discard = int(discard)
    if state.size < 1 or not np.all(np.isfinite(state)):
        raise ValueError("x0 must contain at least one finite state value.")
    if not np.all(np.isfinite(parameter_vector)):
        raise ValueError("parameters must contain only finite values.")
    if n_steps < 1:
        raise ValueError("n_steps must be a positive integer.")
    if discard < 0:
        raise ValueError("discard must be non-negative.")
    times, states = _map_core(
        map_function,
        np.ascontiguousarray(state),
        np.ascontiguousarray(parameter_vector),
        n_steps,
        discard,
    )
    status = "ok" if np.all(np.isfinite(states)) else "nonfinite_solution"
    return NumbaSimulationResult(times, states, "numba", "iteration", "map", status)


__all__ = ["NumbaSimulationResult", "integrate_rk4_numba", "iterate_map_numba"]
