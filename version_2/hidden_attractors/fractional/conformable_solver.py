r"""Conformable-flow integration on the natural local clock.

Stability: experimental

For the shifted Khalil conformable derivative

.. math::

   T_{q,a}x(t)=(t-a)^{1-q}x'(t)=f(t,x(t)),

the clock ``tau=(t-a)**q/q`` gives the ordinary differential equation

.. math::

   dx/d\tau=f(a+(q\tau)^{1/q},x).

HAFO integrates that transformed ODE with classical RK4 on a uniform
``tau`` grid.  This is a local time reparametrization and carries no
hereditary memory.  It must not be interpreted as a Caputo, GL, or
Hadamard solver.

Reference
---------
R. Khalil et al., "A new definition of fractional derivative",
J. Comput. Appl. Math. 264 (2014), 65--70,
https://doi.org/10.1016/j.cam.2014.01.002.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
from numba import njit
from numba.core.registry import CPUDispatcher

from .._rhs import bind_rhs
from ._log_grid import uniform_step_grid_metrics


CONFORMABLE_SOLVER_REFERENCES = (
    "https://doi.org/10.1016/j.cam.2014.01.002",
)


@dataclass(frozen=True, slots=True)
class ConformableSimulationResult:
    """Finite trajectory on physical time and the conformable clock."""

    times: np.ndarray
    clock_times: np.ndarray
    states: np.ndarray
    order: float
    lower_terminal: float
    upper_terminal: float
    clock_step: float
    method: str
    backend: str
    status: str
    solver_info: Mapping[str, Any]
    references: tuple[str, ...] = CONFORMABLE_SOLVER_REFERENCES
    memory_policy: str = "none_local_operator"
    scope: str = "finite_numerical_trajectory_only"

    @property
    def trajectory(self) -> np.ndarray:
        """Return conventional physical-time/state columns."""

        return np.column_stack((self.times, self.states))


def conformable_clock_from_time(
    times: Any,
    order: float,
    lower_terminal: float,
) -> float | np.ndarray:
    """Map physical time ``t`` to ``(t-a)**q/q``."""

    values = np.asarray(times, dtype=np.float64)
    q = float(order)
    terminal = float(lower_terminal)
    offsets = values - terminal
    if np.any(offsets < 0.0):
        raise ValueError("physical times must not precede lower_terminal.")
    result = offsets**q / q
    if values.ndim == 0:
        return float(result)
    return np.asarray(result, dtype=np.float64)


def physical_times_from_conformable_clock(
    clock_times: Any,
    order: float,
    lower_terminal: float,
) -> float | np.ndarray:
    """Map ``tau`` to ``a+(q*tau)**(1/q)``."""

    values = np.asarray(clock_times, dtype=np.float64)
    q = float(order)
    terminal = float(lower_terminal)
    if np.any(values < 0.0):
        raise ValueError("clock_times must be non-negative.")
    result = terminal + (q * values) ** (1.0 / q)
    if values.ndim == 0:
        return float(result)
    return np.asarray(result, dtype=np.float64)


@njit(cache=True, nogil=True)
def _physical_time(clock_time: float, order: float, terminal: float) -> float:
    if clock_time == 0.0:
        return terminal
    return terminal + (order * clock_time) ** (1.0 / order)


@njit(cache=True, nogil=True)
def _conformable_rk4_numba_core(
    rhs,
    initial_state,
    parameters,
    order,
    lower_terminal,
    clock_step,
    n_steps,
    divergence_limit,
):
    dimension = initial_state.size
    clock_times = np.empty(n_steps + 1, dtype=np.float64)
    states = np.empty((n_steps + 1, dimension), dtype=np.float64)
    clock_times[0] = 0.0
    states[0] = initial_state
    completed = 0
    status_code = 0
    for index in range(n_steps):
        tau = index * clock_step
        current = states[index]
        half_tau = tau + 0.5 * clock_step
        next_tau = tau + clock_step
        k1 = rhs(_physical_time(tau, order, lower_terminal), current, parameters)
        k2 = rhs(
            _physical_time(half_tau, order, lower_terminal),
            current + 0.5 * clock_step * k1,
            parameters,
        )
        k3 = rhs(
            _physical_time(half_tau, order, lower_terminal),
            current + 0.5 * clock_step * k2,
            parameters,
        )
        k4 = rhs(
            _physical_time(next_tau, order, lower_terminal),
            current + clock_step * k3,
            parameters,
        )
        states[index + 1] = current + (clock_step / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        clock_times[index + 1] = next_tau
        completed = index + 1
        finite_state = True
        norm_squared = 0.0
        for component in range(dimension):
            value = states[index + 1, component]
            if not np.isfinite(value):
                finite_state = False
            norm_squared += value * value
        if not finite_state:
            status_code = 2
            break
        if np.sqrt(norm_squared) > divergence_limit:
            status_code = 1
            break
    return (
        clock_times[: completed + 1],
        states[: completed + 1],
        status_code,
    )


def _validated_problem_values(
    initial_state: Any,
    order: float,
    lower_terminal: float,
    upper_terminal: float,
    clock_step: float,
    divergence_norm: float | None,
) -> tuple[np.ndarray, float, float, float, float, int, float, float | None]:
    state = np.asarray(initial_state, dtype=np.float64).reshape(-1)
    if state.size < 1 or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain at least one finite value.")
    q = float(order)
    terminal = float(lower_terminal)
    upper = float(upper_terminal)
    step = float(clock_step)
    if not np.isfinite(q) or q <= 0.0 or q > 1.0:
        raise ValueError("order must be finite and lie in (0, 1].")
    if not np.isfinite(terminal) or not np.isfinite(upper) or upper <= terminal:
        raise ValueError("upper_terminal must be finite and exceed lower_terminal.")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("clock_step must be finite and positive.")
    clock_duration = float(conformable_clock_from_time(upper, q, terminal))
    n_steps, residual, tolerance = uniform_step_grid_metrics(clock_duration, step)
    if n_steps < 1 or residual > tolerance:
        raise ValueError(
            "The conformable-clock duration must contain an integer number "
            "of clock_step increments."
        )
    if divergence_norm is None:
        divergence_limit = float("inf")
        recorded_divergence = None
    else:
        divergence_limit = float(divergence_norm)
        if not np.isfinite(divergence_limit) or divergence_limit <= 0.0:
            raise ValueError("divergence_norm must be finite and positive, or None.")
        recorded_divergence = divergence_limit
    return (
        np.ascontiguousarray(state),
        q,
        terminal,
        upper,
        step,
        n_steps,
        divergence_limit,
        recorded_divergence,
    )


def integrate_conformable_rk4(
    rhs: Callable,
    initial_state: Any,
    order: float,
    parameters: Any = None,
    *,
    lower_terminal: float,
    upper_terminal: float,
    clock_step: float,
    use_acceleration: bool = True,
    allow_python_fallback: bool = True,
    divergence_norm: float | None = 120.0,
) -> ConformableSimulationResult:
    """Integrate a commensurate conformable flow with RK4 in ``tau``."""

    if not callable(rhs):
        raise TypeError("rhs must be callable.")
    (
        state,
        q,
        terminal,
        upper,
        step,
        n_steps,
        divergence_limit,
        recorded_divergence,
    ) = _validated_problem_values(
        initial_state,
        order,
        lower_terminal,
        upper_terminal,
        clock_step,
        divergence_norm,
    )

    accelerated = bool(use_acceleration) and isinstance(rhs, CPUDispatcher)
    if accelerated:
        if parameters is None:
            parameter_vector = np.empty(0, dtype=np.float64)
        elif isinstance(parameters, (list, tuple, np.ndarray)):
            parameter_vector = np.asarray(parameters, dtype=np.float64).reshape(-1)
        else:
            raise TypeError(
                "The Numba conformable ABI requires a numeric parameter vector."
            )
        if not np.all(np.isfinite(parameter_vector)):
            raise ValueError("parameters must contain only finite values.")
        clock_times, states, status_code = _conformable_rk4_numba_core(
            rhs,
            state,
            np.ascontiguousarray(parameter_vector),
            q,
            terminal,
            step,
            n_steps,
            divergence_limit,
        )
        status = {0: "ok", 1: "diverged", 2: "nonfinite_solution"}[
            int(status_code)
        ]
        backend = "numba_rk4_conformable_clock"
    else:
        if bool(use_acceleration) and not bool(allow_python_fallback):
            raise RuntimeError(
                "The requested RHS has no Numba conformable backend and "
                "allow_python_fallback=False."
            )
        bound_rhs = bind_rhs(rhs, parameters)
        clock_times = np.empty(n_steps + 1, dtype=np.float64)
        states = np.empty((n_steps + 1, state.size), dtype=np.float64)
        clock_times[0] = 0.0
        states[0] = state
        status = "ok"
        completed = 0

        def clock_rhs(tau: float, current: np.ndarray) -> np.ndarray:
            physical_time = float(
                physical_times_from_conformable_clock(tau, q, terminal)
            )
            value = np.asarray(bound_rhs(physical_time, current), dtype=np.float64)
            if value.shape != state.shape:
                raise ValueError("rhs output shape must match initial_state.")
            return value

        for index in range(n_steps):
            tau = index * step
            current = states[index]
            try:
                k1 = clock_rhs(tau, current)
                k2 = clock_rhs(tau + 0.5 * step, current + 0.5 * step * k1)
                k3 = clock_rhs(tau + 0.5 * step, current + 0.5 * step * k2)
                k4 = clock_rhs(tau + step, current + step * k3)
            except Exception as exc:
                status = f"solver_exception:{type(exc).__name__}:{exc}"
                break
            states[index + 1] = current + (step / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
            clock_times[index + 1] = tau + step
            completed = index + 1
            if not np.all(np.isfinite(states[index + 1])):
                status = "nonfinite_solution"
                break
            if float(np.linalg.norm(states[index + 1])) > divergence_limit:
                status = "diverged"
                break
        clock_times = clock_times[: completed + 1]
        states = states[: completed + 1]
        backend = "python_rk4_conformable_clock"

    times = np.asarray(
        physical_times_from_conformable_clock(clock_times, q, terminal),
        dtype=np.float64,
    )
    info = MappingProxyType(
        {
            "used_numba_backend": accelerated,
            "acceleration_requested": bool(use_acceleration),
            "allow_python_fallback": bool(allow_python_fallback),
            "n_steps_requested": n_steps,
            "n_steps_completed": max(0, states.shape[0] - 1),
            "divergence_norm": recorded_divergence,
            "time_transform": "tau=(t-lower_terminal)^q/q",
            "memory_semantics": "none_local_time_reparametrization",
        }
    )
    return ConformableSimulationResult(
        times=times,
        clock_times=np.asarray(clock_times, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64),
        order=q,
        lower_terminal=terminal,
        upper_terminal=upper,
        clock_step=step,
        method="conformable_rk4_clock",
        backend=backend,
        status=status,
        solver_info=info,
    )


__all__ = [
    "CONFORMABLE_SOLVER_REFERENCES",
    "ConformableSimulationResult",
    "conformable_clock_from_time",
    "integrate_conformable_rk4",
    "physical_times_from_conformable_clock",
]
