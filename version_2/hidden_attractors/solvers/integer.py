"""Integer-order fixed-step solvers used by Lur'e workflows."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


EFORK_Q1_A21 = 0.5
EFORK_Q1_A31 = 0.5
EFORK_Q1_A32 = -0.25
EFORK_Q1_W1 = 2.0 / 3.0
EFORK_Q1_W2 = 5.0 / 3.0
EFORK_Q1_W3 = -4.0 / 3.0


def efork_q1_step(rhs: Callable[[np.ndarray], np.ndarray], state: np.ndarray, h: float) -> np.ndarray:
    """Advance one integer-order step with the q=1 EFORK-3 coefficients."""

    x = np.asarray(state, dtype=float)
    h_value = float(h)
    if h_value <= 0.0:
        raise ValueError("h must be positive.")
    k1 = h_value * np.asarray(rhs(x), dtype=float)
    k2 = h_value * np.asarray(rhs(x + EFORK_Q1_A21 * k1), dtype=float)
    k3 = h_value * np.asarray(rhs(x + EFORK_Q1_A31 * k1 + EFORK_Q1_A32 * k2), dtype=float)
    out = x + EFORK_Q1_W1 * k1 + EFORK_Q1_W2 * k2 + EFORK_Q1_W3 * k3
    if out.shape != x.shape:
        raise ValueError(f"rhs returned incompatible state shape {out.shape}; expected {x.shape}.")
    return out


def efork_q1_integrate(
    rhs: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    t_final: float,
    h: float,
    div_threshold: float | None = None,
) -> tuple[np.ndarray, str]:
    """Integrate an integer-order trajectory with columns ``t,state...``."""

    h_value = float(h)
    final_time = float(t_final)
    if h_value <= 0.0:
        raise ValueError("h must be positive.")
    if final_time < 0.0:
        raise ValueError("t_final must be nonnegative.")
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1 or x.size < 1 or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a finite one-dimensional state vector.")
    n_steps = int(math.ceil(final_time / h_value))
    times = np.empty(n_steps + 1, dtype=float)
    states = np.empty((n_steps + 1, x.size), dtype=float)
    times[0] = 0.0
    states[0] = x
    status = "ok"
    last_index = 0
    for n in range(n_steps):
        if div_threshold is not None and float(np.linalg.norm(x)) >= float(div_threshold):
            status = "diverged"
            break
        try:
            x_next = efork_q1_step(rhs, x, h_value)
        except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
            status = f"solver_exception:{exc}"
            break
        if not np.all(np.isfinite(x_next)):
            status = "nonfinite_solution"
            break
        x = np.asarray(x_next, dtype=float)
        last_index = n + 1
        times[last_index] = last_index * h_value
        states[last_index] = x
        if div_threshold is not None and float(np.linalg.norm(x)) >= float(div_threshold):
            status = "diverged"
            break
    return np.column_stack((times[: last_index + 1], states[: last_index + 1])), status


def _uniform_output_times(t_final: float, h: float) -> np.ndarray:
    """Return a uniform output grid that includes ``t_final`` exactly."""

    n_full = int(math.floor(t_final / h))
    times = h * np.arange(n_full + 1, dtype=float)
    if times[-1] < t_final - 16.0 * np.finfo(float).eps * max(1.0, t_final):
        times = np.append(times, t_final)
    else:
        times[-1] = t_final
    return times


def dop853_q1_integrate(
    rhs: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    t_final: float,
    h: float,
    rtol: float = 1.0e-9,
    atol: float | np.ndarray = 1.0e-12,
    max_step: float = np.inf,
    div_threshold: float | None = None,
) -> tuple[np.ndarray, str]:
    """Integrate a q=1 ODE with adaptive DOP853 and uniform output.

    The return contract matches :func:`efork_q1_integrate`: the first column
    contains time, the remaining columns contain the state, and the second
    return value is a status string.  ``h`` controls only the reported output
    grid; DOP853 chooses its internal steps adaptively.  When ``t_final`` is
    not an integer multiple of ``h``, the exact final time is appended.

    Parameters
    ----------
    rhs
        Autonomous vector field ``rhs(state) -> derivative``.
    x0
        Finite one-dimensional initial state.
    t_final
        Non-negative integration duration.
    h
        Positive output-sampling interval.
    rtol, atol, max_step
        SciPy DOP853 error tolerances and maximum internal step.
    div_threshold
        Optional positive state-norm threshold.  Crossing it terminates the
        integration and returns status ``"diverged"``.

    Notes
    -----
    This is an integer-order, memoryless solver.  It is not a Caputo solver.
    Runtime integration failures are represented in the status string so the
    function remains compatible with the fixed-step q=1 solver contract;
    invalid configuration still raises :class:`ValueError`.
    """

    final_time = float(t_final)
    output_step = float(h)
    relative_tolerance = float(rtol)
    maximum_step = float(max_step)
    if not np.isfinite(final_time) or final_time < 0.0:
        raise ValueError("t_final must be finite and nonnegative.")
    if not np.isfinite(output_step) or output_step <= 0.0:
        raise ValueError("h must be finite and positive.")
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("rtol must be finite and positive.")
    if maximum_step <= 0.0 or np.isnan(maximum_step):
        raise ValueError("max_step must be positive.")
    absolute_tolerance = np.asarray(atol, dtype=float)
    if (
        absolute_tolerance.ndim > 1
        or not np.all(np.isfinite(absolute_tolerance))
        or np.any(absolute_tolerance <= 0.0)
    ):
        raise ValueError("atol must contain only finite positive values.")
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1 or x.size < 1 or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a finite one-dimensional state vector.")
    if absolute_tolerance.ndim == 1 and absolute_tolerance.size not in {1, x.size}:
        raise ValueError("vector atol must have one entry per state component.")
    solver_atol: float | np.ndarray
    if absolute_tolerance.ndim == 0 or absolute_tolerance.size == 1:
        solver_atol = float(absolute_tolerance.ravel()[0])
    else:
        solver_atol = absolute_tolerance
    threshold: float | None = None
    if div_threshold is not None:
        threshold = float(div_threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("div_threshold must be finite and positive.")
        if float(np.linalg.norm(x)) >= threshold:
            return np.column_stack((np.array([0.0]), x[None, :])), "diverged"
    if final_time == 0.0:
        return np.column_stack((np.array([0.0]), x[None, :])), "ok"

    def autonomous_rhs(_time: float, state: np.ndarray) -> np.ndarray:
        derivative = np.asarray(rhs(state), dtype=float)
        if derivative.shape != x.shape:
            raise ValueError(
                f"rhs returned incompatible state shape {derivative.shape}; expected {x.shape}."
            )
        return derivative

    events = None
    if threshold is not None:
        def divergence_event(_time: float, state: np.ndarray) -> float:
            return threshold - float(np.linalg.norm(state))

        divergence_event.direction = -1.0  # type: ignore[attr-defined]
        divergence_event.terminal = True  # type: ignore[attr-defined]
        events = divergence_event

    try:
        solved = solve_ivp(
            autonomous_rhs,
            (0.0, final_time),
            x,
            method="DOP853",
            t_eval=_uniform_output_times(final_time, output_step),
            events=events,
            rtol=relative_tolerance,
            atol=solver_atol,
            max_step=maximum_step,
        )
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        initial = np.column_stack((np.array([0.0]), x[None, :]))
        return initial, f"solver_exception:{exc}"

    trajectory = np.column_stack((solved.t, solved.y.T))
    diverged = bool(threshold is not None and solved.t_events and solved.t_events[0].size)
    if diverged:
        event_time = float(solved.t_events[0][0])
        event_state = np.asarray(solved.y_events[0][0], dtype=float)
        if trajectory.size == 0 or abs(float(trajectory[-1, 0]) - event_time) > 1.0e-12:
            trajectory = np.vstack((trajectory, np.concatenate(([event_time], event_state))))
        status = "diverged"
    elif not solved.success:
        status = f"solver_failure:{solved.message}"
    elif not np.all(np.isfinite(trajectory)):
        status = "nonfinite_solution"
    else:
        status = "ok"
    return trajectory, status


__all__ = [
    "EFORK_Q1_A21",
    "EFORK_Q1_A31",
    "EFORK_Q1_A32",
    "EFORK_Q1_W1",
    "EFORK_Q1_W2",
    "EFORK_Q1_W3",
    "dop853_q1_integrate",
    "efork_q1_integrate",
    "efork_q1_step",
]
