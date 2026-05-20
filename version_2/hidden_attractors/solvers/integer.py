"""Integer-order fixed-step solvers used by Lur'e workflows."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


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
    k3 = h_value * np.asarray(rhs(x + EFORK_Q1_A31 * k2 + EFORK_Q1_A32 * k1), dtype=float)
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


__all__ = [
    "EFORK_Q1_A21",
    "EFORK_Q1_A31",
    "EFORK_Q1_A32",
    "EFORK_Q1_W1",
    "EFORK_Q1_W2",
    "EFORK_Q1_W3",
    "efork_q1_integrate",
    "efork_q1_step",
]
