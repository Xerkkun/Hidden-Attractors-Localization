"""Classical 4th-order Runge-Kutta solver for integer-order ODEs (q=1).

This module provides a minimal, self-contained RK4 implementation for use
in integer-order validation and cross-comparison against scipy.solve_ivp.

Scope
-----
RK4 is valid **only for q=1** (integer-order systems).
It must NOT be used to integrate Caputo fractional systems with q < 1.

The method:
    k1 = h * f(t_n,         y_n)
    k2 = h * f(t_n + h/2,   y_n + k1/2)
    k3 = h * f(t_n + h/2,   y_n + k2/2)
    k4 = h * f(t_n + h,     y_n + k3)
    y_{n+1} = y_n + (k1 + 2*k2 + 2*k3 + k4) / 6

The right-hand side ``rhs`` must accept ``(t, y)`` with ``t`` a float and
``y`` a 1-D numpy array, and return a 1-D array of the same shape.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def rk4_step(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y: np.ndarray,
    t: float,
    h: float,
) -> np.ndarray:
    """Advance one RK4 step.

    Parameters
    ----------
    rhs : callable
        Right-hand side ``f(t, y)`` returning an array of the same shape as ``y``.
    y : np.ndarray, shape (d,)
        Current state.
    t : float
        Current time.
    h : float
        Step size (must be positive).

    Returns
    -------
    y_next : np.ndarray, shape (d,)
        State after one RK4 step.

    Raises
    ------
    ValueError
        If ``h <= 0``.
    """
    h = float(h)
    if h <= 0.0:
        raise ValueError("h must be positive.")
    y = np.asarray(y, dtype=float)
    t = float(t)
    k1 = h * np.asarray(rhs(t,           y),          dtype=float)
    k2 = h * np.asarray(rhs(t + 0.5 * h, y + 0.5 * k1), dtype=float)
    k3 = h * np.asarray(rhs(t + 0.5 * h, y + 0.5 * k2), dtype=float)
    k4 = h * np.asarray(rhs(t + h,        y + k3),        dtype=float)
    return y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def rk4_integrate(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_final: float,
    h: float,
    *,
    div_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate an integer-order ODE using RK4.

    Parameters
    ----------
    rhs : callable
        Right-hand side ``f(t, y)``.
    y0 : np.ndarray, shape (d,)
        Initial condition at ``t=0``.
    t_final : float
        Integration end time.
    h : float
        Step size.
    div_threshold : float or None
        If given, integration stops when ``||y|| >= div_threshold``.

    Returns
    -------
    times : np.ndarray, shape (N+1,)
        Time grid from 0 to ``t_final``.
    states : np.ndarray, shape (N+1, d)
        State at each time point.

    Raises
    ------
    ValueError
        If ``h <= 0`` or ``t_final < 0``.
    """
    h = float(h)
    t_final = float(t_final)
    if h <= 0.0:
        raise ValueError("h must be positive.")
    if t_final < 0.0:
        raise ValueError("t_final must be non-negative.")

    y = np.asarray(y0, dtype=float).copy()
    if y.ndim == 0:
        y = y.reshape(1)
    n_steps = int(round(t_final / h))
    # Ensure t_final grid is exact
    times = np.linspace(0.0, t_final, n_steps + 1)
    states = np.empty((n_steps + 1, y.size), dtype=float)
    states[0] = y

    for n in range(n_steps):
        if div_threshold is not None and float(np.linalg.norm(y)) >= float(div_threshold):
            # Truncate and return
            return times[: n + 1], states[: n + 1]
        y = rk4_step(rhs, y, times[n], h)
        if not np.all(np.isfinite(y)):
            return times[: n + 2], states[: n + 2]
        states[n + 1] = y

    return times, states


__all__ = ["rk4_step", "rk4_integrate"]
