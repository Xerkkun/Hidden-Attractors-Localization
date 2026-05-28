"""Classical Runge-Kutta 4th order (RK4) integrator for integer-order systems (q=1.0).

Used when q = 1.0 or dynamics_order = 'integer' for direct attractor simulation.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def rk4_integrate(
    rhs: Callable[..., np.ndarray],
    x0: np.ndarray,
    h: float,
    N: int,
    divergence_norm: float = 120.0,
) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]:
    """Integrate a system of ODEs using the classical 4th-order Runge-Kutta method.

    Parameters
    ----------
    rhs : Callable
        The vector field of the system. Can have signature rhs(t, x) or rhs(x).
    x0 : np.ndarray
        Initial state vector.
    h : float
        Step size.
    N : int
        Number of steps.
    divergence_norm : float
        Threshold above which the state is considered to have diverged.

    Returns
    -------
    times : np.ndarray
        Time array of shape (M,).
    states : np.ndarray
        State trajectory of shape (M, dim).
    status : str
        Integration status: 'ok', 'diverged', or 'nonfinite'.
    info : dict
        Metadata including step size, number of steps, and method name.
    """
    x0_arr = np.asarray(x0, dtype=float).ravel()
    dim = x0_arr.size

    times = np.empty(N + 1, dtype=float)
    states = np.empty((N + 1, dim), dtype=float)

    times[0] = 0.0
    states[0] = x0_arr

    # Normalize rhs to rhs_t(t, x)
    def rhs_t(t: float, x: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(rhs(t, x), dtype=float)
        except (TypeError, ValueError):
            return np.asarray(rhs(x), dtype=float)

    cx = x0_arr.copy()
    status = "ok"
    last_n = 0

    for n in range(N):
        t_curr = n * h
        t_next = (n + 1) * h

        try:
            k1 = rhs_t(t_curr, cx)
            k2 = rhs_t(t_curr + 0.5 * h, cx + 0.5 * h * k1)
            k3 = rhs_t(t_curr + 0.5 * h, cx + 0.5 * h * k2)
            k4 = rhs_t(t_next, cx + h * k3)
            nx = cx + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        except Exception as exc:
            status = f"solver_exception:{exc}"
            break

        # ── Finite check ─────────────────────────────────────────────────────
        if not np.all(np.isfinite(nx)):
            status = "nonfinite"
            break

        # ── Divergence check ─────────────────────────────────────────────────
        norm = np.linalg.norm(nx)
        if norm > divergence_norm:
            times[n + 1] = t_next
            states[n + 1] = nx
            last_n = n + 1
            status = "diverged"
            break

        times[n + 1] = t_next
        states[n + 1] = nx
        last_n = n + 1
        cx = nx

    info: Dict[str, Any] = {
        "integrator": "rk4",
        "integrator_class": "integer_order_solver",
        "scientific_label": "Classical RK4 integer-order (q=1.0) solver.",
        "hidden_verified": False,
        "h": h,
        "N": N,
        "steps_completed": last_n,
        "t_final_reached": float(times[last_n]),
        "divergence_norm_threshold": divergence_norm,
        "max_norm": float(np.max(np.linalg.norm(states[:last_n + 1], axis=1))),
    }

    return times[:last_n + 1], states[:last_n + 1], status, info
