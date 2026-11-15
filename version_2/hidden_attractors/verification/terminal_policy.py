"""Shared finite-horizon policy for new neighborhood probes.

These tests describe recorded finite trajectories, never asymptotic attraction.
"""
from __future__ import annotations

import copy
import numpy as np

PROBE_POLICY = "neighborhood_terminal_v2"
OPERATIONAL_DESTINATIONS = {
    0: "equilibrium_terminal", 1: "reference_contact", 2: "other_finite",
    3: "threshold_crossed", 4: "numerical_failure", 5: "inconclusive",
}


def full_horizon_early_stop(config=None):
    """Keep explicit norm safety checking, disable heuristic early captures."""
    effective = copy.deepcopy(config or {})
    effective["enabled"] = False
    effective["equilibrium_enabled"] = False
    effective["equilibrium"] = {**effective.get("equilibrium", {}), "enabled": False}
    return effective


def completion_tail(times, states, t_final, t_burn, *, terminal_window=None,
                    min_samples=3):
    """Require the requested horizon and select the tail using actual times."""
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    if (times.ndim != 1 or states.ndim != 2 or len(times) != len(states)
            or len(times) < 2):
        raise ValueError("invalid_time_state_shape")
    if not np.isfinite(times).all() or not np.isfinite(states).all():
        raise ValueError("nonfinite_solution")
    if not np.all(np.diff(times) > 0):
        raise ValueError("nonmonotone_time")
    time_tol = max(1e-10, 16*np.finfo(float).eps*max(1., abs(t_final)))
    if abs(times[0]) > time_tol or abs(times[-1]-t_final) > time_tol:
        raise ValueError("incomplete_horizon")
    start = float(t_burn)
    if terminal_window is not None:
        if not np.isfinite(terminal_window) or terminal_window <= 0:
            raise ValueError("invalid_terminal_window")
        start = max(start, t_final-terminal_window)
    mask = times >= start-time_tol
    if np.count_nonzero(mask) < max(3, min_samples):
        raise ValueError("insufficient_terminal_samples")
    return times[mask], states[mask]


def terminal_equilibrium(tail, equilibria, tolerance, span_tolerance):
    """Final distance, 95th percentile distance, and tail span must all pass."""
    span = float(np.linalg.norm(np.ptp(tail, axis=0)))
    for index, equilibrium in enumerate(equilibria):
        distances = np.linalg.norm(tail-np.asarray(equilibrium), axis=1)
        q95 = float(np.quantile(distances, .95))
        final = float(distances[-1])
        if final <= tolerance and q95 <= tolerance and span <= span_tolerance:
            return index, {"equilibrium_final_distance": final,
                           "equilibrium_q95_distance": q95, "tail_span": span}
    return None, {"tail_span": span}
