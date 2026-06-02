"""Finite-time boundedness diagnostics for sampled trajectories.

These metrics identify bounded numerical candidates. They do not prove chaos,
hiddenness, or asymptotic boundedness.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


ALLOWED_BOUNDEDNESS_STATUSES = {
    "bounded_candidate",
    "unbounded_candidate",
    "nonfinite_trajectory",
    "insufficient_post_transient_data",
    "boundedness_inconclusive",
}


def _coordinate_payload(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float)]


def compute_boundedness_metrics(
    times: Sequence[float],
    trajectory: Sequence[Sequence[float]],
    burn_time: float,
    norm: str = "euclidean",
    divergence_radius: float | None = None,
    growth_window_fraction: float = 0.2,
) -> dict[str, Any]:
    """Compute conservative post-transient boundedness metrics.

    ``bounded_candidate`` is a finite-time numerical label. It is never a
    proof that a trajectory remains bounded for all future time.
    """

    t = np.asarray(times, dtype=float)
    states = np.asarray(trajectory, dtype=float)
    if t.ndim != 1 or states.ndim != 2 or states.shape[0] != t.size:
        raise ValueError("times and trajectory must have shapes (N,) and (N, d).")
    if norm not in {"euclidean", "max"}:
        raise ValueError("norm must be 'euclidean' or 'max'.")
    fraction = float(growth_window_fraction)
    if not (0.0 < fraction <= 0.5):
        raise ValueError("growth_window_fraction must be in (0, 0.5].")
    radius = None if divergence_radius is None else float(divergence_radius)
    if radius is not None and radius <= 0.0:
        raise ValueError("divergence_radius must be positive when provided.")

    retained_mask = t >= float(burn_time)
    post_times = t[retained_mask]
    post_states = states[retained_mask]
    nonfinite_count = int(np.size(post_states) - np.count_nonzero(np.isfinite(post_states)))
    finite_fraction = (
        float(np.count_nonzero(np.isfinite(post_states)) / np.size(post_states))
        if np.size(post_states)
        else 0.0
    )
    base: dict[str, Any] = {
        "burn_time": float(burn_time),
        "norm": norm,
        "post_transient_rows": int(post_states.shape[0]),
        "finite_fraction": finite_fraction,
        "nonfinite_count": nonfinite_count,
        "divergence_radius": radius,
        "boundedness_proves_chaos": False,
        "chaos_certified_by_boundedness": False,
        "hiddenness_certified_by_boundedness": False,
    }
    if post_states.shape[0] < 10:
        return {
            **base,
            "boundedness_status": "insufficient_post_transient_data",
            "norm_timeseries": [],
        }
    if nonfinite_count:
        return {
            **base,
            "boundedness_status": "nonfinite_trajectory",
            "norm_timeseries": [],
        }

    norms = (
        np.linalg.norm(post_states, axis=1)
        if norm == "euclidean"
        else np.max(np.abs(post_states), axis=1)
    )
    window = max(2, int(round(norms.size * fraction)))
    early_mean = float(np.mean(norms[:window]))
    late_mean = float(np.mean(norms[-window:]))
    growth_ratio = late_mean / max(early_mean, np.finfo(float).eps)
    max_norm = float(np.max(norms))
    if radius is not None and max_norm > radius:
        status = "unbounded_candidate"
    elif growth_ratio >= 100.0:
        status = "unbounded_candidate"
    elif np.isfinite(max_norm) and growth_ratio < 10.0:
        status = "bounded_candidate"
    else:
        status = "boundedness_inconclusive"
    if status not in ALLOWED_BOUNDEDNESS_STATUSES:
        status = "boundedness_inconclusive"
    return {
        **base,
        "max_norm": max_norm,
        "R_observed": max_norm,
        "min_norm": float(np.min(norms)),
        "mean_norm": float(np.mean(norms)),
        "median_norm": float(np.median(norms)),
        "final_norm": float(norms[-1]),
        "norm_growth_ratio": float(growth_ratio),
        "coordinate_min": _coordinate_payload(np.min(post_states, axis=0)),
        "coordinate_max": _coordinate_payload(np.max(post_states, axis=0)),
        "coordinate_span": _coordinate_payload(np.ptp(post_states, axis=0)),
        "boundedness_status": status,
        "norm_timeseries": [
            {"time": float(time), "norm": float(value)}
            for time, value in zip(post_times, norms, strict=True)
        ],
    }


__all__ = ["ALLOWED_BOUNDEDNESS_STATUSES", "compute_boundedness_metrics"]
