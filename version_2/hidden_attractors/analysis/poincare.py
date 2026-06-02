"""Reusable Poincare crossing diagnostics for integer and Caputo trajectories.

The Caputo mode is intentionally geometric. It records numerical crossings of
a sampled trajectory and does not claim an exact classical return map.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


ALLOWED_INTERPRETATION_LABELS = {
    "no_crossings",
    "insufficient_crossings",
    "point_like_or_fixed_return",
    "finite_set_like",
    "curve_like",
    "cloud_like",
    "dispersed_cloud_like",
    "inconclusive",
}
_VARIABLE_NAMES = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class PoincareCrossingResult:
    """Numerical section crossings and conservative metadata."""

    points: np.ndarray
    crossing_times: np.ndarray
    crossing_indices: np.ndarray
    crossing_directions: tuple[str, ...]
    crossing_count: int
    section_metadata: dict[str, Any]
    warnings: tuple[str, ...]
    status: str


def _section_index(section_variable: int | str, dimension: int) -> int:
    if isinstance(section_variable, str):
        key = section_variable.strip().lower()
        if key not in _VARIABLE_NAMES:
            raise ValueError("section_variable names must be one of x, y, or z.")
        index = _VARIABLE_NAMES[key]
    else:
        index = int(section_variable)
    if index < 0 or index >= dimension:
        raise ValueError("section_variable index is outside the trajectory dimension.")
    return index


def _call_rhs(rhs: Callable[..., Sequence[float]], time: float, state: np.ndarray) -> np.ndarray:
    try:
        value = rhs(float(time), np.asarray(state, dtype=float))
    except TypeError:
        value = rhs(np.asarray(state, dtype=float))
    return np.asarray(value, dtype=float)


def detect_poincare_crossings(
    times: Sequence[float],
    trajectory: Sequence[Sequence[float]],
    *,
    section_variable: int | str = 0,
    section_value: float = 0.0,
    direction: str = "positive",
    derivative_mode: str = "integer_rhs",
    rhs: Callable[..., Sequence[float]] | None = None,
    interpolation: str = "linear",
    min_crossing_separation: float = 0.0,
    burn_time: float | None = None,
    max_points: int | None = None,
) -> PoincareCrossingResult:
    """Detect linearly interpolated section crossings.

    ``integer_rhs`` evaluates the ODE right-hand side at the interpolated
    crossing. ``geometric_fractional`` and ``finite_difference_diagnostic``
    rely only on the sampled trajectory orientation.
    """

    t = np.asarray(times, dtype=float)
    states = np.asarray(trajectory, dtype=float)
    if t.ndim != 1 or states.ndim != 2 or states.shape[0] != t.size:
        raise ValueError("times and trajectory must have shapes (N,) and (N, d).")
    if t.size < 2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(states)):
        raise ValueError("trajectory must contain at least two finite samples.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing.")
    if interpolation != "linear":
        raise ValueError("only linear interpolation is supported.")
    if derivative_mode not in {"integer_rhs", "geometric_fractional", "finite_difference_diagnostic"}:
        raise ValueError("unsupported derivative_mode.")
    if derivative_mode == "integer_rhs" and rhs is None:
        raise ValueError("integer_rhs mode requires rhs.")
    requested_direction = direction.strip().lower()
    if requested_direction not in {"positive", "negative", "both", "positive_geometric_crossing"}:
        raise ValueError("unsupported direction.")
    normalized_direction = (
        "positive" if requested_direction == "positive_geometric_crossing" else requested_direction
    )
    separation = float(min_crossing_separation)
    if separation < 0.0:
        raise ValueError("min_crossing_separation must be nonnegative.")
    limit = None if max_points is None else int(max_points)
    if limit is not None and limit <= 0:
        raise ValueError("max_points must be positive when provided.")

    section_index = _section_index(section_variable, states.shape[1])
    level = float(section_value)
    accepted_points: list[np.ndarray] = []
    accepted_times: list[float] = []
    accepted_indices: list[int] = []
    accepted_directions: list[str] = []
    filtered_by_separation = 0

    for index in range(states.shape[0] - 1):
        raw_left = float(states[index, section_index])
        raw_right = float(states[index + 1, section_index])
        left = raw_left - level
        right = raw_right - level
        delta = raw_right - raw_left
        positive = left < 0.0 <= right and delta > 0.0
        negative = left > 0.0 >= right and delta < 0.0
        if normalized_direction == "positive" and not positive:
            continue
        if normalized_direction == "negative" and not negative:
            continue
        if normalized_direction == "both" and not (positive or negative):
            continue
        orientation = "positive" if positive else "negative"
        theta = (level - raw_left) / delta
        crossing_time = float(t[index] + theta * (t[index + 1] - t[index]))
        if burn_time is not None and crossing_time < float(burn_time):
            continue
        point = states[index] + theta * (states[index + 1] - states[index])
        point[section_index] = level
        if derivative_mode == "integer_rhs":
            derivative = _call_rhs(rhs, crossing_time, point)
            if derivative.shape != (states.shape[1],):
                raise ValueError("rhs returned an incompatible state derivative.")
            directional_component = float(derivative[section_index])
            if orientation == "positive" and directional_component <= 0.0:
                continue
            if orientation == "negative" and directional_component >= 0.0:
                continue
        if accepted_times and crossing_time - accepted_times[-1] < separation:
            filtered_by_separation += 1
            continue
        accepted_points.append(np.asarray(point, dtype=float))
        accepted_times.append(crossing_time)
        accepted_indices.append(index)
        accepted_directions.append(orientation)
        if limit is not None and len(accepted_points) >= limit:
            break

    points = (
        np.vstack(accepted_points)
        if accepted_points
        else np.empty((0, states.shape[1]), dtype=float)
    )
    caputo_geometric = derivative_mode == "geometric_fractional"
    warnings = (
        (
            "Caputo crossings are numerical geometric diagnostics, not exact "
            "classical Poincare-map returns."
        ),
    ) if caputo_geometric else ()
    metadata = {
        "section_variable": section_variable,
        "section_index": section_index,
        "section_value": level,
        "direction": direction,
        "direction_rule": (
            "rhs(section_crossing)[section_variable] has requested sign"
            if derivative_mode == "integer_rhs"
            else "sampled series changes sign with requested geometric orientation"
        ),
        "derivative_mode": derivative_mode,
        "interpolation": interpolation,
        "min_crossing_separation": separation,
        "filtered_by_min_crossing_separation": filtered_by_separation,
        "burn_time": burn_time,
        "caputo_geometric_crossing": caputo_geometric,
        "exact_poincare_map": derivative_mode == "integer_rhs",
        "uses_classical_rhs_direction": derivative_mode == "integer_rhs",
    }
    status = "no_crossings" if points.shape[0] == 0 else "crossings_detected"
    return PoincareCrossingResult(
        points=points,
        crossing_times=np.asarray(accepted_times, dtype=float),
        crossing_indices=np.asarray(accepted_indices, dtype=int),
        crossing_directions=tuple(accepted_directions),
        crossing_count=int(points.shape[0]),
        section_metadata=metadata,
        warnings=warnings,
        status=status,
    )


def _nearest_neighbor_stats(points: np.ndarray) -> dict[str, float | None]:
    if points.shape[0] < 2:
        return {"minimum": None, "median": None, "mean": None, "maximum": None}
    delta = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    return {
        "minimum": float(np.min(nearest)),
        "median": float(np.median(nearest)),
        "mean": float(np.mean(nearest)),
        "maximum": float(np.max(nearest)),
    }


def summarize_poincare_points(
    points: Sequence[Sequence[float]],
    *,
    retained_after_burn: int | None = None,
    duplicate_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Summarize numerical section geometry without certifying dynamics."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2:
        raise ValueError("points must have shape (N, d).")
    count = int(values.shape[0])
    retained = count if retained_after_burn is None else int(retained_after_burn)
    if count == 0:
        return {
            "crossing_count": 0,
            "retained_after_burn": retained,
            "bounding_box": {},
            "centroid": [],
            "covariance": [],
            "rank_estimate": 0,
            "nearest_neighbor_stats": _nearest_neighbor_stats(values),
            "recurrence_like_score": None,
            "duplicate_fraction": 0.0,
            "section_density_estimate": None,
            "interpretation_label": "no_crossings",
        }
    centroid = np.mean(values, axis=0)
    centered = values - centroid
    covariance = np.cov(values, rowvar=False) if count > 1 else np.zeros((values.shape[1], values.shape[1]))
    covariance = np.atleast_2d(covariance)
    rank = int(np.linalg.matrix_rank(centered, tol=1.0e-8))
    rounded = np.round(values / float(duplicate_tolerance)).astype(np.int64)
    unique_count = int(np.unique(rounded, axis=0).shape[0])
    duplicate_fraction = float(1.0 - unique_count / count)
    nearest = _nearest_neighbor_stats(values)
    spread = float(np.linalg.norm(np.ptp(values, axis=0)))
    if count < 3:
        label = "insufficient_crossings"
    elif spread <= duplicate_tolerance:
        label = "point_like_or_fixed_return"
    elif duplicate_fraction >= 0.5:
        label = "finite_set_like"
    elif rank <= 1:
        label = "curve_like"
    elif nearest["median"] is not None and spread > 0.0 and float(nearest["median"]) / spread > 0.25:
        label = "dispersed_cloud_like"
    else:
        label = "cloud_like"
    if label not in ALLOWED_INTERPRETATION_LABELS:
        label = "inconclusive"
    density = None if spread == 0.0 else float(count / spread)
    median_neighbor = nearest["median"]
    recurrence_score = None if median_neighbor is None else float(1.0 / (1.0 + median_neighbor))
    return {
        "crossing_count": count,
        "retained_after_burn": retained,
        "bounding_box": {
            f"coordinate_{index}": {"minimum": float(np.min(values[:, index])), "maximum": float(np.max(values[:, index]))}
            for index in range(values.shape[1])
        },
        "centroid": centroid.tolist(),
        "covariance": covariance.tolist(),
        "rank_estimate": rank,
        "nearest_neighbor_stats": nearest,
        "recurrence_like_score": recurrence_score,
        "duplicate_fraction": duplicate_fraction,
        "section_density_estimate": density,
        "interpretation_label": label,
    }


def write_poincare_outputs(
    output_dir: str | Path,
    result: PoincareCrossingResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write standardized CSV and JSON outputs for one crossing result."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    point_fields = ["crossing_index", "crossing_time", "direction"] + [
        f"coordinate_{index}" for index in range(result.points.shape[1])
    ]
    rows = []
    for row_index, (sample_index, time, direction, point) in enumerate(
        zip(
            result.crossing_indices,
            result.crossing_times,
            result.crossing_directions,
            result.points,
            strict=True,
        )
    ):
        row = {
            "crossing_index": int(sample_index),
            "crossing_time": float(time),
            "direction": direction,
        }
        row.update({f"coordinate_{index}": float(value) for index, value in enumerate(point)})
        rows.append(row)
    for filename in ("poincare_points.csv", "poincare_section.csv"):
        with (target / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=point_fields)
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize_poincare_points(result.points)
    summary.update(
        {
            "status": summary["interpretation_label"],
            "chaos_certified_by_poincare": False,
            "hiddenness_certified_by_poincare": False,
            "periodic_orbit_exact": False,
            "caputo_periodic_orbit_exact": False,
        }
    )
    (target / "poincare_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    output_metadata = {
        "section_metadata": result.section_metadata,
        "warnings": list(result.warnings),
        "crossing_count": result.crossing_count,
        "status": result.status,
        "certifications": {
            "chaos_verified": False,
            "hidden_verified": False,
            "periodic_orbit_exact": False,
            "caputo_periodic_orbit_exact": False,
        },
        **(metadata or {}),
    }
    (target / "poincare_metadata.json").write_text(
        json.dumps(output_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "points": "poincare_points.csv",
        "section": "poincare_section.csv",
        "summary": "poincare_summary.json",
        "metadata": "poincare_metadata.json",
    }


__all__ = [
    "ALLOWED_INTERPRETATION_LABELS",
    "PoincareCrossingResult",
    "detect_poincare_crossings",
    "summarize_poincare_points",
    "write_poincare_outputs",
]
