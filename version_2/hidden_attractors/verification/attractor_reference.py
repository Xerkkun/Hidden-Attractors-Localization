"""Calibrated point-cloud references for finite attractor comparisons.

The utilities in this module classify sampled post-transient clouds.  They do
not prove equality of invariant sets and do not prove hiddenness.  In
particular, the acceptance threshold is estimated from independent windows of
the candidate trajectory instead of being copied from a periodic reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from ..analysis.trajectory import cloud_median_distance, sample_rows


@dataclass(frozen=True)
class AttractorReferenceCalibration:
    """Finite point-cloud calibration for one numerical attractor reference."""

    scale: float
    within_reference_distances: tuple[float, ...]
    negative_control_distances: tuple[float, ...]
    acceptance_threshold: float
    ambiguity_margin: float
    status: str
    max_points: int


def _finite_cloud(cloud: np.ndarray, *, dimension: int | None = None) -> np.ndarray:
    values = np.asarray(cloud, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("each attractor cloud must be a non-empty two-dimensional array.")
    if dimension is not None and values.shape[1] != int(dimension):
        raise ValueError("all attractor clouds must have the same state dimension.")
    if not np.all(np.isfinite(values)):
        raise ValueError("attractor clouds must contain only finite values.")
    return values


def _normalized_distance(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    return float(cloud_median_distance(a, b) / float(scale))


def calibrate_attractor_reference(
    reference_clouds: Sequence[np.ndarray],
    *,
    negative_control_clouds: Sequence[np.ndarray] = (),
    max_points: int = 1000,
    within_quantile: float = 0.95,
    safety_factor: float = 2.0,
    ambiguity_fraction: float = 0.25,
) -> AttractorReferenceCalibration:
    """Calibrate a normalized cloud-distance threshold from independent windows.

    At least two reference clouds are required.  Negative controls are not
    required, but when supplied they must remain outside the acceptance band;
    otherwise the calibration is explicitly marked ``overlapping_controls``.
    """

    if int(max_points) < 10:
        raise ValueError("max_points must be at least 10.")
    if len(reference_clouds) < 2:
        raise ValueError("at least two independent reference clouds are required.")
    if not 0.0 < float(within_quantile) <= 1.0:
        raise ValueError("within_quantile must lie in (0, 1].")
    if not np.isfinite(safety_factor) or float(safety_factor) <= 1.0:
        raise ValueError("safety_factor must be finite and greater than one.")
    if not np.isfinite(ambiguity_fraction) or float(ambiguity_fraction) <= 0.0:
        raise ValueError("ambiguity_fraction must be finite and positive.")

    first = _finite_cloud(np.asarray(reference_clouds[0], dtype=float))
    dimension = first.shape[1]
    references = [sample_rows(first, int(max_points))]
    for cloud in reference_clouds[1:]:
        references.append(sample_rows(_finite_cloud(np.asarray(cloud), dimension=dimension), int(max_points)))
    negatives = [
        sample_rows(_finite_cloud(np.asarray(cloud), dimension=dimension), int(max_points))
        for cloud in negative_control_clouds
    ]

    pooled = np.vstack(references)
    scale = float(np.linalg.norm(np.ptp(pooled, axis=0)))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError("reference clouds have zero or non-finite attractor scale.")
    within = tuple(
        _normalized_distance(references[i], references[j], scale)
        for i, j in combinations(range(len(references)), 2)
    )
    baseline = max(float(np.quantile(np.asarray(within), float(within_quantile))), 10.0 * np.finfo(float).eps)
    threshold = float(safety_factor) * baseline
    margin = max(float(ambiguity_fraction) * threshold, baseline)
    negative = tuple(
        min(_normalized_distance(control, reference, scale) for reference in references)
        for control in negatives
    )
    status = "calibrated"
    if negative and min(negative) <= threshold + margin:
        status = "overlapping_controls"
    return AttractorReferenceCalibration(
        scale=scale,
        within_reference_distances=within,
        negative_control_distances=negative,
        acceptance_threshold=threshold,
        ambiguity_margin=margin,
        status=status,
        max_points=int(max_points),
    )


def classify_cloud_against_reference(
    cloud: np.ndarray,
    reference_clouds: Sequence[np.ndarray],
    calibration: AttractorReferenceCalibration,
) -> dict[str, Any]:
    """Classify a finite cloud as same, different, or ambiguous."""

    if calibration.status != "calibrated":
        return {
            "classification": "inconclusive",
            "distance_norm": float("nan"),
            "distances_norm": [],
            "calibration_status": calibration.status,
        }
    if not reference_clouds:
        raise ValueError("reference_clouds cannot be empty.")
    first = _finite_cloud(np.asarray(reference_clouds[0], dtype=float))
    values = sample_rows(
        _finite_cloud(np.asarray(cloud, dtype=float), dimension=first.shape[1]),
        calibration.max_points,
    )
    distances = tuple(
        _normalized_distance(
            values,
            sample_rows(_finite_cloud(np.asarray(reference), dimension=first.shape[1]), calibration.max_points),
            calibration.scale,
        )
        for reference in reference_clouds
    )
    distance = float(np.median(np.asarray(distances)))
    lower = calibration.acceptance_threshold
    upper = lower + calibration.ambiguity_margin
    if distance <= lower:
        label = "same_attractor_under_calibrated_cloud_test"
    elif distance < upper:
        label = "inconclusive"
    else:
        label = "different_from_target_under_calibrated_cloud_test"
    return {
        "classification": label,
        "distance_norm": distance,
        "distances_norm": list(distances),
        "acceptance_threshold": lower,
        "ambiguity_upper_bound": upper,
        "calibration_status": calibration.status,
        "finite_sample_only": True,
    }


__all__ = [
    "AttractorReferenceCalibration",
    "calibrate_attractor_reference",
    "classify_cloud_against_reference",
]
