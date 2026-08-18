"""Conservative finite-time destination classifier for trajectory campaigns.

Stability: experimental
    The classifier separates equilibrium convergence, projected near-periodic
    motion, bounded recurrence, finite-radius escape, unsettled transients, and
    ambiguous/numerically invalid outcomes.  Every label is explicitly a
    finite-time numerical diagnosis, not a proof of an invariant set, chaos,
    attraction, or hiddenness.

For Caputo systems with a finite lower terminal, ``periodic`` means a
near-periodic sampled projection.  It must not be read as an exact nonconstant
periodic solution of the hereditary initial-value problem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..analysis.trajectory import cloud_median_distance, sample_rows


DESTINATION_CLASSIFIER_SCHEMA_VERSION = "1.1"
DESTINATION_CLASSIFIER_ID = "destination_classifier_v1_1"
DESTINATION_LABELS = (
    "equilibrium",
    "periodic",
    "recurrent",
    "escape",
    "transient",
    "ambiguous",
)

DestinationLabel = Literal[
    "equilibrium",
    "periodic",
    "recurrent",
    "escape",
    "transient",
    "ambiguous",
]


@dataclass(frozen=True, slots=True)
class DestinationClassifierContract:
    """Explicit thresholds for one destination-classification campaign.

    Distances and radii are measured after division by ``coordinate_scale``.
    By default a missing scale yields an ambiguous campaign decision.  A
    deterministic scale is inferred and reported only when
    ``require_declared_scale=False`` is chosen deliberately for a diagnostic
    run; campaign-grade runs should declare the scale in their manifest.
    """

    burn_time: float
    order_kind: Literal["integer", "caputo"] = "integer"
    coordinate_scale: tuple[float, ...] | None = None
    periodic_coordinates: Mapping[int, float] = field(default_factory=dict)
    require_declared_scale: bool = True
    min_tail_samples: int = 128
    max_cloud_points: int = 500
    divergence_radius: float = 120.0
    divergence_radius_kind: Literal["scaled", "absolute"] = "scaled"
    escape_persistence_fraction: float = 0.50
    equilibrium_tolerance: float = 1.0e-3
    equilibrium_quantile_factor: float = 5.0
    equilibrium_span_factor: float = 10.0
    equilibrium_window_fraction: float = 0.20
    minimum_nontrivial_span: float = 1.0e-2
    periodic_dominant_power_min: float = 0.45
    periodic_entropy_max: float = 0.35
    periodic_frequency_drift_max: float = 0.08
    periodic_return_error_max: float = 0.12
    periodic_windows: int = 4
    recurrence_cloud_distance_max: float = 0.18
    recurrence_centroid_drift_max: float = 0.20
    recurrence_span_drift_max: float = 0.35
    recurrence_covariance_drift_max: float = 0.50
    recurrence_radius: float = 0.08
    recurrence_rate_min: float = 1.0e-4
    recurrence_theiler_samples: int = 10
    recurrence_min_lag_time: float = 1.0
    recurrence_excursion_radius: float = 0.25
    recurrence_return_fraction_min: float = 2.0e-3
    recurrence_min_return_count: int = 3
    reference_distance_max: float = 0.18
    reference_separation_margin: float = 0.05
    minimum_confidence: float = 0.05

    def __post_init__(self) -> None:
        float_fields = (
            "burn_time",
            "divergence_radius",
            "escape_persistence_fraction",
            "equilibrium_tolerance",
            "equilibrium_quantile_factor",
            "equilibrium_span_factor",
            "equilibrium_window_fraction",
            "minimum_nontrivial_span",
            "periodic_dominant_power_min",
            "periodic_entropy_max",
            "periodic_frequency_drift_max",
            "periodic_return_error_max",
            "recurrence_cloud_distance_max",
            "recurrence_centroid_drift_max",
            "recurrence_span_drift_max",
            "recurrence_covariance_drift_max",
            "recurrence_radius",
            "recurrence_rate_min",
            "recurrence_min_lag_time",
            "recurrence_excursion_radius",
            "recurrence_return_fraction_min",
            "reference_distance_max",
            "reference_separation_margin",
            "minimum_confidence",
        )
        for name in float_fields:
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be a real numeric value.") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        for name in (
            "min_tail_samples",
            "max_cloud_points",
            "periodic_windows",
            "recurrence_theiler_samples",
            "recurrence_min_return_count",
        ):
            raw = getattr(self, name)
            if isinstance(raw, bool):
                raise TypeError(f"{name} must be an integer, not Boolean.")
            try:
                value = int(raw)
                numeric = float(raw)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be an integer.") from exc
            if not math.isfinite(numeric) or numeric != value:
                raise ValueError(f"{name} must be an exact integer.")
            object.__setattr__(self, name, value)
        if not isinstance(self.require_declared_scale, bool):
            raise TypeError("require_declared_scale must be Boolean.")
        if self.order_kind not in {"integer", "caputo"}:
            raise ValueError("order_kind must be 'integer' or 'caputo'.")
        if self.divergence_radius_kind not in {"scaled", "absolute"}:
            raise ValueError("divergence_radius_kind must be 'scaled' or 'absolute'.")
        for name in (
            "divergence_radius",
            "equilibrium_tolerance",
            "equilibrium_quantile_factor",
            "equilibrium_span_factor",
            "minimum_nontrivial_span",
            "periodic_dominant_power_min",
            "periodic_entropy_max",
            "periodic_frequency_drift_max",
            "periodic_return_error_max",
            "recurrence_cloud_distance_max",
            "recurrence_centroid_drift_max",
            "recurrence_span_drift_max",
            "recurrence_covariance_drift_max",
            "recurrence_radius",
            "recurrence_min_lag_time",
            "recurrence_excursion_radius",
            "reference_distance_max",
            "reference_separation_margin",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in (
            "escape_persistence_fraction",
            "equilibrium_window_fraction",
            "recurrence_rate_min",
            "recurrence_return_fraction_min",
            "minimum_confidence",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if not 0.0 < float(self.periodic_dominant_power_min) < 1.0:
            raise ValueError("periodic_dominant_power_min must lie in (0, 1).")
        if not 0.0 < float(self.periodic_entropy_max) <= 1.0:
            raise ValueError("periodic_entropy_max must lie in (0, 1].")
        if int(self.min_tail_samples) < 16:
            raise ValueError("min_tail_samples must be at least 16.")
        if int(self.max_cloud_points) < 20:
            raise ValueError("max_cloud_points must be at least 20.")
        if int(self.periodic_windows) < 2:
            raise ValueError("periodic_windows must be at least 2.")
        if int(self.recurrence_theiler_samples) < 0:
            raise ValueError("recurrence_theiler_samples must be non-negative.")
        if int(self.recurrence_min_return_count) < 1:
            raise ValueError("recurrence_min_return_count must be positive.")
        if self.coordinate_scale is not None:
            scale = tuple(float(value) for value in self.coordinate_scale)
            if not scale or not np.all(np.isfinite(scale)) or any(value <= 0.0 for value in scale):
                raise ValueError("coordinate_scale must be finite and strictly positive.")
            object.__setattr__(self, "coordinate_scale", scale)
        periodic: dict[int, float] = {}
        for raw_index, raw_period in dict(self.periodic_coordinates).items():
            try:
                index = int(raw_index)
                period = float(raw_period)
            except (TypeError, ValueError) as exc:
                raise TypeError("periodic_coordinates must map integer indices to periods.") from exc
            if isinstance(raw_index, bool) or float(raw_index) != index:
                raise ValueError("periodic coordinate indices must be exact integers.")
            if index < 0 or not math.isfinite(period) or period <= 0.0:
                raise ValueError("periodic coordinates require non-negative indices and positive periods.")
            periodic[index] = period
        object.__setattr__(self, "periodic_coordinates", periodic)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.coordinate_scale is not None:
            payload["coordinate_scale"] = list(self.coordinate_scale)
        return payload


@dataclass(frozen=True, slots=True)
class DestinationClassification:
    """Machine-readable finite-time destination decision."""

    label: DestinationLabel
    destination_id: str
    subtype: str
    confidence: float
    is_ambiguous: bool
    reasons: tuple[str, ...]
    metrics: Mapping[str, Any]
    evidence_status: str = "finite_time_destination_diagnostic"
    scientific_warnings: tuple[str, ...] = (
        "This classification does not prove attraction, chaos, or hiddenness.",
    )
    schema_version: str = DESTINATION_CLASSIFIER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.label not in DESTINATION_LABELS:
            raise ValueError(f"label must be one of: {', '.join(DESTINATION_LABELS)}.")
        if not str(self.destination_id).strip():
            raise ValueError("destination_id must be non-empty.")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1].")
        object.__setattr__(self, "confidence", confidence)
        if bool(self.is_ambiguous) != (self.label == "ambiguous"):
            raise ValueError("is_ambiguous must agree with label='ambiguous'.")

    @property
    def edge_label(self) -> str:
        """Specific label intended for edge-tracking callbacks."""

        return self.destination_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "destination_id": self.destination_id,
            "subtype": self.subtype,
            "confidence": self.confidence,
            "is_ambiguous": self.is_ambiguous,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "evidence_status": self.evidence_status,
            "scientific_warnings": list(self.scientific_warnings),
        }


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _strict_jsonable(value: Any) -> Any:
    """Convert diagnostics to strict JSON values, using null for unavailable metrics."""

    if isinstance(value, Mapping):
        return {str(key): _strict_jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_strict_jsonable(item) for item in value.tolist()]
    if isinstance(value, (tuple, list)):
        return [_strict_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _strict_jsonable(value.item())
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value


def _result(
    label: DestinationLabel,
    destination_id: str,
    subtype: str,
    confidence: float,
    reasons: Sequence[str],
    metrics: Mapping[str, Any],
    *,
    warnings: Sequence[str] = (),
) -> DestinationClassification:
    base_warning = "This classification does not prove attraction, chaos, or hiddenness."
    return DestinationClassification(
        label=label,
        destination_id=destination_id,
        subtype=subtype,
        confidence=_clip01(confidence),
        is_ambiguous=label == "ambiguous",
        reasons=tuple(str(reason) for reason in reasons),
        metrics=_strict_jsonable(dict(metrics)),
        scientific_warnings=(base_warning, *(str(item) for item in warnings)),
    )


def _coordinate_scale(states: np.ndarray, contract: DestinationClassifierContract) -> tuple[np.ndarray, str]:
    if contract.coordinate_scale is not None:
        scale = np.asarray(contract.coordinate_scale, dtype=float)
        if scale.shape != (states.shape[1],):
            raise ValueError("coordinate_scale dimension must match trajectory states.")
        return scale, "declared"
    spread = np.ptp(states, axis=0)
    magnitude = np.quantile(np.abs(states), 0.90, axis=0)
    scale = np.maximum(np.maximum(spread, magnitude), 1.0)
    return np.asarray(scale, dtype=float), "inferred_diagnostic_only"


def _principal_states(
    states: np.ndarray,
    periodic_coordinates: Mapping[int, float],
) -> np.ndarray:
    values = np.asarray(states, dtype=float).copy()
    for index, period in periodic_coordinates.items():
        values[:, int(index)] = (
            values[:, int(index)] + 0.5 * float(period)
        ) % float(period) - 0.5 * float(period)
    return values


def _topology_embedding(
    states: np.ndarray,
    scale: np.ndarray,
    periodic_coordinates: Mapping[int, float],
) -> np.ndarray:
    """Embed periodic coordinates as scaled sine/cosine pairs."""

    columns: list[np.ndarray] = []
    for index in range(states.shape[1]):
        if index in periodic_coordinates:
            period = float(periodic_coordinates[index])
            angle = 2.0 * np.pi * states[:, index] / period
            radius = period / (2.0 * np.pi * float(scale[index]))
            columns.extend((radius * np.cos(angle), radius * np.sin(angle)))
        else:
            columns.append(states[:, index] / float(scale[index]))
    return np.column_stack(columns)


def _wrapped_scaled_distances(
    states: np.ndarray,
    point: np.ndarray,
    scale: np.ndarray,
    periodic_coordinates: Mapping[int, float],
) -> np.ndarray:
    differences = np.asarray(states, dtype=float) - np.asarray(point, dtype=float)
    for index, period in periodic_coordinates.items():
        differences[:, int(index)] = (
            differences[:, int(index)] + 0.5 * float(period)
        ) % float(period) - 0.5 * float(period)
    return np.linalg.norm(differences / scale, axis=1)


def _spectral_periodicity(
    times: np.ndarray,
    states_scaled: np.ndarray,
    contract: DestinationClassifierContract,
) -> dict[str, Any]:
    dt_values = np.diff(times)
    dt = float(np.median(dt_values))
    relative_jitter = float(np.max(np.abs(dt_values - dt)) / max(abs(dt), np.finfo(float).eps))
    result: dict[str, Any] = {
        "sampling_interval": dt,
        "sampling_relative_jitter": relative_jitter,
        "periodicity_available": bool(relative_jitter <= 1.0e-3),
    }
    if relative_jitter > 1.0e-3:
        result.update(
            {
                "dominant_component": -1,
                "dominant_frequency_hz": float("nan"),
                "dominant_power_ratio": float("nan"),
                "spectral_entropy": float("nan"),
                "relative_frequency_drift": float("nan"),
                "period_return_error": float("nan"),
                "periodic_gate_passed": False,
            }
        )
        return result
    variances = np.var(states_scaled, axis=0)
    component = int(np.argmax(variances))
    values = states_scaled[:, component] - float(np.mean(states_scaled[:, component]))

    def spectrum_summary(signal: np.ndarray) -> tuple[float, float, float]:
        if signal.size < 16 or not np.any(np.abs(signal) > np.finfo(float).eps):
            return 0.0, 0.0, 0.0
        power = np.abs(np.fft.rfft(signal * np.hanning(signal.size))) ** 2
        frequency = np.fft.rfftfreq(signal.size, d=dt)
        if power.size <= 1 or not np.any(power[1:] > 0.0):
            return 0.0, 0.0, 0.0
        positive = power[1:]
        probability = positive / max(float(np.sum(positive)), np.finfo(float).tiny)
        index = int(np.argmax(positive))
        ratio = float(probability[index])
        entropy = -float(np.sum(probability * np.log(probability + np.finfo(float).tiny)))
        entropy /= max(math.log(probability.size), 1.0)
        return float(frequency[index + 1]), ratio, entropy

    frequency, ratio, entropy = spectrum_summary(values)
    chunks = [
        chunk
        for chunk in np.array_split(values, int(contract.periodic_windows))
        if chunk.size >= 16
    ]
    window_peaks = [spectrum_summary(chunk)[0] for chunk in chunks]
    if len(window_peaks) < 2 or any(peak <= 0.0 for peak in window_peaks):
        frequency_drift = float("inf")
    else:
        frequency_drift = float(
            max(window_peaks) - min(window_peaks)
        ) / max(max(window_peaks), np.finfo(float).eps)
    if frequency <= 0.0:
        return_error = float("inf")
        lag = 0
    else:
        lag = int(round(1.0 / (frequency * dt)))
        if lag < 1 or lag >= states_scaled.shape[0] // 2:
            return_error = float("inf")
        else:
            differences = states_scaled[lag:] - states_scaled[:-lag]
            radius = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (states_scaled - np.mean(states_scaled, axis=0)) ** 2,
                            axis=1,
                        )
                    )
                )
            )
            return_error = float(
                np.sqrt(np.mean(np.sum(differences**2, axis=1)))
                / max(radius, np.finfo(float).eps)
            )
    passed = bool(
        ratio >= contract.periodic_dominant_power_min
        and entropy <= contract.periodic_entropy_max
        and frequency_drift <= contract.periodic_frequency_drift_max
        and return_error <= contract.periodic_return_error_max
    )
    result.update(
        {
            "dominant_component": component,
            "dominant_frequency_hz": frequency,
            "dominant_power_ratio": ratio,
            "spectral_entropy": entropy,
            "relative_frequency_drift": frequency_drift,
            "period_lag_samples": lag,
            "period_return_error": return_error,
            "periodic_gate_passed": passed,
        }
    )
    return result


def _recurrence_and_stationarity(
    times: np.ndarray,
    states_scaled: np.ndarray,
    contract: DestinationClassifierContract,
) -> dict[str, Any]:
    half = states_scaled.shape[0] // 2
    first = sample_rows(states_scaled[:half], contract.max_cloud_points)
    second = sample_rows(states_scaled[half:], contract.max_cloud_points)
    cloud_distance = float(cloud_median_distance(first, second))
    centroid_drift = float(np.linalg.norm(np.mean(first, axis=0) - np.mean(second, axis=0)))
    span_drift = float(np.linalg.norm(np.ptp(first, axis=0) - np.ptp(second, axis=0)))
    cov_first = np.atleast_2d(np.cov(first, rowvar=False))
    cov_second = np.atleast_2d(np.cov(second, rowvar=False))
    covariance_scale = max(
        float(np.linalg.norm(cov_first)),
        float(np.linalg.norm(cov_second)),
        np.finfo(float).eps,
    )
    covariance_drift = float(np.linalg.norm(cov_first - cov_second) / covariance_scale)
    sample_count = min(states_scaled.shape[0], contract.max_cloud_points, 400)
    sample_indices = np.linspace(0, states_scaled.shape[0] - 1, sample_count).astype(int)
    sample = states_scaled[sample_indices]
    sample_times = np.asarray(times, dtype=float)[sample_indices]
    recurrence_sample_count = int(sample.shape[0])
    if sample.shape[0] < 2:
        recurrence_rate = 0.0
        eligible_pairs = 0
        excursion_return_pair_count = 0
        excursion_return_anchor_count = 0
        excursion_return_fraction = 0.0
        excursion_return_pairs_per_sample = 0.0
        excursion_return_mean_multiplicity = 0.0
    else:
        differences = sample[:, None, :] - sample[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
        sample_positions = np.arange(sample.shape[0], dtype=int)
        temporal_separation = np.abs(
            sample_positions[:, None] - sample_positions[None, :]
        )
        eligible = temporal_separation > int(contract.recurrence_theiler_samples)
        eligible_pairs = int(np.count_nonzero(eligible))
        recurrence_rate = float(
            np.count_nonzero((distances <= contract.recurrence_radius) & eligible)
            / max(eligible_pairs, 1)
        )
        excursion_return_pair_count = 0
        returning_anchor_mask = np.zeros(recurrence_sample_count, dtype=bool)
        for left in range(sample.shape[0] - 1):
            distances_from_left = np.linalg.norm(sample[left + 1 :] - sample[left], axis=1)
            excursion_so_far = np.maximum.accumulate(distances_from_left)
            for right in range(left + 1, sample.shape[0]):
                if right - left <= contract.recurrence_theiler_samples:
                    continue
                if sample_times[right] - sample_times[left] < contract.recurrence_min_lag_time:
                    continue
                offset = right - left - 1
                if (
                    distances[left, right] <= contract.recurrence_radius
                    and excursion_so_far[offset] >= contract.recurrence_excursion_radius
                ):
                    excursion_return_pair_count += 1
                    returning_anchor_mask[left] = True
        excursion_return_anchor_count = int(np.count_nonzero(returning_anchor_mask))
        excursion_return_fraction = float(
            excursion_return_anchor_count / max(recurrence_sample_count, 1)
        )
        excursion_return_pairs_per_sample = float(
            excursion_return_pair_count / max(recurrence_sample_count, 1)
        )
        excursion_return_mean_multiplicity = float(
            excursion_return_pair_count / max(excursion_return_anchor_count, 1)
        )
    stationary = bool(
        cloud_distance <= contract.recurrence_cloud_distance_max
        and centroid_drift <= contract.recurrence_centroid_drift_max
        and span_drift <= contract.recurrence_span_drift_max
        and covariance_drift <= contract.recurrence_covariance_drift_max
    )
    return {
        "half_cloud_distance_norm": cloud_distance,
        "half_centroid_drift_norm": centroid_drift,
        "half_span_drift_norm": span_drift,
        "half_covariance_drift_norm": covariance_drift,
        "recurrence_radius_norm": contract.recurrence_radius,
        "recurrence_theiler_samples": int(contract.recurrence_theiler_samples),
        "recurrence_theiler_coordinate": "subsampled_cloud_index",
        "recurrence_eligible_pairs": eligible_pairs,
        "recurrence_rate": recurrence_rate,
        "recurrence_min_lag_time": contract.recurrence_min_lag_time,
        "recurrence_excursion_radius_norm": contract.recurrence_excursion_radius,
        "recurrence_sample_count": recurrence_sample_count,
        # Historical alias: this remains the number of qualifying return pairs.
        "excursion_return_count": excursion_return_pair_count,
        "excursion_return_pair_count": excursion_return_pair_count,
        "excursion_return_anchor_count": excursion_return_anchor_count,
        "excursion_return_fraction": excursion_return_fraction,
        "excursion_return_fraction_basis": (
            "unique_start_anchors_over_recurrence_sample_count"
        ),
        "excursion_return_pairs_per_sample": excursion_return_pairs_per_sample,
        "excursion_return_mean_multiplicity": excursion_return_mean_multiplicity,
        "stationarity_gate_passed": stationary,
    }


def _reference_match(
    states_scaled: np.ndarray,
    references: Mapping[str, Sequence[Sequence[float]] | np.ndarray] | None,
    scale: np.ndarray,
    periodic_coordinates: Mapping[int, float],
    contract: DestinationClassifierContract,
) -> dict[str, Any]:
    if not references:
        return {
            "reference_match_status": "not_requested",
            "reference_match": "",
            "reference_distance_norm": float("nan"),
            "reference_second_distance_norm": float("nan"),
            "reference_distances_norm": {},
        }
    query = sample_rows(states_scaled, contract.max_cloud_points)
    distances: dict[str, float] = {}
    for name, cloud in references.items():
        values = np.asarray(cloud, dtype=float)
        if values.ndim != 2 or values.shape[1] != scale.size or values.shape[0] == 0:
            raise ValueError(
                f"reference {name!r} must have shape (N, {scale.size})."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"reference {name!r} must be finite.")
        reference_scaled = sample_rows(
            _topology_embedding(values, scale, periodic_coordinates),
            contract.max_cloud_points,
        )
        distances[str(name)] = float(cloud_median_distance(query, reference_scaled))
    ordered = sorted(distances.items(), key=lambda item: (item[1], item[0]))
    best_name, best_distance = ordered[0]
    second_distance = ordered[1][1] if len(ordered) > 1 else float("inf")
    separated = second_distance - best_distance >= contract.reference_separation_margin
    if best_distance <= contract.reference_distance_max and separated:
        status = "matched"
        matched = best_name
    elif best_distance <= contract.reference_distance_max + contract.reference_separation_margin:
        status = "ambiguous"
        matched = ""
    else:
        status = "different_from_references"
        matched = ""
    return {
        "reference_match_status": status,
        "reference_match": matched,
        "reference_distance_norm": best_distance,
        "reference_second_distance_norm": second_distance,
        "reference_distances_norm": distances,
    }


def classify_destination(
    times: Sequence[float] | np.ndarray,
    states: Sequence[Sequence[float]] | np.ndarray,
    *,
    contract: DestinationClassifierContract,
    equilibria: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    references: Mapping[str, Sequence[Sequence[float]] | np.ndarray] | None = None,
    integration_status: str = "ok",
) -> DestinationClassification:
    """Classify one trajectory under an explicit finite-time contract.

    ``destination_id`` refines the broad ``label`` for edge tracking.  It is
    ``equilibrium:<name>`` or ``reference:<name>`` when a specific destination
    is resolved, and otherwise records an unlabeled broad destination.
    """

    t = np.asarray(times, dtype=float)
    X = np.asarray(states, dtype=float)
    if t.ndim != 1 or t.size < 2 or not np.all(np.isfinite(t)):
        raise ValueError("times must be a finite one-dimensional array with at least two samples.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing.")
    if X.ndim != 2 or X.shape[0] != t.size or X.shape[1] < 1:
        raise ValueError("states must have shape (len(times), dimension).")
    for index in contract.periodic_coordinates:
        if int(index) >= X.shape[1]:
            raise ValueError("periodic coordinate index is outside the trajectory dimension.")
    status = str(integration_status).strip().lower()
    finite_mask = np.all(np.isfinite(X), axis=1)
    accepted_statuses = {
        "ok",
        "success",
        "completed",
        "complete",
        "event_escape",
        "escape",
        "diverged",
    }
    explicit_escape_statuses = {"event_escape", "escape", "diverged"}
    base_metrics: dict[str, Any] = {
        "integration_status": status,
        "dimension": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "finite_fraction": float(np.count_nonzero(finite_mask) / X.shape[0]),
        "burn_time": float(contract.burn_time),
        "classifier_contract": contract.to_dict(),
    }
    if not np.any(finite_mask):
        return _result(
            "ambiguous",
            "ambiguous:numerical_failure",
            "nonfinite_trajectory",
            1.0,
            ("nonfinite_state_values",),
            base_metrics,
        )
    tail_mask = t >= contract.burn_time
    tail_times = t[tail_mask]
    tail = X[tail_mask]
    base_metrics["n_tail_samples"] = int(tail.shape[0])
    scale_basis = tail[np.all(np.isfinite(tail), axis=1)]
    if scale_basis.shape[0] == 0:
        scale_basis = X[finite_mask]
    scale, scale_source = _coordinate_scale(scale_basis, contract)
    principal_finite = _principal_states(X[finite_mask], contract.periodic_coordinates)
    if contract.divergence_radius_kind == "scaled":
        finite_norms = np.linalg.norm(principal_finite / scale, axis=1)
    else:
        finite_norms = np.linalg.norm(principal_finite, axis=1)
    finite_tail_mask = tail_mask & finite_mask
    principal_tail_finite = _principal_states(
        X[finite_tail_mask],
        contract.periodic_coordinates,
    )
    if principal_tail_finite.shape[0]:
        if contract.divergence_radius_kind == "scaled":
            tail_radius_norms = np.linalg.norm(principal_tail_finite / scale, axis=1)
        else:
            tail_radius_norms = np.linalg.norm(principal_tail_finite, axis=1)
        escape_fraction = float(
            np.mean(tail_radius_norms >= contract.divergence_radius)
        )
    else:
        tail_radius_norms = np.empty(0, dtype=float)
        escape_fraction = 0.0
    final_radius_norm = float(finite_norms[-1])
    maximum_radius_norm = float(np.max(finite_norms))
    radius_evidence = bool(
        maximum_radius_norm >= contract.divergence_radius
        and final_radius_norm >= contract.divergence_radius
    )
    base_metrics.update(
        {
            "coordinate_scale": [float(value) for value in scale],
            "coordinate_scale_source": scale_source,
            "periodic_coordinates": {
                str(index): period for index, period in contract.periodic_coordinates.items()
            },
            "metric_space": (
                "cylindrical_product_embedding"
                if contract.periodic_coordinates
                else "scaled_euclidean"
            ),
            "divergence_radius_kind": contract.divergence_radius_kind,
            "final_radius_norm": final_radius_norm,
            "maximum_radius_norm": maximum_radius_norm,
            "tail_escape_fraction": escape_fraction,
        }
    )
    if (
        status in explicit_escape_statuses
        and radius_evidence
        and (
            scale_source == "declared"
            or contract.divergence_radius_kind == "absolute"
        )
    ):
        margin = (
            final_radius_norm - contract.divergence_radius
        ) / contract.divergence_radius
        return _result(
            "escape",
            "escape",
            "explicit_solver_escape_with_finite_radius_evidence",
            0.5 + 0.5 * _clip01(margin),
            ("explicit_escape_status_and_finite_radius_crossing",),
            base_metrics,
        )
    if not np.all(finite_mask):
        return _result(
            "ambiguous",
            "ambiguous:numerical_failure",
            "nonfinite_trajectory_without_resolved_escape",
            1.0,
            ("nonfinite_state_values",),
            base_metrics,
        )
    if status not in accepted_statuses:
        return _result(
            "ambiguous",
            "ambiguous:numerical_status",
            "integration_status_not_accepted",
            1.0,
            (f"integration_status:{status or 'missing'}",),
            base_metrics,
        )
    if status in explicit_escape_statuses:
        return _result(
            "ambiguous",
            "ambiguous:escape_status_conflict",
            "solver_status_conflicts_with_radius_test",
            1.0,
            ("explicit_escape_status_without_finite_radius_evidence",),
            base_metrics,
        )
    if (
        radius_evidence
        and tail.shape[0] >= contract.min_tail_samples
        and escape_fraction >= contract.escape_persistence_fraction
        and (
            scale_source == "declared"
            or contract.divergence_radius_kind == "absolute"
        )
    ):
        margin = (
            final_radius_norm - contract.divergence_radius
        ) / contract.divergence_radius
        confidence = 0.5 * _clip01(margin) + 0.5 * _clip01(
            escape_fraction
            / max(contract.escape_persistence_fraction, np.finfo(float).eps)
        )
        return _result(
            "escape",
            "escape",
            "persistent_finite_radius_escape",
            confidence,
            ("divergence_radius_reached_and_persisted",),
            base_metrics,
        )
    if contract.require_declared_scale and scale_source != "declared":
        return _result(
            "ambiguous",
            "ambiguous:scale_not_declared",
            "inferred_scale_cannot_support_campaign_destination",
            1.0,
            ("declare_coordinate_scale_before_non_escape_classification",),
            base_metrics,
        )
    if tail.shape[0] < contract.min_tail_samples:
        return _result(
            "ambiguous",
            "ambiguous:insufficient_tail",
            "insufficient_post_transient_data",
            1.0,
            ("insufficient_post_transient_samples",),
            base_metrics,
        )
    tail_scaled = _topology_embedding(tail, scale, contract.periodic_coordinates)
    spans = np.ptp(tail_scaled, axis=0)
    max_span = float(np.max(spans))
    base_metrics.update(
        {
            "tail_span_scaled": [float(value) for value in spans],
            "max_tail_span_scaled": max_span,
        }
    )

    equilibrium_metrics: dict[str, Any] = {
        "closest_equilibrium": "",
        "closest_equilibrium_final_distance_norm": float("nan"),
        "closest_equilibrium_q95_distance_norm": float("nan"),
        "closest_equilibrium_window_span_norm": float("nan"),
    }
    equilibrium_hit: tuple[str, float, float, float] | None = None
    if equilibria:
        window_size = max(10, int(round(tail.shape[0] * contract.equilibrium_window_fraction)))
        late = tail[-window_size:]
        late_embedded = tail_scaled[-window_size:]
        for name, point in equilibria.items():
            equilibrium = np.asarray(point, dtype=float)
            if equilibrium.shape != (X.shape[1],) or not np.all(np.isfinite(equilibrium)):
                raise ValueError(
                    f"equilibrium {name!r} must be a finite vector of dimension {X.shape[1]}."
                )
            distances = _wrapped_scaled_distances(
                late,
                equilibrium,
                scale,
                contract.periodic_coordinates,
            )
            final_distance = float(distances[-1])
            q95_distance = float(np.quantile(distances, 0.95))
            window_span = float(np.linalg.norm(np.ptp(late_embedded, axis=0)))
            candidate = (str(name), final_distance, q95_distance, window_span)
            if equilibrium_hit is None or final_distance < equilibrium_hit[1]:
                equilibrium_hit = candidate
        assert equilibrium_hit is not None
        name, final_distance, q95_distance, window_span = equilibrium_hit
        equilibrium_metrics = {
            "closest_equilibrium": name,
            "closest_equilibrium_final_distance_norm": final_distance,
            "closest_equilibrium_q95_distance_norm": q95_distance,
            "closest_equilibrium_window_span_norm": window_span,
        }
        base_metrics.update(equilibrium_metrics)
        converged = bool(
            final_distance <= contract.equilibrium_tolerance
            and q95_distance <= contract.equilibrium_quantile_factor * contract.equilibrium_tolerance
            and window_span <= contract.equilibrium_span_factor * contract.equilibrium_tolerance
        )
        if converged:
            q95_limit = contract.equilibrium_quantile_factor * contract.equilibrium_tolerance
            span_limit = contract.equilibrium_span_factor * contract.equilibrium_tolerance
            confidence = np.mean(
                [
                    _clip01(1.0 - final_distance / contract.equilibrium_tolerance),
                    _clip01(1.0 - q95_distance / q95_limit),
                    _clip01(1.0 - window_span / span_limit),
                ]
            )
            return _result(
                "equilibrium",
                f"equilibrium:{name}",
                "tail_converged_to_known_equilibrium",
                float(confidence),
                ("final_and_tail_equilibrium_tests_passed",),
                base_metrics,
            )

    if max_span < contract.minimum_nontrivial_span:
        return _result(
            "ambiguous",
            "ambiguous:collapsed_without_equilibrium_match",
            "collapsed_tail_without_known_equilibrium",
            1.0,
            ("tail_is_nearly_constant_but_no_equilibrium_convergence_was_resolved",),
            base_metrics,
        )

    periodicity = _spectral_periodicity(tail_times, tail_scaled, contract)
    stationarity = _recurrence_and_stationarity(tail_times, tail_scaled, contract)
    reference = _reference_match(
        tail_scaled,
        references,
        scale,
        contract.periodic_coordinates,
        contract,
    )
    base_metrics.update(periodicity)
    base_metrics.update(stationarity)
    base_metrics.update(reference)
    if reference["reference_match_status"] == "ambiguous":
        return _result(
            "ambiguous",
            "ambiguous:reference_overlap",
            "reference_destination_not_separated",
            1.0,
            ("best_reference_lies_inside_the_ambiguity_band",),
            base_metrics,
        )

    specific_reference = str(reference["reference_match"])
    if bool(periodicity["periodic_gate_passed"]):
        ratio_margin = (
            float(periodicity["dominant_power_ratio"]) - contract.periodic_dominant_power_min
        ) / max(1.0 - contract.periodic_dominant_power_min, np.finfo(float).eps)
        entropy_margin = 1.0 - float(periodicity["spectral_entropy"]) / contract.periodic_entropy_max
        drift_margin = 1.0 - float(periodicity["relative_frequency_drift"]) / contract.periodic_frequency_drift_max
        return_margin = 1.0 - float(periodicity["period_return_error"]) / contract.periodic_return_error_max
        confidence = float(
            np.mean([_clip01(value) for value in (ratio_margin, entropy_margin, drift_margin, return_margin)])
        )
        if confidence < contract.minimum_confidence:
            return _result(
                "ambiguous",
                "ambiguous:periodic_threshold_margin",
                "near_periodic_threshold_boundary",
                1.0 - confidence,
                ("periodicity_tests_pass_but_with_insufficient_margin",),
                base_metrics,
            )
        subtype = (
            "projected_near_periodic_caputo"
            if contract.order_kind == "caputo"
            else "near_periodic_integer_flow"
        )
        destination_id = f"reference:{specific_reference}" if specific_reference else "periodic:unlabeled"
        warnings = (
            "Caputo periodicity is a projected finite-time label, not an exact periodic-orbit claim.",
        ) if contract.order_kind == "caputo" else ()
        return _result(
            "periodic",
            destination_id,
            subtype,
            confidence,
            ("spectral_window_and_period_return_tests_passed",),
            base_metrics,
            warnings=warnings,
        )

    recurrent = bool(
        stationarity["stationarity_gate_passed"]
        and float(stationarity["recurrence_rate"]) >= contract.recurrence_rate_min
        and int(stationarity["excursion_return_pair_count"])
        >= contract.recurrence_min_return_count
        and float(stationarity["excursion_return_fraction"])
        >= contract.recurrence_return_fraction_min
    )
    if recurrent:
        margins = (
            1.0 - float(stationarity["half_cloud_distance_norm"]) / contract.recurrence_cloud_distance_max,
            1.0 - float(stationarity["half_centroid_drift_norm"]) / contract.recurrence_centroid_drift_max,
            1.0 - float(stationarity["half_span_drift_norm"]) / contract.recurrence_span_drift_max,
            1.0 - float(stationarity["half_covariance_drift_norm"]) / contract.recurrence_covariance_drift_max,
            float(stationarity["recurrence_rate"]) / max(contract.recurrence_rate_min, np.finfo(float).eps),
            float(stationarity["excursion_return_pair_count"])
            / max(contract.recurrence_min_return_count, 1),
            float(stationarity["excursion_return_fraction"])
            / max(contract.recurrence_return_fraction_min, np.finfo(float).eps),
        )
        confidence = float(np.mean([_clip01(value) for value in margins]))
        if confidence < contract.minimum_confidence:
            return _result(
                "ambiguous",
                "ambiguous:recurrence_threshold_margin",
                "near_recurrence_threshold_boundary",
                1.0 - confidence,
                ("recurrence_tests_pass_but_with_insufficient_margin",),
                base_metrics,
            )
        destination_id = f"reference:{specific_reference}" if specific_reference else "recurrent:unlabeled"
        return _result(
            "recurrent",
            destination_id,
            "bounded_recurrent_candidate",
            confidence,
            ("bounded_noncollapsed_stationary_recurrent_tail",),
            base_metrics,
        )

    drift_ratios = (
        float(stationarity["half_cloud_distance_norm"]) / contract.recurrence_cloud_distance_max,
        float(stationarity["half_centroid_drift_norm"]) / contract.recurrence_centroid_drift_max,
        float(stationarity["half_span_drift_norm"]) / contract.recurrence_span_drift_max,
        float(stationarity["half_covariance_drift_norm"]) / contract.recurrence_covariance_drift_max,
    )
    drift_confidence = _clip01(
        (max(drift_ratios) - 1.0) / max(max(drift_ratios), 1.0)
    )
    missing_return_confidence = max(
        _clip01(
            1.0
            - float(stationarity["excursion_return_pair_count"])
            / max(contract.recurrence_min_return_count, 1)
        ),
        _clip01(
            1.0
            - float(stationarity["excursion_return_fraction"])
            / max(contract.recurrence_return_fraction_min, np.finfo(float).eps)
        ),
    )
    confidence = max(drift_confidence, missing_return_confidence)
    if confidence <= contract.minimum_confidence:
        return _result(
            "ambiguous",
            "ambiguous:transient_threshold_margin",
            "unsettled_without_sufficient_transient_margin",
            1.0 - confidence,
            ("transient_and_recurrence_thresholds_are_not_separated",),
            base_metrics,
        )
    return _result(
        "transient",
        "transient:unsettled",
        "bounded_unsettled_transient",
        confidence,
        ("tail_statistics_or_clouds_have_not_stabilized", "extend_time_horizon_before_promotion"),
        base_metrics,
    )


__all__ = [
    "DESTINATION_CLASSIFIER_ID",
    "DESTINATION_CLASSIFIER_SCHEMA_VERSION",
    "DESTINATION_LABELS",
    "DestinationClassification",
    "DestinationClassifierContract",
    "classify_destination",
]
