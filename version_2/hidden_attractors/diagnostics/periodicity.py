"""Post-continuation periodicity gate for target-system trajectories.

Stability: experimental
    Periodicity is an exclusion diagnostic for chaotic-attractor validation.
    It is applied after the seed has been integrated in the target system;
    harmonic seed shape alone is never a rejection reason.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from ..analysis.spectral import fft_spectrum, psd_welch


COMPONENT_NAMES = ("x", "y", "z")


def _post_transient_segment(trajectory: np.ndarray, t_transient: float) -> np.ndarray:
    values = np.asarray(trajectory, dtype=float)
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError("trajectory must contain columns t,x,y,z.")
    if t_transient <= 0.0:
        return values
    return values[values[:, 0] >= float(t_transient)]


def _power_summary(values: np.ndarray, h: float) -> tuple[float, float, float]:
    spectrum = fft_spectrum(values, h)
    power = np.asarray(spectrum.values, dtype=float) ** 2
    frequencies = np.asarray(spectrum.frequency_hz, dtype=float)
    if power.size <= 1 or not np.any(power[1:] > 0.0):
        return 0.0, 0.0, 0.0
    power = power[1:]
    frequencies = frequencies[1:]
    total = float(np.sum(power))
    probability = power / max(total, 1.0e-300)
    ratio = float(np.max(probability))
    entropy = -float(np.sum(probability * np.log(probability + 1.0e-300))) / max(math.log(probability.size), 1.0)
    peak = float(frequencies[int(np.argmax(power))])
    return ratio, entropy, peak


def _window_frequency_drift(values: np.ndarray, h: float, n_windows: int) -> float:
    chunks = [chunk for chunk in np.array_split(values, max(1, int(n_windows))) if chunk.size >= 16]
    peaks = [_power_summary(chunk, h)[2] for chunk in chunks]
    if len(peaks) < 2 or any(peak <= 0.0 for peak in peaks):
        return float("inf")
    return float(max(abs(left - right) / max(left, right) for i, left in enumerate(peaks) for right in peaks[i + 1 :]))


def _component_metrics(values: np.ndarray, h: float, config: Mapping[str, Any], name: str) -> dict[str, Any]:
    ratio, entropy, peak_hz = _power_summary(values, h)
    psd = psd_welch(values, h, nperseg=int(config.get("psd_nperseg", 512)))
    psd_power = np.asarray(psd.values, dtype=float)
    psd_ratio = float(np.max(psd_power[1:]) / np.sum(psd_power[1:])) if psd_power.size > 1 and np.sum(psd_power[1:]) > 0 else 0.0
    drift = _window_frequency_drift(values, h, int(config.get("n_windows", 3)))
    component_range = float(np.ptp(values)) if values.size else 0.0
    min_range = float(config.get("min_range", 0.01))
    entropy_max = float(config.get("entropy_min", 0.25))
    ratio_min = float(config.get("dominant_ratio_max", 0.65))
    relaxed_ratio = float(config.get("relaxed_dominant_ratio", 0.45))
    drift_max = float(config.get("freq_drift_max", 0.05))
    narrowband = entropy < entropy_max and ratio > ratio_min
    stable_peak = drift < drift_max and ratio > relaxed_ratio
    periodic = component_range >= min_range and (narrowband or stable_peak)
    return {
        "component": name,
        "fft_dominant_power_ratio": ratio,
        "psd_dominant_power_ratio": psd_ratio,
        "spectral_entropy": entropy,
        "dominant_frequency_hz": peak_hz,
        "relative_frequency_drift": drift,
        "range": component_range,
        "narrowband_periodic": bool(narrowband),
        "stable_frequency_periodic": bool(stable_peak),
        "periodic_component": bool(periodic),
    }


def classify_post_transient_periodicity(
    trajectory: np.ndarray,
    *,
    h: float,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Label target-system dynamics using FFT/PSD and multi-component agreement."""

    settings = dict(config or {})
    segment = _post_transient_segment(trajectory, float(settings.get("t_transient", 0.0)))
    states = segment[:, 1:4]
    finite = bool(states.size and np.all(np.isfinite(states)))
    divergence_norm = float(settings.get("divergence_norm", 120.0))
    diverged = bool(not finite or np.max(np.linalg.norm(states, axis=1)) > divergence_norm)
    requested = tuple(str(item) for item in settings.get("components", COMPONENT_NAMES))
    rows = []
    if not diverged:
        for name in requested:
            if name in COMPONENT_NAMES:
                rows.append(_component_metrics(states[:, COMPONENT_NAMES.index(name)], h, settings, name))
    periodic_components = [str(row["component"]) for row in rows if row["periodic_component"]]
    required = 2 if bool(settings.get("require_two_components", True)) else 1
    periodic = len(periodic_components) >= required
    maximum_range = max((float(row["range"]) for row in rows), default=0.0)
    thin_limit = float(settings.get("thin_range_max", 0.05))
    if periodic:
        candidate_label = "thin_periodic_rejected" if maximum_range <= thin_limit else "regular_periodic_rejected"
    else:
        broad_components = sum(
            float(row["spectral_entropy"]) >= float(settings.get("entropy_min", 0.25))
            and float(row["range"]) >= float(settings.get("min_range", 0.01))
            for row in rows
        )
        candidate_label = "chaotic_candidate_pending_robustness" if broad_components >= required else "nonperiodic_candidate"
    legacy_status = "periodic_post_transient" if periodic else ("divergent_post_transient" if diverged else "nonperiodic_post_transient")
    return {
        "candidate_label": candidate_label,
        "periodicity_status": legacy_status,
        "early_periodicity_status": legacy_status,
        "periodic_post_transient": bool(periodic),
        "periodic_early": bool(periodic),
        "diverged_post_transient": diverged,
        "diverged_early": diverged,
        "analysis_rows": int(segment.shape[0]),
        "t_transient": float(settings.get("t_transient", 0.0)),
        "periodic_components": ";".join(periodic_components),
        "n_periodic_components": len(periodic_components),
        "component_metrics": rows,
        "hidden_candidate_allowed": False if periodic else None,
        "hiddenness_status": "not_tested",
    }


def promotion_label_after_hiddenness_probe(
    periodicity: Mapping[str, Any],
    *,
    target_contacts_from_equilibria: int | None,
) -> str:
    """Prevent a periodic orbit from being promoted as a hidden chaotic attractor."""

    label = str(periodicity.get("candidate_label", "nonperiodic_candidate"))
    if label == "regular_periodic_rejected" and target_contacts_from_equilibria == 0:
        return "regular_hidden_like_not_chaotic"
    return label


__all__ = [
    "classify_post_transient_periodicity",
    "promotion_label_after_hiddenness_probe",
]
