"""Tests for F5.3 FFT/PSD supporting diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hidden_attractors.analysis.spectral import (
    ALLOWED_SPECTRAL_STATES,
    compute_fft_psd,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation" / "chaos_validation" / "dynamics_diagnostics" / "psd_fft"


def _times() -> np.ndarray:
    return np.arange(5000, dtype=float) * 0.01


def test_pure_sine_has_dominant_periodic_peak() -> None:
    times = _times()
    result = compute_fft_psd(times, np.sin(2.0 * np.pi * 2.0 * times))
    assert result["state"] == "dominant_periodic_peak"
    assert result["peak_dominance"] > 0.6
    assert np.isclose(result["peak_dominance"], result["dominant_power"] / sum(result["power"]))


def test_two_frequency_signal_is_quasiperiodic_candidate() -> None:
    times = _times()
    signal = np.sin(2.0 * np.pi * 2.0 * times) + 0.6 * np.sin(2.0 * np.pi * np.sqrt(2.0) * times)
    result = compute_fft_psd(times, signal)
    assert result["state"] == "quasiperiodic_candidate"


def test_broadband_synthetic_is_broadband_or_documented_inconclusive() -> None:
    times = _times()
    result = compute_fft_psd(times, np.random.default_rng(12345).normal(size=times.size))
    assert result["state"] in {"broadband_spectrum", "spectral_inconclusive"}
    assert result["psd_proves_chaos"] is False


def test_spectral_allowed_states_are_exact() -> None:
    assert ALLOWED_SPECTRAL_STATES == {
        "broadband_spectrum",
        "dominant_periodic_peak",
        "quasiperiodic_candidate",
        "spectral_inconclusive",
    }


def test_spectral_standard_outputs_exist() -> None:
    summary = json.loads((OUTPUT / "psd_fft_diagnostics_summary.json").read_text(encoding="utf-8"))
    assert summary["standardized_outputs"] is True
    assert summary["psd_proves_chaos"] is False
    assert summary["method_validation"]["pure_sine"] == "passed"
    assert summary["method_validation"]["two_frequency_signal"] == "passed"
