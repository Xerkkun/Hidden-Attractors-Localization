"""Numerical contract for the public one-sided Welch PSD."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import welch

from hidden_attractors.analysis import psd_welch


@pytest.mark.parametrize("nperseg", [255, 256])
def test_psd_welch_matches_scipy_density_for_same_window(nperseg: int) -> None:
    h = 0.01
    overlap = 0.5
    rng = np.random.default_rng(20260729)
    signal = (
        0.7 * np.sin(2.0 * np.pi * 3.0 * np.arange(1400) * h)
        + 0.2 * rng.normal(size=1400)
    )
    step = max(1, int(round(nperseg * (1.0 - overlap))))
    noverlap = nperseg - step
    window = np.hanning(nperseg)

    result = psd_welch(
        signal,
        h=h,
        nperseg=nperseg,
        overlap=overlap,
    )
    expected_frequency, expected_density = welch(
        signal,
        fs=1.0 / h,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        average="mean",
    )

    np.testing.assert_allclose(result.frequency_hz, expected_frequency, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.values, expected_density, rtol=2.0e-13, atol=1.0e-15)


@pytest.mark.parametrize(
    ("h", "overlap"),
    [
        (0.0, 0.5),
        (-0.01, 0.5),
        (float("nan"), 0.5),
        (0.01, -0.1),
        (0.01, 1.0),
        (0.01, float("nan")),
    ],
)
def test_psd_welch_rejects_invalid_sampling_contract(h: float, overlap: float) -> None:
    with pytest.raises(ValueError):
        psd_welch(np.ones(64), h=h, nperseg=32, overlap=overlap)
