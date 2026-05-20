"""Reusable spectral diagnostics for trajectory components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SpectrumResult:
    """One component spectrum."""

    frequency_hz: np.ndarray
    frequency_rad_s: np.ndarray
    values: np.ndarray
    component: int
    method: str


def infer_step(times: np.ndarray, fallback: float | None = None) -> float:
    """Infer a positive sample step from trajectory times."""

    t = np.asarray(times, dtype=float)
    if t.size >= 2:
        diffs = np.diff(t)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size:
            return float(np.median(diffs))
    if fallback is None:
        raise ValueError("could not infer sample step and no fallback was provided.")
    return float(fallback)


def fft_spectrum(values: np.ndarray, h: float, *, window: str = "hann", component: int = 0) -> SpectrumResult:
    """Return one-sided FFT amplitude spectrum for a component."""

    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 2:
        return SpectrumResult(np.empty(0), np.empty(0), np.empty(0), int(component), "fft")
    centered = data - float(np.mean(data))
    if window == "hann":
        weights = np.hanning(centered.size)
    elif window in {"none", "boxcar"}:
        weights = np.ones(centered.size)
    else:
        raise ValueError("window must be 'hann' or 'none'.")
    spec = np.abs(np.fft.rfft(centered * weights)) / max(float(np.sum(weights)), 1.0e-300)
    freq = np.fft.rfftfreq(centered.size, d=float(h))
    return SpectrumResult(freq, 2.0 * np.pi * freq, spec, int(component), "fft")


def psd_welch(
    values: np.ndarray,
    h: float,
    *,
    nperseg: int = 512,
    overlap: float = 0.5,
    component: int = 0,
) -> SpectrumResult:
    """Return a simple NumPy Welch PSD estimate."""

    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 2:
        return SpectrumResult(np.empty(0), np.empty(0), np.empty(0), int(component), "psd_welch")
    n = min(max(16, int(nperseg)), data.size)
    step = max(1, int(round(n * (1.0 - float(overlap)))))
    window = np.hanning(n)
    scale = float(np.sum(window**2)) * float(h)
    chunks = []
    for start in range(0, data.size - n + 1, step):
        segment = data[start : start + n]
        segment = segment - float(np.mean(segment))
        chunks.append((np.abs(np.fft.rfft(segment * window)) ** 2) / max(scale, 1.0e-300))
    if not chunks:
        return fft_spectrum(data, h, component=component)
    psd = np.mean(np.vstack(chunks), axis=0)
    freq = np.fft.rfftfreq(n, d=float(h))
    return SpectrumResult(freq, 2.0 * np.pi * freq, psd, int(component), "psd_welch")


def trajectory_component_spectra(
    trajectory: np.ndarray,
    *,
    components: Sequence[int] | None = None,
    h: float | None = None,
    method: str = "fft",
) -> list[SpectrumResult]:
    """Compute FFT or PSD spectra for state components in ``t,state...`` data."""

    X = np.asarray(trajectory, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("trajectory must have columns t,state....")
    h_value = infer_step(X[:, 0], h)
    dims = list(components) if components is not None else list(range(X.shape[1] - 1))
    out: list[SpectrumResult] = []
    for component in dims:
        values = X[:, 1 + int(component)]
        if method == "fft":
            out.append(fft_spectrum(values, h_value, component=int(component)))
        elif method in {"psd", "welch", "psd_welch"}:
            out.append(psd_welch(values, h_value, component=int(component)))
        else:
            raise ValueError("method must be 'fft' or 'psd'.")
    return out


__all__ = [
    "SpectrumResult",
    "fft_spectrum",
    "infer_step",
    "psd_welch",
    "trajectory_component_spectra",
]
