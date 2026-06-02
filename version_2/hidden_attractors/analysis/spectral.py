"""Reusable spectral diagnostics for trajectory components.

FFT and PSD classifications are supporting numerical indicators. They do not
certify chaos or hiddenness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


ALLOWED_SPECTRAL_STATES = {
    "broadband_spectrum",
    "dominant_periodic_peak",
    "quasiperiodic_candidate",
    "spectral_inconclusive",
}
_COORDINATE_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class SpectrumResult:
    """One component amplitude or power spectrum.

    Attributes
    ----------
    frequency_hz : np.ndarray, shape (K,)
        Frequency axis in hertz.
    frequency_rad_s : np.ndarray, shape (K,)
        Frequency axis in rad/s (``2π × frequency_hz``).
    values : np.ndarray, shape (K,)
        Amplitude (FFT) or power spectral density (Welch) values.
    component : int
        State-component index this spectrum was computed from.
    method : str
        ``'fft'`` or ``'psd_welch'``.
    """

    frequency_hz: np.ndarray
    frequency_rad_s: np.ndarray
    values: np.ndarray
    component: int
    method: str


def infer_step(times: np.ndarray, fallback: float | None = None) -> float:
    """Infer a positive sample step from trajectory times.

    Parameters
    ----------
    times : np.ndarray
        Time column of a trajectory array.
    fallback : float or None, default None
        Value to return when the step cannot be inferred.  If ``None``
        and inference fails, a :exc:`ValueError` is raised.

    Returns
    -------
    h : float
        Median of positive finite inter-sample intervals.

    Raises
    ------
    ValueError
        If the step cannot be inferred and *fallback* is ``None``.
    """

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
    """Return the one-sided FFT amplitude spectrum for one trajectory component.

    Parameters
    ----------
    values : np.ndarray, shape (N,)
        Time series for one state component.
    h : float
        Sample step size (seconds).
    window : str, default 'hann'
        Window function: ``'hann'`` (Hanning) or ``'none'`` / ``'boxcar'``
        (rectangular).
    component : int, default 0
        State-component index stored in the result for bookkeeping.

    Returns
    -------
    spectrum : SpectrumResult
        Amplitude spectrum normalised by the window sum.
        Empty arrays are returned if the series has fewer than 2 finite values.

    Raises
    ------
    ValueError
        If *window* is not ``'hann'``, ``'none'``, or ``'boxcar'``.

    Examples
    --------
    >>> import numpy as np
    >>> from hidden_attractors.analysis.spectral import fft_spectrum
    >>> t = np.linspace(0, 10, 1000)
    >>> values = np.sin(2 * np.pi * 2.0 * t)  # 2 Hz sine
    >>> sp = fft_spectrum(values, h=t[1]-t[0])
    >>> float(sp.frequency_hz[sp.values.argmax()])  # doctest: +ELLIPSIS
    2.0...
    """

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
    """Return a simple NumPy Welch power spectral density estimate.

    Parameters
    ----------
    values : np.ndarray, shape (N,)
        Time series for one state component.
    h : float
        Sample step size (seconds).
    nperseg : int, default 512
        Segment length for Welch averaging.  Clipped to ``[16, N]``.
    overlap : float, default 0.5
        Fractional overlap between adjacent segments (0–1).
    component : int, default 0
        State-component index stored in the result for bookkeeping.

    Returns
    -------
    spectrum : SpectrumResult
        PSD estimate.  Falls back to :func:`fft_spectrum` if no full
        segment can be formed.
    """

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
    """Compute FFT or Welch PSD spectra for state components in a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray, shape (N, d+1)
        Trajectory array with time in column 0 and *d* state columns.
    components : sequence of int or None, default None
        State-component indices to analyse.  If ``None``, all components
        (columns 1 … d) are processed.
    h : float or None, default None
        Sample step.  If ``None``, inferred from the time column via
        :func:`infer_step`.
    method : str, default 'fft'
        Spectral method: ``'fft'`` or ``'psd'`` / ``'welch'`` / ``'psd_welch'``.

    Returns
    -------
    spectra : list[SpectrumResult]
        One :class:`SpectrumResult` per requested component, in order.

    Raises
    ------
    ValueError
        If *trajectory* is not 2-D with at least two columns, or if
        *method* is unrecognised.

    Examples
    --------
    >>> import numpy as np
    >>> from hidden_attractors.analysis.spectral import trajectory_component_spectra
    >>> t = np.linspace(0, 10, 500)
    >>> traj = np.column_stack([t, np.sin(2*np.pi*3*t), np.zeros_like(t)])
    >>> spectra = trajectory_component_spectra(traj)
    >>> len(spectra)  # two state components
    2
    """

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


def _window_weights(length: int, window: str) -> np.ndarray:
    if window == "hann":
        return np.hanning(length)
    if window in {"none", "boxcar"}:
        return np.ones(length)
    raise ValueError("window must be 'hann', 'none', or 'boxcar'.")


def _prominent_peak_indices(power: np.ndarray) -> np.ndarray:
    if power.size < 3 or float(np.max(power)) <= 0.0:
        return np.empty(0, dtype=int)
    local = np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])) + 1
    return local[power[local] >= 0.05 * float(np.max(power))]


def compute_fft_psd(
    times: Sequence[float],
    signal: Sequence[float],
    burn_time: float | None = None,
    detrend: bool = True,
    window: str = "hann",
    normalize_power: bool = True,
    remove_dc: bool = True,
) -> dict[str, Any]:
    """Compute one-sided FFT power metrics and a conservative spectral label."""

    t = np.asarray(times, dtype=float)
    values = np.asarray(signal, dtype=float).reshape(-1)
    if t.ndim != 1 or values.ndim != 1 or t.size != values.size:
        raise ValueError("times and signal must be one-dimensional and aligned.")
    if burn_time is not None:
        mask = t >= float(burn_time)
        t = t[mask]
        values = values[mask]
    finite = np.isfinite(t) & np.isfinite(values)
    t = t[finite]
    values = values[finite]
    if values.size < 16:
        return {
            "state": "spectral_inconclusive",
            "signal_length": int(values.size),
            "psd_proves_chaos": False,
            "chaos_certified_by_psd": False,
            "hiddenness_certified_by_psd": False,
        }
    step = infer_step(t)
    data = values.copy()
    if detrend:
        index = np.arange(data.size, dtype=float)
        slope, intercept = np.polyfit(index, data, 1)
        data = data - (slope * index + intercept)
    elif remove_dc:
        data = data - float(np.mean(data))
    weights = _window_weights(data.size, window)
    power = np.abs(np.fft.rfft(data * weights)) ** 2
    frequencies = np.fft.rfftfreq(data.size, d=step)
    if remove_dc and power.size:
        power[0] = 0.0
    total_power_raw = float(np.sum(power))
    if normalize_power and total_power_raw > 0.0:
        power = power / total_power_raw
    total_power = float(np.sum(power))
    dominant_index = int(np.argmax(power)) if power.size else 0
    dominant_power = float(power[dominant_index]) if power.size else 0.0
    peak_dominance = dominant_power / max(total_power, np.finfo(float).eps)
    positive_power = power[power > 0.0]
    entropy = (
        float(-np.sum(positive_power * np.log(positive_power)) / np.log(power.size))
        if positive_power.size and power.size > 1 and normalize_power
        else 0.0
    )
    sorted_power = np.sort(power)[::-1]
    cumulative = np.cumsum(sorted_power)
    bins_for_90 = int(np.searchsorted(cumulative, 0.9 * max(total_power, 0.0)) + 1)
    bandwidth_fraction = float(bins_for_90 / max(power.size, 1))
    prominent = _prominent_peak_indices(power)
    peak_count = int(prominent.size)
    if peak_dominance > 0.6:
        state = "dominant_periodic_peak"
    elif 2 <= peak_count <= 5 and peak_dominance >= 0.1:
        state = "quasiperiodic_candidate"
    elif entropy >= 0.65 and peak_dominance < 0.2:
        state = "broadband_spectrum"
    else:
        state = "spectral_inconclusive"
    return {
        "state": state,
        "peak_dominance": float(peak_dominance),
        "spectral_entropy": entropy,
        "bandwidth_fraction": bandwidth_fraction,
        "number_of_prominent_peaks": peak_count,
        "dominant_frequency": float(frequencies[dominant_index]) if frequencies.size else None,
        "dominant_power": dominant_power,
        "total_power": total_power,
        "total_power_before_normalization": total_power_raw,
        "dc_removed": bool(remove_dc),
        "detrend": bool(detrend),
        "window": window,
        "normalize_power": bool(normalize_power),
        "sampling_interval": step,
        "sampling_rate": 1.0 / step,
        "signal_length": int(values.size),
        "frequencies": frequencies.tolist(),
        "power": power.tolist(),
        "psd_proves_chaos": False,
        "chaos_certified_by_psd": False,
        "hiddenness_certified_by_psd": False,
    }


def spectral_diagnostics_multicoordinate(
    times: Sequence[float],
    trajectory: Sequence[Sequence[float]],
    burn_time: float,
    coordinates: Sequence[str] = _COORDINATE_NAMES,
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply :func:`compute_fft_psd` to selected trajectory coordinates."""

    t = np.asarray(times, dtype=float)
    states = np.asarray(trajectory, dtype=float)
    if t.ndim != 1 or states.ndim != 2 or states.shape[0] != t.size:
        raise ValueError("times and trajectory must have shapes (N,) and (N, d).")
    results: dict[str, dict[str, Any]] = {}
    for name in coordinates:
        if name not in _COORDINATE_NAMES:
            raise ValueError("coordinates must be selected from x, y, and z.")
        index = _COORDINATE_NAMES.index(name)
        if index >= states.shape[1]:
            raise ValueError(f"trajectory does not contain coordinate {name}.")
        results[name] = compute_fft_psd(t, states[:, index], burn_time=burn_time, **kwargs)
    states_observed = [result["state"] for result in results.values()]
    if "broadband_spectrum" in states_observed:
        state_global = "broadband_spectrum"
    elif "dominant_periodic_peak" in states_observed:
        state_global = "dominant_periodic_peak"
    elif "quasiperiodic_candidate" in states_observed:
        state_global = "quasiperiodic_candidate"
    else:
        state_global = "spectral_inconclusive"
    return {
        "coordinate_results": results,
        "state_global": state_global,
        "psd_alone_does_not_certify_chaos": True,
        "psd_proves_chaos": False,
        "chaos_certified_by_psd": False,
        "hiddenness_certified_by_psd": False,
    }


__all__ = [
    "ALLOWED_SPECTRAL_STATES",
    "SpectrumResult",
    "compute_fft_psd",
    "fft_spectrum",
    "infer_step",
    "psd_welch",
    "spectral_diagnostics_multicoordinate",
    "trajectory_component_spectra",
]
