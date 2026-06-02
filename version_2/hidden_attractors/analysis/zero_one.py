"""Reproducible 0-1 chaos-test diagnostics for sampled signals.

The 0-1 statistic is a supporting numerical indicator. Noise may produce a
high statistic, so this module never certifies deterministic chaos.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


ALLOWED_ZERO_ONE_STATES = {
    "zero_one_chaotic_candidate",
    "zero_one_regular_candidate",
    "zero_one_inconclusive",
}
_COORDINATE_NAMES = ("x", "y", "z")


def _prepare_signal(
    signal: Sequence[float],
    *,
    detrend: bool,
    normalize: bool,
    max_samples: int | None,
) -> np.ndarray:
    values = np.asarray(signal, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if max_samples is not None and values.size > int(max_samples):
        indices = np.linspace(0, values.size - 1, int(max_samples), dtype=int)
        values = values[indices]
    if values.size < 100:
        raise ValueError("zero_one_test requires at least 100 finite samples.")
    if detrend:
        index = np.arange(values.size, dtype=float)
        slope, intercept = np.polyfit(index, values, 1)
        values = values - (slope * index + intercept)
    if normalize:
        scale = float(np.std(values))
        if scale > np.finfo(float).eps:
            values = values / scale
    return values


def _classify_k(value: float) -> str:
    if value > 0.8:
        return "zero_one_chaotic_candidate"
    if value < 0.2:
        return "zero_one_regular_candidate"
    return "zero_one_inconclusive"


def _k_for_c(signal: np.ndarray, c_value: float) -> float:
    """Return correlation statistic K_c using modified mean-square displacement."""

    index = np.arange(1, signal.size + 1, dtype=float)
    p_values = np.cumsum(signal * np.cos(index * c_value))
    q_values = np.cumsum(signal * np.sin(index * c_value))
    max_lag = min(max(20, signal.size // 10), 500)
    lags = np.arange(1, max_lag + 1, dtype=float)
    displacement = np.empty(max_lag, dtype=float)
    mean_signal = float(np.mean(signal))
    denominator = max(1.0 - float(np.cos(c_value)), np.finfo(float).eps)
    for offset, lag in enumerate(range(1, max_lag + 1)):
        dp = p_values[lag:] - p_values[:-lag]
        dq = q_values[lag:] - q_values[:-lag]
        raw = float(np.mean(dp * dp + dq * dq))
        oscillatory = mean_signal**2 * (1.0 - float(np.cos(lag * c_value))) / denominator
        displacement[offset] = raw - oscillatory
    if np.std(displacement) <= np.finfo(float).eps:
        return 0.0
    correlation = float(np.corrcoef(lags, displacement)[0, 1])
    return correlation if np.isfinite(correlation) else 0.0


def zero_one_test(
    signal: Sequence[float],
    n_c: int = 100,
    c_values: Sequence[float] | None = None,
    random_seed: int = 12345,
    detrend: bool = True,
    normalize: bool = True,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Compute a median robust 0-1 statistic over reproducible values of ``c``."""

    values = _prepare_signal(
        signal,
        detrend=bool(detrend),
        normalize=bool(normalize),
        max_samples=max_samples,
    )
    if c_values is None:
        if int(n_c) < 1:
            raise ValueError("n_c must be positive.")
        rng = np.random.default_rng(int(random_seed))
        c_array = rng.uniform(np.pi / 5.0, 4.0 * np.pi / 5.0, size=int(n_c))
    else:
        c_array = np.asarray(c_values, dtype=float).reshape(-1)
    if c_array.size == 0 or np.any((c_array <= 0.0) | (c_array >= np.pi)):
        raise ValueError("c_values must be non-empty and lie in (0, pi).")
    k_values = np.asarray([_k_for_c(values, float(c_value)) for c_value in c_array])
    k_median = float(np.median(k_values))
    return {
        "K": k_median,
        "K_median": k_median,
        "K_mean": float(np.mean(k_values)),
        "K_std": float(np.std(k_values)),
        "K_min": float(np.min(k_values)),
        "K_max": float(np.max(k_values)),
        "c_values_count": int(c_array.size),
        "state": _classify_k(k_median),
        "signal_length": int(values.size),
        "detrend": bool(detrend),
        "normalize": bool(normalize),
        "random_seed": int(random_seed),
        "zero_one_alone_does_not_certify_chaos": True,
        "chaos_certified_by_zero_one": False,
        "hiddenness_certified_by_zero_one": False,
    }


def zero_one_multicoordinate(
    times: Sequence[float],
    trajectory: Sequence[Sequence[float]],
    burn_time: float,
    coordinates: Sequence[str] = _COORDINATE_NAMES,
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply :func:`zero_one_test` to selected post-transient coordinates."""

    t = np.asarray(times, dtype=float)
    states = np.asarray(trajectory, dtype=float)
    if t.ndim != 1 or states.ndim != 2 or states.shape[0] != t.size:
        raise ValueError("times and trajectory must have shapes (N,) and (N, d).")
    mask = t >= float(burn_time)
    retained = states[mask]
    coordinate_results: dict[str, dict[str, Any]] = {}
    for name in coordinates:
        if name not in _COORDINATE_NAMES:
            raise ValueError("coordinates must be selected from x, y, and z.")
        index = _COORDINATE_NAMES.index(name)
        if index >= retained.shape[1]:
            raise ValueError(f"trajectory does not contain coordinate {name}.")
        coordinate_results[name] = zero_one_test(retained[:, index], **kwargs)
    statistics = np.asarray([item["K"] for item in coordinate_results.values()], dtype=float)
    chaotic_count = sum(item["K"] > 0.8 for item in coordinate_results.values())
    if chaotic_count >= 2:
        state = "zero_one_chaotic_candidate"
    elif all(item["K"] < 0.2 for item in coordinate_results.values()):
        state = "zero_one_regular_candidate"
    else:
        state = "zero_one_inconclusive"
    return {
        "coordinate_results": coordinate_results,
        "K_global_median": float(np.median(statistics)),
        "K_global_max": float(np.max(statistics)),
        "K_global_min": float(np.min(statistics)),
        "state_global": state,
        "post_transient_rows": int(retained.shape[0]),
        "zero_one_alone_does_not_certify_chaos": True,
        "chaos_certified_by_zero_one": False,
        "hiddenness_certified_by_zero_one": False,
    }


__all__ = [
    "ALLOWED_ZERO_ONE_STATES",
    "zero_one_multicoordinate",
    "zero_one_test",
]
