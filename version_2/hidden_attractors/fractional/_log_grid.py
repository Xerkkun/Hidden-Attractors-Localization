"""Numerically stable helpers for logarithmic fractional grids."""

from __future__ import annotations

from typing import Any

import numpy as np


def stable_log_ratio(numerator: Any, denominator: float) -> float | np.ndarray:
    """Return ``log(numerator / denominator)`` without close-value loss.

    ``log1p((numerator-denominator)/denominator)`` preserves adjacent floating
    values, while the difference of logarithms avoids overflow when the ratio
    itself is outside the floating-point range.  Inputs are expected to be
    finite, positive, and no smaller than ``denominator``; public contracts
    validate those conditions before calling this helper.
    """

    values = np.asarray(numerator, dtype=np.float64)
    lower = float(denominator)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative_increment = (values - lower) / lower
        close_values = np.isfinite(relative_increment) & (
            relative_increment <= 0.5
        )
        result = np.where(
            close_values,
            np.log1p(relative_increment),
            np.log(values) - np.log(lower),
        )
    if values.ndim == 0:
        return float(result)
    return np.asarray(result, dtype=np.float64)


def physical_times_from_log(
    lower_terminal: float,
    log_times: Any,
) -> float | np.ndarray:
    """Map ``u`` to ``a*exp(u)`` without avoidable cancellation or overflow.

    Direct multiplication makes ``u=0`` exactly equal to ``a`` and preserves
    close grid points.  When ``exp(u)`` or the product overflows although the
    final physical time may still be finite, evaluation falls back to log-space.
    """

    coordinates = np.asarray(log_times, dtype=np.float64)
    terminal = float(lower_terminal)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        direct = terminal * np.exp(coordinates)
        fallback = np.exp(np.log(terminal) + coordinates)
    result = np.where(np.isfinite(direct) & (direct > 0.0), direct, fallback)
    if coordinates.ndim == 0:
        return float(result)
    return np.asarray(result, dtype=np.float64)


def uniform_step_grid_metrics(duration: float, step: float) -> tuple[int, float, float]:
    """Return nearest step count, residual, and scale-aware roundoff tolerance."""

    ratio = float(duration) / float(step)
    if not np.isfinite(ratio):
        return 0, np.inf, 0.0
    nearest_steps = int(round(ratio))
    reconstructed = nearest_steps * float(step)
    residual = abs(reconstructed - float(duration))
    scale = max(abs(float(duration)), abs(reconstructed), abs(float(step)))
    spacing = max(
        abs(float(np.spacing(float(duration)))),
        abs(float(np.spacing(reconstructed))),
        float(np.finfo(np.float64).tiny),
    )
    tolerance = max(
        64.0 * float(np.finfo(np.float64).eps) * scale,
        8.0 * spacing,
    )
    return nearest_steps, residual, tolerance


__all__ = [
    "physical_times_from_log",
    "stable_log_ratio",
    "uniform_step_grid_metrics",
]
