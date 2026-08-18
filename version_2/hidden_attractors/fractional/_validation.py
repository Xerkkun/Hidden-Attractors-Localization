"""Private validation primitives shared by sampled fractional methods."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np


def sample_matrix(samples: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalize non-empty finite samples to a contiguous two-dimensional matrix."""

    values = np.asarray(samples, dtype=np.float64)
    was_vector = values.ndim == 1
    if was_vector:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(
            "samples must have shape (n_times,) or (n_times, dimension)."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values.")
    return np.ascontiguousarray(values, dtype=np.float64), was_vector


def real_sample_matrix(samples: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalize samples while explicitly rejecting unsupported complex data."""

    if np.iscomplexobj(samples):
        raise TypeError("samples must be real-valued; complex CQ is not implemented.")
    return sample_matrix(samples)


def strict_count(value: Any, *, name: str, minimum: int) -> int:
    """Return an integer-like value satisfying a declared lower bound."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    if normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return int(normalized)
