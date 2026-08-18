"""Shared validation for fixed-step integration horizons."""

from __future__ import annotations

import math
from operator import index as operator_index
from typing import Iterable

import numpy as np


def exact_fixed_step_count(
    h: float,
    t_final: float,
    *,
    caller: str,
    max_steps: int | None = None,
) -> int:
    """Return the uniform-step count or reject a misaligned horizon.

    Fixed-step Caputo formulas and their native counterparts assume one
    constant step size. Rounding upward changes the experiment, while a short
    final step changes the fractional weights. Only floating-point
    reconstruction error is tolerated.
    """

    try:
        step = float(h)
        horizon = float(t_final)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{caller}: h and t_final must be finite numbers.") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError(f"{caller}: h must be a positive finite number; got {h!r}.")
    if not math.isfinite(horizon) or horizon < 0.0:
        raise ValueError(
            f"{caller}: t_final must be a non-negative finite number; got {t_final!r}."
        )

    ratio = horizon / step
    if not math.isfinite(ratio):
        raise ValueError(
            f"{caller}: t_final / h must be finite; got "
            f"t_final={horizon!r}, h={step!r}."
        )
    steps = int(round(ratio))
    platform_limit = int(np.iinfo(np.intp).max)
    limit = platform_limit
    if max_steps is not None:
        if isinstance(max_steps, (bool, np.bool_)):
            raise TypeError(f"{caller}: max_steps must be a positive integer.")
        try:
            requested_limit = operator_index(max_steps)
        except TypeError as exc:
            raise TypeError(f"{caller}: max_steps must be a positive integer.") from exc
        if requested_limit < 1:
            raise ValueError(f"{caller}: max_steps must be positive.")
        limit = min(limit, requested_limit)
    if steps > limit:
        raise ValueError(
            f"{caller}: t_final/h requires {steps} fixed steps, exceeding "
            f"the supported limit {limit}."
        )
    reconstructed = steps * step
    scale = max(abs(horizon), abs(reconstructed), abs(step))
    tolerance = max(
        64.0 * float(np.finfo(np.float64).eps) * scale,
        8.0 * math.ulp(horizon),
        8.0 * math.ulp(reconstructed),
    )
    if abs(reconstructed - horizon) > tolerance:
        raise ValueError(
            f"{caller}: t_final must contain an integer number of fixed steps; "
            f"got t_final={horizon!r}, h={step!r}. This solver will not "
            "silently integrate beyond the requested horizon."
        )
    return steps


def checked_array_capacity(
    shape: Iterable[int],
    dtype: np.dtype | type,
    *,
    caller: str,
    max_bytes: int | None = None,
) -> tuple[int, int]:
    """Return element/byte counts after checking platform-safe capacity."""

    dimensions: list[int] = []
    for raw_dimension in shape:
        if isinstance(raw_dimension, (bool, np.bool_)):
            raise TypeError(f"{caller}: array dimensions must be non-negative integers.")
        try:
            dimension = operator_index(raw_dimension)
        except TypeError as exc:
            raise TypeError(
                f"{caller}: array dimensions must be non-negative integers."
            ) from exc
        if dimension < 0:
            raise ValueError(f"{caller}: array dimensions must be non-negative.")
        dimensions.append(dimension)
    elements = math.prod(dimensions)
    itemsize = int(np.dtype(dtype).itemsize)
    bytes_required = elements * itemsize
    platform_limit = int(np.iinfo(np.intp).max)
    if elements > platform_limit or bytes_required > platform_limit:
        raise ValueError(f"{caller}: requested array capacity exceeds platform limits.")
    if max_bytes is not None:
        byte_limit = operator_index(max_bytes)
        if byte_limit < 0:
            raise ValueError(f"{caller}: max_bytes must be non-negative.")
        if bytes_required > byte_limit:
            raise ValueError(
                f"{caller}: requested array requires {bytes_required} bytes, "
                f"exceeding max_bytes={byte_limit}."
            )
    return elements, bytes_required


__all__ = ["checked_array_capacity", "exact_fixed_step_count"]
