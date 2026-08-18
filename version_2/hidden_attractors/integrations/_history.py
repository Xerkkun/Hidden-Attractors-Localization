"""Shared validation for fractional prehistory and memory policies."""

from __future__ import annotations

from operator import index as operator_index
from typing import Any

import numpy as np


def canonical_history_times(
    times: Any,
    h: float,
    *,
    caller: str,
    require_zero_anchor: bool,
) -> np.ndarray:
    """Validate a uniform source grid and return an exact grid ending at zero."""

    step = float(h)
    raw = np.asarray(times, dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{caller}: history_times must have non-empty shape (H,).")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError(f"{caller}: h must be finite and positive.")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{caller}: history_times must contain only finite values.")

    history_len = int(raw.size)
    relative = raw - raw[-1]
    expected = (
        np.arange(history_len, dtype=np.float64) - (history_len - 1)
    ) * step
    time_scale = max(
        1.0,
        float(np.max(np.abs(raw))),
        float(max(1, history_len - 1)) * abs(step),
    )
    tolerance = 64.0 * np.finfo(np.float64).eps * time_scale
    if require_zero_anchor and abs(float(raw[-1])) > tolerance:
        raise ValueError(f"{caller}: history_times must end at t=0.")
    if history_len > 1 and (
        np.any(np.diff(raw) <= 0.0)
        or np.any(np.abs(relative - expected) > tolerance)
    ):
        raise ValueError(
            f"{caller}: history_times must be strictly increasing on the same h grid."
        )
    return np.ascontiguousarray(expected)


def validate_prehistory(
    history_times: Any | None,
    history_states: Any | None,
    *,
    x0: np.ndarray,
    h: float,
    caller: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Validate paired, anchored prehistory arrays and canonicalize their grid."""

    if (history_times is None) != (history_states is None):
        raise ValueError(
            f"{caller}: history_times and history_states must be provided together."
        )
    if history_times is None:
        return None, None

    state0 = np.asarray(x0, dtype=np.float64)
    if state0.ndim != 1 or state0.size == 0 or not np.all(np.isfinite(state0)):
        raise ValueError(f"{caller}: x0 must be a non-empty finite vector.")
    canonical_times = canonical_history_times(
        history_times,
        h,
        caller=caller,
        require_zero_anchor=True,
    )
    states = np.asarray(history_states, dtype=np.float64)
    expected_shape = (canonical_times.size, state0.size)
    if states.shape != expected_shape:
        raise ValueError(
            f"{caller}: history_states must have shape {expected_shape}, got {states.shape}."
        )
    if not np.all(np.isfinite(states)):
        raise ValueError(f"{caller}: history_states must contain only finite values.")
    state_tolerance = 64.0 * np.finfo(np.float64).eps * max(
        1.0,
        float(np.max(np.abs(state0))),
        float(np.max(np.abs(states[-1]))),
    )
    if not np.allclose(states[-1], state0, rtol=0.0, atol=state_tolerance):
        raise ValueError(f"{caller}: the last history state must equal x0 at t=0.")
    # Always own the canonical buffer.  ``ascontiguousarray`` may alias an
    # already contiguous caller array, and normalizing its last row would then
    # mutate user-owned prehistory.
    canonical_states = np.array(states, dtype=np.float64, order="C", copy=True)
    canonical_states[-1] = state0
    return canonical_times, canonical_states


def validate_rhs_result(
    value: Any,
    *,
    dim: int,
    caller: str,
) -> np.ndarray:
    """Require an RHS result to be a finite vector of the state dimension."""

    try:
        derivative = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{caller}: rhs must return a real vector with shape ({dim},)."
        ) from exc
    if derivative.shape != (dim,):
        raise ValueError(
            f"{caller}: rhs must return shape ({dim},), got {derivative.shape}."
        )
    if not np.all(np.isfinite(derivative)):
        raise ValueError(f"{caller}: rhs must return only finite derivatives.")
    return np.ascontiguousarray(derivative)


def validate_equilibria(
    equilibria: Any | None,
    *,
    dim: int,
    caller: str,
) -> list[np.ndarray] | None:
    """Validate equilibrium vectors before convergence checks may broadcast."""

    if equilibria is None:
        return None
    validated: list[np.ndarray] = []
    for index, equilibrium in enumerate(equilibria):
        try:
            candidate = np.asarray(equilibrium, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{caller}: equilibria[{index}] must be a real vector with shape ({dim},)."
            ) from exc
        if candidate.shape != (dim,):
            raise ValueError(
                f"{caller}: equilibria[{index}] must have shape ({dim},), "
                f"got {candidate.shape}."
            )
        if not np.all(np.isfinite(candidate)):
            raise ValueError(
                f"{caller}: equilibria[{index}] must contain only finite values."
            )
        validated.append(np.ascontiguousarray(candidate))
    return validated


def validate_memory_policy(
    memory_mode: Any,
    memory_window_length: Any | None,
    *,
    caller: str,
) -> tuple[str, int | None]:
    """Validate the two supported memory modes and an optional sample count."""

    if memory_mode not in {"full", "window"}:
        raise ValueError(f"{caller}: memory_mode must be exactly 'full' or 'window'.")
    count: int | None = None
    if memory_window_length is not None:
        if isinstance(memory_window_length, (bool, np.bool_)):
            raise TypeError(f"{caller}: memory_window_length must be a positive integer.")
        try:
            count = operator_index(memory_window_length)
        except TypeError as exc:
            raise TypeError(
                f"{caller}: memory_window_length must be a positive integer."
            ) from exc
        if count < 1:
            raise ValueError(f"{caller}: memory_window_length must be positive.")
    if memory_mode == "window" and count is None:
        raise ValueError(
            f"{caller}: memory_window_length is required when memory_mode='window'."
        )
    return memory_mode, count


def validate_fractional_state_and_order(
    x0: Any,
    q: Any,
    *,
    caller: str,
    allow_integer_limit: bool,
) -> tuple[np.ndarray, float]:
    """Validate a direct fractional solver's state and scalar order."""

    try:
        order = float(q)
        state = np.asarray(x0, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{caller}: q and x0 must contain real numbers.") from exc
    upper_ok = order <= 1.0 if allow_integer_limit else order < 1.0
    if not np.isfinite(order) or order <= 0.0 or not upper_ok:
        interval = "0 < q <= 1" if allow_integer_limit else "0 < q < 1"
        raise ValueError(f"{caller}: q must satisfy {interval}; got {q!r}.")
    if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
        raise ValueError(f"{caller}: x0 must be a non-empty finite one-dimensional array.")
    return np.ascontiguousarray(state), order


def validate_divergence_norm(value: Any | None, *, caller: str) -> float | None:
    """Normalize a hard cutoff; ``None`` and positive infinity disable it."""

    if value is None:
        return None
    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{caller}: divergence_norm must be positive, None, or +inf."
        ) from exc
    if np.isnan(threshold) or threshold == -np.inf or threshold <= 0.0:
        raise ValueError(
            f"{caller}: divergence_norm must be positive, None, or +inf."
        )
    if threshold == np.inf:
        return None
    return threshold


__all__ = [
    "canonical_history_times",
    "validate_memory_policy",
    "validate_prehistory",
    "validate_fractional_state_and_order",
    "validate_rhs_result",
    "validate_equilibria",
    "validate_divergence_norm",
]
