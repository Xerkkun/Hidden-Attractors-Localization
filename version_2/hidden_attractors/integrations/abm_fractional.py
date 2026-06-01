"""Block-restarted ABM predictor-corrector for component-wise Caputo orders.

This module is intentionally separate from the full-history workflow
integrators.  It implements the block-local memory contract used by the
Fischer et al. (2020) cloned-dynamics reproduction lane.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Mapping

import numpy as np
from scipy.special import gamma


def normalize_component_orders(
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    dimension: int,
) -> np.ndarray:
    """Return one validated derivative order per state component."""

    values = np.asarray(orders, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, dimension)
    if values.size != dimension:
        raise ValueError(
            f"orders must contain one value or {dimension} component values; "
            f"received {values.size}."
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0) or np.any(values > 1.0):
        raise ValueError("orders must contain finite values in the interval (0, 1].")
    return values


def classify_component_orders(orders: np.ndarray) -> str:
    """Classify normalized orders for result metadata."""

    values = np.asarray(orders, dtype=float)
    if np.allclose(values, 1.0):
        return "integer"
    if np.allclose(values, values[0]):
        return "commensurate_fractional"
    return "incommensurate_fractional"


def _eval_rhs(
    rhs: Callable,
    t: float,
    state: np.ndarray,
    parameters: Mapping[str, float] | None,
) -> np.ndarray:
    """Evaluate common autonomous and time-dependent RHS signatures."""

    if parameters is not None:
        try:
            return np.asarray(rhs(t, state, parameters), dtype=float)
        except TypeError:
            try:
                return np.asarray(rhs(state, parameters), dtype=float)
            except TypeError:
                pass
    try:
        return np.asarray(rhs(t, state), dtype=float)
    except TypeError:
        return np.asarray(rhs(state), dtype=float)


@lru_cache(maxsize=64)
def _weights_for_order(q: float, h: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Precompute predictor and corrector history weights for one order."""

    predictor = np.zeros((n_steps, n_steps), dtype=float)
    corrector = np.zeros((n_steps, n_steps), dtype=float)
    pred_scale = h**q / float(gamma(q + 1.0))
    corr_scale = h**q / float(gamma(q + 2.0))

    for n in range(n_steps):
        j = np.arange(n + 1, dtype=float)
        predictor[n, : n + 1] = pred_scale * (
            (n + 1.0 - j) ** q - (n - j) ** q
        )

        a0 = float(n) ** (q + 1.0) - (float(n) - q) * (float(n) + 1.0) ** q
        corrector[n, 0] = corr_scale * a0
        if n:
            j_mid = np.arange(1, n + 1, dtype=float)
            corrector[n, 1 : n + 1] = corr_scale * (
                (n - j_mid + 2.0) ** (q + 1.0)
                + (n - j_mid) ** (q + 1.0)
                - 2.0 * (n - j_mid + 1.0) ** (q + 1.0)
            )
    return predictor, corrector


def integrate_fractional_abm(
    rhs: Callable,
    x0: np.ndarray,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    h: float,
    n_steps: int,
    parameters: Mapping[str, float] | None = None,
    *,
    memory_protocol: str = "block_restart",
    divergence_norm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Integrate one ABM block with scalar or component-wise Caputo orders.

    ``memory_protocol="block_restart"`` means that the supplied ``x0`` is the
    lower-limit anchor for this block.  History from prior blocks is not
    retained.  The same implementation also permits ``q=1`` for integer
    cloned-dynamics benchmarks.
    """

    if memory_protocol not in {"block_restart", "published_block_restart"}:
        raise ValueError("integrate_fractional_abm supports block restart memory only.")
    h = float(h)
    n_steps = int(n_steps)
    if h <= 0.0:
        raise ValueError("h must be positive.")
    if n_steps < 1:
        raise ValueError("n_steps must be a positive integer.")

    anchor = np.asarray(x0, dtype=float).reshape(-1)
    component_orders = normalize_component_orders(orders, anchor.size)
    times = np.arange(n_steps + 1, dtype=float) * h
    states = np.zeros((n_steps + 1, anchor.size), dtype=float)
    rhs_values = np.zeros_like(states)
    states[0] = anchor
    rhs_values[0] = _eval_rhs(rhs, 0.0, anchor, parameters)
    if rhs_values[0].shape != anchor.shape:
        raise ValueError("rhs output shape must match x0.")

    order_groups = {
        float(q): np.flatnonzero(np.isclose(component_orders, q))
        for q in np.unique(component_orders)
    }
    weights = {
        q: _weights_for_order(q, h, n_steps)
        for q in order_groups
    }

    for n in range(n_steps):
        predictor = anchor.copy()
        for q, indices in order_groups.items():
            predictor[indices] += (
                weights[q][0][n, : n + 1] @ rhs_values[: n + 1, indices]
            )
        try:
            predicted_rhs = _eval_rhs(rhs, times[n + 1], predictor, parameters)
        except Exception:
            return times[: n + 1], states[: n + 1], "solver_exception"

        corrected = anchor.copy()
        for q, indices in order_groups.items():
            corrected[indices] += (
                weights[q][1][n, : n + 1] @ rhs_values[: n + 1, indices]
                + h**q / float(gamma(q + 2.0)) * predicted_rhs[indices]
            )
        if not np.all(np.isfinite(corrected)):
            return times[: n + 1], states[: n + 1], "nonfinite_solution"
        if divergence_norm is not None and np.linalg.norm(corrected) > float(divergence_norm):
            states[n + 1] = corrected
            return times[: n + 2], states[: n + 2], "diverged"
        states[n + 1] = corrected
        try:
            rhs_values[n + 1] = _eval_rhs(rhs, times[n + 1], corrected, parameters)
        except Exception:
            return times[: n + 2], states[: n + 2], "solver_exception"

    return times, states, "ok"


__all__ = [
    "classify_component_orders",
    "integrate_fractional_abm",
    "normalize_component_orders",
]
