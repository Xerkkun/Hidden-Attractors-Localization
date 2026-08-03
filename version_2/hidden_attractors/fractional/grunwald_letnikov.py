"""Numba-accelerated direct Grunwald-Letnikov derivative.

Stability: experimental

The operator accepts scalar or component-wise orders on an equally spaced
sample grid.  ``definition='grunwald_letnikov'`` applies the raw binomial
history. ``definition='caputo_shifted'`` applies the same discretization to
``x(t)-x(t0)`` and therefore approximates a Caputo derivative for 0 < q < 1.
These initial-condition conventions are intentionally not interchangeable.

References
----------
I. Podlubny, *Fractional Differential Equations*, Academic Press, 1999,
ISBN 978-0-12-558840-9.
C. Lubich, "Discretized Fractional Calculus", SIAM Journal on Mathematical
Analysis 17 (1986), https://doi.org/10.1137/0517050.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .contracts import normalize_fractional_orders


@dataclass(frozen=True, slots=True)
class FractionalDerivativeResult:
    """Structured output from a discrete fractional derivative."""

    values: np.ndarray
    orders: np.ndarray
    definition: str
    method: str
    step: float
    memory_policy: str
    history_window: int | None
    status: str = "finite_numerical_diagnostic"


@njit(cache=True, nogil=True)
def _weights_numba(order: float, count: int) -> np.ndarray:
    weights = np.empty(count, dtype=np.float64)
    if count == 0:
        return weights
    weights[0] = 1.0
    for k in range(1, count):
        weights[k] = weights[k - 1] * (1.0 - (order + 1.0) / k)
    return weights


@njit(cache=True, nogil=True, parallel=True)
def _gl_derivative_numba(
    samples: np.ndarray,
    step: float,
    orders: np.ndarray,
    shift_initial: bool,
    history_window: int,
) -> np.ndarray:
    n_times, dimension = samples.shape
    output = np.empty_like(samples)
    for component in prange(dimension):
        order = orders[component]
        scale = step ** (-order)
        weights = _weights_numba(order, n_times)
        anchor = samples[0, component] if shift_initial else 0.0
        for n in range(n_times):
            lower = 0
            if history_window > 0 and n + 1 > history_window:
                lower = n + 1 - history_window
            total = 0.0
            for sample_index in range(lower, n + 1):
                lag = n - sample_index
                total += weights[lag] * (samples[sample_index, component] - anchor)
            output[n, component] = scale * total
    return output


def grunwald_letnikov_weights(order: float, count: int) -> np.ndarray:
    """Return binomial GL weights ``(-1)^k * binom(order, k)`` recursively."""

    order = float(order)
    count = int(count)
    if not np.isfinite(order) or order <= 0.0 or order > 1.0:
        raise ValueError("order must be finite and lie in (0, 1].")
    if count < 0:
        raise ValueError("count must be non-negative.")
    return _weights_numba(order, count)


def grunwald_letnikov_derivative(
    samples: np.ndarray,
    step: float,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    *,
    definition: str = "grunwald_letnikov",
    history_window: int | None = None,
) -> FractionalDerivativeResult:
    """Approximate a left-sided fractional derivative on a uniform grid.

    Parameters
    ----------
    samples:
        Shape ``(n_times,)`` or ``(n_times, dimension)``.
    step:
        Positive uniform time step.
    orders:
        One order or one value per component, each in ``(0, 1]``.
    definition:
        ``"grunwald_letnikov"`` for the raw GL history,
        ``"riemann_liouville_gl"`` for the same discretization explicitly
        labelled as an RL approximation, or ``"caputo_shifted"`` for GL
        applied to ``x-x0``.
    history_window:
        Optional positive number of recent samples. Omitting it retains full
        history. A finite value changes the numerical memory contract.
    """

    array = np.asarray(samples, dtype=float)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("samples must have shape (n_times,) or (n_times, dimension).")
    if not np.all(np.isfinite(array)):
        raise ValueError("samples must contain only finite values.")
    step = float(step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be a finite positive number.")
    normalized_orders = normalize_fractional_orders(orders, array.shape[1])
    definition = str(definition).strip().lower()
    allowed = {"grunwald_letnikov", "riemann_liouville_gl", "caputo_shifted"}
    if definition not in allowed:
        raise ValueError(f"definition must be one of {sorted(allowed)}.")
    if history_window is None:
        window_value = 0
        memory_policy = "full_history"
    else:
        window_value = int(history_window)
        if window_value < 1:
            raise ValueError("history_window must be a positive integer.")
        memory_policy = "finite_window"

    output = _gl_derivative_numba(
        np.ascontiguousarray(array),
        step,
        np.ascontiguousarray(normalized_orders),
        definition == "caputo_shifted",
        window_value,
    )
    if was_vector:
        output = output[:, 0]
    return FractionalDerivativeResult(
        values=output,
        orders=normalized_orders,
        definition=definition,
        method="gl_direct_numba",
        step=step,
        memory_policy=memory_policy,
        history_window=history_window,
    )


__all__ = [
    "FractionalDerivativeResult",
    "grunwald_letnikov_derivative",
    "grunwald_letnikov_weights",
]
