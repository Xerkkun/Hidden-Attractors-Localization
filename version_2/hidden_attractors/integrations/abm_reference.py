"""Full-history component-wise PECE reference rails with linear storage.

The convolution is split into direct sums within the active block and FFT
contributions from completed blocks. The Caputo lower limit stays at zero;
FFT blocks do not restart or truncate the fractional history.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.special import gamma

from .._rhs import bind_rhs
from .._time_grid import exact_fixed_step_count
from .abm_fractional import normalize_component_orders


def integrate_reference_abm(
    rhs: Callable,
    x0: np.ndarray,
    orders: float | list[float] | np.ndarray,
    h: float,
    t_final: float,
    *,
    convolution_block_size: int = 256,
    divergence_norm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Integrate one fixed-lower-limit PECE history without dense weight tables.

    Uses one predictor and one corrector evaluation per step, with the same
    analytical weights as ``integrate_fractional_abm`` on a single long block.
    Storage is O(N*d); completed FFT blocks retain contributions to every
    future step. This is not the block-restarted clone protocol.
    """
    n_steps = exact_fixed_step_count(h, t_final, caller="integrate_reference_abm")
    if isinstance(convolution_block_size, (bool, np.bool_)) or int(convolution_block_size) != convolution_block_size or convolution_block_size < 1:
        raise ValueError("convolution_block_size must be a positive integer")
    block_size = int(convolution_block_size)
    anchor = np.asarray(x0, dtype=float)
    if anchor.ndim != 1 or not anchor.size or not np.all(np.isfinite(anchor)):
        raise ValueError("x0 must be a nonempty finite vector")
    if divergence_norm is not None and (not np.isscalar(divergence_norm)
            or isinstance(divergence_norm, (bool, np.bool_))
            or not np.isfinite(divergence_norm) or divergence_norm <= 0):
        raise ValueError("divergence_norm must be positive and finite, or None")
    component_orders = normalize_component_orders(orders, anchor.size)
    bound_rhs = bind_rhs(rhs)
    times = np.arange(n_steps + 1, dtype=float) * h
    states = np.zeros((n_steps + 1, anchor.size), dtype=float)
    values = np.zeros_like(states)
    states[0] = anchor
    initial_rhs = np.asarray(bound_rhs(0.0, anchor), dtype=float)
    if initial_rhs.shape != anchor.shape or not np.all(np.isfinite(initial_rhs)):
        raise ValueError("initial RHS must be finite and match x0")
    values[0] = initial_rhs
    lag = np.arange(n_steps, dtype=float)[:, None]
    q = component_orders[None, :]
    predictor_kernel = ((lag + 1) ** q - lag ** q) * h**q / gamma(q + 1)
    corrector_scale = h**component_orders / gamma(component_orders + 2)
    corrector_kernel = ((lag + 2) ** (q + 1) + lag ** (q + 1)
                        - 2 * (lag + 1) ** (q + 1)) * corrector_scale
    past_predictor = np.zeros((n_steps, anchor.size), dtype=float)
    past_corrector = np.zeros_like(past_predictor)
    for n in range(n_steps):
        block_start = (n // block_size) * block_size
        recent_values = values[block_start:n + 1]
        recent_lags = n - np.arange(block_start, n + 1)
        predicted = anchor + past_predictor[n] + np.sum(predictor_kernel[recent_lags] * recent_values, axis=0)
        if not np.all(np.isfinite(predicted)):
            return times[:n + 1], states[:n + 1], "nonfinite_solution"
        try:
            predicted_rhs = np.asarray(bound_rhs(times[n + 1], predicted), dtype=float)
        except Exception:
            return times[:n + 1], states[:n + 1], "solver_exception"
        if predicted_rhs.shape != anchor.shape:
            return times[:n + 1], states[:n + 1], "solver_exception"
        if not np.all(np.isfinite(predicted_rhs)):
            return times[:n + 1], states[:n + 1], "nonfinite_solution"
        # f(0) has a distinct, step-dependent corrector coefficient.
        first_recent = max(1, block_start)
        corrector_recent_lags = n - np.arange(first_recent, n + 1)
        recent_corrector = np.sum(corrector_kernel[corrector_recent_lags] * values[first_recent:n + 1], axis=0)
        a0 = float(n) ** (component_orders + 1) - (float(n) - component_orders) * float(n + 1) ** component_orders
        corrected = anchor + past_corrector[n] + recent_corrector + corrector_scale * (a0 * values[0] + predicted_rhs)
        if not np.all(np.isfinite(corrected)):
            return times[:n + 1], states[:n + 1], "nonfinite_solution"
        states[n + 1] = corrected
        if divergence_norm is not None and np.linalg.norm(corrected) > divergence_norm:
            return times[:n + 2], states[:n + 2], "diverged"
        try:
            corrected_rhs = np.asarray(bound_rhs(times[n + 1], corrected), dtype=float)
        except Exception:
            return times[:n + 2], states[:n + 2], "solver_exception"
        if corrected_rhs.shape != anchor.shape:
            return times[:n + 2], states[:n + 2], "solver_exception"
        if not np.all(np.isfinite(corrected_rhs)):
            return times[:n + 2], states[:n + 2], "nonfinite_solution"
        values[n + 1] = corrected_rhs
        if (n + 1) % block_size == 0 and n + 1 < n_steps:
            future_start = n + 1
            # Completed sources j in [block_start,n] contribute only at
            # future convolution indices. f[n+1] belongs to the next block.
            source = values[block_start:n + 1]
            remaining = n_steps - block_start
            fft_size = next_fast_len(remaining + len(source) - 1)
            transformed_source = rfft(source, n=fft_size, axis=0)
            pred_convolution = irfft(transformed_source * rfft(predictor_kernel[:remaining], n=fft_size, axis=0), n=fft_size, axis=0)
            offset = future_start - block_start
            past_predictor[future_start:] += pred_convolution[offset:remaining]
            if block_start == 0:
                source = source.copy()
                source[0] = 0.0
                transformed_source = rfft(source, n=fft_size, axis=0)
            corr_convolution = irfft(transformed_source * rfft(corrector_kernel[:remaining], n=fft_size, axis=0), n=fft_size, axis=0)
            past_corrector[future_start:] += corr_convolution[offset:remaining]
    return times, states, "ok"


__all__ = ["integrate_reference_abm"]
