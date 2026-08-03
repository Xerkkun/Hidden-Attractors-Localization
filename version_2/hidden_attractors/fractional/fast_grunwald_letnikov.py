"""Offline FFT acceleration for full-history Grunwald--Letnikov operators.

Stability: experimental

This module evaluates the same causal, uniform-grid discrete convolution as
``grunwald_letnikov_derivative``.  It changes only how that convolution is
computed: zero-padded real FFTs replace the quadratic history loop.  It is a
batch/offline sampled-data operator, not an online fractional ODE solver.

For ``N`` samples and ``d`` components, the FFT route has arithmetic
complexity ``O(d N log N)`` and working storage ``O(d N)``.  Those asymptotic
bounds do not imply a universal speedup: transform setup and allocation can
make the direct Numba kernel faster for short histories.  The public selector
therefore exposes a deterministic, user-configurable crossover threshold.

The FFT length is always at least ``2*N - 1``.  Cropping the resulting linear
convolution to its first ``N`` entries gives the causal GL history and avoids
the wraparound aliasing produced by an unpadded length-``N`` circular
convolution.

References
----------
C. Lubich, "Discretized Fractional Calculus", SIAM Journal on Mathematical
Analysis 17 (1986), https://doi.org/10.1137/0517050.

M. Matusiak, "Fast Evaluation of Grunwald-Letnikov Variable Fractional-Order
Differentiation and Integration Based on the FFT Convolution", in *Advanced,
Contemporary Control*, Springer, 2020,
https://doi.org/10.1007/978-3-030-50936-1_74.

The implementation here is a fixed-order, one-shot linear convolution.  It
does not claim to reproduce the variable-order algorithm of the latter work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numba
import numpy as np
import scipy
from scipy.fft import irfft, next_fast_len, rfft

from .contracts import normalize_fractional_orders
from .grunwald_letnikov import (
    grunwald_letnikov_derivative,
    grunwald_letnikov_weights,
)


FFT_AUTO_THRESHOLD: Final[int] = 1024
"""Default sample count at which ``backend='auto'`` selects the FFT route."""

FAST_GL_REFERENCES: Final[tuple[str, ...]] = (
    "https://doi.org/10.1137/0517050",
    "https://doi.org/10.1007/978-3-030-50936-1_74",
)

_ALLOWED_DEFINITIONS: Final[frozenset[str]] = frozenset(
    {"grunwald_letnikov", "riemann_liouville_gl", "caputo_shifted"}
)
_ALLOWED_BACKENDS: Final[frozenset[str]] = frozenset({"auto", "direct", "fft"})


@dataclass(frozen=True, slots=True)
class FastGLDerivativeResult:
    """Structured result and reproducibility metadata for a batch GL operator.

    ``estimated_workspace_bytes`` is an implementation-level estimate of the
    principal arrays allocated by this module.  It excludes opaque temporary
    work buffers internal to SciPy/pocketfft, Numba, NumPy, and the allocator.
    """

    values: np.ndarray
    orders: np.ndarray
    definition: str
    method: str
    step: float
    backend: str
    backend_version: str
    sample_count: int
    dimension: int
    fft_length: int | None
    requested_backend: str
    auto_threshold: int
    arithmetic_complexity: str
    working_memory_complexity: str
    estimated_workspace_bytes: int
    execution_mode: str = "batch_offline"
    memory_policy: str = "full_history"
    history_window: None = None
    status: str = "finite_numerical_diagnostic"
    references: tuple[str, ...] = FAST_GL_REFERENCES


def gl_linear_convolution_fft_length(sample_count: int) -> int:
    """Return a fast real-FFT length sufficient for exact linear convolution.

    The word *exact* refers to convolution support rather than floating-point
    arithmetic: a length of at least ``2*N - 1`` prevents circular wraparound,
    while FFT roundoff remains present.
    """

    if isinstance(sample_count, (bool, np.bool_)):
        raise TypeError("sample_count must be a positive integer, not bool.")
    try:
        count = int(sample_count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("sample_count must be a positive integer.") from exc
    if count != sample_count or count < 1:
        raise ValueError("sample_count must be a positive integer.")
    return int(next_fast_len(2 * count - 1, real=True))


def _validate_auto_threshold(auto_threshold: int) -> int:
    if isinstance(auto_threshold, (bool, np.bool_)):
        raise TypeError("auto_threshold must be a positive integer, not bool.")
    try:
        threshold = int(auto_threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("auto_threshold must be a positive integer.") from exc
    if threshold != auto_threshold or threshold < 1:
        raise ValueError("auto_threshold must be a positive integer.")
    return threshold


def _validate_inputs(
    samples: np.ndarray,
    step: float,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    definition: str,
) -> tuple[np.ndarray, bool, float, np.ndarray, str]:
    array = np.asarray(samples, dtype=float)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("samples must have shape (n_times,) or (n_times, dimension).")
    if not np.all(np.isfinite(array)):
        raise ValueError("samples must contain only finite values.")

    step_value = float(step)
    if not np.isfinite(step_value) or step_value <= 0.0:
        raise ValueError("step must be a finite positive number.")
    normalized_orders = normalize_fractional_orders(orders, array.shape[1])

    definition_value = str(definition).strip().lower()
    if definition_value not in _ALLOWED_DEFINITIONS:
        raise ValueError(f"definition must be one of {sorted(_ALLOWED_DEFINITIONS)}.")
    return (
        np.ascontiguousarray(array),
        was_vector,
        step_value,
        np.ascontiguousarray(normalized_orders),
        definition_value,
    )


def _fft_workspace_estimate(sample_count: int, dimension: int, fft_length: int) -> int:
    """Estimate bytes for the principal real/complex arrays in the FFT path."""

    frequency_count = fft_length // 2 + 1
    real_bytes = np.dtype(np.float64).itemsize
    complex_bytes = np.dtype(np.complex128).itemsize

    # Shifted samples, weights, full inverse transform, and cropped output.
    real_storage = (2 * sample_count + fft_length + sample_count) * dimension
    # Sample spectrum, weight spectrum, and their product.
    complex_storage = 3 * frequency_count * dimension
    return int(real_storage * real_bytes + complex_storage * complex_bytes)


def _direct_workspace_estimate(sample_count: int, dimension: int) -> int:
    # Contiguous input, output, and one recurrent weight vector per worker in
    # the worst case.  Numba/runtime allocator internals are deliberately
    # excluded, just as pocketfft internals are excluded above.
    return int((2 * sample_count * dimension + sample_count * dimension) * 8)


def _fft_full_history_derivative(
    samples: np.ndarray,
    step: float,
    orders: np.ndarray,
    shift_initial: bool,
) -> tuple[np.ndarray, int]:
    sample_count, dimension = samples.shape
    fft_length = gl_linear_convolution_fft_length(sample_count)
    working_samples = samples - samples[0:1, :] if shift_initial else samples.copy()

    # Equal component orders share one recurrent weight sequence.  The
    # frequency-domain kernels are still materialized by component to keep the
    # multiplication contiguous and make the workspace estimate transparent.
    weights = np.empty((sample_count, dimension), dtype=np.float64)
    for order in np.unique(orders):
        component_mask = orders == order
        recurrent = grunwald_letnikov_weights(float(order), sample_count)
        weights[:, component_mask] = recurrent[:, None]

    sample_spectrum = rfft(working_samples, n=fft_length, axis=0)
    weight_spectrum = rfft(weights, n=fft_length, axis=0)
    linear_convolution = irfft(
        sample_spectrum * weight_spectrum,
        n=fft_length,
        axis=0,
    )
    causal = linear_convolution[:sample_count, :]
    causal *= np.power(step, -orders)[None, :]
    return causal, fft_length


def fast_grunwald_letnikov_derivative(
    samples: np.ndarray,
    step: float,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    *,
    definition: str = "grunwald_letnikov",
    backend: Literal["auto", "direct", "fft"] = "auto",
    auto_threshold: int = FFT_AUTO_THRESHOLD,
) -> FastGLDerivativeResult:
    """Evaluate a full-history GL/RL or Caputo-shifted sampled operator.

    Parameters
    ----------
    samples:
        Scalar history ``(n_times,)`` or component history
        ``(n_times, dimension)`` on a uniform grid.
    step:
        Positive grid spacing.
    orders:
        One order or one order per component, each in ``(0, 1]``.
    definition:
        ``"grunwald_letnikov"`` and ``"riemann_liouville_gl"`` select the
        raw binomial history.  ``"caputo_shifted"`` applies that history to
        ``x-x[0]`` and is the same discrete convention as the direct public
        operator for ``0 < q <= 1``.
    backend:
        ``"fft"`` forces zero-padded FFT convolution; ``"direct"`` delegates
        to the public Numba reference; ``"auto"`` selects FFT exactly when
        ``n_times >= auto_threshold``.
    auto_threshold:
        Explicit deterministic crossover for ``backend="auto"``.  It is a
        policy knob, not a claim that FFT is faster on every host.

    Notes
    -----
    This function intentionally has no finite-window option: it is specialized
    for full-history, one-shot convolution.  Online integration and streaming
    memory updates require a different algorithm and contract.
    """

    array, was_vector, step_value, normalized_orders, definition_value = _validate_inputs(
        samples,
        step,
        orders,
        definition,
    )
    backend_value = str(backend).strip().lower()
    if backend_value not in _ALLOWED_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_ALLOWED_BACKENDS)}.")
    threshold = _validate_auto_threshold(auto_threshold)
    sample_count, dimension = array.shape
    selected_backend = (
        "fft"
        if backend_value == "fft"
        or (backend_value == "auto" and sample_count >= threshold)
        else "direct"
    )

    if selected_backend == "direct":
        direct_result = grunwald_letnikov_derivative(
            array,
            step_value,
            normalized_orders,
            definition=definition_value,
        )
        values = np.asarray(direct_result.values, dtype=np.float64)
        result_values = values[:, 0] if was_vector else values
        return FastGLDerivativeResult(
            values=result_values,
            orders=normalized_orders,
            definition=definition_value,
            method="gl_direct_numba_reference",
            step=step_value,
            backend="numba",
            backend_version=numba.__version__,
            sample_count=sample_count,
            dimension=dimension,
            fft_length=None,
            requested_backend=backend_value,
            auto_threshold=threshold,
            arithmetic_complexity="O(d*N^2)",
            working_memory_complexity="O(d*N)",
            estimated_workspace_bytes=_direct_workspace_estimate(sample_count, dimension),
        )

    fft_values, fft_length = _fft_full_history_derivative(
        array,
        step_value,
        normalized_orders,
        definition_value == "caputo_shifted",
    )
    result_values = fft_values[:, 0] if was_vector else fft_values
    return FastGLDerivativeResult(
        values=result_values,
        orders=normalized_orders,
        definition=definition_value,
        method="gl_fft_linear_convolution",
        step=step_value,
        backend="scipy.fft.pocketfft",
        backend_version=scipy.__version__,
        sample_count=sample_count,
        dimension=dimension,
        fft_length=fft_length,
        requested_backend=backend_value,
        auto_threshold=threshold,
        arithmetic_complexity="O(d*N*log(N))",
        working_memory_complexity="O(d*N)",
        estimated_workspace_bytes=_fft_workspace_estimate(
            sample_count,
            dimension,
            fft_length,
        ),
    )


__all__ = [
    "FAST_GL_REFERENCES",
    "FFT_AUTO_THRESHOLD",
    "FastGLDerivativeResult",
    "fast_grunwald_letnikov_derivative",
    "gl_linear_convolution_fft_length",
]
