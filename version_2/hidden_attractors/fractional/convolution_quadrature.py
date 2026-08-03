"""Lubich convolution quadrature for sampled fractional operators.

Stability: experimental

This module evaluates a *sampled operator*.  It is deliberately not an FDE
solver.  For a BDF generating polynomial ``delta(z)`` it forms the weights

``delta(z)**q = sum(omega[k] * z**k, k=0..infinity)``

and applies ``h**(-q) * sum(omega[k] * x[n-k], k=0..n)``.  BDF1 is exactly
the Grunwald--Letnikov formula.  BDF2 uses
``delta(z) = 3/2 - 2*z + z**2/2``.  Only the left-sided, terminal-truncated,
uniform-grid convention is implemented.

No starting corrections are implemented.  In particular, the first BDF2
values are the direct truncated-history convolution; they are not silently
replaced by a BDF1 start.  Starting corrections can be important for
nonsmooth or incompatible data, as analysed by Jin, Li, and Zhou (2017).

References
----------
C. Lubich, "Discretized Fractional Calculus", SIAM Journal on Mathematical
Analysis 17 (1986), https://doi.org/10.1137/0517050.
C. Lubich, "Convolution Quadrature Revisited", BIT Numerical Mathematics 44
(2004), https://doi.org/10.1023/B:BITN.0000046813.23911.2D.
B. Jin, B. Li, and Z. Zhou, "Correction of High-Order BDF Convolution
Quadrature for Fractional Evolution Equations", SIAM Journal on Scientific
Computing 39 (2017), https://doi.org/10.1137/17M1118816.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as operator_index

import numpy as np
from numba import njit, prange

from .contracts import normalize_fractional_orders
from .grunwald_letnikov import grunwald_letnikov_weights


RL_OPERATOR_ONLY_INITIAL_CONDITION = "operator_only_no_ivp"
"""Acknowledgement token for raw RL sampled-operator evaluation."""

CAPUTO_SHIFTED_INITIAL_CONDITION = "point_value_shift_x_minus_x0"
"""Initial-value convention used by the Caputo-shifted discrete operator."""

_DEFINITIONS = frozenset({"riemann_liouville", "caputo_shifted"})
_BACKENDS = frozenset({"numba", "python", "fft"})
_REFERENCES = (
    "https://doi.org/10.1137/0517050",
    "https://doi.org/10.1023/B:BITN.0000046813.23911.2D",
    "https://doi.org/10.1137/17M1118816",
)


@dataclass(frozen=True, slots=True)
class LubichConvolutionQuadratureResult:
    """Structured output of a finite sampled-operator evaluation.

    ``weights`` has shape ``(n_times, dimension)`` even when components share
    an order.  This makes the precise coefficients used for every component
    auditable.  ``values`` preserves a one-dimensional input as a vector.
    """

    values: np.ndarray
    times: np.ndarray
    orders: np.ndarray
    weights: np.ndarray
    definition: str
    bdf_order: int
    delta_formula: str
    generating_formula: str
    backend: str
    time_complexity: str
    working_memory: str
    step: float
    lower_terminal: float
    initial_condition_semantics: str
    startup_convention: str
    starting_corrections: str
    references: tuple[str, ...]
    scope: str = "sampled_fractional_operator_only_not_an_fde_solver"
    status: str = "finite_numerical_diagnostic"


def _bdf_coefficients(bdf_order: int) -> tuple[np.ndarray, str]:
    if isinstance(bdf_order, (bool, np.bool_)):
        raise ValueError("bdf_order must be 1 or 2, not a Boolean value.")
    try:
        normalized = int(bdf_order)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bdf_order must be 1 or 2.") from exc
    if normalized != bdf_order or normalized not in (1, 2):
        raise ValueError("bdf_order must be 1 or 2.")
    if normalized == 1:
        return np.array([1.0, -1.0]), "delta(z) = 1 - z"
    return (
        np.array([1.5, -2.0, 0.5]),
        "delta(z) = 3/2 - 2*z + z^2/2",
    )


@njit(cache=True, nogil=True)
def _bdf2_weights_numba(order: float, count: int) -> np.ndarray:
    weights = np.empty(count, dtype=np.float64)
    if count == 0:
        return weights
    # For q=1 the generating series is the finite BDF2 polynomial.  Assigning
    # it directly prevents roundoff-sized recurrence tails.
    if order == 1.0:
        weights[:] = 0.0
        weights[0] = 1.5
        if count > 1:
            weights[1] = -2.0
        if count > 2:
            weights[2] = 0.5
        return weights

    a0 = 1.5
    a1 = -2.0
    a2 = 0.5
    weights[0] = a0**order
    for k in range(1, count):
        # From delta W' = q delta' W for W(z)=delta(z)^q:
        # w_k = sum_j (j(q+1)-k) a_j w_(k-j) / (a_0 k).
        total = ((order + 1.0) - k) * a1 * weights[k - 1]
        if k >= 2:
            total += (2.0 * (order + 1.0) - k) * a2 * weights[k - 2]
        weights[k] = total / (a0 * k)
    return weights


def lubich_bdf_weights(
    order: float,
    count: int,
    *,
    bdf_order: int = 1,
) -> np.ndarray:
    """Return coefficients of ``delta(z)**order`` for BDF1 or BDF2.

    BDF1 delegates to the canonical GL recurrence, so the returned array is
    exactly the same array of coefficients used by
    :func:`grunwald_letnikov_weights`.  For BDF2 the stable two-term
    recurrence follows by differentiating the generating identity and
    equating coefficients.  Weight generation costs ``O(count)`` time and
    ``O(count)`` memory.
    """

    if isinstance(order, (bool, np.bool_)) or np.iscomplexobj(order):
        raise TypeError("order must be a real number, not Boolean or complex.")
    order = float(order)
    if isinstance(count, (bool, np.bool_)):
        raise TypeError("count must be a non-negative integer, not Boolean.")
    try:
        count = operator_index(count)
    except TypeError as exc:
        raise TypeError("count must be a non-negative integer.") from exc
    if not np.isfinite(order) or order <= 0.0 or order > 1.0:
        raise ValueError("order must be finite and lie in (0, 1].")
    if count < 0:
        raise ValueError("count must be non-negative.")
    _, _ = _bdf_coefficients(bdf_order)
    if bdf_order == 1:
        return grunwald_letnikov_weights(order, count)
    return _bdf2_weights_numba(order, count)


def _sample_matrix(samples: np.ndarray) -> tuple[np.ndarray, bool]:
    if np.iscomplexobj(samples):
        raise TypeError("samples must be real-valued; complex CQ is not implemented.")
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
    return np.ascontiguousarray(values), was_vector


def _time_grid(
    n_times: int,
    *,
    step: float | None,
    times: np.ndarray | None,
    lower_terminal: float,
) -> tuple[np.ndarray, float, float]:
    terminal = float(lower_terminal)
    if not np.isfinite(terminal):
        raise ValueError("lower_terminal must be finite.")
    if (step is None) == (times is None):
        raise ValueError("Pass exactly one of step or times.")

    if times is None:
        normalized_step = float(step)
        if not np.isfinite(normalized_step) or normalized_step <= 0.0:
            raise ValueError("step must be a finite positive number.")
        with np.errstate(over="ignore", invalid="ignore"):
            next_time = terminal + normalized_step
            grid = terminal + normalized_step * np.arange(n_times, dtype=np.float64)
        if not np.isfinite(next_time) or next_time <= terminal:
            raise ValueError(
                "step is not representable as a positive increment at lower_terminal."
            )
        if not np.all(np.isfinite(grid)) or (
            n_times > 1 and np.any(np.diff(grid) <= 0.0)
        ):
            raise ValueError("the grid constructed from step is not finite and increasing.")
        return grid, normalized_step, terminal

    if np.iscomplexobj(times):
        raise TypeError("times must be real-valued.")
    grid = np.asarray(times, dtype=np.float64).reshape(-1)
    if grid.size != n_times:
        raise ValueError("times length must match the number of sample rows.")
    if not np.all(np.isfinite(grid)):
        raise ValueError("times must contain only finite values.")
    if grid[0] != terminal:
        raise ValueError("times[0] must equal lower_terminal.")
    if n_times < 2:
        raise ValueError("times requires at least two samples to infer the step.")
    differences = np.diff(grid)
    normalized_step = float(differences[0])
    if not np.isfinite(normalized_step) or normalized_step <= 0.0:
        raise ValueError("times must be strictly increasing.")
    step_scale = max(1.0, abs(normalized_step))
    if not np.allclose(
        differences,
        normalized_step,
        rtol=2.0e-12,
        atol=64.0 * np.finfo(np.float64).eps * step_scale,
    ):
        raise ValueError("Lubich BDF convolution quadrature requires a uniform grid.")
    return np.ascontiguousarray(grid), normalized_step, terminal


def _validate_definition_and_initial_condition(
    definition: str,
    initial_condition_semantics: str,
) -> tuple[str, str]:
    normalized_definition = str(definition).strip().lower()
    if normalized_definition not in _DEFINITIONS:
        raise ValueError(f"definition must be one of {sorted(_DEFINITIONS)}.")
    normalized_semantics = str(initial_condition_semantics).strip().lower()
    required = (
        RL_OPERATOR_ONLY_INITIAL_CONDITION
        if normalized_definition == "riemann_liouville"
        else CAPUTO_SHIFTED_INITIAL_CONDITION
    )
    if normalized_semantics != required:
        raise ValueError(
            f"definition={normalized_definition!r} requires "
            f"initial_condition_semantics={required!r}."
        )
    return normalized_definition, normalized_semantics


def _component_weights(
    orders: np.ndarray,
    count: int,
    bdf_order: int,
) -> np.ndarray:
    weights = np.empty((count, orders.size), dtype=np.float64)
    for component, order in enumerate(orders):
        weights[:, component] = lubich_bdf_weights(
            float(order), count, bdf_order=bdf_order
        )
    return weights


@njit(cache=True, nogil=True, parallel=True)
def _direct_convolution_numba(
    samples: np.ndarray,
    weights: np.ndarray,
    orders: np.ndarray,
    step: float,
) -> np.ndarray:
    n_times, dimension = samples.shape
    output = np.empty_like(samples)
    for component in prange(dimension):
        scale = step ** (-orders[component])
        for n in range(n_times):
            total = 0.0
            for lag in range(n + 1):
                total += weights[lag, component] * samples[n - lag, component]
            output[n, component] = scale * total
    return output


def _direct_convolution_python(
    samples: np.ndarray,
    weights: np.ndarray,
    orders: np.ndarray,
    step: float,
) -> np.ndarray:
    n_times, dimension = samples.shape
    output = np.empty_like(samples)
    for component in range(dimension):
        scale = step ** (-float(orders[component]))
        for n in range(n_times):
            total = 0.0
            for lag in range(n + 1):
                total += weights[lag, component] * samples[n - lag, component]
            output[n, component] = scale * total
    return output


def _fft_convolution(
    samples: np.ndarray,
    weights: np.ndarray,
    orders: np.ndarray,
    step: float,
) -> np.ndarray:
    """Return a linear (never circular) convolution via zero-padded FFTs."""

    n_times, dimension = samples.shape
    required = 2 * n_times - 1
    fft_length = 1 << max(0, (required - 1).bit_length())
    output = np.empty_like(samples)
    for component in range(dimension):
        sample_spectrum = np.fft.rfft(samples[:, component], n=fft_length)
        weight_spectrum = np.fft.rfft(weights[:, component], n=fft_length)
        convolution = np.fft.irfft(
            sample_spectrum * weight_spectrum, n=fft_length
        )[:n_times]
        output[:, component] = (
            step ** (-float(orders[component])) * convolution
        )
    return output


def lubich_convolution_quadrature(
    samples: np.ndarray,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    *,
    bdf_order: int = 1,
    definition: str = "riemann_liouville",
    step: float | None = None,
    times: np.ndarray | None = None,
    lower_terminal: float = 0.0,
    initial_condition_semantics: str,
    backend: str = "numba",
) -> LubichConvolutionQuadratureResult:
    """Evaluate a left fractional operator with BDF convolution quadrature.

    Parameters
    ----------
    samples:
        Scalar or component-wise history with shape ``(n_times,)`` or
        ``(n_times, dimension)``.
    orders:
        One order or one value per component.  This vertical is intentionally
        limited to ``0 < q <= 1``; higher-order Caputo shifts require more
        initial polynomial data than ``x(t0)``.
    bdf_order:
        ``1`` for ``delta(z)=1-z`` or ``2`` for
        ``delta(z)=3/2-2*z+z**2/2``.
    definition:
        ``"riemann_liouville"`` applies the raw terminal-truncated history.
        ``"caputo_shifted"`` applies the same discrete operator to
        ``x(t)-x(t0)`` component by component.
    step, times:
        Pass exactly one.  ``times`` must be uniformly spaced and start at
        ``lower_terminal``; ``step`` constructs that grid explicitly.
    initial_condition_semantics:
        Raw RL evaluation requires ``"operator_only_no_ivp"``.  The shifted
        convention requires ``"point_value_shift_x_minus_x0"``.  Neither
        token turns this sampled operator into an IVP solver.
    backend:
        Explicit selector: ``"numba"`` or ``"python"`` performs the direct
        ``O(d*N**2)`` history sum.  ``"fft"`` performs a zero-padded batch
        convolution in ``O(d*N*log(N))`` time and ``O(d*N)`` working memory.

    Notes
    -----
    No starting corrections are applied.  At ``q=1``, BDF2 therefore returns
    ``(3/2*x[0])/h`` at the terminal and
    ``(3/2*x[1]-2*x[0])/h`` at the next sample under the raw convention; the
    unavailable pre-terminal samples are absent, not extrapolated.  For the
    shifted convention the same startup is applied to ``x-x[0]``.
    """

    values, was_vector = _sample_matrix(samples)
    _, delta_formula = _bdf_coefficients(bdf_order)
    bdf_order = int(bdf_order)
    selected_backend = str(backend).strip().lower()
    if selected_backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}.")
    normalized_definition, semantics = _validate_definition_and_initial_condition(
        definition, initial_condition_semantics
    )
    grid, normalized_step, terminal = _time_grid(
        values.shape[0],
        step=step,
        times=times,
        lower_terminal=lower_terminal,
    )
    if np.iscomplexobj(orders):
        raise TypeError("orders must be real-valued.")
    normalized_orders = normalize_fractional_orders(orders, values.shape[1])
    normalized_orders = np.ascontiguousarray(normalized_orders)
    weights = _component_weights(normalized_orders, values.shape[0], bdf_order)

    operator_samples = values
    if normalized_definition == "caputo_shifted":
        operator_samples = np.ascontiguousarray(values - values[0:1, :])

    if selected_backend == "numba":
        output = _direct_convolution_numba(
            operator_samples, weights, normalized_orders, normalized_step
        )
        time_complexity = "O(d*N^2) direct full-history convolution"
        working_memory = "O(d*N) weights and output"
    elif selected_backend == "python":
        output = _direct_convolution_python(
            operator_samples, weights, normalized_orders, normalized_step
        )
        time_complexity = "O(d*N^2) direct full-history convolution"
        working_memory = "O(d*N) weights and output"
    else:
        output = _fft_convolution(
            operator_samples, weights, normalized_orders, normalized_step
        )
        time_complexity = "O(d*N*log(N)) zero-padded batch convolution"
        working_memory = "O(d*N) zero-padded FFT work arrays"

    if was_vector:
        output = output[:, 0]
    return LubichConvolutionQuadratureResult(
        values=output,
        times=grid,
        orders=normalized_orders,
        weights=weights,
        definition=normalized_definition,
        bdf_order=bdf_order,
        delta_formula=delta_formula,
        generating_formula="(delta(z)/h)^q with delta(z)^q = sum_k omega_k z^k",
        backend=selected_backend,
        time_complexity=time_complexity,
        working_memory=working_memory,
        step=normalized_step,
        lower_terminal=terminal,
        initial_condition_semantics=semantics,
        startup_convention="terminal_truncated_history_no_prehistory_extrapolation",
        starting_corrections="none_implemented",
        references=_REFERENCES,
    )


__all__ = [
    "CAPUTO_SHIFTED_INITIAL_CONDITION",
    "LubichConvolutionQuadratureResult",
    "RL_OPERATOR_ONLY_INITIAL_CONDITION",
    "lubich_bdf_weights",
    "lubich_convolution_quadrature",
]
