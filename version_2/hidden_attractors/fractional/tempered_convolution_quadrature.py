r"""Tempered Lubich convolution quadrature for sampled operators.

Stability: experimental

This module evaluates left-sided *sampled operators* on a uniform physical-
time grid.  It is not an FDE solver.  For ``tau=t-a`` HAFO fixes the
exponential-conjugation definitions

.. math::

   D_{a+}^{q,\lambda}x(t)
   =e^{-\lambda\tau}D_{a+}^{q}
      [e^{\lambda(\,\cdot-a)}x](t),

and

.. math::

   {}^C D_{a+}^{q,\lambda}x(t)
   =e^{-\lambda\tau}{}^C D_{a+}^{q}
      [e^{\lambda(\,\cdot-a)}x](t).

Materialising ``exp(+lambda*tau)`` is numerically unsafe and unnecessary.  If
``omega[k]`` are the canonical HAFO BDF1/BDF2 Lubich weights, the raw tempered
history is evaluated with

``omega[k] * exp(-lambda*k*h)``.

For the conjugated Caputo definition the exact finite-history correction is

``-x(a)*exp(-lambda*n*h)*sum(omega[k], k=0..n)``.

Both expressions are algebraically identical to explicit exponential
conjugation, but contain only non-growing exponentials for ``lambda >= 0``.
Underflow of a damping factor to zero is therefore safe: it discards a term
that is already below the representable floating-point range.  No positive
exponential is formed anywhere in this implementation.

The convention is the **unnormalized conjugated derivative**.  HAFO does not
subtract ``lambda**q*x``.  Its discrete symbol is
``h**(-q)*delta(exp(-lambda*h)*z)**q``; it does not silently substitute the
distinct finite-step approximation ``(delta(z)/h + lambda)**q``.  Those
normalizations and symbols must be exposed as separate operators if they are
implemented in the future.

BDF1 and BDF2 inherit the canonical HAFO weights, direct Python and Numba
history kernels, and zero-padded linear FFT convolution from
``convolution_quadrature``.  No starting corrections or pre-terminal samples
are introduced.  In particular, BDF2 starts with its terminal-truncated BDF2
formula rather than a silent BDF1 step.

References
----------
F. Sabzikar, M. M. Meerschaert, and J. Chen, "Tempered fractional calculus",
Journal of Computational Physics 293 (2015), 14--28,
https://doi.org/10.1016/j.jcp.2014.04.024.
C. Lubich, "Discretized Fractional Calculus", SIAM Journal on Mathematical
Analysis 17 (1986), https://doi.org/10.1137/0517050.
C. Lubich, "Convolution Quadrature Revisited", BIT Numerical Mathematics 44
(2004), https://doi.org/10.1023/B:BITN.0000046813.23911.2D.
C. Li, W. Deng, and L. Zhao, "Well-posedness and numerical algorithm for the
tempered fractional differential equations", DCDS-B 24 (2019),
https://doi.org/10.3934/dcdsb.2019026.
M. Chen and W. Deng, "Discretized fractional substantial calculus",
ESAIM: Mathematical Modelling and Numerical Analysis 49 (2015), 373--394,
https://doi.org/10.1051/m2an/2014037.
L. Guo, F. Zeng, I. Turner, K. Burrage, and G. E. Karniadakis, "Efficient
Multistep Methods for Tempered Fractional Calculus: Algorithms and
Simulations", SIAM Journal on Scientific Computing 41 (2019),
https://doi.org/10.1137/18M1230153.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import normalize_fractional_orders
from .convolution_quadrature import (
    CAPUTO_SHIFTED_INITIAL_CONDITION,
    RL_OPERATOR_ONLY_INITIAL_CONDITION,
    _bdf_coefficients,
    _direct_convolution_numba,
    _direct_convolution_python,
    _fft_convolution,
    _sample_matrix,
    _time_grid,
    lubich_bdf_weights,
    lubich_convolution_quadrature,
)


TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION = "tempered_operator_only_no_ivp"
"""Acknowledgement token for raw tempered Riemann--Liouville evaluation."""

TEMPERED_CAPUTO_INITIAL_CONDITION = (
    "tempered_caputo_conjugated_point_value_shift"
)
"""Point-value token for the exponentially conjugated Caputo convention."""

_DEFINITIONS = frozenset({"tempered_riemann_liouville", "tempered_caputo"})
_BACKENDS = frozenset({"numba", "python", "fft"})
_REFERENCES = (
    "https://doi.org/10.1016/j.jcp.2014.04.024",
    "https://doi.org/10.1137/0517050",
    "https://doi.org/10.1023/B:BITN.0000046813.23911.2D",
    "https://doi.org/10.3934/dcdsb.2019026",
    "https://doi.org/10.1051/m2an/2014037",
    "https://doi.org/10.1137/18M1230153",
)


@dataclass(frozen=True, slots=True)
class TemperedConvolutionQuadratureResult:
    """Structured finite-grid tempered-operator evaluation.

    ``base_weights`` contains the coefficients of ``delta(z)**q`` and
    ``weights`` contains the coefficients actually convolved with the samples,
    ``base_weights[k] * exp(-tempering*k*step)``.  Both arrays have shape
    ``(n_times, dimension)`` so component-wise orders and tempering parameters
    remain auditable.
    """

    values: np.ndarray
    times: np.ndarray
    orders: np.ndarray
    tempering: np.ndarray
    weights: np.ndarray
    base_weights: np.ndarray
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
    conjugation: str
    tempering_convention: str
    normalization_correction: str
    caputo_initial_correction: str
    startup_convention: str
    starting_corrections: str
    positive_exponential_materialized: bool
    damping_underflowed: bool
    references: tuple[str, ...]
    scope: str = "sampled_fractional_operator_only_not_an_fde_solver"
    status: str = "finite_numerical_diagnostic"


def _strict_real_array(value: Any, *, name: str) -> np.ndarray:
    """Return a real numeric array without silently accepting bool/strings."""

    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be real-valued, not complex.")
    raw = np.asarray(value)
    if raw.dtype.kind == "b":
        raise TypeError(f"{name} must be real-valued, not Boolean.")
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")
    return np.asarray(raw, dtype=np.float64)


def _strict_real_scalar(value: Any, *, name: str) -> float:
    array = _strict_real_array(value, name=name)
    if array.ndim != 0:
        raise TypeError(f"{name} must be a real scalar.")
    normalized = float(array)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _normalize_tempering(tempering: Any, dimension: int) -> np.ndarray:
    raw = _strict_real_array(tempering, name="tempering")
    if raw.ndim > 1:
        raise ValueError("tempering must be a scalar or one-dimensional array.")
    values = raw.reshape(-1)
    if values.size == 1:
        values = np.repeat(values, dimension)
    if values.size != dimension:
        raise ValueError(
            f"tempering must contain one value or {dimension} values; "
            f"received {values.size}."
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("tempering must be finite and non-negative.")
    # Canonicalise negative zero so metadata and the exact lambda=0 reduction
    # do not depend on a signed-zero input spelling.
    values[values == 0.0] = 0.0
    return np.ascontiguousarray(values)


def _normalize_orders(orders: Any, dimension: int) -> np.ndarray:
    raw = _strict_real_array(orders, name="orders")
    if raw.ndim > 1:
        raise ValueError("orders must be a scalar or one-dimensional array.")
    normalized = normalize_fractional_orders(orders, dimension)
    return np.ascontiguousarray(normalized)


def _definition_contract(
    definition: str,
    initial_condition_semantics: str,
) -> tuple[str, str, str]:
    normalized_definition = str(definition).strip().lower()
    if normalized_definition not in _DEFINITIONS:
        raise ValueError(f"definition must be one of {sorted(_DEFINITIONS)}.")
    normalized_semantics = str(initial_condition_semantics).strip().lower()
    if normalized_definition == "tempered_riemann_liouville":
        required = TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION
        cq_definition = "riemann_liouville"
        cq_semantics = RL_OPERATOR_ONLY_INITIAL_CONDITION
    else:
        required = TEMPERED_CAPUTO_INITIAL_CONDITION
        cq_definition = "caputo_shifted"
        cq_semantics = CAPUTO_SHIFTED_INITIAL_CONDITION
    if normalized_semantics != required:
        raise ValueError(
            f"definition={normalized_definition!r} requires "
            f"initial_condition_semantics={required!r}."
        )
    return normalized_definition, cq_definition, cq_semantics


def _base_component_weights(
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


def _damped_weights(
    base_weights: np.ndarray,
    tempering: np.ndarray,
    step: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    lags = np.arange(base_weights.shape[0], dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        exponents = (lags * step)[:, None] * tempering[None, :]
        damping = np.exp(-exponents)
    # Finite non-negative inputs can produce +inf only by multiplication
    # overflow.  exp(-inf)=0 is the correct representable limiting damping.
    if np.any(np.isnan(damping)) or np.any(damping < 0.0) or np.any(damping > 1.0):
        raise ValueError("tempering and grid produced invalid damping factors.")
    damped = np.ascontiguousarray(base_weights * damping)
    if not np.all(np.isfinite(damped)):
        raise ValueError("tempered CQ weights are non-finite.")
    underflowed = bool(np.any((damping == 0.0) & (exponents > 0.0)))
    return damped, damping, underflowed


def _unscaled_convolution(
    samples: np.ndarray,
    weights: np.ndarray,
    backend: str,
) -> np.ndarray:
    """Reuse canonical CQ convolution kernels with unit external scaling."""

    zero_orders = np.zeros(samples.shape[1], dtype=np.float64)
    if backend == "numba":
        return _direct_convolution_numba(samples, weights, zero_orders, 1.0)
    if backend == "python":
        return _direct_convolution_python(samples, weights, zero_orders, 1.0)
    return _fft_convolution(samples, weights, zero_orders, 1.0)


def _cost_metadata(backend: str) -> tuple[str, str]:
    if backend == "fft":
        return (
            "O(d*N*log(N)) zero-padded batch convolution",
            "O(d*N) base/damped weights, output, and FFT work arrays",
        )
    return (
        "O(d*N^2) direct full-history convolution",
        "O(d*N) base/damped weights and output",
    )


def tempered_convolution_quadrature(
    samples: np.ndarray,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    *,
    tempering: float | list[float] | tuple[float, ...] | np.ndarray,
    bdf_order: int = 1,
    definition: str = "tempered_riemann_liouville",
    step: float | None = None,
    times: np.ndarray | None = None,
    lower_terminal: float = 0.0,
    initial_condition_semantics: str,
    backend: str = "numba",
) -> TemperedConvolutionQuadratureResult:
    """Evaluate a tempered RL or conjugated-Caputo sampled operator.

    Parameters
    ----------
    samples:
        Real history with shape ``(n_times,)`` or ``(n_times, dimension)``.
    orders:
        One order or one order per component, each in ``0 < q <= 1``.
    tempering:
        One tempering parameter or one per component.  Every value must be
        finite and non-negative.  ``lambda=0`` reduces through the canonical
        untempered CQ implementation exactly.
    bdf_order:
        ``1`` or ``2``.  No starting corrections are applied.
    definition:
        ``"tempered_riemann_liouville"`` is the raw conjugated RL operator.
        ``"tempered_caputo"`` is the conjugated Caputo operator, including
        the exact point-value correction after exponential conjugation.
    step, times:
        Pass exactly one.  ``times`` must be uniform and start exactly at
        ``lower_terminal``; ``step`` constructs that grid.
    initial_condition_semantics:
        The raw definition requires
        ``TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION``.  The Caputo definition
        requires ``TEMPERED_CAPUTO_INITIAL_CONDITION``.  These explicit tokens
        prevent an operator evaluation from being mistaken for an IVP solve.
    backend:
        ``"python"`` and ``"numba"`` are direct full-history evaluations;
        ``"fft"`` is a zero-padded, one-shot linear convolution.  FFT is not
        a streaming or online fast-history method.
    """

    _ = _strict_real_array(samples, name="samples")
    values, was_vector = _sample_matrix(samples)
    normalized_orders = _normalize_orders(orders, values.shape[1])
    normalized_tempering = _normalize_tempering(tempering, values.shape[1])
    normalized_definition, cq_definition, cq_semantics = _definition_contract(
        definition, initial_condition_semantics
    )

    selected_backend = str(backend).strip().lower()
    if selected_backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}.")
    _, delta_formula = _bdf_coefficients(bdf_order)
    bdf_order = int(bdf_order)

    terminal = _strict_real_scalar(lower_terminal, name="lower_terminal")
    normalized_step_input: float | None = None
    normalized_times_input: np.ndarray | None = None
    if step is not None:
        normalized_step_input = _strict_real_scalar(step, name="step")
    if times is not None:
        normalized_times_input = _strict_real_array(times, name="times")
    grid, normalized_step, terminal = _time_grid(
        values.shape[0],
        step=normalized_step_input,
        times=normalized_times_input,
        lower_terminal=terminal,
    )

    # Exact software reduction, including the base routine's arithmetic order.
    if np.all(normalized_tempering == 0.0):
        try:
            base_result = lubich_convolution_quadrature(
                values,
                normalized_orders,
                bdf_order=bdf_order,
                definition=cq_definition,
                times=grid if times is not None else None,
                step=normalized_step if times is None else None,
                lower_terminal=terminal,
                initial_condition_semantics=cq_semantics,
                backend=selected_backend,
            )
        except OverflowError as exc:
            raise ValueError(
                "step**(-orders) or the scaled tempered convolution overflows "
                "float64."
            ) from exc
        output_matrix = np.asarray(base_result.values, dtype=np.float64)
        if output_matrix.ndim == 1:
            output_matrix = output_matrix[:, None]
        if not np.all(np.isfinite(output_matrix)):
            raise ValueError(
                "step**(-orders) or the scaled tempered convolution overflows "
                "float64."
            )
        base_weights = np.ascontiguousarray(base_result.weights)
        tempered_weights = np.array(base_weights, copy=True)
        damping_underflowed = False
        time_complexity = base_result.time_complexity
        working_memory = base_result.working_memory
    else:
        base_weights = _base_component_weights(
            normalized_orders, values.shape[0], bdf_order
        )
        tempered_weights, damping, damping_underflowed = _damped_weights(
            base_weights, normalized_tempering, normalized_step
        )
        with np.errstate(over="ignore", invalid="ignore"):
            unscaled = _unscaled_convolution(
                values, tempered_weights, selected_backend
            )
        if not np.all(np.isfinite(unscaled)):
            raise ValueError(
                "tempered convolution overflowed or produced non-finite values."
            )

        if normalized_definition == "tempered_caputo":
            base_partial_sums = np.cumsum(base_weights, axis=0)
            with np.errstate(over="ignore", invalid="ignore"):
                correction = (
                    values[0:1, :]
                    * damping
                    * base_partial_sums
                )
                unscaled = unscaled - correction
            # The conjugated Caputo terminal is exactly zero by construction;
            # assign it explicitly so signed roundoff cannot leak into metadata
            # or q=1 startup tests.
            unscaled[0, :] = 0.0
            if not np.all(np.isfinite(unscaled)):
                raise ValueError(
                    "tempered Caputo initial correction overflowed or produced "
                    "non-finite values."
                )

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scales = np.power(normalized_step, -normalized_orders)
            output_matrix = unscaled * scales[None, :]
        if not np.all(np.isfinite(scales)) or not np.all(np.isfinite(output_matrix)):
            raise ValueError(
                "step**(-orders) or the scaled tempered convolution overflows "
                "float64."
            )
        time_complexity, working_memory = _cost_metadata(selected_backend)

    output: np.ndarray = output_matrix
    if was_vector:
        output = output_matrix[:, 0]
    caputo_correction = (
        "-x(a)*exp(-lambda*n*h)*sum_{k=0}^n omega_k"
        if normalized_definition == "tempered_caputo"
        else "not_applicable_raw_tempered_riemann_liouville"
    )
    return TemperedConvolutionQuadratureResult(
        values=output,
        times=grid,
        orders=normalized_orders,
        tempering=normalized_tempering,
        weights=tempered_weights,
        base_weights=base_weights,
        definition=normalized_definition,
        bdf_order=bdf_order,
        delta_formula=delta_formula,
        generating_formula=(
            "h^(-q)*delta(exp(-lambda*h)*z)^q; "
            "weights_k=omega_k*exp(-lambda*k*h)"
        ),
        backend=selected_backend,
        time_complexity=time_complexity,
        working_memory=working_memory,
        step=normalized_step,
        lower_terminal=terminal,
        initial_condition_semantics=str(initial_condition_semantics).strip().lower(),
        conjugation=(
            "exp(-lambda*(t-a))*D^q[exp(lambda*(.-a))*x]"
        ),
        tempering_convention="unnormalized_exponential_conjugation",
        normalization_correction=(
            "none_no_minus_lambda_power_q_times_x; "
            "discrete_symbol_is_not_(delta(z)/h+lambda)^q"
        ),
        caputo_initial_correction=caputo_correction,
        startup_convention="terminal_truncated_history_no_prehistory_extrapolation",
        starting_corrections="none_implemented",
        positive_exponential_materialized=False,
        damping_underflowed=damping_underflowed,
        references=_REFERENCES,
    )


__all__ = [
    "TEMPERED_CAPUTO_INITIAL_CONDITION",
    "TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION",
    "TemperedConvolutionQuadratureResult",
    "tempered_convolution_quadrature",
]
