r"""Fast recurrent history for tempered fractional multistep operators.

Stability: experimental

This module implements the real-axis ``Fast Method II`` construction of
Guo, Zeng, Turner, Burrage, and Karniadakis for a *sampled operator*.  It is
not an FDE solver.  A short local window is evaluated with exact multistep
weights; older samples are compressed into real recurrent states.

For ``0 < q < 1`` HAFO supports two generating functions here:

``fbdf1``
    ``Omega(z) = (1-z)**q``.

``gngf2``
    ``Omega(z) = (1-z)**q * (1 + q*(1-z)/2)``.

The second formula is the second-order generalized Newton--Gregory formula
used in the numerical examples of the primary source.  It must not be
silently renamed BDF2: fractional BDF2 makes ``F_omega(-lambda)`` cross a
fractional power of a negative base, so a real-only implementation requires
an additional branch analysis.  At ``q=1``, however, GNGF2 reduces exactly to
the ordinary BDF2 polynomial.

Writing ``r = h*lambda`` makes the quadrature dimensionless.  For a lag
``ell`` the untempered coefficient is approximated by

.. math::

   \omega_\ell \approx \sum_j a_j(1+r_j)^{-\ell-1}.

The tempered recurrence is

.. math::

   y_m^{(j)} = \frac{e^{-\sigma h}}{1+r_j}
       \left(y_{m-1}^{(j)} + u_{m-1}\right),

and therefore uses no growing exponential.  HAFO checks the compressed
weights against the exact finite-grid recurrence at *every* nonlocal lag and
increases the number of quadrature nodes until the requested relative
``L1`` weight tolerance is met.  This is an a posteriori finite-grid check,
not the analytic strip bound of the infinite trapezoidal rule and not an FDE
discretization-error estimate.

The conjugated tempered-Caputo convention is identical to
``tempered_convolution_quadrature``.  Its initial anchor is subtracted with
the exact partial sum of the multistep weights, so compression affects only
the raw history and never approximates the anchor itself.

References
----------
L. Guo, F. Zeng, I. Turner, K. Burrage, and G. E. Karniadakis, "Efficient
Multistep Methods for Tempered Fractional Calculus: Algorithms and
Simulations", SIAM Journal on Scientific Computing 41 (2019),
https://doi.org/10.1137/18M1230153.
C. Lubich, "Discretized Fractional Calculus", SIAM Journal on Mathematical
Analysis 17 (1986), https://doi.org/10.1137/0517050.
L. N. Trefethen and J. A. C. Weideman, "The Exponentially Convergent
Trapezoidal Rule", SIAM Review 56 (2014),
https://doi.org/10.1137/130932132.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as operator_index
from typing import Any

import numpy as np
from numba import njit

from .convolution_quadrature import (
    _sample_matrix,
    _time_grid,
)
from .tempered_convolution_quadrature import (
    TEMPERED_CAPUTO_INITIAL_CONDITION,
    TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
    _definition_contract,
    _normalize_orders,
    _normalize_tempering,
    _strict_real_array,
    _strict_real_scalar,
)


TEMPERED_FAST_HISTORY_REFERENCES = (
    "https://doi.org/10.1137/18M1230153",
    "https://doi.org/10.1137/0517050",
    "https://doi.org/10.1137/130932132",
)

_METHODS = frozenset({"fbdf1", "gngf2"})
_BACKENDS = frozenset({"python", "numba"})
_METHOD_CODES = {"fbdf1": 1, "gngf2": 2}


@dataclass(frozen=True, slots=True)
class TemperedFastHistoryResult:
    """Structured evaluation of a recurrent tempered sampled operator.

    ``quadrature_nodes`` stores the dimensionless nodes ``r_j=h*lambda_j``;
    ``quadrature_weights`` stores the corresponding signed approximation
    weights ``a_j``.  Their shape is ``(quadrature_points, dimension)``.
    ``final_history_state`` has the same shape and is sufficient to audit the
    active memory of the completed batch evaluation.

    ``l1_relative_weight_error`` compares every compressed untempered weight
    on this finite grid with the exact generating-function recurrence.  The
    comparison combines trapezoidal and tail truncation errors; neither part
    is advertised as a separate certified analytic bound.
    """

    values: np.ndarray
    times: np.ndarray
    orders: np.ndarray
    tempering: np.ndarray
    local_base_weights: np.ndarray
    local_tempered_weights: np.ndarray
    quadrature_nodes: np.ndarray
    quadrature_weights: np.ndarray
    final_history_state: np.ndarray
    definition: str
    multistep_method: str
    generating_formula: str
    formal_order: int
    backend: str
    step: float
    lower_terminal: float
    local_history_steps: int
    quadrature_points: int
    tail_cutoff: float
    requested_relative_tolerance: float
    l1_absolute_weight_error: np.ndarray
    l1_relative_weight_error: np.ndarray
    max_absolute_weight_error: np.ndarray
    max_relative_weight_error: np.ndarray
    operator_absolute_error_bound: np.ndarray
    compression_tolerance_satisfied: bool
    calibration_kind: str
    trapezoidal_error_bound: str
    tail_error_bound: str
    time_complexity: str
    active_working_memory: str
    output_memory: str
    initial_condition_semantics: str
    caputo_initial_correction: str
    startup_convention: str
    starting_corrections: str
    conjugation: str
    positive_exponential_materialized: bool
    references: tuple[str, ...]
    scope: str = "sampled_fractional_operator_only_not_an_fde_solver"
    status: str = "finite_numerical_diagnostic"


def _integer_parameter(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not Boolean.")
    try:
        normalized = operator_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    normalized = int(normalized)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return normalized


def _open_unit_scalar(value: Any, *, name: str) -> float:
    normalized = _strict_real_scalar(value, name=name)
    if not 0.0 < normalized < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one.")
    return normalized


def _method_contract(method: str) -> tuple[str, int, str]:
    normalized = str(method).strip().lower()
    if normalized not in _METHODS:
        raise ValueError(f"multistep_method must be one of {sorted(_METHODS)}.")
    if normalized == "fbdf1":
        return normalized, 1, "Omega(z)=(1-z)^q"
    return normalized, 2, "Omega(z)=(1-z)^q*(1+q*(1-z)/2)"


def _base_weights(order: float, count: int, method_code: int) -> np.ndarray:
    """Return exact local FBDF1 or GNGF2 weights."""

    weights = np.empty(count, dtype=np.float64)
    if count == 0:
        return weights
    gl_previous = 1.0
    if method_code == 1:
        weights[0] = 1.0
    else:
        weights[0] = 1.0 + 0.5 * order
    for lag in range(1, count):
        gl_current = ((lag - 1.0 - order) / lag) * gl_previous
        if method_code == 1:
            weights[lag] = gl_current
        else:
            weights[lag] = (
                (1.0 + 0.5 * order) * gl_current
                - 0.5 * order * gl_previous
            )
        gl_previous = gl_current
    return weights


def _local_component_weights(
    orders: np.ndarray,
    tempering: np.ndarray,
    count: int,
    step: float,
    method_code: int,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.empty((count, orders.size), dtype=np.float64)
    tempered = np.empty_like(base)
    lags = np.arange(count, dtype=np.float64)
    for component, order in enumerate(orders):
        base[:, component] = _base_weights(float(order), count, method_code)
        damping = np.exp(-float(tempering[component]) * step * lags)
        tempered[:, component] = (
            np.power(step, -float(order))
            * damping
            * base[:, component]
        )
    if not np.all(np.isfinite(tempered)):
        raise ValueError("local tempered weights overflowed or became non-finite.")
    return np.ascontiguousarray(base), np.ascontiguousarray(tempered)


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, values)


def _log_abs_one_minus_exp(values: np.ndarray) -> np.ndarray:
    """Stable ``log(abs(1-exp(values)))`` including its exact zero."""

    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    negative = values < 0.0
    positive = values > 0.0
    zero = ~(negative | positive)
    result[negative] = np.log(-np.expm1(values[negative]))
    result[positive] = (
        values[positive] + np.log1p(-np.exp(-values[positive]))
    )
    result[zero] = -np.inf
    return result


def _log_abs_phi_n(
    log_nodes: np.ndarray,
    order: float,
    lag: int,
    method_code: int,
) -> np.ndarray:
    log_coefficient = np.log(abs(-np.sin(np.pi * order) / np.pi))
    result = (
        log_coefficient
        + (1.0 + order) * log_nodes
        - (lag + 1.0) * _softplus(log_nodes)
    )
    if method_code == 2:
        crossing = log_nodes + np.log(0.5 * order)
        result = result + _log_abs_one_minus_exp(crossing)
    return result


def _significant_interval_for_lag(
    order: float,
    lag: int,
    method_code: int,
    tail_cutoff: float,
) -> tuple[float, float]:
    """Locate the outer level-set crossings of ``abs(phi_lag)``."""

    lower = -64.0
    upper = 64.0
    log_cutoff = np.log(tail_cutoff)
    grid_size = 16385
    for _ in range(5):
        grid = np.linspace(lower, upper, grid_size, dtype=np.float64)
        log_values = _log_abs_phi_n(grid, order, lag, method_code)
        maximum = float(np.max(log_values))
        threshold = maximum + log_cutoff
        significant = np.flatnonzero(log_values >= threshold)
        if significant.size == 0:
            raise RuntimeError("could not locate a significant quadrature interval.")
        first = int(significant[0])
        last = int(significant[-1])
        if first > 0 and last < grid_size - 1:
            break
        lower *= 2.0
        upper *= 2.0
    else:
        raise RuntimeError(
            "quadrature interval exceeds the stable real-axis search range; "
            "increase local_history_steps or relax tail_cutoff."
        )

    def scalar_difference(point: float) -> float:
        value = _log_abs_phi_n(
            np.array([point]), order, lag, method_code
        )[0]
        return float(value - threshold)

    left_low = float(grid[first - 1])
    left_high = float(grid[first])
    for _ in range(60):
        midpoint = 0.5 * (left_low + left_high)
        if scalar_difference(midpoint) >= 0.0:
            left_high = midpoint
        else:
            left_low = midpoint

    right_low = float(grid[last])
    right_high = float(grid[last + 1])
    for _ in range(60):
        midpoint = 0.5 * (right_low + right_high)
        if scalar_difference(midpoint) >= 0.0:
            right_low = midpoint
        else:
            right_high = midpoint
    return left_high, right_low


def _quadrature_interval(
    order: float,
    first_compressed_lag: int,
    final_lag: int,
    method_code: int,
    tail_cutoff: float,
) -> tuple[float, float]:
    intervals = [
        _significant_interval_for_lag(
            order, first_compressed_lag, method_code, tail_cutoff
        )
    ]
    if final_lag != first_compressed_lag:
        intervals.append(
            _significant_interval_for_lag(
                order, final_lag, method_code, tail_cutoff
            )
        )
    return (
        min(interval[0] for interval in intervals),
        max(interval[1] for interval in intervals),
    )


def _quadrature_for_component(
    order: float,
    first_compressed_lag: int,
    final_lag: int,
    method_code: int,
    tail_cutoff: float,
    quadrature_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = _quadrature_interval(
        order,
        first_compressed_lag,
        final_lag,
        method_code,
        tail_cutoff,
    )
    log_nodes = np.linspace(lower, upper, quadrature_points, dtype=np.float64)
    spacing = (upper - lower) / (quadrature_points - 1)
    nodes = np.exp(log_nodes)
    coefficient = -np.sin(np.pi * order) / np.pi
    log_magnitude = (
        np.log(abs(coefficient)) + (1.0 + order) * log_nodes
    )
    signs = np.full(quadrature_points, np.sign(coefficient), dtype=np.float64)
    if method_code == 2:
        crossing = log_nodes + np.log(0.5 * order)
        log_magnitude = log_magnitude + _log_abs_one_minus_exp(crossing)
        signs *= np.where(crossing < 0.0, 1.0, -1.0)
        signs[crossing == 0.0] = 0.0
    weights = spacing * signs * np.exp(log_magnitude)
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(weights)):
        raise RuntimeError(
            "quadrature nodes or weights are not representable in float64; "
            "increase local_history_steps or relax tail_cutoff."
        )
    return np.ascontiguousarray(nodes), np.ascontiguousarray(weights)


def _calibrate_component(
    order: float,
    local_history_steps: int,
    final_lag: int,
    method_code: int,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, float]:
    if final_lag <= local_history_steps or order == 1.0:
        return 0.0, 0.0, 0.0, 0.0

    first_lag = local_history_steps + 1
    inverse_factors = 1.0 / (1.0 + nodes)
    powers = np.power(inverse_factors, first_lag + 1.0)

    gl_previous = 1.0
    exact_weight = 1.0 if method_code == 1 else 1.0 + 0.5 * order
    l1_exact = 0.0
    l1_error = 0.0
    maximum_absolute = 0.0
    maximum_relative = 0.0
    reference_scale = 0.0

    for lag in range(1, final_lag + 1):
        gl_current = ((lag - 1.0 - order) / lag) * gl_previous
        if method_code == 1:
            exact_weight = gl_current
        else:
            exact_weight = (
                (1.0 + 0.5 * order) * gl_current
                - 0.5 * order * gl_previous
            )
        gl_previous = gl_current
        if lag < first_lag:
            continue
        approximation = float(np.dot(weights, powers))
        error = abs(approximation - exact_weight)
        absolute_exact = abs(exact_weight)
        l1_exact += absolute_exact
        l1_error += error
        maximum_absolute = max(maximum_absolute, error)
        reference_scale = max(reference_scale, absolute_exact)
        if absolute_exact > 64.0 * np.finfo(np.float64).eps * max(
            reference_scale, np.finfo(np.float64).tiny
        ):
            maximum_relative = max(
                maximum_relative, error / absolute_exact
            )
        powers *= inverse_factors

    relative_l1 = l1_error / max(l1_exact, np.finfo(np.float64).tiny)
    return l1_error, relative_l1, maximum_absolute, maximum_relative


def _build_and_calibrate_quadrature(
    orders: np.ndarray,
    local_history_steps: int,
    final_lag: int,
    method_code: int,
    tail_cutoff: float,
    requested_points: int | None,
    maximum_points: int,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active = orders < 1.0
    if final_lag <= local_history_steps or not np.any(active):
        empty = np.empty((0, orders.size), dtype=np.float64)
        zeros = np.zeros(orders.size, dtype=np.float64)
        return empty, empty.copy(), 0, zeros, zeros.copy(), zeros.copy(), zeros.copy()

    if requested_points is None and maximum_points < 65:
        raise ValueError(
            "max_quadrature_points must be at least 65 when automatic "
            "quadrature refinement is required."
        )

    points = requested_points if requested_points is not None else 65
    while True:
        nodes = np.zeros((points, orders.size), dtype=np.float64)
        weights = np.zeros_like(nodes)
        l1_absolute = np.zeros(orders.size, dtype=np.float64)
        l1_relative = np.zeros(orders.size, dtype=np.float64)
        max_absolute = np.zeros(orders.size, dtype=np.float64)
        max_relative = np.zeros(orders.size, dtype=np.float64)
        for component, order in enumerate(orders):
            if order == 1.0:
                continue
            component_nodes, component_weights = _quadrature_for_component(
                float(order),
                local_history_steps + 1,
                final_lag,
                method_code,
                tail_cutoff,
                points,
            )
            nodes[:, component] = component_nodes
            weights[:, component] = component_weights
            (
                l1_absolute[component],
                l1_relative[component],
                max_absolute[component],
                max_relative[component],
            ) = _calibrate_component(
                float(order),
                local_history_steps,
                final_lag,
                method_code,
                component_nodes,
                component_weights,
            )

        if np.all(l1_relative <= relative_tolerance):
            return (
                np.ascontiguousarray(nodes),
                np.ascontiguousarray(weights),
                points,
                l1_absolute,
                l1_relative,
                max_absolute,
                max_relative,
            )
        if requested_points is not None:
            raise RuntimeError(
                "quadrature_points does not satisfy relative_tolerance on "
                "the complete finite history; increase quadrature_points or "
                "relax relative_tolerance."
            )
        next_points = 2 * points - 1
        if next_points > maximum_points:
            raise RuntimeError(
                "automatic Fast Method II quadrature did not satisfy the "
                "requested finite-grid tolerance before max_quadrature_points."
            )
        points = next_points


def _history_coefficients(
    nodes: np.ndarray,
    weights: np.ndarray,
    orders: np.ndarray,
    tempering: np.ndarray,
    step: float,
    local_history_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    if nodes.shape[0] == 0:
        return nodes.copy(), nodes.copy()
    dimension = orders.size
    decays = np.empty_like(nodes)
    coefficients = np.empty_like(nodes)
    for component in range(dimension):
        inverse = 1.0 / (1.0 + nodes[:, component])
        if orders[component] == 1.0:
            decays[:, component] = 0.0
            coefficients[:, component] = 0.0
            continue
        damping = np.exp(-tempering[component] * step)
        decays[:, component] = damping * inverse
        coefficients[:, component] = (
            np.power(step, -orders[component])
            * np.exp(-local_history_steps * tempering[component] * step)
            * weights[:, component]
            * np.power(inverse, local_history_steps + 1.0)
        )
    if not np.all(np.isfinite(decays)) or not np.all(np.isfinite(coefficients)):
        raise ValueError("fast-history recurrence coefficients are non-finite.")
    return np.ascontiguousarray(decays), np.ascontiguousarray(coefficients)


def _evaluate_python(
    samples: np.ndarray,
    orders: np.ndarray,
    tempering: np.ndarray,
    local_weights: np.ndarray,
    decays: np.ndarray,
    history_coefficients: np.ndarray,
    step: float,
    local_history_steps: int,
    method_code: int,
    caputo: bool,
) -> tuple[np.ndarray, np.ndarray]:
    count, dimension = samples.shape
    quadrature_points = decays.shape[0]
    output = np.zeros_like(samples)
    state = np.zeros((quadrature_points, dimension), dtype=np.float64)
    for component in range(dimension):
        order = float(orders[component])
        sigma = float(tempering[component])
        gl_previous = 1.0
        partial_sum = 1.0 if method_code == 1 else 1.0 + 0.5 * order
        anchor_damping = 1.0
        anchor_step = np.exp(-sigma * step)
        scale = np.power(step, -order)
        for n in range(count):
            if n > 0:
                gl_current = ((n - 1.0 - order) / n) * gl_previous
                if method_code == 1:
                    current_weight = gl_current
                else:
                    current_weight = (
                        (1.0 + 0.5 * order) * gl_current
                        - 0.5 * order * gl_previous
                    )
                gl_previous = gl_current
                partial_sum += current_weight
                anchor_damping *= anchor_step

            local_stop = min(n, local_history_steps)
            local = 0.0
            for lag in range(local_stop + 1):
                local += local_weights[lag, component] * samples[n - lag, component]

            history = 0.0
            if n > local_history_steps and order < 1.0:
                source = n - local_history_steps - 1
                for point in range(quadrature_points):
                    state[point, component] = decays[point, component] * (
                        state[point, component] + samples[source, component]
                    )
                    history += (
                        history_coefficients[point, component]
                        * state[point, component]
                    )
            value = local + history
            if caputo:
                value -= (
                    scale
                    * samples[0, component]
                    * anchor_damping
                    * partial_sum
                )
                if n == 0:
                    value = 0.0
            output[n, component] = value
    return output, state


@njit(cache=True, nogil=True)
def _evaluate_numba(
    samples: np.ndarray,
    orders: np.ndarray,
    tempering: np.ndarray,
    local_weights: np.ndarray,
    decays: np.ndarray,
    history_coefficients: np.ndarray,
    step: float,
    local_history_steps: int,
    method_code: int,
    caputo: bool,
) -> tuple[np.ndarray, np.ndarray]:
    count, dimension = samples.shape
    quadrature_points = decays.shape[0]
    output = np.zeros_like(samples)
    state = np.zeros((quadrature_points, dimension), dtype=np.float64)
    for component in range(dimension):
        order = orders[component]
        sigma = tempering[component]
        gl_previous = 1.0
        if method_code == 1:
            partial_sum = 1.0
        else:
            partial_sum = 1.0 + 0.5 * order
        anchor_damping = 1.0
        anchor_step = np.exp(-sigma * step)
        scale = step ** (-order)
        for n in range(count):
            if n > 0:
                gl_current = ((n - 1.0 - order) / n) * gl_previous
                if method_code == 1:
                    current_weight = gl_current
                else:
                    current_weight = (
                        (1.0 + 0.5 * order) * gl_current
                        - 0.5 * order * gl_previous
                    )
                gl_previous = gl_current
                partial_sum += current_weight
                anchor_damping *= anchor_step

            local_stop = n
            if local_stop > local_history_steps:
                local_stop = local_history_steps
            local = 0.0
            for lag in range(local_stop + 1):
                local += local_weights[lag, component] * samples[n - lag, component]

            history = 0.0
            if n > local_history_steps and order < 1.0:
                source = n - local_history_steps - 1
                for point in range(quadrature_points):
                    state[point, component] = decays[point, component] * (
                        state[point, component] + samples[source, component]
                    )
                    history += (
                        history_coefficients[point, component]
                        * state[point, component]
                    )
            value = local + history
            if caputo:
                value -= (
                    scale
                    * samples[0, component]
                    * anchor_damping
                    * partial_sum
                )
                if n == 0:
                    value = 0.0
            output[n, component] = value
    return output, state


def tempered_fast_multistep_history(
    samples: np.ndarray,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    *,
    tempering: float | list[float] | tuple[float, ...] | np.ndarray,
    multistep_method: str = "gngf2",
    definition: str = "tempered_riemann_liouville",
    step: float | None = None,
    times: np.ndarray | None = None,
    lower_terminal: float = 0.0,
    initial_condition_semantics: str,
    local_history_steps: int = 50,
    quadrature_points: int | None = None,
    relative_tolerance: float = 1.0e-8,
    tail_cutoff: float = 1.0e-20,
    max_quadrature_points: int = 2049,
    backend: str = "numba",
) -> TemperedFastHistoryResult:
    """Evaluate a tempered RL or conjugated-Caputo history recurrently.

    Parameters
    ----------
    samples, orders, tempering, step, times, lower_terminal:
        Same uniform-grid and componentwise conventions as
        :func:`tempered_convolution_quadrature`.  Orders lie in ``(0, 1]``
        and tempering parameters are finite and non-negative.
    multistep_method:
        ``"fbdf1"`` or ``"gngf2"``.  GNGF2 is a distinct fractional
        second-order generator; it is not fractional BDF2.
    definition, initial_condition_semantics:
        Raw tempered RL requires
        ``TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION``.  Conjugated tempered
        Caputo requires ``TEMPERED_CAPUTO_INITIAL_CONDITION``.
    local_history_steps:
        Largest lag evaluated with exact weights.  It must be at least two;
        the primary paper uses 50.  The active history therefore contains
        ``local_history_steps + 1`` exact coefficients plus recurrent states.
    quadrature_points:
        Explicit number of real-axis nodes, or ``None`` to increase a nested
        grid automatically until ``relative_tolerance`` is met.
    relative_tolerance:
        Required relative ``L1`` error of *all* compressed weights on this
        finite grid against the exact coefficient recurrence.  This controls
        compression only, not CQ discretization or an FDE solution error.
    tail_cutoff:
        Relative level-set cutoff used to truncate the real line.  It is not
        independently a certified error bound.
    max_quadrature_points:
        Safety cap for automatic node refinement.  It must be at least 65
        whenever a compressed history requires the automatic grid.
    backend:
        ``"python"`` or ``"numba"``.  FFT is intentionally absent: it is an
        offline batch convolution, not a recurrent fast-history backend.
    """

    _ = _strict_real_array(samples, name="samples")
    values, was_vector = _sample_matrix(samples)
    normalized_orders = _normalize_orders(orders, values.shape[1])
    normalized_tempering = _normalize_tempering(tempering, values.shape[1])
    normalized_definition, _, _ = _definition_contract(
        definition, initial_condition_semantics
    )
    method, formal_order, generating_formula = _method_contract(multistep_method)
    method_code = _METHOD_CODES[method]

    selected_backend = str(backend).strip().lower()
    if selected_backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}.")

    requested_local_steps = _integer_parameter(
        local_history_steps, name="local_history_steps", minimum=2
    )
    maximum_points = _integer_parameter(
        max_quadrature_points, name="max_quadrature_points", minimum=17
    )
    requested_points: int | None = None
    if quadrature_points is not None:
        requested_points = _integer_parameter(
            quadrature_points, name="quadrature_points", minimum=17
        )
        if requested_points > maximum_points:
            raise ValueError(
                "quadrature_points cannot exceed max_quadrature_points."
            )
    tolerance = _open_unit_scalar(relative_tolerance, name="relative_tolerance")
    cutoff = _open_unit_scalar(tail_cutoff, name="tail_cutoff")

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

    effective_local_steps = min(requested_local_steps, values.shape[0] - 1)
    local_count = effective_local_steps + 1
    local_base, local_tempered = _local_component_weights(
        normalized_orders,
        normalized_tempering,
        local_count,
        normalized_step,
        method_code,
    )
    (
        quadrature_nodes,
        quadrature_weights,
        used_points,
        l1_absolute,
        l1_relative,
        max_absolute,
        max_relative,
    ) = _build_and_calibrate_quadrature(
        normalized_orders,
        effective_local_steps,
        values.shape[0] - 1,
        method_code,
        cutoff,
        requested_points,
        maximum_points,
        tolerance,
    )
    decays, history_coefficients = _history_coefficients(
        quadrature_nodes,
        quadrature_weights,
        normalized_orders,
        normalized_tempering,
        normalized_step,
        effective_local_steps,
    )

    caputo = normalized_definition == "tempered_caputo"
    evaluator = _evaluate_numba if selected_backend == "numba" else _evaluate_python
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        output_matrix, final_state = evaluator(
            values,
            normalized_orders,
            normalized_tempering,
            local_tempered,
            decays,
            history_coefficients,
            normalized_step,
            effective_local_steps,
            method_code,
            caputo,
        )
    if not np.all(np.isfinite(output_matrix)) or not np.all(np.isfinite(final_state)):
        raise ValueError("fast-history evaluation overflowed or became non-finite.")

    sample_maximum = np.max(np.abs(values), axis=0)
    operator_bound = (
        np.power(normalized_step, -normalized_orders)
        * sample_maximum
        * l1_absolute
    )
    output: np.ndarray = output_matrix
    if was_vector:
        output = output_matrix[:, 0]

    return TemperedFastHistoryResult(
        values=output,
        times=grid,
        orders=normalized_orders,
        tempering=normalized_tempering,
        local_base_weights=local_base,
        local_tempered_weights=local_tempered,
        quadrature_nodes=quadrature_nodes,
        quadrature_weights=quadrature_weights,
        final_history_state=final_state,
        definition=normalized_definition,
        multistep_method=method,
        generating_formula=generating_formula,
        formal_order=formal_order,
        backend=selected_backend,
        step=normalized_step,
        lower_terminal=terminal,
        local_history_steps=effective_local_steps,
        quadrature_points=used_points,
        tail_cutoff=cutoff,
        requested_relative_tolerance=tolerance,
        l1_absolute_weight_error=l1_absolute,
        l1_relative_weight_error=l1_relative,
        max_absolute_weight_error=max_absolute,
        max_relative_weight_error=max_relative,
        operator_absolute_error_bound=operator_bound,
        compression_tolerance_satisfied=bool(np.all(l1_relative <= tolerance)),
        calibration_kind=(
            "all_nonlocal_finite_grid_weights_against_exact_recurrence_l1"
        ),
        trapezoidal_error_bound=(
            "not_separately_instantiated_unknown_analytic_strip_constants; "
            "included_in_finite_grid_weight_calibration"
        ),
        tail_error_bound=(
            "not_separately_certified; included_in_finite_grid_weight_calibration"
        ),
        time_complexity="O(d*(Q+n0)*N) recurrent batch evaluation",
        active_working_memory=(
            "O(d*(Q+n0)) history state and exact local weights, excluding "
            "input and returned output"
        ),
        output_memory="O(d*N) returned sampled values",
        initial_condition_semantics=str(initial_condition_semantics).strip().lower(),
        caputo_initial_correction=(
            "exact_-x(a)*exp(-lambda*n*h)*sum_{k=0}^n_omega_k"
            if caputo
            else "not_applicable_raw_tempered_riemann_liouville"
        ),
        startup_convention=(
            "exact_terminal_truncated_history_through_local_window_then_"
            "recurrent_compressed_history"
        ),
        starting_corrections="none_implemented",
        conjugation="exp(-lambda*(t-a))*D^q[exp(lambda*(.-a))*x]",
        positive_exponential_materialized=False,
        references=TEMPERED_FAST_HISTORY_REFERENCES,
    )


__all__ = [
    "TEMPERED_CAPUTO_INITIAL_CONDITION",
    "TEMPERED_FAST_HISTORY_REFERENCES",
    "TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION",
    "TemperedFastHistoryResult",
    "tempered_fast_multistep_history",
]
