"""Sampled fractional operators beyond the Caputo-only case.

Stability: experimental

This module deliberately separates *operator evaluation* from fractional
initial-value problems.  The Riemann--Liouville (RL), tempered RL, and
variable-order functions below consume a history sampled from an explicit
lower terminal.  They do not interpret ``f(a)`` as a classical RL initial
condition.  Callers must acknowledge that contract with
``initial_condition_semantics=OPERATOR_ONLY_INITIAL_CONDITION``.

The hot history loops have equivalent Numba and pure-Python implementations.
``backend="auto"`` currently selects Numba; ``backend="python"`` remains a
transparent reference path for cross-checks and environments where compiling
small inputs would cost more than it saves.

References
----------
I. Podlubny, *Fractional Differential Equations*, Academic Press, 1999,
ISBN 978-0-12-558840-9.
C. Lubich, "Discretized Fractional Calculus", SIAM Journal on Mathematical
Analysis 17 (1986), https://doi.org/10.1137/0517050.
F. Sabzikar, M. M. Meerschaert, and J. Chen, "Tempered fractional
calculus", Journal of Computational Physics 293 (2015),
https://doi.org/10.1016/j.jcp.2014.04.024.
S. G. Samko and B. Ross, "Integration and differentiation to a variable
fractional order", Integral Transforms and Special Functions 1 (1993),
https://doi.org/10.1080/10652469308819027.
R. Khalil, M. Al Horani, A. Yousef, and M. Sababheh, "A new definition of
fractional derivative", Journal of Computational and Applied Mathematics
264 (2014), https://doi.org/10.1016/j.cam.2014.01.002.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numba import njit, prange

from ._validation import sample_matrix as _sample_matrix
from .contracts import normalize_fractional_orders


OPERATOR_ONLY_INITIAL_CONDITION = "operator_only_no_ivp"
"""Acknowledgement token for sampled non-Caputo operator evaluation."""

_BACKENDS = frozenset({"auto", "numba", "python"})


@dataclass(frozen=True, slots=True)
class SampledFractionalDerivativeResult:
    """Structured result carrying the definition and boundary convention."""

    values: np.ndarray
    times: np.ndarray
    orders: np.ndarray
    derivative: str
    discretization: str
    lower_terminal: float
    initial_condition_semantics: str
    convention: str
    backend: str
    step: float | None
    parameters: dict[str, Any] = field(default_factory=dict)
    status: str = "finite_numerical_diagnostic"


def _select_backend(backend: str) -> str:
    selected = str(backend).strip().lower()
    if selected not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}.")
    return "numba" if selected == "auto" else selected


def _validate_operator_initial_condition(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized != OPERATOR_ONLY_INITIAL_CONDITION:
        raise ValueError(
            "Sampled RL/GL evaluation is not an IVP solver. Pass "
            "initial_condition_semantics='operator_only_no_ivp'; a classical "
            "RL IVP requires fractional-integral initial data and a solver "
            "contract of its own."
        )
    return normalized


def _uniform_grid(
    times: np.ndarray,
    n_times: int,
    lower_terminal: float,
) -> tuple[np.ndarray, float, float]:
    grid = np.asarray(times, dtype=np.float64).reshape(-1)
    if grid.size != n_times:
        raise ValueError("times length must match the number of sample rows.")
    if grid.size < 2:
        raise ValueError("At least two uniformly spaced samples are required.")
    if not np.all(np.isfinite(grid)):
        raise ValueError("times must contain only finite values.")
    terminal = float(lower_terminal)
    if not np.isfinite(terminal):
        raise ValueError("lower_terminal must be finite.")
    scale = max(1.0, abs(terminal), abs(float(grid[0])))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if not np.isclose(grid[0], terminal, rtol=0.0, atol=tolerance):
        raise ValueError(
            "times[0] must equal lower_terminal; pre-terminal history is not "
            "implicitly reconstructed."
        )
    differences = np.diff(grid)
    step = float(differences[0])
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("times must be strictly increasing.")
    grid_scale = max(1.0, abs(step), float(np.max(np.abs(grid))))
    if not np.allclose(
        differences,
        step,
        rtol=2.0e-12,
        atol=64.0 * np.finfo(np.float64).eps * grid_scale,
    ):
        raise ValueError("RL/GL sampled operators require a uniform time grid.")
    return np.ascontiguousarray(grid), step, terminal


@njit(cache=True, nogil=True, parallel=True)
def _constant_order_gl_numba(
    samples: np.ndarray,
    step: float,
    orders: np.ndarray,
    tempering: float,
) -> np.ndarray:
    n_times, dimension = samples.shape
    output = np.empty_like(samples)
    decay_step = np.exp(-tempering * step)
    for component in prange(dimension):
        weights = np.empty(n_times, dtype=np.float64)
        weights[0] = 1.0
        decay = 1.0
        binomial_weight = 1.0
        order = orders[component]
        for lag in range(1, n_times):
            binomial_weight *= 1.0 - (order + 1.0) / lag
            decay *= decay_step
            weights[lag] = binomial_weight * decay
        scale = step ** (-order)
        for n in range(n_times):
            total = 0.0
            for lag in range(n + 1):
                total += weights[lag] * samples[n - lag, component]
            output[n, component] = scale * total
    return output


def _constant_order_gl_python(
    samples: np.ndarray,
    step: float,
    orders: np.ndarray,
    tempering: float,
) -> np.ndarray:
    n_times, dimension = samples.shape
    output = np.empty_like(samples)
    decay_step = float(np.exp(-tempering * step))
    for component in range(dimension):
        order = float(orders[component])
        weights = np.empty(n_times, dtype=np.float64)
        weights[0] = 1.0
        decay = 1.0
        binomial_weight = 1.0
        for lag in range(1, n_times):
            binomial_weight *= 1.0 - (order + 1.0) / lag
            decay *= decay_step
            weights[lag] = binomial_weight * decay
        scale = step ** (-order)
        for n in range(n_times):
            total = 0.0
            for lag in range(n + 1):
                total += weights[lag] * samples[n - lag, component]
            output[n, component] = scale * total
    return output


def _constant_order_gl(
    samples: np.ndarray,
    step: float,
    orders: np.ndarray,
    tempering: float,
    backend: str,
) -> np.ndarray:
    if backend == "numba":
        return _constant_order_gl_numba(samples, step, orders, tempering)
    return _constant_order_gl_python(samples, step, orders, tempering)


def riemann_liouville_gl_derivative(
    samples: np.ndarray,
    times: np.ndarray,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    *,
    lower_terminal: float,
    initial_condition_semantics: str,
    backend: str = "auto",
) -> SampledFractionalDerivativeResult:
    r"""Approximate a left RL derivative with the direct GL formula.

    At ``t_n = a + n h`` this computes

    ``h**(-q) * sum((-1)**k * binom(q, k) * f(t_n-k*h), k=0..n)``.

    The sample history starts exactly at ``a=lower_terminal``.  This is an
    operator evaluation, not an RL initial-value solver: for ``0<q<1``, a
    classical RL IVP is normally posed using a fractional-integral datum such
    as ``(I_a**(1-q) x)(a+)``, not merely ``x(a)``.  Podlubny (1999) gives the
    continuous definitions and Lubich (1986), DOI 10.1137/0517050, develops
    the convolution-quadrature foundation of GL-type discretizations.  At
    ``n=0`` the finite-grid formula is ``h**(-q) f(a)``; that mesh-dependent
    value must not be mistaken for a finite pointwise RL derivative at a
    potentially singular terminal.
    """

    values, was_vector = _sample_matrix(samples)
    grid, step, terminal = _uniform_grid(times, values.shape[0], lower_terminal)
    normalized_orders = normalize_fractional_orders(orders, values.shape[1])
    semantics = _validate_operator_initial_condition(initial_condition_semantics)
    selected_backend = _select_backend(backend)
    output = _constant_order_gl(
        values,
        step,
        np.ascontiguousarray(normalized_orders),
        0.0,
        selected_backend,
    )
    if was_vector:
        output = output[:, 0]
    return SampledFractionalDerivativeResult(
        values=output,
        times=grid,
        orders=normalized_orders,
        derivative="riemann_liouville",
        discretization="grunwald_letnikov_direct",
        lower_terminal=terminal,
        initial_condition_semantics=semantics,
        convention="left_sided_terminal_truncated_history",
        backend=selected_backend,
        step=step,
    )


def tempered_grunwald_letnikov_derivative(
    samples: np.ndarray,
    times: np.ndarray,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    *,
    tempering: float,
    lower_terminal: float,
    initial_condition_semantics: str,
    backend: str = "auto",
) -> SampledFractionalDerivativeResult:
    r"""Approximate the left exponentially tempered RL derivative.

    The definition fixed here is

    ``D_a**(q,lambda) f(t) = exp(-lambda*(t-a)) *``
    ``D_a**q(exp(lambda*(.-a))*f(.))(t)``,

    discretized by the weights
    ``h**(-q) (-1)**k binom(q,k) exp(-lambda*k*h)``.  It is the uncorrected
    tempered RL operator of Sabzikar, Meerschaert, and Chen (2015), DOI
    10.1016/j.jcp.2014.04.024.  It is not a tempered Caputo derivative and no
    optional ``-lambda**q f`` correction is silently applied.  ``lambda=0``
    reduces exactly to :func:`riemann_liouville_gl_derivative`.
    """

    values, was_vector = _sample_matrix(samples)
    grid, step, terminal = _uniform_grid(times, values.shape[0], lower_terminal)
    normalized_orders = normalize_fractional_orders(orders, values.shape[1])
    lambda_value = float(tempering)
    if not np.isfinite(lambda_value) or lambda_value < 0.0:
        raise ValueError("tempering must be a finite non-negative number.")
    semantics = _validate_operator_initial_condition(initial_condition_semantics)
    selected_backend = _select_backend(backend)
    output = _constant_order_gl(
        values,
        step,
        np.ascontiguousarray(normalized_orders),
        lambda_value,
        selected_backend,
    )
    if was_vector:
        output = output[:, 0]
    return SampledFractionalDerivativeResult(
        values=output,
        times=grid,
        orders=normalized_orders,
        derivative="tempered_riemann_liouville",
        discretization="tempered_grunwald_letnikov_direct",
        lower_terminal=terminal,
        initial_condition_semantics=semantics,
        convention="exponential_conjugation_uncorrected",
        backend=selected_backend,
        step=step,
        parameters={"tempering": lambda_value},
    )


def _variable_orders(
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    n_times: int,
    dimension: int,
) -> np.ndarray:
    array = np.asarray(orders, dtype=np.float64)
    if array.ndim == 0:
        normalized = np.full((n_times, dimension), float(array), dtype=np.float64)
    elif array.ndim == 1 and array.size == n_times:
        normalized = np.repeat(array[:, None], dimension, axis=1)
    elif array.ndim == 2 and array.shape == (n_times, dimension):
        normalized = np.array(array, dtype=np.float64, copy=True)
    else:
        raise ValueError(
            "Variable orders must be scalar, shape (n_times,), or shape "
            "(n_times, dimension)."
        )
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Variable orders must contain only finite values.")
    if np.any(normalized <= 0.0) or np.any(normalized > 1.0):
        raise ValueError("Every variable order must lie in (0, 1].")
    return np.ascontiguousarray(normalized)


@njit(cache=True, nogil=True, parallel=True)
def _variable_order_gl_numba(
    samples: np.ndarray,
    step: float,
    orders: np.ndarray,
) -> np.ndarray:
    n_times, dimension = samples.shape
    output = np.empty_like(samples)
    for component in prange(dimension):
        for n in range(n_times):
            order = orders[n, component]
            weight = 1.0
            total = samples[n, component]
            for lag in range(1, n + 1):
                weight *= 1.0 - (order + 1.0) / lag
                total += weight * samples[n - lag, component]
            output[n, component] = step ** (-order) * total
    return output


def _variable_order_gl_python(
    samples: np.ndarray,
    step: float,
    orders: np.ndarray,
) -> np.ndarray:
    n_times, dimension = samples.shape
    output = np.empty_like(samples)
    for component in range(dimension):
        for n in range(n_times):
            order = float(orders[n, component])
            weight = 1.0
            total = float(samples[n, component])
            for lag in range(1, n + 1):
                weight *= 1.0 - (order + 1.0) / lag
                total += weight * samples[n - lag, component]
            output[n, component] = step ** (-order) * total
    return output


def variable_order_grunwald_letnikov_derivative(
    samples: np.ndarray,
    times: np.ndarray,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    *,
    lower_terminal: float,
    initial_condition_semantics: str,
    backend: str = "auto",
) -> SampledFractionalDerivativeResult:
    r"""Evaluate one explicit variable-order GL convention.

    At each output time ``t_n``, this implementation *freezes* the order at
    ``q_n=q(t_n)`` and uses that same value in every binomial coefficient and
    in ``h**(-q_n)``.  Thus

    ``D f(t_n) ~= h**(-q_n) sum((-1)**k binom(q_n,k) f(t_n-k*h))``.

    Variable-order fractional derivatives are not unique.  In particular,
    this convention does not add terms involving derivatives of ``q(t)`` and
    must not be relabelled as another type-I/type-II convention without a new
    derivation.  The broader theory begins with Samko and Ross (1993), DOI
    10.1080/10652469308819027.  The direct implementation costs
    ``O(n_times**2 * dimension)`` because weights cannot be reused when the
    order changes at every evaluation time.
    """

    values, was_vector = _sample_matrix(samples)
    grid, step, terminal = _uniform_grid(times, values.shape[0], lower_terminal)
    normalized_orders = _variable_orders(orders, *values.shape)
    semantics = _validate_operator_initial_condition(initial_condition_semantics)
    selected_backend = _select_backend(backend)
    if selected_backend == "numba":
        output = _variable_order_gl_numba(values, step, normalized_orders)
    else:
        output = _variable_order_gl_python(values, step, normalized_orders)
    if was_vector:
        output = output[:, 0]
    return SampledFractionalDerivativeResult(
        values=output,
        times=grid,
        orders=normalized_orders,
        derivative="variable_order_grunwald_letnikov",
        discretization="evaluation_time_frozen_order_direct_gl",
        lower_terminal=terminal,
        initial_condition_semantics=semantics,
        convention="order_q_at_output_time_used_for_all_history_weights",
        backend=selected_backend,
        step=step,
    )


def _strictly_increasing_grid(
    times: np.ndarray,
    lower_terminal: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    grid = np.asarray(times, dtype=np.float64).reshape(-1)
    if grid.size < 1 or not np.all(np.isfinite(grid)):
        raise ValueError("times must contain at least one finite value.")
    if grid.size > 1 and np.any(np.diff(grid) <= 0.0):
        raise ValueError("times must be strictly increasing.")
    terminal = float(lower_terminal)
    if not np.isfinite(terminal):
        raise ValueError("lower_terminal must be finite.")
    scale = max(1.0, abs(terminal), float(np.max(np.abs(grid))))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    offsets = grid - terminal
    if np.any(offsets < -tolerance):
        raise ValueError("Conformable evaluation times cannot precede lower_terminal.")
    offsets[np.abs(offsets) <= tolerance] = 0.0
    return np.ascontiguousarray(grid), np.ascontiguousarray(offsets), terminal


def _ordinary_derivative_samples(
    times: np.ndarray,
    ordinary_derivative: np.ndarray | Callable[[float], Any],
) -> tuple[np.ndarray, bool]:
    if callable(ordinary_derivative):
        rows: list[np.ndarray] = []
        for time in times:
            row = np.asarray(ordinary_derivative(float(time)), dtype=np.float64)
            rows.append(row.reshape(-1))
        dimension = rows[0].size
        if dimension < 1 or any(row.size != dimension for row in rows):
            raise ValueError(
                "ordinary_derivative callable must return a fixed non-empty shape."
            )
        values = np.vstack(rows)
        was_vector = dimension == 1
    else:
        values = np.asarray(ordinary_derivative, dtype=np.float64)
        was_vector = values.ndim == 1
        if was_vector:
            values = values[:, None]
        if values.ndim != 2 or min(values.shape) < 1:
            raise ValueError(
                "ordinary_derivative samples must have shape (n_times,) or "
                "(n_times, dimension)."
            )
    if values.shape[0] != times.size:
        raise ValueError(
            "ordinary_derivative sample rows must match the number of times."
        )
    return np.ascontiguousarray(values), was_vector


def _terminal_values(
    terminal_value: float | list[float] | tuple[float, ...] | np.ndarray | None,
    dimension: int,
) -> np.ndarray:
    if terminal_value is None:
        raise ValueError("terminal_value is required when terminal_policy='provided'.")
    array = np.asarray(terminal_value, dtype=np.float64).reshape(-1)
    if array.size == 1:
        array = np.full(dimension, float(array[0]), dtype=np.float64)
    if array.size != dimension or not np.all(np.isfinite(array)):
        raise ValueError("terminal_value must be finite and scalar or componentwise.")
    return np.ascontiguousarray(array)


@njit(cache=True, nogil=True, parallel=True)
def _conformable_numba(
    ordinary_derivative: np.ndarray,
    offsets: np.ndarray,
    orders: np.ndarray,
    terminal_values: np.ndarray,
) -> np.ndarray:
    n_times, dimension = ordinary_derivative.shape
    output = np.empty_like(ordinary_derivative)
    for component in prange(dimension):
        order = orders[component]
        for n in range(n_times):
            if offsets[n] == 0.0 and order < 1.0:
                output[n, component] = terminal_values[component]
            else:
                output[n, component] = (
                    offsets[n] ** (1.0 - order)
                    * ordinary_derivative[n, component]
                )
    return output


def _conformable_python(
    ordinary_derivative: np.ndarray,
    offsets: np.ndarray,
    orders: np.ndarray,
    terminal_values: np.ndarray,
) -> np.ndarray:
    n_times, dimension = ordinary_derivative.shape
    output = np.empty_like(ordinary_derivative)
    for component in range(dimension):
        order = float(orders[component])
        for n in range(n_times):
            if offsets[n] == 0.0 and order < 1.0:
                output[n, component] = terminal_values[component]
            else:
                output[n, component] = (
                    offsets[n] ** (1.0 - order)
                    * ordinary_derivative[n, component]
                )
    return output


def conformable_khalil_derivative(
    times: np.ndarray,
    ordinary_derivative: np.ndarray | Callable[[float], Any],
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    *,
    lower_terminal: float = 0.0,
    terminal_policy: str = "raise",
    terminal_value: float | list[float] | tuple[float, ...] | np.ndarray | None = None,
    backend: str = "auto",
) -> SampledFractionalDerivativeResult:
    r"""Apply the local conformable derivative to known ordinary derivatives.

    For differentiable ``f`` and ``t>a`` this evaluates the Khalil identity
    ``T_q f(t) = (t-a)**(1-q) f'(t)``.  ``a=0`` is the original convention;
    another ``lower_terminal`` requests the explicitly labelled shifted
    extension.  ``ordinary_derivative`` may be sampled ``f'``/RHS values or a
    callable of one scalar time returning a scalar or vector.

    At ``t=a`` and ``q<1`` the algebraic expression alone does not determine
    the conformable right limit when ``f'`` can be singular.  Choose one:

    - ``terminal_policy="raise"`` (default) refuses to guess;
    - ``"bounded_derivative_zero"`` sets zero and requires finite ``f'(a)``;
    - ``"provided"`` uses an explicit finite ``terminal_value``.

    For ``q=1``, the ordinary derivative is returned including at the terminal.
    This is a local reparametrization, not a nonlocal memory operator.  See
    Khalil et al. (2014), DOI 10.1016/j.cam.2014.01.002.
    """

    grid, offsets, terminal = _strictly_increasing_grid(times, lower_terminal)
    derivatives, was_vector = _ordinary_derivative_samples(grid, ordinary_derivative)
    normalized_orders = normalize_fractional_orders(orders, derivatives.shape[1])
    policy = str(terminal_policy).strip().lower()
    allowed_policies = {"raise", "bounded_derivative_zero", "provided"}
    if policy not in allowed_policies:
        raise ValueError(f"terminal_policy must be one of {sorted(allowed_policies)}.")

    terminal_mask = offsets == 0.0
    fractional_components = normalized_orders < 1.0
    needs_terminal_contract = bool(np.any(terminal_mask) and np.any(fractional_components))
    if needs_terminal_contract and policy == "raise":
        raise ValueError(
            "The conformable derivative at the lower terminal needs an explicit "
            "terminal_policy for q < 1."
        )
    if needs_terminal_contract and policy == "provided":
        terminal_values = _terminal_values(terminal_value, derivatives.shape[1])
    else:
        terminal_values = np.zeros(derivatives.shape[1], dtype=np.float64)

    nonterminal_mask = ~terminal_mask
    if np.any(nonterminal_mask) and not np.all(np.isfinite(derivatives[nonterminal_mask])):
        raise ValueError("ordinary_derivative must be finite away from the terminal.")
    if np.any(terminal_mask):
        terminal_rows = derivatives[terminal_mask]
        for component, order in enumerate(normalized_orders):
            if order == 1.0 or policy == "bounded_derivative_zero":
                if not np.all(np.isfinite(terminal_rows[:, component])):
                    raise ValueError(
                        "ordinary_derivative at the terminal must be finite for "
                        "q=1 and for terminal_policy='bounded_derivative_zero'."
                    )
    elif not np.all(np.isfinite(derivatives)):
        raise ValueError("ordinary_derivative must contain only finite values.")

    selected_backend = _select_backend(backend)
    if selected_backend == "numba":
        output = _conformable_numba(
            derivatives,
            offsets,
            np.ascontiguousarray(normalized_orders),
            terminal_values,
        )
    else:
        output = _conformable_python(
            derivatives,
            offsets,
            normalized_orders,
            terminal_values,
        )
    if was_vector:
        output = output[:, 0]
    return SampledFractionalDerivativeResult(
        values=output,
        times=grid,
        orders=normalized_orders,
        derivative="conformable_khalil",
        discretization="exact_scaling_of_supplied_ordinary_derivative",
        lower_terminal=terminal,
        initial_condition_semantics="local_operator_no_memory_no_ivp",
        convention="khalil_original" if terminal == 0.0 else "shifted_khalil",
        backend=selected_backend,
        step=None,
        parameters={"terminal_policy": policy},
    )


__all__ = [
    "OPERATOR_ONLY_INITIAL_CONDITION",
    "SampledFractionalDerivativeResult",
    "conformable_khalil_derivative",
    "riemann_liouville_gl_derivative",
    "tempered_grunwald_letnikov_derivative",
    "variable_order_grunwald_letnikov_derivative",
]
