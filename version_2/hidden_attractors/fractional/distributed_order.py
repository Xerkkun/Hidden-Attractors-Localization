"""Distributed-order GL operators on uniformly sampled trajectories.

Stability: experimental

This module evaluates the finite order-quadrature approximation

``sum_j omega_j D**q_j x(t_n)``

where each base derivative is approximated independently by the direct
Grunwald--Letnikov (GL) history formula.  It is an operator on supplied
samples, not a distributed-order fractional differential-equation solver.
Two discretizations are therefore present and reported in every result:
quadrature in the order variable and GL convolution on the time grid.

The weight contract is deliberately explicit.  ``*_mass`` means that the
supplied weights are already the effective masses of a discrete measure.
``*_quadrature_density`` means that the supplied quadrature weights are
multiplied by explicit density values at the order nodes.  A ``signed_*``
mode is required before any negative factor is accepted.

The Numba kernel accumulates one order at a time into the output and retains
only one GL-weight vector.  It does not materialize an
``(n_orders, n_times, dimension)`` derivative tensor.

References
----------
M. Caputo, *Elasticita e dissipazione*, Zanichelli, Bologna, 1969.
M. Caputo, "Distributed order differential equations modelling dielectric
induction and diffusion", Fractional Calculus and Applied Analysis 4 (2001),
421--442.  (No DOI is asserted here.)
K. Diethelm and N. J. Ford, "Numerical analysis for distributed-order
differential equations", Journal of Computational and Applied Mathematics
225 (2009), 96--104, https://doi.org/10.1016/j.cam.2008.07.018.
I. Podlubny, *Fractional Differential Equations*, Academic Press, 1999,
ISBN 978-0-12-558840-9.
C. Lubich, "Discretized Fractional Calculus", SIAM Journal on Mathematical
Analysis 17 (1986), 704--719, https://doi.org/10.1137/0517050.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange


_BACKENDS = frozenset({"auto", "numba", "python"})
_DEFINITIONS = frozenset(
    {"grunwald_letnikov", "riemann_liouville_gl", "caputo_shifted"}
)
_NORMALIZATIONS = frozenset({"none", "unit_mass"})
_WEIGHT_SEMANTICS = frozenset(
    {
        "nonnegative_mass",
        "signed_mass",
        "nonnegative_quadrature_density",
        "signed_quadrature_density",
    }
)


@dataclass(frozen=True, slots=True)
class DistributedOrderDerivativeResult:
    """Structured result from a discrete distributed-order operator.

    ``raw_mass`` and ``raw_l1_norm`` describe the effective order weights
    before optional normalization.  ``mass`` and ``l1_norm`` describe the
    weights actually used by the time-history kernel.
    """

    values: np.ndarray
    order_nodes: np.ndarray
    quadrature_weights: np.ndarray
    density_values: np.ndarray | None
    effective_weights: np.ndarray
    weight_semantics: str
    normalization: str
    raw_mass: float
    raw_l1_norm: float
    mass: float
    l1_norm: float
    base_definition: str
    method: str
    backend: str
    step: float
    lower_terminal: float
    grid_convention: str
    memory_policy: str
    history_window: int | None
    order_quadrature: str
    complexity: str
    working_memory: str
    approximation: str = "double_discretization_order_quadrature_and_time_gl"
    status: str = "finite_numerical_diagnostic"


def _sample_matrix(samples: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(samples, dtype=np.float64)
    was_vector = values.ndim == 1
    if was_vector:
        values = values[:, None]
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError(
            "samples must have shape (n_times,) or (n_times, dimension)."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values.")
    return np.ascontiguousarray(values), was_vector


def _finite_vector(values: object, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind == "b" or np.iscomplexobj(raw):
        raise TypeError(f"{name} must contain real non-Boolean values.")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be convertible to a finite real vector."
        ) from exc
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(array)


def _order_measure(
    order_nodes: object,
    order_weights: object,
    density_values: object | None,
    weight_semantics: str,
    normalization: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    float,
    float,
    float,
    float,
]:
    nodes = _finite_vector(order_nodes, "order_nodes")
    if np.any(nodes <= 0.0) or np.any(nodes > 1.0):
        raise ValueError("Every order node must lie in (0, 1].")
    weights = _finite_vector(order_weights, "order_weights")
    if weights.size != nodes.size:
        raise ValueError("order_weights must have one value per order node.")

    semantics = str(weight_semantics).strip().lower()
    if semantics not in _WEIGHT_SEMANTICS:
        raise ValueError(
            f"weight_semantics must be one of {sorted(_WEIGHT_SEMANTICS)}."
        )
    signed = semantics.startswith("signed_")
    uses_density = semantics.endswith("_quadrature_density")
    if uses_density:
        if density_values is None:
            raise ValueError(
                "density_values are required for *_quadrature_density semantics."
            )
        density = _finite_vector(density_values, "density_values")
        if density.size != nodes.size:
            raise ValueError("density_values must have one value per order node.")
        if not signed and (np.any(weights < 0.0) or np.any(density < 0.0)):
            raise ValueError(
                "Negative quadrature or density values require signed semantics."
            )
        raw_effective = weights * density
    else:
        if density_values is not None:
            raise ValueError("density_values are not used with *_mass semantics.")
        density = None
        if not signed and np.any(weights < 0.0):
            raise ValueError("Negative order weights require signed semantics.")
        raw_effective = np.array(weights, copy=True)

    raw_mass = float(np.sum(raw_effective, dtype=np.float64))
    raw_l1_norm = float(np.sum(np.abs(raw_effective), dtype=np.float64))
    if not np.isfinite(raw_l1_norm) or raw_l1_norm == 0.0:
        raise ValueError("The effective order measure must have non-zero mass norm.")

    normalization_value = str(normalization).strip().lower()
    if normalization_value not in _NORMALIZATIONS:
        raise ValueError(
            f"normalization must be one of {sorted(_NORMALIZATIONS)}."
        )
    if normalization_value == "unit_mass":
        cancellation_tolerance = (
            32.0 * np.finfo(np.float64).eps * raw_l1_norm
        )
        if abs(raw_mass) <= cancellation_tolerance:
            raise ValueError(
                "unit_mass normalization requires non-zero algebraic mass; "
                "the signed weights cancel."
            )
        effective = raw_effective / raw_mass
    else:
        effective = raw_effective

    effective = np.ascontiguousarray(effective)
    mass = float(np.sum(effective, dtype=np.float64))
    l1_norm = float(np.sum(np.abs(effective), dtype=np.float64))
    return (
        nodes,
        weights,
        density,
        effective,
        raw_mass,
        raw_l1_norm,
        mass,
        l1_norm,
    )


@njit(cache=True, nogil=True, parallel=True)
def _distributed_order_gl_numba(
    samples: np.ndarray,
    step: float,
    order_nodes: np.ndarray,
    effective_weights: np.ndarray,
    shift_initial: bool,
    history_window: int,
) -> np.ndarray:
    n_times, dimension = samples.shape
    output = np.zeros_like(samples)
    kernel_length = n_times
    if history_window > 0 and history_window < n_times:
        kernel_length = history_window

    # Only one order-weight vector is alive at a time.  In particular, this
    # avoids a (n_orders, n_times, dimension) intermediate derivative tensor.
    for order_index in range(order_nodes.size):
        order = order_nodes[order_index]
        order_mass = effective_weights[order_index]
        gl_weights = np.empty(kernel_length, dtype=np.float64)
        gl_weights[0] = 1.0
        for lag in range(1, kernel_length):
            gl_weights[lag] = gl_weights[lag - 1] * (
                1.0 - (order + 1.0) / lag
            )
        scale = step ** (-order)

        for component in prange(dimension):
            anchor = samples[0, component] if shift_initial else 0.0
            for n in range(n_times):
                number_of_lags = n + 1
                if number_of_lags > kernel_length:
                    number_of_lags = kernel_length
                total = 0.0
                # Descending lags reproduce the summation order of the public
                # single-order GL operator (oldest retained sample first).
                for lag in range(number_of_lags - 1, -1, -1):
                    total += gl_weights[lag] * (
                        samples[n - lag, component] - anchor
                    )
                output[n, component] += order_mass * scale * total
    return output


def _distributed_order_gl_python(
    samples: np.ndarray,
    step: float,
    order_nodes: np.ndarray,
    effective_weights: np.ndarray,
    shift_initial: bool,
    history_window: int,
) -> np.ndarray:
    """Transparent Python reference for the accumulated GL formula."""

    n_times, dimension = samples.shape
    output = np.zeros_like(samples)
    kernel_length = n_times
    if history_window > 0:
        kernel_length = min(kernel_length, history_window)

    for order, order_mass in zip(order_nodes, effective_weights, strict=True):
        gl_weights = np.empty(kernel_length, dtype=np.float64)
        gl_weights[0] = 1.0
        for lag in range(1, kernel_length):
            gl_weights[lag] = gl_weights[lag - 1] * (
                1.0 - (float(order) + 1.0) / lag
            )
        scale = step ** (-float(order))
        for component in range(dimension):
            anchor = float(samples[0, component]) if shift_initial else 0.0
            for n in range(n_times):
                number_of_lags = min(n + 1, kernel_length)
                total = 0.0
                for lag in range(number_of_lags - 1, -1, -1):
                    total += gl_weights[lag] * (
                        float(samples[n - lag, component]) - anchor
                    )
                output[n, component] += float(order_mass) * scale * total
    return output


def distributed_order_gl_derivative(
    samples: np.ndarray,
    step: float,
    order_nodes: list[float] | tuple[float, ...] | np.ndarray,
    order_weights: list[float] | tuple[float, ...] | np.ndarray,
    *,
    definition: str = "grunwald_letnikov",
    weight_semantics: str = "nonnegative_mass",
    density_values: list[float] | tuple[float, ...] | np.ndarray | None = None,
    normalization: str = "none",
    lower_terminal: float = 0.0,
    history_window: int | None = None,
    backend: str = "auto",
) -> DistributedOrderDerivativeResult:
    r"""Approximate a left distributed-order derivative on a uniform grid.

    For samples ``x_n = x(a+n*h)``, this evaluates

    ``sum_j Omega_j h**(-q_j) sum_k g_k(q_j) y_(n-k)``,

    where ``g_k(q)=(-1)**k binom(q,k)`` and ``y=x`` for the raw GL/RL
    definitions or ``y=x-x(a)`` for ``definition="caputo_shifted"``.

    Parameters
    ----------
    samples:
        Finite values of shape ``(n_times,)`` or ``(n_times, dimension)``.
    step:
        Positive spacing of the uniform time grid.
    order_nodes:
        Explicit order-quadrature nodes ``q_j`` in ``(0, 1]``.
    order_weights:
        Masses or quadrature weights, interpreted only according to
        ``weight_semantics``.
    definition:
        ``"grunwald_letnikov"`` or ``"riemann_liouville_gl"`` applies raw
        GL history. ``"caputo_shifted"`` applies GL to ``x-x(a)``; the latter
        is the usual shifted approximation for a Caputo derivative only in
        the supported range ``0 < q <= 1``.
    weight_semantics:
        One of ``"nonnegative_mass"``, ``"signed_mass"``,
        ``"nonnegative_quadrature_density"``, or
        ``"signed_quadrature_density"``.  The density variants require
        ``density_values`` and use ``Omega_j=order_weights[j]*density_values[j]``.
    normalization:
        ``"none"`` retains the effective factors. ``"unit_mass"`` divides
        them by their algebraic sum.  A signed zero-mass measure cannot be
        normalized this way.
    lower_terminal:
        Finite metadata value ``a`` identifying the first sample time.  No
        pre-terminal history is reconstructed.
    history_window:
        Optional positive count of recent samples retained in every GL
        convolution.  This changes the operator from full to finite memory.
    backend:
        ``"numba"`` for the compiled accumulating kernel, ``"python"`` for
        the direct reference implementation, or ``"auto"`` (currently
        Numba).

    Notes
    -----
    This routine supplies the finite sum after the caller has chosen an
    order quadrature.  It does not estimate quadrature error, adapt the nodes,
    solve an FDE, or establish any dynamical/chaotic property.  Diethelm and
    Ford (2009), DOI 10.1016/j.cam.2008.07.018, analyze the order-quadrature
    reduction for distributed-order equations; the time discretization here
    is specifically the direct GL formula described in the module references.
    """

    values, was_vector = _sample_matrix(samples)
    step_value = float(step)
    if not np.isfinite(step_value) or step_value <= 0.0:
        raise ValueError("step must be a finite positive number.")
    terminal = float(lower_terminal)
    if not np.isfinite(terminal):
        raise ValueError("lower_terminal must be finite.")

    definition_value = str(definition).strip().lower()
    if definition_value not in _DEFINITIONS:
        raise ValueError(f"definition must be one of {sorted(_DEFINITIONS)}.")
    backend_value = str(backend).strip().lower()
    if backend_value not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}.")
    selected_backend = "numba" if backend_value == "auto" else backend_value
    semantics_value = str(weight_semantics).strip().lower()
    normalization_value = str(normalization).strip().lower()

    (
        nodes,
        weights,
        density,
        effective,
        raw_mass,
        raw_l1_norm,
        mass,
        l1_norm,
    ) = _order_measure(
        order_nodes,
        order_weights,
        density_values,
        semantics_value,
        normalization_value,
    )

    if history_window is None:
        window_value = 0
        memory_policy = "full_history"
    else:
        if isinstance(history_window, (bool, np.bool_)):
            raise ValueError("history_window must be a positive integer.")
        try:
            window_as_float = float(history_window)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "history_window must be a positive integer."
            ) from error
        if (
            not np.isfinite(window_as_float)
            or window_as_float < 1.0
            or not window_as_float.is_integer()
        ):
            raise ValueError("history_window must be a positive integer.")
        window_value = int(window_as_float)
        memory_policy = "finite_window"

    shift_initial = definition_value == "caputo_shifted"
    if selected_backend == "numba":
        output = _distributed_order_gl_numba(
            values,
            step_value,
            nodes,
            effective,
            shift_initial,
            window_value,
        )
        method = "distributed_order_gl_direct_numba_accumulated"
    else:
        output = _distributed_order_gl_python(
            values,
            step_value,
            nodes,
            effective,
            shift_initial,
            window_value,
        )
        method = "distributed_order_gl_direct_python_reference"

    if was_vector:
        output = output[:, 0]

    retained = "n_times" if history_window is None else "min(n_times, history_window)"
    complexity = (
        f"O(n_orders * dimension * n_times * {retained}) time"
    )
    working_memory = (
        f"O(n_times * dimension + {retained}) excluding input; "
        "no order-by-time derivative tensor"
    )
    return DistributedOrderDerivativeResult(
        values=output,
        order_nodes=np.array(nodes, copy=True),
        quadrature_weights=np.array(weights, copy=True),
        density_values=None if density is None else np.array(density, copy=True),
        effective_weights=np.array(effective, copy=True),
        weight_semantics=semantics_value,
        normalization=normalization_value,
        raw_mass=raw_mass,
        raw_l1_norm=raw_l1_norm,
        mass=mass,
        l1_norm=l1_norm,
        base_definition=definition_value,
        method=method,
        backend=selected_backend,
        step=step_value,
        lower_terminal=terminal,
        grid_convention="t_n=lower_terminal+n*step",
        memory_policy=memory_policy,
        history_window=history_window,
        order_quadrature="explicit_nodes_and_declared_weights",
        complexity=complexity,
        working_memory=working_memory,
    )


__all__ = [
    "DistributedOrderDerivativeResult",
    "distributed_order_gl_derivative",
]
