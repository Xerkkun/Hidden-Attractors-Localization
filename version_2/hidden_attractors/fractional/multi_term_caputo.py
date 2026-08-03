r"""Semantic facade for finite multi-term Caputo equations.

Stability: experimental

This module represents

.. math::

   \sum_{j=1}^{R} c_j\,{}_{a}^{C}D_t^{\alpha_j}x(t)=f(t,x(t)),

with ``0 < alpha_j <= 1`` and nonnegative equation coefficients ``c_j``.
The finite sum is an atomic order measure.  It is *not* presented as a
quadrature approximation to a continuous distributed-order density.

The numerical work is deliberately delegated to HAFO's existing combined-L1
distributed-order kernel.  The facade only validates and canonicalizes the
finite equation terms, then records their distinct semantics.  It therefore
does not copy or reconstruct the ``O(R*N + N**2*d)`` solver.

References
----------
K. Diethelm and N. J. Ford, Applied Mathematics and Computation 154 (2004),
621--640, https://doi.org/10.1016/S0096-3003(03)00739-2.

J. Ren and Z.-Z. Sun, East Asian Journal on Applied Mathematics 4 (2014),
242--266, https://doi.org/10.4208/EAJAM.181113.280514A.

M. She, D. Li, and H.-W. Sun, Mathematics and Computers in Simulation 193
(2022), 584--606, https://doi.org/10.1016/j.matcom.2021.11.005.

M. A. Zaky and J. A. Tenreiro Machado, Computers & Mathematics with
Applications 79 (2020), 476--488,
https://doi.org/10.1016/j.camwa.2019.07.008.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from .distributed_order_caputo_solver import (
    DistributedOrderCaputoResult,
    integrate_distributed_order_caputo_l1,
)


MULTI_TERM_CAPUTO_L1_REFERENCES = (
    "https://doi.org/10.1016/S0096-3003(03)00739-2",
    "https://doi.org/10.4208/EAJAM.181113.280514A",
    "https://doi.org/10.1016/j.matcom.2021.11.005",
    "https://doi.org/10.1016/j.camwa.2019.07.008",
)

MULTI_TERM_CAPUTO_SCISPACE_PAPER_IDS = (
    "3tp7pod1yv",
    "1o5cvoa44t",
    "3hf80rfehx",
    "3ued1yt7yp",
)


def _finite_real_vector(values: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind == "b" or np.iscomplexobj(raw):
        raise TypeError(f"{name} must contain real non-Boolean values.")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be convertible to a real one-dimensional vector."
        ) from exc
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class MultiTermCaputoTerms:
    """Original and canonical terms of one finite Caputo equation.

    Canonical terms are sorted by increasing order.  Duplicate orders are
    coalesced only when their normalized ``float64`` values are exactly equal;
    no tolerance is used because merging nearby orders would change the model.
    Coefficients are accumulated with :func:`math.fsum`.
    """

    original_orders: np.ndarray
    original_coefficients: np.ndarray
    orders: np.ndarray
    coefficients: np.ndarray
    source_indices: tuple[tuple[int, ...], ...]
    zero_coefficient_indices: tuple[int, ...]
    zero_coefficient_policy: str
    canonicalization: str = "sort_ascending_exact_order_coalescence_math_fsum"
    measure_kind: str = "finite_discrete_atomic_order_measure"
    normalization: str = "none"

    @property
    def original_term_count(self) -> int:
        """Number of terms supplied by the caller, including zero terms."""

        return int(self.original_orders.size)

    @property
    def term_count(self) -> int:
        """Number of nonzero canonical terms sent to the numerical kernel."""

        return int(self.orders.size)

    @property
    def zero_terms_removed(self) -> int:
        """Number of exact zero coefficients removed before integration."""

        return len(self.zero_coefficient_indices)

    @property
    def duplicate_terms_coalesced(self) -> int:
        """Number of positive duplicate-order terms eliminated by grouping."""

        return (
            self.original_term_count
            - self.zero_terms_removed
            - self.term_count
        )

    @property
    def coefficient_sum(self) -> float:
        """Algebraic mass of the equation coefficients without normalization."""

        return float(math.fsum(float(value) for value in self.coefficients))


def canonicalize_multi_term_caputo_terms(
    orders: Any,
    coefficients: Any,
    *,
    zero_coefficient_policy: str = "drop",
) -> MultiTermCaputoTerms:
    """Validate and canonicalize a finite multi-term Caputo equation.

    Parameters
    ----------
    orders, coefficients:
        One-dimensional sequences with one coefficient per order.
    zero_coefficient_policy:
        ``"drop"`` removes exact zeros and records their original indices;
        ``"raise"`` rejects them.  Negative coefficients are rejected by both
        policies because the reused implicit solver has a positive-measure
        well-posedness contract.
    """

    original_orders = _finite_real_vector(orders, name="orders")
    original_coefficients = _finite_real_vector(
        coefficients,
        name="coefficients",
    )
    if original_orders.size != original_coefficients.size:
        raise ValueError("coefficients must contain one value per order.")
    if np.any(original_orders <= 0.0) or np.any(original_orders > 1.0):
        raise ValueError("Every order must lie in (0, 1].")
    if np.any(original_coefficients < 0.0):
        raise ValueError(
            "Multi-term Caputo L1 requires nonnegative equation coefficients."
        )

    zero_policy = str(zero_coefficient_policy).strip().lower()
    if zero_policy not in {"drop", "raise"}:
        raise ValueError("zero_coefficient_policy must be 'drop' or 'raise'.")
    zero_indices = tuple(
        int(index)
        for index in np.flatnonzero(original_coefficients == 0.0)
    )
    if zero_policy == "raise" and zero_indices:
        raise ValueError(
            "Zero equation coefficients are forbidden by "
            "zero_coefficient_policy='raise'."
        )

    positive_indices = np.flatnonzero(original_coefficients > 0.0)
    if positive_indices.size == 0:
        raise ValueError("At least one equation coefficient must be positive.")

    grouped_indices: dict[float, list[int]] = {}
    for index in positive_indices:
        order = float(original_orders[int(index)])
        grouped_indices.setdefault(order, []).append(int(index))

    canonical_orders = np.array(sorted(grouped_indices), dtype=np.float64)
    canonical_coefficients_values: list[float] = []
    source_indices: list[tuple[int, ...]] = []
    for order in canonical_orders:
        indices = tuple(grouped_indices[float(order)])
        try:
            coefficient = math.fsum(
                float(original_coefficients[index]) for index in indices
            )
        except OverflowError as exc:
            raise ValueError(
                "The coalesced equation coefficients overflow float64."
            ) from exc
        if not np.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError(
                "Every canonical equation coefficient must remain finite and positive."
            )
        canonical_coefficients_values.append(coefficient)
        source_indices.append(indices)

    canonical_coefficients = np.asarray(
        canonical_coefficients_values,
        dtype=np.float64,
    )
    try:
        coefficient_sum = math.fsum(
            float(value) for value in canonical_coefficients
        )
    except OverflowError as exc:
        raise ValueError(
            "The total equation coefficient sum overflows float64."
        ) from exc
    if not np.isfinite(coefficient_sum) or coefficient_sum <= 0.0:
        raise ValueError("The equation coefficient sum must be finite and positive.")

    return MultiTermCaputoTerms(
        original_orders=_readonly_copy(original_orders),
        original_coefficients=_readonly_copy(original_coefficients),
        orders=_readonly_copy(canonical_orders),
        coefficients=_readonly_copy(canonical_coefficients),
        source_indices=tuple(source_indices),
        zero_coefficient_indices=zero_indices,
        zero_coefficient_policy=zero_policy,
    )


@dataclass(frozen=True, slots=True)
class MultiTermCaputoResult:
    """Finite trajectory from the semantic multi-term Caputo facade."""

    distributed_result: DistributedOrderCaputoResult
    terms: MultiTermCaputoTerms
    solver_info: Mapping[str, Any]
    method: str = "multi_term_caputo_l1"
    definition: str = "caputo_multi_term_finite_sum"
    measure_kind: str = "finite_discrete_atomic_order_measure"
    normalization: str = "none"
    references: tuple[str, ...] = MULTI_TERM_CAPUTO_L1_REFERENCES
    scope: str = "finite_numerical_trajectory_only"

    @property
    def times(self) -> np.ndarray:
        return self.distributed_result.times

    @property
    def states(self) -> np.ndarray:
        return self.distributed_result.states

    @property
    def trajectory(self) -> np.ndarray:
        return self.distributed_result.trajectory

    @property
    def orders(self) -> np.ndarray:
        return self.terms.orders

    @property
    def coefficients(self) -> np.ndarray:
        return self.terms.coefficients

    @property
    def original_orders(self) -> np.ndarray:
        return self.terms.original_orders

    @property
    def original_coefficients(self) -> np.ndarray:
        return self.terms.original_coefficients

    @property
    def corrector_iterations(self) -> np.ndarray:
        return self.distributed_result.corrector_iterations

    @property
    def corrector_residuals(self) -> np.ndarray:
        return self.distributed_result.corrector_residuals

    @property
    def l1_coefficients(self) -> np.ndarray:
        return self.distributed_result.l1_coefficients

    @property
    def combined_l1_kernel(self) -> np.ndarray:
        return self.distributed_result.combined_l1_kernel

    @property
    def backend(self) -> str:
        return self.distributed_result.backend

    @property
    def status(self) -> str:
        return self.distributed_result.status

    @property
    def memory_policy(self) -> str:
        return self.distributed_result.memory_policy

    @property
    def lower_terminal(self) -> float:
        return self.distributed_result.lower_terminal

    @property
    def requested_upper_terminal(self) -> float:
        return self.distributed_result.requested_upper_terminal

    @property
    def actual_upper_terminal(self) -> float:
        return self.distributed_result.actual_upper_terminal

    @property
    def step(self) -> float:
        return self.distributed_result.step

    @property
    def n_steps_requested(self) -> int:
        return self.distributed_result.n_steps_requested

    @property
    def initial_regularity(self) -> str:
        return self.distributed_result.initial_regularity


def integrate_multi_term_caputo_l1(
    rhs: Callable,
    initial_state: Any,
    parameters: Any = None,
    *,
    orders: Any,
    coefficients: Any,
    step: float,
    n_steps: int,
    lower_terminal: float = 0.0,
    zero_coefficient_policy: str = "drop",
    corrector_atol: float = 1.0e-12,
    corrector_rtol: float = 1.0e-10,
    corrector_max_iterations: int = 50,
    on_nonconvergence: str = "raise",
    initial_regularity: str = "unknown",
    compatibility_tolerance: float = 1.0e-10,
    use_acceleration: bool = True,
    allow_python_fallback: bool = True,
    divergence_norm: float | None = 120.0,
) -> MultiTermCaputoResult:
    """Integrate a finite positive-coefficient multi-term Caputo equation.

    Coefficients are equation coefficients and are never normalized.  Exact
    duplicate orders are coalesced before the existing combined-L1 kernel is
    called, reducing its one-time ``O(R*N)`` construction without changing the
    equation.  The ``O(N**2*d)`` full-history loop remains the same tested Numba
    or Python implementation used by the distributed-order solver.
    """

    terms = canonicalize_multi_term_caputo_terms(
        orders,
        coefficients,
        zero_coefficient_policy=zero_coefficient_policy,
    )
    distributed_result = integrate_distributed_order_caputo_l1(
        rhs,
        initial_state,
        parameters,
        order_nodes=terms.orders,
        order_weights=terms.coefficients,
        step=step,
        n_steps=n_steps,
        lower_terminal=lower_terminal,
        weight_semantics="nonnegative_mass",
        density_values=None,
        normalization="none",
        order_quadrature_name="finite_atomic_multi_term_caputo_equation",
        corrector_atol=corrector_atol,
        corrector_rtol=corrector_rtol,
        corrector_max_iterations=corrector_max_iterations,
        on_nonconvergence=on_nonconvergence,
        initial_regularity=initial_regularity,
        compatibility_tolerance=compatibility_tolerance,
        use_acceleration=use_acceleration,
        allow_python_fallback=allow_python_fallback,
        divergence_norm=divergence_norm,
    )

    underlying_info = dict(distributed_result.solver_info)
    solver_info: dict[str, Any] = {
        **underlying_info,
        "definition": "caputo_multi_term_finite_sum",
        "equation_form": "sum_j c_j * C_D_a^alpha_j x = rhs(t, x)",
        "order_measure_kind": terms.measure_kind,
        "continuous_order_quadrature_used": False,
        "continuous_order_density_inferred": False,
        "coefficient_normalization": "none",
        "coefficient_sum": terms.coefficient_sum,
        "original_term_count": terms.original_term_count,
        "canonical_term_count": terms.term_count,
        "zero_coefficient_policy": terms.zero_coefficient_policy,
        "zero_coefficient_indices": list(terms.zero_coefficient_indices),
        "zero_terms_removed": terms.zero_terms_removed,
        "duplicate_terms_coalesced": terms.duplicate_terms_coalesced,
        "duplicate_order_coalescence": "exact_float64_equality",
        "coefficient_accumulation": "math_fsum",
        "canonical_ordering": "ascending",
        "canonicalization": terms.canonicalization,
        "original_orders": terms.original_orders.tolist(),
        "original_coefficients": terms.original_coefficients.tolist(),
        "canonical_orders": terms.orders.tolist(),
        "canonical_coefficients": terms.coefficients.tolist(),
        "source_indices_by_canonical_term": [
            list(indices) for indices in terms.source_indices
        ],
        "underlying_definition": underlying_info.get("definition"),
        "underlying_method": distributed_result.method,
        "underlying_order_rule_name": distributed_result.order_quadrature_name,
        "implementation_reuse": (
            "distributed_order_combined_l1_kernel_without_solver_reconstruction"
        ),
        "facade_constructed_kernel": False,
        "all_terms_share_lower_terminal": True,
        "all_terms_share_initial_state": True,
        "signed_coefficients_supported": False,
        "zero_order_term_supported": False,
        "scispace_paper_ids": list(MULTI_TERM_CAPUTO_SCISPACE_PAPER_IDS),
        "reference_urls": list(MULTI_TERM_CAPUTO_L1_REFERENCES),
        "claims": "finite_numerical_trajectory_only",
    }
    return MultiTermCaputoResult(
        distributed_result=distributed_result,
        terms=terms,
        solver_info=MappingProxyType(solver_info),
    )


__all__ = [
    "MULTI_TERM_CAPUTO_L1_REFERENCES",
    "MULTI_TERM_CAPUTO_SCISPACE_PAPER_IDS",
    "MultiTermCaputoResult",
    "MultiTermCaputoTerms",
    "canonicalize_multi_term_caputo_terms",
    "integrate_multi_term_caputo_l1",
]
