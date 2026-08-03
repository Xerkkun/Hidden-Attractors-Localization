#!/usr/bin/env python
r"""Forced multi-scale relaxation with a finite Caputo derivative sum.

The manufactured affine response is attached to a standard relaxation term so
that the complete numerical trajectory can be checked without confusing a
finite multi-term equation with a continuous distributed-order quadrature.
The example includes a repeated order, a zero coefficient, a non-unit total
coefficient, and the exact integer-order branch ``alpha=1``.

This is a solver and provenance example.  It makes no chaos, attraction, or
hiddenness claim.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from scipy.special import gamma

from hidden_attractors.fractional import integrate_multi_term_caputo_l1


LOWER_TERMINAL = 0.0
STEP = 0.02
N_STEPS = 25
INITIAL_VALUE = 0.8
AFFINE_SLOPE = -0.12
RELAXATION_RATE = 0.35

# The duplicate 2/3 terms are equation coefficients, not quadrature weights.
INPUT_ORDERS = np.array([2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 0.55, 1.0])
INPUT_COEFFICIENTS = np.array([0.2, 0.4, 0.5, 0.0, 0.75])


def exact_state(time: float | np.ndarray) -> np.ndarray:
    """Return the manufactured affine relaxation response."""

    values = np.asarray(time, dtype=float)
    return INITIAL_VALUE + AFFINE_SLOPE * (values - LOWER_TERMINAL)


def _affine_caputo_derivative(time: float, order: float) -> float:
    elapsed = max(0.0, float(time) - LOWER_TERMINAL)
    if order == 1.0:
        return AFFINE_SLOPE
    return float(
        AFFINE_SLOPE
        * elapsed ** (1.0 - order)
        / gamma(2.0 - order)
    )


def forcing(time: float) -> float:
    """Manufactured forcing for the non-unit finite Caputo sum."""

    canonical_orders = (1.0 / 3.0, 2.0 / 3.0, 1.0)
    canonical_coefficients = (0.4, 0.7, 0.75)
    derivative_sum = math.fsum(
        coefficient * _affine_caputo_derivative(time, order)
        for order, coefficient in zip(
            canonical_orders,
            canonical_coefficients,
            strict=True,
        )
    )
    return derivative_sum + RELAXATION_RATE * float(exact_state(time))


def relaxation_rhs(time: float, state: np.ndarray) -> np.ndarray:
    """Return ``-lambda*x + g(t)`` for the forced relaxation equation."""

    return -RELAXATION_RATE * np.asarray(state, dtype=float) + forcing(time)


def run_example(*, use_acceleration: bool = True) -> dict[str, Any]:
    """Integrate the model and return a strict JSON-compatible audit record."""

    result = integrate_multi_term_caputo_l1(
        relaxation_rhs,
        [INITIAL_VALUE],
        orders=INPUT_ORDERS,
        coefficients=INPUT_COEFFICIENTS,
        step=STEP,
        n_steps=N_STEPS,
        lower_terminal=LOWER_TERMINAL,
        zero_coefficient_policy="drop",
        corrector_atol=1.0e-13,
        corrector_rtol=1.0e-12,
        corrector_max_iterations=100,
        initial_regularity="smooth",
        use_acceleration=use_acceleration,
        allow_python_fallback=True,
        divergence_norm=None,
    )
    expected = exact_state(result.times)
    errors = np.abs(result.states[:, 0] - expected)
    info = result.solver_info
    return {
        "model": "forced_multi_scale_caputo_relaxation",
        "equation": "sum_j c_j C_D^alpha_j x = -lambda*x + g(t)",
        "method": result.method,
        "underlying_method": info["underlying_method"],
        "status": result.status,
        "backend": result.backend,
        "memory_policy": result.memory_policy,
        "input_orders": result.original_orders.tolist(),
        "input_coefficients": result.original_coefficients.tolist(),
        "canonical_orders": result.orders.tolist(),
        "canonical_coefficients": result.coefficients.tolist(),
        "coefficient_sum": float(info["coefficient_sum"]),
        "normalization": result.normalization,
        "measure_kind": result.measure_kind,
        "continuous_order_quadrature_used": bool(
            info["continuous_order_quadrature_used"]
        ),
        "duplicate_terms_coalesced": int(info["duplicate_terms_coalesced"]),
        "zero_terms_removed": int(info["zero_terms_removed"]),
        "alpha_one_handling": info["alpha_one_handling"],
        "sample_count": int(result.times.size),
        "actual_upper_terminal": float(result.actual_upper_terminal),
        "maximum_absolute_error": float(np.max(errors)),
        "final_state": float(result.states[-1, 0]),
        "exact_final_state": float(expected[-1]),
        "implementation_reuse": info["implementation_reuse"],
        "references": list(result.references),
        "scispace_paper_ids": list(info["scispace_paper_ids"]),
        "claims": (
            "Finite manufactured-trajectory consistency only; no general "
            "convergence, stability, chaos, attraction, or hiddenness claim."
        ),
    }


def main() -> None:
    """Print the example record as strict JSON."""

    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
