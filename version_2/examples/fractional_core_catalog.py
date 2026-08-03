#!/usr/bin/env python
"""Exercise HAFO's diverse sampled fractional operators on known functions.

The example distinguishes sampled operators from FDE solvers.  Its endpoint
numbers are finite numerical demonstrations, not a proof of convergence or of
any dynamical property.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from hidden_attractors.fractional import (
    CAPUTO_HADAMARD_INITIAL_CONDITION,
    CAPUTO_SHIFTED_INITIAL_CONDITION,
    OPERATOR_ONLY_INITIAL_CONDITION,
    atangana_baleanu_caputo_derivative,
    atangana_baleanu_normalization,
    caputo_fabrizio_derivative,
    conformable_khalil_derivative,
    distributed_order_gl_derivative,
    fast_grunwald_letnikov_derivative,
    get_fractional_derivative,
    get_fractional_method,
    gl_linear_convolution_fft_length,
    grunwald_letnikov_derivative,
    grunwald_letnikov_weights,
    hadamard_convolution_quadrature,
    integrate_conformable_rk4,
    integrate_abc_predictor_corrector,
    integrate_distributed_order_caputo_l1,
    integrate_gl_explicit,
    integrate_tempered_caputo_abm,
    integrate_variable_order_caputo_type3_l1,
    list_fractional_derivatives,
    list_fractional_methods,
    lubich_bdf_weights,
    lubich_convolution_quadrature,
    riemann_liouville_gl_derivative,
    tempered_grunwald_letnikov_derivative,
    variable_order_grunwald_letnikov_derivative,
)


def _last(values: np.ndarray) -> float:
    return float(np.asarray(values).reshape(-1)[-1])


def run_catalog(sample_count: int = 257) -> dict[str, Any]:
    """Return a compact, JSON-compatible run record for the operator catalog."""

    if sample_count < 17:
        raise ValueError("sample_count must be at least 17.")
    times = np.linspace(0.0, 1.0, sample_count)
    step = float(times[1] - times[0])
    samples = 1.0 + times**3
    order = 0.62

    gl = grunwald_letnikov_derivative(
        samples, step, order, definition="caputo_shifted"
    )
    fast_gl = fast_grunwald_letnikov_derivative(
        samples,
        step,
        order,
        definition="caputo_shifted",
        backend="fft",
    )
    rl = riemann_liouville_gl_derivative(
        samples,
        times,
        order,
        lower_terminal=0.0,
        initial_condition_semantics=OPERATOR_ONLY_INITIAL_CONDITION,
    )
    tempered = tempered_grunwald_letnikov_derivative(
        samples,
        times,
        order,
        tempering=0.35,
        lower_terminal=0.0,
        initial_condition_semantics=OPERATOR_ONLY_INITIAL_CONDITION,
    )
    variable = variable_order_grunwald_letnikov_derivative(
        samples,
        times,
        0.45 + 0.25 * times,
        lower_terminal=0.0,
        initial_condition_semantics=OPERATOR_ONLY_INITIAL_CONDITION,
    )
    conformable = conformable_khalil_derivative(
        times,
        3.0 * times**2,
        order,
        lower_terminal=0.0,
        terminal_policy="bounded_derivative_zero",
    )
    caputo_fabrizio = caputo_fabrizio_derivative(
        samples, step, order, lower_terminal=0.0
    )
    abc_order = 0.5
    atangana_baleanu = atangana_baleanu_caputo_derivative(
        samples,
        step,
        abc_order,
        lower_terminal=0.0,
        normalization=atangana_baleanu_normalization,
        normalization_name="B(alpha)=1-alpha+alpha/Gamma(alpha)",
        backend="fft",
    )
    distributed = distributed_order_gl_derivative(
        samples,
        step,
        [0.3, 0.6, 0.9],
        [0.2, 0.5, 0.3],
        definition="caputo_shifted",
        weight_semantics="nonnegative_mass",
        normalization="unit_mass",
    )
    lubich = lubich_convolution_quadrature(
        samples,
        order,
        bdf_order=2,
        definition="caputo_shifted",
        times=times,
        lower_terminal=0.0,
        initial_condition_semantics=CAPUTO_SHIFTED_INITIAL_CONDITION,
        backend="fft",
    )

    log_times = times
    lower_terminal = 2.0
    hadamard_samples = 1.0 + log_times**3
    hadamard = hadamard_convolution_quadrature(
        hadamard_samples,
        order,
        bdf_order=2,
        definition="caputo_hadamard",
        times=lower_terminal * np.exp(log_times),
        lower_terminal=lower_terminal,
        initial_condition_semantics=CAPUTO_HADAMARD_INITIAL_CONDITION,
        backend="fft",
    )

    def constant_forcing(time: float, state: np.ndarray) -> np.ndarray:
        del time
        return np.ones_like(state)

    gl_solver = integrate_gl_explicit(
        constant_forcing,
        np.array([1.0]),
        order,
        step=0.01,
        n_steps=50,
        use_acceleration=False,
    )
    conformable_solver = integrate_conformable_rk4(
        constant_forcing,
        np.array([1.0]),
        0.5,
        lower_terminal=0.0,
        upper_terminal=1.0,
        clock_step=0.01,
        use_acceleration=False,
        divergence_norm=None,
    )
    abc_solver_order = 0.7

    def compatible_quadratic_forcing(
        time: float,
        state: np.ndarray,
    ) -> np.ndarray:
        return np.full_like(state, time * time)

    abc_solver = integrate_abc_predictor_corrector(
        compatible_quadratic_forcing,
        np.array([1.0]),
        abc_solver_order,
        step=0.01,
        n_steps=50,
        normalization=1.0,
        normalization_name="B(alpha)=1",
        use_acceleration=False,
        divergence_norm=None,
    )
    tempering = 0.4
    tempered_solver_order = 0.7
    power = 2.0
    tempered_coefficient = math.gamma(power + 1.0) / math.gamma(
        power + 1.0 - tempered_solver_order
    )

    def tempered_manufactured_forcing(
        time: float,
        state: np.ndarray,
    ) -> np.ndarray:
        value = (
            math.exp(-tempering * time)
            * tempered_coefficient
            * time ** (power - tempered_solver_order)
        )
        return np.full_like(state, value)

    tempered_caputo_solver = integrate_tempered_caputo_abm(
        tempered_manufactured_forcing,
        np.array([1.0]),
        tempered_solver_order,
        tempering=tempering,
        lower_terminal=0.0,
        upper_terminal=0.5,
        step=0.01,
        use_acceleration=False,
        divergence_norm=None,
    )
    variable_order_power = 2.0

    def type3_order(time: float, initial_state: np.ndarray) -> float:
        del initial_state
        return 0.52 + 0.2 * time

    def type3_manufactured_forcing(
        time: float,
        state: np.ndarray,
    ) -> np.ndarray:
        del state
        alpha = type3_order(time, np.array([1.0]))
        coefficient = math.gamma(variable_order_power + 1.0) / math.gamma(
            variable_order_power + 1.0 - alpha
        )
        return np.array([coefficient * time ** (variable_order_power - alpha)])

    variable_order_type3_solver = integrate_variable_order_caputo_type3_l1(
        type3_manufactured_forcing,
        np.array([1.0]),
        step=0.01,
        n_steps=50,
        lower_terminal=0.0,
        order_function=type3_order,
        order_function_name="alpha(t)=0.52+0.2t",
        initial_regularity="smooth",
        use_acceleration=False,
        divergence_norm=None,
    )
    distributed_order_nodes = np.array([0.35, 0.80])
    distributed_order_weights = np.array([0.40, 0.60])

    def distributed_order_manufactured_forcing(
        time: float,
        state: np.ndarray,
    ) -> np.ndarray:
        del state
        value = 0.0
        for alpha, mass in zip(
            distributed_order_nodes,
            distributed_order_weights,
            strict=True,
        ):
            value += (
                mass
                * math.gamma(3.0)
                / math.gamma(3.0 - alpha)
                * time ** (2.0 - alpha)
            )
        return np.array([value])

    distributed_order_caputo_solver = integrate_distributed_order_caputo_l1(
        distributed_order_manufactured_forcing,
        np.array([1.0]),
        order_nodes=distributed_order_nodes,
        order_weights=distributed_order_weights,
        step=0.01,
        n_steps=50,
        lower_terminal=0.0,
        initial_regularity="smooth",
        use_acceleration=False,
        divergence_norm=None,
    )

    direct_fast_difference = float(
        np.max(np.abs(np.asarray(gl.values) - np.asarray(fast_gl.values)))
    )
    return {
        "scope": "finite_sampled_operator_and_manufactured_solver_demo",
        "sample_count": sample_count,
        "registry": {
            "derivative_count": len(list_fractional_derivatives()),
            "method_count": len(list_fractional_methods()),
            "caputo_hadamard_kernel": get_fractional_derivative(
                "caputo_hadamard"
            ).kernel_family,
            "hadamard_cq_execution_kind": get_fractional_method(
                "hadamard_convolution_quadrature"
            ).execution_kind,
        },
        "weights": {
            "gl_first_four": grunwald_letnikov_weights(order, 4).tolist(),
            "lubich_bdf2_first_four": lubich_bdf_weights(
                order, 4, bdf_order=2
            ).tolist(),
            "fft_linear_length": gl_linear_convolution_fft_length(sample_count),
        },
        "operator_endpoints": {
            "caputo_shifted_gl": _last(gl.values),
            "caputo_shifted_gl_fft": _last(fast_gl.values),
            "riemann_liouville_gl": _last(rl.values),
            "tempered_riemann_liouville_gl": _last(tempered.values),
            "variable_order_frozen_gl": _last(variable.values),
            "conformable_khalil": _last(conformable.values),
            "caputo_fabrizio": _last(caputo_fabrizio.values),
            "atangana_baleanu_caputo": _last(atangana_baleanu.values),
            "distributed_order_gl": _last(distributed.values),
            "lubich_bdf2_caputo_shifted": _last(lubich.values),
            "caputo_hadamard_bdf2": _last(hadamard.values),
        },
        "backend_checks": {
            "gl_direct_vs_fft_max_abs": direct_fast_difference,
            "lubich_backend": lubich.backend,
            "hadamard_backend": hadamard.backend,
            "atangana_baleanu_backend": atangana_baleanu.backend,
        },
        "manufactured_gl_solver": {
            "status": gl_solver.status,
            "backend": gl_solver.backend,
            "final_time": float(gl_solver.times[-1]),
            "final_state": gl_solver.states[-1].tolist(),
        },
        "manufactured_conformable_solver": {
            "status": conformable_solver.status,
            "backend": conformable_solver.backend,
            "physical_final_time": float(conformable_solver.times[-1]),
            "clock_final_time": float(conformable_solver.clock_times[-1]),
            "final_state": conformable_solver.states[-1].tolist(),
            "memory_policy": conformable_solver.memory_policy,
        },
        "manufactured_abc_solver": {
            "status": abc_solver.status,
            "backend": abc_solver.backend,
            "final_time": float(abc_solver.times[-1]),
            "final_state": abc_solver.states[-1].tolist(),
            "normalization": abc_solver.normalization_description,
            "compatibility_residual": abc_solver.compatibility_residual,
            "memory_policy": abc_solver.memory_policy,
        },
        "manufactured_tempered_caputo_solver": {
            "status": tempered_caputo_solver.status,
            "backend": tempered_caputo_solver.backend,
            "final_time": float(tempered_caputo_solver.times[-1]),
            "final_state": tempered_caputo_solver.states[-1].tolist(),
            "tempering": tempered_caputo_solver.tempering,
            "memory_policy": tempered_caputo_solver.memory_policy,
            "lambda_zero_reduction": tempered_caputo_solver.solver_info[
                "lambda_zero_reduction"
            ],
        },
        "manufactured_variable_order_type3_solver": {
            "status": variable_order_type3_solver.status,
            "backend": variable_order_type3_solver.backend,
            "final_time": float(variable_order_type3_solver.times[-1]),
            "final_state": variable_order_type3_solver.states[-1].tolist(),
            "order_min": float(np.min(variable_order_type3_solver.orders)),
            "order_max": float(np.max(variable_order_type3_solver.orders)),
            "definition": variable_order_type3_solver.solver_info["definition"],
            "memory_policy": variable_order_type3_solver.memory_policy,
        },
        "manufactured_distributed_order_caputo_solver": {
            "status": distributed_order_caputo_solver.status,
            "backend": distributed_order_caputo_solver.backend,
            "final_time": float(distributed_order_caputo_solver.times[-1]),
            "final_state": distributed_order_caputo_solver.states[-1].tolist(),
            "order_nodes": distributed_order_caputo_solver.order_nodes.tolist(),
            "effective_weights": (
                distributed_order_caputo_solver.effective_weights.tolist()
            ),
            "definition": distributed_order_caputo_solver.solver_info[
                "definition"
            ],
            "memory_policy": distributed_order_caputo_solver.memory_policy,
            "final_manufactured_abs_error": float(
                abs(distributed_order_caputo_solver.states[-1, 0] - 1.25)
            ),
        },
        "claims": (
            "No stability, convergence, chaos, attraction, or hiddenness claim "
            "is inferred from this single finite run."
        ),
    }


def main() -> None:
    print(json.dumps(run_catalog(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
