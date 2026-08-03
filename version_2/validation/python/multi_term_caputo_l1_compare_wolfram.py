"""Compare the finite multi-term Caputo facade with independent Wolfram data.

The Wolfram case derives fractional L1 weights by integrating the Caputo
kernel, adds the alpha=1 backward-Euler branch explicitly, and advances a
directly solved scalar linear recurrence. It never reads HAFO source. This
module reaches the implementation only through ``integrate_multi_term_caputo_l1``.

Passing is finite algebraic/numerical consistency evidence. It is not a
convergence theorem or evidence of nonlinear stability, chaos, attraction, or
hiddenness.
"""

from __future__ import annotations

import argparse
import json
from math import gamma
from pathlib import Path
from typing import Any

import numpy as np

from hidden_attractors.fractional import integrate_multi_term_caputo_l1


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_ID = "multi_term_caputo_l1"
DEFAULT_SUMMARY = (
    ROOT
    / "validation"
    / "outputs"
    / "wolfram"
    / SYSTEM_ID
    / f"{SYSTEM_ID}_validation_summary.json"
)
NUMERIC_TOLERANCE = 8.0e-12

SOURCE_ANCHORS = {
    "diethelm_ford_doi": "10.1016/S0096-3003(03)00739-2",
    "ren_sun_doi": "10.4208/EAJAM.181113.280514A",
    "she_li_sun_doi": "10.1016/j.matcom.2021.11.005",
    "zaky_machado_doi": "10.1016/j.camwa.2019.07.008",
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _max_abs(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError(
            "shape mismatch in Wolfram/Python comparison: "
            f"{left_array.shape} != {right_array.shape}"
        )
    return float(np.max(np.abs(left_array - right_array), initial=0.0))


def _independent_l1_weight(order: float, lag: int) -> float:
    if order == 1.0:
        return 1.0 if lag == 0 else 0.0
    return float((lag + 1.0) ** (1.0 - order) - lag ** (1.0 - order))


def _independent_kernel(
    orders: np.ndarray,
    coefficients: np.ndarray,
    *,
    step: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    l1_coefficients = np.asarray(
        [
            coefficient * step ** (-order) / gamma(2.0 - order)
            for order, coefficient in zip(orders, coefficients, strict=True)
        ],
        dtype=np.float64,
    )
    per_term = np.asarray(
        [
            [
                coefficient * _independent_l1_weight(float(order), lag)
                for lag in range(n_steps)
            ]
            for order, coefficient in zip(
                orders,
                l1_coefficients,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    return l1_coefficients, per_term


def _exact_state(
    time: float,
    *,
    lower_terminal: float,
    initial_value: float,
    slope: float,
) -> float:
    return initial_value + slope * (time - lower_terminal)


def _multi_term_affine_derivative(
    time: float,
    *,
    lower_terminal: float,
    slope: float,
    orders: np.ndarray,
    coefficients: np.ndarray,
) -> float:
    elapsed = max(0.0, float(time) - lower_terminal)
    total = 0.0
    for order, coefficient in zip(orders, coefficients, strict=True):
        if order == 1.0:
            total += float(coefficient) * slope
        elif elapsed > 0.0:
            total += (
                float(coefficient)
                * slope
                * elapsed ** (1.0 - float(order))
                / gamma(2.0 - float(order))
            )
    return float(total)


def _recurrence_residuals(
    states: np.ndarray,
    kernel: np.ndarray,
    forcing: np.ndarray,
    *,
    linear_coefficient: float,
) -> np.ndarray:
    residuals = np.zeros(states.size - 1, dtype=np.float64)
    for output_index in range(1, states.size):
        lhs = 0.0
        for lag in range(output_index):
            lhs += kernel[lag] * (
                states[output_index - lag]
                - states[output_index - lag - 1]
            )
        residuals[output_index - 1] = (
            lhs
            - linear_coefficient * states[output_index]
            - forcing[output_index]
        )
    return residuals


def compare_wolfram_summary(
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    tolerance: float = NUMERIC_TOLERANCE,
) -> dict[str, Any]:
    """Recompute the independent rational fixture through the public facade."""

    summary_path = Path(summary_path).resolve()
    payload = _load(summary_path)
    if payload.get("system_id") != SYSTEM_ID:
        raise ValueError(
            f"unexpected system_id {payload.get('system_id')!r}; "
            f"expected {SYSTEM_ID!r}"
        )

    parameters = payload["parameters"]
    lower_terminal = float(parameters["lower_terminal"])
    step = float(parameters["step"])
    n_steps = int(parameters["n_steps"])
    original_orders = np.asarray(parameters["original_orders"], dtype=np.float64)
    original_coefficients = np.asarray(
        parameters["original_coefficients"],
        dtype=np.float64,
    )
    canonical_orders = np.asarray(
        parameters["canonical_orders"],
        dtype=np.float64,
    )
    canonical_coefficients = np.asarray(
        parameters["canonical_coefficients"],
        dtype=np.float64,
    )
    expected_source_indices = tuple(
        tuple(int(index) for index in indices)
        for indices in parameters["expected_source_indices_zero_based"]
    )
    expected_zero_indices = tuple(
        int(index) for index in parameters["expected_zero_indices_zero_based"]
    )
    initial_value = float(parameters["initial_value"])
    slope = float(parameters["slope"])
    linear_coefficient = float(parameters["lambda"])

    expected_times = lower_terminal + step * np.arange(n_steps + 1)
    exact_states = np.asarray(
        [
            _exact_state(
                float(time),
                lower_terminal=lower_terminal,
                initial_value=initial_value,
                slope=slope,
            )
            for time in expected_times
        ],
        dtype=np.float64,
    )
    exact_derivative = np.asarray(
        [
            _multi_term_affine_derivative(
                float(time),
                lower_terminal=lower_terminal,
                slope=slope,
                orders=canonical_orders,
                coefficients=canonical_coefficients,
            )
            for time in expected_times
        ],
        dtype=np.float64,
    )
    forcing = exact_derivative - linear_coefficient * exact_states

    expected_l1_coefficients, expected_per_term_kernel = _independent_kernel(
        canonical_orders,
        canonical_coefficients,
        step=step,
        n_steps=n_steps,
    )
    expected_combined_kernel = np.sum(expected_per_term_kernel, axis=0)

    def manufactured_rhs(time: float, state: np.ndarray) -> np.ndarray:
        exact_at_time = _exact_state(
            time,
            lower_terminal=lower_terminal,
            initial_value=initial_value,
            slope=slope,
        )
        derivative_at_time = _multi_term_affine_derivative(
            time,
            lower_terminal=lower_terminal,
            slope=slope,
            orders=canonical_orders,
            coefficients=canonical_coefficients,
        )
        forcing_at_time = (
            derivative_at_time - linear_coefficient * exact_at_time
        )
        return linear_coefficient * state + forcing_at_time

    result = integrate_multi_term_caputo_l1(
        manufactured_rhs,
        [initial_value],
        orders=original_orders,
        coefficients=original_coefficients,
        step=step,
        n_steps=n_steps,
        lower_terminal=lower_terminal,
        zero_coefficient_policy="drop",
        corrector_atol=1.0e-14,
        corrector_rtol=1.0e-14,
        corrector_max_iterations=100,
        on_nonconvergence="raise",
        initial_regularity="smooth",
        use_acceleration=False,
        divergence_norm=None,
    )

    kernel = payload["kernel"]
    kernel_metrics = {
        "independent_to_wolfram_integrated_per_term_max_diff": _max_abs(
            expected_per_term_kernel,
            kernel["per_term_integrated_values"],
        ),
        "independent_to_wolfram_formula_per_term_max_diff": _max_abs(
            expected_per_term_kernel,
            kernel["per_term_formula_values"],
        ),
        "independent_to_wolfram_combined_max_diff": _max_abs(
            expected_combined_kernel,
            kernel["combined_integrated_values"],
        ),
        "facade_l1_coefficients_max_diff": _max_abs(
            result.l1_coefficients,
            expected_l1_coefficients,
        ),
        "facade_combined_kernel_max_diff": _max_abs(
            result.combined_l1_kernel,
            expected_combined_kernel,
        ),
        "current_step_coefficient_max_diff": abs(
            float(expected_combined_kernel[0])
            - float(kernel["current_step_coefficient"])
        ),
    }
    kernel_metrics["max_diff"] = max(kernel_metrics.values())

    manufactured = payload["manufactured_case"]
    recurrence_residuals = _recurrence_residuals(
        result.states[:, 0],
        result.combined_l1_kernel,
        forcing,
        linear_coefficient=linear_coefficient,
    )
    trajectory_metrics = {
        "time_grid_facade_to_wolfram_max_diff": _max_abs(
            result.times,
            manufactured["times"],
        ),
        "forcing_python_to_wolfram_max_diff": _max_abs(
            forcing,
            manufactured["forcing_values"],
        ),
        "facade_states_to_wolfram_max_diff": _max_abs(
            result.states[:, 0],
            manufactured["states"],
        ),
        "facade_states_to_exact_max_diff": _max_abs(
            result.states[:, 0],
            exact_states,
        ),
        "wolfram_states_to_exact_max_diff": _max_abs(
            manufactured["states"],
            exact_states,
        ),
        "facade_recurrence_residual_max": float(
            np.max(np.abs(recurrence_residuals), initial=0.0)
        ),
        "wolfram_reported_recurrence_residual": abs(
            float(manufactured["max_recurrence_residual"])
        ),
        "wolfram_reported_exact_error": abs(
            float(manufactured["max_exact_error"])
        ),
    }
    trajectory_metrics["max_diff"] = max(trajectory_metrics.values())

    quadratic = payload["quadratic_caputo_identity"]
    elapsed = float(quadratic["probe_time"]) - lower_terminal
    quadratic_probe_expected = float(
        sum(
            coefficient
            * gamma(3.0)
            / gamma(3.0 - order)
            * elapsed ** (2.0 - order)
            for order, coefficient in zip(
                canonical_orders,
                canonical_coefficients,
                strict=True,
            )
        )
    )
    quadratic_probe_diff = abs(
        quadratic_probe_expected - float(quadratic["probe_value"])
    )

    source = payload["source"]
    source_anchors_match = all(
        source.get(key) == expected
        for key, expected in SOURCE_ANCHORS.items()
    )
    independence_flags_match = bool(
        source.get("hafo_source_read") is False
        and source.get("report_input_used") is False
        and source.get("hafo_formula_imported") is False
    )
    wolfram_tests_pass = bool(
        payload.get("passed") is True
        and payload.get("tests")
        and all(test.get("passed") is True for test in payload["tests"])
    )
    wolfram_symbolic_checks_pass = bool(
        payload["canonicalization"]["coefficient_sum_preserved"] is True
        and payload["canonicalization"]["symbolic_match"] is True
        and payload["canonicalization"]["single_term_reduction_match"] is True
        and kernel["symbolic_match"] is True
        and kernel["alpha_one_backward_euler_match"] is True
        and quadratic["fractional_symbolic_match"] is True
        and quadratic["alpha_one_symbolic_match"] is True
        and quadratic["multi_term_symbolic_match"] is True
    )
    canonical_contract_match = bool(
        np.array_equal(result.original_orders, original_orders)
        and np.array_equal(result.original_coefficients, original_coefficients)
        and np.array_equal(result.orders, canonical_orders)
        and np.array_equal(result.coefficients, canonical_coefficients)
        and result.terms.source_indices == expected_source_indices
        and result.terms.zero_coefficient_indices == expected_zero_indices
        and result.terms.duplicate_terms_coalesced == 1
        and result.terms.zero_terms_removed == 1
        and result.terms.coefficient_sum == float(parameters["coefficient_sum"])
        and result.normalization == "none"
        and result.solver_info["coefficient_normalization"] == "none"
    )
    facade_contract_match = bool(
        result.status == "ok"
        and result.method == "multi_term_caputo_l1"
        and result.definition == "caputo_multi_term_finite_sum"
        and result.measure_kind == "finite_discrete_atomic_order_measure"
        and result.backend == "python_numpy_combined_l1_picard"
        and result.memory_policy == "full_history"
        and result.solver_info["continuous_order_quadrature_used"] is False
        and result.solver_info["continuous_order_density_inferred"] is False
        and result.solver_info["underlying_method"]
        == "distributed_order_caputo_l1"
        and result.solver_info["n_steps_completed"] == n_steps
    )

    cross_implementation_max_diff = max(
        kernel_metrics["max_diff"],
        trajectory_metrics["max_diff"],
        quadratic_probe_diff,
    )
    passed = bool(
        source_anchors_match
        and independence_flags_match
        and wolfram_tests_pass
        and wolfram_symbolic_checks_pass
        and canonical_contract_match
        and facade_contract_match
        and cross_implementation_max_diff <= float(tolerance)
    )
    return {
        "validation_scope": (
            "independent_Wolfram_to_HAFO_multi_term_Caputo_L1_"
            "finite_consistency"
        ),
        "evidence_boundary": payload["evidence_boundary"],
        "summary_path": str(summary_path),
        "tolerance": float(tolerance),
        "source_anchors_match": source_anchors_match,
        "independence_flags_match": independence_flags_match,
        "wolfram_tests_pass": wolfram_tests_pass,
        "wolfram_symbolic_checks_pass": wolfram_symbolic_checks_pass,
        "canonical_contract_match": canonical_contract_match,
        "facade_contract_match": facade_contract_match,
        "python_backend": result.backend,
        "kernel_metrics": kernel_metrics,
        "trajectory_metrics": trajectory_metrics,
        "quadratic_probe_diff": quadratic_probe_diff,
        "cross_implementation_max_diff": cross_implementation_max_diff,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--tolerance", type=float, default=NUMERIC_TOLERANCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination for the Python comparison JSON.",
    )
    args = parser.parse_args()
    result = compare_wolfram_summary(args.summary, tolerance=args.tolerance)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

