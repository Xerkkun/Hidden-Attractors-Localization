"""Compare HAFO's variable-order Caputo Type III L1 solver with Wolfram.

The independent Wolfram case fixes the Type III convention (the current
``alpha(t_n)`` is frozen throughout the history integral), derives L1
coefficients from a symbolic kernel antiderivative, and advances a finite
manufactured recurrence.  This comparator recomputes the same finite data
through HAFO's public weight and solver APIs.

Passing is finite algebraic/numerical consistency evidence only.  It is not a
global convergence theorem and is not evidence of stability, chaos,
attraction, or hiddenness.
"""

from __future__ import annotations

import argparse
import json
from math import gamma
from pathlib import Path
from typing import Any

import numpy as np

from hidden_attractors.fractional.variable_order_caputo_type3 import (
    integrate_variable_order_caputo_type3_l1,
    variable_order_l1_weight,
)


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_ID = "variable_order_caputo_type3_l1"
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
    "tavares_almeida_torres_doi": "10.1016/j.cnsns.2015.10.027",
    "fang_sun_wang_doi": "10.1016/j.camwa.2020.07.009",
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


def _order_function(time: float) -> float:
    return 0.5 + (time - 0.75) ** 2 / 10.0


def _power_derivative(time: float, order: float, power: float) -> float:
    if time == 0.0:
        return 0.0
    return gamma(power + 1.0) / gamma(power + 1.0 - order) * time ** (
        power - order
    )


def _sampled_l1_derivative(
    samples: np.ndarray,
    orders: np.ndarray,
    step: float,
) -> np.ndarray:
    derivative = np.zeros_like(samples, dtype=np.float64)
    for output_index in range(1, samples.size):
        alpha = float(orders[output_index])
        weighted_increments = 0.0
        for history_index in range(output_index):
            lag_index = output_index - history_index - 1
            weighted_increments += variable_order_l1_weight(
                alpha, lag_index
            ) * (samples[history_index + 1] - samples[history_index])
        derivative[output_index] = (
            step ** (-alpha)
            / gamma(2.0 - alpha)
            * weighted_increments
        )
    return derivative


def compare_wolfram_summary(
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    tolerance: float = NUMERIC_TOLERANCE,
) -> dict[str, Any]:
    """Recompute the Wolfram Type III L1 case with public HAFO APIs."""

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
    initial_value = float(parameters["initial_value"])
    power = float(parameters["power"])
    constant_order = float(parameters["constant_order"])
    if lower_terminal != 0.0:
        raise ValueError("the independent power oracle requires lower_terminal=0")

    wolfram_orders = np.asarray(parameters["orders"], dtype=np.float64)
    expected_times = lower_terminal + step * np.arange(n_steps + 1)
    python_orders = np.asarray(
        [_order_function(float(time)) for time in expected_times],
        dtype=np.float64,
    )

    weights = payload["weights"]
    probe_index = int(weights["probe_output_index"])
    probe_order = float(weights["probe_order"])
    public_probe_weights = np.asarray(
        [
            step ** (-probe_order)
            / gamma(2.0 - probe_order)
            * variable_order_l1_weight(
                probe_order, probe_index - history_index - 1
            )
            for history_index in range(probe_index)
        ],
        dtype=np.float64,
    )
    weight_metrics = {
        "public_to_wolfram_formula_max_diff": _max_abs(
            public_probe_weights, weights["formula_values"]
        ),
        "public_to_wolfram_symbolic_integral_max_diff": _max_abs(
            public_probe_weights, weights["symbolic_integral_values"]
        ),
        "wolfram_integral_numeric_max_residual": abs(
            float(weights["numeric_max_residual"])
        ),
    }
    weight_metrics["max_diff"] = max(weight_metrics.values())

    power_formula = payload["power_formula"]
    exact_power_derivative = np.asarray(
        [
            _power_derivative(float(time), float(alpha), power)
            for time, alpha in zip(expected_times, python_orders, strict=True)
        ]
    )
    power_samples = initial_value + expected_times**power
    public_l1_derivative = _sampled_l1_derivative(
        power_samples, python_orders, step
    )
    power_metrics = {
        "order_grid_python_to_wolfram_max_diff": _max_abs(
            python_orders, wolfram_orders
        ),
        "exact_formula_python_to_wolfram_max_diff": _max_abs(
            exact_power_derivative, power_formula["exact_derivative_values"]
        ),
        "sampled_l1_python_to_wolfram_max_diff": _max_abs(
            public_l1_derivative, power_formula["l1_derivative_values"]
        ),
        "finite_grid_error": _max_abs(
            public_l1_derivative, exact_power_derivative
        ),
    }
    power_metrics["finite_grid_error_report_diff"] = abs(
        power_metrics["finite_grid_error"]
        - float(power_formula["l1_max_finite_grid_error"])
    )

    def manufactured_rhs(time: float, state: np.ndarray) -> np.ndarray:
        return np.full_like(
            state,
            _power_derivative(time, _order_function(time), power),
        )

    variable_result = integrate_variable_order_caputo_type3_l1(
        manufactured_rhs,
        [initial_value],
        step=step,
        n_steps=n_steps,
        lower_terminal=lower_terminal,
        order_function=_order_function,
        order_function_name="alpha(t)=1/2+(t-3/4)^2/10",
        initial_regularity="smooth",
        use_acceleration=False,
        divergence_norm=None,
    )
    manufactured = payload["manufactured_case"]
    exact_states = initial_value + variable_result.times**power
    trajectory_metrics = {
        "time_grid_python_to_wolfram_max_diff": _max_abs(
            variable_result.times, manufactured["times"]
        ),
        "order_grid_solver_to_wolfram_max_diff": _max_abs(
            variable_result.orders, manufactured["orders"]
        ),
        "states_python_to_wolfram_max_diff": _max_abs(
            variable_result.states[:, 0], manufactured["states"]
        ),
        "exact_states_python_to_wolfram_max_diff": _max_abs(
            exact_states, manufactured["exact_states"]
        ),
        "finite_grid_error": _max_abs(
            variable_result.states[:, 0], exact_states
        ),
    }
    trajectory_metrics["finite_grid_error_report_diff"] = abs(
        trajectory_metrics["finite_grid_error"]
        - float(manufactured["max_finite_grid_error"])
    )

    def constant_rhs(time: float, state: np.ndarray) -> np.ndarray:
        return np.full_like(
            state,
            _power_derivative(time, constant_order, power),
        )

    constant_result = integrate_variable_order_caputo_type3_l1(
        constant_rhs,
        [initial_value],
        step=step,
        n_steps=n_steps,
        lower_terminal=lower_terminal,
        order_function=lambda time: constant_order,
        order_function_name=f"constant alpha={constant_order:.17g}",
        initial_regularity="smooth",
        use_acceleration=False,
        divergence_norm=None,
    )
    constant = payload["constant_order_reduction"]
    constant_metrics = {
        "solver_to_variable_path_max_diff": _max_abs(
            constant_result.states[:, 0], constant["variable_order_path_states"]
        ),
        "solver_to_constant_reference_max_diff": _max_abs(
            constant_result.states[:, 0], constant["constant_order_reference_states"]
        ),
        "wolfram_trajectory_max_residual": abs(
            float(constant["trajectory_max_residual"])
        ),
        "wolfram_weight_max_residual": abs(
            float(constant["weight_max_residual"])
        ),
    }
    constant_metrics["max_diff"] = max(constant_metrics.values())

    cross_implementation_max_diff = max(
        weight_metrics["max_diff"],
        power_metrics["order_grid_python_to_wolfram_max_diff"],
        power_metrics["exact_formula_python_to_wolfram_max_diff"],
        power_metrics["sampled_l1_python_to_wolfram_max_diff"],
        power_metrics["finite_grid_error_report_diff"],
        trajectory_metrics["time_grid_python_to_wolfram_max_diff"],
        trajectory_metrics["order_grid_solver_to_wolfram_max_diff"],
        trajectory_metrics["states_python_to_wolfram_max_diff"],
        trajectory_metrics["exact_states_python_to_wolfram_max_diff"],
        trajectory_metrics["finite_grid_error_report_diff"],
        constant_metrics["max_diff"],
    )

    source = payload["source"]
    source_anchors_match = all(
        source.get(key) == expected for key, expected in SOURCE_ANCHORS.items()
    )
    independence_flags_match = bool(
        source.get("hafo_source_read") is False
        and source.get("report_input_used") is False
    )
    wolfram_tests_pass = bool(
        payload.get("passed") is True
        and payload.get("tests")
        and all(test.get("passed") is True for test in payload["tests"])
    )
    symbolic_checks_pass = bool(
        power_formula["symbolic_match"] is True
        and weights["symbolic_match"] is True
    )
    solver_contract_match = bool(
        variable_result.status == constant_result.status == "ok"
        and variable_result.method == constant_result.method
        == "vo_caputo_type3_l1"
        and variable_result.backend == constant_result.backend
        == "python_numpy_l1_picard"
        and variable_result.memory_policy == constant_result.memory_policy
        == "full_history"
        and variable_result.solver_info["definition"]
        == "tavares_type_iii_current_time"
    )
    passed = bool(
        source_anchors_match
        and independence_flags_match
        and wolfram_tests_pass
        and symbolic_checks_pass
        and solver_contract_match
        and cross_implementation_max_diff <= float(tolerance)
    )
    return {
        "validation_scope": (
            "independent_Wolfram_to_HAFO_variable_order_Type_III_L1_"
            "finite_consistency"
        ),
        "evidence_boundary": payload["evidence_boundary"],
        "summary_path": str(summary_path),
        "tolerance": float(tolerance),
        "source_anchors_match": source_anchors_match,
        "independence_flags_match": independence_flags_match,
        "wolfram_tests_pass": wolfram_tests_pass,
        "symbolic_checks_pass": symbolic_checks_pass,
        "solver_contract_match": solver_contract_match,
        "python_backend": variable_result.backend,
        "weight_metrics": weight_metrics,
        "power_metrics": power_metrics,
        "trajectory_metrics": trajectory_metrics,
        "constant_order_metrics": constant_metrics,
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
