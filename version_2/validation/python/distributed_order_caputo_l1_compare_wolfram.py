"""Compare HAFO's distributed-order Caputo L1 solver with Wolfram.

The Wolfram case independently integrates the Caputo power-law kernel for
three discrete order masses and advances a directly solved linear recurrence
for an affine manufactured solution.  It does not read HAFO source or import
HAFO's coefficient formula.  This comparator reaches HAFO only through the
public weight and solver APIs.

Passing is finite algebraic/numerical consistency evidence only.  It is not a
global convergence theorem and is not evidence of nonlinear stability, chaos,
attraction, or hiddenness.
"""

from __future__ import annotations

import argparse
import json
from math import gamma
from pathlib import Path
from typing import Any

import numpy as np

from hidden_attractors.fractional.distributed_order_caputo_solver import (
    distributed_order_l1_weight,
    integrate_distributed_order_caputo_l1,
)


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_ID = "distributed_order_caputo_l1"
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
    "caputo_distributed_order_index": (
        "https://www.math.bas.bg/complan/fcaa/volume4/index.html"
    ),
    "diethelm_ford_doi": "10.1016/j.cam.2008.07.018",
    "hu_liu_anh_turner_doi": "10.21914/ANZIAMJ.V55I0.7888",
    "lin_xu_doi": "10.1016/j.jcp.2007.02.001",
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


def _exact_state(
    time: float,
    *,
    lower_terminal: float,
    initial_value: float,
    slope: float,
) -> float:
    return initial_value + slope * (time - lower_terminal)


def _distributed_affine_caputo_derivative(
    time: float,
    *,
    lower_terminal: float,
    slope: float,
    order_nodes: np.ndarray,
    order_masses: np.ndarray,
) -> float:
    elapsed = max(0.0, float(time) - lower_terminal)
    if elapsed == 0.0:
        return 0.0
    return float(
        sum(
            mass
            * slope
            * elapsed ** (1.0 - order)
            / gamma(2.0 - order)
            for order, mass in zip(
                order_nodes, order_masses, strict=True
            )
        )
    )


def _public_per_order_kernel(
    order_nodes: np.ndarray,
    order_masses: np.ndarray,
    *,
    step: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.asarray(
        [
            mass * step ** (-order) / gamma(2.0 - order)
            for order, mass in zip(
                order_nodes, order_masses, strict=True
            )
        ],
        dtype=np.float64,
    )
    kernel = np.asarray(
        [
            [
                coefficient
                * distributed_order_l1_weight(float(order), lag)
                for lag in range(n_steps)
            ]
            for order, coefficient in zip(
                order_nodes, coefficients, strict=True
            )
        ],
        dtype=np.float64,
    )
    return coefficients, kernel


def _recurrence_residuals(
    states: np.ndarray,
    combined_kernel: np.ndarray,
    times: np.ndarray,
    *,
    forcing: np.ndarray,
    linear_coefficient: float,
) -> np.ndarray:
    residuals = np.zeros(states.size - 1, dtype=np.float64)
    for output_index in range(1, states.size):
        lhs = 0.0
        for lag in range(output_index):
            lhs += combined_kernel[lag] * (
                states[output_index - lag]
                - states[output_index - lag - 1]
            )
        residuals[output_index - 1] = (
            lhs
            - linear_coefficient * states[output_index]
            - forcing[output_index]
        )
    if times.size != states.size:
        raise ValueError("time and state grids must have equal length")
    return residuals


def compare_wolfram_summary(
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    tolerance: float = NUMERIC_TOLERANCE,
) -> dict[str, Any]:
    """Recompute the independent multinode case with public HAFO APIs."""

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
    order_nodes = np.asarray(parameters["order_nodes"], dtype=np.float64)
    order_masses = np.asarray(parameters["order_masses"], dtype=np.float64)
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
            _distributed_affine_caputo_derivative(
                float(time),
                lower_terminal=lower_terminal,
                slope=slope,
                order_nodes=order_nodes,
                order_masses=order_masses,
            )
            for time in expected_times
        ],
        dtype=np.float64,
    )
    forcing = exact_derivative - linear_coefficient * exact_states

    public_coefficients, public_per_order_kernel = _public_per_order_kernel(
        order_nodes,
        order_masses,
        step=step,
        n_steps=n_steps,
    )
    public_combined_kernel = np.sum(public_per_order_kernel, axis=0)

    kernel = payload["kernel"]
    kernel_metrics = {
        "per_order_public_to_integrated_max_diff": _max_abs(
            public_per_order_kernel,
            kernel["per_order_integrated_values"],
        ),
        "per_order_public_to_formula_max_diff": _max_abs(
            public_per_order_kernel,
            kernel["per_order_formula_values"],
        ),
        "combined_public_to_integrated_max_diff": _max_abs(
            public_combined_kernel,
            kernel["combined_integrated_values"],
        ),
        "combined_public_to_formula_max_diff": _max_abs(
            public_combined_kernel,
            kernel["combined_formula_values"],
        ),
        "current_coefficient_max_diff": abs(
            float(np.sum(public_coefficients))
            - float(kernel["current_step_coefficient"])
        ),
        "wolfram_integral_numeric_max_residual": abs(
            float(kernel["numeric_max_residual"])
        ),
    }
    kernel_metrics["max_diff"] = max(kernel_metrics.values())

    manufactured = payload["manufactured_case"]
    reference_metrics = {
        "time_grid_python_to_wolfram_max_diff": _max_abs(
            expected_times, manufactured["times"]
        ),
        "forcing_python_to_wolfram_max_diff": _max_abs(
            forcing, manufactured["forcing_values"]
        ),
        "exact_states_python_to_wolfram_max_diff": _max_abs(
            exact_states, manufactured["exact_states"]
        ),
        "wolfram_states_to_exact_max_diff": _max_abs(
            manufactured["states"], exact_states
        ),
        "wolfram_reported_exact_error": abs(
            float(manufactured["max_exact_error"])
        ),
        "wolfram_reported_recurrence_residual": abs(
            float(manufactured["max_recurrence_residual"])
        ),
    }
    reference_metrics["max_diff"] = max(reference_metrics.values())

    def manufactured_rhs(time: float, state: np.ndarray) -> np.ndarray:
        exact_at_time = _exact_state(
            time,
            lower_terminal=lower_terminal,
            initial_value=initial_value,
            slope=slope,
        )
        derivative_at_time = _distributed_affine_caputo_derivative(
            time,
            lower_terminal=lower_terminal,
            slope=slope,
            order_nodes=order_nodes,
            order_masses=order_masses,
        )
        forcing_at_time = (
            derivative_at_time - linear_coefficient * exact_at_time
        )
        return linear_coefficient * state + forcing_at_time

    result = integrate_distributed_order_caputo_l1(
        manufactured_rhs,
        [initial_value],
        order_nodes=order_nodes,
        order_weights=order_masses,
        step=step,
        n_steps=n_steps,
        lower_terminal=lower_terminal,
        weight_semantics="nonnegative_mass",
        normalization="none",
        order_quadrature_name="independent_discrete_mass_rule",
        corrector_atol=1.0e-14,
        corrector_rtol=1.0e-14,
        corrector_max_iterations=100,
        initial_regularity="smooth",
        use_acceleration=False,
        divergence_norm=None,
    )

    solver_states = result.states[:, 0]
    solver_recurrence_residuals = _recurrence_residuals(
        solver_states,
        result.combined_l1_kernel,
        result.times,
        forcing=forcing,
        linear_coefficient=linear_coefficient,
    )
    solver_metrics = {
        "order_nodes_max_diff": _max_abs(result.order_nodes, order_nodes),
        "order_masses_max_diff": _max_abs(
            result.effective_weights, order_masses
        ),
        "l1_coefficients_max_diff": _max_abs(
            result.l1_coefficients, public_coefficients
        ),
        "combined_kernel_max_diff": _max_abs(
            result.combined_l1_kernel, public_combined_kernel
        ),
        "time_grid_python_to_wolfram_max_diff": _max_abs(
            result.times, manufactured["times"]
        ),
        "states_python_to_wolfram_max_diff": _max_abs(
            solver_states, manufactured["states"]
        ),
        "states_python_to_exact_max_diff": _max_abs(
            solver_states, exact_states
        ),
        "max_discrete_recurrence_residual": float(
            np.max(np.abs(solver_recurrence_residuals), initial=0.0)
        ),
    }
    solver_metrics["max_diff"] = max(solver_metrics.values())

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
    symbolic_checks_pass = bool(
        kernel["per_order_symbolic_match"] is True
        and kernel["combined_symbolic_match"] is True
        and payload["affine_caputo_identity"]["symbolic_match"] is True
    )
    kernel_shape_checks_pass = bool(
        kernel["positive"] is True
        and kernel["strictly_decreasing"] is True
    )
    solver_contract_match = bool(
        result.status == "ok"
        and result.method == "distributed_order_caputo_l1"
        and result.backend == "python_numpy_combined_l1_picard"
        and result.memory_policy == "full_history"
        and result.weight_semantics == "nonnegative_mass"
        and result.normalization == "none"
        and result.solver_info["definition"]
        == "caputo_distributed_order_discrete_measure"
        and result.solver_info["n_steps_completed"] == n_steps
    )

    cross_implementation_max_diff = max(
        kernel_metrics["max_diff"],
        reference_metrics["max_diff"],
        solver_metrics["max_diff"],
    )
    passed = bool(
        source_anchors_match
        and independence_flags_match
        and wolfram_tests_pass
        and symbolic_checks_pass
        and kernel_shape_checks_pass
        and solver_contract_match
        and cross_implementation_max_diff <= float(tolerance)
    )
    return {
        "validation_scope": (
            "independent_Wolfram_to_HAFO_distributed_order_Caputo_L1_"
            "finite_consistency"
        ),
        "evidence_boundary": payload["evidence_boundary"],
        "summary_path": str(summary_path),
        "tolerance": float(tolerance),
        "source_anchors_match": source_anchors_match,
        "independence_flags_match": independence_flags_match,
        "wolfram_tests_pass": wolfram_tests_pass,
        "symbolic_checks_pass": symbolic_checks_pass,
        "kernel_shape_checks_pass": kernel_shape_checks_pass,
        "solver_contract_match": solver_contract_match,
        "python_backend": result.backend,
        "kernel_metrics": kernel_metrics,
        "reference_metrics": reference_metrics,
        "solver_metrics": solver_metrics,
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
