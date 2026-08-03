"""Compare HAFO's ABC PCM with an independent Wolfram recurrence.

The Wolfram case derives both linear product-integration weights with
``Integrate`` and evaluates one finite Lee--Kim--Jang recurrence for the
manufactured scalar right-hand side ``f(t, u) = t**2`` with ``B(alpha)=1``.
It does not read HAFO source code or generated report data.

Passing is finite algebraic/numerical consistency evidence.  It is not a
convergence theorem and is not evidence of stability, chaos, attraction, or
hiddenness.
"""

from __future__ import annotations

import argparse
from math import gamma
import json
from pathlib import Path
from typing import Any

import numpy as np

from hidden_attractors.fractional.abc_solver import (
    abc_linear_product_weights,
    integrate_abc_predictor_corrector,
)


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_ID = "abc_predictor_corrector"
DEFAULT_SUMMARY = (
    ROOT
    / "validation"
    / "outputs"
    / "wolfram"
    / SYSTEM_ID
    / f"{SYSTEM_ID}_validation_summary.json"
)
NUMERIC_TOLERANCE = 5.0e-12

SOURCE_ANCHORS = {
    "lee_kim_jang_doi": "10.3390/fractalfract8010065",
    "atangana_baleanu_doi": "10.2298/TSCI160111018A",
    "diethelm_garrappa_giusti_stynes_doi": "10.1515/fca-2020-0032",
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


def _manufactured_exact(
    times: np.ndarray,
    *,
    initial_value: float,
    order: float,
    lower_terminal: float,
    normalization: float,
) -> np.ndarray:
    elapsed = times - lower_terminal
    local = (1.0 - order) * elapsed**2 / normalization
    memory = (
        order
        * gamma(3.0)
        * elapsed ** (order + 2.0)
        / (normalization * gamma(order + 3.0))
    )
    return initial_value + local + memory


def compare_wolfram_summary(
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    tolerance: float = NUMERIC_TOLERANCE,
) -> dict[str, Any]:
    """Recompute the independent finite ABC PCM case with public HAFO APIs."""

    summary_path = Path(summary_path).resolve()
    payload = _load(summary_path)
    if payload.get("system_id") != SYSTEM_ID:
        raise ValueError(
            f"unexpected system_id {payload.get('system_id')!r}; "
            f"expected {SYSTEM_ID!r}"
        )

    parameters = payload["parameters"]
    alpha = float(parameters["alpha"])
    step = float(parameters["step"])
    n_steps = int(parameters["n_steps"])
    lower_terminal = float(parameters["lower_terminal"])
    initial_value = float(parameters["initial_value"])
    normalization = float(parameters["normalization"])

    wolfram_weights = payload["weights"]
    theta0, theta1 = abc_linear_product_weights(alpha, step, n_steps)
    weight_metrics = {
        "theta0_python_to_wolfram_formula_max_diff": _max_abs(
            theta0[1:], wolfram_weights["theta0_formula"]
        ),
        "theta1_python_to_wolfram_formula_max_diff": _max_abs(
            theta1[1:], wolfram_weights["theta1_formula"]
        ),
        "theta0_python_to_symbolic_integral_max_diff": _max_abs(
            theta0[1:], wolfram_weights["theta0_symbolic_integral"]
        ),
        "theta1_python_to_symbolic_integral_max_diff": _max_abs(
            theta1[1:], wolfram_weights["theta1_symbolic_integral"]
        ),
        "wolfram_integral_max_residual": abs(
            float(wolfram_weights["integral_max_residual"])
        ),
        "wolfram_partition_max_residual": abs(
            float(wolfram_weights["partition_max_residual"])
        ),
    }
    weight_metrics["max_diff"] = max(weight_metrics.values())

    def quadratic_rhs(time: float, state: np.ndarray) -> np.ndarray:
        return np.full_like(state, (time - lower_terminal) ** 2)

    python_result = integrate_abc_predictor_corrector(
        quadratic_rhs,
        [initial_value],
        alpha,
        step=step,
        n_steps=n_steps,
        lower_terminal=lower_terminal,
        normalization=normalization,
        normalization_name="B(alpha)=1",
        use_acceleration=False,
        divergence_norm=None,
    )
    wolfram_case = payload["manufactured_case"]
    exact = _manufactured_exact(
        python_result.times,
        initial_value=initial_value,
        order=alpha,
        lower_terminal=lower_terminal,
        normalization=normalization,
    )
    wolfram_states = np.asarray(wolfram_case["states"], dtype=np.float64)
    trajectory_metrics = {
        "time_grid_python_to_wolfram_max_diff": _max_abs(
            python_result.times, wolfram_case["times"]
        ),
        "states_python_to_wolfram_max_diff": _max_abs(
            python_result.states[:, 0], wolfram_states
        ),
        "wolfram_exact_volterra_to_python_exact_max_diff": _max_abs(
            exact, wolfram_case["exact_volterra_states"]
        ),
        "python_finite_grid_error": _max_abs(
            python_result.states[:, 0], exact
        ),
        "wolfram_finite_grid_error": _max_abs(
            wolfram_states, wolfram_case["exact_volterra_states"]
        ),
        "wolfram_recurrence_direct_max_residual": abs(
            float(wolfram_case["recurrence_direct_max_residual"])
        ),
    }
    cross_implementation_max_diff = max(
        weight_metrics["max_diff"],
        trajectory_metrics["time_grid_python_to_wolfram_max_diff"],
        trajectory_metrics["states_python_to_wolfram_max_diff"],
        trajectory_metrics["wolfram_exact_volterra_to_python_exact_max_diff"],
        trajectory_metrics["wolfram_recurrence_direct_max_residual"],
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
    solver_contract_match = bool(
        python_result.status == "ok"
        and python_result.backend == "python_abc_pcm_full_history"
        and python_result.normalization_value == normalization
        and python_result.startup_iterations
        == int(wolfram_case["startup_iterations"])
        and wolfram_case["startup_converged"] is True
    )
    symbolic_integrals_pass = bool(wolfram_weights["symbolic_match"])
    passed = bool(
        source_anchors_match
        and independence_flags_match
        and wolfram_tests_pass
        and solver_contract_match
        and symbolic_integrals_pass
        and cross_implementation_max_diff <= float(tolerance)
    )
    return {
        "validation_scope": (
            "independent_Wolfram_to_HAFO_ABC_PCM_finite_consistency"
        ),
        "evidence_boundary": payload["evidence_boundary"],
        "summary_path": str(summary_path),
        "tolerance": float(tolerance),
        "source_anchors_match": source_anchors_match,
        "independence_flags_match": independence_flags_match,
        "wolfram_tests_pass": wolfram_tests_pass,
        "solver_contract_match": solver_contract_match,
        "symbolic_integrals_pass": symbolic_integrals_pass,
        "python_backend": python_result.backend,
        "python_status": python_result.status,
        "weight_metrics": weight_metrics,
        "trajectory_metrics": trajectory_metrics,
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
