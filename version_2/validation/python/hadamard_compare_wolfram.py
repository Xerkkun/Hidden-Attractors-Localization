"""Compare HAFO Hadamard operators with an independent Wolfram artifact.

The Wolfram case derives the logarithmic coordinate transformation, Gamma/Beta
identities, and BDF1/BDF2 weights without importing or reading HAFO.  This
module loads only its exported JSON values and recomputes the sampled operators
and manufactured Caputo--Hadamard trajectory through the public Python API.

Passing is finite-grid implementation evidence.  It does not establish
stability, chaos, hidden-attractor existence, or convergence for arbitrary
nonlinear fractional differential equations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gamma

from hidden_attractors.fractional import (
    CAPUTO_HADAMARD_INITIAL_CONDITION,
    HADAMARD_RL_OPERATOR_ONLY_INITIAL_CONDITION,
    hadamard_convolution_quadrature,
    integrate_caputo_hadamard_abm,
    lubich_bdf_weights,
)


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_ID = "hadamard_fractional_operator"
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
    "jarad_abdeljawad_baleanu_doi": "10.1186/1687-1847-2012-142",
    "yin_zhang_liu_li_doi": "10.1016/j.cnsns.2024.108221",
    "zheng_doi": "10.1016/j.aml.2021.107366",
    "green_liu_yan_doi": "10.3390/math9212728",
    "lubich_doi": "10.1137/0517050",
    "diethelm_ford_freed_doi": "10.1023/B:NUMA.0000027736.85078.be",
}


def _load(path: Path) -> dict[str, Any]:
    """Load one summary and reject missing or non-object JSON artifacts."""

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


def _compare_sample_case(payload: dict[str, Any]) -> dict[str, Any]:
    parameters = payload["parameters"]
    sample = payload["cq"]["sample_case"]
    order = float(parameters["order"])
    lower_terminal = float(parameters["lower_terminal"])
    physical_times = np.asarray(sample["physical_times"], dtype=np.float64)
    samples = np.asarray(sample["samples"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    metrics: list[float] = []

    for wolfram_row in sample["rows"]:
        bdf_order = int(round(float(wolfram_row["bdf_order"])))
        raw = hadamard_convolution_quadrature(
            samples,
            order,
            bdf_order=bdf_order,
            definition="hadamard_riemann_liouville",
            times=physical_times,
            lower_terminal=lower_terminal,
            initial_condition_semantics=(
                HADAMARD_RL_OPERATOR_ONLY_INITIAL_CONDITION
            ),
            backend="python",
        )
        shifted = hadamard_convolution_quadrature(
            samples,
            order,
            bdf_order=bdf_order,
            definition="caputo_hadamard",
            times=physical_times,
            lower_terminal=lower_terminal,
            initial_condition_semantics=CAPUTO_HADAMARD_INITIAL_CONDITION,
            backend="python",
        )
        shifted_constant = hadamard_convolution_quadrature(
            np.full_like(samples, samples[0]),
            order,
            bdf_order=bdf_order,
            definition="caputo_hadamard",
            times=physical_times,
            lower_terminal=lower_terminal,
            initial_condition_semantics=CAPUTO_HADAMARD_INITIAL_CONDITION,
            backend="python",
        )
        weight_diff = _max_abs(raw.weights[:, 0], wolfram_row["weights"])
        raw_diff = _max_abs(raw.values, wolfram_row["raw_values"])
        shifted_diff = _max_abs(
            shifted.values, wolfram_row["caputo_shifted_values"]
        )
        log_grid_diff = _max_abs(raw.log_times, sample["log_times"])
        physical_grid_diff = _max_abs(raw.times, physical_times)
        shifted_sample_diff = _max_abs(
            samples - samples[0], sample["shifted_samples"]
        )
        transformed_grid_diff = _max_abs(
            lower_terminal * np.exp(np.asarray(sample["log_times"], dtype=float)),
            physical_times,
        )
        shifted_constant_diff = abs(
            float(np.max(np.abs(shifted_constant.values), initial=0.0))
            - float(wolfram_row["caputo_constant_max_abs"])
        )
        row_metrics = [
            weight_diff,
            raw_diff,
            shifted_diff,
            log_grid_diff,
            physical_grid_diff,
            shifted_sample_diff,
            transformed_grid_diff,
            shifted_constant_diff,
        ]
        metrics.extend(row_metrics)
        rows.append(
            {
                "bdf_order": bdf_order,
                "weight_max_diff": weight_diff,
                "raw_value_max_diff": raw_diff,
                "caputo_shifted_max_diff": shifted_diff,
                "log_grid_max_diff": log_grid_diff,
                "physical_grid_max_diff": physical_grid_diff,
                "shifted_sample_max_diff": shifted_sample_diff,
                "transformed_grid_max_diff": transformed_grid_diff,
                "caputo_constant_max_diff": shifted_constant_diff,
                "max_diff": max(row_metrics),
            }
        )

    return {"rows": rows, "max_diff": max(metrics, default=0.0)}


def _compare_convergence_rows(payload: dict[str, Any]) -> dict[str, Any]:
    parameters = payload["parameters"]
    order = float(parameters["order"])
    lower_terminal = float(parameters["lower_terminal"])
    degree = int(round(float(parameters["log_power_degree"])))
    constant_value = float(parameters["constant_value"])
    analytic_log_power = float(gamma(degree + 1.0) / gamma(degree + 1.0 - order))
    analytic_raw_constant = float(constant_value / gamma(1.0 - order))
    rows: list[dict[str, Any]] = []
    metrics: list[float] = []

    for wolfram_row in payload["cq"]["convergence_rows"]:
        bdf_order = int(round(float(wolfram_row["bdf_order"])))
        n_steps = int(round(float(wolfram_row["n_steps"])))
        log_times = np.arange(n_steps + 1, dtype=np.float64) / n_steps
        physical_times = lower_terminal * np.exp(log_times)
        shifted = hadamard_convolution_quadrature(
            log_times**degree,
            order,
            bdf_order=bdf_order,
            definition="caputo_hadamard",
            times=physical_times,
            lower_terminal=lower_terminal,
            initial_condition_semantics=CAPUTO_HADAMARD_INITIAL_CONDITION,
            backend="python",
        )
        raw_constant = hadamard_convolution_quadrature(
            np.full(n_steps + 1, constant_value, dtype=np.float64),
            order,
            bdf_order=bdf_order,
            definition="hadamard_riemann_liouville",
            times=physical_times,
            lower_terminal=lower_terminal,
            initial_condition_semantics=(
                HADAMARD_RL_OPERATOR_ONLY_INITIAL_CONDITION
            ),
            backend="python",
        )
        shifted_endpoint = float(shifted.values[-1])
        raw_endpoint = float(raw_constant.values[-1])
        shifted_error = abs(shifted_endpoint - analytic_log_power)
        raw_error = abs(raw_endpoint - analytic_raw_constant)
        row_metrics = [
            abs(shifted_endpoint - float(wolfram_row["caputo_log_power_endpoint"])),
            abs(raw_endpoint - float(wolfram_row["raw_constant_endpoint"])),
            abs(
                analytic_log_power
                - float(wolfram_row["caputo_log_power_analytic"])
            ),
            abs(
                analytic_raw_constant
                - float(wolfram_row["raw_constant_analytic"])
            ),
            abs(shifted_error - float(wolfram_row["caputo_log_power_abs_error"])),
            abs(raw_error - float(wolfram_row["raw_constant_abs_error"])),
        ]
        metrics.extend(row_metrics)
        rows.append(
            {
                "bdf_order": bdf_order,
                "n_steps": n_steps,
                "caputo_endpoint_diff": row_metrics[0],
                "raw_constant_endpoint_diff": row_metrics[1],
                "caputo_analytic_diff": row_metrics[2],
                "raw_constant_analytic_diff": row_metrics[3],
                "caputo_reported_error_diff": row_metrics[4],
                "raw_constant_reported_error_diff": row_metrics[5],
                "max_diff": max(row_metrics),
            }
        )

    return {"rows": rows, "max_diff": max(metrics, default=0.0)}


def _compare_q1_weights(payload: dict[str, Any]) -> dict[str, float]:
    cq = payload["cq"]
    wolfram_bdf1 = np.asarray(cq["q1_bdf1_weights"], dtype=np.float64)
    wolfram_bdf2 = np.asarray(cq["q1_bdf2_weights"], dtype=np.float64)
    python_bdf1 = lubich_bdf_weights(1.0, wolfram_bdf1.size, bdf_order=1)
    python_bdf2 = lubich_bdf_weights(1.0, wolfram_bdf2.size, bdf_order=2)
    return {
        "bdf1_max_diff": _max_abs(python_bdf1, wolfram_bdf1),
        "bdf2_max_diff": _max_abs(python_bdf2, wolfram_bdf2),
    }


def _compare_manufactured_abm(payload: dict[str, Any]) -> dict[str, Any]:
    case = payload["abm_manufactured"]
    order = float(case["order"])
    lower_terminal = float(case["lower_terminal"])
    initial_state = float(case["initial_state"])
    forcing = float(case["forcing"])
    log_step = float(case["log_step"])
    n_steps = int(round(float(case["n_steps"])))
    upper_terminal = lower_terminal * np.exp(n_steps * log_step)

    def constant_rhs(_time: float, state: np.ndarray) -> np.ndarray:
        return np.full_like(state, forcing)

    result = integrate_caputo_hadamard_abm(
        constant_rhs,
        [initial_state],
        order,
        lower_terminal=lower_terminal,
        upper_terminal=upper_terminal,
        log_step=log_step,
        use_acceleration=False,
        divergence_norm=None,
    )
    analytic_states = np.asarray(case["analytic_states"], dtype=np.float64)
    metrics = {
        "log_times_max_diff": _max_abs(result.log_times, case["log_times"]),
        "physical_times_max_diff": _max_abs(result.times, case["physical_times"]),
        "states_max_diff": _max_abs(result.states[:, 0], analytic_states),
        "analytic_formula_max_diff": _max_abs(
            initial_state
            + forcing
            * np.asarray(case["log_times"], dtype=np.float64) ** order
            / gamma(order + 1.0),
            analytic_states,
        ),
    }
    return {
        **metrics,
        "status": result.status,
        "backend": result.backend,
        "max_diff": max(metrics.values()),
    }


def compare_wolfram_summary(
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    tolerance: float = NUMERIC_TOLERANCE,
) -> dict[str, Any]:
    """Recompute exported Wolfram samples with HAFO's public APIs."""

    summary_path = Path(summary_path).resolve()
    payload = _load(summary_path)
    if payload.get("system_id") != SYSTEM_ID:
        raise ValueError(
            f"unexpected system_id {payload.get('system_id')!r}; "
            f"expected {SYSTEM_ID!r}"
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

    sample_case = _compare_sample_case(payload)
    convergence = _compare_convergence_rows(payload)
    q1_weights = _compare_q1_weights(payload)
    manufactured_abm = _compare_manufactured_abm(payload)
    worst_numeric_diff = max(
        sample_case["max_diff"],
        convergence["max_diff"],
        *q1_weights.values(),
        manufactured_abm["max_diff"],
    )
    passed = bool(
        wolfram_tests_pass
        and source_anchors_match
        and independence_flags_match
        and manufactured_abm["status"] == "ok"
        and worst_numeric_diff <= float(tolerance)
    )
    return {
        "validation_scope": (
            "independent_wolfram_to_hafo_hadamard_caputo_hadamard_consistency"
        ),
        "evidence_boundary": payload["evidence_boundary"],
        "summary_path": str(summary_path),
        "tolerance": float(tolerance),
        "source_anchors_match": source_anchors_match,
        "independence_flags_match": independence_flags_match,
        "wolfram_tests_pass": wolfram_tests_pass,
        "cq_sample_case": sample_case,
        "cq_convergence": convergence,
        "q1_weights": q1_weights,
        "manufactured_abm": manufactured_abm,
        "worst_numeric_diff": worst_numeric_diff,
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
        help="Optional JSON destination for the Python consistency summary.",
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
