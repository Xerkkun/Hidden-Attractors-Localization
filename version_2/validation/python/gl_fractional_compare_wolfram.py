"""Compare HAFO's GL implementation with an independent Wolfram artifact.

The Wolfram case derives the signed-binomial weights and scalar recurrences
without importing or reading HAFO.  This module performs the opposite half of
the cross-check: it loads only the exported JSON numbers and recomputes every
sample with the public Python/Numba implementation.

Source anchors shared with the independent case:

* Podlubny, *Fractional Differential Equations* (1999),
  ISBN 978-0-12-558840-9.
* Lubich, "Discretized Fractional Calculus" (1986),
  DOI 10.1137/0517050.

This is finite-grid method validation.  Passing does not establish nonlinear
stability, chaos, or hidden-attractor existence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit
from scipy.special import gamma

from hidden_attractors.fractional.gl_solver import integrate_gl_explicit_numba
from hidden_attractors.fractional.grunwald_letnikov import (
    grunwald_letnikov_derivative,
    grunwald_letnikov_weights,
)


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_ID = "gl_fractional_operator_validation"
DEFAULT_SUMMARY = (
    ROOT
    / "validation"
    / "outputs"
    / "wolfram"
    / SYSTEM_ID
    / f"{SYSTEM_ID}_summary.json"
)
NUMERIC_TOLERANCE = 5.0e-12


@njit
def _constant_forcing_rhs(
    _time: float, state: np.ndarray, _parameters: np.ndarray
) -> np.ndarray:
    output = np.empty_like(state)
    for index in range(state.size):
        output[index] = 1.0
    return output


@njit
def _linear_rhs(
    _time: float, state: np.ndarray, parameters: np.ndarray
) -> np.ndarray:
    output = np.empty_like(state)
    for index in range(state.size):
        output[index] = parameters[0] * state[index]
    return output


def _load(path: Path) -> dict[str, Any]:
    """Load one Wolfram summary and reject non-object JSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _max_abs(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"shape mismatch in Wolfram/Python comparison: "
            f"{left_array.shape} != {right_array.shape}"
        )
    return float(np.max(np.abs(left_array - right_array), initial=0.0))


def _compare_monomial(payload: dict[str, Any], order: float) -> dict[str, Any]:
    section = payload["monomial_caputo_shifted"]
    degree = int(round(float(payload["parameters"]["monomial_degree"])))
    value_diffs: list[float] = []
    analytic_diffs: list[float] = []
    error_diffs: list[float] = []
    rows: list[dict[str, float | int]] = []
    analytic = float(gamma(degree + 1.0) / gamma(degree + 1.0 - order))

    for wolfram_row in section["rows"]:
        n_steps = int(round(float(wolfram_row["n_steps"])))
        step = 1.0 / n_steps
        times = np.arange(n_steps + 1, dtype=float) * step
        samples = times**degree
        result = grunwald_letnikov_derivative(
            samples,
            step,
            order,
            definition="caputo_shifted",
        )
        python_value = float(result.values[-1])
        python_error = abs(python_value - analytic)
        value_diff = abs(python_value - float(wolfram_row["value"]))
        analytic_diff = abs(analytic - float(wolfram_row["analytic"]))
        error_diff = abs(python_error - float(wolfram_row["abs_error"]))
        value_diffs.append(value_diff)
        analytic_diffs.append(analytic_diff)
        error_diffs.append(error_diff)
        rows.append(
            {
                "n_steps": n_steps,
                "python_value": python_value,
                "wolfram_value": float(wolfram_row["value"]),
                "value_diff": value_diff,
            }
        )

    return {
        "rows": rows,
        "value_max_diff": max(value_diffs),
        "analytic_max_diff": max(analytic_diffs),
        "reported_error_max_diff": max(error_diffs),
    }


def _compare_constant(payload: dict[str, Any], order: float) -> dict[str, Any]:
    section = payload["constant_raw_gl"]
    constant_value = float(payload["parameters"]["constant_value"])
    analytic = float(constant_value / gamma(1.0 - order))
    raw_diffs: list[float] = []
    analytic_diffs: list[float] = []
    shifted_diffs: list[float] = []

    for wolfram_row in section["rows"]:
        n_steps = int(round(float(wolfram_row["n_steps"])))
        step = 1.0 / n_steps
        samples = np.full(n_steps + 1, constant_value, dtype=float)
        raw = grunwald_letnikov_derivative(
            samples,
            step,
            order,
            definition="riemann_liouville_gl",
        )
        shifted = grunwald_letnikov_derivative(
            samples,
            step,
            order,
            definition="caputo_shifted",
        )
        raw_diffs.append(abs(float(raw.values[-1]) - float(wolfram_row["raw_value"])))
        analytic_diffs.append(abs(analytic - float(wolfram_row["rl_analytic"])))
        shifted_diffs.append(
            abs(
                float(np.max(np.abs(shifted.values), initial=0.0))
                - float(wolfram_row["caputo_shifted_max_abs"])
            )
        )

    return {
        "raw_value_max_diff": max(raw_diffs),
        "analytic_max_diff": max(analytic_diffs),
        "shifted_constant_max_diff": max(shifted_diffs),
    }


def _compare_q1_operator(payload: dict[str, Any]) -> dict[str, float]:
    section = payload["q1_operator"]
    step = float(section["step"])
    samples = np.asarray(section["samples"], dtype=float)
    raw = grunwald_letnikov_derivative(
        samples,
        step,
        1.0,
        definition="grunwald_letnikov",
    )
    shifted = grunwald_letnikov_derivative(
        samples,
        step,
        1.0,
        definition="caputo_shifted",
    )
    backward = np.diff(samples) / step
    return {
        "raw_max_diff": _max_abs(raw.values, section["raw_values"]),
        "shifted_max_diff": _max_abs(shifted.values, section["shifted_values"]),
        "backward_difference_max_diff": _max_abs(
            backward, section["backward_differences"]
        ),
    }


def _compare_fractional_solver(
    payload: dict[str, Any], order: float
) -> dict[str, Any]:
    section = payload["fractional_solver"]
    value_diffs: list[float] = []
    analytic_diffs: list[float] = []
    error_diffs: list[float] = []
    analytic = float(2.0 + 1.0 / gamma(1.0 + order))

    for wolfram_row in section["rows"]:
        n_steps = int(round(float(wolfram_row["n_steps"])))
        result = integrate_gl_explicit_numba(
            _constant_forcing_rhs,
            np.asarray([2.0]),
            order,
            step=1.0 / n_steps,
            n_steps=n_steps,
            initialization="caputo_shifted",
        )
        python_value = float(result.states[-1, 0])
        python_error = abs(python_value - analytic)
        value_diffs.append(abs(python_value - float(wolfram_row["value"])))
        analytic_diffs.append(abs(analytic - float(wolfram_row["analytic"])))
        error_diffs.append(abs(python_error - float(wolfram_row["abs_error"])))

    return {
        "value_max_diff": max(value_diffs),
        "analytic_max_diff": max(analytic_diffs),
        "reported_error_max_diff": max(error_diffs),
    }


def _compare_q1_solver(payload: dict[str, Any]) -> dict[str, float]:
    section = payload["q1_solver"]
    step = float(section["step"])
    n_steps = int(round(float(section["n_steps"])))
    initial = float(section["initial"])
    coefficient = float(section["lambda"])
    parameters = np.asarray([coefficient], dtype=float)
    shifted = integrate_gl_explicit_numba(
        _linear_rhs,
        np.asarray([initial]),
        1.0,
        parameters,
        step=step,
        n_steps=n_steps,
        initialization="caputo_shifted",
    )
    raw = integrate_gl_explicit_numba(
        _linear_rhs,
        np.asarray([initial]),
        1.0,
        parameters,
        step=step,
        n_steps=n_steps,
        initialization="discrete_gl",
    )
    indices = np.arange(n_steps + 1, dtype=float)
    euler = initial * (1.0 + coefficient * step) ** indices
    return {
        "shifted_max_diff": _max_abs(shifted.states[:, 0], section["shifted_values"]),
        "raw_max_diff": _max_abs(raw.states[:, 0], section["raw_values"]),
        "euler_max_diff": _max_abs(euler, section["euler_values"]),
    }


def compare_wolfram_summary(
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    tolerance: float = NUMERIC_TOLERANCE,
) -> dict[str, Any]:
    """Recompute every exported Wolfram sample with HAFO.

    Parameters
    ----------
    summary_path:
        JSON artifact exported by ``gl_fractional_operator_validation.wl``.
    tolerance:
        Maximum allowed Wolfram/Python difference.  This compares two
        floating-point implementations; it is not the discretization error
        against the continuum formulas, which is reported separately by the
        Wolfram artifact.
    """

    summary_path = Path(summary_path).resolve()
    payload = _load(summary_path)
    if payload.get("system_id") != SYSTEM_ID:
        raise ValueError(
            f"unexpected system_id {payload.get('system_id')!r}; expected {SYSTEM_ID!r}"
        )

    order = float(payload["parameters"]["order"])
    wolfram_weights = payload["weight_identity"]["sample"]
    python_weights = grunwald_letnikov_weights(order, len(wolfram_weights))
    weights = {"max_diff": _max_abs(python_weights, wolfram_weights)}
    monomial = _compare_monomial(payload, order)
    constant = _compare_constant(payload, order)
    q1_operator = _compare_q1_operator(payload)
    fractional_solver = _compare_fractional_solver(payload, order)
    q1_solver = _compare_q1_solver(payload)

    source = payload["source"]
    source_anchors_match = bool(
        source.get("podlubny_isbn") == "978-0-12-558840-9"
        and source.get("oldham_spanier_isbn") == "978-0-12-525550-9"
        and source.get("lubich_doi") == "10.1137/0517050"
    )
    independence_flags_match = bool(
        source.get("hafo_source_read") is False
        and source.get("report_input_used") is False
    )
    wolfram_tests_pass = bool(
        payload.get("passed") is True
        and all(test.get("passed") is True for test in payload.get("tests", []))
    )

    numeric_metrics = [
        weights["max_diff"],
        monomial["value_max_diff"],
        monomial["analytic_max_diff"],
        monomial["reported_error_max_diff"],
        constant["raw_value_max_diff"],
        constant["analytic_max_diff"],
        constant["shifted_constant_max_diff"],
        *q1_operator.values(),
        fractional_solver["value_max_diff"],
        fractional_solver["analytic_max_diff"],
        fractional_solver["reported_error_max_diff"],
        *q1_solver.values(),
    ]
    worst_numeric_diff = max(float(value) for value in numeric_metrics)
    passed = bool(
        wolfram_tests_pass
        and source_anchors_match
        and independence_flags_match
        and worst_numeric_diff <= float(tolerance)
    )

    return {
        "validation_scope": "independent_wolfram_to_hafo_GL_consistency",
        "evidence_boundary": payload["evidence_boundary"],
        "summary_path": str(summary_path),
        "tolerance": float(tolerance),
        "source_anchors_match": source_anchors_match,
        "independence_flags_match": independence_flags_match,
        "wolfram_tests_pass": wolfram_tests_pass,
        "weights": weights,
        "monomial_caputo_shifted": monomial,
        "constant_raw_gl": constant,
        "q1_operator": q1_operator,
        "fractional_solver": fractional_solver,
        "q1_solver": q1_solver,
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
