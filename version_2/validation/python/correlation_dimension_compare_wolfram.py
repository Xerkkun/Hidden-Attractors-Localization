"""Compare HAFO's public correlation-dimension API with Wolfram.

The Wolfram case constructs an exact two-dimensional point set, enumerates
unordered pairs after a positive Theiler exclusion, applies the strict
``distance < radius`` criterion, and performs an explicit least-squares
log--log fit.  It does not read HAFO source or generated reports.  This
comparator reaches HAFO only through :func:`correlation_sum_curve` and
:func:`fit_correlation_dimension`, using the transparent Python backend.

Passing is finite-set pair-count and regression consistency evidence only.
It is not evidence that the declared fit interval is an asymptotic scaling
region, nor proof of a fractal dimension, chaos, attraction, or hiddenness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hidden_attractors.analysis.correlation_dimension import (
    correlation_sum_curve,
    fit_correlation_dimension,
)


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_ID = "correlation_dimension"
DEFAULT_SUMMARY = (
    ROOT
    / "validation"
    / "outputs"
    / "wolfram"
    / SYSTEM_ID
    / f"{SYSTEM_ID}_validation_summary.json"
)
NUMERIC_TOLERANCE = 5.0e-13

SOURCE_ANCHORS = {
    "grassberger_procaccia_doi": "10.1103/PhysRevLett.50.346",
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


def compare_wolfram_summary(
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    tolerance: float = NUMERIC_TOLERANCE,
) -> dict[str, Any]:
    """Recompute the exact finite-set curve and explicit fit with public APIs."""

    summary_path = Path(summary_path).resolve()
    payload = _load(summary_path)
    if payload.get("system_id") != SYSTEM_ID:
        raise ValueError(
            f"unexpected system_id {payload.get('system_id')!r}; "
            f"expected {SYSTEM_ID!r}"
        )

    parameters = payload["parameters"]
    points = np.asarray(parameters["points"], dtype=np.float64)
    radii = np.asarray(parameters["radii"], dtype=np.float64)
    theiler_window = int(parameters["theiler_window"])
    metric = str(parameters["metric"])
    fit_radius_range_values = np.asarray(
        parameters["fit_radius_range"],
        dtype=np.float64,
    )
    if fit_radius_range_values.shape != (2,):
        raise ValueError("fit_radius_range must contain exactly two values")
    fit_radius_range = (
        float(fit_radius_range_values[0]),
        float(fit_radius_range_values[1]),
    )

    curve = correlation_sum_curve(
        points,
        radii,
        theiler_window=theiler_window,
        metric=metric,
        backend="python",
        fallback=False,
        sampling="declared exact synthetic point ordering",
        projection="identity projection in R^2",
    )
    fit = fit_correlation_dimension(
        curve,
        fit_radius_range=fit_radius_range,
        minimum_points=3,
    )

    pair_geometry = payload["pair_geometry"]
    wolfram_curve = payload["correlation_curve"]
    curve_metrics = {
        "radii_max_diff": _max_abs(curve.radii, wolfram_curve["radii"]),
        "strict_pair_counts_max_diff": _max_abs(
            curve.counts,
            wolfram_curve["strict_pair_counts"],
        ),
        "correlation_sums_max_diff": _max_abs(
            curve.correlation_sums,
            wolfram_curve["correlation_sums"],
        ),
        "eligible_pair_denominator_diff": abs(
            int(curve.eligible_pairs) - int(pair_geometry["denominator"])
        ),
        "curve_denominator_internal_diff": abs(
            int(curve.eligible_pairs) - int(wolfram_curve["denominator"])
        ),
    }
    curve_metrics["max_diff"] = max(curve_metrics.values())

    wolfram_fit = payload["fit"]
    predicted = fit.intercept + fit.slope * fit.log_radii
    residuals = fit.log_correlation_sums - predicted
    residual_sum_squares = float(residuals @ residuals)
    selected_radii = curve.radii[fit.fit_indices]
    selected_sums = curve.correlation_sums[fit.fit_indices]
    finite_local_slopes = np.isfinite(fit.local_slopes)
    fit_metrics = {
        "fit_indices_max_diff": _max_abs(
            fit.fit_indices,
            wolfram_fit["selected_indices_zero_based"],
        ),
        "fit_radii_max_diff": _max_abs(
            selected_radii,
            wolfram_fit["radii"],
        ),
        "fit_correlation_sums_max_diff": _max_abs(
            selected_sums,
            wolfram_fit["correlation_sums"],
        ),
        "log_radii_max_diff": _max_abs(
            fit.log_radii,
            wolfram_fit["log_radii"],
        ),
        "log_correlation_sums_max_diff": _max_abs(
            fit.log_correlation_sums,
            wolfram_fit["log_correlation_sums"],
        ),
        "intercept_abs_diff": abs(
            float(fit.intercept) - float(wolfram_fit["intercept"])
        ),
        "slope_abs_diff": abs(
            float(fit.slope) - float(wolfram_fit["slope"])
        ),
        "r_squared_abs_diff": abs(
            float(fit.r_squared) - float(wolfram_fit["r_squared"])
        ),
        "regression_standard_error_abs_diff": abs(
            float(fit.regression_standard_error)
            - float(wolfram_fit["regression_standard_error"])
        ),
        "predicted_log_sums_max_diff": _max_abs(
            predicted,
            wolfram_fit["predicted_log_correlation_sums"],
        ),
        "residuals_max_diff": _max_abs(
            residuals,
            wolfram_fit["residuals"],
        ),
        "residual_sum_squares_abs_diff": abs(
            residual_sum_squares
            - float(wolfram_fit["residual_sum_squares"])
        ),
        "local_slopes_max_diff": _max_abs(
            fit.local_slopes[finite_local_slopes],
            wolfram_fit["local_slopes"],
        ),
        "local_slope_radii_max_diff": _max_abs(
            fit.local_slope_radii[finite_local_slopes],
            wolfram_fit["local_slope_radii"],
        ),
    }
    fit_metrics["max_diff"] = max(fit_metrics.values())

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
    curve_contract_match = bool(
        curve.backend == "python"
        and curve.metric == "euclidean"
        and curve.theiler_window == theiler_window == 1
        and int(curve.eligible_pairs) == 10
        and parameters.get("radius_criterion")
        == "Norm[x_i-x_j] < r (strict)"
        and parameters.get("pair_criterion")
        == "unordered_i_less_than_j_and_j_minus_i_greater_than_w"
    )
    fit_contract_match = bool(
        fit.curve is curve
        and fit.fit_radius_range == fit_radius_range
        and fit.minimum_points == 3
        and np.all(selected_radii >= fit_radius_range[0])
        and np.all(selected_radii <= fit_radius_range[1])
        and np.all(selected_sums > 0.0)
        and np.all(selected_sums < 1.0)
        and wolfram_fit.get("method")
        == "LeastSquares[{1,log(r)},log(C)]"
    )

    cross_implementation_max_diff = max(
        curve_metrics["max_diff"],
        fit_metrics["max_diff"],
    )
    passed = bool(
        source_anchors_match
        and independence_flags_match
        and wolfram_tests_pass
        and curve_contract_match
        and fit_contract_match
        and cross_implementation_max_diff <= float(tolerance)
    )
    return {
        "validation_scope": (
            "independent_Wolfram_to_HAFO_correlation_sum_and_fit_"
            "finite_consistency"
        ),
        "evidence_boundary": payload["evidence_boundary"],
        "summary_path": str(summary_path),
        "tolerance": float(tolerance),
        "source_anchors_match": source_anchors_match,
        "independence_flags_match": independence_flags_match,
        "wolfram_tests_pass": wolfram_tests_pass,
        "curve_contract_match": curve_contract_match,
        "fit_contract_match": fit_contract_match,
        "python_backend": curve.backend,
        "curve_metrics": curve_metrics,
        "fit_metrics": fit_metrics,
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
        help="Optional JSON destination for the Python comparison summary.",
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
