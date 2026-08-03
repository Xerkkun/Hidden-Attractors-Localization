"""Compare HAFO's public ABC operator with an independent Wolfram artifact.

The Wolfram case fixes ``alpha=1/2``, evaluates the Mittag--Leffler kernel
without reading HAFO, and exports interval weights plus constant, ramp, and
non-polynomial sampled cases.  This comparator recomputes those values through
the public Python, Numba, and offline FFT paths.

Passing is finite-grid operator evidence.  It is not evidence of an ABC FDE
solver, stability, chaos, attraction, hiddenness, or initial compatibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import erfcx

from hidden_attractors.fractional import (
    abc_piecewise_linear_weights,
    atangana_baleanu_caputo_derivative,
    atangana_baleanu_caputo_derivative_reference,
    atangana_baleanu_normalization,
)


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_ID = "atangana_baleanu_operator"
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
    "atangana_baleanu_doi": "10.2298/TSCI160111018A",
    "yadav_pandey_shukla_doi": "10.1016/j.chaos.2018.11.009",
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


def _compare_weights(
    payload: dict[str, Any],
    *,
    step: float,
    alpha: float,
) -> dict[str, Any]:
    wolfram = np.asarray(payload["weights"]["values"], dtype=np.float64)
    python = abc_piecewise_linear_weights(
        step, alpha, wolfram.size, backend="python"
    )
    numba = abc_piecewise_linear_weights(
        step, alpha, wolfram.size, backend="numba"
    )
    metrics = {
        "python_to_wolfram_max_diff": _max_abs(python.values, wolfram),
        "numba_to_wolfram_max_diff": _max_abs(numba.values, wolfram),
        "numba_to_python_max_diff": _max_abs(numba.values, python.values),
        "wolfram_primitive_max_diff": _max_abs(
            wolfram, payload["weights"]["primitive_values"]
        ),
    }
    return {
        **metrics,
        "python_backend": python.backend,
        "numba_backend": numba.backend,
        "max_diff": max(metrics.values()),
    }


def _compare_operator_case(
    case: dict[str, Any],
    *,
    step: float,
    alpha: float,
    lower_terminal: float,
) -> dict[str, Any]:
    samples = np.asarray(case["samples"], dtype=np.float64)
    expected = np.asarray(case["derivative_values"], dtype=np.float64)
    common = {
        "step": step,
        "alpha": alpha,
        "lower_terminal": lower_terminal,
        "normalization": atangana_baleanu_normalization,
        "normalization_name": "B(alpha)=1-alpha+alpha/Gamma(alpha)",
    }
    python = atangana_baleanu_caputo_derivative_reference(samples, **common)
    numba = atangana_baleanu_caputo_derivative(
        samples, backend="numba", **common
    )
    fft = atangana_baleanu_caputo_derivative(samples, backend="fft", **common)
    metrics = {
        "python_to_wolfram_max_diff": _max_abs(python.values, expected),
        "numba_to_wolfram_max_diff": _max_abs(numba.values, expected),
        "fft_to_wolfram_max_diff": _max_abs(fft.values, expected),
        "numba_to_python_max_diff": _max_abs(numba.values, python.values),
        "fft_to_python_max_diff": _max_abs(fft.values, python.values),
        "python_time_grid_max_diff": _max_abs(
            python.times,
            lower_terminal + step * np.arange(samples.size, dtype=np.float64),
        ),
    }
    return {
        **metrics,
        "python_backend": python.backend,
        "numba_backend": numba.backend,
        "fft_backend": fft.backend,
        "fft_length": fft.fft_length,
        "max_diff": max(metrics.values()),
    }


def _compare_kernel_identity(payload: dict[str, Any]) -> dict[str, float]:
    section = payload["kernel_identity"]
    elapsed = np.asarray(section["elapsed_times"], dtype=np.float64)
    independent = erfcx(np.sqrt(elapsed))
    metrics = {
        "mittag_leffler_to_scipy_erfcx_max_diff": _max_abs(
            section["mittag_leffler_values"], independent
        ),
        "wolfram_erfc_to_scipy_erfcx_max_diff": _max_abs(
            section["erfc_values"], independent
        ),
        "wolfram_internal_max_residual": abs(float(section["max_residual"])),
    }
    return {**metrics, "max_diff": max(metrics.values())}


def _compare_ramp_closed_form(
    payload: dict[str, Any],
    *,
    alpha: float,
    normalization: float,
) -> dict[str, float]:
    case = payload["ramp_case"]
    elapsed = np.asarray(case["elapsed_times"], dtype=np.float64)
    slope = float(case["slope"])
    kernel_integral = (
        erfcx(np.sqrt(elapsed))
        - 1.0
        + 2.0 * np.sqrt(elapsed) / np.sqrt(np.pi)
    )
    independent = normalization / (1.0 - alpha) * slope * kernel_integral
    metrics = {
        "wolfram_closed_form_to_scipy_max_diff": _max_abs(
            case["closed_form_values"], independent
        ),
        "wolfram_discrete_to_scipy_max_diff": _max_abs(
            case["derivative_values"], independent
        ),
        "wolfram_internal_max_residual": abs(
            float(case["closed_form_max_residual"])
        ),
    }
    return {**metrics, "max_diff": max(metrics.values())}


def compare_wolfram_summary(
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    tolerance: float = NUMERIC_TOLERANCE,
) -> dict[str, Any]:
    """Recompute the exported half-order ABC cases with public HAFO APIs."""

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
    lower_terminal = float(parameters["lower_terminal"])
    normalization = float(parameters["normalization"])
    if alpha != 0.5:
        raise ValueError("the independent ABC oracle must use alpha=1/2")

    public_normalization_diff = abs(
        atangana_baleanu_normalization(alpha) - normalization
    )
    weights = _compare_weights(payload, step=step, alpha=alpha)
    constant = _compare_operator_case(
        payload["constant_case"],
        step=step,
        alpha=alpha,
        lower_terminal=lower_terminal,
    )
    ramp = _compare_operator_case(
        payload["ramp_case"],
        step=step,
        alpha=alpha,
        lower_terminal=lower_terminal,
    )
    sample = _compare_operator_case(
        payload["sample_case"],
        step=step,
        alpha=alpha,
        lower_terminal=lower_terminal,
    )
    kernel_identity = _compare_kernel_identity(payload)
    ramp_closed_form = _compare_ramp_closed_form(
        payload, alpha=alpha, normalization=normalization
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
    worst_numeric_diff = max(
        public_normalization_diff,
        weights["max_diff"],
        constant["max_diff"],
        ramp["max_diff"],
        sample["max_diff"],
        kernel_identity["max_diff"],
        ramp_closed_form["max_diff"],
    )
    backends_match = bool(
        weights["python_backend"] == "python"
        and weights["numba_backend"] == "numba"
        and all(
            case["python_backend"] == "python_direct"
            and case["numba_backend"] == "numba_direct"
            and case["fft_backend"] == "numpy_fft_offline"
            and case["fft_length"] is not None
            for case in (constant, ramp, sample)
        )
    )
    passed = bool(
        wolfram_tests_pass
        and source_anchors_match
        and independence_flags_match
        and backends_match
        and worst_numeric_diff <= float(tolerance)
    )
    return {
        "validation_scope": "independent_wolfram_to_hafo_ABC_consistency",
        "evidence_boundary": payload["evidence_boundary"],
        "summary_path": str(summary_path),
        "tolerance": float(tolerance),
        "source_anchors_match": source_anchors_match,
        "independence_flags_match": independence_flags_match,
        "wolfram_tests_pass": wolfram_tests_pass,
        "backends_match": backends_match,
        "public_normalization_diff": public_normalization_diff,
        "weights": weights,
        "constant_case": constant,
        "ramp_case": ramp,
        "sample_case": sample,
        "kernel_identity": kernel_identity,
        "ramp_closed_form": ramp_closed_form,
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
