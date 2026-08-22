"""Validate the independent Wolfram fast tempered-history fixture.

The Wolfram case constructs FBDF1/GNGF2 weights, direct tempered
convolutions, and real-axis recurrent histories at 80-digit precision without
reading HAFO.  This module first validates that artifact with an independent
float64 direct convolution.  Only the optional public-core comparison imports
HAFO.

Passing is finite sampled-grid implementation evidence.  It is not a general
trapezoidal-tail certificate, an FDE solver theorem, or evidence of chaos,
attraction, or hiddenness.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_ID = "tempered_fast_multistep_history"
TEMP_ROOT = Path(tempfile.gettempdir())
DEFAULT_SUMMARY = (
    TEMP_ROOT
    / "hafo_tempered_fast_multistep_history"
    / f"{SYSTEM_ID}_validation_summary.json"
)
PERSISTED_SUMMARY = (
    ROOT
    / "validation"
    / "reference_cases"
    / "tempered_fast_multistep_history"
    / f"{SYSTEM_ID}_validation_summary.json"
)

SOURCE_ANCHORS = {
    "guo_fast_method_ii_doi": "10.1137/18M1230153",
    "lubich_flmm_doi": "10.1137/0517050",
    "trefethen_weideman_trapezoidal_doi": "10.1137/130932132",
}
EXPECTED_METHODS = ("fbdf1", "gngf2")
EXPECTED_DEFINITIONS = (
    "tempered_riemann_liouville",
    "tempered_caputo",
)
INDEPENDENT_TOLERANCE = 3.0e-12
CORE_TOLERANCE = 5.0e-10
PUBLIC_API_MODULE = "hidden_attractors.fractional"
PUBLIC_CORE_NAME = "tempered_fast_multistep_history"


class PublicFastHistoryAPIUnavailable(RuntimeError):
    """The optional public HAFO implementation cannot be imported."""


def _load(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _array(value: Any, *, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions, received {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("Wolfram fixture contains a non-finite value")
    return result


def _max_abs(left: Any, right: Any) -> float:
    left_array = _array(left)
    right_array = _array(right)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"shape mismatch: {left_array.shape} != {right_array.shape}"
        )
    return float(np.max(np.abs(left_array - right_array), initial=0.0))


def _weights(order: float, count: int, method: str) -> np.ndarray:
    if method not in EXPECTED_METHODS:
        raise ValueError(f"unexpected multistep method {method!r}")
    gl = np.empty(int(count), dtype=np.float64)
    if gl.size == 0:
        return gl
    gl[0] = 1.0
    for lag in range(1, gl.size):
        gl[lag] = ((lag - 1.0 - float(order)) / lag) * gl[lag - 1]
    if method == "fbdf1":
        return gl
    result = (1.0 + 0.5 * float(order)) * gl
    if result.size > 1:
        result[1:] -= 0.5 * float(order) * gl[:-1]
    return result


def _direct_tempered(
    samples: Any,
    order: float,
    tempering: float,
    step: float,
    method: str,
    definition: str,
) -> np.ndarray:
    values = _array(samples, ndim=1)
    if definition not in EXPECTED_DEFINITIONS:
        raise ValueError(f"unexpected definition {definition!r}")
    weights = _weights(order, values.size, method)
    lags = np.arange(values.size, dtype=np.float64)
    damped = weights * np.exp(-float(tempering) * float(step) * lags)
    result = np.convolve(values, damped)[: values.size]
    if definition == "tempered_caputo":
        result -= (
            values[0]
            * np.exp(-float(tempering) * float(step) * lags)
            * np.cumsum(weights)
        )
        result[0] = 0.0
    return result / (float(step) ** float(order))


def _rows(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    raw = payload["fixture"]["rows"]
    if not isinstance(raw, list):
        raise TypeError("fixture rows must be a list")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in raw:
        if not isinstance(row, dict):
            raise TypeError("every fixture row must be an object")
        key = (str(row["method"]), str(row["definition"]))
        if key in result:
            raise ValueError(f"duplicate fixture row {key!r}")
        result[key] = row
    expected = {
        (method, definition)
        for method in EXPECTED_METHODS
        for definition in EXPECTED_DEFINITIONS
    }
    if set(result) != expected:
        raise ValueError(f"unexpected fixture rows: {tuple(result)!r}")
    return result


def validate_wolfram_summary(path: Path = DEFAULT_SUMMARY) -> dict[str, Any]:
    """Validate source anchors, independence flags, tests, and fixtures."""

    payload = _load(path)
    if payload.get("system_id") != SYSTEM_ID:
        raise ValueError(
            f"unexpected system_id {payload.get('system_id')!r}; "
            f"expected {SYSTEM_ID!r}"
        )
    source = payload.get("source")
    if not isinstance(source, dict):
        raise TypeError("source metadata must be an object")
    source_anchors_match = all(
        source.get(key) == value for key, value in SOURCE_ANCHORS.items()
    )
    independence_flags_match = all(
        source.get(key) is False
        for key in (
            "hafo_source_read",
            "hafo_formula_imported",
            "hafo_output_used",
            "report_input_used",
        )
    )
    conventions = payload.get("conventions")
    if not isinstance(conventions, dict):
        raise TypeError("conventions metadata must be an object")
    convention_contract_match = (
        int(conventions.get("working_precision", 0)) == 80
        and conventions.get("dimensionless_node") == "r=h*lambda"
        and conventions.get("fractional_bdf2_claimed") is False
        and conventions.get("positive_exponential_materialized") is False
    )
    tests = payload.get("tests")
    if not isinstance(tests, list) or len(tests) != 13:
        raise ValueError("expected exactly 13 Wolfram tests")
    wolfram_tests_pass = bool(payload.get("passed")) and all(
        isinstance(test, dict) and test.get("passed") is True for test in tests
    )

    fixture = payload["fixture"]
    order = float(fixture["order"])
    tempering = float(fixture["tempering"])
    step = float(fixture["step"])
    samples = _array(fixture["samples"], ndim=1)
    elapsed = _array(fixture["elapsed"], ndim=1)
    metrics: dict[str, float] = {
        "elapsed_grid_max_diff": _max_abs(
            elapsed, step * np.arange(samples.size, dtype=np.float64)
        )
    }
    for (method, definition), row in _rows(payload).items():
        direct = _direct_tempered(
            samples, order, tempering, step, method, definition
        )
        prefix = f"{method}_{definition}"
        metrics[f"{prefix}_direct_max_diff"] = _max_abs(
            direct, row["direct_values"]
        )
        metrics[f"{prefix}_fast_max_diff"] = _max_abs(
            direct, row["fast_values"]
        )
        metrics[f"{prefix}_wolfram_residual"] = abs(
            float(row["max_fast_direct_residual"])
        )
        minimum_decay = float(row["minimum_decay"])
        maximum_decay = float(row["maximum_decay"])
        if not 0.0 < minimum_decay <= maximum_decay <= 1.0:
            raise ValueError("Wolfram recurrence decay lies outside (0, 1]")

    anchor_fixture = payload["anchor_fixture"]
    if not isinstance(anchor_fixture, dict):
        raise TypeError("anchor_fixture metadata must be an object")
    anchor_samples = _array(anchor_fixture["samples"], ndim=1)
    if anchor_samples.shape != samples.shape:
        raise ValueError(
            "anchor_fixture samples must match the main fixture length"
        )
    anchor_rows = anchor_fixture["rows"]
    if not isinstance(anchor_rows, list) or len(anchor_rows) != 2:
        raise ValueError("expected two anchor rows")
    observed_anchor_methods: set[str] = set()
    for row in anchor_rows:
        if not isinstance(row, dict):
            raise TypeError("every anchor row must be an object")
        method = str(row["method"])
        if method not in EXPECTED_METHODS:
            raise ValueError(f"unexpected anchor method {method!r}")
        if method in observed_anchor_methods:
            raise ValueError(f"duplicate anchor method {method!r}")
        observed_anchor_methods.add(method)
        reported_values = _array(row["values"], ndim=1)
        reconstructed = _direct_tempered(
            anchor_samples,
            order,
            tempering,
            step,
            method,
            "tempered_caputo",
        )
        metrics[f"{method}_anchor_reconstruction_max_diff"] = _max_abs(
            reconstructed, reported_values
        )
        metrics[f"{method}_anchor_reconstructed_residual"] = float(
            np.max(np.abs(reconstructed), initial=0.0)
        )
        metrics[f"{method}_anchor_reported_residual"] = max(
            float(np.max(np.abs(reported_values), initial=0.0)),
            abs(float(row["max_anchor_residual"])),
        )
    if observed_anchor_methods != set(EXPECTED_METHODS):
        raise ValueError(
            f"unexpected anchor methods: {tuple(sorted(observed_anchor_methods))!r}"
        )

    numeric_max_diff = max(metrics.values(), default=0.0)
    passed = (
        source_anchors_match
        and independence_flags_match
        and convention_contract_match
        and wolfram_tests_pass
        and numeric_max_diff <= INDEPENDENT_TOLERANCE
    )
    return {
        "system_id": SYSTEM_ID,
        "path": str(Path(path).resolve()),
        "source_anchors_match": source_anchors_match,
        "independence_flags_match": independence_flags_match,
        "convention_contract_match": convention_contract_match,
        "wolfram_tests_pass": wolfram_tests_pass,
        "numeric_metrics": metrics,
        "numeric_max_diff": numeric_max_diff,
        "tolerance": INDEPENDENT_TOLERANCE,
        "passed": passed,
    }


@lru_cache(maxsize=1)
def _load_public_api() -> tuple[Callable[..., Any], str, str]:
    try:
        module = importlib.import_module(PUBLIC_API_MODULE)
    except Exception as exc:  # pragma: no cover - environment diagnostic
        raise PublicFastHistoryAPIUnavailable(
            f"cannot import {PUBLIC_API_MODULE}: {exc}"
        ) from exc
    evaluator = getattr(module, PUBLIC_CORE_NAME, None)
    if not callable(evaluator):
        raise PublicFastHistoryAPIUnavailable(
            f"{PUBLIC_API_MODULE}.{PUBLIC_CORE_NAME} is unavailable"
        )
    rl_token = getattr(
        module, "TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION", None
    )
    caputo_token = getattr(module, "TEMPERED_CAPUTO_INITIAL_CONDITION", None)
    if not isinstance(rl_token, str) or not isinstance(caputo_token, str):
        raise PublicFastHistoryAPIUnavailable(
            "public tempered initial-condition tokens are unavailable"
        )
    return evaluator, rl_token, caputo_token


def public_api_status() -> tuple[bool, str]:
    try:
        evaluator, _, _ = _load_public_api()
    except PublicFastHistoryAPIUnavailable as exc:
        return False, str(exc)
    return True, f"{evaluator.__module__}.{evaluator.__name__}"


def _compare_public_core(payload: dict[str, Any]) -> dict[str, Any]:
    evaluator, rl_token, caputo_token = _load_public_api()
    fixture = payload["fixture"]
    order = float(fixture["order"])
    tempering = float(fixture["tempering"])
    step = float(fixture["step"])
    samples = _array(fixture["samples"], ndim=1)
    local_steps = int(fixture["local_history_steps"])
    requested_compression_tolerance = 1.0e-10
    difference_metrics: dict[str, float] = {}
    calibration_metrics: dict[str, float] = {}
    used_points: dict[str, int] = {}
    calibration_satisfied = True
    for (method, definition), row in _rows(payload).items():
        token = caputo_token if definition == "tempered_caputo" else rl_token
        result = evaluator(
            samples,
            order,
            tempering=tempering,
            multistep_method=method,
            definition=definition,
            step=step,
            initial_condition_semantics=token,
            local_history_steps=local_steps,
            relative_tolerance=requested_compression_tolerance,
            backend="numba",
        )
        prefix = f"{method}_{definition}"
        difference_metrics[f"{prefix}_max_diff"] = _max_abs(
            result.values, row["direct_values"]
        )
        calibration_metrics[f"{prefix}_reported_l1_relative_error"] = float(
            np.max(result.l1_relative_weight_error, initial=0.0)
        )
        used_points[prefix] = int(result.quadrature_points)
        calibration_satisfied = calibration_satisfied and bool(
            result.compression_tolerance_satisfied
        )
    maximum_difference = max(difference_metrics.values(), default=0.0)
    maximum_l1_relative_error = max(
        calibration_metrics.values(), default=0.0
    )
    calibration_satisfied = (
        calibration_satisfied
        and maximum_l1_relative_error <= requested_compression_tolerance
    )
    return {
        "metrics": difference_metrics,
        "calibration_metrics": calibration_metrics,
        "quadrature_points": used_points,
        "max_diff": maximum_difference,
        "tolerance": CORE_TOLERANCE,
        "requested_compression_tolerance": requested_compression_tolerance,
        "max_reported_l1_relative_weight_error": maximum_l1_relative_error,
        "compression_tolerance_satisfied": calibration_satisfied,
        "passed": (
            maximum_difference <= CORE_TOLERANCE and calibration_satisfied
        ),
    }


def compare_wolfram_summary(
    path: Path = DEFAULT_SUMMARY,
    *,
    require_core: bool = False,
) -> dict[str, Any]:
    independent = validate_wolfram_summary(path)
    try:
        core = _compare_public_core(_load(path))
        available = True
        diagnostic = f"{PUBLIC_API_MODULE}.{PUBLIC_CORE_NAME}"
    except PublicFastHistoryAPIUnavailable as exc:
        core = None
        available = False
        diagnostic = str(exc)
    performed = core is not None
    core_passed = bool(core and core["passed"])
    passed = independent["passed"] and (
        core_passed if require_core else (not performed or core_passed)
    )
    return {
        "system_id": SYSTEM_ID,
        "independent": independent,
        "public_core_available": available,
        "public_core_diagnostic": diagnostic,
        "cross_implementation_performed": performed,
        "cross_implementation_passed": core_passed,
        "cross_implementation_result": core,
        "require_core": bool(require_core),
        "passed": passed,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--require-core", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = compare_wolfram_summary(
        arguments.summary, require_core=arguments.require_core
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CORE_TOLERANCE",
    "DEFAULT_SUMMARY",
    "EXPECTED_DEFINITIONS",
    "EXPECTED_METHODS",
    "INDEPENDENT_TOLERANCE",
    "PERSISTED_SUMMARY",
    "PUBLIC_API_MODULE",
    "PUBLIC_CORE_NAME",
    "SOURCE_ANCHORS",
    "SYSTEM_ID",
    "compare_wolfram_summary",
    "public_api_status",
    "validate_wolfram_summary",
]
