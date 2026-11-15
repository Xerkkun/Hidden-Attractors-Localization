"""The published-row verdict must enforce the declared sign contract."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation" / "python"))

import cloned_dynamics_benchmarks as benchmarks  # noqa: E402


@pytest.fixture
def expected() -> dict:
    return {"tolerance_abs_target": 0.05, "tolerance_abs_initial": 0.15,
            "sign_pattern_required": True}


def test_small_absolute_error_does_not_override_required_sign(expected: dict) -> None:
    result = benchmarks.evaluate_benchmark_spectrum([0.1, -0.002, -1], [0.1, 0.002, -1], expected)
    assert result["quantitative_tolerance_met"] is True
    assert result["sign_match"] is False
    assert result["status"] == "published_benchmark_failed_quantitative"
    assert result["legacy_status"] == "published_benchmark_passed_quantitative"
    assert result["evaluation_criteria_version"] == benchmarks.CURRENT_CRITERIA_VERSION


def test_matching_signs_outside_target_only_support_sign_pattern(expected: dict) -> None:
    result = benchmarks.evaluate_benchmark_spectrum([0.2, 0.002, -1.5], [0.1, 0.002, -1], expected)
    assert result["status"] == "published_benchmark_passed_sign_pattern"
    assert result["quantitative_tolerance_met"] is False


def test_matching_signs_still_require_initial_tolerance(expected: dict) -> None:
    result = benchmarks.evaluate_benchmark_spectrum([0.3, 0.002, -1], [0.1, 0.002, -1], expected)
    assert result["status"] == "published_benchmark_failed_quantitative"


def test_near_zero_is_a_diagnostic_and_never_rounds_sign(expected: dict) -> None:
    result = benchmarks.evaluate_benchmark_spectrum([1e-12, -1], [-1e-12, -1], expected)
    assert result["component_sign_diagnostics"][0] == "near_zero_boundary_crossing"
    assert result["near_zero_diagnostic_changes_verdict"] is False
    assert result["status"] == "published_benchmark_failed_quantitative"
    assert result["sign_match"] is False


def test_exact_zero_is_not_positive_or_negative(expected: dict) -> None:
    result = benchmarks.evaluate_benchmark_spectrum([0, -1], [1e-12, -1], expected)
    assert result["sign_match"] is False
    assert result["status"] == "published_benchmark_failed_quantitative"


def test_yaml_can_explicitly_omit_required_sign_gate(expected: dict) -> None:
    expected["sign_pattern_required"] = False
    result = benchmarks.evaluate_benchmark_spectrum([0.002, -1], [-0.002, -1], expected)
    assert result["sign_match"] is False
    assert result["required_sign_gate_met"] is True
    assert result["status"] == "published_benchmark_passed_quantitative"


def test_explicit_legacy_preserves_historical_tolerance_first_verdict(expected: dict) -> None:
    result = benchmarks.evaluate_benchmark_spectrum(
        [0.002, -1], [-0.002, -1], expected,
        criteria_version=benchmarks.LEGACY_CRITERIA_VERSION,
    )
    assert result["status"] == result["legacy_status"] == "published_benchmark_passed_quantitative"
    assert result["required_sign_gate_met"] is False


def test_yaml_version_and_explicit_version_override(expected: dict) -> None:
    expected["evaluation_criteria_version"] = benchmarks.LEGACY_CRITERIA_VERSION
    assert benchmarks.evaluate_benchmark_spectrum([0.002], [-0.002], expected)["status"] == (
        "published_benchmark_passed_quantitative")
    assert benchmarks.evaluate_benchmark_spectrum(
        [0.002], [-0.002], expected, criteria_version=benchmarks.CURRENT_CRITERIA_VERSION,
    )["status"] == "published_benchmark_failed_quantitative"


@pytest.mark.parametrize("solver_status,values", [("failed", [0.1, -1]), ("ok", [np.nan, -1])])
def test_nonfinite_or_failed_solver_is_inconclusive(expected: dict, solver_status: str, values: list) -> None:
    result = benchmarks.evaluate_benchmark_spectrum(
        values, [0.1, -1], expected, solver_status=solver_status, fast=True,
    )
    assert result["status"] == "published_benchmark_inconclusive"


def test_smoke_is_never_quantitative_validation(expected: dict) -> None:
    result = benchmarks.evaluate_benchmark_spectrum([0.1, -1], [0.1, -1], expected, fast=True)
    assert result["status"] == "published_benchmark_smoke_passed"


def test_run_adapter_preserves_yaml_parameters_and_evaluator(monkeypatch, expected: dict) -> None:
    calls = []

    def fake_solver(rhs, x0, **kwargs):
        calls.append(rhs(0.0, x0))
        return SimpleNamespace(exponents=[0.002, -1, -2], status="ok")

    monkeypatch.setattr(benchmarks, "compute_cloned_dynamics_spectrum", fake_solver)
    case = {"system": {"kind": "four_wing", "parameters": {"b": 0.9, "c": 4}},
            "method_id": benchmarks.METHOD_ID, "expected": expected,
            "integration": {"h_clone": .01, "t_clone": 1, "K": 2, "delta": .001,
                            "memory_protocol": "published_block_restart"}}
    row = {"orders": [1, 1, 1], "lyapunov": [-0.002, -1, -2], "K01": 0, "type": "Comm"}
    record, _ = benchmarks.run_benchmark_row(case, row)
    np.testing.assert_allclose(calls[0], [-.09, .09, .51])
    assert record["status"] == "published_benchmark_failed_quantitative"
    assert record["legacy_status"] == "published_benchmark_passed_quantitative"


def test_invalid_criteria_is_rejected(expected: dict) -> None:
    with pytest.raises(ValueError, match="unknown benchmark"):
        benchmarks.evaluate_benchmark_spectrum([1], [1], expected, criteria_version="silent_relaxation")


def test_published_source_must_be_finite(expected: dict) -> None:
    result = benchmarks.evaluate_benchmark_spectrum([1], [np.nan], expected)
    assert result["status"] == "published_benchmark_inconclusive"
