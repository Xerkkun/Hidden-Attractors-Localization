"""F6 integrates evidence conservatively without producing certifications."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from hidden_attractors.analysis.integrated_chaos_validator import integrate_case_evidence
from validation.python.run_integrated_chaos_validator import _f4_status


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation" / "chaos_validation" / "integrated_chaos_validator"
SUMMARY = OUTPUT / "integrated_chaos_summary.json"


def test_f6_runner_writes_integrated_outputs() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "validation" / "python" / "run_integrated_chaos_validator.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert SUMMARY.is_file()
    assert (OUTPUT / "integrated_chaos_by_case.csv").is_file()
    assert (OUTPUT / "integrated_chaos_evidence_matrix.csv").is_file()
    assert (OUTPUT / "integrated_chaos_rules.json").is_file()


def test_f6_loads_f5_and_all_three_cases_have_evidence() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["inputs"]["f5_summary"].endswith("f5_diagnostics_summary.json")
    assert summary["inputs"]["lyapunov_registry"] == "hidden_attractors/analysis/lyapunov_methods.py"
    assert all(
        item["available"]
        for item in summary["inputs"]["lyapunov_validation_summaries"].values()
    )
    assert summary["cases_total"] == 3
    assert len(summary["cases"]) == 3
    assert all(case["evidence"] for case in summary["cases"])


def test_current_f5_conflicts_remain_inconclusive_even_with_integer_positive_lambda() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert {case["chaos_evidence_level"] for case in summary["cases"]} == {
        "chaos_evidence_inconclusive"
    }
    integer = next(case for case in summary["cases"] if case["case_id"] == "chua_integer_q1_reference")
    method = next(
        item for item in integer["evidence"]["lyapunov_methods"] if item["method_id"] == "integer_qr_benettin"
    )
    assert method["validated"] is True
    assert method["lambda_max"] > 0.0


def test_explicit_non_conflicting_rule_can_label_chaotic_candidate() -> None:
    decision = integrate_case_evidence(
        case_id="synthetic",
        boundedness_status="bounded_candidate",
        zero_one_status="zero_one_chaotic_candidate",
        psd_fft_status="broadband_spectrum",
        poincare_status="cloud_like",
        lyapunov_evidence=[],
        f4_status="f4_internal_validation_missing_or_pending",
    )
    assert decision["chaos_evidence_level"] == "strong_chaos_evidence"
    assert "chaos_verified" not in decision


def test_f6_never_certifies_chaos_or_hiddenness() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert "certifications" not in summary
    assert "chaos_verified" not in summary
    assert all("chaos_verified" not in case for case in summary["cases"])
    assert all("hidden_verified" not in case for case in summary["cases"])
    assert summary["chaos_evidence_level"] == "chaos_evidence_inconclusive"
    assert summary["hiddenness_evidence_level"] == "not_evaluated_by_this_stage"


def test_missing_f4_is_reported_without_blocking_f6_classification() -> None:
    assert _f4_status(None) == "f4_internal_validation_missing_or_pending"
