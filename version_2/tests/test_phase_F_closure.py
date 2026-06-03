"""Phase F freezes as a structured finite-time chaos-evidence layer."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from hidden_attractors.analysis.phase_f_closure import assess_phase_f_closure


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation" / "chaos_validation" / "phase_F_closure"
SUMMARY = OUTPUT / "phase_F_closure_summary.json"
MATRIX = OUTPUT / "phase_F_closure_matrix.csv"
STATEMENT = OUTPUT / "phase_F_diagnostic_scope_statement.md"
CHAOS_ROOT = ROOT / "validation" / "chaos_validation"


def test_phase_f_runner_writes_closure_artifacts() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "validation" / "python" / "run_phase_F_closure_assessment.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert SUMMARY.is_file()
    assert MATRIX.is_file()
    assert STATEMENT.is_file()
    assert (OUTPUT / "phase_F_closure_decision.md").is_file()
    assert (OUTPUT / "phase_F_closure_rules.json").is_file()


def test_phase_f_freezes_as_finite_time_evidence_layer() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["status"] == "phase_F_frozen"
    assert summary["phase_F_frozen"] is True
    assert summary["evidence_layer"] == "finite_time_chaos_evidence"
    assert summary["available_evidence_level"] == "chaos_evidence_inconclusive"
    assert summary["structured_diagnostics_closed"] is True


def test_routes_a_and_b_are_assessed_not_discarded_or_promoted() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    routes = summary["closure_routes"]
    route_a = routes["route_A_fractional_variational_abm_qr"]
    route_b = routes["route_B_fractional_cloned_dynamics"]
    assert route_a["status"] == "assessed_with_documented_validation_gap"
    assert route_a["strict_requirement_satisfied"] is False
    assert route_a["f4_method_control"]["status"] == "separate_internal_control_only_published_validation_pending"
    assert route_a["related_dk2018_reproduction_lane"]["does_not_promote_full_history_qr"] is True
    assert route_b["status"] == "assessed_with_documented_discrepancies"
    assert route_b["strict_requirement_satisfied"] is False
    assert route_b["rows_total"] == 24
    assert route_b["rows_passed_quantitative"] == 10
    assert route_b["rows_passed_sign_pattern"] == 6
    assert route_b["rows_failed"] == 8
    assert route_b["diagnostic_closure"]["status"] == "closed_with_documented_discrepancies"
    assert routes["route_C_diagnostic_scope_closure"]["status"] == "completed"


def test_fractional_method_criterion_is_labeled_not_flattened_to_false() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert (
        summary["criteria"]["valid_fractional_lyapunov_method_per_candidate"]
        == "not_strictly_validated_with_documented_attempts"
    )
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        rows = {row["criterion_id"]: row for row in csv.DictReader(handle)}
    assert rows["valid_fractional_lyapunov_method_per_candidate"]["blocker"] == ""
    assert (
        rows["valid_fractional_lyapunov_method_per_candidate"]["current_status"]
        == "not_strictly_validated_with_documented_attempts"
    )


def test_scope_statement_documents_methods_discrepancies_and_boundaries() -> None:
    statement = STATEMENT.read_text(encoding="utf-8")
    for required in (
        "fractional_variational_abm_qr",
        "fractional_cloned_dynamics_abm_gs_published",
        "published_benchmarks_pending_discrepancy",
        "F4",
        "F5",
        "finite-time chaos-evidence layer",
        "sampled-neighborhood candidate gate",
        "assessed_with_documented_validation_gap",
        "assessed_with_documented_discrepancies",
    ):
        assert required in statement


def test_missing_f6_f7_are_reported_without_failing_diagnostic_closure() -> None:
    def read(relative: str) -> dict:
        return json.loads((CHAOS_ROOT / relative).read_text(encoding="utf-8"))

    summary = assess_phase_f_closure(
        f4_summary=read("lyapunov_methods/F4_internal_validation/f4_internal_validation_summary.json"),
        f5_summary=read("dynamics_diagnostics/f5_diagnostics_summary.json"),
        dk2018_summary=read(
            "lyapunov_methods/fractional_variational_dk2018_block_restart_abm_gs_published/validation_summary.json"
        ),
        fischer2020_summary=read(
            "lyapunov_methods/fractional_cloned_dynamics_abm_gs_published/validation_summary.json"
        ),
        diagnostic_scope_statement_exists=True,
    )
    assert summary["input_statuses"]["f6_status"] == "not_available"
    assert summary["input_statuses"]["f7_status"] == "not_available"
    assert summary["structured_diagnostics_closed"] is True
