"""F5 diagnostics remain structured but conservatively non-certifying."""

from __future__ import annotations

import json
from pathlib import Path

from validation.python.run_poincare_diagnostics import _final_f5_status


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "validation"
    / "chaos_validation"
    / "dynamics_diagnostics"
    / "f5_diagnostics_summary.json"
)
CONTRACT = SUMMARY.parent / "diagnostics_contract.json"


def test_f5_summary_tracks_all_required_subphases() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert {"boundedness", "zero_one", "psd_fft", "poincare"} <= set(summary)
    assert summary["poincare"]["status"] == "completed_structured_outputs"
    assert summary["poincare"]["standardized_outputs"] is True
    assert summary["poincare_method_validation"]["status"] == "passed_synthetic_crossing_tests"
    assert (
        summary["poincare_application_to_published_cases"]["status"]
        == "completed_structured_outputs"
    )


def test_zero_one_is_required_by_f5_contract() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert "zero_one_test" in contract["required_diagnostics"]
    assert "zero_one_test" not in contract["optional_diagnostics"]


def test_generated_subphases_are_standardized_and_structured_ready() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    subphases = [summary[name] for name in ("boundedness", "zero_one", "psd_fft", "poincare")]
    assert all(item["standardized_outputs"] for item in subphases)
    assert summary["final_f5_status"] == "f5_diagnostics_structured_outputs_ready"


def test_pending_subphase_prevents_structured_ready_promotion() -> None:
    subphases = {
        name: {"status": "completed", "standardized_outputs": name != "zero_one"}
        for name in ("boundedness", "zero_one", "psd_fft", "poincare")
    }
    assert _final_f5_status(subphases) == "diagnostics_partial_current_protocol"


def test_all_standardized_outputs_can_mark_structured_ready_without_certification() -> None:
    subphases = {
        name: {"status": "completed", "standardized_outputs": True}
        for name in ("boundedness", "zero_one", "psd_fft", "poincare")
    }
    assert _final_f5_status(subphases) == "f5_diagnostics_structured_outputs_ready"


def test_no_single_diagnostic_certifies_chaos_or_hiddenness() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["certifications"] == {"chaos_verified": False, "hidden_verified": False}
    assert summary["invariants"]["single_indicator_cannot_certify_chaos"] is True
    assert summary["invariants"]["diagnostics_cannot_certify_hiddenness"] is True
    assert summary["invariants"]["poincare_cannot_certify_caputo_periodic_orbits"] is True
    serialized = json.dumps(summary)
    assert '"chaos_verified": true' not in serialized
    assert '"hidden_verified": true' not in serialized
    assert '"poincare_proves_chaos": true' not in serialized
    assert '"boundedness_proves_chaos": true' not in serialized
    assert '"zero_one_proves_chaos": true' not in serialized
    assert '"psd_proves_chaos": true' not in serialized
