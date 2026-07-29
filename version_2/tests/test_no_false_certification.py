from __future__ import annotations

import json
from pathlib import Path
from hidden_attractors.paths import PROJECT_ROOT

def test_no_false_certification() -> None:
    # Verify that non-evaluated/inconclusive stages remain non-certifying.
    validation_root = PROJECT_ROOT / "validation"
    manifest_path = validation_root / "00_manifest" / "validation_manifest.json"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
        
    incomplete_stages = ["robustness", "diagnostics"]
    
    for stage_name in incomplete_stages:
        stage_path = manifest_data["stages"][stage_name]
        summary_path = validation_root / stage_path
        
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
            
        # Ensure that although current_contract_applied is True, the status is NOT "completed"
        assert summary_data["evidence_scope"]["current_contract_applied"] is True
        assert summary_data["status"] != "completed"
        assert summary_data["status"] != "passed_python_wolfram"
        
        # Verify specific honest statuses
        if stage_name == "robustness":
            assert summary_data["status"] == "not_evaluated_under_frozen_contract"
        elif stage_name == "diagnostics":
            assert summary_data["status"] == "diagnostics_partial_current_protocol"

    hiddenness_path = validation_root / manifest_data["stages"]["hiddenness_tests"]
    with open(hiddenness_path, "r", encoding="utf-8") as f:
        hiddenness_data = json.load(f)

    assert hiddenness_data["status"] == "completed_self_excited_contact_detected"
    assert hiddenness_data["verdict"] == "chaotic_self_excited_candidate_not_hidden_under_tested_equilibrium_neighborhoods"
    assert hiddenness_data["metrics"]["target_hits_E_plus"] > 0
    assert hiddenness_data["metrics"]["target_hits_E_minus"] > 0
    serialized_hiddenness = json.dumps(hiddenness_data).lower()
    assert '"hidden_verified": true' not in serialized_hiddenness
    assert '"hiddenness_verified": true' not in serialized_hiddenness
            
    assert manifest_data["pending_stages"] == []
    assert manifest_data["failed_or_incomplete_stages"] == []
    assert manifest_data["not_evaluated_stages"] == ["robustness"]
    assert manifest_data["inconclusive_stages"] == ["diagnostics"]
    assert (
        manifest_data["final_report"]["status"]
        == "not_produced_under_frozen_contract"
    )
    assert manifest_data["final_report"]["source"] is None
    assert manifest_data["final_report"]["compiled"] is None


def test_f5_outputs_have_no_false_certification() -> None:
    root = PROJECT_ROOT / "validation" / "chaos_validation" / "dynamics_diagnostics"
    serialized = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in root.rglob("*.json")
    )
    for forbidden in (
        '"boundedness_proves_chaos": true',
        '"zero_one_proves_chaos": true',
        '"psd_proves_chaos": true',
        '"poincare_proves_chaos": true',
        '"chaos_verified": true',
        '"hidden_verified": true',
    ):
        assert forbidden not in serialized


def test_f6_f7_outputs_have_no_false_certification() -> None:
    root = PROJECT_ROOT / "validation" / "chaos_validation"
    serialized = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for folder in ("integrated_chaos_validator", "method_comparison")
        for path in (root / folder).rglob("*.json")
    )
    for forbidden in (
        '"integrated_validator_proves_chaos": true',
        '"method_comparison_proves_chaos": true',
        '"chaos_verified": true',
        '"hidden_verified": true',
        '"f6_chaos_proof": true',
        '"f7_chaos_proof": true',
        '"fractional_lyapunov_validated": true',
        '"caputo_lyapunov_validated": true',
    ):
        assert forbidden not in serialized


def test_phase_f_closure_has_no_false_certification() -> None:
    root = PROJECT_ROOT / "validation" / "chaos_validation"
    serialized = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for folder in ("phase_F_closure", "lyapunov_methods/F4_internal_validation")
        for path in (root / folder).rglob("*.json")
    )
    for forbidden in (
        '"strict_chaos_validation_closed": true',
        '"chaos_verified": true',
        '"hiddenness_verified": true',
        "f_strict_chaos_validation_closed",
        '"fractional_lyapunov_validated_by_f4": true',
        '"caputo_lyapunov_validated_by_f4": true',
    ):
        assert forbidden not in serialized
