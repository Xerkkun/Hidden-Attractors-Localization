from __future__ import annotations

import json
from pathlib import Path
from hidden_attractors.paths import PROJECT_ROOT

def test_no_false_certification() -> None:
    # Verify that stages 08, 09, and 10 are NOT falsely certified as complete
    validation_root = PROJECT_ROOT / "validation"
    manifest_path = validation_root / "00_manifest" / "validation_manifest.json"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
        
    incomplete_stages = ["robustness", "hiddenness_tests", "diagnostics"]
    
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
            assert summary_data["status"] == "pending_not_in_scope_current_phase"
        elif stage_name == "hiddenness_tests":
            assert summary_data["status"] == "hiddenness_exploratory_only"
        elif stage_name == "diagnostics":
            assert summary_data["status"] == "diagnostics_partial_current_protocol"
            
    # Verify they are correctly placed under pending/failed_or_incomplete stages in the manifest
    assert "robustness" in manifest_data["pending_stages"]
    assert "hiddenness_tests" in manifest_data["pending_stages"]
    assert "diagnostics" in manifest_data["pending_stages"]
    
    assert "robustness" in manifest_data["failed_or_incomplete_stages"]
    assert "hiddenness_tests" in manifest_data["failed_or_incomplete_stages"]
    assert "diagnostics" in manifest_data["failed_or_incomplete_stages"]


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
