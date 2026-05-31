from __future__ import annotations

import json
from pathlib import Path
from hidden_attractors.paths import PROJECT_ROOT

def test_diagnostics_partial_scope() -> None:
    # 1. Verify stage 09 (hiddenness_tests) is hiddenness_exploratory_only
    summary09_path = PROJECT_ROOT / "validation" / "09_hiddenness_tests" / "hiddenness_tests_validation_summary.json"
    assert summary09_path.exists()
    
    with open(summary09_path, "r", encoding="utf-8") as f:
        summary09_data = json.load(f)
        
    assert summary09_data["stage"] == "hiddenness_tests"
    assert summary09_data["status"] == "hiddenness_exploratory_only"
    assert summary09_data["evidence_scope"]["current_contract_applied"] is True
    assert summary09_data["evidence_scope"]["classification"] == "official_validation_run"
    
    # 2. Verify stage 10 (diagnostics) is diagnostics_partial_current_protocol
    summary10_path = PROJECT_ROOT / "validation" / "10_diagnostics" / "diagnostics_validation_summary.json"
    assert summary10_path.exists()
    
    with open(summary10_path, "r", encoding="utf-8") as f:
        summary10_data = json.load(f)
        
    assert summary10_data["stage"] == "diagnostics"
    assert summary10_data["status"] == "diagnostics_partial_current_protocol"
    assert summary10_data["evidence_scope"]["current_contract_applied"] is True
    assert summary10_data["evidence_scope"]["classification"] == "official_validation_run"
