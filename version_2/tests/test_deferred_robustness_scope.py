from __future__ import annotations

import json
from pathlib import Path
from hidden_attractors.paths import PROJECT_ROOT

def test_deferred_robustness_scope() -> None:
    # Verify that robustness (stage 08) is marked as pending_not_in_scope_current_phase and has no verdict
    summary_path = PROJECT_ROOT / "validation" / "08_robustness" / "robustness_validation_summary.json"
    assert summary_path.exists()
    
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
        
    assert summary_data["stage"] == "robustness"
    assert summary_data["status"] == "pending_not_in_scope_current_phase"
    assert summary_data["verdict"] is None
    assert summary_data["evidence_scope"]["current_contract_applied"] is True
    assert summary_data["evidence_scope"]["classification"] == "official_validation_run"
