from __future__ import annotations

import json
from pathlib import Path
from hidden_attractors.paths import PROJECT_ROOT

def test_deferred_robustness_scope() -> None:
    # The frozen record closes this stage as not evaluated, without a verdict.
    summary_path = PROJECT_ROOT / "validation" / "08_robustness" / "robustness_validation_summary.json"
    assert summary_path.exists()
    
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
        
    assert summary_data["stage"] == "robustness"
    assert summary_data["status"] == "not_evaluated_under_frozen_contract"
    assert summary_data["verdict"] is None
    assert summary_data["evidence_scope"]["current_contract_applied"] is True
    assert (
        summary_data["evidence_scope"]["classification"]
        == "not_evaluated_under_frozen_contract"
    )
    assert summary_data["closure"]["classification"] == "not_evaluated"
