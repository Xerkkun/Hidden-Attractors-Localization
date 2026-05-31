from __future__ import annotations

import json
from pathlib import Path
from hidden_attractors.paths import PROJECT_ROOT

def test_current_protocol_evidence_scope_and_contract() -> None:
    # 1. Verify alignment of effective_contract.json with unified_caputo_protocol.json
    protocol_path = PROJECT_ROOT / "configs" / "unified_caputo_protocol.json"
    effective_path = PROJECT_ROOT / "validation" / "01_numerical_contract" / "effective_contract.json"
    
    assert protocol_path.exists()
    assert effective_path.exists()
    
    with open(protocol_path, "r", encoding="utf-8") as f:
        protocol_data = json.load(f)
        
    with open(effective_path, "r", encoding="utf-8") as f:
        effective_data = json.load(f)
        
    # Check key fields align
    assert effective_data["q"] == protocol_data["numerical_contract"]["q"]
    assert effective_data["h"] == protocol_data["numerical_contract"]["h"]
    assert effective_data["t_final"] == protocol_data["numerical_contract"]["t_final"]
    assert effective_data["t_transient"] == protocol_data["numerical_contract"]["t_transient"]
    assert effective_data["backend"] == protocol_data["numerical_contract"]["backend"]
    assert effective_data["memory_policy"] == protocol_data["numerical_contract"]["memory_policy"]
    assert effective_data["memory_length"] == protocol_data["numerical_contract"]["memory_length"]
    
    # 2. Verify stage summaries 01-07 are marked as official_validation_run and current_contract_applied: True
    validation_root = PROJECT_ROOT / "validation"
    manifest_path = validation_root / "00_manifest" / "validation_manifest.json"
    assert manifest_path.exists()
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
        
    for stage_name in [
        "numerical_contract", "algebraic_validation", "seed_generation",
        "soft_precheck", "continuation", "post_continuation_filter", "dynamic_reference"
    ]:
        # Check in manifest
        assert manifest_data["stage_evidence_scopes"][stage_name] == "official_validation_run"
        
        # Check in individual summary file
        stage_path = manifest_data["stages"][stage_name]
        summary_path = validation_root / stage_path
        assert summary_path.exists()
        
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
            
        assert summary_data["evidence_scope"]["classification"] == "official_validation_run"
        assert summary_data["evidence_scope"]["current_contract_applied"] is True

        # 3. Verify embedded numerical_contract block in stage summaries exactly matches effective_contract.json
        if stage_name != "algebraic_validation":
            assert summary_data["numerical_contract"]["hiddenness_radii"] == effective_data["hiddenness_radii"]
            assert summary_data["numerical_contract"]["samples_per_radius"] == effective_data["samples_per_radius"]
            assert summary_data["numerical_contract"]["sample_growth_per_radius"] == effective_data["sample_growth_per_radius"]
            assert summary_data["numerical_contract"]["h"] == effective_data["h"]
            assert summary_data["numerical_contract"]["t_final"] == effective_data["t_final"]
            assert summary_data["numerical_contract"]["random_seed"] == effective_data["random_seed"]

    # 4. Verify seed families classification (Stage 03)
    seed_summary_path = validation_root / manifest_data["stages"]["seed_generation"]
    with open(seed_summary_path, "r", encoding="utf-8") as f:
        seed_summary_data = json.load(f)
    assert seed_summary_data["outputs"]["families"] == ["lure_classical_centered"]
    assert seed_summary_data["outputs"]["legacy_or_exploratory_seed_families"] == [
        "lure_classical_biased", "machado_centered", "machado_biased"
    ]

    # 5. Verify Robustness Deferment Transparency (Stage 08)
    robustness_summary_path = validation_root / manifest_data["stages"]["robustness"]
    with open(robustness_summary_path, "r", encoding="utf-8") as f:
        robustness_summary_data = json.load(f)
    assert robustness_summary_data["executed"] is False
    assert robustness_summary_data["retained_legacy_artifacts"] is True
    assert robustness_summary_data["numerical_contract"]["hiddenness_radii"] == effective_data["hiddenness_radii"]

    # 6. Verify Hiddenness Classification Precision (Stage 09)
    hiddenness_summary_path = validation_root / manifest_data["stages"]["hiddenness_tests"]
    with open(hiddenness_summary_path, "r", encoding="utf-8") as f:
        hiddenness_summary_data = json.load(f)
    assert hiddenness_summary_data["evidence_scope"]["classification"] == "official_summary_for_current_protocol"
    assert hiddenness_summary_data["evidence_scope"]["current_contract_applied"] is True
    assert hiddenness_summary_data["evidence_scope"]["numerical_evidence_executed_under_current_contract"] is False
    assert hiddenness_summary_data["evidence_scope"]["legacy_exploratory_evidence_retained"] is True
    assert hiddenness_summary_data["numerical_contract"]["hiddenness_radii"] == effective_data["hiddenness_radii"]
