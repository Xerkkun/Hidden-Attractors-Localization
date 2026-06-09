"""Automated tests verifying Caputo methodology homogenization and protocol compliance rules."""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path
import pytest
import numpy as np

# Add active legacy directory to sys.path
ACTIVE_LEGACY = Path(__file__).resolve().parents[1] / "tools" / "legacy"
if str(ACTIVE_LEGACY) not in sys.path:
    sys.path.insert(0, str(ACTIVE_LEGACY))

from hidden_attractors.workflows.protocol import (
    HiddennessTestResult,
)


def test_strong_hiddenness_label_requires_all_six_basin_planes() -> None:
    # 3. Verify that hiddenness_tests validator fails if final_label is strong
    # but the 6 required basin planes are missing
    incomplete = HiddennessTestResult(
        candidate_id="c1",
        tested_equilibria=("E0", "E+", "E-"),
        tested_radii=(1.0e-4,),
        neighborhood_sampling_mode="ball",
        target_contacts=0,
        numerical_failures=0,
        basin_planes=("xy_close",), # Missing others
        reference_was_robust=True,
        final_label="hidden_verified_only_if_full_protocol_passed",
    )
    assert len(incomplete.validate()) > 0
    assert "hidden_verified_only_if_full_protocol_passed requires the complete tested protocol" in incomplete.validate()[0]

def test_algebraic_validation_included_in_manifest() -> None:
    # 4. Verify that algebraic_validation is in stages and in pending_stages when cross-tool validation is pending
    manifest_path = Path(__file__).resolve().parents[1] / "validation" / "00_manifest" / "validation_manifest.json"
    assert manifest_path.exists(), f"Official validation manifest must exist at {manifest_path}"
    
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = manifest.get("stages", {})
    pending = manifest.get("pending_stages", [])
    
    assert "algebraic_validation" in stages, "algebraic_validation must be in official stages"
    
    summary_path = Path(__file__).resolve().parents[1] / "validation" / "02_algebraic_validation" / "algebraic_validation_validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") == "passed_python_wolfram":
        assert "algebraic_validation" not in pending
    else:
        assert "algebraic_validation" in pending, "algebraic_validation must remain pending until cross-tool comparison passes"

def test_pre_continuation_periodic_seed_discard_fails_validation() -> None:
    from hidden_attractors.workflows.protocol import SoftPrecheckResult
    
    # Validation must fail if a pre_continuation_periodic seed is marked as NOT admissible
    invalid_result = SoftPrecheckResult(
        candidate_id="c1",
        label="pre_continuation_periodic",
        admissible_for_continuation=False,
        finite_trajectory=True,
    )
    errors = invalid_result.validate()
    assert len(errors) > 0
    assert "pre-continuation periodicity is diagnostic and cannot reject a seed" in errors[0]
    
    # It should pass if admissible_for_continuation is True
    valid_result = SoftPrecheckResult(
        candidate_id="c1",
        label="pre_continuation_periodic",
        admissible_for_continuation=True,
        finite_trajectory=True,
    )
    assert len(valid_result.validate()) == 0

def test_validation_jsons_do_not_contain_old_efork_label() -> None:
    # Verify that no JSON in version_2/validation/ contains the old EFORK label "K3 = a31*K1 + a32*K2"
    validation_dir = Path(__file__).resolve().parents[1] / "validation"
    for json_file in validation_dir.glob("**/*.json"):
        content = json_file.read_text(encoding="utf-8")
        assert "K3 = a31*K1 + a32*K2" not in content, f"{json_file.name} contains old EFORK label"

def test_pending_stages_not_empty_if_any_stage_incomplete() -> None:
    # Verify that pending_stages is not empty if any stage is incomplete, incomplete_or_failed_tolerance_check or incomplete_pending_basin_slices
    manifest_path = Path(__file__).resolve().parents[1] / "validation" / "00_manifest" / "validation_manifest.json"
    assert manifest_path.exists()
    
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pending = manifest.get("pending_stages", [])
    
    # Read all summary jsons to see if any are incomplete
    validation_dir = Path(__file__).resolve().parents[1] / "validation"
    incomplete_stages = []
    for json_file in validation_dir.glob("**/0*_validation_summary.json"):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        status = data.get("status", "")
        if status.startswith("incomplete") or status.startswith("failed") or status == "passed_internal_pending_external_cross_tool":
            incomplete_stages.append(data.get("stage"))
            
    if incomplete_stages:
        assert len(pending) > 0, "pending_stages must not be empty if any stage is incomplete"
        for stage in incomplete_stages:
            assert stage in pending, f"{stage} is incomplete and must be in pending_stages"


def test_official_manifest_pending_stages_match_real_stage_summaries() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "configs" / "validation_contract.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "validation" / "00_manifest" / "validation_manifest.json").read_text(encoding="utf-8"))
    closed_statuses = {
        "completed",
        "passed_python_wolfram",
        "completed_self_excited_contact_detected",
    }
    expected_pending = []
    for stage in contract["stages"]:
        summary_path = root / "validation" / stage["id"] / stage["summary"]
        if not summary_path.exists():
            expected_pending.append(stage["slug"])
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") not in closed_statuses:
            expected_pending.append(stage["slug"])
    assert manifest["pending_stages"] == expected_pending


def test_algebraic_validation_failed_cross_tool_cannot_be_closed() -> None:
    # Verify that if algebraic_validation fails cross-tool comparison, it remains in failed_or_incomplete_stages or pending_stages
    manifest_path = Path(__file__).resolve().parents[1] / "validation" / "00_manifest" / "validation_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    algebra_summary_path = Path(__file__).resolve().parents[1] / "validation" / "02_algebraic_validation" / "algebraic_validation_validation_summary.json"
    assert algebra_summary_path.exists()
    summary = json.loads(algebra_summary_path.read_text(encoding="utf-8"))
    
    status = summary.get("status", "")
    metrics = summary.get("metrics", {})
    
    has_failed_cross_tool = not (metrics.get("equilibrium_cross_tool_pass", False) and metrics.get("jacobian_cross_tool_pass", False) and metrics.get("eigenvalue_cross_tool_pass", False))
    if has_failed_cross_tool:
        assert status in {"passed_internal_pending_external_cross_tool", "incomplete_or_failed_tolerance_check", "failed_cross_tool_comparison"}
        if status == "incomplete_or_failed_tolerance_check" or status == "failed_cross_tool_comparison":
            assert "algebraic_validation" in manifest.get("failed_or_incomplete_stages", [])
        elif status == "passed_internal_pending_external_cross_tool":
            assert "algebraic_validation" in manifest.get("pending_stages", [])

def test_lightweight_hiddenness_cannot_be_promoted_to_full() -> None:
    # Verify that if a hiddenness run is lightweight, it cannot be promoted to full
    from hidden_attractors.workflows.protocol import HiddennessTestResult
    
    # Attempting to validate with a strong label under a lightweight protocol (e.g. no basin planes or not robust reference) must fail
    res = HiddennessTestResult(
        candidate_id="c1",
        tested_equilibria=("E0", "E+", "E-"),
        tested_radii=(1.0e-4,),
        neighborhood_sampling_mode="ball",
        target_contacts=0,
        numerical_failures=0,
        basin_planes=(), # Empty basin planes -> lightweight/incomplete
        reference_was_robust=True,
        final_label="hidden_verified_only_if_full_protocol_passed",
    )
    assert len(res.validate()) > 0
    assert "hidden_verified_only_if_full_protocol_passed requires the complete tested protocol" in res.validate()[0]
