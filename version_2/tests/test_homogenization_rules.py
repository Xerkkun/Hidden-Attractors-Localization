"""Automated tests verifying Caputo methodology homogenization and protocol compliance rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest
import numpy as np

# Add legacy tools path to sys.path
LEGACY_ROOT = Path(__file__).resolve().parents[1] / "tools" / "legacy"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

import lure_biased_multiparam_continuation as continuation
import lure_biased_multiparam_search as search
from hidden_attractors.workflows.protocol import (
    HiddennessTestResult,
)

def test_legacy_scripts_raise_error_without_historical_reproduction_mode() -> None:
    # 1. Verify that both search.py and continuation.py raise errors without historical_reproduction_only: true
    bad_cfg = {
        "q": 0.9998,
        "historical_reproduction_only": False,
    }
    tmp_file = Path("configs_tmp_bad_test.yaml")
    try:
        import yaml
        tmp_file.write_text(yaml.dump(bad_cfg), encoding="utf-8")
        
        with pytest.raises(RuntimeError, match="Legacy Route Execution Blocked"):
            continuation.run_continuation_pipeline(tmp_file)
            
        with pytest.raises(RuntimeError, match="Legacy Route Execution Blocked"):
            import argparse
            mock_args = argparse.Namespace(
                output_root=None,
                resume=False,
                search_worker_index=None,
                aggregate_search_workers=False,
                periodicity_only=False,
                aggregate_periodicity_workers=False,
                n_samples=None,
                periodicity_worker_count=1,
                periodicity_worker_index=None,
                run_id=None,
                search_worker_count=1,
                prepare_only=False,
            )
            search.run_search(tmp_file, mock_args)
    finally:
        if tmp_file.exists():
            tmp_file.unlink()

def test_survivor_preserves_early_periodicity_status(monkeypatch, tmp_path: Path) -> None:
    # 2. Verify that a survivor correctly retains its early_periodicity_status
    # and records post_continuation_dynamics_status separately
    item = {
        "candidate_id": "c1",
        "seed_id": "s1",
        "seed_early_periodicity_status": "pre_continuation_periodic",
        "seed_vec": np.array([1.0, 2.0, 3.0]),
        "sigma0": 0.5,
    }
    cfg = {
        "q": 0.9998,
        "continuation": {
            "h": 0.02,
            "memory_length": 20.0,
            "t_block": 10.0,
            "n_blocks": 2,
            "smooth_width": 0.2,
            "routes": ["C1"],
            "divergence_norm": 120.0,
            "equilibrium_tol": 0.001,
        }
    }
    p = {"alpha_chua": 1.0}
    eqs = {"E0": np.array([0.0, 0.0, 0.0])}
    
    # Mock integration to run fast and return bounded classification
    class FakeHistory:
        def __init__(self):
            self.memory_points = 5
            
        @classmethod
        def from_trajectory(cls, *args, **kwargs):
            return cls()

        def as_efork_history(self):
            return None
            
    monkeypatch.setattr("lure_biased_multiparam_continuation.chua.efork3_integrate", lambda *args, **kwargs: np.zeros((10, 4)))
    monkeypatch.setattr("lure_biased_multiparam_continuation.classify_traj", lambda *args, **kwargs: {
        "bounded": True,
        "diverged": False,
        "equilibrium_hit": False,
        "final_class": "bounded_nontrivial",
        "final_x": 1.0,
        "final_y": 2.0,
        "final_z": 3.0,
    })
    monkeypatch.setattr("lure_biased_multiparam_continuation.FractionalHistory", FakeHistory)
    
    rows, path_rows, survivor, elapsed = continuation.run_one_continuation_item(
        item, cfg, p, eqs, tmp_path, []
    )
    
    assert survivor is not None
    # Verify early_periodicity_status is preserved and not overwritten to nonperiodic_post_transient
    assert survivor["early_periodicity_status"] == "pre_continuation_periodic"
    # Verify separate post_continuation_dynamics_status field exists
    assert survivor["post_continuation_dynamics_status"] == "nonperiodic_post_transient"

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
    # 4. Verify that algebraic_validation is not in pending_stages of manifest
    manifest_path = Path(__file__).resolve().parents[1] / "validation" / "00_manifest" / "validation_manifest.json"
    assert manifest_path.exists(), f"Official validation manifest must exist at {manifest_path}"
    
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = manifest.get("stages", {})
    pending = manifest.get("pending_stages", [])
    
    assert "algebraic_validation" in stages, "algebraic_validation must be in official stages"
    assert stages["algebraic_validation"] != "pending", "algebraic_validation stage must not be pending"
    assert "algebraic_validation" not in pending, "algebraic_validation must not be in pending_stages list"

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

