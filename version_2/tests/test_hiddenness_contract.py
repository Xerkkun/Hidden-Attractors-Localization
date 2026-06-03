from __future__ import annotations

import sys
import pytest
import numpy as np
from pathlib import Path

# Add workspace root and version_2 to sys.path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

from hidden_attractors.verification.hiddenness_contract import (
    HiddennessVerificationStatus,
    verify_hiddenness_contract,
)
from hidden_attractors.workflows.config_loader import load_config
from hidden_attractors.verification.sphere_tests import run_sphere_probe_sweep


# 1. Test HiddennessVerificationStatus enum values
def test_verification_status_values():
    assert HiddennessVerificationStatus.NOT_RUN == "NOT_RUN"
    assert HiddennessVerificationStatus.INCOMPLETE_PROTOCOL == "INCOMPLETE_PROTOCOL"
    assert HiddennessVerificationStatus.SEED_NOT_AVAILABLE == "SEED_NOT_AVAILABLE"
    assert HiddennessVerificationStatus.CANDIDATE_NOT_AVAILABLE == "CANDIDATE_NOT_AVAILABLE"
    assert HiddennessVerificationStatus.SELF_EXCITED_CONTACT_DETECTED == "SELF_EXCITED_CONTACT_DETECTED"
    assert HiddennessVerificationStatus.NUMERICAL_FAILURE == "NUMERICAL_FAILURE"
    assert HiddennessVerificationStatus.HIDDEN_COMPATIBLE == "HIDDEN_COMPATIBLE"
    assert HiddennessVerificationStatus.HIDDEN_VERIFIED == "HIDDEN_VERIFIED"


# 2. Test candidate not available check
def test_candidate_not_available():
    equilibria = {"E0": np.zeros(3)}
    res = verify_hiddenness_contract(
        equilibria=equilibria,
        sphere_summary_records=[],
        probe_runs=[],
        required_radii=[1e-2, 1e-3, 1e-4, 1e-5],
        require_candidate_attractor=True,
        seed_reached_attractor=False,
    )
    assert res["hiddenness_status"] == HiddennessVerificationStatus.CANDIDATE_NOT_AVAILABLE
    assert res["hidden_verified"] is False
    assert "Seed did not reach the target attractor candidate." in res["failed_requirements"]


# 3. Test reference tail length requirement
def test_ref_tail_too_short():
    equilibria = {"E0": np.zeros(3)}
    res = verify_hiddenness_contract(
        equilibria=equilibria,
        sphere_summary_records=[],
        probe_runs=[{"equilibrium": "E0", "radius": 1e-2, "destination": "stable_equilibrium"}],
        required_radii=[1e-2],
        min_ref_tail_points=1000,
        ref_tail_size=500,  # Less than min_ref_tail_points
    )
    # The tail verification failure registers in failed_requirements and marks incomplete protocol
    assert res["hiddenness_status"] == HiddennessVerificationStatus.INCOMPLETE_PROTOCOL
    assert res["hidden_verified"] is False
    assert any("Reference attractor tail has fewer than" in req for req in res["failed_requirements"])


# 4. Test unsupported match metric error logging
def test_unsupported_match_metric():
    equilibria = {"E0": np.zeros(3)}
    res = verify_hiddenness_contract(
        equilibria=equilibria,
        sphere_summary_records=[],
        probe_runs=[{"equilibrium": "E0", "radius": 1e-2, "destination": "stable_equilibrium"}],
        required_radii=[1e-2],
        target_match_metric="invalid_metric",
    )
    assert res["hiddenness_status"] == HiddennessVerificationStatus.INCOMPLETE_PROTOCOL
    assert res["hidden_verified"] is False
    assert any("Target match metric" in req and "is not documented" in req for req in res["failed_requirements"])


# 5. Test missing equilibria when require_all_equilibria is True
def test_missing_equilibria():
    equilibria = {"E0": np.zeros(3), "E1": np.ones(3)}
    # Only E0 is tested
    probe_runs = [
        {"equilibrium": "E0", "radius": 1e-2, "destination": "stable_equilibrium"},
        {"equilibrium": "E0", "radius": 1e-3, "destination": "stable_equilibrium"},
        {"equilibrium": "E0", "radius": 1e-4, "destination": "stable_equilibrium"},
        {"equilibrium": "E0", "radius": 1e-5, "destination": "stable_equilibrium"},
    ]
    res = verify_hiddenness_contract(
        equilibria=equilibria,
        sphere_summary_records=[],
        probe_runs=probe_runs,
        required_radii=[1e-2, 1e-3, 1e-4, 1e-5],
        require_all_equilibria=True,
    )
    assert res["hiddenness_status"] == HiddennessVerificationStatus.INCOMPLETE_PROTOCOL
    assert res["hidden_verified"] is False
    assert res["hidden_compatible"] is True  # Compatible because no hits
    assert "E1" in res["missing_equilibria"]


# 6. Test missing radii tested for tested equilibria
def test_missing_radii():
    equilibria = {"E0": np.zeros(3)}
    # Radii 1e-4 and 1e-5 are missing
    probe_runs = [
        {"equilibrium": "E0", "radius": 1e-2, "destination": "stable_equilibrium"},
        {"equilibrium": "E0", "radius": 1e-3, "destination": "stable_equilibrium"},
    ]
    res = verify_hiddenness_contract(
        equilibria=equilibria,
        sphere_summary_records=[],
        probe_runs=probe_runs,
        required_radii=[1e-2, 1e-3, 1e-4, 1e-5],
    )
    assert res["hiddenness_status"] == HiddennessVerificationStatus.INCOMPLETE_PROTOCOL
    assert res["hidden_verified"] is False
    assert len(res["missing_radii_by_equilibrium"]["E0"]) == 2


# 7. Test self-excited contact detection (target attractor hit)
def test_self_excited_contact_detected():
    equilibria = {"E0": np.zeros(3)}
    # At least one target hit
    probe_runs = [
        {"equilibrium": "E0", "radius": 1e-2, "destination": "target_attractor"},
        {"equilibrium": "E0", "radius": 1e-3, "destination": "stable_equilibrium"},
        {"equilibrium": "E0", "radius": 1e-4, "destination": "stable_equilibrium"},
        {"equilibrium": "E0", "radius": 1e-5, "destination": "stable_equilibrium"},
    ]
    res = verify_hiddenness_contract(
        equilibria=equilibria,
        sphere_summary_records=[],
        probe_runs=probe_runs,
        required_radii=[1e-2, 1e-3, 1e-4, 1e-5],
    )
    assert res["hiddenness_status"] == HiddennessVerificationStatus.SELF_EXCITED_CONTACT_DETECTED
    assert res["hidden_verified"] is False
    assert res["hidden_compatible"] is False
    assert res["self_excited_contact_detected"] is True


# 8. Test numerical failures behavior
def test_numerical_failure():
    equilibria = {"E0": np.zeros(3)}
    # Numerical failure in probes, and allow_numerical_failures=False
    probe_runs = [
        {"equilibrium": "E0", "radius": 1e-2, "destination": "numerical_failure"},
        {"equilibrium": "E0", "radius": 1e-3, "destination": "stable_equilibrium"},
        {"equilibrium": "E0", "radius": 1e-4, "destination": "stable_equilibrium"},
        {"equilibrium": "E0", "radius": 1e-5, "destination": "stable_equilibrium"},
    ]
    res = verify_hiddenness_contract(
        equilibria=equilibria,
        sphere_summary_records=[],
        probe_runs=probe_runs,
        required_radii=[1e-2, 1e-3, 1e-4, 1e-5],
        allow_numerical_failures=False,
    )
    assert res["hiddenness_status"] == HiddennessVerificationStatus.NUMERICAL_FAILURE
    assert res["hidden_verified"] is False
    assert res["hidden_compatible"] is True  # Compatible because no target hits


# 9. Test unstable-only limitation (strict_all_equilibria flag)
def test_unstable_only_limitation():
    equilibria = {"E0": np.zeros(3), "E1": np.ones(3)}  # E0 is stable, E1 is unstable
    # We only test unstable equilibrium E1
    probe_runs = [
        {"equilibrium": "E1", "radius": 1e-2, "destination": "stable_equilibrium"},
        {"equilibrium": "E1", "radius": 1e-3, "destination": "stable_equilibrium"},
        {"equilibrium": "E1", "radius": 1e-4, "destination": "stable_equilibrium"},
        {"equilibrium": "E1", "radius": 1e-5, "destination": "stable_equilibrium"},
    ]
    # Even if strict_all_equilibria is False in config, the contract verification
    # should restrict the status to HIDDEN_COMPATIBLE if not all equilibria are tested.
    res_strict = verify_hiddenness_contract(
        equilibria=equilibria,
        sphere_summary_records=[],
        probe_runs=probe_runs,
        required_radii=[1e-2, 1e-3, 1e-4, 1e-5],
        require_all_equilibria=True,
    )
    assert res_strict["hiddenness_status"] == HiddennessVerificationStatus.INCOMPLETE_PROTOCOL
    assert res_strict["hidden_verified"] is False
    assert res_strict["hidden_compatible"] is True

    res_unst = verify_hiddenness_contract(
        equilibria=equilibria,
        sphere_summary_records=[],
        probe_runs=probe_runs,
        required_radii=[1e-2, 1e-3, 1e-4, 1e-5],
        require_all_equilibria=False,
    )
    assert res_unst["hiddenness_status"] == HiddennessVerificationStatus.INCOMPLETE_PROTOCOL
    assert res_unst["hidden_verified"] is False
    assert res_unst["hidden_compatible"] is True


# 10. Test contract satisfied (HIDDEN_VERIFIED)
def test_contract_satisfied(valid_run_metadata):
    equilibria = {"E0": np.zeros(3), "E1": np.ones(3)}
    probe_runs = []
    for eq in equilibria:
        for r in [1e-2, 1e-3, 1e-4, 1e-5]:
            probe_runs.append({"equilibrium": eq, "radius": r, "destination": "stable_equilibrium"})
            
    res = verify_hiddenness_contract(
        equilibria=equilibria,
        sphere_summary_records=[],
        probe_runs=probe_runs,
        required_radii=[1e-2, 1e-3, 1e-4, 1e-5],
        require_all_equilibria=True,
        allow_numerical_failures=False,
        ref_tail_size=1200,
        min_ref_tail_points=1000,
        run_metadata=valid_run_metadata,
        reference_was_robust=True,
        basin_planes=("xy_close", "xy_large", "xz_close", "xz_large", "yz_close", "yz_large"),
    )
    assert res["hiddenness_status"] == HiddennessVerificationStatus.HIDDEN_VERIFIED
    assert res["hidden_verified"] is True
    assert res["hidden_compatible"] is True
    assert len(res["failed_requirements"]) == 0


def test_complete_neighborhood_evidence_without_metadata_is_only_compatible():
    equilibria = {"E0": np.zeros(3)}
    probe_runs = [
        {"equilibrium": "E0", "radius": radius, "destination": "stable_equilibrium"}
        for radius in [1e-2, 1e-3, 1e-4, 1e-5]
    ]
    res = verify_hiddenness_contract(
        equilibria=equilibria,
        sphere_summary_records=[],
        probe_runs=probe_runs,
        required_radii=[1e-2, 1e-3, 1e-4, 1e-5],
        ref_tail_size=1200,
        reference_was_robust=True,
        basin_planes=("xy_close", "xy_large", "xz_close", "xz_large", "yz_close", "yz_large"),
    )
    assert res["hidden_verified"] is False
    assert res["promotion_verdict"] == "compatible_with_hiddenness_under_tested_radii"
    assert "run_metadata is required for a strong candidate promotion" in res["metadata_validation_errors"]


# 11. Test config loader default values
def test_config_loader_hiddenness_defaults(tmp_path):
    yaml_content = """
system:
  system_id: "chua_fractional_saturation"
  q: 0.99
  parameters:
    alpha: 9.0
    beta: 15.0
modes:
  transfer_mode: "fractional"
integrator:
  name: "efork3"
  h: 0.005
stages:
  attractor_only: true
"""
    yaml_file = tmp_path / "test_hiddenness_cfg.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    cfg = load_config(yaml_file)
    
    # Verify that the hiddenness sub-dictionary exists with all defaults
    assert "hiddenness" in cfg
    hid = cfg["hiddenness"]
    assert hid["required_radii"] == [1e-2, 1e-3, 1e-4, 1e-5]
    assert hid["strict_all_equilibria"] is True
    assert hid["allow_numerical_failures"] is False
    assert hid["min_ref_tail_points"] == 1000
    assert hid["min_probe_tail_points"] == 200
    assert hid["target_match_metric"] == "nn_percentile"
    assert hid["target_match_tol"] == 0.5
    assert hid["target_match_nn_percentile"] == 90.0
