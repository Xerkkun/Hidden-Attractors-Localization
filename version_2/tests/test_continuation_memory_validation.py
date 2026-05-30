"""Tests for continuation_memory_validation phase (Phase D).

Tests are designed to be fast (unit-level). Long integration tests are
guarded by fast=True to use short t_final from fast_test config.

No test asserts hidden_verified, chaotic, or similar claims.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup: ensure repo root is on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.python.continuation_memory_validation import (
    load_continuation_config,
    get_continuation_system,
    build_eta_rhs,
    extract_history,
    classify_continuation_segment,
    compute_segment_metrics,
    compare_restart_vs_history,
    compare_eta_grids,
    run_continuation_memory_validation,
    _NO_CLAIM,
    _ALLOWED_DYN_CLASSES,
    _ALLOWED_ETA_REFINEMENT_STATUSES,
    _ALLOWED_RESTART_VS_HISTORY,
    _ALLOWED_OVERALL_STATUSES,
)

# ---------------------------------------------------------------------------
# Paths to YAML configs
# ---------------------------------------------------------------------------
_VALIDATION_DIR = REPO_ROOT / "validation" / "continuation_memory_validation"
_SATURATION_YAML = _VALIDATION_DIR / "chua_fractional_saturation_continuation.yaml"
_ARCTAN_YAML = _VALIDATION_DIR / "chua_fractional_arctan_continuation.yaml"


# ===========================================================================
# 1. Configs exist and basic schema checks
# ===========================================================================

class TestConfigsExistAndSchema:
    """Verify that both required YAML config files exist and load properly."""

    def test_configs_exist(self) -> None:
        assert _SATURATION_YAML.exists(), f"Missing config: {_SATURATION_YAML}"
        assert _ARCTAN_YAML.exists(), f"Missing config: {_ARCTAN_YAML}"

    @pytest.mark.parametrize("yaml_path", [_SATURATION_YAML, _ARCTAN_YAML])
    def test_load_config_basic(self, yaml_path: Path) -> None:
        config = load_continuation_config(yaml_path)
        assert config["integrator"]["method"] == "ABM"
        q = float(config["q"])
        assert 0.0 < q < 1.0
        assert config["comparison_policy"]["pointwise_comparison_used"] is False


# ===========================================================================
# 2. System loader and deformed RHS building
# ===========================================================================

class TestSystemLoaderAndDeformedRHS:
    """Test loading chaotic system matrices and constructing deformed RHS."""

    def test_saturation_system_loader(self) -> None:
        sys_obj, P, b, r, psi = get_continuation_system("chua_fractional_saturation")
        assert P.shape == (3, 3)
        assert b.shape == (3,)
        assert r.shape == (3,)
        assert callable(psi)

    def test_build_eta_rhs_saturation(self) -> None:
        sys_obj, P, b, r, psi = get_continuation_system("chua_fractional_saturation")
        k = -1.2  # Arbitrary DF seed gain

        # Build RHS at eta=0, 0.5, 1
        rhs_0, status_0 = build_eta_rhs(sys_obj, eta=0.0, k=k)
        rhs_mid, status_mid = build_eta_rhs(sys_obj, eta=0.5, k=k)
        rhs_1, status_1 = build_eta_rhs(sys_obj, eta=1.0, k=k)

        assert status_0 == "available"
        assert status_mid == "available"
        assert status_1 == "available"
        assert callable(rhs_0)
        assert callable(rhs_mid)
        assert callable(rhs_1)

        # Evaluate at x = [1.0, 2.0, 3.0]
        x = np.array([1.0, 2.0, 3.0])
        sigma = float(r @ x)

        val_0 = rhs_0(0.0, x)
        val_mid = rhs_mid(0.0, x)
        val_1 = rhs_1(0.0, x)

        expected_0 = P @ x + b * (k * sigma)
        expected_1 = P @ x + b * psi(sigma)
        expected_mid = P @ x + b * (k * sigma + 0.5 * (psi(sigma) - k * sigma))

        assert np.allclose(val_0, expected_0)
        assert np.allclose(val_1, expected_1)
        assert np.allclose(val_mid, expected_mid)

    def test_build_eta_rhs_no_seed(self) -> None:
        """When k=None, build_eta_rhs must return None and status unavailable."""
        sys_obj, P, b, r, psi = get_continuation_system("chua_fractional_arctan")
        rhs, status = build_eta_rhs(sys_obj, eta=0.5, k=None)
        assert rhs is None
        assert status == "continuation_auxiliary_unavailable"


# ===========================================================================
# 3. History extraction test
# ===========================================================================

class TestHistoryExtraction:
    """Test extracting last M points from times and states."""

    def test_extract_history_basic(self) -> None:
        times = np.linspace(0.0, 10.0, 100)
        states = np.column_stack([times, 2 * times, 3 * times])
        h_t, h_s = extract_history(times, states, M=10)
        assert len(h_t) == 10
        assert h_s.shape == (10, 3)
        assert np.allclose(h_t, times[-10:])
        assert np.allclose(h_s, states[-10:])

    def test_extract_history_short(self) -> None:
        """If trajectory has fewer than M points, extract all of them."""
        times = np.linspace(0.0, 2.0, 5)
        states = np.ones((5, 3))
        h_t, h_s = extract_history(times, states, M=10)
        assert len(h_t) == 5
        assert h_s.shape == (5, 3)
        assert np.allclose(h_t, times)
        assert np.allclose(h_s, states)


# ===========================================================================
# 4. Classifier and metrics delegation
# ===========================================================================

class TestClassifierDelegation:
    """Verify that trajectory classification delegates and works correctly."""

    def test_nan_detected(self) -> None:
        times = np.linspace(0.0, 5.0, 10)
        states = np.ones((10, 3))
        states[5, 0] = float("nan")
        config = {"divergence_norm": 120.0, "collapse_variance_tolerance": 1e-8, "min_range_tolerance": 1e-5}
        cls = classify_continuation_segment(times, states, t_burn=0.0, config=config)
        assert cls == "nan_detected"

    def test_diverged(self) -> None:
        times = np.linspace(0.0, 5.0, 10)
        states = np.ones((10, 3))
        states[-1, 0] = 150.0  # above divergence_norm
        config = {"divergence_norm": 120.0, "collapse_variance_tolerance": 1e-8, "min_range_tolerance": 1e-5}
        cls = classify_continuation_segment(times, states, t_burn=0.0, config=config)
        assert cls == "diverged"

    def test_collapsed(self) -> None:
        times = np.linspace(0.0, 5.0, 10)
        states = np.tile([1.0, -1.0, 0.0], (10, 1))
        config = {"divergence_norm": 120.0, "collapse_variance_tolerance": 1e-8, "min_range_tolerance": 1e-5}
        cls = classify_continuation_segment(times, states, t_burn=0.0, config=config)
        assert cls == "collapsed_to_equilibrium"

    def test_metrics_translation_invariant(self) -> None:
        times = np.linspace(0.0, 10.0, 50)
        states = np.column_stack([np.sin(times), np.cos(times), np.zeros_like(times)])
        m1 = compute_segment_metrics(times, states, t_burn=2.0)
        m2 = compute_segment_metrics(times, states + 10.0, t_burn=2.0)
        assert abs(m1["rho_attractor"] - m2["rho_attractor"]) < 1e-10
        assert abs(m1["rho_max"] - m2["rho_max"]) < 1e-10


# ===========================================================================
# 5. compare_restart_vs_history logic
# ===========================================================================

class TestCompareRestartVsHistory:
    """Test decision logic of restart vs history comparison."""

    def test_consistent(self) -> None:
        restart_grid = [{
            "eta": 1.0,
            "dynamic_class": "bounded_nontrivial",
            "final_state_x": 1.0, "final_state_y": 1.0, "final_state_z": 1.0,
            "rho_attractor": 2.0, "rho_max": 3.0,
            "range_x": 10.0, "range_y": 10.0, "range_z": 10.0,
            "availability": "available",
        }]
        history_grid = [{
            "eta": 1.0,
            "dynamic_class": "bounded_nontrivial",
            "final_state_x": 1.01, "final_state_y": 1.01, "final_state_z": 1.01,
            "rho_attractor": 2.01, "rho_max": 3.01,
            "range_x": 10.01, "range_y": 10.01, "range_z": 10.01,
            "availability": "available",
        }]

        policy = {
            "pointwise_comparison_used": False,
            "final_state_relative_tolerance": 0.50,
            "rho_jump_tolerance": 0.35,
            "range_relative_tolerance": 0.35,
        }

        status, comp, warns = compare_restart_vs_history(restart_grid, history_grid, policy)
        assert status == "restart_and_history_consistent"
        assert comp["class_changed"] is False
        assert comp["warning"] is False

    def test_class_differs(self) -> None:
        restart_grid = [{
            "eta": 1.0,
            "dynamic_class": "collapsed_to_equilibrium",
            "final_state_x": 0.0, "final_state_y": 0.0, "final_state_z": 0.0,
            "rho_attractor": 0.0, "rho_max": 0.0,
            "range_x": 0.01, "range_y": 0.01, "range_z": 0.01,
            "availability": "available",
        }]
        history_grid = [{
            "eta": 1.0,
            "dynamic_class": "bounded_nontrivial",
            "final_state_x": 1.0, "final_state_y": 1.0, "final_state_z": 1.0,
            "rho_attractor": 2.0, "rho_max": 3.0,
            "range_x": 10.0, "range_y": 10.0, "range_z": 10.0,
            "availability": "available",
        }]

        policy = {
            "pointwise_comparison_used": False,
            "final_state_relative_tolerance": 0.50,
            "rho_jump_tolerance": 0.35,
            "range_relative_tolerance": 0.35,
        }

        status, comp, warns = compare_restart_vs_history(restart_grid, history_grid, policy)
        # Class changed and exceeded threshold (final state dist is > 0.50)
        assert status == "restart_artifact_possible"
        assert comp["class_changed"] is True
        assert comp["warning"] is True

    def test_metric_differs(self) -> None:
        # Classes same, but rho differs significantly (5.0 vs 2.0 -> > 0.35 rel diff)
        restart_grid = [{
            "eta": 1.0,
            "dynamic_class": "bounded_nontrivial",
            "final_state_x": 1.0, "final_state_y": 1.0, "final_state_z": 1.0,
            "rho_attractor": 2.0, "rho_max": 3.0,
            "range_x": 10.0, "range_y": 10.0, "range_z": 10.0,
            "availability": "available",
        }]
        history_grid = [{
            "eta": 1.0,
            "dynamic_class": "bounded_nontrivial",
            "final_state_x": 1.0, "final_state_y": 1.0, "final_state_z": 1.0,
            "rho_attractor": 5.0, "rho_max": 3.0,
            "range_x": 10.0, "range_y": 10.0, "range_z": 10.0,
            "availability": "available",
        }]

        policy = {
            "pointwise_comparison_used": False,
            "final_state_relative_tolerance": 0.50,
            "rho_jump_tolerance": 0.35,
            "range_relative_tolerance": 0.35,
        }

        status, comp, warns = compare_restart_vs_history(restart_grid, history_grid, policy)
        assert status == "paper_style_restart_differs_from_caputo_history_transport"
        assert comp["class_changed"] is False
        assert comp["warning"] is True


# ===========================================================================
# 6. compare_eta_grids stability check
# ===========================================================================

class TestCompareEtaGrids:
    """Test grid refinement stability logic."""

    def test_stable_grids(self) -> None:
        grids = {
            25: [{"eta": 1.0, "dynamic_class": "bounded_nontrivial"}],
            50: [{"eta": 1.0, "dynamic_class": "bounded_nontrivial"}],
        }
        status, warns = compare_eta_grids(grids)
        assert status == "continuation_stable_under_eta_refinement"
        assert len(warns) == 0

    def test_unstable_grids(self) -> None:
        # Two largest (50, 100) agree on "bounded_nontrivial", but 25 disagrees
        grids = {
            25: [{"eta": 1.0, "dynamic_class": "collapsed_to_equilibrium"}],
            50: [{"eta": 1.0, "dynamic_class": "bounded_nontrivial"}],
            100: [{"eta": 1.0, "dynamic_class": "bounded_nontrivial"}],
        }
        status, warns = compare_eta_grids(grids)
        assert status == "continuation_requires_eta_refinement"
        assert len(warns) > 0



# ===========================================================================
# 7. run_continuation_memory_validation Fast Smoke Tests
# ===========================================================================

class TestRunContinuationMemoryValidationSmoke:
    """Run fast smoke tests on the two configs and check outputs."""

    def test_saturation_case_smoke(self, tmp_path: Path) -> None:
        summary = run_continuation_memory_validation(
            config_path=_SATURATION_YAML,
            output_dir=tmp_path,
            fast=True,
            save_histories=False,
        )

        case_id = summary["case_id"]
        case_out = tmp_path / case_id

        assert (case_out / "continuation_grid_summary.csv").exists()
        assert (case_out / "restart_vs_history_comparison.csv").exists()
        assert (case_out / "continuation_validation_summary.json").exists()

        assert summary["overall_status"] in _ALLOWED_OVERALL_STATUSES
        assert summary["restart_vs_history_status"] in _ALLOWED_RESTART_VS_HISTORY
        assert summary["eta_refinement_status"] in _ALLOWED_ETA_REFINEMENT_STATUSES

    def test_arctan_case_smoke(self, tmp_path: Path) -> None:
        """Wu 2023 case has k=null, so should return continuation_auxiliary_unavailable."""
        summary = run_continuation_memory_validation(
            config_path=_ARCTAN_YAML,
            output_dir=tmp_path,
            fast=True,
            save_histories=False,
        )

        case_id = summary["case_id"]
        case_out = tmp_path / case_id

        assert (case_out / "continuation_validation_summary.json").exists()
        assert summary["overall_status"] == "continuation_validation_inconclusive"
        assert summary["restart_vs_history_status"] == "continuation_auxiliary_unavailable"


# ===========================================================================
# 8. No-claims enforcement
# ===========================================================================

class TestNoClaimsEnforcement:
    """Ensure no hidden attractor or chaos claims are certified by the code."""

    def test_no_claim_dict(self) -> None:
        assert "hidden_verified" not in _NO_CLAIM
        assert _NO_CLAIM["hiddenness_certified_by_this_pipeline"] is False
        assert _NO_CLAIM["chaos_certified_by_this_pipeline"] is False
        assert _NO_CLAIM["no_hidden_verified_claim"] is True
        assert _NO_CLAIM["pointwise_comparison_used"] is False

    def test_saturation_summary_no_claims(self, tmp_path: Path) -> None:
        summary = run_continuation_memory_validation(
            config_path=_SATURATION_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        assert "hidden_verified" not in summary
        assert summary["hiddenness_certified_by_this_pipeline"] is False
        assert summary["chaos_certified_by_this_pipeline"] is False
        assert summary["no_hidden_verified_claim"] is True
        assert summary["pointwise_comparison_used"] is False

        # Read JSON file too
        json_path = tmp_path / summary["case_id"] / "continuation_validation_summary.json"
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert "hidden_verified" not in data
        assert data["hiddenness_certified_by_this_pipeline"] is False
        assert data["chaos_certified_by_this_pipeline"] is False
        assert data["no_hidden_verified_claim"] is True
