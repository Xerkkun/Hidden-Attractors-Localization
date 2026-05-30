"""Tests for fractional_memory_validation phase (Phase C).

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

from validation.python.fractional_memory_validation import (
    classify_fractional_trajectory,
    compare_window_to_full,
    compute_memory_metrics,
    estimate_caputo_tail_bound,
    load_memory_config,
    run_fractional_memory_validation,
    _NO_CLAIM,
    _ALLOWED_DYNAMIC_CLASSES,
    _ALLOWED_WINDOW_STATUSES,
    _ALLOWED_OVERALL_STATUSES,
)

# ---------------------------------------------------------------------------
# Paths to YAML configs
# ---------------------------------------------------------------------------
_VALIDATION_DIR = REPO_ROOT / "validation" / "fractional_memory_validation"
_SATURATION_YAML = _VALIDATION_DIR / "chua_fractional_saturation_memory.yaml"
_ARCTAN_YAML = _VALIDATION_DIR / "chua_fractional_arctan_memory.yaml"


# ===========================================================================
# 1. test_fractional_memory_configs_exist
# ===========================================================================

class TestConfigsExist:
    """Verify that both required YAML config files exist."""

    def test_saturation_yaml_exists(self) -> None:
        assert _SATURATION_YAML.exists(), (
            f"Missing config: {_SATURATION_YAML}"
        )

    def test_arctan_yaml_exists(self) -> None:
        assert _ARCTAN_YAML.exists(), (
            f"Missing config: {_ARCTAN_YAML}"
        )


# ===========================================================================
# 2. test_configs_use_caputo_abm
# ===========================================================================

class TestConfigsUseCaputoABM:
    """Verify that both configs use ABM integrator with q < 1 and full_reference."""

    @pytest.mark.parametrize("yaml_path", [_SATURATION_YAML, _ARCTAN_YAML])
    def test_integrator_is_abm(self, yaml_path: Path) -> None:
        config = load_memory_config(yaml_path)
        assert config["integrator"]["method"] == "ABM", (
            f"Expected ABM integrator in {yaml_path.name}"
        )

    @pytest.mark.parametrize("yaml_path", [_SATURATION_YAML, _ARCTAN_YAML])
    def test_q_less_than_one(self, yaml_path: Path) -> None:
        config = load_memory_config(yaml_path)
        q = float(config["q"])
        assert 0.0 < q < 1.0, (
            f"q={q} must be in (0, 1) in {yaml_path.name}"
        )

    @pytest.mark.parametrize("yaml_path", [_SATURATION_YAML, _ARCTAN_YAML])
    def test_full_reference_true(self, yaml_path: Path) -> None:
        config = load_memory_config(yaml_path)
        assert config["memory"]["full_reference"] is True, (
            f"memory.full_reference must be true in {yaml_path.name}"
        )


# ===========================================================================
# 3. test_window_sizes_ordered
# ===========================================================================

class TestWindowSizesOrdered:
    """Verify M1 < M2 < M3 in both configs."""

    @pytest.mark.parametrize("yaml_path", [_SATURATION_YAML, _ARCTAN_YAML])
    def test_windows_strictly_increasing(self, yaml_path: Path) -> None:
        config = load_memory_config(yaml_path)
        windows = config["memory"]["windows"]
        sizes = [w["M"] for w in windows]
        assert len(sizes) >= 2, f"Need at least 2 windows in {yaml_path.name}"
        for i in range(len(sizes) - 1):
            assert sizes[i] < sizes[i + 1], (
                f"Window sizes not strictly increasing in {yaml_path.name}: {sizes}"
            )


# ===========================================================================
# 4. test_no_pointwise_comparison
# ===========================================================================

class TestNoPointwiseComparison:
    """Verify comparison_policy.pointwise_comparison_used == false."""

    @pytest.mark.parametrize("yaml_path", [_SATURATION_YAML, _ARCTAN_YAML])
    def test_pointwise_comparison_false(self, yaml_path: Path) -> None:
        config = load_memory_config(yaml_path)
        policy = config["comparison_policy"]
        assert policy.get("pointwise_comparison_used") is False, (
            f"pointwise_comparison_used must be false in {yaml_path.name}"
        )


# ===========================================================================
# 5. test_rho_attractor_translation_invariant
# ===========================================================================

class TestRhoAttractorTranslationInvariant:
    """rho_attractor must be invariant under constant shifts of the trajectory."""

    def test_translation_invariance(self) -> None:
        rng = np.random.default_rng(42)
        N = 200
        dim = 3
        t_burn = 0.0
        times = np.linspace(0.0, 10.0, N)
        states = rng.standard_normal((N, dim))

        metrics_orig = compute_memory_metrics(times, states, t_burn)

        # Shift by a constant vector
        shift = np.array([10.0, -5.0, 3.14])
        states_shifted = states + shift[np.newaxis, :]
        metrics_shifted = compute_memory_metrics(times, states_shifted, t_burn)

        rho_orig = metrics_orig["rho_attractor"]
        rho_shifted = metrics_shifted["rho_attractor"]

        assert abs(rho_orig - rho_shifted) < 1e-10, (
            f"rho_attractor not translation-invariant: "
            f"orig={rho_orig}, shifted={rho_shifted}"
        )

    def test_rho_max_translation_invariant(self) -> None:
        rng = np.random.default_rng(7)
        N = 100
        times = np.linspace(0.0, 5.0, N)
        states = rng.standard_normal((N, 3))
        shift = np.array([100.0, -200.0, 50.0])

        m1 = compute_memory_metrics(times, states, 0.0)
        m2 = compute_memory_metrics(times, states + shift, 0.0)

        assert abs(m1["rho_max"] - m2["rho_max"]) < 1e-10


# ===========================================================================
# 6. test_classifier_detects_nan
# ===========================================================================

class TestClassifierDetectsNaN:
    """Classifier returns 'nan_detected' when trajectory has NaN."""

    def test_nan_in_states(self) -> None:
        times = np.linspace(0.0, 10.0, 50)
        states = np.ones((50, 3))
        states[25, 1] = float("nan")
        result = classify_fractional_trajectory(
            times, states, t_burn=0.0,
            divergence_norm=120.0,
            collapse_variance_tolerance=1e-8,
            min_range_tolerance=1e-5,
        )
        assert result == "nan_detected"

    def test_inf_in_states(self) -> None:
        times = np.linspace(0.0, 10.0, 50)
        states = np.ones((50, 3))
        states[-1, 0] = float("inf")
        result = classify_fractional_trajectory(
            times, states, t_burn=0.0,
            divergence_norm=120.0,
            collapse_variance_tolerance=1e-8,
            min_range_tolerance=1e-5,
        )
        assert result == "nan_detected"


# ===========================================================================
# 7. test_classifier_detects_divergence
# ===========================================================================

class TestClassifierDetectsDivergence:
    """Classifier returns 'diverged' when max ||X|| exceeds divergence_norm."""

    def test_diverged(self) -> None:
        times = np.linspace(0.0, 10.0, 50)
        states = np.ones((50, 3)) * 5.0
        # One point very large
        states[40] = np.array([200.0, 0.0, 0.0])
        result = classify_fractional_trajectory(
            times, states, t_burn=0.0,
            divergence_norm=120.0,
            collapse_variance_tolerance=1e-8,
            min_range_tolerance=1e-5,
        )
        assert result == "diverged"

    def test_not_diverged_below_norm(self) -> None:
        times = np.linspace(0.0, 10.0, 50)
        states = np.ones((50, 3)) * 5.0
        result = classify_fractional_trajectory(
            times, states, t_burn=0.0,
            divergence_norm=120.0,
            collapse_variance_tolerance=1e-8,
            min_range_tolerance=1e-5,
        )
        assert result != "diverged"


# ===========================================================================
# 8. test_classifier_detects_collapse
# ===========================================================================

class TestClassifierDetectsCollapse:
    """Classifier returns 'collapsed_to_equilibrium' for near-constant trajectory."""

    def test_collapsed_constant_trajectory(self) -> None:
        times = np.linspace(0.0, 10.0, 100)
        states = np.ones((100, 3)) * 3.0 + np.random.default_rng(0).standard_normal((100, 3)) * 1e-12
        result = classify_fractional_trajectory(
            times, states, t_burn=0.0,
            divergence_norm=120.0,
            collapse_variance_tolerance=1e-8,
            min_range_tolerance=1e-5,
        )
        assert result == "collapsed_to_equilibrium"

    def test_collapsed_at_equilibrium(self) -> None:
        times = np.linspace(0.0, 10.0, 100)
        # All states exactly at one point
        states = np.tile([1.0, -2.0, 0.5], (100, 1))
        result = classify_fractional_trajectory(
            times, states, t_burn=0.0,
            divergence_norm=120.0,
            collapse_variance_tolerance=1e-8,
            min_range_tolerance=1e-5,
        )
        assert result == "collapsed_to_equilibrium"


# ===========================================================================
# 9. test_classifier_detects_bounded_nontrivial
# ===========================================================================

class TestClassifierDetectsBoundedNontrivial:
    """Classifier returns 'bounded_nontrivial' for well-spread bounded trajectory."""

    def test_bounded_nontrivial(self) -> None:
        rng = np.random.default_rng(123)
        N = 200
        times = np.linspace(0.0, 20.0, N)
        # Oscillating trajectory, bounded, high variance
        t = np.linspace(0.0, 20.0, N)
        states = np.column_stack([
            5.0 * np.sin(t),
            3.0 * np.cos(2.0 * t),
            2.0 * np.sin(3.0 * t + 1.0),
        ])
        result = classify_fractional_trajectory(
            times, states, t_burn=0.0,
            divergence_norm=120.0,
            collapse_variance_tolerance=1e-8,
            min_range_tolerance=1e-5,
        )
        assert result == "bounded_nontrivial"


# ===========================================================================
# 10. test_memory_warning_on_class_change
# ===========================================================================

class TestMemoryWarningOnClassChange:
    """compare_window_to_full must warn and set status=window_memory_insufficient
    when dynamic class changes between full and window runs."""

    def _make_metrics(self, rho: float, rho_max: float) -> dict:
        """Helper: create artificial metrics."""
        return {
            "rho_attractor": rho,
            "rho_max": rho_max,
            "mean_vector": [0.0, 0.0, 0.0],
            "std_vector": [1.0, 1.0, 1.0],
            "variance_vector": [1.0, 1.0, 1.0],
            "min_vector": [-5.0, -5.0, -5.0],
            "max_vector": [5.0, 5.0, 5.0],
            "range_vector": [10.0, 10.0, 10.0],
        }

    def test_class_change_gives_insufficient(self) -> None:
        full_metrics = self._make_metrics(rho=3.0, rho_max=5.0)
        window_metrics = self._make_metrics(rho=3.1, rho_max=5.1)

        policy = {
            "pointwise_comparison_used": False,
            "rho_relative_tolerance": 0.25,
            "rho_max_relative_tolerance": 0.35,
            "range_relative_tolerance": 0.35,
            "center_relative_tolerance": 0.60,
        }

        result = compare_window_to_full(
            window_metrics=window_metrics,
            full_metrics=full_metrics,
            dynamic_class_window="collapsed_to_equilibrium",
            dynamic_class_full="bounded_nontrivial",
            policy=policy,
        )

        assert result["warning"] is True
        assert result["class_changed"] is True
        assert result["status"] == "window_memory_insufficient"

    def test_no_class_change_no_excess_gives_sufficient(self) -> None:
        metrics = self._make_metrics(rho=3.0, rho_max=5.0)
        policy = {
            "pointwise_comparison_used": False,
            "rho_relative_tolerance": 0.25,
            "rho_max_relative_tolerance": 0.35,
            "range_relative_tolerance": 0.35,
            "center_relative_tolerance": 0.60,
        }
        result = compare_window_to_full(
            window_metrics=metrics,
            full_metrics=metrics,
            dynamic_class_window="bounded_nontrivial",
            dynamic_class_full="bounded_nontrivial",
            policy=policy,
        )
        assert result["warning"] is False
        assert result["status"] == "window_memory_sufficient"


# ===========================================================================
# 11. test_memory_warning_on_large_rho_error
# ===========================================================================

class TestMemoryWarningOnLargeRhoError:
    """compare_window_to_full must warn when rho_relative_error exceeds tolerance."""

    def _make_metrics(self, rho: float, rho_max: float) -> dict:
        return {
            "rho_attractor": rho,
            "rho_max": rho_max,
            "mean_vector": [0.0, 0.0, 0.0],
            "std_vector": [1.0, 1.0, 1.0],
            "variance_vector": [1.0, 1.0, 1.0],
            "min_vector": [-5.0, -5.0, -5.0],
            "max_vector": [5.0, 5.0, 5.0],
            "range_vector": [10.0, 10.0, 10.0],
        }

    def test_large_rho_error_triggers_warning(self) -> None:
        full_metrics = self._make_metrics(rho=3.0, rho_max=5.0)
        # rho 100% bigger → rho_relative_error ≈ 1.0 >> 0.25 tolerance
        window_metrics = self._make_metrics(rho=6.0, rho_max=5.0)

        policy = {
            "pointwise_comparison_used": False,
            "rho_relative_tolerance": 0.25,
            "rho_max_relative_tolerance": 0.35,
            "range_relative_tolerance": 0.35,
            "center_relative_tolerance": 0.60,
        }

        result = compare_window_to_full(
            window_metrics=window_metrics,
            full_metrics=full_metrics,
            dynamic_class_window="bounded_nontrivial",
            dynamic_class_full="bounded_nontrivial",
            policy=policy,
        )

        assert result["warning"] is True
        assert result["rho_relative_error"] > 0.25
        # class not changed → sensitive, not insufficient
        assert result["status"] == "window_memory_sensitive"


# ===========================================================================
# 12. test_tail_bound_nonnegative
# ===========================================================================

class TestTailBoundNonnegative:
    """estimate_caputo_tail_bound must always return a non-negative value."""

    @pytest.mark.parametrize("q,t_final,M,h,K", [
        (0.9, 10.0, 256, 0.01, 1.0),
        (0.5, 5.0, 100, 0.01, 5.0),
        (0.99, 80.0, 1024, 0.01, 10.0),
        (0.9998, 80.0, 256, 0.01, 2.5),
        (0.7, 10.0, 2000, 0.01, 0.1),   # window > t_final → 0
    ])
    def test_bound_nonnegative(self, q, t_final, M, h, K) -> None:
        bound = estimate_caputo_tail_bound(
            q=q,
            t_final=t_final,
            t_burn=t_final / 2,
            memory_window_steps=M,
            h=h,
            derivative_bound=K,
        )
        assert bound >= 0.0, (
            f"tail_bound={bound} is negative for q={q}, M={M}, K={K}"
        )

    def test_window_larger_than_tspan_gives_zero(self) -> None:
        """If L = M*h >= t_final, tail bound must be 0."""
        bound = estimate_caputo_tail_bound(
            q=0.9, t_final=1.0, t_burn=0.5,
            memory_window_steps=200, h=0.01,   # L=2.0 > t_final=1.0
            derivative_bound=5.0,
        )
        assert bound == 0.0


# ===========================================================================
# 13. test_run_memory_validation_fast_smoke
# ===========================================================================

class TestRunMemoryValidationFastSmoke:
    """Fast smoke test: run one case with fast=True and verify output files."""

    def test_saturation_case_creates_outputs(self, tmp_path: Path) -> None:
        """Run saturation case fast and check all three output files exist."""
        summary = run_fractional_memory_validation(
            config_path=_SATURATION_YAML,
            output_dir=tmp_path,
            fast=True,
            save_trajectories=False,
        )

        case_id = summary["case_id"]
        case_out = tmp_path / case_id

        assert (case_out / "memory_window_summary.csv").exists(), (
            "memory_window_summary.csv not created"
        )
        assert (case_out / "memory_comparison.csv").exists(), (
            "memory_comparison.csv not created"
        )
        assert (case_out / "memory_validation_summary.json").exists(), (
            "memory_validation_summary.json not created"
        )

    def test_arctan_case_creates_outputs(self, tmp_path: Path) -> None:
        """Run arctan case fast and check all three output files exist."""
        summary = run_fractional_memory_validation(
            config_path=_ARCTAN_YAML,
            output_dir=tmp_path,
            fast=True,
            save_trajectories=False,
        )

        case_id = summary["case_id"]
        case_out = tmp_path / case_id

        assert (case_out / "memory_window_summary.csv").exists()
        assert (case_out / "memory_comparison.csv").exists()
        assert (case_out / "memory_validation_summary.json").exists()

    def test_summary_json_is_valid(self, tmp_path: Path) -> None:
        """Summary JSON must be parseable and contain required keys."""
        summary = run_fractional_memory_validation(
            config_path=_SATURATION_YAML,
            output_dir=tmp_path,
            fast=True,
        )

        required_keys = [
            "stage", "case_id", "system_id", "q", "integrator",
            "full_memory_reference_present", "windows_tested",
            "overall_status", "automatic_warnings",
            "pointwise_comparison_used",
            "hiddenness_certified_by_this_pipeline",
            "no_hidden_verified_claim",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key '{key}' in summary"

        assert summary["overall_status"] in _ALLOWED_OVERALL_STATUSES, (
            f"Unexpected overall_status: {summary['overall_status']}"
        )


# ===========================================================================
# 14. test_summary_no_hidden_verified
# ===========================================================================

class TestSummaryNoHiddenVerified:
    """No output summary should declare hidden_verified."""

    def test_no_claim_dict_has_no_hidden_verified(self) -> None:
        assert "hidden_verified" not in _NO_CLAIM
        assert _NO_CLAIM["hiddenness_certified_by_this_pipeline"] is False
        assert _NO_CLAIM["no_hidden_verified_claim"] is True
        assert _NO_CLAIM["pointwise_comparison_used"] is False

    def test_run_summary_no_hidden_verified(self, tmp_path: Path) -> None:
        summary = run_fractional_memory_validation(
            config_path=_SATURATION_YAML,
            output_dir=tmp_path,
            fast=True,
        )

        # Key must not appear
        assert "hidden_verified" not in summary, (
            "summary must not contain 'hidden_verified' key"
        )
        # Mandatory flags
        assert summary["hiddenness_certified_by_this_pipeline"] is False
        assert summary["no_hidden_verified_claim"] is True
        assert summary["pointwise_comparison_used"] is False

    def test_run_summary_hiddenness_false(self, tmp_path: Path) -> None:
        summary = run_fractional_memory_validation(
            config_path=_ARCTAN_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        assert summary["hiddenness_certified_by_this_pipeline"] is False

    def test_summary_json_file_no_hidden_verified(self, tmp_path: Path) -> None:
        """Also check the written JSON file."""
        summary = run_fractional_memory_validation(
            config_path=_SATURATION_YAML,
            output_dir=tmp_path,
            fast=True,
        )
        json_path = tmp_path / summary["case_id"] / "memory_validation_summary.json"
        with json_path.open("r") as fh:
            loaded = json.load(fh)

        assert "hidden_verified" not in loaded
        assert loaded["hiddenness_certified_by_this_pipeline"] is False
        assert loaded["no_hidden_verified_claim"] is True
