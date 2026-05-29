"""Tests for the integrator crosscheck validation phase.

These tests verify:
1. Config files exist and have correct schema.
2. Trajectory classifier returns correct states for synthetic trajectories.
3. Metric comparison does not use pointwise comparison.
4. h-sensitivity and memory sensitivity evaluation logic.
5. No hidden_verified claim is made.
6. Fast smoke test of a full case run.

Tests in this file are intentionally fast: they avoid long integrations.
Use ``fast=True`` (or the ``--fast`` flag on the runner) for smoke tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ── Module-level path helpers ──────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
CROSSCHECK_DIR = REPO_ROOT / "validation" / "integrator_crosscheck"
OUTPUT_DIR = REPO_ROOT / "validation" / "outputs" / "integrator_crosscheck"

# ── Import crosscheck module ──────────────────────────────────────────────

import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.python.integrator_crosscheck import (
    classify_trajectory,
    compute_post_transient_metrics,
    compare_metric_to_reference,
    evaluate_h_sensitivity,
    evaluate_memory_sensitivity,
    load_crosscheck_config,
    run_integrator_crosscheck_case,
    _ALLOWED_MEMORY_SENSITIVITY_STATES,
    _ALLOWED_H_SENSITIVITY_STATES,
    _ALLOWED_TRAJECTORY_STATES,
    _NO_CLAIM,
)


# =============================================================================
# 1. Config files exist
# =============================================================================

EXPECTED_CONFIGS = [
    "chua_integer_saturation.yaml",
    "chua_fractional_saturation.yaml",
    "chua_fractional_arctan.yaml",
]


@pytest.mark.parametrize("fname", EXPECTED_CONFIGS)
def test_integrator_crosscheck_configs_exist(fname: str) -> None:
    """All required YAML configuration files must exist."""
    path = CROSSCHECK_DIR / fname
    assert path.exists(), f"Config file not found: {path}"


# =============================================================================
# 2. All configs have pointwise_comparison_used = false
# =============================================================================

@pytest.mark.parametrize("fname", EXPECTED_CONFIGS)
def test_crosscheck_policy_not_pointwise_by_default(fname: str) -> None:
    """Every YAML config must declare pointwise_comparison_used = false."""
    config = load_crosscheck_config(CROSSCHECK_DIR / fname)
    policy = config.get("comparison_policy", {})
    assert policy.get("pointwise_comparison_used") is False, (
        f"{fname}: comparison_policy.pointwise_comparison_used must be false"
    )


# =============================================================================
# 3. Trajectory classifier: nan_detected
# =============================================================================

def test_trajectory_classifier_detects_nan() -> None:
    """Classifier must return nan_detected when the trajectory contains NaN."""
    t = np.linspace(0.0, 10.0, 100)
    states = np.ones((100, 3)) * 0.5
    states[50, 1] = float("nan")
    result = classify_trajectory(t, states, t_burn=5.0)
    assert result == "nan_detected", f"Expected nan_detected, got {result}"


# =============================================================================
# 4. Trajectory classifier: diverged
# =============================================================================

def test_trajectory_classifier_detects_divergence() -> None:
    """Classifier must return diverged when norm exceeds divergence_norm."""
    t = np.linspace(0.0, 10.0, 100)
    states = np.ones((100, 3)) * 1.0
    states[80, :] = np.array([200.0, 200.0, 200.0])  # norm >> 120
    result = classify_trajectory(t, states, t_burn=5.0, divergence_norm=120.0)
    assert result == "diverged", f"Expected diverged, got {result}"


# =============================================================================
# 5. Trajectory classifier: collapsed_to_equilibrium
# =============================================================================

def test_trajectory_classifier_detects_collapse() -> None:
    """Classifier must return collapsed_to_equilibrium for a near-constant trajectory."""
    t = np.linspace(0.0, 10.0, 500)
    # Almost constant post-transient
    states = np.zeros((500, 3))
    states[:, 0] = 3.14  # constant x
    states[:, 1] = 2.71  # constant y
    states[:, 2] = 1.41  # constant z
    result = classify_trajectory(t, states, t_burn=5.0, collapse_variance_tolerance=1e-8)
    assert result == "collapsed_to_equilibrium", (
        f"Expected collapsed_to_equilibrium, got {result}"
    )


# =============================================================================
# 6. Trajectory classifier: bounded_nontrivial
# =============================================================================

def test_trajectory_classifier_detects_bounded_nontrivial() -> None:
    """Classifier must return bounded_nontrivial for a sinusoidal trajectory."""
    t = np.linspace(0.0, 20.0, 2000)
    states = np.column_stack([
        5.0 * np.sin(2.0 * np.pi * 0.3 * t),
        3.0 * np.cos(2.0 * np.pi * 0.7 * t),
        2.0 * np.sin(2.0 * np.pi * 1.1 * t + 0.5),
    ])
    result = classify_trajectory(t, states, t_burn=5.0, divergence_norm=120.0)
    assert result == "bounded_nontrivial", f"Expected bounded_nontrivial, got {result}"


# =============================================================================
# 7. Metric comparison: accepts shifted chaotic cloud (not pointwise)
# =============================================================================

def test_metric_comparison_accepts_shifted_chaotic_cloud() -> None:
    """Two clouds with a small shift should be geometrically consistent.

    This verifies that the comparison is NOT pointwise; a shifted cloud with
    similar range and scale should pass with liberal tolerances.
    """
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 20.0, 2000)
    t_burn = 5.0

    base_cloud = rng.standard_normal((2000, 3)) * np.array([5.0, 3.0, 7.0])
    shifted_cloud = base_cloud + np.array([0.2, 0.1, 0.3])  # small shift

    metrics_ref = compute_post_transient_metrics(t, base_cloud, t_burn)
    metrics_run = compute_post_transient_metrics(t, shifted_cloud, t_burn)

    tolerances = {
        "pointwise_comparison_used": False,
        "range_relative_tolerance": 0.35,
        "center_relative_tolerance": 0.60,
        "scale_relative_tolerance": 0.50,
        "cloud_distance_tolerance": 0.35,
    }
    result = compare_metric_to_reference(metrics_run, metrics_ref, tolerances)

    assert result["pointwise_comparison_used"] is False
    assert result["geometric_consistency"] is True, (
        f"Small shift should be geometrically consistent: {result}"
    )


# =============================================================================
# 8. h-sensitivity: requires_smaller_h
# =============================================================================

def test_integrator_specific_h_allowed() -> None:
    """Large h diverged + small h bounded → requires_smaller_h."""
    results = [
        {
            "run_id": "efork3_h05",
            "method": "EFORK3",
            "h": 0.05,
            "memory_mode": "full",
            "trajectory_class": "diverged",
        },
        {
            "run_id": "efork3_h01",
            "method": "EFORK3",
            "h": 0.01,
            "memory_mode": "full",
            "trajectory_class": "bounded_nontrivial",
        },
        {
            "run_id": "efork3_h005",
            "method": "EFORK3",
            "h": 0.005,
            "memory_mode": "full",
            "trajectory_class": "bounded_nontrivial",
        },
    ]
    status = evaluate_h_sensitivity(results)
    assert status == "requires_smaller_h", (
        f"Expected requires_smaller_h for coarse-fails/fine-ok pattern, got {status}"
    )


# =============================================================================
# 9. Memory sensitivity statuses are from the allowed set
# =============================================================================

def test_memory_window_statuses_exist() -> None:
    """evaluate_memory_sensitivity must return a status from the allowed set."""
    # q=1 case → not_applicable_q1
    results_q1 = [
        {"method": "EFORK_Q1", "memory_mode": "not_applicable", "trajectory_class": "bounded_nontrivial"},
    ]
    s1 = evaluate_memory_sensitivity(results_q1, q=1.0)
    assert s1 in _ALLOWED_MEMORY_SENSITIVITY_STATES, f"Unexpected status: {s1}"
    assert s1 == "not_applicable_q1"

    # No window runs
    results_no_win = [
        {"method": "EFORK3", "memory_mode": "full", "trajectory_class": "bounded_nontrivial"},
    ]
    s2 = evaluate_memory_sensitivity(results_no_win, q=0.9)
    assert s2 in _ALLOWED_MEMORY_SENSITIVITY_STATES, f"Unexpected status: {s2}"
    assert s2 == "no_memory_window_runs"

    # Full = bounded, window = bounded → sufficient
    results_ok = [
        {"method": "ABM", "memory_mode": "full", "trajectory_class": "bounded_nontrivial"},
        {"method": "EFORK3", "memory_mode": "window", "trajectory_class": "bounded_nontrivial"},
    ]
    s3 = evaluate_memory_sensitivity(results_ok, q=0.9)
    assert s3 in _ALLOWED_MEMORY_SENSITIVITY_STATES, f"Unexpected status: {s3}"
    assert s3 == "memory_window_sufficient"

    # Full = bounded, window = diverged → insufficient
    results_bad = [
        {"method": "ABM", "memory_mode": "full", "trajectory_class": "bounded_nontrivial"},
        {"method": "EFORK3", "memory_mode": "window", "trajectory_class": "diverged"},
    ]
    s4 = evaluate_memory_sensitivity(results_bad, q=0.9)
    assert s4 in _ALLOWED_MEMORY_SENSITIVITY_STATES, f"Unexpected status: {s4}"
    assert s4 == "memory_window_insufficient"


# =============================================================================
# 10. No hidden_verified claim in crosscheck outputs
# =============================================================================

def test_no_hidden_verified_claim_in_crosscheck() -> None:
    """The _NO_CLAIM dict must not contain hidden_verified and must have correct flags."""
    # The _NO_CLAIM sentinel used in every summary
    assert "hidden_verified" not in _NO_CLAIM, (
        "'hidden_verified' key must not appear in crosscheck outputs"
    )
    assert _NO_CLAIM.get("hiddenness_certified_by_this_pipeline") is False
    assert _NO_CLAIM.get("no_hidden_verified_claim") is True


# =============================================================================
# 11. Fast smoke test: chua_integer_saturation
# =============================================================================

def test_run_crosscheck_fast_smoke(tmp_path: Path) -> None:
    """Fast smoke test: run the integer saturation case and verify output files."""
    config_path = CROSSCHECK_DIR / "chua_integer_saturation.yaml"
    if not config_path.exists():
        pytest.skip("Config file not found; skipping smoke test.")

    summary = run_integrator_crosscheck_case(
        config_path=config_path,
        output_dir=tmp_path,
        fast=True,
        save_trajectories=False,
        make_figures=False,
    )

    case_id = summary["case_id"]
    case_dir = tmp_path / case_id

    # Required output files
    assert (case_dir / "crosscheck_summary.json").exists(), "crosscheck_summary.json not created"
    assert (case_dir / "individual_runs.csv").exists(), "individual_runs.csv not created"
    assert (case_dir / "run_summary.json").exists(), "run_summary.json not created"

    # Verify no hidden_verified claim
    assert "hidden_verified" not in summary
    assert summary.get("hiddenness_certified_by_this_pipeline") is False
    assert summary.get("no_hidden_verified_claim") is True

    # Verify not pointwise
    assert summary.get("pointwise_comparison_used") is False

    # Verify overall_status is a recognised value
    from validation.python.integrator_crosscheck import _ALLOWED_OVERALL_STATES
    assert summary.get("overall_status") in _ALLOWED_OVERALL_STATES, (
        f"Unexpected overall_status: {summary.get('overall_status')}"
    )

    # Verify crosscheck_summary.json content matches return value
    with open(case_dir / "crosscheck_summary.json", "r", encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved.get("hiddenness_certified_by_this_pipeline") is False
    assert saved.get("no_hidden_verified_claim") is True
