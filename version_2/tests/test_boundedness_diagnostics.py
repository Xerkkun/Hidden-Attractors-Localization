"""Tests for F5.1 finite-time boundedness diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hidden_attractors.analysis.boundedness import compute_boundedness_metrics


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation" / "chaos_validation" / "dynamics_diagnostics" / "boundedness"


def test_bounded_synthetic_trajectory_is_candidate() -> None:
    times = np.linspace(0.0, 20.0, 1001)
    states = np.column_stack((np.sin(times), np.cos(times), 0.5 * np.sin(2.0 * times)))
    result = compute_boundedness_metrics(times, states, burn_time=5.0, divergence_radius=100.0)
    assert result["boundedness_status"] == "bounded_candidate"
    assert result["R_observed"] < 2.0
    assert result["boundedness_proves_chaos"] is False


def test_divergent_synthetic_trajectory_is_unbounded_candidate() -> None:
    times = np.linspace(0.0, 10.0, 1001)
    values = np.exp(times)
    states = np.column_stack((values, values, values))
    result = compute_boundedness_metrics(times, states, burn_time=0.0, divergence_radius=100.0)
    assert result["boundedness_status"] == "unbounded_candidate"


def test_nonfinite_synthetic_trajectory_is_reported() -> None:
    times = np.linspace(0.0, 10.0, 101)
    states = np.column_stack((np.sin(times), np.cos(times), times))
    states[50, 0] = np.nan
    result = compute_boundedness_metrics(times, states, burn_time=0.0)
    assert result["boundedness_status"] == "nonfinite_trajectory"
    assert result["nonfinite_count"] == 1


def test_boundedness_standard_outputs_exist() -> None:
    summary = json.loads((OUTPUT / "boundedness_diagnostics_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed_structured_outputs"
    assert summary["standardized_outputs"] is True
    assert summary["boundedness_proves_chaos"] is False
    for case in summary["case_summaries"]:
        assert (OUTPUT / case["case_id"] / "boundedness_case_summary.json").is_file()
        assert (OUTPUT / case["case_id"] / "README.md").is_file()
