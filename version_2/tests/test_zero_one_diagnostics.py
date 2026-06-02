"""Tests for F5.2 0-1 supporting diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hidden_attractors.analysis.zero_one import (
    ALLOWED_ZERO_ONE_STATES,
    zero_one_multicoordinate,
    zero_one_test,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation" / "chaos_validation" / "dynamics_diagnostics" / "zero_one"


def test_regular_sine_has_low_k() -> None:
    index = np.arange(2500, dtype=float)
    result = zero_one_test(np.sin(2.0 * np.pi * 0.037 * index), n_c=40, max_samples=2000)
    assert result["K"] < 0.2
    assert result["state"] == "zero_one_regular_candidate"


def test_logistic_map_has_high_k() -> None:
    values = np.empty(2500)
    values[0] = 0.123
    for index in range(values.size - 1):
        values[index + 1] = 4.0 * values[index] * (1.0 - values[index])
    result = zero_one_test(values, n_c=40, max_samples=2000)
    assert result["K"] > 0.8
    assert result["state"] == "zero_one_chaotic_candidate"
    assert result["chaos_certified_by_zero_one"] is False


def test_multicoordinate_output_has_k_per_coordinate() -> None:
    times = np.arange(1500, dtype=float) * 0.01
    states = np.column_stack((np.sin(times), np.cos(times), np.sin(2.0 * times)))
    result = zero_one_multicoordinate(times, states, burn_time=1.0, n_c=20, max_samples=1000)
    assert set(result["coordinate_results"]) == {"x", "y", "z"}
    assert all("K" in values for values in result["coordinate_results"].values())
    assert result["chaos_certified_by_zero_one"] is False


def test_zero_one_allowed_states_are_exact() -> None:
    assert ALLOWED_ZERO_ONE_STATES == {
        "zero_one_chaotic_candidate",
        "zero_one_regular_candidate",
        "zero_one_inconclusive",
    }


def test_zero_one_standard_outputs_exist() -> None:
    summary = json.loads((OUTPUT / "zero_one_diagnostics_summary.json").read_text(encoding="utf-8"))
    assert summary["standardized_outputs"] is True
    assert summary["zero_one_proves_chaos"] is False
    assert summary["method_validation"]["regular_sine"] == "passed"
    assert summary["method_validation"]["logistic_map"] == "passed"
    assert summary["method_validation"]["noise_limitation_documented"] is True
