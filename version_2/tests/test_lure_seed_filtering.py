"""Focused tests for biased-Lur'e seed preservation and early screening."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# Add active legacy directory to sys.path
ACTIVE_LEGACY = Path(__file__).resolve().parents[1] / "tools" / "legacy"
if str(ACTIVE_LEGACY) not in sys.path:
    sys.path.insert(0, str(ACTIVE_LEGACY))

from early_periodicity_filter import classify_early_periodicity, run_early_periodicity_filter  # noqa: E402


def test_periodic_post_transient_classification_uses_multiple_components() -> None:
    h = 0.01
    t = np.arange(0.0, 30.0, h)
    traj = np.column_stack([t, np.sin(2.0 * np.pi * t), np.cos(2.0 * np.pi * t), 0.5 * np.sin(2.0 * np.pi * t)])
    result = classify_early_periodicity(
        traj,
        h,
        {
            "entropy_min": 0.25,
            "dominant_ratio_max": 0.55,
            "relaxed_dominant_ratio": 0.40,
            "freq_drift_max": 0.05,
            "n_windows": 3,
            "min_range": 0.01,
            "components": ["x", "y", "z"],
            "require_two_components": True,
        },
    )

    assert result["periodic_early"] is True
    assert result["early_periodicity_status"] == "periodic_post_transient"
    assert result["n_periodic_components"] >= 2


def test_post_transient_filter_ignores_periodicity_only_before_burn_in() -> None:
    h = 0.01
    t = np.arange(0.0, 30.0, h)
    rng = np.random.default_rng(1234)
    tail = rng.normal(size=(t.size, 3))
    states = tail.copy()
    transient = t < 10.0
    states[transient, 0] = np.sin(2.0 * np.pi * t[transient])
    states[transient, 1] = np.cos(2.0 * np.pi * t[transient])
    states[transient, 2] = np.sin(2.0 * np.pi * t[transient])
    result = classify_early_periodicity(
        np.column_stack([t, states]),
        h,
        {
            "t_transient": 10.0,
            "entropy_min": 0.25,
            "dominant_ratio_max": 0.55,
            "relaxed_dominant_ratio": 0.40,
            "freq_drift_max": 0.05,
            "n_windows": 3,
            "min_range": 0.01,
            "components": ["x", "y", "z"],
            "require_two_components": True,
        },
    )

    assert result["early_periodicity_status"] == "nonperiodic_post_transient"
    assert result["t_transient"] == 10.0


def test_post_transient_filter_rejects_periodic_primary_case(monkeypatch) -> None:
    h = 0.01
    t = np.arange(0.0, 30.0, h)
    traj = np.column_stack([t, np.sin(2.0 * np.pi * t), np.cos(2.0 * np.pi * t), np.sin(2.0 * np.pi * t)])
    monkeypatch.setattr("early_periodicity_filter.chua.efork3_integrate", lambda *_args, **_kwargs: traj)
    cfg = {
        "q": 0.9998,
        "early_periodicity_filter": {
            "enabled": True,
            "gate_before_continuation": True,
            "historical_reproduction_mode": True,
            "backend": "python_legacy",
            "h": h,
            "memory_length": 8.0,
            "t_transient": 5.0,
            "observation_time": 25.0,
            "discard_if_periodic": True,
            "entropy_min": 0.25,
            "dominant_ratio_max": 0.55,
            "relaxed_dominant_ratio": 0.40,
            "freq_drift_max": 0.05,
            "n_windows": 3,
            "components": ["x", "y", "z"],
            "require_two_components": True,
        },
    }
    seed = {"seed_id": "s1", "candidate_id": "c1", "valid_seed": True, "x0": 0.1, "y0": 0.2, "z0": 0.3}

    result = run_early_periodicity_filter([seed], cfg, {})

    assert result["kept_seeds"] == []
    assert result["rejected_seeds"][0]["candidate_status"] == "rejected_periodic_post_transient"
    assert result["summary"]["n_nonperiodic_seeds_for_continuation"] == 0


def test_periodic_direct_seed_is_diagnostic_without_precontinuation_gate(monkeypatch) -> None:
    h = 0.01
    t = np.arange(0.0, 30.0, h)
    traj = np.column_stack([t, np.sin(2.0 * np.pi * t), np.cos(2.0 * np.pi * t), np.sin(2.0 * np.pi * t)])
    monkeypatch.setattr("early_periodicity_filter.chua.efork3_integrate", lambda *_args, **_kwargs: traj)
    cfg = {
        "q": 0.9998,
        "early_periodicity_filter": {
            "enabled": True,
            "gate_before_continuation": False,
            "backend": "python_legacy",
            "h": h,
            "memory_length": 8.0,
            "t_transient": 5.0,
            "observation_time": 25.0,
            "discard_if_periodic": True,
            "entropy_min": 0.25,
            "dominant_ratio_max": 0.55,
            "relaxed_dominant_ratio": 0.40,
            "freq_drift_max": 0.05,
            "n_windows": 3,
            "components": ["x", "y", "z"],
            "require_two_components": True,
        },
    }
    seed = {
        "seed_id": "s1",
        "candidate_id": "c1",
        "candidate_status": "hard_candidate_accepted",
        "valid_seed": True,
        "x0": 0.1,
        "y0": 0.2,
        "z0": 0.3,
    }

    result = run_early_periodicity_filter([seed], cfg, {})

    assert result["rejected_seeds"] == []
    assert result["kept_seeds"][0]["early_periodicity_status"] == "pre_continuation_periodic"
    assert result["summary"]["n_seeds_released_to_continuation"] == 1


def test_periodicity_matrix_evaluates_four_solver_memory_cells(monkeypatch, tmp_path: Path) -> None:
    h = 0.01
    t = np.arange(0.0, 20.0 + h, h)
    traj = np.column_stack([t, np.sin(2.0 * np.pi * t), np.cos(2.0 * np.pi * t), np.sin(2.0 * np.pi * t)])

    class FakeEfork:
        @classmethod
        def build(cls, **_kwargs):
            return cls()

        def set_nonsmooth_params(self, _params) -> None:
            return None

        def integrate_efork3(self, *_args, **_kwargs):
            return traj

    class FakeAbm:
        @classmethod
        def build(cls, **_kwargs):
            return cls()

        def set_nonsmooth_params(self, _params) -> None:
            return None

        def integrate(self, *_args, **_kwargs):
            return traj

        def integrate_truncated(self, *_args, **_kwargs):
            return traj

    monkeypatch.setattr("early_periodicity_filter.FractionalChuaBackend", FakeEfork)
    monkeypatch.setattr("early_periodicity_filter.FullHistoryABMBackend", FakeAbm)
    cfg = {
        "q": 0.9998,
        "params": {},
        "early_periodicity_filter": {
            "enabled": True,
            "h": h,
            "t_transient": 5.0,
            "observation_time": 15.0,
            "entropy_min": 0.25,
            "dominant_ratio_max": 0.55,
            "relaxed_dominant_ratio": 0.40,
            "freq_drift_max": 0.05,
            "n_windows": 3,
            "components": ["x", "y", "z"],
            "require_two_components": True,
            "matrix": {
                "enabled": True,
                "primary_case_id": "efork_full_history",
                "cases": [
                    {"case_id": "efork_full_history", "solver": "efork3", "memory_policy": "full_history"},
                    {"case_id": "efork_truncated", "solver": "efork3", "memory_policy": "truncated", "memory_length": 8.0},
                    {"case_id": "abm_full_history", "solver": "abm", "memory_policy": "full_history"},
                    {"case_id": "abm_truncated", "solver": "abm", "memory_policy": "truncated", "memory_length": 8.0},
                ],
            },
        },
    }
    seed = {"seed_id": "s1", "candidate_id": "c1", "valid_seed": True, "x0": 0.1, "y0": 0.2, "z0": 0.3}

    checkpoint = tmp_path / "periodicity_checkpoint.json"
    result = run_early_periodicity_filter([seed], cfg, {}, checkpoint_path=checkpoint)

    assert len(result["diagnostics"]) == 4
    assert checkpoint.exists()
    assert result["summary"]["primary_case_id"] == "efork_full_history"
    assert set(result["summary"]["matrix_case_counts"]) == {
        "efork_full_history",
        "efork_truncated",
        "abm_full_history",
        "abm_truncated",
    }
    assert len(result["kept_seeds"]) == 1
    assert result["kept_seeds"][0]["early_periodicity_status"] == "pre_continuation_periodic"
    assert result["kept_seeds"][0]["periodicity_case_id"] == "efork_full_history"

    resumed = run_early_periodicity_filter([seed], cfg, {}, checkpoint_path=checkpoint, resume=True)
    assert len(resumed["diagnostics"]) == 4
    assert resumed["summary"]["n_rejected_by_primary_post_transient_gate"] == 0
    assert resumed["summary"]["n_seeds_released_to_continuation"] == 1

