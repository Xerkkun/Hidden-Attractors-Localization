"""Focused tests for biased-Lur'e seed preservation and early screening."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


LEGACY_ROOT = Path(__file__).resolve().parents[1] / "tools" / "legacy"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from early_periodicity_filter import classify_early_periodicity, run_early_periodicity_filter  # noqa: E402
from lure_biased_multiparam_continuation import selected_seed_items  # noqa: E402
from lure_biased_multiparam_search import Quadrature, candidate_filter, evaluate_family_point, evaluate_point  # noqa: E402


def _filter_cfg() -> dict:
    return {
        "search": {
            "residual_keep": 0.05,
            "rhoH_keep": 0.30,
            "rhoH_priority": 0.15,
        },
        "fallback": {
            "enabled": True,
            "keep_best": 2,
            "lambda_rho": 0.05,
            "allow_rhoH_up_to": 1.0,
            "allow_residual_up_to": 0.5,
        },
    }


def _evaluation(candidate_id: str, residual: float, rho: float, amplitude: float = 1.0) -> dict:
    return {
        "candidate_id": candidate_id,
        "df_family": "classical_biased",
        "q": 0.9998,
        "mu": "",
        "A": amplitude,
        "sigma0": 0.0,
        "omega": 1.0,
        "residual_abs": residual,
        "rho_H": rho,
        "notes": "",
    }


def test_fallback_preserves_ranked_seeds_without_hidden_label() -> None:
    result = candidate_filter(
        [_evaluation("row_a", 0.10, 0.40), _evaluation("row_b", 0.08, 0.35, amplitude=1.1)],
        _filter_cfg(),
    )

    assert result["accepted_candidates"] == []
    assert len(result["fallback_candidates"]) == 2
    assert all(row["candidate_status"] == "best_available_seed_not_accepted" for row in result["selected_candidates"])
    assert all(row["hiddenness_status"] == "not_tested" for row in result["selected_candidates"])
    assert "candidate_hidden_like" not in {row["candidate_status"] for row in result["selected_candidates"]}
    assert result["selected_candidates"][0]["score"] <= result["selected_candidates"][1]["score"]


def test_hard_filter_emits_explicit_acceptance_status() -> None:
    result = candidate_filter([_evaluation("accepted", 0.01, 0.10)], _filter_cfg())

    assert len(result["accepted_candidates"]) == 1
    assert result["fallback_candidates"] == []
    assert result["selected_candidates"][0]["candidate_status"] == "hard_candidate_accepted"


def test_hard_filter_limits_candidates_sent_to_early_screen() -> None:
    cfg = _filter_cfg()
    cfg["early_periodicity_filter"] = {"max_candidates_for_screen": 1}
    result = candidate_filter(
        [_evaluation("first", 0.01, 0.10), _evaluation("second", 0.02, 0.10, amplitude=1.1)],
        cfg,
    )

    assert len(result["accepted_candidates"]) == 2
    assert len(result["selected_candidates"]) == 1


def test_machado_fdf_closure_applies_configured_theta(monkeypatch) -> None:
    quad = Quadrature(2, 64)
    precomputed_y = np.array([2.0 + 0.0j, 0.0 + 0.0j])
    precomputed_w = [-0.5 + 0.0j, 0.0 + 0.0j]
    monkeypatch.setattr(
        "lure_biased_multiparam_search.chua.machado_complex_power",
        lambda base_n, _mu, branch=0: complex(base_n),
    )
    params = {"alpha_chua": 1.0}

    zero_phase = evaluate_point(
        candidate_id="theta_zero",
        A=1.0,
        sigma0=0.0,
        omega=1.0,
        q=0.9998,
        p=params,
        quad=quad,
        source_hint="test",
        stage="test",
        df_family="machado_biased",
        mu=1.0,
        theta=0.0,
        precomputed_Y=precomputed_y,
        precomputed_W=precomputed_w,
    )
    opposite_phase = evaluate_point(
        candidate_id="theta_pi",
        A=1.0,
        sigma0=0.0,
        omega=1.0,
        q=0.9998,
        p=params,
        quad=quad,
        source_hint="test",
        stage="test",
        df_family="machado_biased",
        mu=1.0,
        theta=np.pi,
        precomputed_Y=precomputed_y,
        precomputed_W=precomputed_w,
    )

    assert zero_phase["residual_abs"] == pytest.approx(0.0, abs=1.0e-12)
    assert opposite_phase["residual_abs"] == pytest.approx(2.0, abs=1.0e-12)


def test_machado_fdf_selects_best_theta_from_configured_grid(monkeypatch) -> None:
    quad = Quadrature(2, 64)
    monkeypatch.setattr(
        "lure_biased_multiparam_search.chua.machado_complex_power",
        lambda base_n, _mu, branch=0: complex(base_n),
    )
    best = evaluate_family_point(
        candidate_id="theta_grid",
        A=1.0,
        sigma0=0.0,
        omega=1.0,
        q=0.9998,
        p={"alpha_chua": 1.0},
        quad=quad,
        source_hint="test",
        stage="test",
        df_family="machado_biased",
        mu=1.0,
        cfg={"machado": {"theta_values": [np.pi, 0.0]}},
        precomputed_Y=np.array([2.0 + 0.0j, 0.0 + 0.0j]),
        precomputed_W=[-0.5 + 0.0j, 0.0 + 0.0j],
    )

    assert best["theta"] == pytest.approx(0.0)
    assert best["residual_abs"] == pytest.approx(0.0, abs=1.0e-12)


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
    assert result["rejected_seeds"][0]["periodicity_case_id"] == "efork_full_history"

    resumed = run_early_periodicity_filter([seed], cfg, {}, checkpoint_path=checkpoint, resume=True)
    assert len(resumed["diagnostics"]) == 4
    assert resumed["summary"]["n_rejected_by_primary_post_transient_gate"] == 1


def test_continuation_accepts_only_nonperiodic_unrejected_seeds() -> None:
    candidates = [
        {"candidate_id": "keep", "candidate_status": "hard_candidate_accepted"},
        {"candidate_id": "periodic", "candidate_status": "best_available_seed_not_accepted"},
    ]
    seeds = [
        {
            "candidate_id": "keep",
            "seed_id": "s_keep",
            "candidate_status": "hard_candidate_accepted",
            "early_periodicity_status": "nonperiodic_post_transient",
            "valid_seed": True,
            "x0": 0.0,
            "y0": 0.0,
            "z0": 0.0,
        },
        {
            "candidate_id": "periodic",
            "seed_id": "s_periodic",
            "candidate_status": "best_available_seed_not_accepted",
            "early_periodicity_status": "periodic_post_transient",
            "valid_seed": True,
            "x0": 1.0,
            "y0": 1.0,
            "z0": 1.0,
        },
    ]

    selected = selected_seed_items(
        {"continuation": {"max_candidates": 6, "max_seeds_per_candidate": 1}},
        candidates,
        seeds,
        {"s_periodic"},
    )

    assert [row["seed_id"] for row in selected] == ["s_keep"]


def test_continuation_accepts_periodic_diagnostic_seed_when_gate_is_disabled() -> None:
    candidates = [{"candidate_id": "periodic", "candidate_status": "hard_candidate_accepted"}]
    seeds = [{
        "candidate_id": "periodic",
        "seed_id": "s_periodic",
        "candidate_status": "hard_candidate_accepted",
        "early_periodicity_status": "pre_continuation_periodic",
        "valid_seed": True,
        "x0": 1.0,
        "y0": 1.0,
        "z0": 1.0,
    }]

    selected = selected_seed_items(
        {
            "early_periodicity_filter": {"gate_before_continuation": False},
            "continuation": {"max_candidates": 6, "max_seeds_per_candidate": 1},
        },
        candidates,
        seeds,
        {"s_periodic"},
    )

    assert [row["seed_id"] for row in selected] == ["s_periodic"]
