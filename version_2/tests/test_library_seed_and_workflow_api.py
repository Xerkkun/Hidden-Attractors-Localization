"""Tests for migrated library APIs that replace env-only legacy calls."""

from __future__ import annotations

import numpy as np

from hidden_attractors.seed_generation import (
    find_harmonic_seed,
    find_omega_gain_candidates,
    reconstruct_biased_lure_seed,
)
from hidden_attractors.solvers import FractionalHistory
from hidden_attractors.workflows.unified_chua import UnifiedChuaConfig


def test_harmonic_seed_generation_is_importable_without_env(monkeypatch) -> None:
    for name in list(__import__("os").environ):
        if name.startswith("HIDDEN_ATTRACTORS_"):
            monkeypatch.delenv(name, raising=False)

    pairs = find_omega_gain_candidates(0.9998, nscan=2000)
    assert pairs

    seed = find_harmonic_seed(q=0.9998, branch_index=0, nscan=2000)
    assert seed.seed.shape == (3,)
    assert np.all(np.isfinite(seed.seed))
    assert seed.omega > 0.0
    assert seed.amplitude > 0.0


def test_biased_seed_reconstruction_returns_finite_state() -> None:
    pairs = find_omega_gain_candidates(0.9998, nscan=2000)
    omega = pairs[0][0]

    seed = reconstruct_biased_lure_seed(q=0.9998, amplitude=5.0, sigma0=0.0, omega=omega)

    assert seed.seed.shape == (3,)
    assert seed.mean_state.shape == (3,)
    assert np.all(np.isfinite(seed.seed))


def test_fractional_history_extracts_efork_window() -> None:
    t = np.linspace(0.0, 1.0, 11)
    traj = np.column_stack([t, t, t + 1.0, t + 2.0])

    history = FractionalHistory.from_trajectory(traj, q=0.9998, h=0.1, memory_length=0.3)

    assert history.memory_points == 4
    assert history.as_efork_history().shape == (4, 4)
    assert history.t_window[-1] == 0.0


def test_unified_chua_config_uses_cli_arguments_not_hidden_env() -> None:
    cfg = UnifiedChuaConfig(
        output_dir="outputs/example_run",
        run_mode="q_sweep",
        q_values=(0.99, 0.9998),
        psd=True,
        basin_planes=False,
    )

    args = cfg.to_argv()
    joined = " ".join(args)

    assert "--run-mode q_sweep" in joined
    assert "--q-values 0.99,0.9998" in joined
    assert "--psd" in args
    assert "--no-basin-planes" in args
    assert "HIDDEN_ATTRACTORS" not in joined
