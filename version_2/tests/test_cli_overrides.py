from __future__ import annotations

import sys
from pathlib import Path
import pytest

# Add version_2 to sys.path if not present
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

from hidden_attractors.workflows.config_loader import apply_cli_overrides, load_config
from hidden_attractors.paths import get_packaged_examples_path


def test_cli_override_case_a():
    cfg = {"bifurcation": {"values": {"n": 300}}}
    # Since apply_cli_overrides validates the dict, let's add minimal valid fields to avoid validation errors.
    cfg.update({
        "system_id": "chua_fractional_saturation",
        "integrator": "efork3",
        "h": 0.001,
        "memory_mode": "full",
        "memory_policy": "full_caputo",
    })
    res = apply_cli_overrides(cfg, {"bifurcation.values.n": 3})
    assert res["bifurcation"]["values"]["n"] == 3


def test_cli_override_case_b():
    cfg = {"final_simulation": {"t_final": 500.0}}
    cfg.update({
        "system_id": "chua_fractional_saturation",
        "integrator": "efork3",
        "h": 0.001,
        "memory_mode": "full",
        "memory_policy": "full_caputo",
    })
    res = apply_cli_overrides(cfg, {"final_simulation.t_final": 0.2})
    assert res["final_simulation"]["t_final"] == 0.2


def test_cli_override_case_c():
    cfg = {"basin": {"grid_n": 150}}
    cfg.update({
        "system_id": "chua_fractional_saturation",
        "integrator": "efork3",
        "h": 0.001,
        "memory_mode": "full",
        "memory_policy": "full_caputo",
    })
    res = apply_cli_overrides(cfg, {"basin.grid_n": 3})
    assert res["basin"]["grid_n"] == 3


def test_cli_override_case_d():
    # Start with memory_mode: full, memory_policy: full_caputo
    cfg = {
        "system_id": "chua_fractional_saturation",
        "integrator": "efork3",
        "h": 0.001,
        "memory_mode": "full",
        "memory_policy": "full_caputo",
    }
    res = apply_cli_overrides(cfg, {"memory_policy": "finite_window", "memory_window_steps": 20})
    assert res["memory_mode"] == "window"
    assert res["memory_policy"] == "finite_window"
    assert res["memory_window_steps"] == 20


def test_cli_override_case_e():
    # Start with memory_mode: full, memory_policy: full_caputo
    cfg = {
        "system_id": "chua_fractional_saturation",
        "integrator": "efork3",
        "h": 0.001,
        "memory_mode": "full",
        "memory_policy": "full_caputo",
    }
    res = apply_cli_overrides(cfg, {"memory_mode": "window", "memory_window_steps": 20})
    assert res["memory_mode"] == "window"
    assert res["memory_policy"] == "finite_window"
    assert res["memory_window_steps"] == 20


def test_cli_override_case_f_real_cli(capsys):
    from hidden_attractors.cli.run import main
    main(["inspect-config", "--preset", "chua_bifurcation", "--bifurcation.values.n", "3"])
    captured = capsys.readouterr()
    assert "values" in captured.out
    assert "'n': 3" in captured.out


def test_global_q_and_transfer_overrides_propagate_to_all_fractional_contracts():
    cfg = {
        "system_id": "chua_fractional_arctan",
        "q": 0.95,
        "integrator": "abm",
        "h": 0.01,
        "memory_mode": "full",
        "memory_policy": "full_caputo",
        "transfer_mode": "published_integer_laplace",
        "seed": {
            "df_order": "fractional",
            "transfer_mode": "published_integer_laplace",
            "q_seed": 0.95,
        },
        "continuation": {"continuation_order": "fractional", "q_continuation": 0.95},
        "dynamics": {"dynamics_order": "fractional", "q_dynamics": 0.95},
    }
    result = apply_cli_overrides(
        cfg,
        {"q": 0.9999, "transfer_mode": "fractional_spectral"},
    )
    assert result["q"] == pytest.approx(0.9999)
    assert result["seed"]["q_seed"] == pytest.approx(0.9999)
    assert result["continuation"]["q_continuation"] == pytest.approx(0.9999)
    assert result["dynamics"]["q_dynamics"] == pytest.approx(0.9999)
    assert result["transfer_mode"] == "fractional_spectral"
    assert result["seed"]["transfer_mode"] == "fractional_spectral"
