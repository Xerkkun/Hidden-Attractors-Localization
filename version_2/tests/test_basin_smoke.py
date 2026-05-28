from __future__ import annotations

import sys
import json
import pytest
from pathlib import Path

# Add version_2 to sys.path if not present
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

from hidden_attractors.workflows.config_loader import load_config
from hidden_attractors.workflows.basin_runner import run_basin_workflow
from hidden_attractors.paths import get_packaged_examples_path


def test_basin_smoke_all(tmp_path):
    config_path = get_packaged_examples_path() / "chua_fractional_basin.yaml"
    cfg = load_config(config_path)
    
    cfg["output_dir"] = str(tmp_path)
    cfg["plot_enabled"] = False
    cfg["use_c_backend"] = False
    
    # Configure fast smoke run
    cfg["basin"]["grid_n"] = 3
    cfg["basin"]["t_final"] = 0.1
    cfg["basin"]["t_burn"] = 0.02
    cfg["basin"]["h"] = 0.01
    cfg["basin"]["equilibrium_selection"] = "all"
    cfg["memory_mode"] = "window"
    cfg["memory_window_steps"] = 10
    cfg["final_simulation"] = {
        "t_final": 0.1,
        "t_burn": 0.02,
        "divergence_norm": 120.0
    }
    
    summary = run_basin_workflow(cfg)
    
    assert summary["workflow_mode"] == "basin"
    assert summary["grid_n"] == 3
    assert len(summary["targets_used"]) > 1  # E0, E+, E-
    assert summary["memory_mode"] == "window"
    assert summary["memory_window_length"] == 10
    
    # Verify CSV and NPY exist
    csv_path = tmp_path / "basin_data.csv"
    assert csv_path.exists()
    
    # Check that NPY grids are saved for targets
    for target in summary["targets_used"]:
        for plane in summary["planes"]:
            npy_path = tmp_path / f"basin_grid_{target}_{plane}.npy"
            assert npy_path.exists()
            
            # Verify no figures exist
            png_path = tmp_path / f"basin_slice_{target}_{plane}.png"
            assert not png_path.exists()


def test_basin_smoke_single_eq(tmp_path):
    config_path = get_packaged_examples_path() / "chua_fractional_basin.yaml"
    cfg = load_config(config_path)
    
    cfg["output_dir"] = str(tmp_path)
    cfg["plot_enabled"] = False
    cfg["use_c_backend"] = False
    
    cfg["basin"]["grid_n"] = 3
    cfg["basin"]["t_final"] = 0.1
    cfg["basin"]["t_burn"] = 0.02
    cfg["basin"]["h"] = 0.01
    cfg["basin"]["equilibrium_selection"] = "E+"
    cfg["final_simulation"] = {
        "t_final": 0.1,
        "t_burn": 0.02,
        "divergence_norm": 120.0
    }
    
    summary = run_basin_workflow(cfg)
    
    assert summary["grid_n"] == 3
    assert summary["targets_used"] == ["E+"]
