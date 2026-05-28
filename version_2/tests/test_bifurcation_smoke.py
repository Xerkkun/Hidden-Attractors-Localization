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
from hidden_attractors.workflows.bifurcation import run_bifurcation_workflow
from hidden_attractors.paths import get_packaged_examples_path


def test_bifurcation_smoke(tmp_path):
    config_path = get_packaged_examples_path() / "chua_fractional_bifurcation.yaml"
    cfg = load_config(config_path)
    
    # Configure fast smoke run
    cfg["output_dir"] = str(tmp_path)
    cfg["plot_enabled"] = False
    cfg["bifurcation"]["values"]["n"] = 3
    cfg["bifurcation"]["discard_time"] = 0.1
    cfg["bifurcation"]["sample_time"] = 0.1
    cfg["bifurcation"]["h"] = 0.01
    cfg["bifurcation"]["save_csv"] = True
    cfg["bifurcation"]["save_plot"] = False
    cfg["use_c_backend"] = False  # Python fallback for test simplicity
    
    summary = run_bifurcation_workflow(cfg)
    
    assert summary["workflow_mode"] == "bifurcation"
    assert summary["n_swept_points"] == 3
    assert "n_success" in summary
    assert "n_failed" in summary
    assert "failed_parameter_values" in summary
    assert "integrator" in summary
    assert "memory_mode" in summary
    assert "memory_window_length" in summary
    assert "q_base" in summary
    assert "parameter_swept" in summary
    
    # Verify CSV file exists
    csv_path = tmp_path / "bifurcation_data.csv"
    assert csv_path.exists()
    
    # Verify no plot is generated
    plot_path = tmp_path / "bifurcation_plot.png"
    assert not plot_path.exists()


def test_bifurcation_no_csv(tmp_path):
    config_path = get_packaged_examples_path() / "chua_fractional_bifurcation.yaml"
    cfg = load_config(config_path)
    
    cfg["output_dir"] = str(tmp_path)
    cfg["plot_enabled"] = False
    cfg["bifurcation"]["values"]["n"] = 2
    cfg["bifurcation"]["discard_time"] = 0.1
    cfg["bifurcation"]["sample_time"] = 0.1
    cfg["bifurcation"]["h"] = 0.01
    cfg["bifurcation"]["save_csv"] = False
    cfg["bifurcation"]["save_plot"] = False
    cfg["use_c_backend"] = False
    
    summary = run_bifurcation_workflow(cfg)
    
    csv_path = tmp_path / "bifurcation_data.csv"
    assert not csv_path.exists()
