from __future__ import annotations

import sys
import os
import json
import pytest
from pathlib import Path

# Add workspace root and version_2 to sys.path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

from hidden_attractors.workflows.attractor_only import run_attractor_only_workflow


def test_attractor_only_integer(tmp_path):
    config = {
        "system_id": "chua_integer_saturation",
        "q": 1.0,
        "integrator": "heun",
        "h": 0.01,
        "final_simulation": {
            "t_final": 5.0,
            "t_burn": 1.0,
            "initial_condition": [0.1, 0.1, 0.1]
        },
        "output_dir": str(tmp_path),
        "save_timeseries": True,
        "save_attractor": True,
        "diagnostics": True,
        "plot_enabled": False,  # disable plots to speed up tests
    }

    summary = run_attractor_only_workflow(config)

    assert summary["system_id"] == "chua_integer_saturation"
    assert summary["q"] == 1.0
    assert summary["integrator"] == "heun"
    assert len(summary["results"]) == 1
    
    res = summary["results"][0]
    assert res["ic_label"] == "x0"
    assert res["status"] == "ok"
    assert "diagnostics" in res
    assert "classification" in res
    
    # Check that files were written
    assert os.path.exists(os.path.join(tmp_path, "summary.json"))
    assert os.path.exists(os.path.join(tmp_path, "final_timeseries.csv"))
    assert os.path.exists(os.path.join(tmp_path, "final_attractor.csv"))


def test_attractor_only_fractional(tmp_path):
    config = {
        "system_id": "chua_fractional_saturation",
        "q": 0.98,
        "integrator": "efork3",
        "h": 0.01,
        "memory_mode": "window",
        "memory_window_length": 50,
        "final_simulation": {
            "t_final": 2.0,
            "t_burn": 0.5,
            "initial_condition": [0.1, 0.1, 0.1]
        },
        "output_dir": str(tmp_path),
        "save_timeseries": True,
        "save_attractor": True,
        "diagnostics": True,
        "plot_enabled": False,
        "use_c_backend": True,  # test C backend integration
    }

    summary = run_attractor_only_workflow(config)

    assert summary["system_id"] == "chua_fractional_saturation"
    assert summary["q"] == 0.98
    assert summary["integrator"] == "efork3"
    
    res = summary["results"][0]
    assert res["status"] == "ok"
    assert "diagnostics" in res
