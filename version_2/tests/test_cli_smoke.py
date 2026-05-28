from __future__ import annotations

import sys
import os
import pytest
from pathlib import Path

# Add workspace root and version_2 to sys.path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

from hidden_attractors.cli.run import main


def test_cli_inspect_config(capsys):
    # Test inspecting a built-in preset
    main(["inspect-config", "-p", "chua_integer"])
    captured = capsys.readouterr()
    assert "EFFECTIVE CONFIGURATION" in captured.out
    assert "chua_integer_saturation" in captured.out


def test_cli_init_single(tmp_path, monkeypatch):
    # Change working directory to tmp_path
    monkeypatch.chdir(tmp_path)
    
    # Init a single example preset
    main(["init", "-e", "chua_integer"])
    
    assert (tmp_path / "chua_integer_centered_lure_df.yaml").exists()


def test_cli_init_all(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    
    # Init all example presets
    main(["init"])
    
    examples_dir = tmp_path / "configs" / "examples"
    assert examples_dir.exists()
    assert (examples_dir / "chua_integer_centered_lure_df.yaml").exists()
    assert (examples_dir / "chua_fractional_centered_lure_df.yaml").exists()
    assert (examples_dir / "chua_full_protocol.yaml").exists()


def test_cli_run_preset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    
    # Run a short run of chua_integer with fast overrides
    main([
        "run", "-p", "chua_integer",
        "--final_simulation.t_final", "2.0",
        "--final_simulation.t_burn", "0.5",
        "--output_dir", str(tmp_path / "out"),
        "--plot_enabled", "False",
    ])
    
    summary_path = tmp_path / "out" / "summary.json"
    assert summary_path.exists()
