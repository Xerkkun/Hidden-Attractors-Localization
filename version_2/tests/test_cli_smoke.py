from __future__ import annotations

import sys
import pytest
from pathlib import Path
from hidden_attractors.cli.run import main

# Add version_2 to sys.path if not present
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))


def test_cli_inspect_config_chua_fractional(capsys):
    # Test inspecting a built-in preset with --preset
    main(["inspect-config", "--preset", "chua_fractional"])
    captured = capsys.readouterr()
    assert "EFFECTIVE CONFIGURATION" in captured.out
    assert "chua_fractional_saturation" in captured.out


def test_cli_run_chua_arctan_only_integer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    
    # Run the exact CLI command from command list
    main([
        "run", "--preset", "chua_arctan_only_integer",
        "--final_simulation.t_final", "0.2",
        "--final_simulation.t_burn", "0.05",
        "--h", "0.01",
        "--plot_enabled", "false",
        "--output_dir", str(tmp_path / "out_arctan_int"),
    ])
    
    summary_path = tmp_path / "out_arctan_int" / "summary.json"
    assert summary_path.exists()


def test_cli_init_single(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["init", "-e", "chua_integer"])
    assert (tmp_path / "chua_integer_centered_lure_df.yaml").exists()


def test_cli_init_all(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["init"])
    examples_dir = tmp_path / "configs" / "examples"
    assert examples_dir.exists()
    assert (examples_dir / "chua_integer_centered_lure_df.yaml").exists()
