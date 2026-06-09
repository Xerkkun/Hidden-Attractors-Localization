from __future__ import annotations

import sys
import pytest
import json
from pathlib import Path

# Add version_2 to sys.path if not present
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

from hidden_attractors.cli.run import main

# ---------------------------------------------------------------------------
# Grouped CLI help tests
# ---------------------------------------------------------------------------
@pytest.mark.cli
@pytest.mark.parametrize("args", [
    [],
    ["run"],
    ["init"],
    ["inspect-config"],
    ["validate"],
    ["protocol"],
    ["hiddenness"],
    ["basin"],
    ["bifurcation"],
    ["lyapunov"],
    ["chaos-test"],
])
def test_grouped_cli_help(args, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(args + ["--help"])
    assert excinfo.value.code == 0

# ---------------------------------------------------------------------------
# Deprecated aliases
# ---------------------------------------------------------------------------
@pytest.mark.deprecated_alias
@pytest.mark.parametrize("entrypoint_mod,entrypoint_func", [
    ("hidden_attractors.protocol_cli", "main"),
    ("hidden_attractors.workflows.sphere_controls", "main"),
    ("hidden_attractors.workflows.refined_basin", "main"),
    ("hidden_attractors.workflows.fractional_report_run", "main"),
])
def test_deprecated_aliases_do_not_crash(entrypoint_mod, entrypoint_func, capsys):
    """Temporary compatibility checks for deprecated CLI entry points.
    Scheduled for removal in v1.0.0.
    """
    import importlib
    mod = importlib.import_module(entrypoint_mod)
    func = getattr(mod, entrypoint_func)
    with pytest.raises(SystemExit) as excinfo:
        func(["--help"])
    assert excinfo.value.code == 0

# ---------------------------------------------------------------------------
# Subcommand smoke tests
# ---------------------------------------------------------------------------
@pytest.mark.cli
def test_cli_inspect_config_chua_fractional(capsys):
    # Test inspecting a built-in preset with --preset
    main(["inspect-config", "--preset", "chua_fractional"])
    captured = capsys.readouterr()
    assert "EFFECTIVE CONFIGURATION" in captured.out
    assert "chua_fractional_saturation" in captured.out

@pytest.mark.cli
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

@pytest.mark.cli
def test_cli_init_single(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["init", "-e", "chua_integer"])
    assert (tmp_path / "chua_integer_centered_lure_df.yaml").exists()

@pytest.mark.cli
def test_cli_init_all(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["init"])
    examples_dir = tmp_path / "configs" / "examples"
    assert examples_dir.exists()
    assert (examples_dir / "chua_integer_centered_lure_df.yaml").exists()

@pytest.mark.cli
def test_cli_inspect_candidates(capsys):
    # Test hidden-attractors inspect candidates
    main(["inspect", "candidates"])
    captured = capsys.readouterr()
    assert captured.err == ""

@pytest.mark.cli
def test_cli_inspect_systems(capsys):
    # Test hidden-attractors inspect systems
    main(["inspect", "systems"])
    captured = capsys.readouterr()
    assert "chua-nonsmooth" in captured.out
    assert captured.err == ""

@pytest.mark.cli
def test_cli_inspect_systems_specific(capsys):
    # Test hidden-attractors inspect systems --system chua-nonsmooth
    main(["inspect", "systems", "--system", "chua-nonsmooth", "--equilibria"])
    captured = capsys.readouterr()
    assert "chua-nonsmooth" in captured.out or "nonsmooth" in captured.out
    assert "equilibrium." in captured.out
    assert captured.err == ""

@pytest.mark.cli
def test_cli_inspect_workflow_requirements(capsys):
    # Test hidden-attractors inspect workflow-requirements --example-spec
    main(["inspect", "workflow-requirements", "--example-spec"])
    captured = capsys.readouterr()
    assert "basin" in captured.out
    assert captured.err == ""

@pytest.mark.cli
def test_cli_validate_bibliography(capsys):
    # Test validate bibliography
    try:
        main(["validate", "bibliography"])
    except SystemExit as e:
        assert e.code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
