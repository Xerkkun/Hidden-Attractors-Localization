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
])
def test_deprecated_aliases_do_not_crash(entrypoint_mod, entrypoint_func, capsys):
    """Compatibility checks for retained deprecated CLI entry points."""
    import importlib
    mod = importlib.import_module(entrypoint_mod)
    func = getattr(mod, entrypoint_func)
    with pytest.raises(SystemExit) as excinfo:
        func(["--help"])
    assert excinfo.value.code == 0


@pytest.mark.cli
def test_case_specific_workflow_modules_and_routes_are_not_public():
    import importlib.util
    from hidden_attractors.cli.main import GROUPS

    assert importlib.util.find_spec("hidden_attractors.workflows.sphere_controls") is None
    assert importlib.util.find_spec("hidden_attractors.workflows.robustness_overlay") is None
    assert "hiddenness" not in GROUPS
    assert "robustness" not in GROUPS

# ---------------------------------------------------------------------------
# Subcommand smoke tests
# ---------------------------------------------------------------------------
@pytest.mark.cli
def test_cli_inspect_config_explicit_fixture(capsys):
    config_path = Path(__file__).parent / "fixtures" / "software_validation_fractional.yaml"
    main(["inspect-config", "--config", str(config_path)])
    captured = capsys.readouterr()
    assert "EFFECTIVE CONFIGURATION" in captured.out
    assert "chua-nonsmooth" in captured.out

@pytest.mark.cli
def test_cli_has_no_runnable_presets():
    from hidden_attractors.cli.run import PRESETS

    assert PRESETS == {}

@pytest.mark.cli
def test_cli_init_single(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["init", "-e", "workflow_contract"])
    assert (tmp_path / "workflow_contract.yaml").exists()

@pytest.mark.cli
def test_cli_init_all(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["init"])
    examples_dir = tmp_path / "configs" / "examples"
    assert examples_dir.exists()
    assert {path.name for path in examples_dir.glob("*.yaml")} == {"workflow_contract.yaml"}

@pytest.mark.cli
def test_cli_inspect_candidates(tmp_path, capsys):
    source = tmp_path / "selected_candidates.json"
    source.write_text(
        '{"candidates":[{"candidate_id":"portable","q":1.0,'
        '"seed":[0,0],"robust_start":[0,0]}]}',
        encoding="utf-8",
    )
    main(["inspect", "candidates", "--source", str(source)])
    captured = capsys.readouterr()
    assert "portable" in captured.out
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
    payload = json.loads(captured.out)
    assert payload["system_name"] == "example-system"
    assert payload["target_reference"] is None
    assert payload["sphere_controls"] is None
    assert payload["parameter_sweep"] is None
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
