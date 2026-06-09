from __future__ import annotations

import sys
import pytest
from pathlib import Path
from hidden_attractors.cli.run import main

# Add version_2 to sys.path if not present
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))


def test_cli_inspect_candidates(capsys):
    # Test hidden-attractors inspect candidates
    main(["inspect", "candidates"])
    captured = capsys.readouterr()
    # If no candidates are found, it might print nothing, but it shouldn't crash.
    # We just assert it executes successfully.
    assert captured.err == ""


def test_cli_inspect_systems(capsys):
    # Test hidden-attractors inspect systems
    main(["inspect", "systems"])
    captured = capsys.readouterr()
    assert "chua-nonsmooth" in captured.out
    assert captured.err == ""


def test_cli_inspect_systems_specific(capsys):
    # Test hidden-attractors inspect systems --system chua-nonsmooth
    main(["inspect", "systems", "--system", "chua-nonsmooth", "--equilibria"])
    captured = capsys.readouterr()
    assert "chua-nonsmooth" in captured.out or "nonsmooth" in captured.out
    assert "equilibrium." in captured.out
    assert captured.err == ""


def test_cli_inspect_workflow_requirements(capsys):
    # Test hidden-attractors inspect workflow-requirements --example-spec
    main(["inspect", "workflow-requirements", "--example-spec"])
    captured = capsys.readouterr()
    assert "basin" in captured.out
    assert captured.err == ""


def test_cli_validate_bibliography(capsys):
    # Test validate bibliography
    # By default, it looks at bibliography.json, let's see if it works or runs without crash
    try:
        main(["validate", "bibliography"])
    except SystemExit as e:
        assert e.code == 0
    captured = capsys.readouterr()
    # It might print validation status
    assert captured.err == ""
