from __future__ import annotations

import sys
import tomllib
from pathlib import Path

def test_public_entry_points_are_unified():
    # Find pyproject.toml in version_2 root
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert pyproject_path.exists(), f"Could not find pyproject.toml at {pyproject_path}"

    with open(pyproject_path, "rb") as f:
        project_data = tomllib.load(f)

    scripts = project_data.get("project", {}).get("scripts", {})
    
    # Assert that only 'hidden-attractors' is exposed
    assert set(scripts.keys()) == {"hidden-attractors"}, f"Expected only 'hidden-attractors' script, found: {list(scripts.keys())}"
    
    # Assert that 'hidden-attractors' points to the correct dispatcher
    assert scripts["hidden-attractors"] == "hidden_attractors.cli.main:main", f"Expected entry point to be hidden_attractors.cli.main:main, found: {scripts['hidden-attractors']}"
