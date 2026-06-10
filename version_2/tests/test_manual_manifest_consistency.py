# -*- coding: utf-8 -*-
import tomllib
from pathlib import Path
import pytest
from tests.test_manual_manifest import load_manifest

ROOT_DIR = Path(__file__).resolve().parents[1]  # version_2 directory

@pytest.mark.hygiene
def test_manual_manifest_pyproject_consistency():
    """Verify that manual_manifest.yaml public_cli and entry_point match pyproject.toml."""
    pyproject_path = ROOT_DIR / "pyproject.toml"
    assert pyproject_path.exists(), f"pyproject.toml not found at {pyproject_path}"
    
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
        
    scripts = pyproject_data.get("project", {}).get("scripts", {})
    
    # Extract public CLI from manifest
    manifest_data = load_manifest()
    manifest_cli = manifest_data.get("public_cli")
    manifest_entry = manifest_data.get("entry_point")
    
    # 1. Assert that hidden-attractors is in pyproject.toml project.scripts
    assert "hidden-attractors" in scripts, (
        "pyproject.toml project.scripts does not define 'hidden-attractors'"
    )
    
    # 2. Confirm that the only public script installed is hidden-attractors
    allowed_scripts = {"hidden-attractors"}
    defined_scripts = set(scripts.keys())
    assert defined_scripts == allowed_scripts, (
        f"pyproject.toml contains unexpected scripts: {defined_scripts - allowed_scripts}"
    )
    
    # 3. Confirm entry point matches manual_manifest.yaml
    expected_entry = scripts["hidden-attractors"]
    assert manifest_entry == expected_entry, (
        f"Manifest entry_point '{manifest_entry}' does not match pyproject.toml script entry point '{expected_entry}'"
    )
    
    # 4. Fail if the manifest declares any other public CLI command
    assert manifest_cli == "hidden-attractors", (
        f"Manifest declares an invalid public CLI command: '{manifest_cli}'"
    )
