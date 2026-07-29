# -*- coding: utf-8 -*-
import pytest
import tomllib
from pathlib import Path
from tests.helpers.test_documentation_text import read, normalize, active_doc_paths, ROOT
from tests.test_manual_manifest import load_manifest

@pytest.mark.hygiene
def test_manual_cli_consistency_and_metadata():
    # Load manifest and pyproject.toml
    manifest_data = load_manifest()
    
    pyproject_path = ROOT / "pyproject.toml"
    assert pyproject_path.exists(), f"pyproject.toml not found at {pyproject_path}"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
        
    scripts = pyproject_data.get("project", {}).get("scripts", {})
    
    # 5, 6, 7. Check manual_manifest.yaml and pyproject.toml CLI matching
    manifest_cli = manifest_data.get("public_cli")
    manifest_entry = manifest_data.get("entry_point")
    
    assert manifest_cli == "hidden-attractors", "manual_manifest.yaml public_cli is not 'hidden-attractors'"
    assert manifest_entry == "hidden_attractors.cli.main:main", "manual_manifest.yaml entry_point is incorrect"
    
    assert "hidden-attractors" in scripts, "pyproject.toml [project.scripts] lacks 'hidden-attractors'"
    assert scripts["hidden-attractors"] == "hidden_attractors.cli.main:main", "pyproject.toml entry point mismatch"
    
    # Check active documentation content for required commands
    docs = active_doc_paths()
    combined_content = ""
    for p in docs:
        combined_content += "\n" + read(p)
        
    combined_content_normalized = normalize(combined_content)
    
    required_commands = [
        "hidden-attractors validate contract",
        "hidden-attractors inspect systems",
        "hidden-attractors inspect candidates",
        "hidden-attractors protocol",
    ]
    for cmd in required_commands:
        assert cmd in combined_content_normalized, f"Unified CLI command '{cmd}' is missing in active documentation"
