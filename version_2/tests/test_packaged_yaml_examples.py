from __future__ import annotations

import sys
import importlib.resources
import pytest
from pathlib import Path

# Add version_2 to sys.path if not present
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

from hidden_attractors.workflows.config_loader import load_config
from hidden_attractors.paths import get_packaged_examples_path


def test_packaged_yamls():
    """List, load and validate all packaged YAML configurations."""
    ref_dir = importlib.resources.files("hidden_attractors").joinpath("configs", "examples")
    
    yaml_files = []
    try:
        yaml_files = [f for f in ref_dir.iterdir() if f.is_file() and f.name.endswith(".yaml")]
    except Exception:
        pass
        
    if not yaml_files:
        # Fallback to local files
        local_src = get_packaged_examples_path()
        yaml_files = list(local_src.glob("*.yaml"))

    assert len(yaml_files) > 0, "No packaged YAML example files found!"

    for f_path in yaml_files:
        filename = f_path.name
        
        # Load and parse with config_loader
        with importlib.resources.as_file(f_path) as local_p:
            # Check raw file contents for forbidden legacy keys 'm' or 'n' in the parameters section
            with open(local_p, "r", encoding="utf-8") as fh:
                raw_text = fh.read()
                # Ensure they are not defined as parameter mappings (like "m: " or "n: ")
                assert "\n  m:" not in raw_text, f"{filename} contains legacy parameter 'm'"
                assert "\n  n:" not in raw_text, f"{filename} contains legacy parameter 'n'"

            cfg = load_config(local_p)
            
            # Validate required normalized keys
            assert "system_id" in cfg, f"{filename} is missing system_id"
            assert "integrator" in cfg, f"{filename} is missing integrator"
            assert "output_dir" in cfg, f"{filename} is missing output_dir"
            assert cfg["system_id"] is not None
            assert cfg["integrator"] is not None
            assert cfg["output_dir"] is not None
