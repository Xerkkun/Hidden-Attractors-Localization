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
    from hidden_attractors.paths import list_packaged_example_configs, get_example_config_resource
    
    yaml_filenames = list_packaged_example_configs()
    assert len(yaml_filenames) > 0, "No packaged YAML example files found!"

    for filename in yaml_filenames:
        ref = get_example_config_resource(filename)
        
        # Load and parse with config_loader
        with importlib.resources.as_file(ref) as local_p:
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
