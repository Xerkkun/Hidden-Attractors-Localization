from __future__ import annotations

import sys
import pytest
from pathlib import Path

# Add workspace root and version_2 to sys.path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

from hidden_attractors.workflows.config_loader import load_config
from hidden_attractors.cli.run import PRESETS


def test_all_example_yamls_load_successfully():
    from hidden_attractors.paths import CONFIGS
    
    examples_dir = CONFIGS / "examples"
    if not examples_dir.exists():
        # fallback for dev workspace layout
        examples_dir = Path(__file__).resolve().parents[1] / "configs" / "examples"
        
    assert examples_dir.exists(), f"Examples directory not found at {examples_dir}"
    
    for preset_name, filename in PRESETS.items():
        config_path = examples_dir / filename
        assert config_path.exists(), f"Config file {filename} for preset {preset_name} not found"
        
        # Load and validate
        cfg = load_config(config_path)
        
        # Simple structural sanity checks
        assert "system_id" in cfg
        assert "integrator" in cfg
        assert "h" in cfg
        assert isinstance(cfg["h"], float)
