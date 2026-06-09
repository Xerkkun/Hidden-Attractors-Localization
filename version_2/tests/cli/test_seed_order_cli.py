from __future__ import annotations

import sys
from pathlib import Path
import pytest

# Add version_2 to path
version_2_dir = Path(__file__).resolve().parents[2]
if str(version_2_dir) not in sys.path:
    sys.path.insert(0, str(version_2_dir))

from hidden_attractors.cli.seed import lure_centered

def test_seed_order_cli_centered(tmp_path):
    config_path = version_2_dir / "configs/examples/chua_integer_centered_lure_df.yaml"
    output_dir = tmp_path / "output"
    
    # Run lure_centered with CLI overrides
    argv = [
        "-c", str(config_path),
        "-o", str(output_dir),
        "--df-order", "integer",
        "--q-seed", "1.0",
        "--integrator", "heun",
        "--h", "0.01"
    ]
    
    lure_centered(argv)
    
    # Check that outputs are created
    assert (output_dir / "seeds.csv").exists()
    assert (output_dir / "run_metadata.json").exists()
    assert (output_dir / "effective_config.json").exists()
