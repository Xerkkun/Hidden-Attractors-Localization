from __future__ import annotations

import sys
from pathlib import Path
import pytest

# Add version_2 to path
version_2_dir = Path(__file__).resolve().parents[2]
if str(version_2_dir) not in sys.path:
    sys.path.insert(0, str(version_2_dir))

from hidden_attractors.cli.continuation import run_scalar_continuation

def test_continuation_order_cli_invalid_integrator(tmp_path):
    config_data = """
system:
  system_id: "chua_fractional_saturation"
  q: 1.0
modes:
  transfer_mode: "fractional"
  seed_mode: "fractional"
  continuation_mode: "fractional"
integrator:
  name: "rk4"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_data, encoding="utf-8")
    
    output_dir = tmp_path / "output"
    seed_file = output_dir / "seeds.csv"
    
    # Write a mock seed to the seed file to bypass early exit
    import csv
    import json
    seed_file.parent.mkdir(parents=True, exist_ok=True)
    with open(seed_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "family", "centered_or_biased", "A", "sigma0", "omega", "q", "harmonic_residual", "rho_H", "x0", "reconstruction_metadata", "source_config"])
        w.writerow(["cand1", "lure_classical_centered", "centered", "1.0", "0.0", "1.0", "1.0", "0.0", "0.0", json.dumps([1.0, 2.0, 3.0]), json.dumps({"gain": 1.0}), "none"])
    
    # Pass rk4 with fractional continuation order, which is invalid
    argv = [
        "-c", str(config_path),
        "-s", str(seed_file),
        "-o", str(output_dir),
        "--continuation-order", "fractional",
        "--q-continuation", "0.95",
        "--integrator", "rk4"
    ]
    
    with pytest.raises(ValueError, match="only supports integer-order dynamics"):
        run_scalar_continuation(argv)
