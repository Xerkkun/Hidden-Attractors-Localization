from __future__ import annotations

import sys
import pytest
import warnings
from pathlib import Path

# Add workspace root and version_2 to sys.path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

from hidden_attractors.workflows.config_loader import load_config, apply_cli_overrides


def test_load_hierarchical_yaml(tmp_path):
    yaml_content = """
experiment:
  name: "Test run"
  output_dir: "outputs/test"
system:
  system_id: "chua_fractional_saturation"
  q: 0.99
  parameters:
    alpha: 9.0
    beta: 15.0
modes:
  transfer_mode: "fractional"
integrator:
  name: "efork3"
  h: 0.005
stages:
  attractor_only: true
"""
    yaml_file = tmp_path / "test_hier.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    cfg = load_config(yaml_file)

    assert cfg["system_id"] == "chua_fractional_saturation"
    assert cfg["q"] == 0.99
    assert cfg["alpha"] == 9.0
    assert cfg["beta"] == 15.0
    assert cfg["transfer_mode"] == "fractional"
    assert cfg["integrator"] == "efork3"
    assert cfg["h"] == 0.005
    assert cfg["run_attractor_only"] is True


def test_load_legacy_flat_yaml_warning(tmp_path):
    yaml_content = """
system_id: chua_integer_saturation
q: 1.0
integrator: heun
h: 0.002
t_final: 100.0
"""
    yaml_file = tmp_path / "test_legacy.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="Detected legacy flat YAML keys"):
        cfg = load_config(yaml_file)

    assert cfg["system_id"] == "chua_integer_saturation"
    assert cfg["q"] == 1.0
    assert cfg["integrator"] == "heun"
    assert cfg["h"] == 0.002
    assert cfg["final_simulation"]["t_final"] == 100.0


def test_apply_cli_overrides():
    base_cfg = {
        "system_id": "chua_integer_saturation",
        "q": 1.0,
        "integrator": "heun",
        "h": 0.002,
        "final_simulation": {
            "t_final": 100.0,
            "t_burn": 20.0,
        }
    }

    overrides = {
        "q": 0.95,
        "integrator": "efork3",
        "final_simulation.t_final": 250.0,
    }

    updated = apply_cli_overrides(base_cfg, overrides)

    assert updated["q"] == 0.95
    assert updated["integrator"] == "efork3"
    assert updated["final_simulation"]["t_final"] == 250.0


def test_invalid_config_validation(tmp_path):
    # Invalid integrator/q combination (RK4 with q < 1)
    yaml_content = """
system:
  system_id: "chua_fractional_saturation"
  q: 0.95
integrator:
  name: "rk4"
"""
    yaml_file = tmp_path / "test_invalid.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="only supports integer-order systems"):
        load_config(yaml_file)
