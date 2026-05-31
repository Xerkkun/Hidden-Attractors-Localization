from __future__ import annotations

import pytest
from pathlib import Path
from hidden_attractors.paths import PROJECT_ROOT
from hidden_attractors.workflows.config_loader import load_config

def test_classical_route_scope_defaults() -> None:
    # 1. Load the default classical Lure YAML config
    yaml_path = PROJECT_ROOT / "configs" / "chua_classical_lure_default.yaml"
    assert yaml_path.exists()
    
    cfg = load_config(yaml_path)
    
    # 2. Verify all experimental features are disabled by default
    assert cfg["machado_enabled"] is False
    assert cfg["biased_enabled"] is False
    assert cfg["seed_filter"]["enabled"] is False
    assert cfg["robustness"]["enabled"] is False
    assert cfg["run_basin_slices"] is False
    assert cfg["run_bifurcation"] is False
    assert cfg["workers"] == 1

def test_route_separation_enforced() -> None:
    # 3. Verify route separation rules
    # If transfer_mode is "integer" but seed_mode is "fractional", should raise ValueError
    invalid_mix = {
        "experiment": {"name": "invalid mix"},
        "system": {"system_id": "chua_fractional_saturation", "q": 0.9998},
        "modes": {
            "transfer_mode": "integer",
            "seed_mode": "fractional"
        }
    }
    # Wait, we can test validation via a temporary YAML or by calling load_config on mock raw data
    from hidden_attractors.workflows.config_loader import _validate, _apply_defaults, _normalize
    
    with pytest.raises(ValueError, match="Invalid mode mixture: transfer_mode is 'integer' but seed_mode or continuation_mode is 'fractional'"):
        _validate({"transfer_mode": "integer", "seed_mode": "fractional", "continuation_mode": "fractional"})
        
    # Verify Machado requires transfer_mode fractional
    with pytest.raises(ValueError, match="Generalised Machado Describing Function is only supported when transfer_mode is 'fractional'"):
        _validate({"machado_enabled": True, "transfer_mode": "integer"})
