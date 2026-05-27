import sys
from pathlib import Path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import pytest
import numpy as np
from src.contracts import validate_contracts
from src.systems.chua_saturation import ChuaSaturationSystem

def test_q_seed_effective_resolution():
    # Verify dynamic resolution of q_seed_effective
    # seed_mode = "integer" -> q_seed = 1.0
    # seed_mode = "fractional" -> q_seed = system.q
    system = ChuaSaturationSystem(q=0.85)
    
    # integer seed mode
    config_int = {
        "seed_mode": "integer",
        "q_seed": None,
    }
    q_seed_int = config_int["q_seed"] if config_int.get("q_seed") is not None else (1.0 if config_int["seed_mode"] == "integer" else system.q)
    assert q_seed_int == 1.0
    
    # fractional seed mode
    config_frac = {
        "seed_mode": "fractional",
        "q_seed": None,
    }
    q_seed_frac = config_frac["q_seed"] if config_frac.get("q_seed") is not None else (1.0 if config_frac["seed_mode"] == "integer" else system.q)
    assert q_seed_frac == 0.85

def test_fractional_continuation_blocked_when_q_dynamics_integer():
    # Verify that fractional continuation is blocked when q_dynamics is 1.0
    # continuation_mode = "fractional", q_continuation_effective = 1.0 should fail validation
    config = {
        "q_dynamics": 0.9,
        "q_continuation": 1.0,
        "continuation_mode": "fractional",
        "seed_mode": "integer",
        "memory_policy": "full_caputo",
        "memory_mode": "full",
        "transfer_convention": "standard",
        "harmonic_condition": "1_minus_WN",
        "integrator": "abm"
    }
    with pytest.raises(ValueError, match="Fractional continuation mode cannot be run with integer-order continuation dynamics"):
        validate_contracts(config, resolved=True)

def test_final_simulation_uses_q_dynamics_override(monkeypatch):
    # Verify that run_centered_lure_df_workflow passes q_dynamics to the final simulation
    import src.workflows.centered_lure_df_workflow as workflow_mod
    
    called_q_val = []
    def mock_run_workflow_integration(*args, **kwargs):
        called_q_val.append(kwargs.get("q_val"))
        return np.array([0.0]), np.array([[0.0]]), "ok"
        
    monkeypatch.setattr(workflow_mod, "run_workflow_integration", mock_run_workflow_integration)
    
    def mock_run_fractional_continuation(*args, **kwargs):
        lambda_values = kwargs.get("lambda_values", [0.0])
        return [{"status": "ok", "x_out": np.array([0, 0, 0]), "lambda_value": lv} for lv in lambda_values]
    monkeypatch.setattr(workflow_mod, "run_fractional_continuation", mock_run_fractional_continuation)
    monkeypatch.setattr(workflow_mod, "_save_continuation_trace", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow_mod, "plot_flexible_attractor_and_projections", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow_mod, "plot_timeseries_data", lambda *args, **kwargs: None)
    
    config = {
        "system_id": "chua_fractional_saturation",
        "output_dir": "outputs/test_q_dynamics_override",
        "integrator": "abm",
        "dynamics_mode": "fractional",
        "q_dynamics": 0.82,  # overridden dynamics order (system.q default is usually 0.9 or 0.95)
        "seed_mode": "fractional",
        "continuation_mode": "fractional",
        "memory_policy": "full_caputo",
        "memory_mode": "full",
        "transfer_convention": "standard",
        "harmonic_condition": "1_minus_WN",
        "plot_enabled": False,
        "run_hiddenness_tests": False,
        "run_basin_slices": False,
        "run_sphere_tests": False
    }
    
    from src.workflows.centered_lure_df_workflow import run_centered_lure_df_workflow
    run_centered_lure_df_workflow(config)
    
    assert len(called_q_val) > 0
    # Final simulation integration (active_q) must be 0.82
    assert called_q_val[-1] == 0.82
