import sys
from pathlib import Path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import numpy as np
import pytest
from src.contracts import validate_contracts

def test_validate_contracts_invalid_integrator_for_integer():
    # q_dynamics = 1.0 and integrator = 'abm' must fail
    config = {
        "q_dynamics": 1.0,
        "integrator": "abm",
        "seed_mode": "integer",
        "continuation_mode": "integer",
        "memory_policy": "none",
        "transfer_convention": "standard",
        "harmonic_condition": "1_minus_WN"
    }
    with pytest.raises(ValueError, match="ABM integrator is not allowed for integer-order dynamics"):
        validate_contracts(config)

def test_validate_contracts_invalid_integrator_for_fractional():
    # q_dynamics = 0.9 and integrator = 'heun' must fail
    config = {
        "q_dynamics": 0.9,
        "integrator": "heun",
        "seed_mode": "fractional",
        "continuation_mode": "fractional",
        "memory_policy": "full_caputo",
        "memory_mode": "full",
        "transfer_convention": "standard",
        "harmonic_condition": "1_minus_WN"
    }
    with pytest.raises(ValueError, match="is not allowed for fractional-order dynamics"):
        validate_contracts(config)

def test_validate_contracts_invalid_continuation_integer():
    # continuation_mode = 'integer' and integrator = 'abm' must fail
    config = {
        "q_dynamics": 1.0,
        "integrator": "abm",
        "seed_mode": "integer",
        "continuation_mode": "integer",
        "memory_policy": "none",
        "transfer_convention": "standard",
        "harmonic_condition": "1_minus_WN"
    }
    with pytest.raises(ValueError, match="ABM integrator is not allowed for integer-order dynamics"):
        validate_contracts(config)

def test_validate_contracts_invalid_continuation_fractional():
    # continuation_mode = 'fractional' and integrator = 'efork_q1' must fail
    config = {
        "q_dynamics": 1.0, # set to 1.0 to bypass the q_dynamics < 1.0 check
        "integrator": "efork_q1",
        "seed_mode": "fractional",
        "continuation_mode": "fractional",
        "memory_policy": "full_caputo",
        "memory_mode": "full",
        "transfer_convention": "standard",
        "harmonic_condition": "1_minus_WN"
    }
    with pytest.raises(ValueError, match="is not allowed for fractional continuation"):
        validate_contracts(config)

def test_workflow_prevents_abm_q1(monkeypatch):
    # Mock caputo_abm_integrate to raise an assertion if q=1.0 is passed
    import src.integrators.abm as abm_mod
    
    def mock_caputo_abm_integrate(*args, **kwargs):
        q_val = kwargs.get("q")
        if q_val is not None and abs(q_val - 1.0) < 1e-9:
            raise AssertionError("Workflow illegally called caputo_abm_integrate with q=1.0!")
        return np.array([0.0]), np.array([[0.0]]), "ok"
        
    monkeypatch.setattr(abm_mod, "caputo_abm_integrate", mock_caputo_abm_integrate)
    
    # Running the workflow with integrator="abm" and dynamics_mode="integer" must be rejected by contracts
    from src.workflows.centered_lure_df_workflow import run_centered_lure_df_workflow
    
    config = {
        "system_id": "chua_fractional_saturation",
        "output_dir": "outputs/test_prevent_abm",
        "integrator": "abm",
        "dynamics_mode": "integer",
        "seed_mode": "integer",
        "continuation_mode": "integer",
        "memory_policy": "none",
        "transfer_convention": "standard",
        "harmonic_condition": "1_minus_WN"
    }
    
    with pytest.raises(ValueError):
        run_centered_lure_df_workflow(config)

