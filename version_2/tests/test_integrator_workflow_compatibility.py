from __future__ import annotations

import sys
import pytest
from pathlib import Path

# Add version_2 to sys.path if not present
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

from hidden_attractors.integrations.selector import validate_integrator_compatibility
from hidden_attractors.workflows.centered_lure_df import run_workflow_integration


class MockSystem:
    def __init__(self):
        self.dimension = 3
        self.parameters = {"q": 1.0}
    def evaluate(self, x, params=None):
        import numpy as np
        return np.zeros_like(x)


def test_q1_rk4_valid():
    """q=1 + rk4 is allowed."""
    canonical = validate_integrator_compatibility("rk4", 1.0)
    assert canonical == "rk4"


def test_qless1_rk4_fails():
    """q<1 + rk4 is rejected."""
    with pytest.raises(ValueError, match="only supports integer-order systems"):
        validate_integrator_compatibility("rk4", 0.99)


def test_q1_abm_fails():
    """q=1 + abm is rejected."""
    with pytest.raises(ValueError, match="requires q < 1"):
        validate_integrator_compatibility("abm", 1.0)


def test_qless1_efork3_valid():
    """q<1 + efork3 is allowed."""
    canonical = validate_integrator_compatibility("efork3", 0.99)
    assert canonical == "efork3"


def test_run_workflow_integration_compat():
    """Test that run_workflow_integration checks integration compatibility."""
    import numpy as np
    sys_obj = MockSystem()
    x0 = np.array([1.0, 1.0, 1.0])
    
    # q=1 + rk4 should run fine (or call the integrator without error)
    config_rk4 = {
        "integrator": "rk4",
        "memory_mode": "full",
        "memory_window_length": 100,
        "divergence_norm": 120.0,
        "early_stop": {"enabled": False},
        "use_c_backend": False,
        "allow_python_fallback": True,
    }
    t_fin, x_fin, status = run_workflow_integration(
        system=sys_obj,
        x0=x0,
        q_val=1.0,
        h=0.01,
        t_final=0.02,
        config=config_rk4,
        equilibria=[]
    )
    assert status == "ok"
    
    # q=1 + abm should fail
    config_abm = {
        "integrator": "abm",
        "memory_mode": "full",
        "memory_window_length": 100,
        "divergence_norm": 120.0,
        "early_stop": {"enabled": False},
        "use_c_backend": False,
        "allow_python_fallback": True,
    }
    with pytest.raises(ValueError, match="requires q < 1"):
        run_workflow_integration(
            system=sys_obj,
            x0=x0,
            q_val=1.0,
            h=0.01,
            t_final=0.02,
            config=config_abm,
            equilibria=[]
        )
