import sys
from pathlib import Path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import pytest
import numpy as np
from src.continuation.continuation_integer import run_integer_continuation
from src.systems.chua_saturation import ChuaSaturationSystem

def test_integer_continuation_solvers():
    system = ChuaSaturationSystem(q=1.0)
    seed = np.array([0.1, 0.1, 0.1])
    k_gain = -0.5
    lambda_values = [0.1, 0.2]
    h = 0.05
    
    # efork_q1 must work
    steps_efork = run_integer_continuation(
        system=system,
        seed_x0=seed,
        k_gain=k_gain,
        lambda_values=lambda_values,
        h=h,
        t_transient=0.2,
        t_keep=0.2,
        integrator="efork_q1"
    )
    assert len(steps_efork) == 2
    assert all(s["status"] == "ok" for s in steps_efork)
    
    # heun must work
    steps_heun = run_integer_continuation(
        system=system,
        seed_x0=seed,
        k_gain=k_gain,
        lambda_values=lambda_values,
        h=h,
        t_transient=0.2,
        t_keep=0.2,
        integrator="heun"
    )
    assert len(steps_heun) == 2
    assert all(s["status"] == "ok" for s in steps_heun)
    
    # abm must raise ValueError
    with pytest.raises(ValueError, match="ABM is not available for integer continuation q=1"):
        run_integer_continuation(
            system=system,
            seed_x0=seed,
            k_gain=k_gain,
            lambda_values=lambda_values,
            h=h,
            t_transient=0.2,
            t_keep=0.2,
            integrator="abm"
        )
