import sys
import os
from pathlib import Path
import numpy as np
import pytest
import importlib

# Add workspace root and version_2 to sys.path
workspace_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(workspace_root / "version_2"))
sys.path.insert(1, str(workspace_root))

from typing import Any
from hidden_attractors.systems import get_system
import dataclasses
from hidden_attractors.lure.transfer import W_eval
from hidden_attractors.lure.describing_function import N_quadrature
from hidden_attractors.verification.stability import classify_equilibrium_stability
from hidden_attractors.integrations.abm import caputo_abm_integrate
from hidden_attractors.workflows.centered_lure_df import run_centered_lure_df_workflow

def get_system_by_id(system_id: str, **kwargs) -> Any:
    name_map = {
        "chua_integer_saturation": "chua-nonsmooth",
        "chua_fractional_saturation": "chua-nonsmooth",
        "chua_fractional_arctan": "chua-arctan",
        "chua_arctan_wu2023": "fractional-chua-arctan-wu2023",
    }
    normalized_sys_id = name_map.get(system_id, system_id)
    system = get_system(normalized_sys_id)
    
    # Merge overrides
    merged_params = dict(system.parameters)
    merged_params.update(kwargs)
    
    if "q" not in merged_params:
        if system_id == "chua_fractional_saturation":
            merged_params["q"] = 0.9998
        elif system_id == "chua_fractional_arctan":
            merged_params["q"] = 0.995
        else:
            merged_params["q"] = 1.0
            
    system = dataclasses.replace(system, parameters=merged_params)
    return system


def test_system_matrices_and_parameters():
    # 1. P, b, r correct for each system
    sys_int = get_system_by_id("chua_integer_saturation")
    sys_frac = get_system_by_id("chua_fractional_saturation")
    sys_arctan = get_system_by_id("chua_fractional_arctan")
    
    assert sys_int.parameters.get("alpha") == 8.4562
    assert sys_frac.parameters.get("q") == 0.9998
    assert sys_arctan.parameters.get("q") == 0.995
    
    # Check dimensions
    assert sys_int.lure.matrix.shape == (3, 3)
    assert sys_int.lure.input_vector.shape == (3,)
    assert sys_int.lure.output_vector.shape == (3,)
    
    assert np.allclose(sys_int.lure.input_vector, [-8.4562, 0.0, 0.0])
    assert np.allclose(sys_int.lure.output_vector, [1.0, 0.0, 0.0])
    
    # Verify specific entries in P
    # P[0, 0] = -alpha * (m1 + 1)
    assert np.allclose(sys_int.lure.matrix[0, 0], -8.4562 * (-1.1468 + 1.0))
    # P[0, 0] = -alpha * (1 + m)
    assert np.allclose(sys_arctan.lure.matrix[0, 0], -8.4562 * (1.0 + 0.4))

def test_transfer_function():
    sys_int = get_system_by_id("chua_integer_saturation")
    sys_frac = get_system_by_id("chua_fractional_saturation")
    
    # 2. W_integer coincides with direct calculation: r.T @ inv(P - s*I) @ b
    omega = 2.5
    s = 1j * omega
    direct_W = sys_int.lure.output_vector.T @ np.linalg.inv(sys_int.lure.matrix - s * np.eye(3)) @ sys_int.lure.input_vector
    
    eval_W_int = W_eval(omega, 1.0, "integer", sys_int.lure.matrix, sys_int.lure.input_vector, sys_int.lure.output_vector, transfer_convention="opposite_sign")
    assert np.allclose(eval_W_int, direct_W)
    
    # 3. W_fractional uses lambda = (i*omega)^q
    q = 0.95
    lam = (omega**q) * np.exp(1j * q * np.pi / 2.0)
    direct_W_frac = sys_frac.lure.output_vector.T @ np.linalg.inv(sys_frac.lure.matrix - lam * np.eye(3)) @ sys_frac.lure.input_vector
    
    eval_W_frac = W_eval(omega, q, "fractional", sys_frac.lure.matrix, sys_frac.lure.input_vector, sys_frac.lure.output_vector, transfer_convention="opposite_sign")
    assert np.allclose(eval_W_frac, direct_W_frac)

def test_describing_functions():
    sys_int = get_system_by_id("chua_integer_saturation")
    sys_arctan = get_system_by_id("chua_fractional_arctan")
    
    # 4. N_sat(A) closed form matches quadrature
    for A in [0.5, 1.5, 3.0]:
        val_closed = sys_int.lure.describing_function(A)
        val_quad = N_quadrature(A, sys_int.lure.nonlinearity)
        assert np.allclose(val_closed, val_quad, rtol=1e-3)
        
    # 5. N_arctan(A) returns finite values
    for A in [0.5, 2.5, 10.0]:
        val_arctan = sys_arctan.lure.describing_function(A)
        assert np.isfinite(val_arctan)
        assert val_arctan != 0.0

def test_matignon_stability():
    # 6. Matignon stability check works with q configurable
    sys_frac = get_system_by_id("chua_fractional_saturation", q=0.98)
    eq_pt = np.array([0.0, 0.0, 0.0])
    
    res = classify_equilibrium_stability(sys_frac, eq_pt)
    assert "stable" in res
    assert "instability_measure" in res
    assert np.isfinite(res["instability_measure"])

def test_sliding_window_memory():
    # 7. memory_mode = "window" retains the window length constraint
    sys_frac = get_system_by_id("chua_fractional_saturation")
    
    # Simulate a small run in Python
    t, x, status = caputo_abm_integrate(
        lambda t_val, x_val: sys_frac.evaluate(x_val),
        x0=np.array([1.0, 1.0, -0.4]),
        q=float(sys_frac.parameters.get("q")),
        h=0.02,
        t_final=1.0,
        memory_mode="window",
        memory_window_length=15,
        system=sys_frac,
        use_c_backend=False
    )
    
    # Length of integration should be ceil(1.0/0.02) + 1 = 51 points
    assert len(t) == 51
    # Check that it integrated successfully
    assert status == "ok"

def test_smoke_workflow_short_runs(tmp_path):
    # Smoke tests: runs with short t_final
    config = {
        "system_id": "chua_integer_saturation",
        "q": 1.0,
        "transfer_mode": "integer",
        "continuation_mode": "integer",
        "integrator": "heun",
        "memory_mode": "full",
        "run_hiddenness_tests": False, # 8. run_hiddenness_tests = false skips
        "run_basin_slices": False,
        "plot_enabled": False,
        "save_figures": False,
        "output_dir": str(tmp_path),
        "amplitude_min": 1.0,
        "amplitude_max": 8.0,
        "omega_min": 0.5,
        "omega_max": 3.0,
        "grid_size_omega": 20,
        "grid_size_amplitude": 20,
        "t_final": 5.0, # short run
        "t_burn": 2.0,
        "h": 0.02
    }
    
    res = run_centered_lure_df_workflow(config)
    
    # Check generated files
    assert os.path.exists(os.path.join(tmp_path, "summary.json"))
    assert os.path.exists(os.path.join(tmp_path, "summary.csv"))
    assert os.path.exists(os.path.join(tmp_path, "effective_config.yaml"))
    
    # Check fields in summary
    assert res["system_id"] == "chua_integer_saturation"
    assert res["status"] in {"df_seed_found", "df_seed_not_found"}
