import sys
from pathlib import Path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import numpy as np
import pytest
from src.integrators.fractional_c import fractional_integrate
from src.systems.registry import get_system_by_id

def test_availability_combinations():
    # Linear test system: D^q x = -a * x
    q = 0.95
    h = 0.05
    t_final = 1.0
    x0 = np.array([1.0], dtype=float)
    
    def rhs_linear(t, x):
        return -0.5 * x
        
    for method in ["abm", "efork"]:
        for mem_mode in ["full", "window"]:
            t, x, status, info = fractional_integrate(
                rhs=rhs_linear,
                x0=x0,
                q=q,
                h=h,
                t_final=t_final,
                method=method,
                memory_mode=mem_mode,
                memory_window_length=5,
                use_c_backend=True,
                allow_python_fallback=False
            )
            assert status == "ok"
            assert len(t) == 21
            assert x.shape == (21, 1)
            assert info["used_c_backend"] is True
            assert info["n_steps"] == 21

def test_dimension_generality():
    # Test 2D system
    def rhs_2d(t, x):
        return np.array([-x[0], -2.0 * x[1]])
    
    t, x, status, info = fractional_integrate(
        rhs=rhs_2d, x0=np.array([1.0, 1.0]), q=0.9, h=0.01, t_final=0.5,
        method="abm", memory_mode="full", use_c_backend=True
    )
    assert status == "ok"
    assert x.shape == (51, 2)
    
    # Test 4D artificial system
    def rhs_4d(t, x):
        return np.array([-x[0], -x[1], -x[2], -x[3]])
        
    t, x, status, info = fractional_integrate(
        rhs=rhs_4d, x0=np.array([1.0, 1.0, 1.0, 1.0]), q=0.9, h=0.01, t_final=0.5,
        method="efork", memory_mode="window", memory_window_length=10, use_c_backend=True
    )
    assert status == "ok"
    assert x.shape == (51, 4)

def test_history_propagation():
    def rhs(t, x):
        return -0.8 * x
        
    q = 0.95
    h = 0.05
    t_final = 0.5
    x0 = np.array([2.0], dtype=float)
    
    # Run full trajectory
    t_full, x_full, status_full, _ = fractional_integrate(
        rhs=rhs, x0=x0, q=q, h=h, t_final=1.0, method="abm", memory_mode="full"
    )
    
    # Run first half
    t_h1, x_h1, status_h1, _ = fractional_integrate(
        rhs=rhs, x0=x0, q=q, h=h, t_final=t_final, method="abm", memory_mode="full"
    )
    
    # Continue second half passing first half as history
    t_h2, x_h2, status_h2, _ = fractional_integrate(
        rhs=rhs,
        x0=x_h1[-1],
        q=q,
        h=h,
        t_final=t_final,
        method="abm",
        memory_mode="full",
        history_times=t_h1 - t_h1[-1],
        history_states=x_h1,
        return_history=True
    )
    
    # Check that they match within small tolerance
    assert np.allclose(x_full, x_h2, atol=1e-8)

def test_memory_truncation_window():
    def rhs(t, x):
        return -x
        
    M = 10
    t, x, status, info = fractional_integrate(
        rhs=rhs, x0=np.array([1.0]), q=0.9, h=0.02, t_final=1.0,
        method="abm", memory_mode="window", memory_window_length=M
    )
    
    assert status == "ok"
    assert info["truncated_memory"] is True
    assert info["memory_window_length"] == M

def test_no_fallback_silence():
    # If C backend execution fails, check that allow_python_fallback=False raises error
    # We force error by passing a wrong/incompatible function/type
    with pytest.raises(Exception):
        fractional_integrate(
            rhs="invalid_callable", x0=np.array([1.0]), q=0.9, h=0.01, t_final=0.2,
            method="abm", memory_mode="full", use_c_backend=True, allow_python_fallback=False
        )

def test_chua_smoke_all_combinations():
    sys_frac = get_system_by_id("chua_fractional_saturation")
    x0 = np.array([0.1, 0.1, 0.1])
    
    for method in ["abm", "efork"]:
        for mem_mode in ["full", "window"]:
            t, x, status, info = fractional_integrate(
                rhs=sys_frac.evaluate_rhs,
                x0=x0,
                q=sys_frac.q,
                h=0.02,
                t_final=0.2,
                method=method,
                memory_mode=mem_mode,
                memory_window_length=5,
                system=sys_frac,
                use_c_backend=True
            )
            assert status == "ok"
            assert len(t) == 11
            assert x.shape == (11, 3)
