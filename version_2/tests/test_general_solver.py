import sys
from pathlib import Path
import numpy as np
import pytest

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import ctypes
import importlib

from typing import Any
from hidden_attractors.integrations.general import integrate_general
from hidden_attractors.integrations.fractional_c import GeneralFractionalCBackend
from hidden_attractors.native.rhs_registry import get_c_rhs_and_params
from hidden_attractors.systems import get_system
import dataclasses

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


def test_integrate_general_linear_system_abm():
    # Test integration of a simple general 2D linear system: D_t^q x = A @ x
    # rhs = lambda t, x: np.array([-x[0], -2.0 * x[1]])
    def rhs(t, x):
        return np.array([-x[0], -2.0 * x[1]])
        
    x0 = np.array([1.0, 2.0])
    
    # 1. Test general ABM solver (q = 0.95)
    t, x, status = integrate_general(
        rhs, x0, q=0.95, h=0.05, t_final=1.0, integrator="abm"
    )
    
    assert status == "ok"
    assert len(t) > 0
    assert x.shape == (len(t), 2)
    # Check that it decays
    assert x[-1, 0] < x[0, 0]
    assert x[-1, 1] < x[0, 1]

def test_integrate_general_linear_system_efork():
    # 2. Test general EFORK-3 solver (q = 0.9)
    def rhs(t, x):
        return np.array([-0.5 * x[0], -x[1]])
        
    x0 = np.array([5.0, 5.0])
    
    t, x, status = integrate_general(
        rhs, x0, q=0.9, h=0.02, t_final=0.5, integrator="efork"
    )
    
    assert status == "ok"
    assert len(t) > 0
    assert x.shape == (len(t), 2)
    assert x[-1, 0] < x[0, 0]

def test_integrate_general_chua_c_backend():
    # 3. Test that passing a Chua system with EFORK successfully leverages the EFORK C backend
    sys_frac = get_system_by_id("chua_fractional_saturation")
    x0 = np.array([1.0, 1.0, -0.4])
    
    t, x, status = integrate_general(
        lambda t_val, x_val: sys_frac.evaluate(x_val), x0, q=float(sys_frac.parameters.get("q")), h=0.01, t_final=1.0,
        integrator="efork", system=sys_frac, use_c_backend=True
    )
    
    assert status == "ok"
    assert len(t) == 101
    assert x.shape == (101, 3)

def test_integrate_general_integer_solver():
    # 4. Test standard 2nd-order Heun integer solver (q = 1.0)
    def rhs(x):
        return np.array([-x[0]])
        
    x0 = np.array([10.0])
    
    t, x, status = integrate_general(
        rhs, x0, q=1.0, h=0.1, t_final=2.0
    )
    
    assert status == "ok"
    assert len(t) == 21
    # Analytical solution for q=1 is x(t) = 10 * exp(-t)
    assert np.allclose(x[-1, 0], 10.0 * np.exp(-2.0), rtol=1e-2)

def test_general_solver_c_vs_python():
    # Verify that general EFORK in C and Python yield strictly equivalent results
    def rhs(t, x):
        return np.array([-0.8 * x[0], x[0] - x[1]])
        
    x0 = np.array([2.0, 1.0])
    
    t_c, x_c, status_c = integrate_general(
        rhs, x0, q=0.85, h=0.01, t_final=0.4, integrator="efork", use_c_backend=True
    )
    
    t_py, x_py, status_py = integrate_general(
        rhs, x0, q=0.85, h=0.01, t_final=0.4, integrator="efork", use_c_backend=False
    )
    
    assert status_c == "ok"
    assert status_py == "ok"
    assert np.allclose(t_c, t_py)
    assert np.allclose(x_c, x_py, atol=1e-6)


def test_registered_arctan_rhs_matches_python_model_with_rho():
    system = get_system_by_id(
        "chua_fractional_arctan",
        q=0.99,
        alpha=8.4562,
        beta=12.0732,
        gamma=0.0052,
        a1=0.4,
        a2=-1.5585,
        rho=0.75,
    )
    backend = GeneralFractionalCBackend.get_instance()
    rhs_ptr, params = get_c_rhs_and_params(system, backend.lib)
    assert rhs_ptr is not None

    c_rhs = backend.RHS_CALLBACK(rhs_ptr)
    state = np.ascontiguousarray([0.73, -0.12, 0.31], dtype=np.float64)
    derivative = np.empty(3, dtype=np.float64)
    c_rhs(
        0.0,
        state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        derivative.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        3,
        ctypes.cast(ctypes.byref(params), ctypes.c_void_p),
    )

    assert np.allclose(derivative, system.evaluate(state), rtol=1e-13, atol=1e-13)


def test_arctan_candidate_abm_native_matches_python_full_memory():
    system = get_system_by_id(
        "chua_fractional_arctan",
        q=0.9999,
        alpha=21.849356906616716,
        beta=19.081840840860202,
        gamma=0.007378011979156531,
        a1=0.04228979343578827,
        a2=-3.3367815123026694,
        rho=1.7984259332820332,
    )
    initial = np.array([0.73, -0.12, 0.31], dtype=float)
    kwargs = dict(
        q=0.9999,
        h=0.01,
        t_final=0.2,
        integrator="abm",
        memory_mode="full",
        system=system,
    )

    t_native, x_native, status_native = integrate_general(
        lambda _t, state: system.evaluate(state),
        initial,
        use_c_backend=True,
        **kwargs,
    )
    t_python, x_python, status_python = integrate_general(
        lambda _t, state: system.evaluate(state),
        initial,
        use_c_backend=False,
        **kwargs,
    )

    assert status_native == status_python == "ok"
    np.testing.assert_allclose(t_native, t_python, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(x_native, x_python, rtol=1.0e-12, atol=1.0e-12)
