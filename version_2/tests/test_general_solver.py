import sys
from pathlib import Path
import numpy as np
import pytest

workspace_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(workspace_root))

from src.integrators.general import integrate_general
from src.systems.registry import get_system_by_id

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
        sys_frac.evaluate_rhs, x0, q=sys_frac.q, h=0.01, t_final=1.0,
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

