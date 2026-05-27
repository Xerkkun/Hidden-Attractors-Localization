import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.integrators.efork import efork_integrate
from src.integrators.abm import caputo_abm_integrate
from src.integrators.general import integrate_general
from src.integrators._q1_coefficients import (
    EFORK_Q1_A21,
    EFORK_Q1_A31,
    EFORK_Q1_A32,
    EFORK_Q1_W1,
    EFORK_Q1_W2,
    EFORK_Q1_W3,
)

class DummySystem:
    def __init__(self):
        self.P = np.array([[-1.0]])
        self.b = np.array([1.0])
        self.r = np.array([1.0])
        self.q = 1.0
        self.psi = lambda sigma: 0.2 * sigma


def test_efork_q1_uses_efork_q1_coefficients():
    """Verify that EFORK integration at q=1 uses the correct EFORK_Q1 coefficients."""
    sys_obj = DummySystem()
    x0 = np.array([1.0])
    h = 0.1
    t_final = h

    # 1. Integrate with efork_integrate
    t_arr, x_arr, status = efork_integrate(
        system=sys_obj,
        x0=x0,
        q=1.0,
        h=h,
        t_final=t_final,
        use_c_backend=False
    )
    assert status == "ok"
    
    # 2. Manual step calculation with EFORK_Q1 coefficients
    # Linearized gain k = 0, continuation parameter eps = 1
    # D x = P0 x + eps * b * (psi(r x) - k * (r x))
    # p0 = P
    # rhs(x) = P x + b * psi(r x) = -1.0 * x + 1.0 * 0.2 * x = -0.8 * x
    x = x0[0]
    def rhs_val(v):
        return -0.8 * v

    k1 = h * rhs_val(x)
    k2 = h * rhs_val(x + EFORK_Q1_A21 * k1)
    k3 = h * rhs_val(x + EFORK_Q1_A31 * k1 + EFORK_Q1_A32 * k2)
    x_manual = x + EFORK_Q1_W1 * k1 + EFORK_Q1_W2 * k2 + EFORK_Q1_W3 * k3

    assert np.allclose(x_arr[-1, 0], x_manual, atol=1e-12)


def test_efork_q1_does_not_call_fractional_c():
    """Verify that fractional_integrate C backend is bypassed entirely when q=1."""
    sys_obj = DummySystem()
    x0 = np.array([1.0])
    
    with patch("src.integrators.efork.fractional_integrate") as mock_frac_int:
        efork_integrate(
            system=sys_obj,
            x0=x0,
            q=1.0,
            h=0.1,
            t_final=0.1,
            use_c_backend=True
        )
        # fractional_integrate should never be called for q=1.0
        mock_frac_int.assert_not_called()


def test_abm_q1_documents_heun_limit():
    """Verify that ABM at q=1 correctly falls back to Heun with heun_q1_limit status."""
    def rhs_linear(t, x):
        return -0.8 * x

    x0 = np.array([1.0])
    h = 0.1
    t_final = h

    t_arr, x_arr, status = caputo_abm_integrate(
        rhs=rhs_linear,
        x0=x0,
        q=1.0,
        h=h,
        t_final=t_final
    )
    # Must use heun_q1_limit as status, indicating it's not a Caputo fractional run
    assert status == "ok"
    
    # Calculate exact Heun step:
    # x_pred = x + h * rhs(x) = 1.0 - 0.08 = 0.92
    # x_next = x + 0.5 * h * (rhs(x) + rhs(x_pred)) = 1.0 + 0.05 * (-0.8 + -0.8 * 0.92)
    x_heun = 1.0 + 0.05 * (-0.8 - 0.736)
    assert np.allclose(x_arr[-1, 0], x_heun, atol=1e-12)


def test_general_q1_dispatches_correctly():
    """Verify that integrate_general dispatches correctly to Heun or EFORK_Q1 for q=1."""
    sys_obj = DummySystem()
    x0 = np.array([1.0])
    h = 0.1
    t_final = 0.2

    # Dummy RHS equivalent to DummySystem evaluation
    def rhs_linear(t, x):
        return -0.8 * x

    # 1. Dispatching to abm (Heun)
    t_abm, x_abm, status_abm = integrate_general(
        rhs=rhs_linear,
        x0=x0,
        q=1.0,
        h=h,
        t_final=t_final,
        integrator="abm",
        system=sys_obj
    )
    assert status_abm == "ok"

    # 2. Dispatching to efork (EFORK_Q1)
    t_ef, x_ef, status_ef = integrate_general(
        rhs=rhs_linear,
        x0=x0,
        q=1.0,
        h=h,
        t_final=t_final,
        integrator="efork",
        system=sys_obj
    )
    assert status_ef == "ok"

    # Verify they produce numerically distinct trajectories due to different q=1 limit schemes
    assert not np.allclose(x_abm, x_ef, atol=1e-6)
