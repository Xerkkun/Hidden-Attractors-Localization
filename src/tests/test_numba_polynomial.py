import sys
from pathlib import Path
import numpy as np
import pytest

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.systems.chua_polynomial import ChuaPolynomialSystem
from src.systems.registry import get_system_by_id
from src.integrators.general import integrate_general
from src.integrators.numba_kernels import NUMBA_AVAILABLE, integrate_efork3_q1_numba

def test_chua_polynomial_system_properties():
    """Verify standard properties, nonlinearity, and describing functions of ChuaPolynomialSystem."""
    system = ChuaPolynomialSystem(alpha=9.0, beta=14.0, gamma=0.01, coeff=2.5, q=1.0)
    
    assert system.alpha == 9.0
    assert system.beta == 14.0
    assert system.gamma == 0.01
    assert system.coeff == 2.5
    assert system.q == 1.0
    assert system.system_id == "chua_integer_polynomial"

    # Test P, b, r
    assert np.allclose(system.P, np.array([
        [-9.0, 9.0, 0.0],
        [1.0, -1.0, 1.0],
        [0.0, -14.0, -0.01]
    ]))
    assert np.allclose(system.b, np.array([-9.0, 0.0, 0.0]))
    assert np.allclose(system.r, np.array([1.0, 0.0, 0.0]))

    # Test psi nonlinearity: psi(sigma) = coeff * (sigma^3 - sigma)
    # For sigma = 2.0: 2.5 * (8.0 - 2.0) = 15.0
    assert np.allclose(system.psi(2.0), 15.0)
    assert np.allclose(system.psi(-2.0), -15.0)

    # Test describing function
    # N(A) = coeff * (0.75 * A^2 - 1.0)
    # For A = 2.0: 2.5 * (0.75 * 4 - 1) = 2.5 * (3 - 1) = 5.0
    assert np.allclose(system.describing_function(2.0), 5.0)
    assert system.is_nonsmooth() is False
    assert system.has_closed_form_describing_function() is True
    assert system.describing_function_capabilities["closed_form"] is True


def test_registry_integration():
    """Verify registering and retrieving the polynomial systems via the factory registry."""
    sys_int = get_system_by_id("chua_integer_polynomial", coeff=1.5)
    assert isinstance(sys_int, ChuaPolynomialSystem)
    assert sys_int.q == 1.0
    assert sys_int.coeff == 1.5

    sys_frac = get_system_by_id("chua_fractional_polynomial", q=0.98)
    assert isinstance(sys_frac, ChuaPolynomialSystem)
    assert sys_frac.q == 0.98


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is not installed")
def test_polynomial_numba_integration_match():
    """Verify that Numba JIT integration of ChuaPolynomialSystem matches Python loop exactly."""
    system = ChuaPolynomialSystem(alpha=8.0, beta=12.0, gamma=0.005, coeff=0.5, q=1.0)
    x0 = np.array([0.1, -0.1, 0.2], dtype=np.float64)
    h = 0.01
    t_final = 0.5

    # 1. Integrate using pure Python (passing system=None inside integrate_general bypasses Numba)
    t_py, x_py, status_py = integrate_general(
        rhs=lambda t, x: system.evaluate_rhs(x),
        x0=x0,
        q=1.0,
        h=h,
        t_final=t_final,
        integrator="efork3",
        system=None,
    )

    # 2. Integrate using JIT (passing system=system triggers Numba path)
    t_nb, x_nb, status_nb = integrate_general(
        rhs=lambda t, x: system.evaluate_rhs(x),
        x0=x0,
        q=1.0,
        h=h,
        t_final=t_final,
        integrator="efork3",
        system=system,
    )

    assert status_py == status_nb
    assert np.allclose(t_py, t_nb, atol=1e-12)
    assert np.allclose(x_py, x_nb, atol=1e-12)
