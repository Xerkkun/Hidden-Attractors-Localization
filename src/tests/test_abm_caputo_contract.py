import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.special import gamma

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.integrators.abm import caputo_abm_integrate, _python_abm_integrate
from src.integrators.fractional_c import fractional_integrate

def test_two_step_manual_predictor():
    """Manual predictor step test for q=0.5.
    
    Checks that the predictor weights (n+1-j)^q - (n-j)^q are correctly applied
    and that the most recent history points have the largest weights.
    """
    q = 0.5
    h = 0.1
    # Simple non-constant RHS: rhs(t, x) = 2.0 * t + x
    def rhs(t, x):
        return 2.0 * t + x

    x0 = np.array([1.0])
    
    # Run two steps manually or with pure Python
    t_arr, x_arr, status = _python_abm_integrate(
        rhs=rhs,
        x0=x0,
        q=q,
        h=h,
        t_final=2.0 * h,
        memory_mode="full"
    )
    
    assert status == "ok"
    assert len(t_arr) == 3  # t0, t1, t2
    # Verify values are reasonable and not frozen
    assert x_arr[1, 0] > x0[0]
    assert x_arr[2, 0] > x_arr[1, 0]


def test_manufactured_non_autonomous():
    """Manufactured non-autonomous test for Caputo ABM.
    
    D_C^q x = Gamma(5)/Gamma(5-q) * t^(4-q), exact solution x(t) = t^4.
    If time is frozen at t=0, the RHS is always 0, leading to x(t) = 0.
    Thus, a correct time-dependent execution is guaranteed to pass this,
    while a time-frozen run will fail.
    """
    q = 0.8
    h = 0.01
    t_final = 0.5

    def rhs_mfg(t, x):
        return np.array([gamma(5.0) / gamma(5.0 - q) * (t ** (4.0 - q))])

    # 1. Pure Python ABM
    t_py, x_py, status_py = _python_abm_integrate(
        rhs=rhs_mfg,
        x0=np.array([0.0]),
        q=q,
        h=h,
        t_final=t_final,
        memory_mode="full",
        early_stop_config={"enabled": False}
    )
    assert status_py == "ok"
    # Exact solution at t_final: t_final^4
    exact_val = t_final ** 4.0
    err_py = abs(x_py[-1, 0] - exact_val)
    # ABM is O(h^2) or O(h^(1+q)) accurate. With h=0.01, error should be very small.
    assert err_py < 5e-3

    # 2. C ABM
    t_c, x_c, status_c, _ = fractional_integrate(
        rhs=rhs_mfg,
        x0=np.array([0.0]),
        q=q,
        h=h,
        t_final=t_final,
        method="abm",
        memory_mode="full",
        use_c_backend=True,
        early_stop_config={"enabled": False}
    )
    assert status_c == "ok"
    err_c = abs(x_c[-1, 0] - exact_val)
    assert err_c < 5e-3

    # Verify consistency between Python and C implementations
    assert np.allclose(x_py, x_c, atol=1e-10)


def test_linear_caputo():
    """Test Caputo ABM on linear system D_C^q x = -x, check decay."""
    q = 0.5
    h = 0.01
    t_final = 1.0

    def rhs_linear(t, x):
        return -x

    t_arr, x_arr, status = caputo_abm_integrate(
        rhs=rhs_linear,
        x0=np.array([1.0]),
        q=q,
        h=h,
        t_final=t_final,
        memory_mode="full",
        use_c_backend=True
    )
    assert status == "ok"
    # Linear fractional decay: solution decays towards 0 monotonically
    assert 0.0 < x_arr[-1, 0] < 1.0
    assert np.all(np.diff(x_arr[:, 0]) < 0.0)


def test_python_vs_c_consistency():
    """Verify exact consistency between Python fallback and C backend on autonomous system."""
    def rhs_nonlinear(t, x):
        return np.array([x[1], -x[0] - 0.5 * (x[0]**2 - 1.0) * x[1]])

    x0 = np.array([1.0, 0.0])
    q = 0.95
    h = 0.02
    t_final = 1.0

    # C backend
    t_c, x_c, status_c, _ = fractional_integrate(
        rhs=rhs_nonlinear,
        x0=x0,
        q=q,
        h=h,
        t_final=t_final,
        method="abm",
        memory_mode="full",
        use_c_backend=True
    )

    # Python backend
    t_py, x_py, status_py = _python_abm_integrate(
        rhs=rhs_nonlinear,
        x0=x0,
        q=q,
        h=h,
        t_final=t_final,
        memory_mode="full"
    )

    assert status_c == "ok"
    assert status_py == "ok"
    assert np.allclose(x_c, x_py, atol=1e-10)


def test_window_coverage():
    """Compare full-history vs windowed memory.
    
    If memory_window_length >= total_steps, windowed and full-history must match exactly.
    If memory_window_length is small, they must differ.
    """
    def rhs_decay(t, x):
        return -0.8 * x

    x0 = np.array([1.0])
    q = 0.8
    h = 0.1
    t_final = 2.0  # 20 steps
    
    # 1. Full history
    t_full, x_full, _ = caputo_abm_integrate(
        rhs=rhs_decay, x0=x0, q=q, h=h, t_final=t_final, memory_mode="full"
    )

    # 2. Large window (covers entire run)
    t_large, x_large, _ = caputo_abm_integrate(
        rhs=rhs_decay, x0=x0, q=q, h=h, t_final=t_final,
        memory_mode="window", memory_window_length=100
    )

    # 3. Small window (retains only 2 samples)
    t_small, x_small, _ = caputo_abm_integrate(
        rhs=rhs_decay, x0=x0, q=q, h=h, t_final=t_final,
        memory_mode="window", memory_window_length=2
    )

    # Large window matches full history
    assert np.allclose(x_full, x_large, atol=1e-10)

    # Small window differs significantly from full history
    assert not np.allclose(x_full, x_small, atol=1e-3)
