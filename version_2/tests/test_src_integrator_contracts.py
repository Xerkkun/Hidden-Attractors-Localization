import sys
from pathlib import Path
import numpy as np
import pytest

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from hidden_attractors.solvers import efork3_caputo_integrate
from hidden_attractors.native.backends import GeneralFDEBackend

def test_efork_python_vs_c_backend():
    """Verify that version_2 Python reference EFORK and C FDE backend match."""
    backend = GeneralFDEBackend.build()
    
    # Simple linear decay system: D^q x = -0.5 * x
    def rhs(t, x):
        return -0.5 * x

    x0 = np.array([1.0])
    q = 0.8
    h = 0.02
    t_final = 0.5
    
    # Python reference EFORK-3
    t_py, x_py = efork3_caputo_integrate(
        rhs=rhs,
        y0=x0,
        alpha=q,
        h=h,
        t_final=t_final
    )
    
    # C backend GeneralFDEBackend
    # Returns raw out array which contains t and x values interleaved as: [t0, x0_0, t1, x1_0, ...]
    nsteps = int(round(t_final / h))
    dim = 1
    out_buf = np.zeros((nsteps + 1) * (dim + 1), dtype=np.float64)
    
    def c_rhs(t_val, x_ptr, f_ptr):
        x_arr = np.ctypeslib.as_array(x_ptr, shape=(dim,))
        f_arr = np.ctypeslib.as_array(f_ptr, shape=(dim,))
        f_arr[0] = -0.5 * x_arr[0]
        
    rc = backend.lib.integrate_general_efork_c(
        backend.RHS_CALLBACK(c_rhs),
        x0,
        dim,
        q,
        h,
        t_final,
        120.0,
        out_buf
    )
    assert rc >= 0
    
    # Reshape C output
    out_reshaped = out_buf.reshape(-1, dim + 1)
    t_c = out_reshaped[:, 0]
    x_c = out_reshaped[:, 1:]
    
    # Check that they match close to tolerance
    assert np.allclose(t_py, t_c, atol=1e-12)
    # Check states
    assert np.allclose(x_py[:, 0], x_c[:, 0], atol=1e-8)


def test_general_fde_solver_arbitrary_dimension():
    """Verify that our dynamic VLA memory refactor successfully solves dimensions > 100.
    
    This test runs EFORK C solver with dim = 105. Prior to our fix, it would have either crashed,
    restricted memory calculation to 100, or returned garbage/segfaulted.
    """
    backend = GeneralFDEBackend.build()
    
    # dim = 105
    dim = 105
    x0 = np.ones(dim, dtype=np.float64)
    
    def rhs_large(t, x):
        # simple decay for each dimension
        return -0.2 * x

    q = 0.95
    h = 0.05
    t_final = 0.1 # 2 steps
    
    nsteps = int(round(t_final / h))
    out_buf = np.zeros((nsteps + 1) * (dim + 1), dtype=np.float64)
    
    # Construct standard wrapper callback
    def c_rhs(t_val, x_ptr, f_ptr):
        x_arr = np.ctypeslib.as_array(x_ptr, shape=(dim,))
        f_arr = np.ctypeslib.as_array(f_ptr, shape=(dim,))
        deriv = rhs_large(t_val, x_arr)
        f_arr[:] = deriv[:]
        
    rc = backend.lib.integrate_general_efork_c(
        backend.RHS_CALLBACK(c_rhs),
        x0,
        dim,
        q,
        h,
        t_final,
        120.0,
        out_buf
    )
    
    assert rc >= 0
    # Reshape and check that all dimensions have integrated correctly without crashing
    out_reshaped = out_buf.reshape(-1, dim + 1)
    states = out_reshaped[:, 1:]
    assert states.shape == (3, 105)
    # Assert they all decayed properly from initial state of 1.0
    assert np.all(states[-1, :] < 1.0)
    assert np.all(states[-1, :] > 0.0)


def test_general_fde_solver_q1_guard():
    """Verify that GeneralFDEBackend general_fde_solver rejects q=1.0 for fractional EFORK."""
    backend = GeneralFDEBackend.build()
    
    x0 = np.array([1.0])
    dim = 1
    q = 1.0 # Invalid for fractional EFORK solver which expects 0 < q < 1
    h = 0.01
    t_final = 0.1
    out_buf = np.zeros(11 * (dim + 1), dtype=np.float64)
    
    rc = backend.lib.integrate_general_efork_c(
        backend.RHS_CALLBACK(lambda t, x, f: None),
        x0,
        dim,
        q,
        h,
        t_final,
        120.0,
        out_buf
    )
    # C solver expects q < 1.0 and returns error code -1 on invalid q
    assert rc == -1
