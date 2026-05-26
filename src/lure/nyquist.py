import numpy as np
from scipy.optimize import minimize, root_scalar
from scipy.integrate import quad
from typing import Any, Dict, List, Tuple, Optional
from .transfer import W_eval

def find_harmonic_candidates(
    system: Any,
    transfer_mode: str,
    seed_strategy: str = "nyquist_df",
    df_residual_tol: float = 1e-2,
    omega_min: float = 0.01,
    omega_max: float = 20.0,
    amplitude_min: float = 0.01,
    amplitude_max: float = 20.0,
    grid_size_omega: int = 200,
    grid_size_amplitude: int = 200,
    root_refinement: bool = True,
    q: Optional[float] = None
) -> List[Tuple[float, float, float]]:
    """Find all candidate pairs (A0, omega0, k) solving the harmonic condition.
    
    Returns:
        List of tuples: (amplitude, omega, gain) sorted by gain k in ascending order.
    """
    if q is None:
        q = getattr(system, "q", 1.0)
        
    candidates = []
    
    if seed_strategy == "nyquist_df":
        # Strategy 1: 2D Grid search + optional refinement of ||W*N + 1||
        ws = np.linspace(omega_min, omega_max, grid_size_omega)
        as_ = np.linspace(amplitude_min, amplitude_max, grid_size_amplitude)
        
        best_pts = []
        res = np.zeros((len(as_), len(ws)))
        W_vals = [W_eval(w, q, transfer_mode, system.P, system.b, system.r) for w in ws]
        for i, A in enumerate(as_):
            N = system.describing_function(A)
            for j, w in enumerate(ws):
                res[i, j] = np.abs(W_vals[j] * N + 1.0)
        
        for i in range(1, len(as_) - 1):
            for j in range(1, len(ws) - 1):
                val = res[i, j]
                if val < 0.2:
                    if (val <= res[i-1:i+2, j-1:j+2]).all():
                        best_pts.append((as_[i], ws[j], val))
        
        for A_grid, w_grid, _ in best_pts:
            if root_refinement:
                def obj_func(z):
                    A, w = z
                    if A <= 0.01 or w <= 0.01:
                        return 1e6
                    try:
                        W = W_eval(w, q, transfer_mode, system.P, system.b, system.r)
                        N = system.describing_function(A)
                        return float(np.abs(W * N + 1.0))
                    except Exception:
                        return 1e6
                
                res_opt = minimize(
                    obj_func,
                    [A_grid, w_grid],
                    bounds=[(amplitude_min, amplitude_max), (omega_min, omega_max)],
                    method="L-BFGS-B"
                )
                if res_opt.success and res_opt.fun < df_residual_tol:
                    A_ref, w_ref = res_opt.x
                    k_ref = system.describing_function(A_ref)
                    if not any(np.allclose([A_ref, w_ref], [c[0], c[1]], rtol=1e-2) for c in candidates):
                        candidates.append((float(A_ref), float(w_ref), float(k_ref)))
            else:
                k_grid = system.describing_function(A_grid)
                candidates.append((float(A_grid), float(w_grid), float(k_grid)))
                
    elif seed_strategy in {"k_phi", "imw_gain"}:
        # Strategy 2: 1D phase crossing Im(W) = 0, then solve Phi(A) = 0
        scan_n = max(grid_size_omega, 20000)
        ws = np.linspace(omega_min, omega_max, scan_n)
        ims = []
        for w in ws:
            try:
                val = W_eval(w, q, transfer_mode, system.P, system.b, system.r)
                ims.append(val.imag)
            except Exception:
                ims.append(np.nan)
        
        omega_roots = []
        for j in range(len(ws) - 1):
            if np.isnan(ims[j]) or np.isnan(ims[j+1]):
                continue
            if ims[j] * ims[j+1] < 0.0:
                try:
                    def root_f(w):
                        return W_eval(w, q, transfer_mode, system.P, system.b, system.r).imag
                    
                    sol = root_scalar(root_f, bracket=[ws[j], ws[j+1]], method="bisect")
                    if sol.converged:
                        omega_roots.append(sol.root)
                except Exception:
                    pass
        
        for w0 in omega_roots:
            W0 = W_eval(w0, q, transfer_mode, system.P, system.b, system.r)
            if abs(W0.real) < 1e-12:
                continue
            k = -1.0 / W0.real
            
            def phi_func(A):
                if A <= 0:
                    return 0.0
                def integrand(t):
                    return (system.psi(A * np.cos(w0 * t)) - k * A * np.cos(w0 * t)) * np.cos(w0 * t)
                val, _ = quad(integrand, 0.0, 2.0 * np.pi / w0, limit=100)
                return val
            
            as_ = np.linspace(amplitude_min, amplitude_max, grid_size_amplitude)
            phi_vals = [phi_func(a) for a in as_]
            
            for i in range(len(as_) - 1):
                if phi_vals[i] * phi_vals[i+1] < 0.0:
                    try:
                        sol_A = root_scalar(phi_func, bracket=[as_[i], as_[i+1]], method="bisect")
                        if sol_A.converged:
                            A0 = sol_A.root
                            if not any(np.allclose([A0, w0], [c[0], c[1]], rtol=1e-2) for c in candidates):
                                candidates.append((float(A0), float(w0), float(k)))
                    except Exception:
                        pass
                        
    else:
        raise ValueError(f"Unknown seed_strategy: {seed_strategy}")
        
    # Sort candidates by gain k in ascending order
    candidates.sort(key=lambda x: x[2])
    return candidates
