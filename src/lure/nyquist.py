import numpy as np
from scipy.optimize import minimize, root_scalar
from scipy.integrate import quad
from typing import Any, Dict, List, Tuple
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
    root_refinement: bool = True
) -> List[Tuple[float, float, float]]:
    """Find all candidate pairs (A0, omega0, k) solving the harmonic condition.
    
    Returns:
        List of tuples: (amplitude, omega, gain)
    """
    candidates = []
    
    if seed_strategy == "nyquist_df":
        # Strategy 1: 2D Grid search + optional refinement of ||W*N + 1||
        ws = np.linspace(omega_min, omega_max, grid_size_omega)
        as_ = np.linspace(amplitude_min, amplitude_max, grid_size_amplitude)
        
        best_pts = []
        # We search for local minima of the residual
        res = np.zeros((len(as_), len(ws)))
        W_vals = [W_eval(w, system.q, transfer_mode, system.P, system.b, system.r) for w in ws]
        for i, A in enumerate(as_):
            N = system.describing_function(A)
            for j, w in enumerate(ws):
                res[i, j] = np.abs(W_vals[j] * N + 1.0)
        
        # Identify grid points with residual < 0.2 (coarse filter)
        # and checking if it's a local minimum in a 3x3 window
        for i in range(1, len(as_) - 1):
            for j in range(1, len(ws) - 1):
                val = res[i, j]
                if val < 0.2:
                    # check if local minimum
                    if (val <= res[i-1:i+2, j-1:j+2]).all():
                        best_pts.append((as_[i], ws[j], val))
        
        for A_grid, w_grid, _ in best_pts:
            if root_refinement:
                # Refine using bounded minimization of the absolute residual
                def obj_func(z):
                    A, w = z
                    if A <= 0.01 or w <= 0.01:
                        return 1e6
                    try:
                        W = W_eval(w, system.q, transfer_mode, system.P, system.b, system.r)
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
                    # Avoid duplicates
                    if not any(np.allclose([A_ref, w_ref], [c[0], c[1]], rtol=1e-2) for c in candidates):
                        candidates.append((float(A_ref), float(w_ref), float(k_ref)))
            else:
                k_grid = system.describing_function(A_grid)
                candidates.append((float(A_grid), float(w_grid), float(k_grid)))
                
    elif seed_strategy == "k_phi":
        # Strategy 2: 1D phase crossing Im(W) = 0, then solve Phi(A) = 0
        ws = np.linspace(omega_min, omega_max, grid_size_omega)
        ims = []
        for w in ws:
            try:
                val = W_eval(w, system.q, transfer_mode, system.P, system.b, system.r)
                ims.append(val.imag)
            except Exception:
                ims.append(np.nan)
        
        # Find sign changes in Im(W)
        omega_roots = []
        for j in range(len(ws) - 1):
            if np.isnan(ims[j]) or np.isnan(ims[j+1]):
                continue
            if ims[j] * ims[j+1] < 0.0:
                # Find the root
                try:
                    def root_f(w):
                        return W_eval(w, system.q, transfer_mode, system.P, system.b, system.r).imag
                    
                    sol = root_scalar(root_f, bracket=[ws[j], ws[j+1]], method="bisection")
                    if sol.converged:
                        omega_roots.append(sol.root)
                except Exception:
                    pass
        
        for w0 in omega_roots:
            W0 = W_eval(w0, system.q, transfer_mode, system.P, system.b, system.r)
            if abs(W0.real) < 1e-12:
                continue
            k = -1.0 / W0.real
            
            # Now solve Phi(A) = 0
            # Phi(A) = integral_0^{2pi/w0} [psi(A*cos(w0*t)) - k*A*cos(w0*t)]*cos(w0*t) dt
            def phi_func(A):
                if A <= 0:
                    return 0.0
                def integrand(t):
                    return (system.psi(A * np.cos(w0 * t)) - k * A * np.cos(w0 * t)) * np.cos(w0 * t)
                val, _ = quad(integrand, 0.0, 2.0 * np.pi / w0, limit=100)
                return val
            
            # Scan A for sign changes of Phi(A)
            as_ = np.linspace(amplitude_min, amplitude_max, grid_size_amplitude)
            phi_vals = [phi_func(a) for a in as_]
            
            for i in range(len(as_) - 1):
                if phi_vals[i] * phi_vals[i+1] < 0.0:
                    try:
                        sol_A = root_scalar(phi_func, bracket=[as_[i], as_[i+1]], method="bisection")
                        if sol_A.converged:
                            A0 = sol_A.root
                            if not any(np.allclose([A0, w0], [c[0], c[1]], rtol=1e-2) for c in candidates):
                                candidates.append((float(A0), float(w0), float(k)))
                    except Exception:
                        pass
                        
    else:
        raise ValueError(f"Unknown seed_strategy: {seed_strategy}")
        
    return candidates
