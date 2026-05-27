import numpy as np
from scipy.optimize import minimize, root_scalar
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from .transfer import W_eval
from .describing_function import evaluate_describing_function, solve_amplitude_from_gain

@dataclass
class HarmonicCandidate:
    A0: float
    omega0: float
    k: float
    df_method_used: str
    df_warning: Optional[str]
    residual_gain: float

class CandidateList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detailed_candidates: List[HarmonicCandidate] = []

def find_harmonic_candidates(
    system: Any,
    transfer_mode: str,
    seed_strategy: str = "k_phi",
    df_residual_tol: float = 1e-2,
    omega_min: float = 0.01,
    omega_max: float = 20.0,
    amplitude_min: float = 0.01,
    amplitude_max: float = 20.0,
    grid_size_omega: int = 200,
    grid_size_amplitude: int = 200,
    root_refinement: bool = True,
    q: Optional[float] = None,
    describing_function_mode: str = "auto"
) -> List[Tuple[float, float, float]]:
    """Find all candidate pairs (A0, omega0, k) solving the harmonic condition.
    
    Returns:
        List of tuples: (amplitude, omega, gain) sorted by gain k in ascending order.
        The returned list also contains an attribute 'detailed_candidates' with more metadata.
    """
    if q is None:
        q = getattr(system, "q", 1.0)
        
    candidates = CandidateList()
    
    if seed_strategy == "nyquist_df":
        # Strategy 1: 2D Grid search + optional refinement of ||W*N + 1||
        ws = np.linspace(omega_min, omega_max, grid_size_omega)
        as_ = np.linspace(amplitude_min, amplitude_max, grid_size_amplitude)
        
        best_pts = []
        res = np.zeros((len(as_), len(ws)))
        W_vals = [W_eval(w, q, transfer_mode, system.P, system.b, system.r) for w in ws]
        for i, A in enumerate(as_):
            N_val = evaluate_describing_function(system, A, mode=describing_function_mode).value
            for j, w in enumerate(ws):
                res[i, j] = np.abs(W_vals[j] * N_val + 1.0)
        
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
                        N_val = evaluate_describing_function(system, A, mode=describing_function_mode).value
                        return float(np.abs(W * N_val + 1.0))
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
                    df_res = evaluate_describing_function(system, A_ref, mode=describing_function_mode)
                    k_ref = df_res.value
                    if not any(np.allclose([A_ref, w_ref], [c[0], c[1]], rtol=1e-2) for c in candidates):
                        candidates.append((float(A_ref), float(w_ref), float(k_ref)))
                        candidates.detailed_candidates.append(
                            HarmonicCandidate(
                                A0=float(A_ref),
                                omega0=float(w_ref),
                                k=float(k_ref),
                                df_method_used=df_res.method,
                                df_warning=df_res.warning,
                                residual_gain=abs(k_ref - k_ref) # 0.0 by definition here
                            )
                        )
            else:
                df_res = evaluate_describing_function(system, A_grid, mode=describing_function_mode)
                k_grid = df_res.value
                candidates.append((float(A_grid), float(w_grid), float(k_grid)))
                candidates.detailed_candidates.append(
                    HarmonicCandidate(
                        A0=float(A_grid),
                        omega0=float(w_grid),
                        k=float(k_grid),
                        df_method_used=df_res.method,
                        df_warning=df_res.warning,
                        residual_gain=0.0
                    )
                )
                
    elif seed_strategy in {"k_phi", "imw_gain"}:
        # Strategy 2: 1D phase crossing Im(W) = 0, then solve N(A0) = k
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
            
            try:
                # Solve amplitude from gain
                A0 = solve_amplitude_from_gain(system, k, amplitude_min, amplitude_max, mode=describing_function_mode)
                df_res = evaluate_describing_function(system, A0, mode=describing_function_mode)
                
                if not any(np.allclose([A0, w0], [c[0], c[1]], rtol=1e-2) for c in candidates):
                    candidates.append((float(A0), float(w0), float(k)))
                    candidates.detailed_candidates.append(
                        HarmonicCandidate(
                            A0=float(A0),
                            omega0=float(w0),
                            k=float(k),
                            df_method_used=df_res.method,
                            df_warning=df_res.warning,
                            residual_gain=abs(df_res.value - k)
                        )
                    )
            except Exception:
                pass
                        
    else:
        raise ValueError(f"Unknown seed_strategy: {seed_strategy}")
        
    # Sort candidates and detailed_candidates in tandem by gain k in ascending order
    sorted_pairs = sorted(zip(candidates, candidates.detailed_candidates), key=lambda pair: pair[0][2])
    candidates.clear()
    candidates.detailed_candidates.clear()
    for cand_tup, detailed_cand in sorted_pairs:
        candidates.append(cand_tup)
        candidates.detailed_candidates.append(detailed_cand)
        
    return candidates

