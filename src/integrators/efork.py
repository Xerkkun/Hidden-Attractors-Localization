import numpy as np
from typing import Any, Callable, Dict, Tuple, Optional, List
from .fractional_c import fractional_integrate

def efork_integrate(
    system: Any,
    x0: np.ndarray,
    q: float,
    h: float,
    t_final: float,
    memory_mode: str = "full",
    memory_window_length: Optional[int] = None,
    k: float = 0.0,
    eps: float = 1.0,
    use_c_backend: bool = True,
    divergence_norm: Optional[float] = None,
    early_stop_config: Optional[dict] = None,
    equilibria: Optional[List[np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Integrate with EFORK using the unified C fractional integrator backend with early stopping."""
    x0_arr = np.asarray(x0, dtype=float)
    dim = x0_arr.size
    
    # 1. Integer order q = 1.0: use Heun/EFORK integer solver in Python with early stopping
    if q == 1.0:
        p0 = system.P + k * np.outer(system.b, system.r)
        
        def rhs(x):
            sigma = float(system.r @ x)
            delta = float(system.psi(sigma)) - k * sigma
            return p0 @ x + eps * system.b * delta
            
        n_steps = int(np.ceil(t_final / h))
        t_arr = np.zeros(n_steps + 1, dtype=float)
        x_arr = np.zeros((n_steps + 1, dim), dtype=float)
        t_arr[0] = 0.0
        x_arr[0] = x0_arr
        
        x = x0_arr.copy()
        status = "ok"
        last_idx = 0
        
        # Parse early stop configs
        esc = early_stop_config if early_stop_config is not None else {}
        es_enabled = esc.get("enabled", True)
        
        div_enabled = esc.get("divergence_enabled", esc.get("divergence", {}).get("enabled", True))
        div_norm = esc.get("divergence_norm", esc.get("divergence", {}).get("norm", 80.0))
        div_consec = esc.get("divergence_consecutive_steps", esc.get("divergence", {}).get("consecutive_steps", 5))
        div_growth = esc.get("divergence_growth_factor", esc.get("divergence", {}).get("growth_factor", 1.25))
        
        eq_enabled = esc.get("equilibrium_enabled", esc.get("equilibrium", {}).get("enabled", True))
        eq_t = esc.get("equilibrium_tol", esc.get("equilibrium", {}).get("tol", 1e-3))
        eq_deriv = esc.get("equilibrium_derivative_tol", esc.get("equilibrium", {}).get("derivative_tol", 1e-4))
        eq_consec = esc.get("equilibrium_consecutive_steps", esc.get("equilibrium", {}).get("consecutive_steps", 200))
        eq_min_t = esc.get("equilibrium_min_time", esc.get("equilibrium", {}).get("min_time", 5.0))
        
        div_consec_count = 0
        growth_consec_count = 0
        prev_norm = -1.0
        eq_consec_counts = [0] * len(equilibria) if equilibria else []
        
        for n in range(n_steps):
            t_curr = n * h
            t_next = (n + 1) * h
            try:
                f_curr = rhs(x)
                x_pred = x + h * f_curr
                f_next = rhs(x_pred)
                x_next = x + 0.5 * h * (f_curr + f_next)
            except Exception as exc:
                status = f"solver_exception:{exc}"
                break
                
            norm = np.linalg.norm(x_next)
            
            if divergence_norm is not None and norm > divergence_norm:
                status = "diverged"
                x_arr[n + 1] = x_next
                t_arr[n + 1] = t_next
                last_idx = n + 1
                break
                
            x = x_next
            x_arr[n + 1] = x
            t_arr[n + 1] = t_next
            last_idx = n + 1
            
            # EARLY STOP CHECKS
            if es_enabled:
                # 1. Divergence checks
                if div_enabled:
                    if norm > div_norm:
                        div_consec_count += 1
                    else:
                        div_consec_count = 0
                    if prev_norm >= 0.0:
                        if norm > div_growth * prev_norm:
                            growth_consec_count += 1
                        else:
                            growth_consec_count = 0
                    prev_norm = norm
                    if div_consec_count >= div_consec or growth_consec_count >= div_consec:
                        status = "diverged_early"
                        break
                else:
                    prev_norm = norm
                    
                # 2. Equilibrium convergence checks
                if eq_enabled and equilibria and t_next >= eq_min_t:
                    converged_eq_idx = -1
                    for k, eq in enumerate(equilibria):
                        diff_norm = np.linalg.norm(x_next - eq)
                        try:
                            deriv_norm = np.linalg.norm(rhs(x_next))
                        except Exception:
                            deriv_norm = 9999.0
                            
                        if diff_norm < eq_t and deriv_norm < eq_deriv:
                            eq_consec_counts[k] += 1
                        else:
                            eq_consec_counts[k] = 0
                            
                        if eq_consec_counts[k] >= eq_consec:
                            converged_eq_idx = k
                            break
                    if converged_eq_idx != -1:
                        status = "converged_equilibrium_early"
                        break
            else:
                prev_norm = norm
                
        return t_arr[:last_idx + 1], x_arr[:last_idx + 1], status
        
    # 2. Fractional order q in (0, 1): use the C or Python general fractional_integrate
    p0 = system.P + k * np.outer(system.b, system.r)
    def rhs_deformed(t_val, x_val):
        sigma = float(system.r @ x_val)
        delta = float(system.psi(sigma)) - k * sigma
        return p0 @ x_val + eps * system.b * delta

    # If k = 0 and eps = 1, it matches the exact registered system.
    # Otherwise, fractional_integrate will wrap the Python callback and compile/run it in C.
    sys_to_pass = system if (abs(k) < 1e-12 and abs(eps - 1.0) < 1e-12) else None

    t_arr, x_arr, status, info = fractional_integrate(
        rhs=rhs_deformed,
        x0=x0_arr,
        q=q,
        h=h,
        t_final=t_final,
        method="efork",
        memory_mode=memory_mode,
        memory_window_length=memory_window_length,
        system=sys_to_pass,
        use_c_backend=use_c_backend,
        divergence_norm=divergence_norm,
        return_history=True,
        allow_python_fallback=True,
        early_stop_config=early_stop_config,
        equilibria=equilibria
    )
    
    return t_arr, x_arr, status
