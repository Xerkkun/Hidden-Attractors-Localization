import numpy as np
from scipy.special import gamma
from typing import Any, Callable, Dict, Tuple, Optional, List
from .fractional_c import fractional_integrate

def _python_abm_integrate(
    rhs: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    q: float,
    h: float,
    t_final: float,
    divergence_norm: Optional[float] = 120.0,
    history_times: Optional[np.ndarray] = None,
    history_states: Optional[np.ndarray] = None,
    memory_mode: str = "full",
    memory_window_length: Optional[int] = None,
    early_stop_config: Optional[dict] = None,
    equilibria: Optional[List[np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, str]:
    q = float(q)
    h = float(h)
    t_final = float(t_final)
    n_steps = int(np.ceil(t_final / h))
    
    x0_arr = np.asarray(x0, dtype=float)
    dim = x0_arr.size
    
    if history_times is not None and history_states is not None:
        history_times = np.asarray(history_times, dtype=float)
        history_states = np.asarray(history_states, dtype=float)
        K = len(history_times)
    else:
        K = 1
        history_times = np.array([0.0])
        history_states = x0_arr.reshape(1, dim)
        
    total_steps = K + n_steps
    t_arr = np.zeros(total_steps, dtype=float)
    x_arr = np.zeros((total_steps, dim), dtype=float)
    f_arr = np.zeros((total_steps, dim), dtype=float)
    
    t_arr[:K] = history_times
    x_arr[:K] = history_states
    for j in range(K):
        f_arr[j] = rhs(history_states[j])
        
    for step_idx in range(n_steps):
        t_arr[K + step_idx] = t_arr[K - 1] + (step_idx + 1) * h
        
    powers = np.arange(total_steps + 2, dtype=float)
    pow_q = powers**q
    pow_q1 = powers**(q + 1.0)
    
    hq = h**q
    pred_scale = hq / float(gamma(q + 1.0))
    # Standard gamma function can overflow/underflow, let's keep it safe
    val_gq2 = float(gamma(q + 2.0))
    corr_scale = hq / val_gq2 if abs(val_gq2) > 1e-15 else 0.0
    
    status = "ok"
    last_idx = K - 1
    
    # Parse early stop config
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
    
    for n in range(K - 1, total_steps - 1):
        if memory_mode == "window" and memory_window_length is not None:
            s_idx = max(0, n - int(memory_window_length))
        else:
            s_idx = 0
            
        j_range = np.arange(s_idx, n + 1)
        b_weights = pow_q[n + 1 - j_range] - pow_q[n - j_range]
        
        predictor = x_arr[s_idx] + pred_scale * (b_weights @ f_arr[s_idx: n + 1])
        
        try:
            fp = rhs(predictor)
        except Exception as exc:
            status = f"solver_exception:{exc}"
            break
            
        n_prime = n - s_idx
        a0 = n_prime**(q + 1.0) - (n_prime - q) * (n_prime + 1.0)**q
        if n_prime > 0:
            mid_indices = n - np.arange(s_idx + 1, n + 1)
            a_mid = pow_q1[mid_indices + 2] + pow_q1[mid_indices] - 2.0 * pow_q1[mid_indices + 1]
            a_weights = np.concatenate(([a0], a_mid))
        else:
            a_weights = np.array([a0])
            
        corrected = x_arr[s_idx] + corr_scale * ((a_weights @ f_arr[s_idx: n + 1]) + fp)
        
        norm = np.linalg.norm(corrected)
        
        if divergence_norm is not None and norm > divergence_norm:
            status = "diverged"
            x_arr[n + 1] = corrected
            last_idx = n + 1
            break
            
        if not np.all(np.isfinite(corrected)):
            status = "nonfinite_solution"
            break
            
        x_arr[n + 1] = corrected
        try:
            f_arr[n + 1] = rhs(corrected)
        except Exception as exc:
            status = f"solver_exception:{exc}"
            break
            
        last_idx = n + 1
        
        # EARLY STOPPING CHECKS
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
            if eq_enabled and equilibria and t_arr[n + 1] >= eq_min_t:
                converged_eq_idx = -1
                for k, eq in enumerate(equilibria):
                    diff_norm = np.linalg.norm(corrected - eq)
                    try:
                        deriv_norm = np.linalg.norm(rhs(corrected))
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

def caputo_abm_integrate(
    rhs: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    q: float,
    h: float,
    t_final: float,
    divergence_norm: Optional[float] = 120.0,
    history_times: Optional[np.ndarray] = None,
    history_states: Optional[np.ndarray] = None,
    memory_mode: str = "full",
    memory_window_length: Optional[int] = None,
    system: Optional[Any] = None,
    use_c_backend: bool = True,
    early_stop_config: Optional[dict] = None,
    equilibria: Optional[List[np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Integrate with ABM. Wraps the unified C and Python general solver fractional_integrate."""
    
    # Non-fractional order q=1.0 is handled using Heun's method in Python
    if q == 1.0:
        h_val = float(h)
        n_steps = int(np.ceil(t_final / h_val))
        dim = x0.size
        
        t_arr = np.zeros(n_steps + 1, dtype=float)
        x_arr = np.zeros((n_steps + 1, dim), dtype=float)
        t_arr[0] = 0.0
        x_arr[0] = x0
        
        x = x0.copy()
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
            t_curr = n * h_val
            t_next = (n + 1) * h_val
            try:
                f_curr = rhs(x)
                x_pred = x + h_val * f_curr
                f_next = rhs(x_pred)
                x_next = x + 0.5 * h_val * (f_curr + f_next)
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

    # Wrap the single argument rhs(x) for fractional_integrate which expects rhs(t, x)
    def rhs_t(t_val, x_val):
        return rhs(x_val)
        
    t_arr, x_arr, status, info = fractional_integrate(
        rhs=rhs_t,
        x0=x0,
        q=q,
        h=h,
        t_final=t_final,
        method="abm",
        memory_mode=memory_mode,
        memory_window_length=memory_window_length,
        history_times=history_times,
        history_states=history_states,
        system=system,
        use_c_backend=use_c_backend,
        divergence_norm=divergence_norm if divergence_norm is not None else 120.0,
        return_history=True,
        allow_python_fallback=True,
        early_stop_config=early_stop_config,
        equilibria=equilibria
    )
    
    return t_arr, x_arr, status
