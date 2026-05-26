import numpy as np
from typing import Any, Dict, List, Sequence, Optional
from ..integrators.fractional_c import fractional_integrate

def run_fractional_continuation(
    system: Any,
    seed_x0: np.ndarray,
    k_gain: float,
    lambda_values: Sequence[float],
    t_transient: float,
    t_keep: float,
    h: float,
    memory_mode: str = "full",
    memory_window_length: Optional[int] = None,
    div_threshold: float = 120.0,
    integrator: str = "abm",
    use_c_backend: bool = True
) -> List[Dict[str, Any]]:
    """Execute fractional-order parameter continuation for parameter eta (lambda_values).
    
    Uses the unified fractional_integrate engine to propagate memory state natively across stages.
    """
    x_in = np.asarray(seed_x0, dtype=float).copy()
    steps = []
    
    history_t = None
    history_x = None
    
    # Deformed system definition: D^q X = P0 X + eta * b * phi(r^T X)
    p0 = system.P + k_gain * np.outer(system.b, system.r)
    
    nsteps_tr = int(np.ceil(t_transient / h))
    nsteps_kp = int(np.ceil(t_keep / h))
    
    for eta in lambda_values:
        def rhs_deformed(t_val, x_val):
            sigma = float(system.r @ x_val)
            delta = float(system.psi(sigma)) - k_gain * sigma
            return p0 @ x_val + eta * system.b * delta

        # If k_gain = 0 and eta = 1.0, it matches the exact registered system.
        # Otherwise, C registry is bypassed and it runs wrapped Python callback in C.
        sys_to_pass = system if (abs(k_gain) < 1e-12 and abs(eta - 1.0) < 1e-12) else None
        
        # 1. Integrate Transient Stage
        t_tr, x_tr, status_tr, info_tr = fractional_integrate(
            rhs=rhs_deformed,
            x0=x_in,
            q=system.q,
            h=h,
            t_final=t_transient,
            method=integrator,
            memory_mode=memory_mode,
            memory_window_length=memory_window_length,
            history_times=history_t,
            history_states=history_x,
            system=sys_to_pass,
            use_c_backend=use_c_backend,
            divergence_norm=div_threshold,
            return_history=True,
            allow_python_fallback=True
        )
        
        if status_tr != "ok":
            t_tr_new = t_tr[-nsteps_tr:] if len(t_tr) >= nsteps_tr else t_tr
            x_tr_new = x_tr[-nsteps_tr:] if len(x_tr) >= nsteps_tr else x_tr
            steps.append({
                "lambda_value": float(eta),
                "x_in": x_in.copy(),
                "x_out": x_tr[-1].copy() if len(x_tr) > 0 else x_in.copy(),
                "trajectory": np.column_stack((t_tr_new, x_tr_new)),
                "status": status_tr
            })
            break
            
        x_mid = x_tr[-1].copy()
        
        # Prepare history for keep stage: shift so last point is at 0.0
        history_t = t_tr - t_tr[-1]
        history_x = x_tr
        
        # 2. Integrate Kept Stage
        t_kp, x_kp, status_kp, info_kp = fractional_integrate(
            rhs=rhs_deformed,
            x0=x_mid,
            q=system.q,
            h=h,
            t_final=t_keep,
            method=integrator,
            memory_mode=memory_mode,
            memory_window_length=memory_window_length,
            history_times=history_t,
            history_states=history_x,
            system=sys_to_pass,
            use_c_backend=use_c_backend,
            divergence_norm=div_threshold,
            return_history=True,
            allow_python_fallback=True
        )
        
        x_out = x_kp[-1].copy()
        t_kp_kept = t_kp[-nsteps_kp:] if len(t_kp) >= nsteps_kp else t_kp
        x_kp_kept = x_kp[-nsteps_kp:] if len(x_kp) >= nsteps_kp else x_kp
        
        steps.append({
            "lambda_value": float(eta),
            "x_in": x_in.copy(),
            "x_out": x_out,
            "trajectory": np.column_stack((t_kp_kept, x_kp_kept)),
            "status": status_kp
        })
        
        if status_kp != "ok":
            break
            
        # Update history for next parameter stage: shift so last point is at 0.0
        history_t = t_kp - t_kp[-1]
        history_x = x_kp
        
        x_in = x_out
        
    return steps
