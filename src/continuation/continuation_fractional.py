import numpy as np
from typing import Any, Dict, List, Sequence, Optional
from hidden_attractors.native.backends import FullHistoryABMBackend, FractionalChuaBackend
from hidden_attractors.models.chua import ChuaParameters
from ..integrators.abm import caputo_abm_integrate

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
    """Execute fractional-order parameter continuation for parameter eta (lambda_values)."""
    
    is_saturation = "saturation" in getattr(system, "system_id", "")
    
    # 1. If C backend is enabled, use compiled C backends if possible
    if use_c_backend and (is_saturation or integrator == "efork"):
        try:
            if integrator == "abm":
                backend = FullHistoryABMBackend.build()
                params = ChuaParameters(
                    model="piecewise",
                    alpha=system.alpha,
                    beta=system.beta,
                    gamma=system.gamma,
                    m0=system.m0,
                    m1=system.m1,
                    a1=0.4,
                    a2=-1.5585,
                    rho=1.0
                )
                backend.set_params(params)
                
                if memory_mode == "window" and memory_window_length is not None:
                    Lm = float(memory_window_length) * h
                    c_res = backend.continue_truncated_history(
                        seed_x0.tolist(),
                        lambda_values=lambda_values,
                        q=system.q,
                        k=k_gain,
                        h=h,
                        Lm=Lm,
                        t_transient=t_transient,
                        t_keep=t_keep
                    )
                else:
                    c_res = backend.continue_full_history(
                        seed_x0.tolist(),
                        lambda_values=lambda_values,
                        q=system.q,
                        k=k_gain,
                        h=h,
                        t_transient=t_transient,
                        t_keep=t_keep
                    )
                    
            elif integrator == "efork":
                if memory_mode == "window":
                    raise ValueError("EFORK no disponible para esta combinación")
                    
                backend = FractionalChuaBackend.build()
                if hasattr(system, "m1"):
                    params = ChuaParameters(
                        model="piecewise",
                        alpha=system.alpha,
                        beta=system.beta,
                        gamma=system.gamma,
                        m0=system.m0,
                        m1=system.m1,
                        a1=0.4,
                        a2=-1.5585,
                        rho=1.0
                    )
                else:
                    params = ChuaParameters(
                        model="arctan",
                        alpha=system.alpha,
                        beta=system.beta,
                        gamma=system.gamma,
                        m0=0.0,
                        m1=0.0,
                        a1=system.m,
                        a2=system.n - system.m,
                        rho=1.0
                    )
                backend.set_params(params)
                
                Lm = t_transient + t_keep # effectively full memory in practice for EFORK
                c_res = backend.continue_efork3(
                    seed_x0.tolist(),
                    lambda_values=lambda_values,
                    q=system.q,
                    k=k_gain,
                    h=h,
                    Lm=Lm,
                    t_transient=t_transient,
                    t_keep=t_keep
                )
            else:
                raise ValueError(f"Unknown integrator: {integrator}")
                
            # Convert C result dictionary to standard list of step dicts
            steps = []
            lambdas = c_res["lambda"]
            x_in_arr = c_res["x_in"]
            x_out_arr = c_res["x_out"]
            trajectories = c_res["trajectories"]
            
            for idx, lam in enumerate(lambdas):
                steps.append({
                    "lambda_value": float(lam),
                    "x_in": x_in_arr[idx],
                    "x_out": x_out_arr[idx],
                    "trajectory": trajectories[idx],
                    "status": "ok"
                })
            return steps
        except Exception as exc:
            if integrator == "efork" and memory_mode == "window":
                raise exc
            # Fallback to Python if C fails
            pass
            
    # EFORK is only implemented in C for fractional order
    if integrator == "efork":
        if memory_mode == "window":
            raise ValueError("EFORK no disponible para esta combinación")
        raise RuntimeError("EFORK fractional integration requires compiled C backend.")
        
    # 2. Python fractional continuation solver with history propagation (e.g. for arctan)
    x_in = np.asarray(seed_x0, dtype=float).copy()
    steps = []
    
    # Deformed system definition: D^q X = P0 X + eta * b * phi(r^T X)
    p0 = system.P + k_gain * np.outer(system.b, system.r)
    
    history_t = None
    history_x = None
    
    for eta in lambda_values:
        def rhs(x):
            sigma = float(system.r @ x)
            delta = float(system.psi(sigma)) - k_gain * sigma
            return p0 @ x + eta * system.b * delta
            
        # Integrate transient
        t_tr, x_tr, status_tr = caputo_abm_integrate(
            rhs, x_in, q=system.q, h=h, t_final=t_transient, divergence_norm=div_threshold,
            history_times=history_t, history_states=history_x,
            memory_mode=memory_mode, memory_window_length=memory_window_length,
            system=system, use_c_backend=False
        )
        
        if status_tr != "ok":
            steps.append({
                "lambda_value": float(eta),
                "x_in": x_in.copy(),
                "x_out": x_tr[-1].copy(),
                "trajectory": np.column_stack((t_tr, x_tr)),
                "status": status_tr
            })
            break
            
        x_mid = x_tr[-1].copy()
        
        # Propagate history by concatenating previous history with the new transient
        if history_t is None:
            comb_t = t_tr
            comb_x = x_tr
        else:
            shifted_t_tr = t_tr + history_t[-1] + h
            comb_t = np.concatenate((history_t, shifted_t_tr))
            comb_x = np.concatenate((history_x, x_tr))
            
        # Integrate kept
        t_kp, x_kp, status_kp = caputo_abm_integrate(
            rhs, x_mid, q=system.q, h=h, t_final=t_keep, divergence_norm=div_threshold,
            history_times=comb_t - comb_t[-1], history_states=comb_x,
            memory_mode=memory_mode, memory_window_length=memory_window_length,
            system=system, use_c_backend=False
        )
        
        x_out = x_kp[-1].copy()
        steps.append({
            "lambda_value": float(eta),
            "x_in": x_in.copy(),
            "x_out": x_out,
            "trajectory": np.column_stack((t_kp, x_kp)),
            "status": status_kp
        })
        
        if status_kp != "ok":
            break
            
        # Update history for next eta stage: concatenate transient and kept
        shifted_t_kp = t_kp + comb_t[-1] + h
        history_t = np.concatenate((comb_t, shifted_t_kp))
        history_x = np.concatenate((comb_x, x_kp))
        
        # Shift history_t so the last point is exactly at 0.0
        history_t = history_t - history_t[-1]
        
        # Truncate sliding window if window memory is enabled
        if memory_mode == "window" and memory_window_length is not None:
            history_t = history_t[-memory_window_length:]
            history_x = history_x[-memory_window_length:]
            
        x_in = x_out
        
    return steps
