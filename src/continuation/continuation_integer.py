import numpy as np
from typing import Any, Dict, List, Sequence
from ..integrators.abm import caputo_abm_integrate
from ..integrators.efork import efork_integrate

def run_integer_continuation(
    system: Any,
    seed_x0: np.ndarray,
    k_gain: float,
    lambda_values: Sequence[float],
    t_transient: float,
    t_keep: float,
    h: float,
    div_threshold: float = 120.0,
    integrator: str = "abm"
) -> List[Dict[str, Any]]:
    """Execute integer-order parameter continuation for parameter eta (lambda_values)."""
    
    x_in = np.asarray(seed_x0, dtype=float).copy()
    steps = []
    
    # Deformed system definition
    # D_t^1 X = P0 X + eta * b * phi(r^T X)
    # where P0 = P + k * b * r^T, and phi(sigma) = psi(sigma) - k * sigma
    p0 = system.P + k_gain * np.outer(system.b, system.r)
    
    for eta in lambda_values:
        def rhs(x):
            sigma = float(system.r @ x)
            delta = float(system.psi(sigma)) - k_gain * sigma
            return p0 @ x + eta * system.b * delta
            
        # 1. Transient integration
        if integrator == "abm":
            t_tr, x_tr, status_tr = caputo_abm_integrate(
                rhs, x_in, q=1.0, h=h, t_final=t_transient, divergence_norm=div_threshold, system=system
            )
        else: # efork
            t_tr, x_tr, status_tr = efork_integrate(
                system, x_in, q=1.0, h=h, t_final=t_transient, k=k_gain, eps=eta
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
        
        # 2. Kept integration
        if integrator == "abm":
            t_kp, x_kp, status_kp = caputo_abm_integrate(
                rhs, x_mid, q=1.0, h=h, t_final=t_keep, divergence_norm=div_threshold, system=system
            )
        else: # efork
            t_kp, x_kp, status_kp = efork_integrate(
                system, x_mid, q=1.0, h=h, t_final=t_keep, k=k_gain, eps=eta
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
            
        x_in = x_out
        
    return steps
