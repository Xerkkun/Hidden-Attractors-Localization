import numpy as np
from typing import Any, Callable, Dict, Tuple, Optional
from .fractional_c import fractional_integrate
from hidden_attractors.solvers.integer import efork_q1_integrate

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
    divergence_norm: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Integrate with EFORK using the unified C fractional integrator backend."""
    x0_arr = np.asarray(x0, dtype=float)
    dim = x0_arr.size
    
    # 1. Integer order q = 1.0: use Heun/EFORK integer solver in Python
    if q == 1.0:
        p0 = system.P + k * np.outer(system.b, system.r)
        
        def rhs(x):
            sigma = float(system.r @ x)
            delta = float(system.psi(sigma)) - k * sigma
            return p0 @ x + eps * system.b * delta
            
        traj, status = efork_q1_integrate(rhs, x0_arr, t_final=t_final, h=h, div_threshold=divergence_norm)
        return traj[:, 0], traj[:, 1:], status
        
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
        allow_python_fallback=True
    )
    
    return t_arr, x_arr, status
