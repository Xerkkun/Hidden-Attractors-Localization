import numpy as np
from typing import Callable, Tuple, Optional, Any
from .fractional_c import fractional_integrate
from .abm import caputo_abm_integrate
from .efork import efork_integrate

def integrate_general(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    x0: np.ndarray,
    q: float,
    h: float,
    t_final: float,
    integrator: str = "efork",
    memory_mode: str = "full",
    memory_window_length: Optional[int] = None,
    divergence_norm: Optional[float] = 120.0,
    system: Optional[Any] = None,
    use_c_backend: bool = True
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Unified general solver facade for integrating any system (fractional or integer).
    Supports ABM and EFORK schemes under full or windowed memory.
    """
    x0_arr = np.asarray(x0, dtype=float)
    dim = x0_arr.size
    
    # Normalize rhs to rhs_t(t, x)
    def rhs_t(t: float, x: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(rhs(t, x), dtype=float)
        except TypeError:
            return np.asarray(rhs(x), dtype=float)
            
    # 1. Non-fractional order q = 1.0: use general Heun's method
    if q == 1.0:
        n_steps = int(np.ceil(t_final / h))
        t_arr = np.zeros(n_steps + 1, dtype=float)
        x_arr = np.zeros((n_steps + 1, dim), dtype=float)
        t_arr[0] = 0.0
        x_arr[0] = x0_arr
        
        x = x0_arr.copy()
        status = "ok"
        last_idx = 0
        
        for n in range(n_steps):
            t_curr = n * h
            t_next = (n + 1) * h
            try:
                f_curr = rhs_t(t_curr, x)
                x_pred = x + h * f_curr
                f_next = rhs_t(t_next, x_pred)
                x_next = x + 0.5 * h * (f_curr + f_next)
            except Exception as exc:
                status = f"solver_exception:{exc}"
                break
                
            if divergence_norm is not None and np.linalg.norm(x_next) > divergence_norm:
                status = "diverged"
                x_arr[n + 1] = x_next
                t_arr[n + 1] = t_next
                last_idx = n + 1
                break
                
            x = x_next
            x_arr[n + 1] = x
            t_arr[n + 1] = t_next
            last_idx = n + 1
            
        return t_arr[:last_idx + 1], x_arr[:last_idx + 1], status
            
    # 2. Fractional order q in (0, 1)
    t_arr, x_arr, status, info = fractional_integrate(
        rhs=rhs_t,
        x0=x0_arr,
        q=q,
        h=h,
        t_final=t_final,
        method=integrator,
        memory_mode=memory_mode,
        memory_window_length=memory_window_length,
        system=system,
        use_c_backend=use_c_backend,
        divergence_norm=divergence_norm if divergence_norm is not None else 120.0,
        return_history=True,
        allow_python_fallback=True
    )
    
    return t_arr, x_arr, status
