import numpy as np
from typing import Any, Callable, Dict, Tuple, Optional
from hidden_attractors.native.backends import FractionalChuaBackend
from hidden_attractors.models.chua import ChuaParameters
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
    use_c_backend: bool = True
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Integrate with EFORK. Uses the compiled C backend by default for fractional-order.
    
    Raises:
        ValueError if EFORK is selected with windowed memory.
    """
    if memory_mode == "window":
        raise ValueError("EFORK no disponible para esta combinación")
        
    x0_arr = np.asarray(x0, dtype=float)
    dim = x0_arr.size
    
    if q == 1.0:
        # Integer order: use efork_q1_integrate in Python
        # RHS represents: P * X + b * eta * phi(r^T * X)
        # where P0 = P + k * b * r^T, and phi(sigma) = psi(sigma) - k * sigma
        # For eps = 1.0 and k = 0.0, this is the original system evaluate_rhs.
        # Let's construct the general deformed RHS:
        p0 = system.P + k * np.outer(system.b, system.r)
        
        def rhs(x):
            sigma = float(system.r @ x)
            delta = float(system.psi(sigma)) - k * sigma
            return p0 @ x + eps * system.b * delta
            
        traj, status = efork_q1_integrate(rhs, x0_arr, t_final=t_final, h=h)
        return traj[:, 0], traj[:, 1:], status
        
    # Fractional order q in (0, 1): use C FractionalChuaBackend
    if use_c_backend:
        try:
            backend = FractionalChuaBackend.build()
            
            # Map parameters to ChuaParameters
            if hasattr(system, "m1"):
                # Saturated case (piecewise)
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
                # Arctan case
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
            
            # Lm represents memory length in time (not points). Lm = window or infinite.
            # EFORK C backend takes Lm as time duration.
            # For memory_mode="full", we pass t_final so there is no truncation in practice.
            Lm = t_final
            
            traj_c = backend.integrate_efork3(
                x0_arr.tolist(),
                q=q,
                h=h,
                Lm=Lm,
                t_final=t_final,
                k=k,
                eps=eps
            )
            return traj_c[:, 0], traj_c[:, 1:], "ok"
        except Exception as exc:
            # If C compilation fails, EFORK is only available in C for fractional order.
            # We raise a RuntimeError.
            raise RuntimeError(f"Failed to execute EFORK C backend: {exc}")
            
    else:
        # EFORK Python reference only exists for non-smooth Chua with full history.
        raise ValueError("EFORK fractional integration requires compiled C backend.")
