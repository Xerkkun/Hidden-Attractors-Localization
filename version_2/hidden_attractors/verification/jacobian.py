import numpy as np
from typing import Any

def compute_jacobian(system: Any, x: np.ndarray) -> np.ndarray:
    """Evaluate the Jacobian matrix at the state x.
    
    J(x) = P + b * psi'(r^T * x) * r^T
    """
    x_val = float(x[0])
    alpha = system.alpha
    
    model = getattr(system, "model", None)
    if model is None:
        if hasattr(system, "m0") or hasattr(system, "m1"):
            model = "nonsmooth"
        else:
            model = "arctan"

    if model == "nonsmooth":
        diff = getattr(system, "m0", -0.1768) - getattr(system, "m1", -1.1468)
        dpsi = diff if abs(x_val) < 1.0 else 0.0
    else:
        a2 = getattr(system, "a2", None)
        if a2 is None:
            n = getattr(system, "n", 0.0)
            m = getattr(system, "m", 0.0)
            a2 = n - m
        rho = getattr(system, "rho", 1.0)
        dpsi = (a2 * rho) / (1.0 + (rho * x_val) ** 2)
        
    J = system.P.copy()
    J[0, 0] += -alpha * dpsi
    return J
