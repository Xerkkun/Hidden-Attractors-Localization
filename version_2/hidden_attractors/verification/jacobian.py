import numpy as np
from typing import Any

def compute_jacobian(system: Any, x: np.ndarray) -> np.ndarray:
    """Evaluate the Jacobian matrix at the state x.
    
    J(x) = P + b * psi'(r^T * x) * r^T
    """
    x_val = float(x[0])
    params = system.parameters
    alpha = params.get("alpha", 8.4562)
    
    model = params.get("model")
    if model is None:
        if "m0" in params or "m1" in params:
            model = "nonsmooth"
        else:
            model = "arctan"

    if model == "nonsmooth":
        diff = params.get("m0", -0.1768) - params.get("m1", -1.1468)
        dpsi = diff if abs(x_val) < 1.0 else 0.0
    else:
        a2 = params.get("a2")
        if a2 is None:
            n = params.get("n", 0.0)
            m = params.get("m", 0.0)
            a2 = n - m
        rho = params.get("rho", 1.0)
        dpsi = (a2 * rho) / (1.0 + (rho * x_val) ** 2)
        
    J = system.lure.matrix.copy()
    J[0, 0] += -alpha * dpsi
    return J
