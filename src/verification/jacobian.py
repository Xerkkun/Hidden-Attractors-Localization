import numpy as np
from typing import Any

def compute_jacobian(system: Any, x: np.ndarray) -> np.ndarray:
    """Evaluate the Jacobian matrix at the state x.
    
    J(x) = P + b * psi'(r^T * x) * r^T
    """
    x_val = float(x[0])
    alpha = system.alpha
    
    # Calculate psi'(x)
    if hasattr(system, "m1"):
        # Saturation case
        diff = system.m0 - system.m1
        dpsi = diff if abs(x_val) < 1.0 else 0.0
    else:
        # Arctan case
        diff = system.n - system.m
        dpsi = diff / (1.0 + x_val * x_val)
        
    # J is self.P but adding -alpha * dpsi to the (0, 0) entry
    J = system.P.copy()
    J[0, 0] += -alpha * dpsi
    return J
