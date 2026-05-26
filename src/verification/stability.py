import numpy as np
from typing import Any, Dict, Tuple
from .jacobian import compute_jacobian

def classify_equilibrium_stability(system: Any, eq_point: np.ndarray) -> Dict[str, Any]:
    """Classify the local stability of an equilibrium point.
    
    Using standard eigenvalue checks for q=1.0 and Matignon's criterion for q < 1.0.
    """
    q = system.q
    J = compute_jacobian(system, eq_point)
    eigvals = np.linalg.eigvals(J)
    
    if q == 1.0:
        # Integer stability: Re(lambda) < 0
        stable = bool(all(np.real(val) < 0.0 for val in eigvals))
        alpha_min = float("nan")
        instability_measure = float("nan")
    else:
        # Fractional stability (Matignon's criterion): |arg(lambda)| > q * pi / 2
        # np.angle returns angle in [-pi, pi], so we take absolute value
        angles = np.abs(np.angle(eigvals))
        stable = bool(all(angle > q * np.pi / 2.0 for angle in angles))
        alpha_min = float(np.min(angles))
        instability_measure = float(q - 2.0 * alpha_min / np.pi)
        
    return {
        "eigenvalues": eigvals,
        "stable": stable,
        "alpha_min": alpha_min,
        "instability_measure": instability_measure
    }
