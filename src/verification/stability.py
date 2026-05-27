import numpy as np
from typing import Any, Dict, Tuple
from .jacobian import compute_jacobian

def classify_equilibrium_stability(system: Any, eq_point: np.ndarray, tol: float = 1e-8) -> Dict[str, Any]:
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
        stability_class = "stable" if stable else "unstable"
        matignon_margin = float("nan")
        matignon_threshold = float("nan")
    else:
        # Fractional stability (Matignon's criterion): |arg(lambda)| > q * pi / 2
        # np.angle returns angle in [-pi, pi], so we take absolute value
        angles = np.abs(np.angle(eigvals))
        threshold = q * np.pi / 2.0
        
        # margin_i = |arg(lambda_i)| - q*pi/2
        margins = angles - threshold
        margin_min = float(np.min(margins))
        
        stable = bool(all(angle > threshold for angle in angles))
        
        if margin_min > tol:
            stability_class = "stable"
        elif margin_min < -tol:
            stability_class = "unstable"
        else:
            stability_class = "marginal_or_inconclusive"
            
        alpha_min = float(np.min(angles))
        instability_measure = float(q - 2.0 * alpha_min / np.pi)
        matignon_margin = margin_min
        matignon_threshold = threshold
        
    return {
        "eigenvalues": eigvals,
        "stable": stable,
        "stability_class": stability_class,
        "matignon_margin": matignon_margin,
        "matignon_threshold": matignon_threshold,
        "alpha_min": alpha_min,
        "instability_measure": instability_measure
    }
