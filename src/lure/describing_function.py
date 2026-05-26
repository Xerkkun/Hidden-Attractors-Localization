import numpy as np
from scipy.integrate import quad

def N_quadrature(A: float, psi_func) -> float:
    """Evaluate describes function by numerical quadrature:
    N(A) = (2 / (pi * A)) * integral_0^pi psi(A * cos(theta)) * cos(theta) dtheta
    """
    if A <= 0.0:
        raise ValueError("Amplitude A must be positive.")
    
    def integrand(theta):
        return psi_func(A * np.cos(theta)) * np.cos(theta)
        
    val, _ = quad(integrand, 0.0, np.pi, limit=100)
    return float((2.0 / (np.pi * A)) * val)
