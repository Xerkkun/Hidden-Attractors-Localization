import numpy as np
from scipy.integrate import quad
from typing import Any, Dict

class ChuaArctanSystem:
    """Representation of the Chua system with arctan nonlinearity in Lur'e form.
    
    D_t^q X = P X + b * psi(r^T X)
    """
    def __init__(self, alpha: float = 8.4562, beta: float = 12.0732, gamma: float = 0.0052,
                 m: float = 0.4, n: float = -1.1585, q: float = 0.995, system_id: str = "chua_fractional_arctan"):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.m = float(m)
        self.n = float(n)
        self.q = float(q)
        self.system_id = system_id
        
        # P matrix
        self.P = np.array([
            [-self.alpha * (1.0 + self.m), self.alpha, 0.0],
            [1.0, -1.0, 1.0],
            [0.0, -self.beta, -self.gamma]
        ], dtype=float)
        
        # b and r vectors
        self.b = np.array([-self.alpha, 0.0, 0.0], dtype=float)
        self.r = np.array([1.0, 0.0, 0.0], dtype=float)

    def psi(self, sigma: float) -> float:
        """Arctan nonlinearity: psi(sigma) = (n - m) * arctan(sigma)"""
        return float((self.n - self.m) * np.arctan(sigma))

    def evaluate_rhs(self, x: np.ndarray) -> np.ndarray:
        """Full non-linear RHS evaluation: P * X + b * psi(r^T * X)"""
        sigma = float(self.r @ x)
        return self.P @ x + self.b * self.psi(sigma)

    def N_arctan(self, A: float) -> float:
        """First-harmonic describing function for arctan nonlinearity by robust numerical quadrature.
        
        N_arctan(A) = (2 / (pi * A)) * integral_0^pi psi(A * cos(theta)) * cos(theta) dtheta
        """
        if A <= 0.0:
            raise ValueError("Amplitude A must be positive.")
        
        def integrand(theta):
            return self.psi(A * np.cos(theta)) * np.cos(theta)
            
        val, _ = quad(integrand, 0.0, np.pi, limit=100)
        return float((2.0 / (np.pi * A)) * val)

    def describing_function(self, A: float) -> float:
        """Evaluates describes function N(A)"""
        return self.N_arctan(A)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "m": self.m,
            "n": self.n,
            "q": self.q,
            "system_id": self.system_id
        }
