import numpy as np
from typing import Any, Dict

class ChuaSaturationSystem:
    """Representation of the Chua system with saturation nonlinearity in Lur'e form.
    
    D_t^q X = P X + b * psi(r^T X)
    """
    def __init__(self, alpha: float = 8.4562, beta: float = 12.0732, gamma: float = 0.0052,
                 m0: float = -0.1768, m1: float = -1.1468, q: float = 1.0, system_id: str = "chua_integer_saturation"):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.m0 = float(m0)
        self.m1 = float(m1)
        self.q = float(q)
        self.system_id = system_id
        
        # P matrix
        self.P = np.array([
            [-self.alpha * (self.m1 + 1.0), self.alpha, 0.0],
            [1.0, -1.0, 1.0],
            [0.0, -self.beta, -self.gamma]
        ], dtype=float)
        
        # b and r vectors
        self.b = np.array([-self.alpha, 0.0, 0.0], dtype=float)
        self.r = np.array([1.0, 0.0, 0.0], dtype=float)

    def psi(self, sigma: float) -> float:
        """Normalized saturation nonlinearity: psi(sigma) = (m0 - m1) * sat(sigma)"""
        sat_val = np.clip(sigma, -1.0, 1.0)
        return float((self.m0 - self.m1) * sat_val)

    def sat(self, sigma: float) -> float:
        """Standard saturation function"""
        return float(np.clip(sigma, -1.0, 1.0))

    def evaluate_rhs(self, x: np.ndarray) -> np.ndarray:
        """Full non-linear RHS evaluation: P * X + b * psi(r^T * X)"""
        sigma = float(self.r @ x)
        return self.P @ x + self.b * self.psi(sigma)

    def N_sat(self, A: float) -> float:
        """First-harmonic describing function for saturated nonlinearity.
        
        If 0 < A <= 1:
            N_sat(A) = m0 - m1
        If A > 1:
            N_sat(A) = (2*(m0 - m1)/pi) * (asin(1/A) + (1/A)*sqrt(1 - 1/A^2))
        """
        if A <= 0.0:
            raise ValueError("Amplitude A must be positive.")
        diff = self.m0 - self.m1
        if A <= 1.0:
            return float(diff)
        else:
            term = np.arcsin(1.0 / A) + (1.0 / A) * np.sqrt(1.0 - 1.0 / (A * A))
            return float((2.0 * diff / np.pi) * term)

    def describing_function(self, A: float) -> float:
        """Evaluates describes function N(A)"""
        return self.N_sat(A)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "m0": self.m0,
            "m1": self.m1,
            "q": self.q,
            "system_id": self.system_id
        }
