import numpy as np
from typing import Any, Dict

class ChuaPolynomialSystem:
    """Representation of the Chua system with polynomial (cubic) nonlinearity in Lur'e form.
    
    D_t^q X = P X + b * psi(r^T X)
    where psi(sigma) = coeff * (sigma^3 - sigma).
    """
    def __init__(self, alpha: float = 8.4562, beta: float = 12.0732, gamma: float = 0.0052,
                 coeff: float = 1.0, q: float = 1.0, system_id: str = "chua_integer_polynomial"):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.coeff = float(coeff)
        self.q = float(q)
        self.system_id = system_id
        
        # P matrix is the same linear structure as Chua saturation/arctan
        # But wait, does the linear matrix depend on the slope of the nonlinearity at the origin?
        # In Lur'e form for Chua, we separate the linear part P and nonlinear part b * psi(r^T X).
        # Let's match the exact matrix P, b, r used in the other Chua systems.
        # In ChuaSaturationSystem:
        # P = [
        #     [-alpha * (m1 + 1.0), alpha, 0.0],
        #     [1.0, -1.0, 1.0],
        #     [0.0, -beta, -gamma]
        # ]
        # In ChuaArctanSystem:
        # P = [
        #     [-alpha * (1.0 + m), alpha, 0.0],
        #     [1.0, -1.0, 1.0],
        #     [0.0, -beta, -gamma]
        # ]
        # Here, let's treat the linear part P similarly. For polynomial Chua, if we write the system as:
        # dx/dt = alpha * (y - x - g(x))
        # dy/dt = x - y + z
        # dz/dt = -beta * y - gamma * z
        # where g(x) = m1 * x + 0.5 * (m0 - m1) * (|x + 1| - |x - 1|) for saturation.
        # If we have a polynomial nonlinearity g(x) = c1 * x + c2 * x^3,
        # we can put it in Lur'e form with P, b, r:
        # dx/dt = -alpha * (1.0 + c1) * x + alpha * y - alpha * c2 * x^3
        # In this representation:
        # P[0, 0] = -alpha * (1.0 + c1)
        # b = [-alpha, 0, 0]
        # psi(sigma) = c2 * sigma^3
        # Alternatively, if psi(sigma) = coeff * (sigma^3 - sigma):
        # We can define the system in Lur'e form:
        # P = [
        #     [-alpha, alpha, 0.0],
        #     [1.0, -1.0, 1.0],
        #     [0.0, -beta, -gamma]
        # ]
        # b = [-alpha, 0.0, 0.0]
        # r = [1.0, 0.0, 0.0]
        # So dx/dt = P @ x + b * psi(x[0]) = alpha * (y - x) - alpha * coeff * (x^3 - x)
        # This matches the classic cubic Chua's equation! Let's verify:
        # dx/dt = alpha * (y - x - h(x)) where h(x) = coeff * (x^3 - x).
        # Yes! That is extremely elegant and standard. Let's use this definition!
        
        self.P = np.array([
            [-self.alpha, self.alpha, 0.0],
            [1.0, -1.0, 1.0],
            [0.0, -self.beta, -self.gamma]
        ], dtype=float)
        
        self.b = np.array([-self.alpha, 0.0, 0.0], dtype=float)
        self.r = np.array([1.0, 0.0, 0.0], dtype=float)

        self.describing_function_capabilities = {
            "closed_form": True,
            "piecewise_closed_form": False,
            "quadrature": True,
            "nonsmooth": False,
            "breakpoints": None
        }

    def psi(self, sigma: float) -> float:
        """Polynomial (cubic) nonlinearity: psi(sigma) = coeff * (sigma^3 - sigma)"""
        return float(self.coeff * (sigma**3 - sigma))

    def is_nonsmooth(self) -> bool:
        return False

    def has_closed_form_describing_function(self) -> bool:
        return True

    def describing_function_closed_form(self, A: float) -> float:
        """First-harmonic describing function in closed form: coeff * (0.75 * A^2 - 1.0)"""
        if A <= 0.0:
            raise ValueError("Amplitude A must be positive.")
        return float(self.coeff * (0.75 * A**2 - 1.0))

    def describing_function(self, A: float) -> float:
        return self.describing_function_closed_form(A)

    def evaluate_rhs(self, x: np.ndarray) -> np.ndarray:
        """Full non-linear RHS evaluation: P * X + b * psi(r^T * X)"""
        sigma = float(self.r @ x)
        return self.P @ x + self.b * self.psi(sigma)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "coeff": self.coeff,
            "q": self.q,
            "system_id": self.system_id
        }
