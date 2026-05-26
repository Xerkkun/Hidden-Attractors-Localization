import numpy as np
from typing import Any

def W_spectral(lam: complex, P: np.ndarray, b: np.ndarray, r: np.ndarray) -> complex:
    """Spectral form using the repository's internal convention: W_hat(lambda) = r^T * (P - lambda * I)^(-1) * b"""
    dimension = P.shape[0]
    # P - lambda * I
    lhs = P - lam * np.eye(dimension, dtype=complex)
    inv_lhs = np.linalg.inv(lhs)
    return complex(r.T @ inv_lhs @ b)

def W_eval(omega: float, q: float, transfer_mode: str, P: np.ndarray, b: np.ndarray, r: np.ndarray) -> complex:
    """Evaluates the transfer function at frequency omega.
    
    If transfer_mode == "integer":
        lambda = i * omega
    If transfer_mode == "fractional":
        lambda = (i * omega)^q = omega^q * exp(i * q * pi / 2)
    """
    if omega <= 0.0:
        raise ValueError("Frequency omega must be positive and finite.")
    
    if transfer_mode == "integer":
        lam = 1j * omega
    elif transfer_mode == "fractional":
        lam = (omega**q) * np.exp(1j * q * np.pi / 2.0)
    else:
        raise ValueError(f"Unknown transfer_mode: {transfer_mode}")
        
    return W_spectral(lam, P, b, r)
