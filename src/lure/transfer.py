import numpy as np
from typing import Any

def validate_fractional_order(q: float) -> float:
    """Validate that 0 < q <= 1.0."""
    q_val = float(q)
    if not np.isfinite(q_val) or not (0.0 < q_val <= 1.0):
        raise ValueError(f"Fractional order q must satisfy 0 < q <= 1, got {q}.")
    return q_val

def W_spectral(lam: complex, P: np.ndarray, b: np.ndarray, r: np.ndarray, transfer_convention: str = "standard") -> complex:
    """Spectral form.
    
    If transfer_convention == "standard":
        W_q(omega) = r^T * ((lam * I - P)^(-1)) * b
    If transfer_convention == "opposite_sign":
        W_q(omega) = r^T * ((P - lam * I)^(-1)) * b
    """
    dimension = P.shape[0]
    if transfer_convention == "standard":
        lhs = lam * np.eye(dimension, dtype=complex) - P
    elif transfer_convention == "opposite_sign":
        lhs = P - lam * np.eye(dimension, dtype=complex)
    else:
        raise ValueError(f"Unknown transfer_convention: {transfer_convention}")
        
    inv_lhs = np.linalg.inv(lhs)
    return complex(r.T @ inv_lhs @ b)

def W_eval(omega: float, q: float, transfer_mode: str, P: np.ndarray, b: np.ndarray, r: np.ndarray, transfer_convention: str = "standard") -> complex:
    """Evaluates the transfer function at frequency omega.
    
    If transfer_mode == "integer":
        lambda = i * omega
    If transfer_mode == "fractional":
        lambda = (i * omega)^q = omega^q * exp(i * q * pi / 2)
    """
    if omega <= 0.0 or not np.isfinite(omega):
        raise ValueError("Frequency omega must be positive and finite.")
        
    if transfer_mode not in {"integer", "fractional"}:
        raise ValueError(f"Invalid transfer_mode: {transfer_mode}. Must be 'integer' or 'fractional'.")
        
    q_val = validate_fractional_order(q)
    
    if transfer_mode == "integer":
        lam = 1j * omega
    else: # fractional
        lam = (omega**q_val) * np.exp(1j * q_val * np.pi / 2.0)
        
    return W_spectral(lam, P, b, r, transfer_convention=transfer_convention)
