import numpy as np
from typing import Any

def validate_fractional_order(q: float) -> float:
    """Validate that 0 < q <= 1.0."""
    q_val = float(q)
    if not np.isfinite(q_val) or not (0.0 < q_val <= 1.0):
        raise ValueError(f"Fractional order q must satisfy 0 < q <= 1, got {q}.")
    return q_val

def W_spectral(lam: complex, P: np.ndarray, b: np.ndarray, r: np.ndarray, transfer_convention: str = "standard") -> complex:
    """Spectral form."""
    dimension = P.shape[0]
    if transfer_convention == "standard":
        lhs = lam * np.eye(dimension, dtype=complex) - P
    elif transfer_convention == "opposite_sign":
        lhs = P - lam * np.eye(dimension, dtype=complex)
    else:
        raise ValueError(f"Unknown transfer_convention: {transfer_convention}")
        
    inv_lhs = np.linalg.inv(lhs)
    return complex(r.T @ inv_lhs @ b)

def W_eval(omega: float | np.ndarray, q: float, transfer_mode: str, P: np.ndarray, b: np.ndarray, r: np.ndarray, transfer_convention: str = "standard") -> complex | np.ndarray:
    """Evaluates the transfer function at frequency omega (supports scalar or numpy array)."""
    omega_arr = np.asarray(omega, dtype=float)
    if np.any(omega_arr <= 0.0) or not np.all(np.isfinite(omega_arr)):
        raise ValueError("Frequency omega must be positive and finite.")
        
    if transfer_mode not in {"integer", "fractional"}:
        raise ValueError(f"Invalid transfer_mode: {transfer_mode}. Must be 'integer' or 'fractional'.")
        
    q_val = validate_fractional_order(q)
    
    if transfer_mode == "integer":
        lam = 1j * omega_arr
    else: # fractional
        lam = (omega_arr**q_val) * np.exp(1j * q_val * np.pi / 2.0)
        
    try:
        evals, V = np.linalg.eig(P)
        V_inv = np.linalg.inv(V)
        c_left = r.T @ V
        b_right = (V_inv @ b).flatten()
        residues = c_left * b_right
        
        if lam.ndim == 0:
            denom = lam - evals
            if transfer_convention == "standard":
                val = np.sum(residues / denom)
            else:
                val = np.sum(residues / (-denom))
            return complex(val)
        else:
            denom = lam[..., None] - evals
            if transfer_convention == "standard":
                vals = np.sum(residues / denom, axis=-1)
            else:
                vals = np.sum(residues / (-denom), axis=-1)
            
            if isinstance(omega, (float, int)):
                return complex(vals.item())
            return vals
            
    except Exception:
        # Fallback to standard slow solver if diagonalization fails
        if lam.ndim == 0:
            return W_spectral(lam.item(), P, b, r, transfer_convention=transfer_convention)
        else:
            out = np.zeros(lam.shape, dtype=complex)
            flat = out.flat
            for idx, l_val in enumerate(lam.flat):
                flat[idx] = W_spectral(l_val, P, b, r, transfer_convention=transfer_convention)
            if isinstance(omega, (float, int)):
                return complex(out.item())
            return out


# ---------------------------------------------------------------------------
# Spectral cache — avoid repeated eigendecompositions across phases
# ---------------------------------------------------------------------------

def W_precompute_spectral(
    P: np.ndarray,
    b: np.ndarray,
    r: np.ndarray,
    transfer_convention: str = "standard",
) -> dict:
    """Pre-compute and cache the spectral decomposition of ``P``."""
    try:
        evals, V = np.linalg.eig(P)
        V_inv = np.linalg.inv(V)
        c_left = r.T @ V            # shape (n,)
        b_right = (V_inv @ b).flatten()  # shape (n,)
        residues = c_left * b_right
        if transfer_convention == "opposite_sign":
            convention_sign = -1
        else:
            convention_sign = 1
        return {
            "evals": evals,
            "residues": residues,
            "convention_sign": convention_sign,
            "fallback": False,
            # Keep originals for fallback path
            "_P": P, "_b": b, "_r": r,
            "_transfer_convention": transfer_convention,
        }
    except Exception:
        return {
            "fallback": True,
            "_P": P, "_b": b, "_r": r,
            "_transfer_convention": transfer_convention,
        }


def W_eval_from_cache(
    omega: "float | np.ndarray",
    q: float,
    transfer_mode: str,
    cache: dict,
) -> "complex | np.ndarray":
    """Evaluate ``W(jω)`` using a pre-computed spectral cache."""
    if cache.get("fallback"):
        return W_eval(
            omega, q, transfer_mode,
            cache["_P"], cache["_b"], cache["_r"],
            transfer_convention=cache["_transfer_convention"],
        )

    omega_arr = np.asarray(omega, dtype=float)
    if np.any(omega_arr <= 0.0) or not np.all(np.isfinite(omega_arr)):
        raise ValueError("Frequency omega must be positive and finite.")

    q_val = validate_fractional_order(q)

    if transfer_mode == "integer":
        lam = 1j * omega_arr
    else:
        lam = (omega_arr ** q_val) * np.exp(1j * q_val * np.pi / 2.0)

    evals = cache["evals"]
    residues = cache["residues"]
    sign = cache["convention_sign"]

    if lam.ndim == 0:
        denom = lam - evals
        vals = sign * np.sum(residues / denom)
        return complex(vals)
    else:
        denom = lam[..., None] - evals   # (..., n_eig)
        vals = sign * np.sum(residues / denom, axis=-1)
        if isinstance(omega, (float, int)):
            return complex(vals.item())
        return vals
