import numpy as np
from typing import Any, Dict, List, Tuple

def build_lure_seed(
    system: Any,
    A0: float,
    omega0: float,
    k: float,
    seed_sign_convention: str = "kuznetsov"
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct the initial state seed X_seed and its symmetric partner -X_seed.
    
    X_seed = [
        a0,
        a0 * (m_linear + 1 + k),
        a0 * (alpha * (m_linear + k) - omega0^2) / alpha
    ]
    
    If seed_sign_convention == "wu", the second component y(0) is flipped.
    """
    a0 = float(A0)
    alpha = float(system.alpha)
    
    # Identify m_linear based on system type
    if hasattr(system, "m1"):
        m_linear = float(system.m1)
    elif hasattr(system, "m"):
        m_linear = float(system.m)
    else:
        raise ValueError("System must have either 'm1' (saturation) or 'm' (arctan) parameters.")
        
    x0 = a0
    y0 = a0 * (m_linear + 1.0 + k)
    z0 = a0 * (alpha * (m_linear + k) - omega0**2) / alpha
    
    if seed_sign_convention == "wu":
        y0 = -y0
        
    X_seed = np.array([x0, y0, z0], dtype=float)
    return X_seed, -X_seed
