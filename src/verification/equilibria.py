import numpy as np
from scipy.optimize import root_scalar
from typing import Any, Dict

def solve_equilibria(system: Any) -> Dict[str, np.ndarray]:
    """Find all equilibrium points for the given system.
    
    Returns:
        Dict: {"E0": E0_arr, "E+": Ep_arr, "E-": Em_arr} (Ep and Em only if they exist)
    """
    alpha = system.alpha
    beta = system.beta
    gamma = system.gamma
    
    # K_eq = gamma / (gamma + beta) - (m_linear + 1)
    if hasattr(system, "m1"):
        m_linear = system.m1
        is_sat = True
        diff = system.m0 - system.m1
    else:
        m_linear = system.m
        is_sat = False
        diff = system.n - system.m
        
    K_eq = gamma / (gamma + beta) - (m_linear + 1.0)
    
    # E0 is always (0, 0, 0)
    eqs = {"E0": np.array([0.0, 0.0, 0.0], dtype=float)}
    
    if is_sat:
        # Saturation case
        if K_eq != 0.0:
            x_star = diff / K_eq
            if x_star > 1.0:
                y_star = x_star * gamma / (gamma + beta)
                z_star = -x_star * beta / (gamma + beta)
                eqs["E+"] = np.array([x_star, y_star, z_star], dtype=float)
                eqs["E-"] = np.array([-x_star, -y_star, -z_star], dtype=float)
    else:
        # Arctan case: psi(x) - K_eq * x = 0  => (n - m)*arctan(x) - K_eq * x = 0
        def eq_func(x):
            return (system.n - system.m) * np.arctan(x) - K_eq * x
            
        # We search for a positive root using bisection in [1e-5, 100.0]
        # We check if there's a sign change
        f_left = eq_func(1e-5)
        f_right = eq_func(100.0)
        if f_left * f_right < 0.0:
            try:
                sol = root_scalar(eq_func, bracket=[1e-5, 100.0], method="bisection")
                if sol.converged and sol.root > 1e-4:
                    x_star = sol.root
                    y_star = x_star * gamma / (gamma + beta)
                    z_star = -x_star * beta / (gamma + beta)
                    eqs["E+"] = np.array([x_star, y_star, z_star], dtype=float)
                    eqs["E-"] = np.array([-x_star, -y_star, -z_star], dtype=float)
            except Exception:
                pass
                
    return eqs
