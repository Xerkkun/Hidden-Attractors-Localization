import numpy as np
from scipy.optimize import root_scalar
from typing import Any, Dict

def solve_equilibria(system: Any) -> Dict[str, np.ndarray]:
    """Find all equilibrium points for the given system.
    
    Returns:
        Dict: {"E0": E0_arr, "E+": Ep_arr, "E-": Em_arr} (Ep and Em only if they exist)
    """
    alpha = float(system.parameters.get("alpha", 8.4562))
    beta = float(system.parameters.get("beta", 12.0))
    gamma = float(system.parameters.get("gamma", 0.0487))
    
    model = system.parameters.get("model")
    if model is None:
        if "m0" in system.parameters or "m1" in system.parameters:
            model = "nonsmooth"
        elif "a1" in system.parameters or "a2" in system.parameters:
            model = "arctan"
        else:
            if "m" in system.parameters or "n" in system.parameters:
                model = "arctan"
            else:
                model = "nonsmooth"
                
    if model == "nonsmooth":
        m_linear = float(system.parameters.get("m1", system.parameters.get("m", -1.1468)))
        diff = float(system.parameters.get("m0", -0.1768)) - m_linear
        is_sat = True
    else:
        m_linear = float(system.parameters.get("a1", system.parameters.get("m", 0.4)))
        is_sat = False
        
    if is_sat:
        K_eq = gamma / (gamma + beta) - (m_linear + 1.0)
        eqs = {"E0": np.array([0.0, 0.0, 0.0], dtype=float)}
        if K_eq != 0.0:
            x_star = diff / K_eq
            if x_star > 1.0:
                y_star = x_star * gamma / (gamma + beta)
                z_star = -x_star * beta / (gamma + beta)
                eqs["E+"] = np.array([x_star, y_star, z_star], dtype=float)
                eqs["E-"] = np.array([-x_star, -y_star, -z_star], dtype=float)
    else:
        a1 = float(system.parameters.get("a1", system.parameters.get("m", 0.4)))
        a2 = system.parameters.get("a2")
        if a2 is None:
            n = float(system.parameters.get("n", 0.0))
            m = float(system.parameters.get("m", 0.0))
            a2 = n - m
        else:
            a2 = float(a2)
        rho = float(system.parameters.get("rho", 1.0))
        
        slope = beta / (beta + gamma)
        def eq_func(x):
            return (a1 + slope) * x + a2 * np.arctan(rho * x)
            
        eqs = {"E0": np.array([0.0, 0.0, 0.0], dtype=float)}
        f_left = eq_func(1e-5)
        f_right = eq_func(100.0)
        if f_left * f_right < 0.0:
            try:
                sol = root_scalar(eq_func, bracket=[1e-5, 100.0], method="bisect")
                if sol.converged and sol.root > 1e-4:
                    x_star = sol.root
                    y_star = x_star * gamma / (gamma + beta)
                    z_star = -x_star * beta / (gamma + beta)
                    eqs["E+"] = np.array([x_star, y_star, z_star], dtype=float)
                    eqs["E-"] = np.array([-x_star, -y_star, -z_star], dtype=float)
            except Exception:
                pass
                
    return eqs
