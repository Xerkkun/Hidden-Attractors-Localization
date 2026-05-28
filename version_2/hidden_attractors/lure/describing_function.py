import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Callable

_warned_systems = set()

@dataclass
class DescribingFunctionResult:
    value: float
    method: str
    notes: str
    warning: Optional[str] = None

def N_quadrature(A: float, psi_func) -> float:
    """Evaluate describing function by standard numerical quadrature:
    N(A) = (2 / (pi * A)) * integral_0^pi psi(A * cos(theta)) * cos(theta) dtheta
    """
    if A <= 0.0:
        raise ValueError("Amplitude A must be positive.")
    
    def integrand(theta):
        return psi_func(A * np.cos(theta)) * np.cos(theta)
        
    val, _ = quad(integrand, 0.0, np.pi, limit=100)
    return float((2.0 / (np.pi * A)) * val)

def N_segmented_quadrature(A: float, psi_func, theta_breaks: List[float]) -> float:
    """Evaluate describing function by segmented numerical quadrature:
    Integrate piecewise sections delimited by theta_breaks in [0, pi] to avoid roundoff errors.
    """
    if A <= 0.0:
        raise ValueError("Amplitude A must be positive.")
        
    def integrand(theta):
        return psi_func(A * np.cos(theta)) * np.cos(theta)
        
    val = 0.0
    for i in range(len(theta_breaks) - 1):
        t0 = theta_breaks[i]
        t1 = theta_breaks[i + 1]
        chunk, _ = quad(integrand, t0, t1, limit=100)
        val += chunk
        
    return float((2.0 / (np.pi * A)) * val)

def get_describing_function_capabilities(system: Any) -> Dict[str, Any]:
    """Retrieve capabilities dictionary or define dynamic default maps."""
    if system.lure is None:
        return {
            "closed_form": False,
            "piecewise_closed_form": False,
            "quadrature": False,
            "nonsmooth": False,
            "breakpoints": None
        }
    
    is_nonsmooth = system.parameters.get("model") == "nonsmooth"
    return {
        "closed_form": True,
        "piecewise_closed_form": is_nonsmooth,
        "quadrature": True,
        "nonsmooth": is_nonsmooth,
        "breakpoints": None
    }

def evaluate_describing_function(system: Any, A: float, mode: str = "auto") -> DescribingFunctionResult:
    """General evaluation interface resolving closed-form, piecewise or quadrature modes."""
    caps = get_describing_function_capabilities(system)
    
    # 1. Resolve active mode
    active_mode = mode
    if mode == "auto":
        if caps["closed_form"]:
            active_mode = "closed_form"
        elif caps["piecewise_closed_form"]:
            active_mode = "piecewise_closed_form"
        elif caps["nonsmooth"]:
            active_mode = "segmented_quadrature"
        else:
            active_mode = "quadrature"
            
    # 2. Evaluate according to mode
    if active_mode in ("closed_form", "piecewise_closed_form"):
        val = system.lure.describing_function(A)
        return DescribingFunctionResult(value=float(val), method=active_mode, notes="Closed-form evaluation")
        
    else:
        # Standard quadrature evaluation using the nonlinearity in system.lure
        val = N_quadrature(A, system.lure.nonlinearity)
        return DescribingFunctionResult(value=val, method="quadrature", notes="Standard numerical quadrature")

def solve_amplitude_from_gain(system: Any, k: float, A_min: float, A_max: float, mode: str = "auto") -> float:
    """Solve N(A0) - k = 0 using a robust 1D bisection search."""
    def obj(A):
        res = evaluate_describing_function(system, A, mode=mode)
        return res.value - k
        
    f_min = obj(A_min)
    f_max = obj(A_max)
    
    if f_min * f_max > 0.0:
        # Sign does not change in the brackets. Scan to find a valid crossing subset.
        grid = np.linspace(A_min, A_max, 100)
        vals = [obj(a) for a in grid]
        for i in range(len(grid) - 1):
            if vals[i] * vals[i + 1] < 0.0:
                sol = root_scalar(obj, bracket=[grid[i], grid[i + 1]], method="bisect")
                return float(sol.root)
        # Fallback to nearest if no crossing, or raise
        raise ValueError(f"No sign change in N(A) - k = 0 found in [{A_min}, {A_max}] for k={k}.")
        
    sol = root_scalar(obj, bracket=[A_min, A_max], method="bisect")
    return float(sol.root)


def evaluate_describing_function_batch(
    system: Any,
    A_array: np.ndarray,
    mode: str = "auto",
) -> np.ndarray:
    """Evaluate the describing function N(A) for an entire array of amplitudes."""
    A_array = np.asarray(A_array, dtype=float)
    if np.any(A_array <= 0.0):
        raise ValueError("All amplitudes in A_array must be positive.")

    caps = get_describing_function_capabilities(system)

    # ── Fast path: closed-form evaluations are vectorisable ──────────────
    if caps["closed_form"] or caps["piecewise_closed_form"]:
        try:
            result = system.lure.describing_function(A_array)
            return np.asarray(result, dtype=float)
        except (TypeError, ValueError):
            vf = np.vectorize(system.lure.describing_function)
            return vf(A_array).astype(float)

    # ── Slow path: quadrature ────────────────
    return np.array(
        [evaluate_describing_function(system, float(A), mode=mode).value
         for A in A_array],
        dtype=float,
    )
