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
    caps = getattr(system, "describing_function_capabilities", {})
    
    has_closed = caps.get("closed_form")
    if has_closed is None:
        has_closed = getattr(system, "has_closed_form_describing_function", lambda: hasattr(system, "describing_function_closed_form"))()
        
    has_piecewise = caps.get("piecewise_closed_form")
    if has_piecewise is None:
        has_piecewise = hasattr(system, "describing_function_piecewise_closed_form") or (
            hasattr(system, "describing_function_closed_form") and getattr(system, "is_nonsmooth", lambda: False)()
        )
        
    has_quad = caps.get("quadrature")
    if has_quad is None:
        has_quad = getattr(system, "has_quadrature_describing_function", lambda: hasattr(system, "psi"))()
        
    is_nonsmooth = caps.get("nonsmooth")
    if is_nonsmooth is None:
        is_nonsmooth = getattr(system, "is_nonsmooth", lambda: False)()
        
    breakpoints = caps.get("breakpoints")
    if breakpoints is None:
        breakpoints = getattr(system, "describing_function_breakpoints", None)
        
    return {
        "closed_form": bool(has_closed),
        "piecewise_closed_form": bool(has_piecewise),
        "quadrature": bool(has_quad),
        "nonsmooth": bool(is_nonsmooth),
        "breakpoints": breakpoints
    }

def evaluate_describing_function(system: Any, A: float, mode: str = "auto") -> DescribingFunctionResult:
    """General evaluation interface resolving closed-form, piecewise or quadrature modes."""
    caps = get_describing_function_capabilities(system)
    system_id = getattr(system, "system_id", "unknown_system")
    
    # 1. Resolve active mode
    active_mode = mode
    if mode == "auto":
        if caps["closed_form"]:
            if caps["nonsmooth"] and caps["piecewise_closed_form"]:
                active_mode = "piecewise_closed_form"
            else:
                active_mode = "closed_form"
        elif caps["piecewise_closed_form"]:
            active_mode = "piecewise_closed_form"
        elif caps["nonsmooth"]:
            active_mode = "segmented_quadrature"
        else:
            active_mode = "quadrature"
            
    # 2. Evaluate according to mode
    if active_mode == "closed_form":
        if hasattr(system, "describing_function_closed_form"):
            val = system.describing_function_closed_form(A)
        else:
            val = system.describing_function(A)
        return DescribingFunctionResult(value=float(val), method=active_mode, notes="Closed-form evaluation")
        
    elif active_mode == "piecewise_closed_form":
        if system_id not in _warned_systems and caps["nonsmooth"]:
            print(f"[{system_id}] DF: no linealidad no suave detectada; se usa función descriptiva cerrada por tramos para evitar cuadratura inestable.")
            _warned_systems.add(system_id)
            
        if hasattr(system, "describing_function_closed_form"):
            val = system.describing_function_closed_form(A)
        elif hasattr(system, "describing_function_piecewise_closed_form"):
            val = system.describing_function_piecewise_closed_form(A)
        else:
            val = system.describing_function(A)
        return DescribingFunctionResult(value=float(val), method=active_mode, notes="Piecewise closed-form evaluation")
        
    elif active_mode == "segmented_quadrature" or (active_mode == "quadrature" and caps["nonsmooth"]):
        if caps["nonsmooth"] and not caps["closed_form"] and not caps["piecewise_closed_form"]:
            if system_id not in _warned_systems:
                print(f"[{system_id}] DF WARNING: no linealidad no suave sin forma cerrada registrada; se usa cuadratura seccionada en puntos de quiebre.")
                _warned_systems.add(system_id)
                
        # Resolve breakpoints
        bp_func = caps["breakpoints"]
        if bp_func is not None:
            if callable(bp_func):
                try:
                    theta_breaks = bp_func(A)
                except TypeError:
                    theta_breaks = bp_func(system, A)
            else:
                theta_breaks = bp_func
        else:
            theta_breaks = [0.0, np.pi]
            
        val = N_segmented_quadrature(A, system.psi, theta_breaks)
        return DescribingFunctionResult(value=val, method="segmented_quadrature", notes="Segmented quadrature")
        
    else:
        # Standard quadrature evaluation
        val = N_quadrature(A, system.psi)
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
    """Evaluate the describing function N(A) for an entire array of amplitudes.

    For systems that have a closed-form or piecewise closed-form describing
    function this is fully vectorised (single NumPy call).  For systems that
    require numerical quadrature it falls back to a sequential loop — scipy's
    ``quad`` is inherently scalar and cannot be batched further.

    Parameters
    ----------
    system : object
        Lur'e system instance (same requirements as ``evaluate_describing_function``).
    A_array : array_like, shape (n,)
        Amplitudes at which to evaluate N.  All values must be positive.
    mode : str
        Evaluation mode forwarded to ``evaluate_describing_function``.

    Returns
    -------
    np.ndarray, shape (n,)
        N(A) values as a float array.
    """
    A_array = np.asarray(A_array, dtype=float)
    if np.any(A_array <= 0.0):
        raise ValueError("All amplitudes in A_array must be positive.")

    caps = get_describing_function_capabilities(system)

    # ── Fast path: closed-form evaluations are vectorisable ──────────────
    if caps["closed_form"] or caps["piecewise_closed_form"]:
        if hasattr(system, "describing_function_closed_form"):
            try:
                # Try direct call with array — many implementations support it
                result = system.describing_function_closed_form(A_array)
                return np.asarray(result, dtype=float)
            except (TypeError, ValueError):
                # Scalar-only implementation — use vectorize
                vf = np.vectorize(system.describing_function_closed_form)
                return vf(A_array).astype(float)
        elif hasattr(system, "describing_function_piecewise_closed_form"):
            try:
                result = system.describing_function_piecewise_closed_form(A_array)
                return np.asarray(result, dtype=float)
            except (TypeError, ValueError):
                vf = np.vectorize(system.describing_function_piecewise_closed_form)
                return vf(A_array).astype(float)
        elif hasattr(system, "describing_function"):
            try:
                result = system.describing_function(A_array)
                return np.asarray(result, dtype=float)
            except (TypeError, ValueError):
                vf = np.vectorize(system.describing_function)
                return vf(A_array).astype(float)

    # ── Slow path: quadrature (scipy.quad is scalar-only) ────────────────
    return np.array(
        [evaluate_describing_function(system, float(A), mode=mode).value
         for A in A_array],
        dtype=float,
    )
