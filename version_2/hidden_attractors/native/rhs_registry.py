import ctypes
import math
from typing import Any, Tuple, Optional

# Define the ctypes Structures mapping the C declarations in fractional_integrators.h
class ChuaSaturationParamsStruct(ctypes.Structure):
    _fields_ = [
        ("alpha", ctypes.c_double),
        ("beta", ctypes.c_double),
        ("gamma", ctypes.c_double),
        ("m0", ctypes.c_double),
        ("m1", ctypes.c_double)
    ]

class ChuaArctanParamsStruct(ctypes.Structure):
    _fields_ = [
        ("alpha", ctypes.c_double),
        ("beta", ctypes.c_double),
        ("gamma", ctypes.c_double),
        ("a1", ctypes.c_double),
        ("a2", ctypes.c_double),
        ("rho", ctypes.c_double),
    ]

# Global registry of C RHS functions
_C_RHS_REGISTRY = {}
_EXPLICIT_SYSTEM_ADAPTERS = {
    "chua-nonsmooth": "chua_fractional_saturation",
    "chua-arctan": "chua_fractional_arctan",
    "fractional-chua-arctan-wu2023": "chua_fractional_arctan",
}

def register_c_rhs(system_id: str, rhs_getter_name: str, params_builder: Any):
    """Register a pre-compiled C RHS by system_id."""
    _C_RHS_REGISTRY[system_id] = {
        "rhs_getter": rhs_getter_name,
        "builder": params_builder
    }

# Helper to retrieve parameter from system (attribute or parameters dictionary)
def _get_param(system: Any, key: str, default: float = 0.0) -> float:
    # 1. Direct attribute
    val = getattr(system, key, None)
    if val is not None:
        return float(val)
    # 2. Dictionary in parameters
    params = getattr(system, "parameters", None)
    if isinstance(params, dict) or (params is not None and hasattr(params, "get")):
        val = params.get(key)
        if val is not None:
            return float(val)
    return default

# Pre-defined builder functions
def build_chua_saturation_params(system: Any) -> ctypes.Structure:
    return ChuaSaturationParamsStruct(
        alpha=_get_param(system, "alpha", 8.4562),
        beta=_get_param(system, "beta", 12.0732),
        gamma=_get_param(system, "gamma", 0.0052),
        m0=_get_param(system, "m0", -0.1768),
        m1=_get_param(system, "m1", -1.1468)
    )

def build_chua_arctan_params(system: Any) -> ctypes.Structure:
    return ChuaArctanParamsStruct(
        alpha=_get_param(system, "alpha", 8.4562),
        beta=_get_param(system, "beta", 12.0732),
        gamma=_get_param(system, "gamma", 0.0052),
        a1=_get_param(system, "a1", _get_param(system, "m", 0.4)),
        a2=_get_param(
            system,
            "a2",
            _get_param(system, "n", -1.1585) - _get_param(system, "m", 0.4),
        ),
        rho=_get_param(system, "rho", 1.0),
    )

# Register default systems
register_c_rhs("chua_integer_saturation", "chua_saturation_rhs_c", build_chua_saturation_params)
register_c_rhs("chua_fractional_saturation", "chua_saturation_rhs_c", build_chua_saturation_params)
register_c_rhs("chua_fractional_arctan", "chua_arctan_rhs_c", build_chua_arctan_params)

def get_c_rhs_and_params(system: Any, lib: Any) -> Tuple[Optional[int], Optional[ctypes.Structure]]:
    """Retrieve the function pointer address and the built ctypes parameters structure for a system."""
    if system is None:
        return None, None

    system_id = getattr(system, "system_id", None)
    if system_id is None:
        system_id = _EXPLICIT_SYSTEM_ADAPTERS.get(getattr(system, "name", None))
    if system_id not in _C_RHS_REGISTRY:
        return None, None

    entry = _C_RHS_REGISTRY[system_id]
    
    # Retrieve the exported typed function directly.  The address conversion is
    # performed by ctypes at the platform ABI boundary, avoiding the non-portable
    # ISO-C conversion from a function pointer to ``void *``.
    rhs_func = getattr(lib, entry["rhs_getter"])
    rhs_func.argtypes = [
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    rhs_func.restype = None
    rhs_ptr = ctypes.cast(rhs_func, ctypes.c_void_p).value
    if rhs_ptr is None:
        raise RuntimeError(f"Native RHS symbol {entry['rhs_getter']!r} has no address.")
    
    # Build parameters structure
    params_struct = entry["builder"](system)
    for field_name, field_type in params_struct._fields_:
        if field_type not in (ctypes.c_double, ctypes.c_float):
            continue
        field_value = float(getattr(params_struct, field_name))
        if not math.isfinite(field_value):
            raise ValueError(
                f"Native RHS parameter {field_name!r} for {system_id!r} "
                "must be finite."
            )
    
    return rhs_ptr, params_struct
