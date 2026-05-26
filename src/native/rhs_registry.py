import ctypes
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
        ("m", ctypes.c_double),
        ("n", ctypes.c_double)
    ]

# Global registry of C RHS functions
_C_RHS_REGISTRY = {}

def register_c_rhs(system_id: str, rhs_getter_name: str, params_builder: Any):
    """Register a pre-compiled C RHS by system_id."""
    _C_RHS_REGISTRY[system_id] = {
        "rhs_getter": rhs_getter_name,
        "builder": params_builder
    }

# Pre-defined builder functions
def build_chua_saturation_params(system: Any) -> ctypes.Structure:
    return ChuaSaturationParamsStruct(
        alpha=float(system.alpha),
        beta=float(system.beta),
        gamma=float(system.gamma),
        m0=float(system.m0),
        m1=float(system.m1)
    )

def build_chua_arctan_params(system: Any) -> ctypes.Structure:
    return ChuaArctanParamsStruct(
        alpha=float(system.alpha),
        beta=float(system.beta),
        gamma=float(system.gamma),
        m=float(system.m),
        n=float(system.n)
    )

# Register default systems
register_c_rhs("chua_integer_saturation", "get_chua_saturation_rhs", build_chua_saturation_params)
register_c_rhs("chua_fractional_saturation", "get_chua_saturation_rhs", build_chua_saturation_params)
register_c_rhs("chua_fractional_arctan", "get_chua_arctan_rhs", build_chua_arctan_params)

def get_c_rhs_and_params(system: Any, lib: Any) -> Tuple[Optional[int], Optional[ctypes.Structure]]:
    """Retrieve the function pointer address and the built ctypes parameters structure for a system."""
    if system is None:
        return None, None
        
    system_id = getattr(system, "system_id", None)
    if system_id not in _C_RHS_REGISTRY:
        return None, None
        
    entry = _C_RHS_REGISTRY[system_id]
    
    # Retrieve the C function pointer address from the loaded library
    getter_func = getattr(lib, entry["rhs_getter"])
    getter_func.argtypes = []
    getter_func.restype = ctypes.c_void_p
    rhs_ptr = getter_func()
    
    # Build parameters structure
    params_struct = entry["builder"](system)
    
    return rhs_ptr, params_struct
