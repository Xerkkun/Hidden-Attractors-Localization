from typing import Any, Dict
from .chua_saturation import ChuaSaturationSystem
from .chua_arctan import ChuaArctanSystem
from .chua_polynomial import ChuaPolynomialSystem

def get_system_by_id(system_id: str, **kwargs) -> Any:
    """Factory function to retrieve system instances with custom overrides."""
    if system_id == "chua_integer_saturation":
        return ChuaSaturationSystem(system_id=system_id, **kwargs)
    elif system_id == "chua_fractional_saturation":
        # Ensure default q is 0.9998 if not provided
        if "q" not in kwargs:
            kwargs["q"] = 0.9998
        return ChuaSaturationSystem(system_id=system_id, **kwargs)
    elif system_id == "chua_fractional_arctan":
        # Ensure default q is 0.995 if not provided
        if "q" not in kwargs:
            kwargs["q"] = 0.995
        return ChuaArctanSystem(system_id=system_id, **kwargs)
    elif system_id == "chua_integer_polynomial":
        if "q" not in kwargs:
            kwargs["q"] = 1.0
        return ChuaPolynomialSystem(system_id=system_id, **kwargs)
    elif system_id == "chua_fractional_polynomial":
        if "q" not in kwargs:
            kwargs["q"] = 0.99
        return ChuaPolynomialSystem(system_id=system_id, **kwargs)
    else:
        raise ValueError(f"Unknown system_id: {system_id}")

