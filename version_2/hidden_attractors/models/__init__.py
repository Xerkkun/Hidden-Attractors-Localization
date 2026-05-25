"""Dynamical-system definitions.

Stability: stable
    Chua parameters, vector fields, and equilibria.  Signatures and return
    types are fixed; breaking changes require a version bump and a deprecation
    cycle.
"""

from .chua import (
    ChuaParameters,
    chua_arctan_parameters,
    chua_arctan_wu2023_parameters,
    chua_nonsmooth_parameters,
    chua_parameters,
    equilibria_arctan,
    equilibria_nonsmooth,
    jacobian_arctan,
    jacobian_nonsmooth,
    nonlinearity_nonsmooth,
    rhs_nonsmooth,
    # Compatibility aliases for recorded runs created with the old label.
    chua_piecewise_parameters,
    equilibria_piecewise,
    jacobian_piecewise,
    nonlinearity_arctan,
    nonlinearity_chua,
    nonlinearity_piecewise,
    normalize_chua_model,
    rhs_arctan,
    rhs_chua,
    rhs_piecewise,
)

__all__ = [
    "ChuaParameters",
    "chua_arctan_parameters",
    "chua_arctan_wu2023_parameters",
    "chua_nonsmooth_parameters",
    "chua_parameters",
    "equilibria_arctan",
    "equilibria_nonsmooth",
    "jacobian_arctan",
    "jacobian_nonsmooth",
    "nonlinearity_nonsmooth",
    "rhs_nonsmooth",
    "nonlinearity_arctan",
    "nonlinearity_chua",
    "normalize_chua_model",
    "rhs_arctan",
    "rhs_chua",
]
