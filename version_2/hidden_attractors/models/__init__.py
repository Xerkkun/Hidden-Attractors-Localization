"""Dynamical-system definitions.

Stability: stable
    Chua parameters, vector fields, and equilibria.  Signatures and return
    types are fixed; breaking changes require a version bump and a deprecation
    cycle.
"""

from .chua import (
    ChuaParameters,
    chua_arctan_parameters,
    chua_parameters,
    chua_piecewise_parameters,
    equilibria_piecewise,
    nonlinearity_arctan,
    nonlinearity_chua,
    nonlinearity_piecewise,
    normalize_chua_model,
    rhs_chua,
    rhs_piecewise,
)

__all__ = [
    "ChuaParameters",
    "chua_arctan_parameters",
    "chua_parameters",
    "chua_piecewise_parameters",
    "equilibria_piecewise",
    "nonlinearity_arctan",
    "nonlinearity_chua",
    "nonlinearity_piecewise",
    "normalize_chua_model",
    "rhs_chua",
    "rhs_piecewise",
]
