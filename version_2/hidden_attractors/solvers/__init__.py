"""Fractional-system solver interfaces.

Stability: experimental
    Currently wraps the C/EFORK backend.  New solver interfaces (full-history
    ABM, etc.) will be added here as they are migrated from legacy scripts.
    Existing interfaces will not be removed without a deprecation notice.

Currently this package exposes the repository C/EFORK wrapper.  Full-history
ABM and other Caputo solvers should live here as reusable modules when they are
migrated out of legacy experiment scripts.
"""

from ..native.backends import FractionalChuaBackend
from .efork_published import EFORK3Coefficients, efork3_caputo_integrate, efork3_coefficients
from .history import FractionalHistory
from .integer import efork_q1_integrate, efork_q1_step
from .rk4 import rk4_integrate, rk4_step

__all__ = [
    "EFORK3Coefficients",
    "FractionalChuaBackend",
    "FractionalHistory",
    "efork3_caputo_integrate",
    "efork3_coefficients",
    "efork_q1_integrate",
    "efork_q1_step",
    "rk4_integrate",
    "rk4_step",
]
