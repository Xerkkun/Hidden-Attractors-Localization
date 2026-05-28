"""Numerical integrators and external-library adapters.

Stability: experimental

Sub-modules
-----------
selector
    Validated entry point (q-compatibility checks).  Use ``integrate()``
    from here for all production workflows.
general
    Unified dispatcher for fractional and integer-order integration.
abm
    Adams–Bashforth–Moulton predictor-corrector (Caputo, q < 1).
efork
    EFORK-3 explicit three-stage Caputo method (q < 1 or q = 1 limit).
rk4
    Classical 4th-order Runge–Kutta (q = 1 only).
adm_wu2023
    Local Adomian Decomposition Method (Wu et al. 2023 reproduction).
fractional_c
    Dispatcher to the compiled C / Python-fallback Caputo backends.
external_tools
    Adapters for ``nolds``, ``antropy``, and similar optional backends.
"""

# ── Validated entry point (recommended for workflows) ───────────────────────
from .selector import integrate, validate_integrator_compatibility

# ── Low-level integrators (for direct / advanced use) ───────────────────────
from .general import integrate_general
from .abm import caputo_abm_integrate
from .efork import efork_integrate
from .rk4 import rk4_integrate
from .adm_wu2023 import adm_wu2023_integrate
from .fractional_c import fractional_integrate

# ── External complexity adapters ─────────────────────────────────────────────
from .external_tools import (
    EXTERNAL_TOOLS,
    ExternalTool,
    available_complexity_backends,
    compute_complexity_measures,
    external_tool_report,
    require_external,
)

__all__ = [
    # selector
    "integrate",
    "validate_integrator_compatibility",
    # general
    "integrate_general",
    # low-level integrators
    "caputo_abm_integrate",
    "efork_integrate",
    "rk4_integrate",
    "adm_wu2023_integrate",
    "fractional_integrate",
    # external adapters
    "EXTERNAL_TOOLS",
    "ExternalTool",
    "available_complexity_backends",
    "compute_complexity_measures",
    "external_tool_report",
    "require_external",
]
