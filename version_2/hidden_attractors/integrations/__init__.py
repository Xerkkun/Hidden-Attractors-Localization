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

from importlib import import_module

_EXPORT_GROUPS = {
    ".selector": ("integrate", "validate_integrator_compatibility"),
    ".general": ("integrate_general",),
    ".abm": ("caputo_abm_integrate",),
    ".efork": ("efork_integrate",),
    ".rk4": ("rk4_integrate",),
    ".adm_wu2023": ("adm_wu2023_integrate",),
    ".fractional_c": ("fractional_integrate",),
    ".abm_fractional": ("integrate_fractional_abm",),
    ".external_tools": (
        "EXTERNAL_TOOLS",
        "ExternalTool",
        "available_complexity_backends",
        "compute_complexity_measures",
        "external_tool_report",
        "require_external",
    ),
}
_LAZY_EXPORTS = {
    name: module_name
    for module_name, names in _EXPORT_GROUPS.items()
    for name in names
}


def __getattr__(name: str):
    """Resolve an integration symbol on first access."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

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
    "integrate_fractional_abm",
    # external adapters
    "EXTERNAL_TOOLS",
    "ExternalTool",
    "available_complexity_backends",
    "compute_complexity_measures",
    "external_tool_report",
    "require_external",
]
