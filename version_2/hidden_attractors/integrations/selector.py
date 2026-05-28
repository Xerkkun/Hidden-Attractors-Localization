"""Integrator selector with q-compatibility validation.

Stability: experimental

This module wraps ``integrate_general`` from the migrated integrators with
strict validation rules ensuring that the chosen numerical method is
compatible with the fractional order ``q``.

Rules
-----
- q == 1.0 + abm           → ValueError  (ABM requires 0 < q < 1)
- q < 1.0  + rk4/heun      → ValueError  (RK4/Heun require integer order)
- q < 1.0  + efork_q1      → ValueError  (efork_q1 is the q=1 limit)
- q == 1.0 + efork3        → UserWarning + redirect to efork_q1
- q < 1.0  + adm_wu2023    → allowed (local ADM for Caputo)
- q < 1.0  + efork3        → allowed
- q < 1.0  + abm           → allowed
- q == 1.0 + rk4/heun/efork_q1 → allowed
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Compatibility matrix
# ---------------------------------------------------------------------------

# Set of integrator names valid for q < 1 (fractional)
_FRACTIONAL_INTEGRATORS = {"abm", "efork3", "efork", "adm_wu2023"}

# Set of integrator names valid for q == 1 (integer-order)
_INTEGER_INTEGRATORS = {"rk4", "heun", "efork_q1", "efork3", "efork"}

# Integrators that are ONLY for fractional order (fail at q=1)
_FRACTIONAL_ONLY = {"abm", "adm_wu2023"}

# Integrators that are ONLY for integer order (fail at q<1)
_INTEGER_ONLY = {"rk4", "heun", "efork_q1"}


def _canonical_name(integrator: str) -> str:
    """Normalize integrator names for internal dispatch."""
    name = integrator.strip().lower()
    if name == "efork":
        return "efork3"
    return name


def validate_integrator_compatibility(integrator: str, q: float) -> str:
    """Validate and return canonical integrator name.

    Parameters
    ----------
    integrator : str
        Requested integrator name (case-insensitive).
    q : float
        Fractional order (0 < q <= 1).

    Returns
    -------
    str
        Canonical integrator name after validation.

    Raises
    ------
    ValueError
        If the integrator is incompatible with the given ``q``.
    """
    name = _canonical_name(integrator)

    if q <= 0.0 or q > 1.0:
        raise ValueError(
            f"Fractional order q must be in (0, 1]. Got q={q}."
        )

    is_fractional = q < 1.0
    is_integer = abs(q - 1.0) < 1e-10

    if is_integer:
        # q == 1.0
        if name in _FRACTIONAL_ONLY:
            raise ValueError(
                f"Integrator '{integrator}' requires q < 1 (fractional Caputo). "
                f"Got q={q}. Use 'rk4', 'heun', or 'efork_q1' for integer-order systems."
            )
        if name in ("efork3",):
            warnings.warn(
                f"Integrator 'efork3' at q=1.0 redirects to the integer-order "
                f"'efork_q1' limit. For pure integer-order work, prefer 'rk4' or "
                f"'heun' which are simpler and faster.",
                UserWarning,
                stacklevel=3,
            )
            return "efork3"  # general.py handles the q=1 path internally
    else:
        # q < 1.0
        if name in _INTEGER_ONLY:
            raise ValueError(
                f"Integrator '{integrator}' only supports integer-order systems (q=1). "
                f"Got q={q}. Use 'abm' or 'efork3' for fractional Caputo integration."
            )

    return name


def get_integrator_fn() -> Callable:  # type: ignore[type-arg]
    """Return the unified ``integrate_general`` function.

    Importing lazily avoids circular-import issues during package init.
    """
    from .general import integrate_general  # type: ignore[import]
    return integrate_general


def integrate(
    rhs: Callable,
    x0: np.ndarray,
    q: float,
    h: float,
    t_final: float,
    integrator: str = "efork3",
    memory_mode: str = "full",
    memory_window_length: Optional[int] = None,
    divergence_norm: Optional[float] = 120.0,
    system: Optional[Any] = None,
    use_c_backend: bool = True,
    allow_python_fallback: bool = True,
    early_stop_config: Optional[dict] = None,
    equilibria: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Validated unified integrator entry point.

    Validates q-integrator compatibility before calling ``integrate_general``.

    Parameters
    ----------
    rhs : callable
        Vector field ``f(t, x) -> ndarray`` or ``f(x) -> ndarray``.
    x0 : array-like (dim,)
        Initial condition.
    q : float
        Fractional order, 0 < q <= 1.
    h : float
        Step size.
    t_final : float
        Integration end time.
    integrator : str
        One of 'rk4', 'heun', 'efork_q1', 'efork3', 'abm', 'adm_wu2023'.
    memory_mode : str
        'full' or 'window'. Ignored for integer-order.
    memory_window_length : int, optional
        Number of steps for windowed memory.
    divergence_norm : float, optional
        Hard-stop threshold on ||x||.
    system : object, optional
        System object passed to Numba fast path.
    use_c_backend : bool
        Attempt the compiled C/Numba backend.
    allow_python_fallback : bool
        Fall back to pure-Python when C/Numba unavailable.
    early_stop_config : dict, optional
        Early-stop configuration (divergence + equilibrium checks).
    equilibria : list of ndarray, optional
        Known equilibria for equilibrium early-stop checks.

    Returns
    -------
    t_arr : ndarray (M,)
    x_arr : ndarray (M, dim)
    status : str  — 'ok', 'diverged', 'diverged_early', etc.
    """
    canonical = validate_integrator_compatibility(integrator, q)

    integrate_general = get_integrator_fn()

    # Unified signature wrapper to handle autonomous/non-autonomous and parametric system callables
    def wrapped_rhs(*args, **kwargs):
        if len(args) == 2:
            t, x = args
        elif len(args) == 1:
            t, x = 0.0, args[0]
        else:
            raise TypeError(f"RHS callable expects 1 or 2 arguments, got {len(args)}")

        if system is not None:
            # Use evaluate to automatically bind default parameters
            return system.evaluate(x)
        else:
            # Direct fallback when no system object is provided
            try:
                return rhs(x)
            except TypeError:
                return rhs(t, x)

    return integrate_general(
        rhs=wrapped_rhs,
        x0=np.asarray(x0, dtype=float),
        q=q,
        h=h,
        t_final=t_final,
        integrator=canonical,
        memory_mode=memory_mode,
        memory_window_length=memory_window_length,
        divergence_norm=divergence_norm,
        system=system,
        use_c_backend=use_c_backend,
        early_stop_config=early_stop_config,
        equilibria=equilibria,
    )
