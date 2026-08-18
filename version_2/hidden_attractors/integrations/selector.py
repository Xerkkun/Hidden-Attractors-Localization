"""Integrator selector with q-compatibility validation.

Stability: experimental

This module wraps ``integrate_general`` from the migrated integrators with
strict validation rules ensuring that the chosen numerical method is
compatible with the fractional order ``q``.

Rules
-----
- q == 1.0 + abm           → ValueError  (ABM requires 0 < q < 1)
- q < 1.0  + rk4           → ValueError  (RK4 requires integer order)
- q < 1.0  + efork_q1      → ValueError  (efork_q1 is the q=1 limit)
- q == 1.0 + efork3        → UserWarning + redirect to efork_q1
- q < 1.0  + adm_wu2023    → compatibility-valid, but the generic
                              ``integrate`` facade rejects it with directions
                              to the specialized ``adm_wu2023_integrate`` API
- q < 1.0  + efork3        → allowed
- q < 1.0  + abm           → allowed
- q == 1.0 + rk4/efork_q1  → allowed
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .._rhs import bind_rhs

# ---------------------------------------------------------------------------
# Compatibility matrix
# ---------------------------------------------------------------------------

# Set of integrator names valid for q < 1 (fractional)
_FRACTIONAL_INTEGRATORS = {"abm", "efork3", "efork", "adm_wu2023"}

# Set of integrator names valid for q == 1 (integer-order)
_INTEGER_INTEGRATORS = {"rk4", "efork_q1", "efork3", "efork"}

# Integrators that are ONLY for fractional order (fail at q=1)
_FRACTIONAL_ONLY = {"abm", "adm_wu2023"}

# Integrators that are ONLY for integer order (fail at q<1)
_INTEGER_ONLY = {"rk4", "efork_q1"}

_KNOWN_INTEGRATORS = _FRACTIONAL_INTEGRATORS | _INTEGER_INTEGRATORS


def _canonical_name(integrator: str) -> str:
    """Normalize integrator names for internal dispatch."""
    name = integrator.strip().lower()
    if name == "efork":
        return "efork3"
    return name


def normalize_fractional_order(q: float) -> float:
    """Return a validated order without coercing near-integer values.

    Only the exactly represented value ``1.0`` selects an integer solver.
    A fractional value such as ``nextafter(1.0, 0.0)`` retains fractional
    memory semantics instead of changing model class through a tolerance.
    """

    try:
        value = float(q)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Fractional order q must be a finite number. Got q={q!r}."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Fractional order q must be a finite number. Got q={q!r}."
        )
    if value <= 0.0 or value > 1.0:
        raise ValueError(
            f"Fractional order q must be in (0, 1]. Got q={q}."
        )
    return value


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
    normalized_q = normalize_fractional_order(q)

    if name not in _KNOWN_INTEGRATORS:
        raise ValueError(
            f"Unknown integrator {integrator!r}. Supported integrators are "
            f"{sorted(_KNOWN_INTEGRATORS)}."
        )

    is_integer = normalized_q == 1.0

    if is_integer:
        # q == 1.0
        if name in _FRACTIONAL_ONLY:
            raise ValueError(
                f"Integrator '{integrator}' requires q < 1 (fractional Caputo). "
                f"Got q={q}. Use 'rk4' or 'efork_q1' for integer-order systems."
            )
        if name in ("efork3",):
            warnings.warn(
                f"Integrator 'efork3' at q=1.0 redirects to the integer-order "
                f"'efork_q1' limit. For pure integer-order work, prefer 'rk4'.",
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
        Integration end time. ``t_final / h`` must be an integer to numerical
        tolerance; fixed-step solvers never overshoot the requested horizon.
    integrator : str
        One of 'rk4', 'efork_q1', 'efork3', or 'abm'.
        ``adm_wu2023`` has a different parameter contract and must be called
        through
        :func:`hidden_attractors.integrations.adm_wu2023_integrate`.
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

    Raises
    ------
    ValueError
        If the integrator is incompatible with the given ``q``, or if
        ``adm_wu2023`` is requested through this generic facade.
    """
    canonical = validate_integrator_compatibility(integrator, q)
    normalized_q = normalize_fractional_order(q)

    if canonical == "adm_wu2023":
        raise ValueError(
            "The generic integrate(...) facade does not dispatch 'adm_wu2023' "
            "because that local ADM method requires its specialized parameter "
            "mapping and step-count contract. Call "
            "hidden_attractors.integrations.adm_wu2023_integrate("
            "params, x0, q, h, N, divergence_norm=...) instead."
        )

    integrate_general = get_integrator_fn()

    # ``ChaoticSystem.rhs`` follows the registry contract
    # ``rhs(state, parameters)``.  That two-argument form is intentionally not
    # guessed by ``bind_rhs`` because it is indistinguishable by arity alone
    # from ``rhs(time, state)``.  When the exact registered vector field is
    # supplied, bind its parameter mapping explicitly through ``evaluate``.
    # A different RHS remains authoritative even when ``system`` is provided
    # only as an acceleration hint.
    if system is not None and rhs is getattr(system, "rhs", None):

        def bound_rhs(time: float, state: np.ndarray) -> np.ndarray:
            del time
            return system.evaluate(state)

    else:
        bound_rhs = bind_rhs(rhs)

    # Unified signature wrapper to handle autonomous/non-autonomous and parametric system callables
    def wrapped_rhs(*args, **kwargs):
        if len(args) == 2:
            t, x = args
        elif len(args) == 1:
            t, x = 0.0, args[0]
        else:
            raise TypeError(f"RHS callable expects 1 or 2 arguments, got {len(args)}")

        return bound_rhs(t, x)

    return integrate_general(
        rhs=wrapped_rhs,
        x0=np.asarray(x0, dtype=float),
        q=normalized_q,
        h=h,
        t_final=t_final,
        integrator=canonical,
        memory_mode=memory_mode,
        memory_window_length=memory_window_length,
        divergence_norm=divergence_norm,
        system=system,
        use_c_backend=use_c_backend,
        allow_python_fallback=allow_python_fallback,
        early_stop_config=early_stop_config,
        equilibria=equilibria,
    )
