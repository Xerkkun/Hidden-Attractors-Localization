"""Common Lyapunov API — F1 dispatcher.

F1 — Common Lyapunov API
=========================
This module provides a **method-agnostic interface** for computing Lyapunov
exponent spectra.  Callers select a method by name; the dispatcher validates
compatibility between method, fractional order *q*, and memory mode, then
routes to the appropriate implementation.

F1 scope
--------
* Provides ``compute_lyapunov_spectrum`` as the single public entry point.
* Validates method/q/memory-mode compatibility via
  ``validate_lyapunov_method_request``.
* Routes ``integer_qr_benettin`` to the frozen F0 implementation.
* Raises ``NotImplementedError`` for registered-but-unimplemented fractional
  methods so that callers get an informative error instead of silent wrong
  results.

F1 does NOT implement
---------------------
* ``fractional_variational_abm_qr``     — registered, NOT implemented.
* ``fractional_cloned_dynamics_abm``    — registered, NOT implemented.
* 0–1 test, PSD/FFT, Poincaré sections, ``chaos_validation_summary``.

F1 does NOT certify
-------------------
* chaos_certified_by_this_pipeline: false
* hiddenness_certified_by_this_pipeline: false

References
----------
F1 inherits the references frozen in F0 (integer_qr_benettin):
.. [Benettin1980] G. Benettin et al., Meccanica 15, 1980.
.. [Wolf1985] A. Wolf et al., Physica D 16, 1985.
.. [Danca2018] M.-F. Danca & N. Kuznetsov, Int. J. Bifurcation Chaos 28(5),
   2018.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .lyapunov import (
    LyapunovResult,
    integer_qr_benettin_lyapunov_exponents,
    integer_system_lyapunov_exponents,
)
from .lyapunov_methods import LYAPUNOV_METHODS, LyapunovMethodInfo

# ---------------------------------------------------------------------------
# Request / Summary dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LyapunovComputationRequest:
    """Structured request for a Lyapunov spectrum computation.

    Attributes
    ----------
    system : object or None
        System object (e.g. ``ChaoticSystem``).  Either *system* or *rhs*
        must be provided.
    rhs : callable or None
        Right-hand side ``F(x) -> dxdt``.  Used when *system* is ``None``.
    jacobian : callable or None
        Analytic Jacobian ``J(x) -> (n, n)``.  ``None`` triggers finite
        differences for methods that support it.
    x0 : np.ndarray
        Initial state vector.
    q : float
        Fractional order.  Must be ``1.0`` for ``integer_qr_benettin``.
    method : str
        Canonical method identifier (e.g. ``'integer_qr_benettin'``).
    h : float
        Integration step size (must be positive).
    t_final : float
        Total integration time.
    t_burn : float, default 0.0
        Burn-in time discarded before accumulating exponents.
    reorthonormalization_time : float or None, default None
        Physical time between reorthonormalisations.  Converted to
        ``reorthonormalize_every`` steps.  Ignored if
        ``reorthonormalize_every`` is also set (warning issued).
    reorthonormalize_every : int or None, default None
        Steps between reorthonormalisations.  If both this and
        ``reorthonormalization_time`` are ``None``, the method default
        (10) is used.
    jacobian_eps : float, default 1e-6
        Finite-difference step when no analytic Jacobian is provided.
    div_threshold : float or None, default None
        Divergence threshold on the state norm.
    memory_mode : str, default ``'not_applicable'``
        Memory handling mode.  Must be ``'not_applicable'`` for
        ``integer_qr_benettin``.  Future fractional methods will accept
        ``'full'`` or ``'window'``.
    memory_window : int or None, default None
        Memory window size (for future fractional methods).
    extra : dict, default {}
        Additional method-specific parameters.

    Notes
    -----
    Fields ``hidden_verified``, ``chaos_verified``,
    ``fractional_lyapunov_validated``, and ``caputo_lyapunov_validated``
    are intentionally absent.
    """

    system: object | None
    rhs: Callable[[np.ndarray], np.ndarray] | None
    jacobian: Callable[[np.ndarray], np.ndarray] | None
    x0: np.ndarray
    q: float
    method: str
    h: float
    t_final: float
    t_burn: float = 0.0
    reorthonormalization_time: float | None = None
    reorthonormalize_every: int | None = None
    jacobian_eps: float = 1e-6
    div_threshold: float | None = None
    memory_mode: str = "not_applicable"
    memory_window: int | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LyapunovComputationSummary:
    """Result of a ``compute_lyapunov_spectrum`` call.

    Attributes
    ----------
    result : LyapunovResult
        The numerical Lyapunov exponent estimates and metadata.
    method_info : LyapunovMethodInfo
        Registry metadata for the method that was used.
    request_summary : dict
        Key parameters from the request (method, q, h, t_final, etc.)
        serialised as a plain dict for logging and reproducibility.
    compatibility_status : str
        ``'compatible'`` if validation passed; otherwise the failure
        status string (should not reach here in normal operation, since
        ``compute_lyapunov_spectrum`` raises on incompatible requests).
    warnings : tuple[str, ...]
        Validation and methodological warnings (e.g.,
        ``'analytic_jacobian_missing_finite_difference_used'``).

    Notes
    -----
    Fields ``hidden_verified``, ``chaos_verified``,
    ``fractional_lyapunov_validated``, and ``caputo_lyapunov_validated``
    are intentionally absent.
    """

    result: LyapunovResult
    method_info: LyapunovMethodInfo
    request_summary: dict[str, object]
    compatibility_status: str
    warnings: tuple[str, ...]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_lyapunov_method_request(
    request: LyapunovComputationRequest,
) -> tuple[bool, str, tuple[str, ...]]:
    """Validate a :class:`LyapunovComputationRequest`.

    Returns a 3-tuple ``(ok, status, warnings)``.

    Parameters
    ----------
    request : LyapunovComputationRequest
        The request to validate.

    Returns
    -------
    ok : bool
        ``True`` if the request is compatible with the requested method.
    status : str
        ``'compatible'`` on success.  On failure one of:
        ``'unknown_method'``, ``'method_not_valid_for_fractional_caputo'``,
        ``'memory_mode_not_applicable_for_integer_method'``,
        ``'method_not_implemented'``, ``'invalid_parameter'``.
    warnings : tuple[str, ...]
        Non-fatal advisory strings.

    Notes
    -----
    This function does **not** raise; the caller decides what to do.
    """
    warnings: list[str] = []

    # 1. Method must be known
    if request.method not in LYAPUNOV_METHODS:
        return (
            False,
            "unknown_method",
            (f"Method '{request.method}' is not in the LYAPUNOV_METHODS registry.",),
        )

    info = LYAPUNOV_METHODS[request.method]

    # 2. Method-specific validation
    if request.method == "integer_qr_benettin":
        # q must be 1
        if abs(float(request.q) - 1.0) > 1e-9:
            return (
                False,
                "method_not_valid_for_fractional_caputo",
                (
                    f"integer_qr_benettin is valid only for q=1 (integer-order ODE); "
                    f"received q={request.q}. "
                    "Use a fractional Lyapunov method for Caputo q<1.",
                ),
            )
        # memory_mode must be 'not_applicable'
        if request.memory_mode not in ("not_applicable", None):
            return (
                False,
                "memory_mode_not_applicable_for_integer_method",
                (
                    f"integer_qr_benettin does not use memory; "
                    f"received memory_mode='{request.memory_mode}'. "
                    "Set memory_mode='not_applicable'.",
                ),
            )
        # Jacobian advisory
        if request.jacobian is None and request.system is None:
            warnings.append("analytic_jacobian_missing_finite_difference_used")
        elif request.jacobian is None:
            # system provided but may not have analytic jacobian — still ok
            pass

    # 3. Registered but not implemented
    elif not info.implemented:
        return (
            False,
            "method_not_implemented",
            (
                f"Method '{request.method}' is registered in LYAPUNOV_METHODS "
                f"(derivative_model='{info.derivative_model}') but is not yet implemented. "
                "It will be available in a future phase.",
            ),
        )

    # 4. Numeric parameter checks
    if float(request.h) <= 0.0:
        return False, "invalid_parameter", ("h must be positive.",)
    if float(request.t_final) <= 0.0:
        return False, "invalid_parameter", ("t_final must be positive.",)
    if float(request.t_burn) < 0.0:
        return False, "invalid_parameter", ("t_burn must be non-negative.",)

    return True, "compatible", tuple(warnings)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def compute_lyapunov_spectrum(
    *,
    system: object | None = None,
    rhs: Callable[[np.ndarray], np.ndarray] | None = None,
    jacobian: Callable[[np.ndarray], np.ndarray] | None = None,
    x0: np.ndarray,
    q: float,
    method: str,
    h: float,
    t_final: float,
    t_burn: float = 0.0,
    reorthonormalization_time: float | None = None,
    reorthonormalize_every: int | None = None,
    jacobian_eps: float = 1e-6,
    div_threshold: float | None = None,
    memory_mode: str = "not_applicable",
    memory_window: int | None = None,
    **extra: object,
) -> LyapunovComputationSummary:
    """Compute the Lyapunov spectrum using a named method.

    **F1 — Common Lyapunov API entry point**

    This is the single, method-agnostic entry point for computing Lyapunov
    exponent spectra.  Pass a *method* name (e.g. ``'integer_qr_benettin'``)
    and the dispatcher:

    1. Builds a :class:`LyapunovComputationRequest`.
    2. Validates compatibility via :func:`validate_lyapunov_method_request`.
    3. Resolves ``reorthonormalize_every`` from ``reorthonormalization_time``
       if needed.
    4. Routes to the correct implementation.
    5. Returns a :class:`LyapunovComputationSummary`.

    Parameters
    ----------
    system : object or None, default None
        System object.  Must expose ``evaluate`` and optionally ``jacobian``/
        ``jacobian_matrix``.  Use this **or** *rhs*, not both.
    rhs : callable or None, default None
        Vector field ``F(x) -> dxdt``.  Used when *system* is ``None``.
    jacobian : callable or None, default None
        Analytic Jacobian ``J(x) -> (n, n)``.  ``None`` → finite differences.
    x0 : np.ndarray
        Initial state.
    q : float
        Fractional order.  Must be ``1.0`` for ``'integer_qr_benettin'``.
    method : str
        Canonical method identifier (see ``LYAPUNOV_METHODS``).
    h : float
        Integration step size (must be positive).
    t_final : float
        Total integration time (burn-in excluded).
    t_burn : float, default 0.0
        Burn-in time.
    reorthonormalization_time : float or None, default None
        Physical time between reorthonormalisations; converted to steps.
        Ignored if *reorthonormalize_every* is also provided (warning issued).
    reorthonormalize_every : int or None, default None
        Steps between QR reorthonormalisations.  If both this and
        *reorthonormalization_time* are ``None``, defaults to 10.
    jacobian_eps : float, default 1e-6
        Finite-difference step when no analytic Jacobian is provided.
    div_threshold : float or None, default None
        Divergence threshold on the state norm.
    memory_mode : str, default ``'not_applicable'``
        ``'not_applicable'`` for ``integer_qr_benettin``.  Future fractional
        methods will accept ``'full'`` or ``'window'``.
    memory_window : int or None, default None
        Memory window for future fractional methods.
    **extra : object
        Additional method-specific parameters stored in
        :attr:`LyapunovComputationRequest.extra`.

    Returns
    -------
    summary : LyapunovComputationSummary
        Exponent estimates, method metadata, request summary, and warnings.

    Raises
    ------
    ValueError
        If the request is invalid (unknown method, q/method mismatch,
        memory_mode mismatch, bad parameters) or neither *system* nor *rhs*
        is provided.
    NotImplementedError
        If the method is registered but not yet implemented (e.g.,
        ``'fractional_variational_abm_qr'``).

    Notes
    -----
    This function is **not a validated Caputo fractional Lyapunov method**.
    It is restricted to ``q=1`` in F1.

    chaos_certified_by_this_pipeline: false
    hiddenness_certified_by_this_pipeline: false

    Examples
    --------
    >>> import numpy as np
    >>> from hidden_attractors.analysis import compute_lyapunov_spectrum
    >>> rhs = lambda x: np.array([-x[0], -2*x[1]])
    >>> summary = compute_lyapunov_spectrum(
    ...     rhs=rhs, x0=np.array([1.0, 1.0]), q=1.0,
    ...     method="integer_qr_benettin", h=0.01, t_final=50.0)
    >>> summary.compatibility_status
    'compatible'
    """
    # ------------------------------------------------------------------
    # B4 — resolve reorthonormalize_every
    # ------------------------------------------------------------------
    _extra_warnings: list[str] = []
    resolved_every: int

    if reorthonormalize_every is not None and reorthonormalization_time is not None:
        _extra_warnings.append(
            "both_reorthonormalization_time_and_every_provided_using_every"
        )
        resolved_every = int(reorthonormalize_every)
    elif reorthonormalize_every is not None:
        resolved_every = int(reorthonormalize_every)
    elif reorthonormalization_time is not None:
        resolved_every = max(1, round(float(reorthonormalization_time) / float(h)))
    else:
        resolved_every = 10  # method default

    # ------------------------------------------------------------------
    # Build request
    # ------------------------------------------------------------------
    request = LyapunovComputationRequest(
        system=system,
        rhs=rhs,
        jacobian=jacobian,
        x0=np.asarray(x0, dtype=float),
        q=float(q),
        method=method,
        h=float(h),
        t_final=float(t_final),
        t_burn=float(t_burn),
        reorthonormalization_time=reorthonormalization_time,
        reorthonormalize_every=resolved_every,
        jacobian_eps=float(jacobian_eps),
        div_threshold=div_threshold,
        memory_mode=memory_mode,
        memory_window=memory_window,
        extra=dict(extra),
    )

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    ok, status, val_warnings = validate_lyapunov_method_request(request)
    all_warnings: tuple[str, ...] = tuple(_extra_warnings) + val_warnings

    if not ok:
        # Registered-but-not-implemented gets NotImplementedError
        if status == "method_not_implemented":
            raise NotImplementedError(
                f"Method '{method}' is registered in LYAPUNOV_METHODS but is not yet "
                f"implemented in F1. "
                f"Status: {status}. "
                f"Details: {'; '.join(val_warnings)}"
            )
        raise ValueError(
            f"Lyapunov request validation failed. "
            f"Status: {status}. "
            f"Details: {'; '.join(val_warnings)}"
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    method_info = LYAPUNOV_METHODS[method]

    request_summary: dict[str, object] = {
        "method": method,
        "q": float(q),
        "h": float(h),
        "t_final": float(t_final),
        "t_burn": float(t_burn),
        "reorthonormalize_every": resolved_every,
        "reorthonormalization_time": reorthonormalization_time,
        "memory_mode": memory_mode,
        "memory_window": memory_window,
    }

    if method == "integer_qr_benettin":
        if system is not None:
            result: LyapunovResult = integer_system_lyapunov_exponents(
                system,
                request.x0,
                h=float(h),
                t_final=float(t_final),
                t_burn=float(t_burn),
                reorthonormalize_every=resolved_every,
                jacobian_eps=float(jacobian_eps),
                div_threshold=div_threshold,
            )
        elif rhs is not None:
            result = integer_qr_benettin_lyapunov_exponents(
                rhs,
                jacobian,
                request.x0,
                h=float(h),
                t_final=float(t_final),
                t_burn=float(t_burn),
                reorthonormalize_every=resolved_every,
                jacobian_eps=float(jacobian_eps),
                div_threshold=div_threshold,
                q=1.0,
            )
        else:
            raise ValueError(
                "compute_lyapunov_spectrum: either 'system' or 'rhs' must be provided."
            )
    else:
        # Should not reach here (validate catches not-implemented),
        # but defensive fallback:
        raise NotImplementedError(
            f"Method '{method}' routing is not implemented in F1."
        )

    return LyapunovComputationSummary(
        result=result,
        method_info=method_info,
        request_summary=request_summary,
        compatibility_status="compatible",
        warnings=all_warnings,
    )


__all__ = [
    "LyapunovComputationRequest",
    "LyapunovComputationSummary",
    "validate_lyapunov_method_request",
    "compute_lyapunov_spectrum",
]
