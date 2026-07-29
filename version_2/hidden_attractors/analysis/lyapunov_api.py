"""Common Lyapunov-method dispatcher.

Common Lyapunov API
===================
This module provides a **method-agnostic interface** for computing Lyapunov
exponent spectra.  Callers select a method by name; the dispatcher validates
compatibility between method, fractional order *q*, and memory mode, then
routes to the appropriate implementation.

Supported routes
----------------
* Routes ``integer_qr_benettin`` to the integer QR implementation.
* Routes ``fractional_variational_abm_qr`` → Caputo extended-variational ABM-QR.
* ``memory_mode`` must be ``'full'`` or ``'window'`` (not ``'not_applicable'``).
* ``q`` must be in (0, 1).
* Routes the GS and QR cloned-dynamics methods.
* Does not require a Jacobian or a variational system.
* Uses block-restarted ABM memory for fractional execution.

The dispatcher does not certify
-------------------------------
* chaos_certified_by_this_pipeline: false
* hiddenness_certified_by_this_pipeline: false

References
----------
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
from .lyapunov_fractional import fractional_variational_abm_qr as _frac_abm_qr
from .lyapunov_cloned import compute_cloned_dynamics_spectrum as _cloned_dynamics
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
        ``integer_qr_benettin``; implemented fractional methods use their
        declared full-history or block-restart contract.
    memory_window : int or None, default None
        Memory window size for methods that support windowed memory.
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
        or ``'invalid_parameter'``.
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

    # 2. Generic numeric validation
    try:
        q_value = float(request.q)
        h_value = float(request.h)
        t_final_value = float(request.t_final)
        t_burn_value = float(request.t_burn)
        jacobian_eps_value = float(request.jacobian_eps)
        state = np.asarray(request.x0, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return False, "invalid_parameter", ("numeric parameters must be real numbers.",)

    if not all(
        np.isfinite(value)
        for value in (q_value, h_value, t_final_value, t_burn_value, jacobian_eps_value)
    ):
        return False, "invalid_parameter", ("q, h, times, and jacobian_eps must be finite.",)
    if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
        return False, "invalid_parameter", ("x0 must be a non-empty finite one-dimensional array.",)
    if h_value <= 0.0:
        return False, "invalid_parameter", ("h must be positive.",)
    if t_final_value <= 0.0:
        return False, "invalid_parameter", ("t_final must be positive.",)
    if t_burn_value < 0.0:
        return False, "invalid_parameter", ("t_burn must be non-negative.",)
    if jacobian_eps_value <= 0.0:
        return False, "invalid_parameter", ("jacobian_eps must be positive.",)
    if request.reorthonormalize_every is not None and int(request.reorthonormalize_every) < 1:
        return False, "invalid_parameter", ("reorthonormalize_every must be positive.",)
    if request.reorthonormalization_time is not None:
        interval = float(request.reorthonormalization_time)
        if not np.isfinite(interval) or interval <= 0.0:
            return False, "invalid_parameter", ("reorthonormalization_time must be finite and positive.",)
    if request.div_threshold is not None:
        threshold = float(request.div_threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            return False, "invalid_parameter", ("div_threshold must be finite and positive.",)

    # 3. Method-specific validation
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

    elif request.method == "fractional_variational_abm_qr":
        # q must be strictly in (0, 1)
        q_val = float(request.q)
        if not (0.0 < q_val < 1.0):
            return (
                False,
                "method_not_valid_for_integer_or_out_of_range_q",
                (
                    f"fractional_variational_abm_qr requires 0 < q < 1; "
                    f"received q={request.q}. "
                    "For q=1, use integer_qr_benettin.",
                ),
            )
        # memory_mode must be 'full' or 'window'
        if request.memory_mode not in ("full", "window"):
            return (
                False,
                "memory_mode_must_be_full_or_window_for_fractional_method",
                (
                    f"fractional_variational_abm_qr requires memory_mode='full' or "
                    f"'window'; received memory_mode='{request.memory_mode}'.",
                ),
            )
        # memory_window required if memory_mode='window'
        if request.memory_mode == "window" and (
            request.memory_window is None or int(request.memory_window) < 1
        ):
            return (
                False,
                "invalid_parameter",
                ("memory_window must be a positive int when memory_mode='window'.",),
            )
        # Check system.q consistency if system provided
        if request.system is not None:
            sys_q = None
            for attr in ("q", "order", "fractional_order"):
                try:
                    v = getattr(request.system, attr, None)
                    if v is not None:
                        sys_q = float(v)
                        break
                except Exception:
                    pass
            if sys_q is not None and abs(sys_q - q_val) > 1e-9:
                return (
                    False,
                    "request_q_does_not_match_system_q",
                    (
                        f"request.q={q_val} does not match system.q={sys_q}. "
                        "Ensure the fractional order is consistent.",
                    ),
                )
        # Jacobian advisory
        if request.jacobian is None and request.system is None:
            warnings.append("analytic_jacobian_missing_finite_difference_used")
        elif request.jacobian is None and request.system is not None:
            if not callable(getattr(request.system, "jacobian_matrix", None)):
                warnings.append("analytic_jacobian_missing_finite_difference_used")

    elif request.method in {
        "fractional_cloned_dynamics_abm_gs_published",
        "fractional_cloned_dynamics_abm_qr",
    }:
        q_val = float(request.q)
        if not (0.0 < q_val <= 1.0):
            return (
                False,
                "method_not_valid_for_out_of_range_q",
                (f"cloned dynamics requires 0 < q <= 1; received q={request.q}.",),
            )
        if request.memory_mode not in (
            "not_applicable",
            "published_block_restart",
            "experimental_qr_block_restart",
        ):
            return (
                False,
                "memory_mode_must_be_block_restart_for_cloned_dynamics",
                (
                    "cloned dynamics supports published_block_restart memory only; "
                    f"received memory_mode='{request.memory_mode}'.",
                ),
            )
        warnings.append("cloned_dynamics_no_jacobian_required")

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

    **Common Lyapunov API entry point**

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
        ``'not_applicable'`` for ``integer_qr_benettin``; fractional methods
        use the memory contract documented by their registry entry.
    memory_window : int or None, default None
        Memory window for methods that support windowed memory.
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
    Notes
    -----
    The dispatcher exposes integer and fractional finite-time numerical
    methods with method-specific validation metadata. A returned spectrum is
    not a chaos or hiddenness certification.

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
    try:
        h_for_interval = float(h)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("h must be a finite positive number.") from exc
    if not np.isfinite(h_for_interval) or h_for_interval <= 0.0:
        raise ValueError("h must be a finite positive number.")
    if reorthonormalization_time is not None:
        interval_time = float(reorthonormalization_time)
        if not np.isfinite(interval_time) or interval_time <= 0.0:
            raise ValueError(
                "reorthonormalization_time must be a finite positive number."
            )
    if reorthonormalize_every is not None:
        every_value = int(reorthonormalize_every)
        if every_value < 1 or float(reorthonormalize_every) != float(every_value):
            raise ValueError("reorthonormalize_every must be a positive integer.")

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
        resolved_every = max(1, round(float(reorthonormalization_time) / h_for_interval))
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

    elif method == "fractional_variational_abm_qr":
        # Resolve rhs / jacobian from system if needed
        _rhs = rhs
        _jac = jacobian
        if system is not None:
            if not callable(getattr(system, "evaluate", None)):
                raise ValueError(
                    "compute_lyapunov_spectrum: system must expose a callable evaluate(state)."
                )
            _rhs = lambda state: system.evaluate(state)
            if callable(getattr(system, "jacobian_matrix", None)):
                _jac = lambda state: system.jacobian_matrix(state)
            else:
                _jac = None
        if _rhs is None:
            raise ValueError(
                "compute_lyapunov_spectrum: either 'system' or 'rhs' must be provided."
            )
        history_aware = extra.get("history_aware_qr", True)
        qr_epsilon = extra.get("qr_epsilon", 1e-300)
        result = _frac_abm_qr(
            _rhs,
            _jac,
            request.x0,
            q=float(q),
            h=float(h),
            t_final=float(t_final),
            t_burn=float(t_burn),
            reorthonormalization_time=reorthonormalization_time,
            reorthonormalize_every=resolved_every,
            memory_mode=memory_mode,
            memory_window=memory_window,
            jacobian_eps=float(jacobian_eps),
            div_threshold=div_threshold,
            history_aware_qr=bool(history_aware),
            qr_epsilon=float(qr_epsilon),
        )

    elif method in {
        "fractional_cloned_dynamics_abm_gs_published",
        "fractional_cloned_dynamics_abm_qr",
    }:
        _rhs = rhs
        if system is not None:
            if not callable(getattr(system, "evaluate", None)):
                raise ValueError(
                    "compute_lyapunov_spectrum: system must expose a callable evaluate(state)."
                )
            _rhs = lambda state: system.evaluate(state)
        if _rhs is None:
            raise ValueError(
                "compute_lyapunov_spectrum: either 'system' or 'rhs' must be provided."
            )
        cloned_method = "gs" if method.endswith("_gs_published") else "qr"
        t_clone = float(
            extra.get(
                "t_clone",
                reorthonormalization_time
                if reorthonormalization_time is not None
                else resolved_every * float(h),
            )
        )
        k_blocks = int(extra.get("k_blocks", max(1, round(float(t_final) / t_clone))))
        orders = extra.get("orders", [float(q)])
        memory_protocol = str(
            extra.get(
                "memory_protocol",
                "published_block_restart"
                if cloned_method == "gs"
                else "experimental_qr_block_restart",
            )
        )
        result = _cloned_dynamics(
            _rhs,
            request.x0,
            orders=orders,
            h=float(h),
            t_clone=t_clone,
            n_clones=extra.get("n_clones"),
            k_blocks=k_blocks,
            delta=float(extra.get("delta", 1e-3)),
            method=cloned_method,
            memory_protocol=memory_protocol,
            system_id=extra.get("system_id"),
            parameters=extra.get("parameters"),
            return_history=bool(extra.get("return_history", False)),
            random_seed=extra.get("random_seed"),
            divergence_norm=div_threshold,
        )
        request_summary.update(
            {
                "orders": [float(value) for value in np.asarray(orders, dtype=float).reshape(-1)],
                "t_clone": t_clone,
                "k_blocks": k_blocks,
                "delta": float(extra.get("delta", 1e-3)),
                "memory_protocol": memory_protocol,
            }
        )

    else:
        raise RuntimeError(
            f"Public Lyapunov registry/dispatcher mismatch for method '{method}'."
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
