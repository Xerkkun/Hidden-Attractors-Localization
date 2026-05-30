"""Reusable Lyapunov exponent estimators.

F0 AUDIT — integer_qr_benettin (frozen)
========================================
This module implements **finite-time local Lyapunov exponents for
integer-order (q=1) ODE systems** using the classical Benettin/QR
reorthonormalisation algorithm.

The canonical identifier for this method is ``integer_qr_benettin``.

Scope
-----
* Valid for: q = 1 (integer-order ODE).
* Variational equation: Φ' = J(X) Φ  (first-order, no memory).
* Orthonormalisation: QR decomposition (numpy.linalg.qr).
* Result: finite-time, local Lyapunov exponent estimates.

Out of scope (F0)
-----------------
* NOT a validated Caputo fractional Lyapunov method.
* Does NOT handle fractional memory (q < 1).
* Does NOT certify chaos by itself.
* Does NOT certify hiddenness of attractors.
* chaos_verified / hidden_verified are NOT asserted here.

Fractional Caputo spectra (q < 1) require a dedicated extended-memory
variational method integrating the full original–variational system with
Caputo memory.  That will be implemented in future phases:
``fractional_variational_abm_qr`` and ``fractional_cloned_dynamics_abm``.

References
----------
.. [Benettin1980] G. Benettin et al., "Lyapunov Characteristic Exponents
   for Smooth Dynamical Systems and for Hamiltonian Systems",
   Meccanica 15, 1980.
.. [Wolf1985] A. Wolf et al., "Determining Lyapunov Exponents from a
   Time Series", Physica D 16, 1985.
.. [Danca2018] M.-F. Danca & N. Kuznetsov, "Matlab Code for Lyapunov
   Exponents of Fractional-Order Systems", Int. J. Bifurcation Chaos
   28(5), 2018. — Establishes that fractional Caputo spectra require
   integrating the extended fractional original–variational system with
   memory; the integer QR method is not valid for q < 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..solvers.integer import efork_q1_step
from ..systems.base import ChaoticSystem

# ---------------------------------------------------------------------------
# F0 canonical references (frozen with this method)
# ---------------------------------------------------------------------------
_INTEGER_QR_BENETTIN_REFS: tuple[str, ...] = (
    "Benettin et al. 1980 — Lyapunov Characteristic Exponents (Meccanica 15)",
    "Wolf et al. 1985 — Determining Lyapunov Exponents from a Time Series (Physica D 16)",
    "Danca & Kuznetsov 2018 — Matlab Code for Lyapunov Exponents of Fractional-Order Systems"
    " (Int. J. Bifurcation Chaos 28(5)): fractional Caputo spectra require extended-memory"
    " variational integration; integer QR is NOT valid for q<1.",
)

_INTEGER_QR_BENETTIN_WARNINGS: tuple[str, ...] = (
    "This routine is not a validated Caputo fractional Lyapunov method."
    " It is restricted to q=1."
    " Fractional Caputo spectra require a dedicated extended-memory variational method.",
    "Finite-time local exponents: convergence depends on integration length and step size.",
    "Does not certify chaos; does not certify hiddenness of attractors.",
)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LyapunovResult:
    """Finite-time Lyapunov exponent estimate.

    Attributes
    ----------
    exponents : np.ndarray, shape (n,)
        Final Lyapunov exponent estimates, one per state dimension.
    times : np.ndarray, shape (K,)
        Times at which intermediate estimates were recorded.
    convergence : np.ndarray, shape (K, n)
        Running Lyapunov estimates at each reorthonormalisation step.
    status : str
        Integration outcome: ``'ok'``, ``'burn_diverged'``,
        ``'solver_exception'``, ``'nonfinite_solution'``, or
        ``'diverged'``.
    method_id : str
        Canonical identifier for the numerical method used.
        ``'integer_qr_benettin'`` for this module (F0).
    derivative_model : str
        Derivative model: ``'integer'`` for q=1 ODE, ``'caputo'`` for
        fractional (not implemented in F0).
    q : float
        Fractional order used.  Must be 1.0 for ``integer_qr_benettin``.
    finite_time_local : bool
        Whether the result is a finite-time local estimate (always True
        for this method).
    jacobian_required : bool
        Whether the method requires a Jacobian (always True here; finite
        differences are used when no analytic Jacobian is provided).
    orthonormalization : str
        Orthonormalisation scheme: ``'qr'`` for this method.
    reference_ids : tuple[str, ...]
        Bibliographic references for the method (F0 frozen set).
    methodological_warnings : tuple[str, ...]
        Human-readable warnings about scope and limitations.
    """

    # Core result fields (original API — must remain first for backward compat)
    exponents: np.ndarray
    times: np.ndarray
    convergence: np.ndarray
    status: str

    # F0 metadata fields (all have defaults → fully backward-compatible)
    method_id: str = "integer_qr_benettin"
    derivative_model: str = "integer"
    q: float = 1.0
    finite_time_local: bool = True
    jacobian_required: bool = True
    orthonormalization: str = "qr"
    reference_ids: tuple[str, ...] = field(default_factory=lambda: _INTEGER_QR_BENETTIN_REFS)
    methodological_warnings: tuple[str, ...] = field(
        default_factory=lambda: _INTEGER_QR_BENETTIN_WARNINGS
    )


# ---------------------------------------------------------------------------
# Jacobian helper
# ---------------------------------------------------------------------------

def finite_difference_jacobian(
    rhs: Callable[[np.ndarray], np.ndarray],
    state: np.ndarray,
    *,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Estimate the Jacobian of *rhs* by central finite differences.

    Parameters
    ----------
    rhs : callable([[np.ndarray], np.ndarray])
        Right-hand side function ``F(x) -> dxdt``, shape ``(n,) -> (n,)``.
    state : np.ndarray, shape (n,)
        Point at which the Jacobian is evaluated.
    eps : float, default 1e-6
        Finite-difference step size.

    Returns
    -------
    J : np.ndarray, shape (n, n)
        Approximate Jacobian ``∂F/∂x`` at *state*.

    Examples
    --------
    >>> import numpy as np
    >>> from hidden_attractors.analysis.lyapunov import finite_difference_jacobian
    >>> rhs = lambda x: np.array([-x[0], x[1]])  # diagonal system
    >>> J = finite_difference_jacobian(rhs, np.array([1.0, 1.0]))
    >>> J.shape
    (2, 2)
    """

    x = np.asarray(state, dtype=float)
    n = x.size
    jac = np.empty((n, n), dtype=float)
    for col in range(n):
        step = np.zeros(n, dtype=float)
        step[col] = float(eps)
        jac[:, col] = (np.asarray(rhs(x + step), dtype=float) - np.asarray(rhs(x - step), dtype=float)) / (2.0 * float(eps))
    return jac


# ---------------------------------------------------------------------------
# Core integer-order estimator (frozen: integer_qr_benettin)
# ---------------------------------------------------------------------------

def integer_lyapunov_exponents(
    rhs: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray] | None,
    x0: np.ndarray,
    *,
    h: float,
    t_final: float,
    t_burn: float = 0.0,
    reorthonormalize_every: int = 10,
    jacobian_eps: float = 1.0e-6,
    div_threshold: float | None = None,
    q: float = 1.0,
) -> LyapunovResult:
    """Estimate integer-order Lyapunov exponents by QR reorthonormalisation.

    **Method identifier: ``integer_qr_benettin`` (F0 — frozen)**

    Uses the Benettin/Wolf algorithm:

    1. Integrate the state ``X' = F(X)`` with an integer-order Euler step
       (:func:`~hidden_attractors.solvers.integer.efork_q1_step`).
    2. Propagate the variational basis ``Φ' = J(X) Φ``
       (first-order, memoryless).
    3. Every ``reorthonormalize_every`` steps apply QR decomposition,
       accumulate ``log|diag(R)|``, and reset the basis to Q.
    4. Divide accumulated sums by elapsed time.

    **Scope**

    * Valid for **q = 1** (integer-order ODE) only.
    * Variational equation: ``Φ' = J(X) Φ`` — first-order, no Caputo memory.
    * Orthonormalisation: QR (``numpy.linalg.qr``).
    * Result: finite-time, local Lyapunov exponent estimates.

    **Methodological warning**

    This routine is **not a validated Caputo fractional Lyapunov method**.
    It is restricted to **q = 1**.
    Fractional Caputo spectra require a dedicated extended-memory variational
    method integrating the full original–variational system with Caputo memory
    (to be implemented in future phases: ``fractional_variational_abm_qr``).

    Parameters
    ----------
    rhs : callable
        Vector field ``F(x) -> dxdt``, shape ``(n,) -> (n,)``.
    jacobian : callable or None
        Analytic Jacobian ``J(x) -> (n, n) array``.  If ``None``,
        :func:`finite_difference_jacobian` is used with *jacobian_eps*.
    x0 : np.ndarray, shape (n,)
        Initial state.
    h : float
        Integration step size (must be positive).
    t_final : float
        Total integration time (burn-in excluded).
    t_burn : float, default 0.0
        Burn-in time discarded before accumulating exponents.
    reorthonormalize_every : int, default 10
        Number of steps between QR reorthonormalisations.
    jacobian_eps : float, default 1e-6
        Finite-difference step used when *jacobian* is ``None``.
    div_threshold : float or None, default None
        If set, integration stops when ``‖x‖ >= div_threshold``.
    q : float, default 1.0
        Fractional order.  This function accepts ``q`` for API
        compatibility but only supports ``q = 1.0``.  If ``q`` differs
        from 1.0 by more than 1e-9, a ``ValueError`` is raised directing
        the user to a fractional method.

    Returns
    -------
    result : LyapunovResult
        Exponent estimates, convergence history, status string, and F0
        method metadata (``method_id='integer_qr_benettin'``).

    Raises
    ------
    ValueError
        If *h* is not positive, *x0* is not one-dimensional, or *q* ≠ 1.0.

    Notes
    -----
    The algorithm integrates a first-order Euler scheme (``q=1``) via
    :func:`~hidden_attractors.solvers.integer.efork_q1_step`.

    References
    ----------
    .. [Benettin1980] G. Benettin et al., "Lyapunov Characteristic Exponents
       for Smooth Dynamical Systems and for Hamiltonian Systems",
       Meccanica 15, 1980.
    .. [Wolf1985] A. Wolf et al., "Determining Lyapunov Exponents from a
       Time Series", Physica D 16, 1985.
    .. [Danca2018] M.-F. Danca & N. Kuznetsov, "Matlab Code for Lyapunov
       Exponents of Fractional-Order Systems", Int. J. Bifurcation Chaos
       28(5), 2018.

    Examples
    --------
    >>> import numpy as np
    >>> from hidden_attractors.analysis.lyapunov import integer_lyapunov_exponents
    >>> rhs = lambda x: np.array([-x[0], -2*x[1]])  # stable linear system
    >>> res = integer_lyapunov_exponents(rhs, None, np.array([1.0, 1.0]),
    ...                                  h=0.01, t_final=50.0)
    >>> res.status
    'ok'
    """
    # --- q-validation (F0 gate) ---
    if abs(float(q) - 1.0) > 1e-9:
        raise ValueError(
            f"integer_qr_benettin is valid only for q=1 (integer-order ODE); "
            f"received q={q}.  "
            "Use a fractional Lyapunov method for Caputo q<1 "
            "(e.g., fractional_variational_abm_qr — not yet implemented in F0)."
        )

    h_value = float(h)
    if h_value <= 0.0:
        raise ValueError("h must be positive.")
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1:
        raise ValueError("x0 must be one-dimensional.")
    n = x.size
    burn_steps = int(max(0, round(float(t_burn) / h_value)))
    total_steps = int(max(0, round(float(t_final) / h_value)))
    interval = max(1, int(reorthonormalize_every))
    jac = jacobian or (lambda state: finite_difference_jacobian(rhs, state, eps=jacobian_eps))

    for _ in range(burn_steps):
        x = efork_q1_step(rhs, x, h_value)
        if not np.all(np.isfinite(x)) or (div_threshold is not None and np.linalg.norm(x) >= float(div_threshold)):
            return LyapunovResult(
                np.full(n, np.nan), np.empty(0), np.empty((0, n)), "burn_diverged"
            )

    basis = np.eye(n, dtype=float)
    sums = np.zeros(n, dtype=float)
    times: list[float] = []
    convergence: list[np.ndarray] = []
    elapsed = 0.0
    status = "ok"
    for step in range(1, total_steps + 1):
        J = np.asarray(jac(x), dtype=float)
        basis = basis + h_value * J @ basis
        try:
            x = efork_q1_step(rhs, x, h_value)
        except (RuntimeError, ValueError, FloatingPointError, OverflowError):
            status = "solver_exception"
            break
        elapsed += h_value
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(basis)):
            status = "nonfinite_solution"
            break
        if div_threshold is not None and float(np.linalg.norm(x)) >= float(div_threshold):
            status = "diverged"
            break
        if step % interval == 0:
            qmat, rmat = np.linalg.qr(basis)
            diag = np.abs(np.diag(rmat))
            diag[diag <= 1.0e-300] = 1.0e-300
            sums += np.log(diag)
            basis = qmat
            times.append(elapsed)
            convergence.append(sums / max(elapsed, 1.0e-300))

    exponents = sums / max(elapsed, 1.0e-300) if elapsed > 0.0 else np.full(n, np.nan)
    return LyapunovResult(
        exponents=np.asarray(exponents, dtype=float),
        times=np.asarray(times, dtype=float),
        convergence=np.asarray(convergence, dtype=float) if convergence else np.empty((0, n), dtype=float),
        status=status,
        # F0 metadata
        method_id="integer_qr_benettin",
        derivative_model="integer",
        q=1.0,
        finite_time_local=True,
        jacobian_required=True,
        orthonormalization="qr",
        reference_ids=_INTEGER_QR_BENETTIN_REFS,
        methodological_warnings=_INTEGER_QR_BENETTIN_WARNINGS,
    )


# ---------------------------------------------------------------------------
# Alias / wrapper with explicit q-gate (F0 canonical entry point)
# ---------------------------------------------------------------------------

def integer_qr_benettin_lyapunov_exponents(
    rhs: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray] | None,
    x0: np.ndarray,
    *,
    h: float,
    t_final: float,
    t_burn: float = 0.0,
    reorthonormalize_every: int = 10,
    jacobian_eps: float = 1.0e-6,
    div_threshold: float | None = None,
    q: float = 1.0,
) -> LyapunovResult:
    """Canonical F0 entry point for integer-order QR-Benettin Lyapunov exponents.

    **Method identifier: ``integer_qr_benettin`` (F0 — frozen)**

    This is the explicitly named wrapper for the integer-order QR-Benettin
    algorithm.  It enforces ``q = 1.0`` strictly and populates all F0
    metadata fields in the returned :class:`LyapunovResult`.

    Calling this function with ``q ≠ 1.0`` always raises ``ValueError``.
    This is intentional: fractional Caputo Lyapunov spectra must use a
    dedicated extended-memory variational method, not this routine.

    Parameters
    ----------
    rhs : callable
        Vector field ``F(x) -> dxdt``, shape ``(n,) -> (n,)``.
    jacobian : callable or None
        Analytic Jacobian or ``None`` (finite differences used).
    x0 : np.ndarray, shape (n,)
        Initial state.
    h : float
        Integration step size (positive).
    t_final : float
        Total integration time.
    t_burn : float, default 0.0
        Burn-in time.
    reorthonormalize_every : int, default 10
        Steps between QR reorthonormalisations.
    jacobian_eps : float, default 1e-6
        Finite-difference step (when ``jacobian`` is ``None``).
    div_threshold : float or None, default None
        Divergence threshold on state norm.
    q : float, default 1.0
        Must equal 1.0.  Any other value raises ``ValueError``.

    Returns
    -------
    result : LyapunovResult
        Full result with ``method_id='integer_qr_benettin'`` and all F0
        metadata populated.

    Raises
    ------
    ValueError
        If ``abs(q - 1.0) > 1e-9``, *h* ≤ 0, or *x0* is not 1-D.

    Notes
    -----
    This routine is **not a validated Caputo fractional Lyapunov method**.
    It is restricted to **q = 1**.
    Fractional Caputo spectra require a dedicated extended-memory variational
    method (to be implemented in future phases).

    References
    ----------
    .. [Benettin1980] G. Benettin et al., Meccanica 15, 1980.
    .. [Wolf1985] A. Wolf et al., Physica D 16, 1985.
    .. [Danca2018] M.-F. Danca & N. Kuznetsov, Int. J. Bifurcation Chaos
       28(5), 2018.
    """
    # Strict q-gate for the canonical wrapper
    if abs(float(q) - 1.0) > 1e-9:
        raise ValueError(
            f"integer_qr_benettin is valid only for q=1 (integer-order ODE); "
            f"received q={q}.  "
            "Use a fractional Lyapunov method for Caputo q<1 "
            "(e.g., fractional_variational_abm_qr — not yet implemented in F0)."
        )
    return integer_lyapunov_exponents(
        rhs,
        jacobian,
        x0,
        h=h,
        t_final=t_final,
        t_burn=t_burn,
        reorthonormalize_every=reorthonormalize_every,
        jacobian_eps=jacobian_eps,
        div_threshold=div_threshold,
        q=1.0,
    )


# ---------------------------------------------------------------------------
# Private helper: defensively infer fractional order from a system object
# ---------------------------------------------------------------------------

def _infer_system_order(system: object) -> float | None:
    """Attempt to infer the fractional order *q* from a system object.

    Checks the following attributes in order, returning the first found:
    - ``system.q``
    - ``system.order``
    - ``system.fractional_order``
    - ``system.metadata.get('q')`` (if ``metadata`` is a mapping)
    - ``system.params.get('q')``   (if ``params``    is a mapping)

    Returns ``None`` if no order attribute is found (backward-compatible
    fallback: allow execution without raising).

    Returns
    -------
    float or None
        Inferred fractional order, or ``None`` if undetermined.
    """
    # Direct scalar attributes
    for attr in ("q", "order", "fractional_order"):
        try:
            val = getattr(system, attr, None)
        except Exception:  # noqa: BLE001
            val = None
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass

    # Mapping attributes: metadata / params
    for attr in ("metadata", "params"):
        try:
            mapping = getattr(system, attr, None)
        except Exception:  # noqa: BLE001
            mapping = None
        if mapping is not None:
            try:
                val = mapping.get("q")
                if val is not None:
                    return float(val)
            except (AttributeError, TypeError, ValueError):
                pass

    return None


# ---------------------------------------------------------------------------
# System-level convenience wrapper
# ---------------------------------------------------------------------------

def integer_system_lyapunov_exponents(
    system: ChaoticSystem,
    x0: np.ndarray,
    *,
    h: float,
    t_final: float,
    t_burn: float = 0.0,
    reorthonormalize_every: int = 10,
    jacobian_eps: float = 1.0e-6,
    div_threshold: float | None = None,
) -> LyapunovResult:
    """Estimate Lyapunov exponents for a registered integer-order system.

    **Method identifier: ``integer_qr_benettin`` (F0 — frozen)**

    Convenience wrapper around :func:`integer_lyapunov_exponents` that reads
    the RHS and analytic Jacobian directly from a
    :class:`~hidden_attractors.systems.base.ChaoticSystem`.

    **Methodological warning**

    This routine is **not a validated Caputo fractional Lyapunov method**.
    It is restricted to **q = 1** (integer-order ODE systems).
    Fractional Caputo spectra require a dedicated extended-memory variational
    method integrating the full original–variational system with Caputo memory
    (to be implemented in future phases: ``fractional_variational_abm_qr``).

    Parameters
    ----------
    system : ChaoticSystem
        Registered system (integer order).  If ``system.jacobian`` is set,
        the analytic Jacobian is used; otherwise finite differences are used.
    x0 : np.ndarray, shape (n,)
        Initial state.
    h : float
        Integration step size.
    t_final : float
        Total integration time.
    t_burn : float, default 0.0
        Burn-in time before accumulating exponents.
    reorthonormalize_every : int, default 10
        Steps between QR reorthonormalisations.
    jacobian_eps : float, default 1e-6
        Finite-difference step when no analytic Jacobian is available.
    div_threshold : float or None, default None
        Divergence threshold on the state norm.

    Returns
    -------
    result : LyapunovResult
        Exponent estimates, convergence history, status string, and F0
        method metadata (``method_id='integer_qr_benettin'``).

    Examples
    --------
    >>> import numpy as np
    >>> from hidden_attractors.systems import get_system
    >>> from hidden_attractors.analysis.lyapunov import integer_system_lyapunov_exponents
    >>> sys = get_system('chua-integer')
    >>> res = integer_system_lyapunov_exponents(
    ...     sys, np.array([0.1, 0.2, 0.3]), h=0.01, t_final=50.0)
    >>> res.status
    'ok'
    """

    # A2 — F0 closure: reject fractional systems defensively
    q_sys = _infer_system_order(system)
    if q_sys is not None and abs(q_sys - 1.0) > 1e-9:
        raise ValueError(
            f"integer_system_lyapunov_exponents uses integer_qr_benettin and is valid only "
            f"for q=1; the supplied system appears to have q={q_sys:.6g}. "
            "Use a fractional Lyapunov method for Caputo q<1."
        )

    # A1 — F1 closure: defensive attribute access for evaluate / jacobian
    if not callable(getattr(system, "evaluate", None)):
        raise ValueError(
            "integer_system_lyapunov_exponents: system must expose a callable "
            "evaluate(state) method."
        )
    rhs = lambda state: system.evaluate(state)
    system_jacobian_attr = getattr(system, "jacobian", None)
    if system_jacobian_attr is not None and callable(getattr(system, "jacobian_matrix", None)):
        jacobian: Callable[[np.ndarray], np.ndarray] | None = lambda state: system.jacobian_matrix(state)
    else:
        jacobian = None
    return integer_lyapunov_exponents(
        rhs,
        jacobian,
        np.asarray(x0, dtype=float),
        h=h,
        t_final=t_final,
        t_burn=t_burn,
        reorthonormalize_every=reorthonormalize_every,
        jacobian_eps=jacobian_eps,
        div_threshold=div_threshold,
        q=1.0,
    )


__all__ = [
    "LyapunovResult",
    "finite_difference_jacobian",
    "integer_lyapunov_exponents",
    "integer_qr_benettin_lyapunov_exponents",
    "integer_system_lyapunov_exponents",
]
