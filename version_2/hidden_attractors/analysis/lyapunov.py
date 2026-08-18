"""Reusable Lyapunov exponent estimators.

Integer QR-Benettin method
==========================
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

Out of scope
------------
* NOT a validated Caputo fractional Lyapunov method.
* Does NOT handle fractional memory (q < 1).
* Does NOT certify chaos by itself.
* Does NOT certify hiddenness of attractors.
* chaos_verified / hidden_verified are NOT asserted here.

Fractional Caputo spectra (q < 1) require a dedicated memory-aware method.
They must not be computed with this integer-order routine.

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

from .._time_grid import exact_fixed_step_count
from ..solvers.integer import efork_q1_step
from ..systems.base import ChaoticSystem
from ._system_order import infer_system_order as _infer_system_order

# ---------------------------------------------------------------------------
# Canonical references for this method
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
        ``'integer_qr_benettin'`` for this module.
    derivative_model : str
        Derivative model: ``'integer'`` for q=1 ODE, ``'caputo'`` for
        fractional.
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
        Bibliographic references for the method.
    methodological_warnings : tuple[str, ...]
        Human-readable warnings about scope and limitations.
    """

    # Core result fields (original API — must remain first for backward compat)
    exponents: np.ndarray
    times: np.ndarray
    convergence: np.ndarray
    status: str

    # Method metadata fields (defaults preserve compatibility)
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

    **Method identifier: ``integer_qr_benettin``**

    Uses the Benettin/Wolf algorithm:

    1. Integrate the state ``X' = F(X)`` with the three-stage
       :func:`~hidden_attractors.solvers.integer.efork_q1_step`.
    2. Propagate the variational basis ``Φ' = J(X) Φ`` with an explicit
       first-order Euler update (memoryless).
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
    Fractional Caputo spectra require a dedicated memory-aware method; they
    must not be computed with this integer-order routine.

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
        Exponent estimates, convergence history, status string, and method
        method metadata (``method_id='integer_qr_benettin'``).

    Raises
    ------
    ValueError
        If *h* is not positive, *x0* is not one-dimensional, or *q* ≠ 1.0.

    Notes
    -----
    The state uses the three-stage
    :func:`~hidden_attractors.solvers.integer.efork_q1_step`; the variational
    basis uses a first-order explicit Euler update before QR.

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
    # --- numeric and q validation ---
    q_value = float(q)
    if not np.isfinite(q_value) or abs(q_value - 1.0) > 1e-9:
        raise ValueError(
            f"integer_qr_benettin is valid only for q=1 (integer-order ODE); "
            f"received q={q}.  "
            "Use a memory-aware fractional Lyapunov method for Caputo q<1."
        )

    h_value = float(h)
    t_final_value = float(t_final)
    t_burn_value = float(t_burn)
    eps_value = float(jacobian_eps)
    if not np.isfinite(h_value) or h_value <= 0.0:
        raise ValueError("h must be finite and positive.")
    if not np.isfinite(t_final_value) or t_final_value <= 0.0:
        raise ValueError("t_final must be finite and positive.")
    if not np.isfinite(t_burn_value) or t_burn_value < 0.0:
        raise ValueError("t_burn must be finite and non-negative.")
    if not np.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError("jacobian_eps must be finite and positive.")
    interval = int(reorthonormalize_every)
    if interval < 1 or float(reorthonormalize_every) != float(interval):
        raise ValueError("reorthonormalize_every must be a positive integer.")
    if div_threshold is not None:
        threshold_value = float(div_threshold)
        if not np.isfinite(threshold_value) or threshold_value <= 0.0:
            raise ValueError("div_threshold must be finite and positive.")
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a non-empty finite one-dimensional array.")
    n = x.size
    burn_steps = exact_fixed_step_count(
        h_value,
        t_burn_value,
        caller="integer_qr_benettin_lyapunov_exponents(t_burn)",
    )
    total_steps = exact_fixed_step_count(
        h_value,
        t_final_value,
        caller="integer_qr_benettin_lyapunov_exponents(t_final)",
    )
    if total_steps < 1:
        raise ValueError("t_final must span at least one integration step.")
    jac = jacobian or (lambda state: finite_difference_jacobian(rhs, state, eps=eps_value))

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
    qr_elapsed = 0.0
    steps_since_qr = 0
    status = "ok"
    for step in range(1, total_steps + 1):
        J = np.asarray(jac(x), dtype=float)
        if J.shape != (n, n) or not np.all(np.isfinite(J)):
            status = "invalid_jacobian"
            break
        next_basis = basis + h_value * J @ basis
        try:
            next_x = efork_q1_step(rhs, x, h_value)
        except (RuntimeError, ValueError, FloatingPointError, OverflowError):
            status = "solver_exception"
            break
        x = np.asarray(next_x, dtype=float)
        basis = next_basis
        elapsed += h_value
        steps_since_qr += 1
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
            qr_elapsed = elapsed
            steps_since_qr = 0
            times.append(elapsed)
            convergence.append(sums / max(elapsed, 1.0e-300))

    if steps_since_qr > 0 and np.all(np.isfinite(basis)):
        try:
            qmat, rmat = np.linalg.qr(basis)
        except np.linalg.LinAlgError:
            status = "qr_failure"
        else:
            diag = np.abs(np.diag(rmat))
            diag[diag <= 1.0e-300] = 1.0e-300
            sums += np.log(diag)
            basis = qmat
            qr_elapsed = elapsed
            times.append(elapsed)
            convergence.append(sums / max(elapsed, 1.0e-300))

    exponents = (
        sums / qr_elapsed if qr_elapsed > 0.0 else np.full(n, np.nan)
    )
    return LyapunovResult(
        exponents=np.asarray(exponents, dtype=float),
        times=np.asarray(times, dtype=float),
        convergence=np.asarray(convergence, dtype=float) if convergence else np.empty((0, n), dtype=float),
        status=status,
        # Method metadata
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
# Alias / wrapper with an explicit q-gate
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
    """Canonical entry point for integer-order QR-Benettin Lyapunov exponents.

    **Method identifier: ``integer_qr_benettin``**

    This is the explicitly named wrapper for the integer-order QR-Benettin
    algorithm.  It enforces ``q = 1.0`` strictly and populates all method
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
        Full result with ``method_id='integer_qr_benettin'`` and all method
        metadata populated.

    Raises
    ------
    ValueError
        If ``abs(q - 1.0) > 1e-9``, *h* ≤ 0, or *x0* is not 1-D.

    Notes
    -----
    This routine is **not a validated Caputo fractional Lyapunov method**.
    It is restricted to **q = 1**.
    Fractional Caputo spectra require a dedicated memory-aware method.

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
            "Use a memory-aware fractional Lyapunov method for Caputo q<1."
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

    **Method identifier: ``integer_qr_benettin``**

    Convenience wrapper around :func:`integer_lyapunov_exponents` that reads
    the RHS and analytic Jacobian directly from a
    :class:`~hidden_attractors.systems.base.ChaoticSystem`.

    **Methodological warning**

    This routine is **not a validated Caputo fractional Lyapunov method**.
    It is restricted to **q = 1** (integer-order ODE systems).
    Fractional Caputo spectra require a dedicated memory-aware method.

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
        Exponent estimates, convergence history, status string, and method
        method metadata (``method_id='integer_qr_benettin'``).

    Examples
    --------
    >>> import numpy as np
    >>> from hidden_attractors.systems import get_system
    >>> from hidden_attractors.analysis.lyapunov import integer_system_lyapunov_exponents
    >>> sys = get_system('chua-nonsmooth')
    >>> res = integer_system_lyapunov_exponents(
    ...     sys, np.array([0.1, 0.2, 0.3]), h=0.01, t_final=50.0)
    >>> res.status
    'ok'
    """

    # Reject fractional systems defensively.
    q_sys = _infer_system_order(system)
    if q_sys is not None and abs(q_sys - 1.0) > 1e-9:
        raise ValueError(
            f"integer_system_lyapunov_exponents uses integer_qr_benettin and is valid only "
            f"for q=1; the supplied system appears to have q={q_sys:.6g}. "
            "Use a fractional Lyapunov method for Caputo q<1."
        )

    # Defensive attribute access for evaluate / jacobian.
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
