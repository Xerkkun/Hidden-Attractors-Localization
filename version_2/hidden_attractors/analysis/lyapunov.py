"""Reusable Lyapunov exponent estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..solvers.integer import efork_q1_step
from ..systems.base import ChaoticSystem


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
    """

    exponents: np.ndarray
    times: np.ndarray
    convergence: np.ndarray
    status: str


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
) -> LyapunovResult:
    """Estimate integer-order Lyapunov exponents by QR reorthonormalisation.

    Uses the Benettin et al. algorithm: integrate the variational equations
    for ``reorthonormalize_every`` steps, apply QR, accumulate log-diagonal
    sums, and divide by elapsed time.

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

    Returns
    -------
    result : LyapunovResult
        Exponent estimates, convergence history, and status string.

    Raises
    ------
    ValueError
        If *h* is not positive or *x0* is not one-dimensional.

    Notes
    -----
    The algorithm integrates a first-order Euler scheme (``q=1``) via
    :func:`~hidden_attractors.solvers.integer.efork_q1_step`.  For
    fractional systems the estimates are *approximate*.

    References
    ----------
    .. [1] G. Benettin et al., "Lyapunov Characteristic Exponents for
           Smooth Dynamical Systems and for Hamiltonian Systems",
           Meccanica 15, 1980.

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
            return LyapunovResult(np.full(n, np.nan), np.empty(0), np.empty((0, n)), "burn_diverged")

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
    )


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

    Convenience wrapper around :func:`integer_lyapunov_exponents` that reads
    the RHS and analytic Jacobian directly from a :class:`~hidden_attractors.systems.base.ChaoticSystem`.

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
        Exponent estimates, convergence history, and status string.

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

    rhs = lambda state: system.evaluate(state)
    jacobian = (lambda state: system.jacobian_matrix(state)) if system.jacobian is not None else None
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
    )


__all__ = [
    "LyapunovResult",
    "finite_difference_jacobian",
    "integer_lyapunov_exponents",
    "integer_system_lyapunov_exponents",
]
