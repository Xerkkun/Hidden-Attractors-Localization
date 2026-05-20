"""Reusable Lyapunov exponent estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..solvers.integer import efork_q1_step
from ..systems.base import ChaoticSystem


@dataclass(frozen=True)
class LyapunovResult:
    """Finite-time Lyapunov exponent estimate."""

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
    """Central finite-difference Jacobian for systems without analytic one."""

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
    """Estimate integer-order Lyapunov exponents by QR reorthonormalization."""

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
    """Estimate Lyapunov exponents for a registered integer-order system."""

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
