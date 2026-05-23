"""Published three-stage EFORK reference implementation.

This module follows the stage order and history evaluation in Ghoreishi,
Ghaffari, and Saad (2023).  It is intentionally kept as a small Python
reference implementation for numerical-method validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class EFORK3Coefficients:
    """Coefficients of the explicit three-stage fractional RK method."""

    alpha: float
    c2: float
    c3: float
    a21: float
    a31: float
    a32: float
    w1: float
    w2: float
    w3: float


def efork3_coefficients(alpha: float) -> EFORK3Coefficients:
    """Return three-stage EFORK coefficients for ``0 < alpha < 1``."""

    q = float(alpha)
    if not 0.0 < q < 1.0:
        raise ValueError("The published Caputo EFORK reference requires 0 < alpha < 1.")
    g1 = math.gamma(1.0 + q)
    g2 = math.gamma(1.0 + 2.0 * q)
    g3 = math.gamma(1.0 + 3.0 * q)
    denominator = 2.0 * g2 * g2 - g3
    return EFORK3Coefficients(
        alpha=q,
        c2=(1.0 / (2.0 * g1)) ** (1.0 / q),
        c3=(1.0 / (4.0 * g1)) ** (1.0 / q),
        a21=1.0 / (2.0 * g1 * g1),
        a31=(g1 * g1 * g2 + 2.0 * g2 * g2 - g3) / (4.0 * g1 * g1 * denominator),
        a32=-g2 / (4.0 * denominator),
        w1=(8.0 * g1**3 * g2**2 - 6.0 * g1**3 * g3 + g2 * g3) / (g1 * g2 * g3),
        w2=2.0 * g1 * g1 * (4.0 * g2 * g2 - g3) / (g2 * g3),
        w3=-8.0 * g1 * g1 * denominator / (g2 * g3),
    )


def _history_term(
    t_eval: float,
    times: np.ndarray,
    states: np.ndarray,
    n: int,
    alpha: float,
    h: float,
) -> np.ndarray:
    if n == 0:
        return np.zeros(states.shape[1], dtype=float)
    increments = states[1 : n + 1] - states[:n]
    powers = (t_eval - times[:n]) ** (1.0 - alpha) - (t_eval - times[1 : n + 1]) ** (1.0 - alpha)
    return (increments.T @ powers) / (h * math.gamma(2.0 - alpha))


def efork3_caputo_integrate(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    *,
    alpha: float,
    h: float,
    t_final: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a Caputo problem using the published EFORK-3 formula.

    ``rhs`` supplies the right-hand side of ``D_C^alpha y = rhs(t, y)``.
    The returned arrays are the time grid and state values.
    """

    step = float(h)
    final_time = float(t_final)
    if step <= 0.0 or final_time < 0.0:
        raise ValueError("h must be positive and t_final must be nonnegative.")
    n_steps = int(round(final_time / step))
    if not math.isclose(n_steps * step, final_time, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("t_final must be an integer multiple of h.")
    coeff = efork3_coefficients(alpha)
    state0 = np.asarray(y0, dtype=float)
    if state0.ndim != 1:
        raise ValueError("y0 must be one-dimensional.")

    times = np.linspace(0.0, final_time, n_steps + 1)
    states = np.zeros((n_steps + 1, state0.size), dtype=float)
    states[0] = state0
    h_alpha = step**coeff.alpha
    for n in range(n_steps):
        tn = times[n]
        yn = states[n]

        def modified_rhs(t_eval: float, state: np.ndarray) -> np.ndarray:
            force = np.asarray(rhs(t_eval, state), dtype=float)
            return force - _history_term(t_eval, times, states, n, coeff.alpha, step)

        k1 = h_alpha * modified_rhs(tn, yn)
        k2 = h_alpha * modified_rhs(tn + coeff.c2 * step, yn + coeff.a21 * k1)
        k3 = h_alpha * modified_rhs(
            tn + coeff.c3 * step,
            yn + coeff.a31 * k1 + coeff.a32 * k2,
        )
        states[n + 1] = yn + coeff.w1 * k1 + coeff.w2 * k2 + coeff.w3 * k3
    return times, states


__all__ = ["EFORK3Coefficients", "efork3_coefficients", "efork3_caputo_integrate"]
