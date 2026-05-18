"""Fractional Chua model helpers.

The functions in this module define only the vector field and algebraic
equilibria.  Fractional memory and numerical integration live in solver/native
modules so that model equations stay independent from a particular numerical
contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ChuaParameters:
    """Parameters for the nonsmooth Chua system used in the project.

    Equations:
        f(x) = m1*x + 0.5*(m0-m1)*(|x+1|-|x-1|)
        D^q x = alpha*(y - x - f(x))
        D^q y = x - y + z
        D^q z = -beta*y - gamma*z

    Validity warning:
        This dataclass does not encode the fractional order.  The order ``q``
        belongs to the numerical contract of a Caputo integration.
    """

    alpha: float = 8.4562
    beta: float = 12.0732
    gamma: float = 0.0052
    m0: float = -0.1768
    m1: float = -1.1468


def chua_piecewise_parameters() -> ChuaParameters:
    """Return the project-default piecewise Chua parameters."""

    return ChuaParameters()


def nonlinearity_piecewise(x: float, p: ChuaParameters) -> float:
    """Evaluate the piecewise-linear Chua nonlinearity."""

    return p.m1 * float(x) + 0.5 * (p.m0 - p.m1) * (abs(float(x) + 1.0) - abs(float(x) - 1.0))


def rhs_piecewise(state: np.ndarray, p: ChuaParameters | None = None) -> np.ndarray:
    """Evaluate the integer-time vector field used by Caputo integrators.

    Purpose:
        Provide the right-hand side ``F(x)`` for ``^C D_t^q x = F(x)``.

    Parameters:
        ``state`` is ``(x, y, z)`` and ``p`` contains the Chua coefficients.

    Output:
        A length-3 NumPy array.
    """

    params = p or chua_piecewise_parameters()
    x, y, z = np.asarray(state, dtype=float)
    fx = params.alpha * (y - x - nonlinearity_piecewise(x, params))
    fy = x - y + z
    fz = -params.beta * y - params.gamma * z
    return np.array([fx, fy, fz], dtype=float)


def equilibria_piecewise(p: ChuaParameters | None = None) -> Dict[str, np.ndarray]:
    """Compute the three equilibria for the piecewise Chua model.

    Purpose:
        Hiddenness tests must sample neighborhoods of all equilibria.  This
        helper gives a single source for the equilibrium coordinates used by
        Python workflows.

    Validity warning:
        These are equilibria of the vector field; local fractional stability
        still requires a Matignon-type check on the Jacobian spectrum.
    """

    params = p or chua_piecewise_parameters()
    slope = -params.beta / (params.gamma + params.beta)
    den = params.m1 - slope
    if abs(den) < 1e-15:
        raise ValueError("Degenerate Chua parameters: cannot solve outer equilibria.")
    x_plus = -(params.m0 - params.m1) / den
    x_minus = (params.m0 - params.m1) / den

    def eq_from_x(x: float) -> np.ndarray:
        fx = nonlinearity_piecewise(x, params)
        return np.array([x, x + fx, fx], dtype=float)

    return {"E0": np.zeros(3, dtype=float), "E+": eq_from_x(x_plus), "E-": eq_from_x(x_minus)}
