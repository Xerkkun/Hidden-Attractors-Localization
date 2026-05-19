"""Fractional Chua model helpers.

The functions in this module define only the vector field and algebraic
equilibria.  Fractional memory and numerical integration live in solver/native
modules so that model equations stay independent from a particular numerical
contract.

Reference notes:
    - R. N. Madan and L. O. Chua, "Chaos in Chua's Circuit".
    - M. F. Danca, "Hidden Chaotic Attractors in Fractional-Order Systems".
    - D. Matignon, "Stability Results for Fractional Differential Equations
      with Applications to Control Processing" for fractional local stability.
    See ``docs/code_reference_map.md`` for the full method-to-code map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def normalize_chua_model(raw: str | int | None = "piecewise") -> str:
    """Normalize project aliases for the supported Chua nonlinearities."""

    if raw is None:
        return "piecewise"
    if not isinstance(raw, str):
        return "arctan" if int(raw) == 1 else "piecewise"
    text = raw.strip().lower().replace("-", "_")
    aliases = {
        "pwl": "piecewise",
        "nonsmooth": "piecewise",
        "non_smooth": "piecewise",
        "no_suave": "piecewise",
        "piecewise_linear": "piecewise",
        "tramos": "piecewise",
        "atan": "arctan",
        "arc_tan": "arctan",
        "smooth": "arctan",
        "suave": "arctan",
    }
    value = aliases.get(text, text)
    if value not in {"piecewise", "arctan"}:
        raise ValueError("model must be 'piecewise' or 'arctan'.")
    return value


@dataclass(frozen=True)
class ChuaParameters:
    """Parameters for the Chua systems used in the project.

    Equations:
        f(x) = m1*x + 0.5*(m0-m1)*(|x+1|-|x-1|)
        D^q x = alpha*(y - x - f(x))
        D^q y = x - y + z
        D^q z = -beta*y - gamma*z

    Validity warning:
        This dataclass does not encode the fractional order.  The order ``q``
        belongs to the numerical contract of a Caputo integration.
    """

    model: str = "piecewise"
    alpha: float = 8.4562
    beta: float = 12.0732
    gamma: float = 0.0052
    m0: float = -0.1768
    m1: float = -1.1468
    a1: float = 0.4
    a2: float = -1.5585
    rho: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", normalize_chua_model(self.model))
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive.")
        if self.rho <= 0.0:
            raise ValueError("rho must be positive.")


def chua_parameters(
    *,
    model: str = "piecewise",
    alpha: float = 8.4562,
    beta: float = 12.0732,
    gamma: float = 0.0052,
    m0: float = -0.1768,
    m1: float = -1.1468,
    a1: float = 0.4,
    a2: float = -1.5585,
    rho: float = 1.0,
) -> ChuaParameters:
    """Build an explicit Chua parameter object.

    Purpose:
        Provide a library-level replacement for legacy modules that selected
        the model and coefficients through environment variables.
    """

    return ChuaParameters(
        model=model,
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        m0=float(m0),
        m1=float(m1),
        a1=float(a1),
        a2=float(a2),
        rho=float(rho),
    )


def chua_piecewise_parameters() -> ChuaParameters:
    """Return the project-default piecewise Chua parameters."""

    return chua_parameters(model="piecewise")


def chua_arctan_parameters() -> ChuaParameters:
    """Return the project-default smooth arctan Chua parameters."""

    return chua_parameters(model="arctan")


def nonlinearity_piecewise(x: float, p: ChuaParameters) -> float:
    """Evaluate the piecewise-linear Chua nonlinearity."""

    return p.m1 * float(x) + 0.5 * (p.m0 - p.m1) * (abs(float(x) + 1.0) - abs(float(x) - 1.0))


def nonlinearity_arctan(x: float, p: ChuaParameters) -> float:
    """Evaluate the smooth arctan Chua nonlinearity used in the project."""

    return p.a1 * float(x) + p.a2 * float(np.arctan(p.rho * float(x)))


def nonlinearity_chua(x: float, p: ChuaParameters) -> float:
    """Evaluate the selected Chua nonlinearity."""

    if normalize_chua_model(p.model) == "arctan":
        return nonlinearity_arctan(x, p)
    return nonlinearity_piecewise(x, p)


def rhs_chua(state: np.ndarray, p: ChuaParameters | None = None) -> np.ndarray:
    """Evaluate the selected Chua vector field."""

    params = p or chua_piecewise_parameters()
    x, y, z = np.asarray(state, dtype=float)
    fx = params.alpha * (y - x - nonlinearity_chua(x, params))
    fy = x - y + z
    fz = -params.beta * y - params.gamma * z
    return np.array([fx, fy, fz], dtype=float)


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
    return rhs_chua(state, chua_parameters(
        model="piecewise",
        alpha=params.alpha,
        beta=params.beta,
        gamma=params.gamma,
        m0=params.m0,
        m1=params.m1,
        a1=params.a1,
        a2=params.a2,
        rho=params.rho,
    ))


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
