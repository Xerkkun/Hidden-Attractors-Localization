"""Fractional Chua model helpers.

Stability: stable
    Vector field, nonlinearity, and equilibrium helpers for the Chua system.
    Signatures and return types are fixed.

The functions in this module define only the vector field and algebraic
equilibria.  Fractional memory and numerical integration live in solver/native
modules so that model equations stay independent from a particular numerical
contract.

References
----------
.. [1] R. N. Madan and L. O. Chua, "Chaos in Chua's Circuit", 1993.
.. [2] M. F. Danca, "Hidden Chaotic Attractors in Fractional-Order Systems",
       Nonlinear Dynamics, 2017.
.. [3] D. Matignon, "Stability Results for Fractional Differential Equations
       with Applications to Control Processing", IMACS, 1996.

See ``docs/code_reference_map.md`` for the full method-to-code map.

Reference notes: see the ``References`` section above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .._stability import STABLE, api_tier


def normalize_chua_model(raw: str | int | None = "piecewise") -> str:
    """Normalize project aliases for the supported Chua nonlinearities.

    Parameters
    ----------
    raw : str or int or None, default 'piecewise'
        Model name or integer code.  ``None`` and ``0`` map to
        ``'piecewise'``; ``1`` maps to ``'arctan'``.  String aliases
        ``'pwl'``, ``'nonsmooth'``, ``'atan'``, ``'smooth'`` (and their
        Spanish variants) are accepted.

    Returns
    -------
    model : str
        Canonical model name: ``'piecewise'`` or ``'arctan'``.

    Raises
    ------
    ValueError
        If *raw* cannot be mapped to a supported nonlinearity.

    Examples
    --------
    >>> normalize_chua_model("pwl")
    'piecewise'
    >>> normalize_chua_model("atan")
    'arctan'
    >>> normalize_chua_model(None)
    'piecewise'
    """

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


@api_tier(STABLE)
@dataclass(frozen=True)
class ChuaParameters:
    """Parameters for the Chua systems used in the project.

    Attributes
    ----------
    model : str, default 'piecewise'
        Nonlinearity family: ``'piecewise'`` (PWL) or ``'arctan'`` (smooth).
    alpha : float, default 8.4562
        Capacitor-ratio coefficient in the first state equation.
    beta : float, default 12.0732
        Inductance-ratio coefficient in the third state equation.
    gamma : float, default 0.0052
        Resistive damping coefficient in the third state equation.
    m0 : float, default -0.1768
        Inner-segment slope for the piecewise nonlinearity.
    m1 : float, default -1.1468
        Outer-segment slope for the piecewise nonlinearity.
    a1 : float, default 0.4
        Linear coefficient for the arctan nonlinearity.
    a2 : float, default -1.5585
        Arctan amplitude coefficient.
    rho : float, default 1.0
        Arctan frequency scaling parameter (must be positive).

    Notes
    -----
    The piecewise system is:

    .. math::

        f(x) = m_1 x + \\tfrac{1}{2}(m_0-m_1)(|x+1|-|x-1|)

        {}^C D_t^q x = \\alpha(y - x - f(x)), \\quad
        {}^C D_t^q y = x - y + z, \\quad
        {}^C D_t^q z = -\\beta y - \\gamma z.

    The fractional order *q* is not stored here; it belongs to the
    numerical integration contract.

    Examples
    --------
    >>> from hidden_attractors.models.chua import ChuaParameters
    >>> p = ChuaParameters()
    >>> p.alpha
    8.4562
    >>> p.model
    'piecewise'
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


@api_tier(STABLE)
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
    """Build a :class:`ChuaParameters` object with explicit coefficients.

    All arguments are keyword-only to prevent positional confusion between
    the many scalar coefficients.

    Parameters
    ----------
    model : str, default 'piecewise'
        Nonlinearity family.  See :func:`normalize_chua_model` for aliases.
    alpha : float, default 8.4562
        Capacitor-ratio coefficient.
    beta : float, default 12.0732
        Inductance-ratio coefficient.
    gamma : float, default 0.0052
        Resistive damping coefficient.
    m0 : float, default -0.1768
        Inner-segment slope (piecewise model).
    m1 : float, default -1.1468
        Outer-segment slope (piecewise model).
    a1 : float, default 0.4
        Linear coefficient (arctan model).
    a2 : float, default -1.5585
        Arctan amplitude (arctan model).
    rho : float, default 1.0
        Arctan frequency scaling (arctan model).

    Returns
    -------
    params : ChuaParameters
        Frozen parameter object.

    Examples
    --------
    >>> from hidden_attractors.models.chua import chua_parameters
    >>> p = chua_parameters(alpha=9.0, beta=14.286)
    >>> p.alpha
    9.0
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


@api_tier(STABLE)
def chua_piecewise_parameters() -> ChuaParameters:
    """Return the project-default piecewise Chua parameters.

    Returns
    -------
    params : ChuaParameters
        Frozen parameter object with ``model='piecewise'`` and the
        project-standard coefficients
        (α=8.4562, β=12.0732, γ=0.0052, m₀=−0.1768, m₁=−1.1468).

    Examples
    --------
    >>> from hidden_attractors.models.chua import chua_piecewise_parameters
    >>> p = chua_piecewise_parameters()
    >>> p.model
    'piecewise'
    >>> p.beta
    12.0732
    """

    return chua_parameters(model="piecewise")


def chua_arctan_parameters() -> ChuaParameters:
    """Return the project-default smooth arctan Chua parameters.

    Returns
    -------
    params : ChuaParameters
        Frozen parameter object with ``model='arctan'``.
    """

    return chua_parameters(model="arctan")


def nonlinearity_piecewise(x: float, p: ChuaParameters) -> float:
    """Evaluate the piecewise-linear Chua nonlinearity.

    Parameters
    ----------
    x : float
        Scalar feedback coordinate.
    p : ChuaParameters
        System coefficients.

    Returns
    -------
    value : float
        ``m1·x + 0.5·(m0−m1)·(|x+1|−|x−1|)``.

    Examples
    --------
    >>> from hidden_attractors.models.chua import nonlinearity_piecewise, chua_piecewise_parameters
    >>> p = chua_piecewise_parameters()
    >>> nonlinearity_piecewise(0.0, p)
    0.0
    """

    return p.m1 * float(x) + 0.5 * (p.m0 - p.m1) * (abs(float(x) + 1.0) - abs(float(x) - 1.0))


def nonlinearity_arctan(x: float, p: ChuaParameters) -> float:
    """Evaluate the smooth arctan Chua nonlinearity.

    Parameters
    ----------
    x : float
        Scalar feedback coordinate.
    p : ChuaParameters
        System coefficients (uses ``a1``, ``a2``, ``rho``).

    Returns
    -------
    value : float
        ``a1·x + a2·arctan(ρ·x)``.

    Examples
    --------
    >>> from hidden_attractors.models.chua import nonlinearity_arctan, chua_parameters
    >>> p = chua_parameters(model='arctan')
    >>> nonlinearity_arctan(0.0, p)
    0.0
    """

    return p.a1 * float(x) + p.a2 * float(np.arctan(p.rho * float(x)))


def nonlinearity_chua(x: float, p: ChuaParameters) -> float:
    """Evaluate the nonlinearity selected by ``p.model``.

    Parameters
    ----------
    x : float
        Scalar feedback coordinate.
    p : ChuaParameters
        System coefficients; the ``model`` field selects the branch.

    Returns
    -------
    value : float
        Result of :func:`nonlinearity_arctan` or :func:`nonlinearity_piecewise`.
    """

    if normalize_chua_model(p.model) == "arctan":
        return nonlinearity_arctan(x, p)
    return nonlinearity_piecewise(x, p)


def rhs_chua(state: np.ndarray, p: ChuaParameters | None = None) -> np.ndarray:
    """Evaluate the Chua vector field for the selected nonlinearity.

    Parameters
    ----------
    state : np.ndarray, shape (3,)
        Current state ``(x, y, z)``.
    p : ChuaParameters or None, default None
        System coefficients.  Defaults to :func:`chua_piecewise_parameters`.

    Returns
    -------
    dxdt : np.ndarray, shape (3,)
        Right-hand side ``(ẋ, ẏ, ż)``.
    """

    params = p or chua_piecewise_parameters()
    x, y, z = np.asarray(state, dtype=float)
    fx = params.alpha * (y - x - nonlinearity_chua(x, params))
    fy = x - y + z
    fz = -params.beta * y - params.gamma * z
    return np.array([fx, fy, fz], dtype=float)


@api_tier(STABLE)
def rhs_piecewise(state: np.ndarray, p: ChuaParameters | None = None) -> np.ndarray:
    """Evaluate the piecewise Chua vector field for Caputo integrators.

    This is the right-hand side ``F(x)`` in ``^C D_t^q x = F(x)``
    restricted to the piecewise nonlinearity regardless of ``p.model``.

    Parameters
    ----------
    state : np.ndarray, shape (3,)
        Current state ``(x, y, z)``.
    p : ChuaParameters or None, default None
        System coefficients.  Defaults to :func:`chua_piecewise_parameters`.
        Non-piecewise coefficients (``a1``, ``a2``, ``rho``) are ignored.

    Returns
    -------
    dxdt : np.ndarray, shape (3,)
        Right-hand side ``(ẋ, ẏ, ż)`` using the piecewise nonlinearity.

    Examples
    --------
    >>> import numpy as np
    >>> from hidden_attractors.models.chua import rhs_piecewise, chua_piecewise_parameters
    >>> p = chua_piecewise_parameters()
    >>> rhs_piecewise(np.array([0.0, 0.0, 0.0]), p)
    array([0., 0., 0.])
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


@api_tier(STABLE)
def equilibria_piecewise(p: ChuaParameters | None = None) -> Dict[str, np.ndarray]:
    """Compute the three equilibria of the piecewise Chua model.

    The origin ``E0`` is always an equilibrium.  The outer equilibria
    ``E+`` and ``E−`` exist when the denominator ``m1 − slope`` is
    non-degenerate.

    Parameters
    ----------
    p : ChuaParameters or None, default None
        System coefficients.  Defaults to :func:`chua_piecewise_parameters`.

    Returns
    -------
    equilibria : dict[str, np.ndarray]
        Dictionary with keys ``'E0'``, ``'E+'``, ``'E-'``, each mapping
        to a state vector of shape ``(3,)``.

    Raises
    ------
    ValueError
        If ``m1 − slope`` is numerically zero (degenerate parameters).

    Notes
    -----
    These are equilibria of the *vector field*; local fractional stability
    requires a Matignon-type check on the Jacobian eigenvalue arguments.

    Examples
    --------
    >>> from hidden_attractors.models.chua import equilibria_piecewise
    >>> eq = equilibria_piecewise()
    >>> list(eq.keys())
    ['E0', 'E+', 'E-']
    >>> eq['E0']
    array([0., 0., 0.])
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
