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


def normalize_chua_model(raw: str | int | None = "nonsmooth") -> str:
    """Normalize project aliases for the supported Chua nonlinearities.

    Parameters
    ----------
    raw : str or int or None, default 'nonsmooth'
        Model name or integer code.  ``None`` and ``0`` map to
        ``'nonsmooth'``; ``1`` maps to ``'arctan'``.  Historical aliases
        ``'piecewise'`` and ``'pwl'`` map to ``'nonsmooth'``.

    Returns
    -------
    model : str
        Canonical model name: ``'nonsmooth'`` or ``'arctan'``.

    Raises
    ------
    ValueError
        If *raw* cannot be mapped to a supported nonlinearity.

    Examples
    --------
    >>> normalize_chua_model("pwl")
    'nonsmooth'
    >>> normalize_chua_model("atan")
    'arctan'
    >>> normalize_chua_model(None)
    'nonsmooth'
    """

    if raw is None:
        return "nonsmooth"
    if not isinstance(raw, str):
        return "arctan" if int(raw) == 1 else "nonsmooth"
    text = raw.strip().lower().replace("-", "_")
    aliases = {
        "piecewise": "nonsmooth",
        "pwl": "nonsmooth",
        "nonsmooth": "nonsmooth",
        "non_smooth": "nonsmooth",
        "no_suave": "nonsmooth",
        "piecewise_linear": "nonsmooth",
        "tramos": "nonsmooth",
        "atan": "arctan",
        "arc_tan": "arctan",
        "smooth": "arctan",
        "suave": "arctan",
    }
    value = aliases.get(text, text)
    if value not in {"nonsmooth", "arctan"}:
        raise ValueError("model must be 'nonsmooth' or 'arctan'.")
    return value


@api_tier(STABLE)
@dataclass(frozen=True)
class ChuaParameters:
    """Parameters for the Chua systems used in the project.

    Attributes
    ----------
    model : str, default 'nonsmooth'
        Nonlinearity family: ``'nonsmooth'`` or ``'arctan'``.
    alpha : float, default 8.4562
        Capacitor-ratio coefficient in the first state equation.
    beta : float, default 12.0732
        Inductance-ratio coefficient in the third state equation.
    gamma : float, default 0.0052
        Resistive damping coefficient in the third state equation.
    m0 : float, default -0.1768
        Inner-segment slope for the non-smooth nonlinearity.
    m1 : float, default -1.1468
        Outer-segment slope for the non-smooth nonlinearity.
    a1 : float, default 0.4
        Linear coefficient for the arctan nonlinearity.
    a2 : float, default -1.5585
        Arctan amplitude coefficient.
    rho : float, default 1.0
        Arctan frequency scaling parameter (must be positive).

    Notes
    -----
    The non-smooth system has a piecewise-linear characteristic:

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
    'nonsmooth'
    """

    model: str = "nonsmooth"
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
    model: str = "nonsmooth",
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
    model : str, default 'nonsmooth'
        Nonlinearity family.  See :func:`normalize_chua_model` for aliases.
    alpha : float, default 8.4562
        Capacitor-ratio coefficient.
    beta : float, default 12.0732
        Inductance-ratio coefficient.
    gamma : float, default 0.0052
        Resistive damping coefficient.
    m0 : float, default -0.1768
        Inner-segment slope (non-smooth model).
    m1 : float, default -1.1468
        Outer-segment slope (non-smooth model).
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
def chua_nonsmooth_parameters() -> ChuaParameters:
    """Return the project-default non-smooth Chua parameters.

    Returns
    -------
    params : ChuaParameters
        Frozen parameter object with ``model='nonsmooth'`` and the
        project-standard coefficients
        (α=8.4562, β=12.0732, γ=0.0052, m₀=−0.1768, m₁=−1.1468).

    Examples
    --------
    >>> from hidden_attractors.models.chua import chua_nonsmooth_parameters
    >>> p = chua_nonsmooth_parameters()
    >>> p.model
    'nonsmooth'
    >>> p.beta
    12.0732
    """

    return chua_parameters(model="nonsmooth")


def chua_arctan_parameters() -> ChuaParameters:
    """Return the project-default smooth arctan Chua parameters.

    Returns
    -------
    params : ChuaParameters
        Frozen parameter object with ``model='arctan'``.
    """

    return chua_parameters(model="arctan")


def nonlinearity_nonsmooth(x: float, p: ChuaParameters) -> float:
    """Evaluate the non-smooth Chua characteristic, which is linear by pieces.

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
    >>> from hidden_attractors.models.chua import nonlinearity_nonsmooth, chua_nonsmooth_parameters
    >>> p = chua_nonsmooth_parameters()
    >>> nonlinearity_nonsmooth(0.0, p)
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
        Result of :func:`nonlinearity_arctan` or :func:`nonlinearity_nonsmooth`.
    """

    if normalize_chua_model(p.model) == "arctan":
        return nonlinearity_arctan(x, p)
    return nonlinearity_nonsmooth(x, p)


def rhs_chua(state: np.ndarray, p: ChuaParameters | None = None) -> np.ndarray:
    """Evaluate the Chua vector field for the selected nonlinearity.

    Parameters
    ----------
    state : np.ndarray, shape (3,)
        Current state ``(x, y, z)``.
    p : ChuaParameters or None, default None
        System coefficients.  Defaults to :func:`chua_nonsmooth_parameters`.

    Returns
    -------
    dxdt : np.ndarray, shape (3,)
        Right-hand side ``(ẋ, ẏ, ż)``.
    """

    params = p or chua_nonsmooth_parameters()
    x, y, z = np.asarray(state, dtype=float)
    fx = params.alpha * (y - x - nonlinearity_chua(x, params))
    fy = x - y + z
    fz = -params.beta * y - params.gamma * z
    return np.array([fx, fy, fz], dtype=float)


@api_tier(STABLE)
def rhs_nonsmooth(state: np.ndarray, p: ChuaParameters | None = None) -> np.ndarray:
    """Evaluate the non-smooth Chua vector field for Caputo integrators.

    This is the right-hand side ``F(x)`` in ``^C D_t^q x = F(x)``
    restricted to the non-smooth nonlinearity regardless of ``p.model``.

    Parameters
    ----------
    state : np.ndarray, shape (3,)
        Current state ``(x, y, z)``.
    p : ChuaParameters or None, default None
        System coefficients.  Defaults to :func:`chua_nonsmooth_parameters`.
        Arctan coefficients (``a1``, ``a2``, ``rho``) are ignored.

    Returns
    -------
    dxdt : np.ndarray, shape (3,)
        Right-hand side ``(ẋ, ẏ, ż)`` using the non-smooth nonlinearity.

    Examples
    --------
    >>> import numpy as np
    >>> from hidden_attractors.models.chua import rhs_nonsmooth, chua_nonsmooth_parameters
    >>> p = chua_nonsmooth_parameters()
    >>> rhs_nonsmooth(np.array([0.0, 0.0, 0.0]), p)
    array([0., 0., 0.])
    """

    params = p or chua_nonsmooth_parameters()
    return rhs_chua(state, chua_parameters(
        model="nonsmooth",
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
def jacobian_nonsmooth(state: np.ndarray, p: ChuaParameters | None = None) -> np.ndarray:
    """Evaluate the regional Jacobian of the non-smooth Chua vector field.

    Parameters
    ----------
    state : np.ndarray, shape (3,)
        State at which the regional slope is selected.
    p : ChuaParameters or None, default None
        System coefficients. Defaults to :func:`chua_nonsmooth_parameters`.

    Returns
    -------
    jacobian : np.ndarray, shape (3, 3)
        Jacobian using ``m0`` in the inner region ``|x| < 1`` and ``m1``
        in the outer region ``|x| > 1``.

    Raises
    ------
    ValueError
        If ``x`` lies on a switching surface, where the non-smooth
        vector field does not have a unique classical Jacobian.
    """

    params = p or chua_nonsmooth_parameters()
    x = float(np.asarray(state, dtype=float)[0])
    if np.isclose(abs(x), 1.0, atol=1.0e-14, rtol=0.0):
        raise ValueError("non-smooth Jacobian is undefined at x = +/-1.")
    slope = params.m0 if abs(x) < 1.0 else params.m1
    return np.array(
        [
            [-params.alpha * (1.0 + slope), params.alpha, 0.0],
            [1.0, -1.0, 1.0],
            [0.0, -params.beta, -params.gamma],
        ],
        dtype=float,
    )


@api_tier(STABLE)
def equilibria_nonsmooth(p: ChuaParameters | None = None) -> Dict[str, np.ndarray]:
    """Compute the three equilibria of the non-smooth Chua model.

    The origin ``E0`` is always an equilibrium.  The outer equilibria
    ``E+`` and ``E−`` exist when the denominator ``m1 − slope`` is
    non-degenerate.

    Parameters
    ----------
    p : ChuaParameters or None, default None
        System coefficients.  Defaults to :func:`chua_nonsmooth_parameters`.

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
    >>> from hidden_attractors.models.chua import equilibria_nonsmooth
    >>> eq = equilibria_nonsmooth()
    >>> list(eq.keys())
    ['E0', 'E+', 'E-']
    >>> eq['E0']
    array([0., 0., 0.])
    """

    params = p or chua_nonsmooth_parameters()
    slope = -params.beta / (params.gamma + params.beta)
    den = params.m1 - slope
    if abs(den) < 1e-15:
        raise ValueError("Degenerate Chua parameters: cannot solve outer equilibria.")
    x_plus = -(params.m0 - params.m1) / den
    x_minus = (params.m0 - params.m1) / den

    def eq_from_x(x: float) -> np.ndarray:
        fx = nonlinearity_nonsmooth(x, params)
        return np.array([x, x + fx, fx], dtype=float)

    return {"E0": np.zeros(3, dtype=float), "E+": eq_from_x(x_plus), "E-": eq_from_x(x_minus)}


# Compatibility aliases for notebooks and recorded runs created before
# ``nonsmooth`` became the canonical public name.
def chua_piecewise_parameters() -> ChuaParameters:
    return chua_nonsmooth_parameters()


def nonlinearity_piecewise(x: float, p: ChuaParameters) -> float:
    return nonlinearity_nonsmooth(x, p)


def rhs_piecewise(state: np.ndarray, p: ChuaParameters | None = None) -> np.ndarray:
    return rhs_nonsmooth(state, p)


def jacobian_piecewise(state: np.ndarray, p: ChuaParameters | None = None) -> np.ndarray:
    return jacobian_nonsmooth(state, p)


def equilibria_piecewise(p: ChuaParameters | None = None) -> Dict[str, np.ndarray]:
    return equilibria_nonsmooth(p)
