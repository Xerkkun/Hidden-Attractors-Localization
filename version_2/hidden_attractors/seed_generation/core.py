"""Shared primitives for harmonic-balance seed generation.

Stability: experimental
    Shared utilities used by both the Chua-specific and generic Lur'e seed
    generators.  The public dataclasses and mathematical helpers here are
    relatively stable, but may gain new fields as new system families are
    added.

Contents
--------
- :class:`HarmonicSeed`       — seed from the classical/Machado DF branch.
- :class:`BiasedHarmonicSeed` — seed from a biased first-harmonic approximation.
- :func:`validate_fractional_order` — validates ``0 < q <= 1``.
- :func:`fractional_iomega_power`   — ``(i omega)^q`` on the principal branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


real_dtype = np.float64
complex_dtype = np.complex128


# ── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HarmonicSeed:
    """Numerical seed produced by the describing-function construction.

    Attributes
    ----------
    seed : np.ndarray, shape (n,)
        State-space initial condition derived from the DF branch.
    eigenvector : np.ndarray, shape (n,), complex
        Dominant eigenvector of the linearised matrix at ``(omega, gain)``.
    matched_eigenvalue : complex
        Eigenvalue of the linearised matrix closest to ``(i omega)^q``.
    omega : float
        Angular frequency satisfying ``Im(W_q(i omega)) = 0``.
    gain : float
        Describing-function gain ``k = -1 / Re(W_q(i omega))``.
    amplitude : float
        Oscillation amplitude ``A`` satisfying ``N(A) = k``.
    branch_index : int
        Index into the sorted ``(omega, gain)`` candidate list.
    method : str, default 'classic'
        ``'classic'`` for the standard DF; ``'machado'`` for the auxiliary
        Machado-family DF.
    mu : float or None
        Machado exponent; ``None`` for classical seeds.
    """

    seed: np.ndarray
    eigenvector: np.ndarray
    matched_eigenvalue: complex
    omega: float
    gain: float
    amplitude: float
    branch_index: int
    method: str = "classic"
    mu: float | None = None


@dataclass(frozen=True)
class BiasedHarmonicSeed:
    """Seed reconstructed from a biased first-harmonic approximation.

    Attributes
    ----------
    seed : np.ndarray, shape (n,)
        State-space initial condition (mean + first harmonic).
    mean_state : np.ndarray, shape (n,)
        DC component of the biased solution.
    harmonic_vector : np.ndarray, shape (n,), complex
        First-harmonic amplitude vector in the Lur'e state space.
    fourier : dict
        Fourier coefficient dictionary from :func:`fourier_coefficients_psi`.
    amplitude : float
        Oscillation amplitude ``A``.
    sigma0 : float
        DC bias value of the feedback coordinate ``sigma = c^T x``.
    omega : float
        Angular frequency used to compute the harmonic vector.
    theta : float, default 0.0
        Initial phase angle (radians) applied to the harmonic.
    """

    seed: np.ndarray
    mean_state: np.ndarray
    harmonic_vector: np.ndarray
    fourier: dict[str, object]
    amplitude: float
    sigma0: float
    omega: float
    theta: float = 0.0


# ── Mathematical primitives ─────────────────────────────────────────────────

def validate_fractional_order(q: float) -> float:
    """Validate a Caputo fractional order and return it as a Python float.

    Parameters
    ----------
    q : float
        Candidate fractional order.

    Returns
    -------
    q_valid : float
        The same value cast to ``float`` if ``0 < q <= 1``.

    Raises
    ------
    ValueError
        If *q* is not finite or does not satisfy ``0 < q <= 1``.

    Examples
    --------
    >>> from hidden_attractors.seed_generation.core import validate_fractional_order
    >>> validate_fractional_order(0.9998)
    0.9998
    >>> import pytest
    >>> validate_fractional_order(0.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: fractional order q must satisfy 0 < q <= 1.
    """

    value = float(q)
    if not np.isfinite(value) or not (0.0 < value <= 1.0):
        raise ValueError("fractional order q must satisfy 0 < q <= 1.")
    return value


def fractional_iomega_power(omega: float, q: float) -> complex:
    """Return ``(i \u03c9)^q`` evaluated on the principal branch.

    Parameters
    ----------
    omega : float
        Angular frequency; must be positive and finite.
    q : float
        Caputo fractional order; must satisfy ``0 < q <= 1``.

    Returns
    -------
    result : complex
        ``omega^q * exp(i pi q / 2)`` as a :class:`complex`.

    Raises
    ------
    ValueError
        If *omega* is not positive and finite, or if *q* fails
        :func:`validate_fractional_order`.

    Examples
    --------
    >>> import numpy as np
    >>> from hidden_attractors.seed_generation.core import fractional_iomega_power
    >>> result = fractional_iomega_power(1.0, 1.0)  # integer case: (i*1)^1 = i
    >>> abs(result.real) < 1e-12 and abs(result.imag - 1.0) < 1e-12
    True
    """

    w = float(omega)
    if not np.isfinite(w) or w <= 0.0:
        raise ValueError("omega must be positive and finite.")
    q_value = validate_fractional_order(q)
    return complex_dtype((w**q_value) * np.exp(1j * np.pi * q_value / 2.0))


# ── Private shared helpers ──────────────────────────────────────────────────

def _bisect_root(
    func,
    left: float,
    right: float,
    *,
    maxiter: int = 100,
    xtol: float = 1.0e-12,
) -> float:
    """Small dependency-free scalar bisection helper."""

    lo = float(left)
    hi = float(right)
    flo = float(func(lo))
    fhi = float(func(hi))
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError("root is not bracketed.")
    for _ in range(int(maxiter)):
        mid = 0.5 * (lo + hi)
        fmid = float(func(mid))
        if abs(fmid) <= xtol or abs(hi - lo) <= xtol:
            return mid
        if flo * fmid <= 0.0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)


def _solve_scalar_gain(
    target_gain: float,
    evaluator,
    *,
    amin: float,
    amax: float,
    nscan: int,
) -> float:
    """Grid + bisection solver for ``evaluator(a) == target_gain``."""

    grid = np.linspace(float(amin), float(amax), int(nscan))
    values = np.array([evaluator(a) - target_gain for a in grid], dtype=float)
    for i in range(len(grid) - 1):
        if values[i] == 0.0:
            return float(grid[i])
        if values[i] * values[i + 1] < 0.0:
            return float(
                _bisect_root(
                    lambda a: evaluator(a) - target_gain,
                    grid[i],
                    grid[i + 1],
                    maxiter=500,
                )
            )
    raise RuntimeError("No amplitude solved the requested describing-function gain.")


__all__ = [
    "BiasedHarmonicSeed",
    "HarmonicSeed",
    "complex_dtype",
    "fractional_iomega_power",
    "real_dtype",
    "validate_fractional_order",
]
