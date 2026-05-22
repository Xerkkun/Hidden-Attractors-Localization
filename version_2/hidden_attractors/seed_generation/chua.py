"""Chua-specific seed generation: describing functions and harmonic seeds.

Stability: experimental
    Functions here are the library version of the Nyquist/DF mathematics
    that previously lived only in legacy scripts.  The API is considered
    useful and tested.  New parameters may be added as additional Chua
    families are supported.

Contents
--------
Transfer functions
    :func:`transfer_function`, :func:`chua_matrices`, :func:`psi_sigma`

Classical describing function
    :func:`describing_function`, :func:`is_describing_gain_compatible`,
    :func:`solve_amplitude_from_gain`

Machado family
    :func:`machado_describing_function`, :func:`solve_machado_amplitude_from_gain`

Biased / Fourier
    :func:`fourier_coefficients_psi`, :func:`biased_describing_function`,
    :func:`reconstruct_biased_lure_seed`

Seed construction
    :func:`build_fractional_seed`, :func:`find_harmonic_seed`,
    :func:`find_omega_gain_candidates`, :func:`format_seed_report`
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..models.chua import ChuaParameters, chua_piecewise_parameters, normalize_chua_model
from .core import (
    BiasedHarmonicSeed,
    HarmonicSeed,
    _bisect_root,
    _solve_scalar_gain,
    complex_dtype,
    fractional_iomega_power,
    real_dtype,
    validate_fractional_order,
)


# ── Chua Lur'e decomposition ─────────────────────────────────────────────────

def chua_gain(params: ChuaParameters) -> float:
    """Return the saturation gain ``m0 - m1`` for the piecewise Chua model.

    Parameters
    ----------
    params : ChuaParameters
        Chua circuit parameters; must use the piecewise model.

    Returns
    -------
    gain : float
        Outer-segment slope minus inner-segment slope.
    """

    return float(params.m0 - params.m1)


def chua_matrices(
    params: ChuaParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the Lur'e decomposition matrices ``(P, b, c)`` for Chua.

    The decomposition is ``D^q x = P x + b \u03c8(c^T x)`` where ``\u03c8 = psi_sigma``.

    Parameters
    ----------
    params : ChuaParameters
        Chua circuit parameters.

    Returns
    -------
    pmat : np.ndarray, shape (3, 3)
        Linear part ``P`` of the Lur'e form.
    qvec : np.ndarray, shape (3,)
        Input direction ``b``.
    rvec : np.ndarray, shape (3,)
        Output direction ``c`` (feedback selector).
    """

    model = normalize_chua_model(params.model)
    base_slope = params.a1 if model == "arctan" else params.m1
    pmat = np.array(
        [
            [-params.alpha * (1.0 + base_slope), params.alpha, 0.0],
            [1.0, -1.0, 1.0],
            [0.0, -params.beta, -params.gamma],
        ],
        dtype=real_dtype,
    )
    qvec = np.array([-params.alpha, 0.0, 0.0], dtype=real_dtype)
    rvec = np.array([1.0, 0.0, 0.0], dtype=real_dtype)
    return pmat, qvec, rvec


def psi_sigma(sigma: float, params: ChuaParameters) -> float:
    """Evaluate the nonlinear residual ``\u03c8(\u03c3)`` used by the Lur'e form.

    Parameters
    ----------
    sigma : float
        Feedback coordinate ``\u03c3 = c^T x``.
    params : ChuaParameters
        Chua circuit parameters; selects piecewise or arctan nonlinearity.

    Returns
    -------
    psi : float
        Nonlinear residual: either a clipped saturation or an arctan term.
    """

    value = float(sigma)
    if normalize_chua_model(params.model) == "arctan":
        return float(params.a2 * np.arctan(params.rho * value))
    return float(chua_gain(params) * np.clip(value, -1.0, 1.0))


def build_linearized_matrix(params: ChuaParameters, gain: float) -> np.ndarray:
    """Return the linearised matrix ``P + k b c^T`` for describing-function gain ``k``.

    Parameters
    ----------
    params : ChuaParameters
        Chua parameters used to compute ``P``, ``b``, ``c``.
    gain : float
        Describing-function gain ``k``.

    Returns
    -------
    A_lin : np.ndarray, shape (3, 3)
        Linearised system matrix.
    """

    pmat, qvec, rvec = chua_matrices(params)
    return pmat + float(gain) * np.outer(qvec, rvec)


# ── Transfer function and frequency scan ─────────────────────────────────────

def transfer_function(omega: float, q: float, params: ChuaParameters) -> complex:
    """Return the fractional Chua transfer function ``W_q(i\u03c9) = c^T (P - (i\u03c9)^q I)^{-1} b``.

    Parameters
    ----------
    omega : float
        Angular frequency (must be positive and finite).
    q : float
        Caputo fractional order.
    params : ChuaParameters
        Chua circuit parameters.

    Returns
    -------
    W : complex
        Transfer function value at ``s = (i omega)^q``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``P - (i omega)^q I`` is singular at the given frequency.
    """

    pmat, qvec, rvec = chua_matrices(params)
    matrix = pmat.astype(complex_dtype) - fractional_iomega_power(omega, q) * np.eye(
        3, dtype=complex_dtype
    )
    value = (
        rvec.astype(complex_dtype).reshape(1, -1)
        @ np.linalg.inv(matrix)
        @ qvec.astype(complex_dtype).reshape(-1, 1)
    )[0, 0]
    return complex_dtype(value)


def find_omega_gain_candidates(
    q: float,
    params: ChuaParameters | None = None,
    *,
    wmin: float = 1.0e-4,
    wmax: float = 10.0,
    nscan: int = 20_000,
    compatible_only: bool = True,
) -> list[tuple[float, float]]:
    """Find all ``(\u03c9, k)`` pairs where ``Im(W_q(i\u03c9)) = 0``.

    Parameters
    ----------
    q : float
        Caputo fractional order.
    params : ChuaParameters or None, default None
        Chua parameters.  Defaults to :func:`~hidden_attractors.models.chua.chua_piecewise_parameters`.
    wmin : float, default 1e-4
        Minimum angular frequency to scan.
    wmax : float, default 10.0
        Maximum angular frequency to scan.
    nscan : int, default 20_000
        Number of frequency samples in the initial scan.
    compatible_only : bool, default True
        If ``True``, only pairs where ``k`` is compatible with the DF
        amplitude model are returned.

    Returns
    -------
    pairs : list[tuple[float, float]]
        Sorted list of ``(omega, gain)`` tuples with
        ``k = -1 / Re(W_q(i omega))``.

    Raises
    ------
    ValueError
        If ``wmin >= wmax`` or ``nscan < 8``.

    Examples
    --------
    >>> from hidden_attractors.seed_generation.chua import find_omega_gain_candidates
    >>> pairs = find_omega_gain_candidates(0.9998)
    >>> len(pairs) >= 1
    True
    """

    q_value = validate_fractional_order(q)
    p = params or chua_piecewise_parameters()
    if wmin <= 0.0 or wmax <= wmin:
        raise ValueError("expected 0 < wmin < wmax.")
    if int(nscan) < 8:
        raise ValueError("nscan must be at least 8.")

    ws = np.linspace(float(wmin), float(wmax), int(nscan))
    vals = np.array(
        [np.imag(transfer_function(w, q_value, p)) for w in ws], dtype=float
    )
    roots: list[float] = []
    for i in range(len(ws) - 1):
        f1 = vals[i]
        f2 = vals[i + 1]
        if not np.isfinite(f1) or not np.isfinite(f2):
            continue
        if f1 == 0.0:
            roots.append(float(ws[i]))
        elif f1 * f2 < 0.0:
            roots.append(
                float(
                    _bisect_root(
                        lambda w: np.imag(transfer_function(w, q_value, p)),
                        ws[i],
                        ws[i + 1],
                    )
                )
            )

    unique_roots: list[float] = []
    for root in sorted(roots):
        if not unique_roots or abs(root - unique_roots[-1]) > 1.0e-7:
            unique_roots.append(root)

    pairs: list[tuple[float, float]] = []
    for omega in unique_roots:
        response = transfer_function(omega, q_value, p)
        re_value = float(np.real(response))
        if abs(re_value) < 1.0e-12:
            continue
        gain = -1.0 / re_value
        if compatible_only and not is_describing_gain_compatible(gain, p):
            continue
        pairs.append((float(omega), float(gain)))
    return sorted(pairs, key=lambda item: item[1])


# ── Classical describing function ─────────────────────────────────────────────

def is_describing_gain_compatible(gain: float, params: ChuaParameters) -> bool:
    """Return whether *gain* is reachable by the classical describing function.

    Parameters
    ----------
    gain : float
        Describing-function gain ``k``.
    params : ChuaParameters
        Chua parameters; selects the piecewise or arctan DF model.

    Returns
    -------
    compatible : bool
        ``True`` if there exists an amplitude ``A > 0`` with ``N(A) = k``.
    """

    k = float(gain)
    if normalize_chua_model(params.model) == "arctan":
        return np.sign(k) == np.sign(params.a2) and 0.0 < abs(k) < abs(params.a2) * params.rho
    sat_gain = chua_gain(params)
    return np.sign(k) == np.sign(sat_gain) and 0.0 < abs(k) <= abs(sat_gain) + 1.0e-10


def describing_function(amplitude: float, params: ChuaParameters) -> float:
    """Classical first-harmonic describing function ``N(A)``.

    Parameters
    ----------
    amplitude : float
        Oscillation amplitude ``A > 0``.
    params : ChuaParameters
        Selects the piecewise or arctan nonlinearity.

    Returns
    -------
    N_A : float
        First-harmonic DF value: ``N(A)``.

    Raises
    ------
    ValueError
        If *amplitude* is not positive and finite.

    Notes
    -----
    For the piecewise model ``N(A) = k_sat`` when ``A <= 1`` (linear regime)
    and a smoothly decreasing function of ``A`` for ``A > 1``.
    For the arctan model a different closed-form is used; see
    Gelb & Vander Velde (1968).
    """

    amp = float(amplitude)
    if amp <= 0.0 or not np.isfinite(amp):
        raise ValueError("amplitude must be positive and finite.")
    if normalize_chua_model(params.model) == "arctan":
        return float(
            params.a2
            * 2.0
            * (np.sqrt(1.0 + (params.rho * amp) ** 2) - 1.0)
            / (params.rho * amp * amp)
        )
    gain = chua_gain(params)
    if amp <= 1.0:
        return gain
    return float(
        (2.0 * gain / np.pi)
        * (np.arcsin(1.0 / amp) + np.sqrt(amp * amp - 1.0) / (amp * amp))
    )


def solve_amplitude_from_gain(
    gain: float,
    params: ChuaParameters,
    *,
    amin: float = 1.0 + 1.0e-9,
    amax: float = 100.0,
    nscan: int = 20_000,
) -> float:
    """Solve ``N(A) = gain`` for the classical describing function.

    Parameters
    ----------
    gain : float
        Target describing-function gain.
    params : ChuaParameters
        Chua parameters.
    amin : float, default 1.0+1e-9
        Lower amplitude bound for grid + bisection.
    amax : float, default 100.0
        Upper amplitude bound.
    nscan : int, default 20_000
        Number of points in the initial amplitude grid.

    Returns
    -------
    amplitude : float
        Positive amplitude ``A`` satisfying ``N(A) = gain``.

    Raises
    ------
    RuntimeError
        If *gain* is not compatible with the DF model or if no root is
        found in ``[amin, amax]``.
    """

    k = float(gain)
    if not is_describing_gain_compatible(k, params):
        raise RuntimeError(
            "gain is not compatible with the selected Chua describing function."
        )
    if normalize_chua_model(params.model) == "arctan":
        amplitude_sq = 4.0 * params.a2 * (params.a2 * params.rho - k) / (k * k * params.rho)
        if amplitude_sq <= 0.0:
            raise RuntimeError("computed arctan amplitude is not real positive.")
        return float(np.sqrt(amplitude_sq))
    sat_gain = chua_gain(params)
    if abs(k - sat_gain) < 1.0e-10:
        return 1.0
    return _solve_scalar_gain(
        k, lambda a: describing_function(a, params), amin=amin, amax=amax, nscan=nscan
    )


# ── Machado family ────────────────────────────────────────────────────────────

def machado_describing_function(
    amplitude: float, params: ChuaParameters, mu: float
) -> float:
    """Auxiliary Machado-family describing function ``N_\u03bc(A) = N(A)^\u03bc``.

    Parameters
    ----------
    amplitude : float
        Oscillation amplitude ``A > 0``.
    params : ChuaParameters
        Must use the piecewise Chua model.
    mu : float
        Machado exponent (must be positive and finite).

    Returns
    -------
    N_mu : float
        ``N(A)^mu``.

    Raises
    ------
    ValueError
        If *amplitude* or *mu* are invalid, or if *params* uses the arctan
        model (Machado DF is piecewise-only).
    ValueError
        If ``N(A) <= 0`` (real branch not defined).
    """

    exponent = float(mu)
    if normalize_chua_model(params.model) != "piecewise":
        raise ValueError(
            "Machado describing-function seeds are implemented for the piecewise model."
        )
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("mu must be positive and finite.")
    base = describing_function(amplitude, params)
    if base <= 0.0:
        raise ValueError("N(a) must be positive for the real Machado branch.")
    return float(base**exponent)


def solve_machado_amplitude_from_gain(
    gain: float,
    params: ChuaParameters,
    mu: float,
    *,
    amin: float = 1.0 + 1.0e-9,
    amax: float = 100.0,
    nscan: int = 20_000,
) -> float:
    """Solve ``N(A)^\u03bc = gain`` for the auxiliary Machado family.

    Parameters
    ----------
    gain : float
        Target Machado describing-function gain (must be positive).
    params : ChuaParameters
        Must use the piecewise Chua model.
    mu : float
        Machado exponent (positive and finite).
    amin : float, default 1.0+1e-9
        Lower amplitude bound.
    amax : float, default 100.0
        Upper amplitude bound.
    nscan : int, default 20_000
        Grid size for bisection search.

    Returns
    -------
    amplitude : float
        Positive amplitude satisfying ``N(A)^mu = gain``.

    Raises
    ------
    RuntimeError
        If the model is not piecewise, if no amplitude brackets the root,
        or if *gain* is not positive.
    """

    k = float(gain)
    exponent = float(mu)
    if normalize_chua_model(params.model) != "piecewise":
        raise RuntimeError("Machado amplitude solving applies only to the piecewise model.")
    if exponent <= 0.0 or not np.isfinite(exponent):
        raise ValueError("mu must be positive and finite.")
    if k <= 0.0:
        raise RuntimeError("Machado gain must be positive.")
    target = float(k ** (1.0 / exponent))
    sat_gain = chua_gain(params)
    if abs(target - sat_gain) < 1.0e-10:
        return 1.0
    if not (0.0 < target < sat_gain):
        raise RuntimeError("Machado gain has no real saturated-branch amplitude.")
    return _solve_scalar_gain(
        target,
        lambda a: describing_function(a, params),
        amin=amin,
        amax=amax,
        nscan=nscan,
    )


# ── Biased / Fourier coefficients ─────────────────────────────────────────────

def fourier_coefficients_psi(
    amplitude: float,
    sigma0: float,
    params: ChuaParameters,
    *,
    harmonics: int = 10,
    n_quad: int = 4096,
) -> dict[str, object]:
    """Compute Fourier coefficients of ``\u03c8(\u03c30 + A cos(\u03b8))``.

    Parameters
    ----------
    amplitude : float
        Oscillation amplitude ``A > 0``.
    sigma0 : float
        DC bias of the feedback coordinate.
    params : ChuaParameters
        Selects the piecewise or arctan nonlinearity.
    harmonics : int, default 10
        Number of harmonics to compute (1 through *harmonics*).
    n_quad : int, default 4096
        Number of quadrature points.  Must satisfy ``n >= max(64, 8*harmonics)``.

    Returns
    -------
    coeffs : dict
        Keys: ``'amplitude'``, ``'sigma0'``, ``'harmonics'``, ``'n_quad'``,
        ``'y_mean'``, ``'coefficients'``.
        ``coefficients[k]`` is a dict with ``'a'``, ``'b'``, and
        ``'Y' = a_k - i b_k`` (legacy convention).

    Raises
    ------
    ValueError
        If *amplitude* is not positive, *harmonics* < 1, or *n_quad* is
        too small.

    Notes
    -----
    The convention ``Y_k = a_k - i b_k`` matches the legacy Chua scripts
    and the project reports.
    """

    amp = float(amplitude)
    center = float(sigma0)
    kmax = int(harmonics)
    n = int(n_quad)
    if amp <= 0.0 or not np.isfinite(amp):
        raise ValueError("amplitude must be positive and finite.")
    if kmax < 1:
        raise ValueError("harmonics must be at least 1.")
    if n < max(64, 8 * kmax):
        raise ValueError("n_quad is too small for the requested number of harmonics.")

    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=real_dtype)
    values = np.array(
        [psi_sigma(center + amp * np.cos(th), params) for th in theta], dtype=real_dtype
    )
    coeffs: dict[int, dict[str, object]] = {}
    for k in range(1, kmax + 1):
        ak = 2.0 * float(np.mean(values * np.cos(k * theta)))
        bk = 2.0 * float(np.mean(values * np.sin(k * theta)))
        coeffs[k] = {"a": ak, "b": bk, "Y": complex_dtype(ak - 1j * bk)}
    return {
        "amplitude": amp,
        "sigma0": center,
        "harmonics": kmax,
        "n_quad": n,
        "y_mean": float(np.mean(values)),
        "coefficients": coeffs,
    }


def biased_describing_function(
    amplitude: float,
    sigma0: float,
    params: ChuaParameters,
    *,
    harmonics: int = 10,
    n_quad: int = 4096,
) -> complex:
    """Return the biased describing function ``N(A, \u03c30) = Y_1(A, \u03c30) / A``.

    Parameters
    ----------
    amplitude : float
        Oscillation amplitude ``A > 0``.
    sigma0 : float
        DC bias of the feedback coordinate.
    params : ChuaParameters
        Chua parameters.
    harmonics : int, default 10
        Passed to :func:`fourier_coefficients_psi`.
    n_quad : int, default 4096
        Quadrature points for coefficient computation.

    Returns
    -------
    N_biased : complex
        First-harmonic biased DF value ``Y_1 / A``.
    """

    data = fourier_coefficients_psi(
        amplitude, sigma0, params, harmonics=max(1, int(harmonics)), n_quad=n_quad
    )
    y1 = complex(data["coefficients"][1]["Y"])  # type: ignore[index]
    return complex_dtype(y1 / float(amplitude))


def reconstruct_biased_lure_seed(
    *,
    q: float,
    params: ChuaParameters | None = None,
    amplitude: float,
    sigma0: float,
    omega: float,
    theta: float = 0.0,
    harmonics: int = 10,
    n_quad: int = 4096,
) -> BiasedHarmonicSeed:
    """Reconstruct a biased Lur'e seed from DC and first-harmonic balance equations.

    Solves two least-squares systems: one for the mean state and one for the
    complex first-harmonic amplitude vector.

    Parameters
    ----------
    q : float
        Caputo fractional order.
    params : ChuaParameters or None, default None
        Chua parameters.  Defaults to :func:`~hidden_attractors.models.chua.chua_piecewise_parameters`.
    amplitude : float
        Oscillation amplitude ``A``.
    sigma0 : float
        DC bias of the feedback coordinate.
    omega : float
        Angular frequency of the first harmonic.
    theta : float, default 0.0
        Initial phase angle (radians).
    harmonics : int, default 10
        Number of harmonics for Fourier computation.
    n_quad : int, default 4096
        Quadrature points.

    Returns
    -------
    seed : BiasedHarmonicSeed
        Frozen dataclass with the reconstructed initial condition and all
        intermediate components.
    """

    q_value = validate_fractional_order(q)
    p = params or chua_piecewise_parameters()
    fourier = fourier_coefficients_psi(
        amplitude, sigma0, p, harmonics=max(1, int(harmonics)), n_quad=n_quad
    )
    y1 = complex(fourier["coefficients"][1]["Y"])  # type: ignore[index]
    y_mean = float(fourier["y_mean"])
    pmat, qvec, rvec = chua_matrices(p)

    lhs_dc = np.vstack([pmat, rvec.reshape(1, -1)]).astype(real_dtype)
    rhs_dc = np.concatenate([-qvec * y_mean, np.array([float(sigma0)], dtype=real_dtype)])
    mean_state, *_ = np.linalg.lstsq(lhs_dc, rhs_dc, rcond=None)

    lam = fractional_iomega_power(omega, q_value)
    lhs_h = np.vstack(
        [
            lam * np.eye(3, dtype=complex_dtype) - pmat.astype(complex_dtype),
            rvec.astype(complex_dtype).reshape(1, -1),
        ]
    )
    rhs_h = np.concatenate(
        [
            qvec.astype(complex_dtype) * y1,
            np.array([complex(float(amplitude), 0.0)], dtype=complex_dtype),
        ]
    )
    harmonic_vector, *_ = np.linalg.lstsq(lhs_h, rhs_h, rcond=None)
    seed = np.asarray(mean_state, dtype=real_dtype) + np.real(
        harmonic_vector * np.exp(1j * float(theta))
    )
    return BiasedHarmonicSeed(
        seed=seed.astype(real_dtype),
        mean_state=np.asarray(mean_state, dtype=real_dtype),
        harmonic_vector=harmonic_vector.astype(complex_dtype),
        fourier=fourier,
        amplitude=float(amplitude),
        sigma0=float(sigma0),
        omega=float(omega),
        theta=float(theta),
    )


# ── Harmonic seed construction ────────────────────────────────────────────────

def build_fractional_seed(
    q: float,
    params: ChuaParameters,
    omega: float,
    gain: float,
    amplitude: float,
    *,
    theta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, complex]:
    """Build the harmonic initial condition for one DF branch.

    Computes the eigenvector of the linearised matrix closest to
    ``(i omega)^q``, normalised so that ``c^T v = 1``, then scales
    by *amplitude*.

    Parameters
    ----------
    q : float
        Caputo fractional order.
    params : ChuaParameters
        Chua parameters.
    omega : float
        Angular frequency from the DF scan.
    gain : float
        Describing-function gain ``k``.
    amplitude : float
        Oscillation amplitude ``A``.
    theta : float, default 0.0
        Initial phase (radians).

    Returns
    -------
    seed : np.ndarray, shape (3,)
        Real initial condition.
    eigenvector : np.ndarray, shape (3,), complex
        Normalised dominant eigenvector.
    matched_eigenvalue : complex
        Eigenvalue closest to ``(i omega)^q``.

    Raises
    ------
    RuntimeError
        If the eigenvector cannot be normalised (first component is zero).
    """

    q_value = validate_fractional_order(q)
    linearized = build_linearized_matrix(params, gain).astype(complex_dtype)
    lam = fractional_iomega_power(omega, q_value)
    eigvals, eigvecs = np.linalg.eig(linearized)
    idx = int(np.argmin(np.abs(eigvals - lam)))
    vector = eigvecs[:, idx]
    if abs(vector[0]) < 1.0e-14:
        raise RuntimeError("cannot normalize eigenvector with r^T v = 1.")
    vector = vector / vector[0]
    seed = float(amplitude) * np.real(vector * np.exp(1j * float(theta)))
    return seed.astype(real_dtype), vector.astype(complex_dtype), complex(eigvals[idx])


def find_harmonic_seed(
    *,
    q: float,
    params: ChuaParameters | None = None,
    branch_index: int = 0,
    method: Literal["classic", "machado"] = "classic",
    mu: float = 1.0,
    theta: float = 0.0,
    wmin: float = 1.0e-4,
    wmax: float = 10.0,
    nscan: int = 20_000,
) -> HarmonicSeed:
    """Locate a DF branch and return a finite harmonic seed.

    Parameters
    ----------
    q : float
        Caputo fractional order.
    params : ChuaParameters or None, default None
        Chua parameters.  Defaults to the canonical piecewise model.
    branch_index : int, default 0
        Index into the sorted ``(omega, gain)`` candidate list.
    method : {'classic', 'machado'}, default 'classic'
        ``'classic'``: classical first-harmonic DF.
        ``'machado'``: Machado-family auxiliary DF (changes amplitude only).
    mu : float, default 1.0
        Machado exponent; ignored when *method* is ``'classic'``.
    theta : float, default 0.0
        Initial phase angle (radians).
    wmin : float, default 1e-4
        Minimum angular frequency for the scan.
    wmax : float, default 10.0
        Maximum angular frequency for the scan.
    nscan : int, default 20_000
        Number of scan points.

    Returns
    -------
    seed : HarmonicSeed
        Frozen dataclass with the harmonic initial condition and DF metadata.

    Raises
    ------
    RuntimeError
        If no omega/gain candidate is found.
    IndexError
        If *branch_index* is outside the candidate list.
    ValueError
        If *method* is not ``'classic'`` or ``'machado'``.

    Notes
    -----
    ``method='machado'`` changes only the amplitude relation; it does not
    constitute a hiddenness verification.

    Examples
    --------
    >>> from hidden_attractors.seed_generation.chua import find_harmonic_seed
    >>> seed = find_harmonic_seed(q=0.9998)
    >>> seed.amplitude > 0
    True
    """

    p = params or chua_piecewise_parameters()
    pairs = find_omega_gain_candidates(
        q, p, wmin=wmin, wmax=wmax, nscan=nscan, compatible_only=(method == "classic")
    )
    if not pairs:
        raise RuntimeError("no omega/gain candidate was found.")
    index = int(branch_index)
    if index < 0 or index >= len(pairs):
        raise IndexError("branch_index is outside the available candidate list.")
    omega, gain = pairs[index]
    if method == "machado":
        amplitude = solve_machado_amplitude_from_gain(gain, p, mu)
    elif method == "classic":
        amplitude = solve_amplitude_from_gain(gain, p)
    else:
        raise ValueError("method must be 'classic' or 'machado'.")
    seed, vector, matched = build_fractional_seed(q, p, omega, gain, amplitude, theta=theta)
    return HarmonicSeed(
        seed=seed,
        eigenvector=vector,
        matched_eigenvalue=matched,
        omega=float(omega),
        gain=float(gain),
        amplitude=float(amplitude),
        branch_index=index,
        method=method,
        mu=float(mu) if method == "machado" else None,
    )


def format_seed_report(seed: HarmonicSeed) -> dict[str, object]:
    """Return a JSON-serialisable summary of a harmonic seed.

    Parameters
    ----------
    seed : HarmonicSeed
        Harmonic seed produced by :func:`find_harmonic_seed`.

    Returns
    -------
    report : dict[str, object]
        Dictionary with keys ``'seed'``, ``'omega'``, ``'gain'``,
        ``'amplitude'``, ``'branch_index'``, ``'method'``, ``'mu'``,
        and ``'matched_eigenvalue'`` (as ``[real, imag]``).
    """

    return {
        "seed": seed.seed.tolist(),
        "omega": seed.omega,
        "gain": seed.gain,
        "amplitude": seed.amplitude,
        "branch_index": seed.branch_index,
        "method": seed.method,
        "mu": seed.mu,
        "matched_eigenvalue": [
            float(np.real(seed.matched_eigenvalue)),
            float(np.imag(seed.matched_eigenvalue)),
        ],
    }


__all__ = [
    "biased_describing_function",
    "build_fractional_seed",
    "build_linearized_matrix",
    "chua_gain",
    "chua_matrices",
    "describing_function",
    "find_harmonic_seed",
    "find_omega_gain_candidates",
    "format_seed_report",
    "fourier_coefficients_psi",
    "is_describing_gain_compatible",
    "machado_describing_function",
    "psi_sigma",
    "reconstruct_biased_lure_seed",
    "solve_amplitude_from_gain",
    "solve_machado_amplitude_from_gain",
    "transfer_function",
]
