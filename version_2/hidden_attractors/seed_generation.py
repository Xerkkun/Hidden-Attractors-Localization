"""Harmonic-balance seed generation for fractional Chua workflows.

This module is the library-facing version of the reusable mathematics that
used to live only in legacy scripts.  All numerical choices are explicit
arguments; no ``HIDDEN_ATTRACTORS_*`` environment variables are read here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .models.chua import ChuaParameters, chua_piecewise_parameters, normalize_chua_model


real_dtype = np.float64
complex_dtype = np.complex128


def _bisect_root(func, left: float, right: float, *, maxiter: int = 100, xtol: float = 1.0e-12) -> float:
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


@dataclass(frozen=True)
class HarmonicSeed:
    """Numerical seed produced by the describing-function construction."""

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
    """Seed reconstructed from a biased first-harmonic approximation."""

    seed: np.ndarray
    mean_state: np.ndarray
    harmonic_vector: np.ndarray
    fourier: dict[str, object]
    amplitude: float
    sigma0: float
    omega: float
    theta: float = 0.0


def validate_fractional_order(q: float) -> float:
    """Validate a Caputo fractional order."""

    value = float(q)
    if not np.isfinite(value) or not (0.0 < value <= 1.0):
        raise ValueError("fractional order q must satisfy 0 < q <= 1.")
    return value


def fractional_iomega_power(omega: float, q: float) -> complex:
    """Return ``(i*omega)^q`` on the principal branch."""

    w = float(omega)
    if not np.isfinite(w) or w <= 0.0:
        raise ValueError("omega must be positive and finite.")
    q_value = validate_fractional_order(q)
    return complex_dtype((w**q_value) * np.exp(1j * np.pi * q_value / 2.0))


def chua_gain(params: ChuaParameters) -> float:
    """Return the saturation gain ``m0 - m1`` for the piecewise model."""

    return float(params.m0 - params.m1)


def chua_matrices(params: ChuaParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``P, qvec, r`` for ``D^q x = P x + qvec psi(r^T x)``."""

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
    """Evaluate the nonlinear residual ``psi`` used by the Lur'e form."""

    value = float(sigma)
    if normalize_chua_model(params.model) == "arctan":
        return float(params.a2 * np.arctan(params.rho * value))
    return float(chua_gain(params) * np.clip(value, -1.0, 1.0))


def build_linearized_matrix(params: ChuaParameters, gain: float) -> np.ndarray:
    """Return ``P + k qvec r^T`` for a describing-function gain ``k``."""

    pmat, qvec, rvec = chua_matrices(params)
    return pmat + float(gain) * np.outer(qvec, rvec)


def transfer_function(omega: float, q: float, params: ChuaParameters) -> complex:
    """Return ``r^T (P - (i omega)^q I)^(-1) qvec``."""

    pmat, qvec, rvec = chua_matrices(params)
    matrix = pmat.astype(complex_dtype) - fractional_iomega_power(omega, q) * np.eye(3, dtype=complex_dtype)
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
    """Find roots of ``Im(W_q(i omega)) = 0`` and their gains.

    The gain convention is ``k = -1 / Re(W_q(i omega))``, matching the legacy
    Chua scripts and the project reports.
    """

    q_value = validate_fractional_order(q)
    p = params or chua_piecewise_parameters()
    if wmin <= 0.0 or wmax <= wmin:
        raise ValueError("expected 0 < wmin < wmax.")
    if int(nscan) < 8:
        raise ValueError("nscan must be at least 8.")

    ws = np.linspace(float(wmin), float(wmax), int(nscan))
    vals = np.array([np.imag(transfer_function(w, q_value, p)) for w in ws], dtype=float)
    roots: list[float] = []
    for i in range(len(ws) - 1):
        f1 = vals[i]
        f2 = vals[i + 1]
        if not np.isfinite(f1) or not np.isfinite(f2):
            continue
        if f1 == 0.0:
            roots.append(float(ws[i]))
        elif f1 * f2 < 0.0:
            roots.append(float(_bisect_root(lambda w: np.imag(transfer_function(w, q_value, p)), ws[i], ws[i + 1])))

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


def is_describing_gain_compatible(gain: float, params: ChuaParameters) -> bool:
    """Return whether a classical describing-function amplitude can match ``gain``."""

    k = float(gain)
    if normalize_chua_model(params.model) == "arctan":
        return np.sign(k) == np.sign(params.a2) and 0.0 < abs(k) < abs(params.a2) * params.rho
    sat_gain = chua_gain(params)
    return np.sign(k) == np.sign(sat_gain) and 0.0 < abs(k) <= abs(sat_gain) + 1.0e-10


def describing_function(amplitude: float, params: ChuaParameters) -> float:
    """Classical first-harmonic describing function for the selected nonlinearity."""

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
    return float((2.0 * gain / np.pi) * (np.arcsin(1.0 / amp) + np.sqrt(amp * amp - 1.0) / (amp * amp)))


def machado_describing_function(amplitude: float, params: ChuaParameters, mu: float) -> float:
    """Auxiliary Machado-family describing function ``N_mu(a)=N(a)^mu``."""

    exponent = float(mu)
    if normalize_chua_model(params.model) != "piecewise":
        raise ValueError("Machado describing-function seeds are implemented for the piecewise model.")
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("mu must be positive and finite.")
    base = describing_function(amplitude, params)
    if base <= 0.0:
        raise ValueError("N(a) must be positive for the real Machado branch.")
    return float(base**exponent)


def fourier_coefficients_psi(
    amplitude: float,
    sigma0: float,
    params: ChuaParameters,
    *,
    harmonics: int = 10,
    n_quad: int = 4096,
) -> dict[str, object]:
    """Compute Fourier coefficients of ``psi(sigma0 + A cos(theta))``.

    The return value uses the legacy convention ``Y_k = a_k - i b_k`` and
    records the mean nonlinear response separately as ``y_mean``.
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
    values = np.array([psi_sigma(center + amp * np.cos(th), params) for th in theta], dtype=real_dtype)
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
    """Return the biased describing function ``N(A,sigma0)=Y_1/A``."""

    data = fourier_coefficients_psi(
        amplitude,
        sigma0,
        params,
        harmonics=max(1, int(harmonics)),
        n_quad=n_quad,
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
    """Reconstruct a biased Lur'e seed from DC and first-harmonic equations."""

    q_value = validate_fractional_order(q)
    p = params or chua_piecewise_parameters()
    fourier = fourier_coefficients_psi(
        amplitude,
        sigma0,
        p,
        harmonics=max(1, int(harmonics)),
        n_quad=n_quad,
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
    seed = np.asarray(mean_state, dtype=real_dtype) + np.real(harmonic_vector * np.exp(1j * float(theta)))
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


def _solve_scalar_gain(
    target_gain: float,
    evaluator,
    *,
    amin: float,
    amax: float,
    nscan: int,
) -> float:
    grid = np.linspace(float(amin), float(amax), int(nscan))
    values = np.array([evaluator(a) - target_gain for a in grid], dtype=float)
    for i in range(len(grid) - 1):
        if values[i] == 0.0:
            return float(grid[i])
        if values[i] * values[i + 1] < 0.0:
            return float(_bisect_root(lambda a: evaluator(a) - target_gain, grid[i], grid[i + 1], maxiter=500))
    raise RuntimeError("No amplitude solved the requested describing-function gain.")


def solve_amplitude_from_gain(
    gain: float,
    params: ChuaParameters,
    *,
    amin: float = 1.0 + 1.0e-9,
    amax: float = 100.0,
    nscan: int = 20_000,
) -> float:
    """Solve ``N(a)=gain`` for the classical describing function."""

    k = float(gain)
    if not is_describing_gain_compatible(k, params):
        raise RuntimeError("gain is not compatible with the selected Chua describing function.")
    if normalize_chua_model(params.model) == "arctan":
        amplitude_sq = 4.0 * params.a2 * (params.a2 * params.rho - k) / (k * k * params.rho)
        if amplitude_sq <= 0.0:
            raise RuntimeError("computed arctan amplitude is not real positive.")
        return float(np.sqrt(amplitude_sq))
    sat_gain = chua_gain(params)
    if abs(k - sat_gain) < 1.0e-10:
        return 1.0
    return _solve_scalar_gain(k, lambda a: describing_function(a, params), amin=amin, amax=amax, nscan=nscan)


def solve_machado_amplitude_from_gain(
    gain: float,
    params: ChuaParameters,
    mu: float,
    *,
    amin: float = 1.0 + 1.0e-9,
    amax: float = 100.0,
    nscan: int = 20_000,
) -> float:
    """Solve ``N(a)^mu=gain`` for the auxiliary Machado family."""

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
    return _solve_scalar_gain(target, lambda a: describing_function(a, params), amin=amin, amax=amax, nscan=nscan)


def build_fractional_seed(
    q: float,
    params: ChuaParameters,
    omega: float,
    gain: float,
    amplitude: float,
    *,
    theta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, complex]:
    """Build the harmonic initial condition associated with one DF branch."""

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

    ``method="machado"`` changes only the auxiliary amplitude relation used to
    propose a seed.  It is not a hiddenness verification.
    """

    p = params or chua_piecewise_parameters()
    pairs = find_omega_gain_candidates(q, p, wmin=wmin, wmax=wmax, nscan=nscan, compatible_only=(method == "classic"))
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
    """Return a JSON-serializable summary for reports and notebooks."""

    return {
        "seed": seed.seed.tolist(),
        "omega": seed.omega,
        "gain": seed.gain,
        "amplitude": seed.amplitude,
        "branch_index": seed.branch_index,
        "method": seed.method,
        "mu": seed.mu,
        "matched_eigenvalue": [float(np.real(seed.matched_eigenvalue)), float(np.imag(seed.matched_eigenvalue))],
    }


__all__ = [
    "BiasedHarmonicSeed",
    "HarmonicSeed",
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
    "fractional_iomega_power",
    "is_describing_gain_compatible",
    "machado_describing_function",
    "psi_sigma",
    "reconstruct_biased_lure_seed",
    "solve_amplitude_from_gain",
    "solve_machado_amplitude_from_gain",
    "transfer_function",
    "validate_fractional_order",
]
