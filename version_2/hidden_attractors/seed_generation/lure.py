"""Generic Lur'e seed generation: system-independent harmonic-balance helpers.

Stability: experimental
    These functions operate on any :class:`~hidden_attractors.systems.LureSystem`
    that provides a Lur'e split ``(A, b, c, psi, N(A))``.  The API is
    considered useful and tested but may gain new parameters as new system
    families are added.

Contents
--------
Transfer functions
    :func:`lure_transfer_function`, :func:`build_lure_linearized_matrix`

Frequency scan
    :func:`find_lure_omega_gain_candidates`

Describing functions
    :func:`lure_describing_function`, :func:`lure_machado_describing_function`,
    :func:`solve_lure_amplitude_from_gain`

Biased / Fourier
    :func:`fourier_coefficients_lure`, :func:`biased_lure_describing_function`,
    :func:`reconstruct_biased_lure_seed_from_system`

Seed construction
    :func:`build_lure_fractional_seed`, :func:`find_lure_harmonic_seed`
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..systems.lure import LureSystem
from .core import (
    BiasedHarmonicSeed,
    HarmonicSeed,
    _bisect_root,
    complex_dtype,
    fractional_iomega_power,
    real_dtype,
    validate_fractional_order,
)


# ── Lur'e transfer function ──────────────────────────────────────────────────

def build_lure_linearized_matrix(system: LureSystem, gain: float) -> np.ndarray:
    """Return the linearised matrix ``A + k b c^T`` for a Lur'e DF gain ``k``.

    Parameters
    ----------
    system : LureSystem
        Lur'e system providing ``matrix``, ``input_vector``, ``output_vector``.
    gain : float
        Describing-function gain ``k``.

    Returns
    -------
    A_lin : np.ndarray, shape (n, n)
        Linearised system matrix.
    """

    return np.asarray(system.matrix, dtype=real_dtype) + float(gain) * np.outer(
        np.asarray(system.input_vector, dtype=real_dtype),
        np.asarray(system.output_vector, dtype=real_dtype),
    )


WEYL_CAPUTO_NOTE = (
    "Weyl-Caputo Note: Para sistemas con orden fraccionario q < 1.0, la frecuencia s = (j*omega)^q "
    "se evalúa formalmente en la rama principal como s = omega^q * exp(j * q * pi / 2) de acuerdo al "
    "operador de Weyl-Caputo."
)


def lure_transfer_function(omega: float, q: float, system: LureSystem) -> complex:
    """Return ``c^T (A - (i\u03c9)^q I)^{-1} b`` for a generic Lur'e system.

    Parameters
    ----------
    omega : float
        Angular frequency (positive and finite).
    q : float
        Caputo fractional order.
    system : LureSystem
        Lur'e system providing ``matrix``, ``input_vector``, ``output_vector``.

    Returns
    -------
    W : complex
        Transfer function value at ``s = (i omega)^q``.

    Raises
    ------
    ValueError
        If evaluating with integer order when q != 1.0 or if system is fractional and q = 1.0.
    numpy.linalg.LinAlgError
        If ``A - (i omega)^q I`` is singular.
    """

    q_val = float(q)
    if q_val == 1.0:
        if "fractional" in getattr(system, "name", "").lower() or "nonsmooth" in getattr(system, "name", "").lower():
            raise ValueError("Prohibited evaluating fractional Lur'e system with integer order (q = 1.0).")
        s = complex_dtype(1j * omega)
    else:
        s = fractional_iomega_power(omega, q_val)

    matrix = np.asarray(system.matrix, dtype=complex_dtype)
    bvec = np.asarray(system.input_vector, dtype=complex_dtype)
    cvec = np.asarray(system.output_vector, dtype=complex_dtype)
    lhs = matrix - s * np.eye(
        system.dimension, dtype=complex_dtype
    )
    value = (cvec.reshape(1, -1) @ np.linalg.inv(lhs) @ bvec.reshape(-1, 1))[0, 0]
    return complex_dtype(value)


# ── Frequency scan ────────────────────────────────────────────────────────────

def find_lure_omega_gain_candidates(
    q: float,
    system: LureSystem,
    *,
    wmin: float = 1.0e-4,
    wmax: float = 10.0,
    nscan: int = 20_000,
    compatible_only: bool = True,
) -> list[tuple[float, float]]:
    """Find all ``(\u03c9, k)`` pairs where ``Im(W_q(i\u03c9)) = 0`` for a Lur'e system.

    Parameters
    ----------
    q : float
        Caputo fractional order.
    system : LureSystem
        Lur'e system with a scalar feedback channel.
    wmin : float, default 1e-4
        Minimum angular frequency.
    wmax : float, default 10.0
        Maximum angular frequency.
    nscan : int, default 20_000
        Number of frequency points in the initial scan.
    compatible_only : bool, default True
        If ``True``, reject pairs where the gain is not compatible with
        the system's amplitude model (calls ``system.is_gain_compatible``).

    Returns
    -------
    pairs : list[tuple[float, float]]
        Sorted list of ``(omega, gain)`` tuples.

    Raises
    ------
    ValueError
        If ``wmin >= wmax`` or ``nscan < 8``.
    """

    q_value = validate_fractional_order(q)
    if wmin <= 0.0 or wmax <= wmin:
        raise ValueError("expected 0 < wmin < wmax.")
    if int(nscan) < 8:
        raise ValueError("nscan must be at least 8.")

    ws = np.linspace(float(wmin), float(wmax), int(nscan))
    vals = np.array(
        [np.imag(lure_transfer_function(w, q_value, system)) for w in ws], dtype=float
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
                        lambda w: np.imag(lure_transfer_function(w, q_value, system)),
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
        response = lure_transfer_function(omega, q_value, system)
        re_value = float(np.real(response))
        if abs(re_value) < 1.0e-12:
            continue
        gain = -1.0 / re_value
        if compatible_only and not system.is_gain_compatible(gain):
            continue
        pairs.append((float(omega), float(gain)))
    return sorted(pairs, key=lambda item: item[1])


# ── Lur'e describing functions ────────────────────────────────────────────────

def lure_describing_function(amplitude: float, system: LureSystem) -> complex:
    """Evaluate the classical first-harmonic describing function for *system*.

    Parameters
    ----------
    amplitude : float
        Oscillation amplitude ``A > 0``.
    system : LureSystem
        Must provide a ``describing_function`` callback.

    Returns
    -------
    N_A : complex
        First-harmonic DF value.

    Raises
    ------
    ValueError
        If *amplitude* is not positive and finite.
    """

    amp = float(amplitude)
    if amp <= 0.0 or not np.isfinite(amp):
        raise ValueError("amplitude must be positive and finite.")
    return complex_dtype(system.describing_function(amp))


def lure_machado_describing_function(
    amplitude: float, system: LureSystem, mu: float
) -> float:
    """Return the real Machado-family describing function ``N_\u03bc(A) = N(A)^\u03bc``.

    Parameters
    ----------
    amplitude : float
        Oscillation amplitude ``A > 0``.
    system : LureSystem
        Must provide a ``machado_describing_function(amplitude, mu)`` callback.
    mu : float
        Machado exponent (positive and finite).

    Returns
    -------
    N_mu : float
        ``N(A)^mu`` on the real branch.

    Raises
    ------
    ValueError
        If *mu* is not positive and finite, or if the base DF returns a
        complex value (imaginary part exceeds ``1e-12``).

    Notes
    -----
    Complex or sign-changing DF variants need a custom branch convention
    and should be implemented by a dedicated amplitude solver.
    """

    exponent = float(mu)
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("mu must be positive and finite.")
    value = complex(system.machado_describing_function(float(amplitude), exponent))
    if abs(float(np.imag(value))) > 1.0e-12:
        raise ValueError("Machado workflow currently requires a real-valued branch.")
    return float(np.real(value))


def solve_lure_amplitude_from_gain(
    gain: float,
    system: LureSystem,
    *,
    method: Literal["classic", "machado"] = "classic",
    mu: float = 1.0,
    amin: float = 1.0 + 1.0e-9,
    amax: float = 100.0,
    nscan: int = 20_000,
) -> float:
    """Solve the amplitude relation ``N(A) = gain`` for a generic Lur'e system.

    Parameters
    ----------
    gain : float
        Target describing-function gain.
    system : LureSystem
        Lur'e system with DF and optionally a closed-form amplitude solver.
    method : {'classic', 'machado'}, default 'classic'
        ``'classic'``: uses ``system.amplitude_from_gain`` if available,
        otherwise numerical bisection on :func:`lure_describing_function`.
        ``'machado'``: numerical bisection on :func:`lure_machado_describing_function`.
    mu : float, default 1.0
        Machado exponent; ignored when *method* is ``'classic'``.
    amin : float, default 1.0+1e-9
        Lower amplitude bound for numerical search.
    amax : float, default 100.0
        Upper amplitude bound.
    nscan : int, default 20_000
        Grid size for bisection.

    Returns
    -------
    amplitude : float
        Positive amplitude satisfying the selected amplitude relation.

    Raises
    ------
    ValueError
        If *method* is not ``'classic'`` or ``'machado'``.
    RuntimeError
        If no root is found in ``[amin, amax]``.
    """

    from .core import _solve_scalar_gain

    k = float(gain)
    if method == "classic" and system.amplitude_from_gain is not None:
        return system.solve_amplitude(k)
    if method == "classic":
        evaluator = lambda a: float(np.real(lure_describing_function(a, system)))
    elif method == "machado":
        evaluator = lambda a: lure_machado_describing_function(a, system, mu)
    else:
        raise ValueError("method must be 'classic' or 'machado'.")
    return _solve_scalar_gain(k, evaluator, amin=amin, amax=amax, nscan=nscan)


# ── Biased Lur'e seed ─────────────────────────────────────────────────────────

def fourier_coefficients_lure(
    amplitude: float,
    sigma0: float,
    system: LureSystem,
    *,
    harmonics: int = 10,
    n_quad: int = 4096,
) -> dict[str, object]:
    """Compute Fourier coefficients of ``\u03c8(\u03c30 + A cos(\u03b8))`` for a Lur'e system.

    Parameters
    ----------
    amplitude : float
        Oscillation amplitude ``A > 0``.
    sigma0 : float
        DC bias of the feedback coordinate.
    system : LureSystem
        Lur'e system providing the ``nonlinearity`` callback.
    harmonics : int, default 10
        Number of harmonics to compute.
    n_quad : int, default 4096
        Number of quadrature points; must satisfy ``n >= max(64, 8*harmonics)``.

    Returns
    -------
    coeffs : dict
        Same structure as :func:`~hidden_attractors.seed_generation.chua.fourier_coefficients_psi`:
        keys ``'amplitude'``, ``'sigma0'``, ``'harmonics'``, ``'n_quad'``,
        ``'y_mean'``, ``'coefficients'``.

    Raises
    ------
    ValueError
        If *amplitude* is not positive, *harmonics* < 1, or *n_quad* is too small.
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
        [system.nonlinearity(center + amp * np.cos(th)) for th in theta], dtype=real_dtype
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


def biased_lure_describing_function(
    amplitude: float,
    sigma0: float,
    system: LureSystem,
    *,
    harmonics: int = 10,
    n_quad: int = 4096,
) -> complex:
    """Return the biased Lur'e describing function ``N(A, \u03c30) = Y_1(A, \u03c30) / A``.

    Parameters
    ----------
    amplitude : float
        Oscillation amplitude ``A > 0``.
    sigma0 : float
        DC bias of the feedback coordinate.
    system : LureSystem
        Lur'e system providing the ``nonlinearity`` callback.
    harmonics : int, default 10
        Passed to :func:`fourier_coefficients_lure`.
    n_quad : int, default 4096
        Quadrature points.

    Returns
    -------
    N_biased : complex
        First-harmonic biased DF value ``Y_1 / A``.
    """

    data = fourier_coefficients_lure(
        amplitude, sigma0, system, harmonics=max(1, int(harmonics)), n_quad=n_quad
    )
    y1 = complex(data["coefficients"][1]["Y"])  # type: ignore[index]
    return complex_dtype(y1 / float(amplitude))


def reconstruct_biased_lure_seed_from_system(
    *,
    q: float,
    system: LureSystem,
    amplitude: float,
    sigma0: float,
    omega: float,
    theta: float = 0.0,
    harmonics: int = 10,
    n_quad: int = 4096,
) -> BiasedHarmonicSeed:
    """Reconstruct a biased Lur'e seed from DC and first-harmonic balance equations.

    Parameters
    ----------
    q : float
        Caputo fractional order.
    system : LureSystem
        Lur'e system providing the ``nonlinearity`` and structural matrices.
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
    fourier = fourier_coefficients_lure(
        amplitude, sigma0, system, harmonics=max(1, int(harmonics)), n_quad=n_quad
    )
    y1 = complex(fourier["coefficients"][1]["Y"])  # type: ignore[index]
    y_mean = float(fourier["y_mean"])
    pmat = np.asarray(system.matrix, dtype=real_dtype)
    bvec = np.asarray(system.input_vector, dtype=real_dtype)
    cvec = np.asarray(system.output_vector, dtype=real_dtype)

    lhs_dc = np.vstack([pmat, cvec.reshape(1, -1)]).astype(real_dtype)
    rhs_dc = np.concatenate([-bvec * y_mean, np.array([float(sigma0)], dtype=real_dtype)])
    mean_state, *_ = np.linalg.lstsq(lhs_dc, rhs_dc, rcond=None)

    lam = fractional_iomega_power(omega, q_value)
    lhs_h = np.vstack(
        [
            lam * np.eye(system.dimension, dtype=complex_dtype) - pmat.astype(complex_dtype),
            cvec.astype(complex_dtype).reshape(1, -1),
        ]
    )
    rhs_h = np.concatenate(
        [
            bvec.astype(complex_dtype) * y1,
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


# ── Lur'e harmonic seed ───────────────────────────────────────────────────────

def build_lure_fractional_seed(
    q: float,
    system: LureSystem,
    omega: float,
    gain: float,
    amplitude: float,
    *,
    theta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, complex]:
    """Build the harmonic initial state for one Lur'e DF branch.

    Parameters
    ----------
    q : float
        Caputo fractional order.
    system : LureSystem
        Lur'e system.
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
    seed : np.ndarray, shape (n,)
        Real initial condition.
    eigenvector : np.ndarray, shape (n,), complex
        Normalised dominant eigenvector (``c^T v = 1``).
    matched_eigenvalue : complex
        Eigenvalue closest to ``(i omega)^q``.

    Raises
    ------
    RuntimeError
        If ``c^T v \u2248 0`` (eigenvector cannot be normalised).
    """

    q_value = validate_fractional_order(q)
    linearized = build_lure_linearized_matrix(system, gain).astype(complex_dtype)
    lam = fractional_iomega_power(omega, q_value)
    eigvals, eigvecs = np.linalg.eig(linearized)
    idx = int(np.argmin(np.abs(eigvals - lam)))
    vector = eigvecs[:, idx]
    scale = complex(np.asarray(system.output_vector, dtype=complex_dtype) @ vector)
    if abs(scale) < 1.0e-14:
        raise RuntimeError("cannot normalize eigenvector with c^T v = 1.")
    vector = vector / scale
    seed = float(amplitude) * np.real(vector * np.exp(1j * float(theta)))
    return seed.astype(real_dtype), vector.astype(complex_dtype), complex(eigvals[idx])


def find_lure_harmonic_seed(
    *,
    q: float,
    system: LureSystem,
    branch_index: int = 0,
    method: Literal["classic", "machado"] = "classic",
    mu: float = 1.0,
    theta: float = 0.0,
    wmin: float = 1.0e-4,
    wmax: float = 10.0,
    nscan: int = 20_000,
) -> HarmonicSeed:
    """Locate a Lur'e DF branch and return a finite harmonic seed.

    Parameters
    ----------
    q : float
        Caputo fractional order.
    system : LureSystem
        Lur'e system to scan.
    branch_index : int, default 0
        Index into the sorted ``(omega, gain)`` candidate list.
    method : {'classic', 'machado'}, default 'classic'
        Amplitude-solving method.
    mu : float, default 1.0
        Machado exponent; ignored when *method* is ``'classic'``.
    theta : float, default 0.0
        Initial phase angle (radians).
    wmin : float, default 1e-4
        Minimum angular frequency for the scan.
    wmax : float, default 10.0
        Maximum angular frequency.
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
    """

    pairs = find_lure_omega_gain_candidates(
        q, system, wmin=wmin, wmax=wmax, nscan=nscan,
        compatible_only=(method == "classic"),
    )
    if not pairs:
        raise RuntimeError("no omega/gain candidate was found.")
    index = int(branch_index)
    if index < 0 or index >= len(pairs):
        raise IndexError("branch_index is outside the available candidate list.")
    omega, gain = pairs[index]
    amplitude = solve_lure_amplitude_from_gain(gain, system, method=method, mu=mu)
    seed, vector, matched = build_lure_fractional_seed(
        q, system, omega, gain, amplitude, theta=theta
    )
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


__all__ = [
    "biased_lure_describing_function",
    "build_lure_fractional_seed",
    "build_lure_linearized_matrix",
    "find_lure_harmonic_seed",
    "find_lure_omega_gain_candidates",
    "fourier_coefficients_lure",
    "lure_describing_function",
    "lure_machado_describing_function",
    "lure_transfer_function",
    "reconstruct_biased_lure_seed_from_system",
    "solve_lure_amplitude_from_gain",
    "WEYL_CAPUTO_NOTE",
]
