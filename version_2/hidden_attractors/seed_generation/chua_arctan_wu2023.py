"""Centered Lur'e seeds for the Wu et al. (2023) arctan Chua case.

Stability: experimental
    This module isolates the smooth Wu2023 convention from the historical
    non-smooth Chua seed artifacts.

Sign convention
---------------
For ``D^q x = P x + b psi(r^T x)`` with
``psi(sigma) = a2*atan(rho*sigma)`` and ``b=(-alpha, 0, 0)^T``, define

``W_q(omega) = r^T ((i*omega)^q I - P)^(-1) b``.

The harmonic characteristic equation is ``1 - W_q*N(A) = 0``.  Writing
``1 + W_q*N(A) = 0`` with the same ``b`` and ``psi`` reverses the feedback
sign and yields no admissible centered Wu2023 arctan branches.
"""

from __future__ import annotations

import numpy as np

from ..models.chua import ChuaParameters, chua_arctan_wu2023_parameters
from .chua import (
    build_fractional_seed,
    chua_matrices,
    describing_function,
    is_describing_gain_compatible,
    solve_amplitude_from_gain,
)
from .core import HarmonicSeed, _bisect_root, complex_dtype, fractional_iomega_power, validate_fractional_order


def transfer_function_arctan_wu2023(
    omega: float,
    q: float = 0.99,
    params: ChuaParameters | None = None,
) -> complex:
    """Return ``r^T ((i*omega)^q I - P)^(-1) b`` on the principal branch."""

    p = params or chua_arctan_wu2023_parameters()
    if p.model != "arctan":
        raise ValueError("Wu2023 transfer requires model='arctan'.")
    pmat, bvec, rvec = chua_matrices(p)
    lam = fractional_iomega_power(omega, q)
    lhs = lam * np.eye(3, dtype=complex_dtype) - pmat.astype(complex_dtype)
    return complex_dtype(rvec.astype(complex_dtype) @ np.linalg.solve(lhs, bvec.astype(complex_dtype)))


def find_centered_arctan_wu2023_branches(
    *,
    q: float = 0.99,
    params: ChuaParameters | None = None,
    wmin: float = 1.0e-4,
    wmax: float = 10.0,
    nscan: int = 20_000,
) -> list[HarmonicSeed]:
    """Return admissible centered classical-DF branches for the Wu2023 model."""

    q_value = validate_fractional_order(q)
    p = params or chua_arctan_wu2023_parameters()
    if p.model != "arctan":
        raise ValueError("Wu2023 centered branches require model='arctan'.")
    if wmin <= 0.0 or wmax <= wmin or int(nscan) < 8:
        raise ValueError("expected 0 < wmin < wmax and nscan >= 8.")
    grid = np.linspace(float(wmin), float(wmax), int(nscan))
    imaginary = np.array([transfer_function_arctan_wu2023(w, q_value, p).imag for w in grid])
    roots: list[float] = []
    for left, right, f_left, f_right in zip(grid, grid[1:], imaginary, imaginary[1:]):
        if not np.isfinite(f_left) or not np.isfinite(f_right):
            continue
        if f_left == 0.0:
            roots.append(float(left))
        elif f_left * f_right < 0.0:
            roots.append(
                _bisect_root(
                    lambda omega: transfer_function_arctan_wu2023(omega, q_value, p).imag,
                    float(left),
                    float(right),
                )
            )
    unique = []
    for root in sorted(roots):
        if not unique or abs(root - unique[-1]) > 1.0e-7:
            unique.append(root)

    seeds: list[HarmonicSeed] = []
    for omega in unique:
        response = transfer_function_arctan_wu2023(omega, q_value, p)
        if abs(response.real) < 1.0e-12:
            continue
        gain = 1.0 / float(response.real)
        if not is_describing_gain_compatible(gain, p):
            continue
        amplitude = solve_amplitude_from_gain(gain, p)
        seed, eigenvector, matched = build_fractional_seed(q_value, p, omega, gain, amplitude)
        seeds.append(
            HarmonicSeed(
                seed=seed,
                eigenvector=eigenvector,
                matched_eigenvalue=matched,
                omega=float(omega),
                gain=float(gain),
                amplitude=float(amplitude),
                branch_index=len(seeds),
                method="classical_centered_arctan",
            )
        )
    return seeds


def _complex_pair(value: complex) -> list[float]:
    return [float(np.real(value)), float(np.imag(value))]


def format_arctan_wu2023_seed_report(
    *,
    q: float = 0.99,
    params: ChuaParameters | None = None,
    nscan: int = 20_000,
) -> dict[str, object]:
    """Create JSON-ready branch evidence with fractional lambda and closure checks."""

    p = params or chua_arctan_wu2023_parameters()
    branches = find_centered_arctan_wu2023_branches(q=q, params=p, nscan=nscan)
    rows: list[dict[str, object]] = []
    for branch in branches:
        response = transfer_function_arctan_wu2023(branch.omega, q, p)
        nonlinear_gain = describing_function(branch.amplitude, p)
        rows.append(
            {
                "branch": branch.branch_index,
                "family": branch.method,
                "omega": branch.omega,
                "k": branch.gain,
                "A": branch.amplitude,
                "lambda": _complex_pair(fractional_iomega_power(branch.omega, q)),
                "W_q": _complex_pair(response),
                "N_A": nonlinear_gain,
                "eigenvalue": _complex_pair(branch.matched_eigenvalue),
                "eigenvector": [_complex_pair(item) for item in branch.eigenvector],
                "seed": branch.seed.tolist(),
                "closure_residual_1_minus_WN": abs(1.0 - response * nonlinear_gain),
                "incompatible_requested_residual_1_plus_WN": abs(1.0 + response * nonlinear_gain),
            }
        )
    return {
        "system": "fractional_chua_arctan_wu2023",
        "q": float(q),
        "method": "classical_centered_arctan",
        "transfer_function": "r^T ((j*omega)^q I - P)^(-1) b",
        "balance_equation": "1 - W_q(j*omega)*N(A) = 0",
        "sign_audit": "With b=[-alpha,0,0]^T and a2<0, 1 + W_q*N(A) = 0 has no admissible centered branch.",
        "branches": rows,
    }


__all__ = [
    "find_centered_arctan_wu2023_branches",
    "format_arctan_wu2023_seed_report",
    "transfer_function_arctan_wu2023",
]
