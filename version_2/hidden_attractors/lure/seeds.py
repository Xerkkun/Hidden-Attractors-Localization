import warnings
import numpy as np
from typing import Any, Optional, Tuple

_Q_INTEGER_TOL = 1e-10

def _lambda_from_frequency(omega0: float, q: float, transfer_mode: str) -> complex:
    """Return the eigenvalue lambda that corresponds to the harmonic frequency."""
    omega0 = float(omega0)
    q = float(q)

    if omega0 <= 0.0 or not np.isfinite(omega0):
        raise ValueError(f"omega0 must be positive and finite, got {omega0}.")
    if not np.isfinite(q) or not (0.0 < q <= 1.0):
        raise ValueError(f"q must satisfy 0 < q <= 1, got {q}.")
    if transfer_mode not in {"integer", "fractional", "auto"}:
        raise ValueError(
            f"transfer_mode must be 'integer', 'fractional', or 'auto', got {transfer_mode!r}."
        )

    if transfer_mode == "integer":
        return complex(0.0, omega0)
    if transfer_mode == "fractional":
        return complex((omega0 ** q) * np.exp(1j * np.pi * q / 2.0))
    if abs(q - 1.0) < _Q_INTEGER_TOL:
        return complex(0.0, omega0)
    return complex((omega0 ** q) * np.exp(1j * np.pi * q / 2.0))


def build_closed_form_integer_seed(
    system: Any,
    A0: float,
    omega0: float,
    k: float,
    seed_sign_convention: str = "kuznetsov",
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct the seed using the algebraic closed-form formula for q = 1."""
    a0 = float(A0)
    alpha = float(system.alpha)

    if hasattr(system, "m1"):
        m_linear = float(system.m1)
    elif hasattr(system, "m"):
        m_linear = float(system.m)
    else:
        raise ValueError(
            "System must have either 'm1' (saturation) or 'm' (arctan) attributes."
        )

    x0 = a0
    y0 = a0 * (m_linear + 1.0 + k)
    z0 = a0 * (alpha * (m_linear + k) - omega0 ** 2) / alpha

    if seed_sign_convention == "wu":
        y0 = -y0

    X_seed = np.array([x0, y0, z0], dtype=float)
    return X_seed, -X_seed


def build_modal_lure_seed(
    system: Any,
    A0: float,
    omega0: float,
    k: float,
    q: float,
    transfer_mode: str = "auto",
    theta: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, complex, complex]:
    """Construct the harmonic seed from the dominant eigenvector of P0 = P + k*b*r^T."""
    A0 = float(A0)
    k = float(k)
    theta = float(theta)

    P0 = system.P.astype(complex) + k * np.outer(
        system.b.astype(complex), system.r.astype(complex)
    )

    lambda0 = _lambda_from_frequency(omega0, q, transfer_mode)

    eigvals, eigvecs = np.linalg.eig(P0)
    idx = int(np.argmin(np.abs(eigvals - lambda0)))
    v = eigvecs[:, idx].copy()

    scale = system.r.astype(complex) @ v
    if abs(scale) < 1e-14:
        raise RuntimeError(
            "Cannot normalise eigenvector: r^T v is essentially zero "
            f"(|scale| = {abs(scale):.3e}). Check system matrices and (omega0, k)."
        )
    v = v / scale

    X_seed = A0 * np.real(v * np.exp(1j * theta))
    return (
        X_seed.astype(float),
        v.astype(complex),
        complex(eigvals[idx]),
        lambda0,
    )


def build_lure_seed(
    system: Any,
    A0: float,
    omega0: float,
    k: float,
    seed_sign_convention: str = "kuznetsov",
    q: Optional[float] = None,
    transfer_mode: str = "auto",
    theta: float = 0.0,
    seed_construction: str = "modal",
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct the initial state seed X_seed and its symmetric partner -X_seed."""
    if q is None:
        if not hasattr(system, "q"):
            raise ValueError(
                "q was not provided and system does not have a 'q' attribute."
            )
        q = float(system.q)
    else:
        q = float(q)

    if seed_construction not in {"modal", "closed_form_integer"}:
        raise ValueError(
            f"seed_construction must be 'modal' or 'closed_form_integer', "
            f"got {seed_construction!r}."
        )

    if seed_construction == "closed_form_integer":
        _is_fractional_mode = (
            transfer_mode == "fractional"
            or (transfer_mode == "auto" and abs(q - 1.0) >= _Q_INTEGER_TOL)
            or abs(q - 1.0) >= _Q_INTEGER_TOL
        )
        if _is_fractional_mode:
            raise ValueError(
                f"seed_construction='closed_form_integer' is only valid for q=1 "
                f"(integer order). Got q={q}, transfer_mode={transfer_mode!r}. "
                "Use seed_construction='modal' for fractional orders."
            )
        return build_closed_form_integer_seed(
            system, A0, omega0, k, seed_sign_convention=seed_sign_convention
        )

    X_seed, _v, _matched_ev, _lambda0 = build_modal_lure_seed(
        system, A0, omega0, k, q=q, transfer_mode=transfer_mode, theta=theta
    )
    return X_seed, -X_seed
