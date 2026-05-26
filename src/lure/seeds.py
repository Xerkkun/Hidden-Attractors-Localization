"""Seed construction for the centered Lur'e describing-function workflow.

For q < 1, the harmonic balance seed must be reconstructed from the
eigenvector of the linearised matrix P0 = P + k*b*r^T associated to the
fractional eigenvalue lambda0 = (i*omega0)^q.
The closed-form formula with omega0^2 corresponds to the integer case q=1
and must NOT be used as a fractional seed.
"""

import warnings
import numpy as np
from typing import Any, Optional, Tuple

# Tolerance to decide whether q is effectively integer
_Q_INTEGER_TOL = 1e-10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lambda_from_frequency(omega0: float, q: float, transfer_mode: str) -> complex:
    """Return the eigenvalue lambda that corresponds to the harmonic frequency.

    Parameters
    ----------
    omega0 : float
        Angular frequency (must be > 0).
    q : float
        Caputo fractional order (must satisfy 0 < q <= 1).
    transfer_mode : str
        One of ``"integer"``, ``"fractional"``, or ``"auto"``.
        - ``"integer"``    → lambda = i * omega0
        - ``"fractional"`` → lambda = omega0^q * exp(i * pi * q / 2)
        - ``"auto"``       → uses ``"integer"`` when |q - 1| < tol, else ``"fractional"``

    Returns
    -------
    complex
        The eigenvalue lambda.

    Raises
    ------
    ValueError
        If omega0 <= 0, q is outside (0, 1], or transfer_mode is unknown.
    """
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
    # "auto"
    if abs(q - 1.0) < _Q_INTEGER_TOL:
        return complex(0.0, omega0)
    return complex((omega0 ** q) * np.exp(1j * np.pi * q / 2.0))


# ---------------------------------------------------------------------------
# Closed-form integer seed (q = 1 only)
# ---------------------------------------------------------------------------

def build_closed_form_integer_seed(
    system: Any,
    A0: float,
    omega0: float,
    k: float,
    seed_sign_convention: str = "kuznetsov",
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct the seed using the algebraic closed-form formula for q = 1.

    .. warning::
        This formula is only valid for the **integer** case q = 1 (or as a
        diagnostic tool).  For q < 1, use :func:`build_modal_lure_seed` instead.

    The closed-form expressions are:

        x0 = a0
        y0 = a0 * (m_linear + 1 + k)
        z0 = a0 * (alpha * (m_linear + k) - omega0^2) / alpha

    Parameters
    ----------
    system : object
        Lur'e system with attributes ``alpha``, and ``m1`` or ``m``.
    A0 : float
        Oscillation amplitude.
    omega0 : float
        Angular frequency.
    k : float
        Describing-function gain.
    seed_sign_convention : str
        ``"kuznetsov"`` (default) leaves the seed unmodified.
        ``"wu"`` flips the sign of the second component.

    Returns
    -------
    seed_pos : np.ndarray, shape (3,)
    seed_neg : np.ndarray, shape (3,)
        ``-seed_pos``
    """
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


# ---------------------------------------------------------------------------
# Modal seed (valid for all 0 < q <= 1)
# ---------------------------------------------------------------------------

def build_modal_lure_seed(
    system: Any,
    A0: float,
    omega0: float,
    k: float,
    q: float,
    transfer_mode: str = "auto",
    theta: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, complex, complex]:
    """Construct the harmonic seed from the dominant eigenvector of P0 = P + k*b*r^T.

    This is the correct construction for all fractional orders 0 < q <= 1.
    For q = 1 it coincides (up to numerical precision) with the algebraic
    closed-form result.

    The eigenvector v is normalised so that r^T v = 1, then scaled:

        X_seed = A0 * Re(v * exp(i * theta))

    Parameters
    ----------
    system : object
        Lur'e system with attributes ``P`` (3x3), ``b`` (3,), ``r`` (3,).
    A0 : float
        Oscillation amplitude.
    omega0 : float
        Angular frequency.
    k : float
        Describing-function gain.
    q : float
        Caputo fractional order (0 < q <= 1).
    transfer_mode : str
        ``"integer"``, ``"fractional"``, or ``"auto"`` (default).
    theta : float
        Initial phase offset in radians (default 0.0).

    Returns
    -------
    X_seed : np.ndarray, shape (n,)
        Real initial condition.
    v_normalized : np.ndarray, shape (n,), complex
        Dominant eigenvector normalised with r^T v = 1.
    matched_eigenvalue : complex
        Eigenvalue of P0 closest to lambda0.
    lambda0 : complex
        Target eigenvalue (i*omega0)^q.

    Raises
    ------
    RuntimeError
        If the eigenvector cannot be normalised (r^T v ≈ 0).
    """
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

    # Normalise with r^T v = 1
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    """Construct the initial state seed X_seed and its symmetric partner -X_seed.

    Parameters
    ----------
    system : object
        Lur'e system instance.
    A0 : float
        Oscillation amplitude from the describing-function scan.
    omega0 : float
        Angular frequency from the describing-function scan.
    k : float
        Describing-function gain from the scan.
    seed_sign_convention : str
        ``"kuznetsov"`` (default): seed returned as-is.
        ``"wu"``: second component flipped (only applied for
        ``seed_construction="closed_form_integer"``).
    q : float or None
        Caputo fractional order.  If ``None``, ``system.q`` is used.
    transfer_mode : str
        ``"integer"``, ``"fractional"``, or ``"auto"`` (default).
    theta : float
        Initial phase offset in radians (default 0.0).
    seed_construction : str
        ``"modal"`` (default): eigenvector construction — valid for all q.
        ``"closed_form_integer"``: algebraic formula — valid only for q = 1.

    Returns
    -------
    seed_pos : np.ndarray, shape (n,)
    seed_neg : np.ndarray, shape (n,)
        ``-seed_pos``

    Raises
    ------
    ValueError
        If ``seed_construction="closed_form_integer"`` is requested with
        ``q < 1`` or ``transfer_mode="fractional"``.  This combination would
        produce an incorrect seed and is therefore forbidden.
    """
    # Resolve q
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

    # Guard: closed-form integer seed is invalid for fractional cases
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
                "Use seed_construction='modal' for fractional orders — "
                "the closed-form formula with omega0^2 is the source of "
                "the fractional seed error."
            )
        return build_closed_form_integer_seed(
            system, A0, omega0, k, seed_sign_convention=seed_sign_convention
        )

    # Modal construction (default, valid for all q)
    X_seed, _v, _matched_ev, _lambda0 = build_modal_lure_seed(
        system, A0, omega0, k, q=q, transfer_mode=transfer_mode, theta=theta
    )
    return X_seed, -X_seed
