"""F2 — Fractional variational ABM-QR Lyapunov exponent estimator.

F2 — fractional_variational_abm_qr
=====================================
Implements finite-time local Lyapunov exponents for Caputo fractional-order
systems (0 < q < 1) by integrating the **extended original–variational system**

    ᶜDᵗq X   = F(X)
    ᶜDᵗq Φ   = J(X) Φ,   Φ(0) = I

with a Caputo Adams–Bashforth–Moulton (ABM) predictor-corrector and
history-aware QR reorthonormalisation.

Method identifier:  ``fractional_variational_abm_qr``
Derivative model:   Caputo
q support:          0 < q < 1
Finite-time local:  True
History-aware QR:   True  (see section below)

History-aware QR
----------------
In Caputo systems the **entire past** matters.  A naive "restart" that
resets Φ(tₖ) = Q without touching the stored variational history
produces a **history-inconsistent** trajectory from step k+1 onward.

This implementation applies the QR transform to every stored
variational block in the history window:

    Φⱼ ← Φⱼ · R⁻¹,     j = memory_start … current

and then recomputes the stored RHS samples:

    G(Yⱼ) ← rhs_ext( pack(Xⱼ, Φⱼ · R⁻¹) )

so that the ABM corrector sums remain coherent with the rotated basis.

If this transformation is omitted the method degrades to a block-restart
approximation (``fractional_variational_abm_qr_block_restart``), which
is **not** full-memory Caputo-aware.

Scope
-----
* Valid for: 0 < q < 1 (Caputo fractional ODE).
* NOT valid for q = 1 (use integer_qr_benettin).
* Results are finite-time local Lyapunov exponents, NOT asymptotic proofs.
* Does NOT certify chaos.
* Does NOT certify hiddenness of attractors.

    chaos_certified_by_this_pipeline: false
    hiddenness_certified_by_this_pipeline: false

References
----------
.. [Danca2018] M.-F. Danca & N. Kuznetsov, "Matlab Code for Lyapunov
   Exponents of Fractional-Order Systems", Int. J. Bifurcation Chaos
   28(5), 2018.  — Primary methodological reference for the extended
   original–variational Caputo system with ABM integration.
.. [Benettin1980] G. Benettin et al., Meccanica 15, 1980.
.. [Wolf1985] A. Wolf et al., Physica D 16, 1985.

ABM weight attribution
----------------------
The predictor and corrector weights used here mirror exactly those in
``hidden_attractors.integrations.abm._python_abm_integrate``:

    predictor scale = hᵠ / Γ(q+1)
    corrector scale = hᵠ / Γ(q+2)
    b_{j,n+1} = (n+1-j)ᵠ - (n-j)ᵠ
    a_s       = (n-s)^(q+1) - (n-s-q)·(n-s+1)ᵠ
    a_j (j>s) = (n-j+2)^(q+1) + (n-j)^(q+1) - 2·(n-j+1)^(q+1)

No alternative weight scheme is introduced.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.special import gamma as _gamma

from .lyapunov import (
    LyapunovResult,
    finite_difference_jacobian,
)

# ---------------------------------------------------------------------------
# Canonical references for F2
# ---------------------------------------------------------------------------

_FRACTIONAL_VARIATIONAL_ABM_QR_REFS: tuple[str, ...] = (
    "Danca & Kuznetsov 2018 — Matlab Code for Lyapunov Exponents of"
    " Fractional-Order Systems (Int. J. Bifurcation Chaos 28(5)):"
    " primary reference for extended original–variational Caputo system"
    " with ABM integration and QR reorthonormalisation.",
    "Benettin et al. 1980 — Lyapunov Characteristic Exponents (Meccanica 15).",
    "Wolf et al. 1985 — Determining Lyapunov Exponents from a Time Series"
    " (Physica D 16).",
)

_FRACTIONAL_VARIATIONAL_ABM_QR_WARNINGS: tuple[str, ...] = (
    "Results are finite-time local Lyapunov exponents, NOT asymptotic proofs.",
    "Caputo memory: history-aware QR transforms the entire stored variational"
    " history at each reorthonormalisation step.",
    "Does not certify chaos; does not certify hiddenness of attractors.",
    "chaos_certified_by_this_pipeline: false",
    "hiddenness_certified_by_this_pipeline: false",
    "validated_against_published_benchmarks: false (F2 — pending).",
    "Not validated for non-smooth systems (e.g., Chua saturation);"
    " derivative undefined at switching surfaces.",
)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FractionalVariationalQRConfig:
    """Configuration for :func:`fractional_variational_abm_qr`.

    Attributes
    ----------
    q : float
        Caputo fractional order.  Must satisfy 0 < q < 1.
    h : float
        Integration step size (positive).
    t_final : float
        Total integration time (positive).
    t_burn : float, default 0.0
        Burn-in time.  The extended system is integrated from t=0;
        exponent accumulation starts only after ``t_burn``.
    reorthonormalization_time : float or None, default None
        Physical time between QR steps; converted to step count.
    reorthonormalize_every : int or None, default None
        Steps between QR reorthonormalisations.  Default 10 if both
        ``reorthonormalization_time`` and this are ``None``.
    memory_mode : str, default ``'full'``
        ``'full'`` — full Caputo history.
        ``'window'`` — finite-memory window approximation.
    memory_window : int or None, default None
        Number of steps in the memory window (required if
        ``memory_mode='window'``).
    jacobian_eps : float, default 1e-6
        Finite-difference step for the Jacobian.
    div_threshold : float or None, default None
        Stop if ``‖X‖ ≥ div_threshold``.
    history_aware_qr : bool, default True
        Apply QR transform to the entire stored variational history
        (Caputo-coherent).  If ``False``, only the current Φ is reset
        (block-restart approximation, not full-memory Caputo-aware).
    qr_epsilon : float, default 1e-300
        Floor for |diag(R)| entries to avoid log(0).
    max_steps : int or None, default None
        Hard cap on total integration steps (safety limit).
    """

    q: float
    h: float
    t_final: float
    t_burn: float = 0.0
    reorthonormalization_time: float | None = None
    reorthonormalize_every: int | None = None
    memory_mode: str = "full"
    memory_window: int | None = None
    jacobian_eps: float = 1e-6
    div_threshold: float | None = None
    history_aware_qr: bool = True
    qr_epsilon: float = 1e-300
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.q < 1.0):
            raise ValueError(
                f"FractionalVariationalQRConfig: q must satisfy 0 < q < 1; got q={self.q}."
            )
        if self.h <= 0.0:
            raise ValueError(f"FractionalVariationalQRConfig: h must be positive; got h={self.h}.")
        if self.t_final <= 0.0:
            raise ValueError(f"FractionalVariationalQRConfig: t_final must be positive.")
        if self.t_burn < 0.0:
            raise ValueError(f"FractionalVariationalQRConfig: t_burn must be non-negative.")
        if self.memory_mode not in ("full", "window"):
            raise ValueError(
                f"FractionalVariationalQRConfig: memory_mode must be 'full' or 'window';"
                f" got '{self.memory_mode}'."
            )
        if self.memory_mode == "window" and (
            self.memory_window is None or int(self.memory_window) < 1
        ):
            raise ValueError(
                "FractionalVariationalQRConfig: memory_window must be a positive int"
                " when memory_mode='window'."
            )


# ---------------------------------------------------------------------------
# Extended-state packing / unpacking
# ---------------------------------------------------------------------------

def pack_extended_state(X: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """Pack state X (shape n) and variational matrix Φ (shape n×n) into a
    flat vector Y of shape n + n² (row-major).

    Parameters
    ----------
    X : np.ndarray, shape (n,)
    Phi : np.ndarray, shape (n, n)

    Returns
    -------
    Y : np.ndarray, shape (n + n*n,)
    """
    return np.concatenate([X.ravel(), Phi.ravel(order="C")])


def unpack_extended_state(Y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Unpack flat vector Y into state X and variational matrix Φ.

    Parameters
    ----------
    Y : np.ndarray, shape (n + n*n,)
    n : int
        State dimension.

    Returns
    -------
    X : np.ndarray, shape (n,)
    Phi : np.ndarray, shape (n, n)  — row-major ordering
    """
    X = Y[:n].copy()
    Phi = Y[n:].reshape(n, n, order="C").copy()
    return X, Phi


# ---------------------------------------------------------------------------
# Extended variational RHS builder
# ---------------------------------------------------------------------------

def build_extended_variational_rhs(
    rhs: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray] | None,
    n: int,
    jacobian_eps: float = 1e-6,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build ``G(Y) = [F(X), J(X)Φ]`` for the extended variational system.

    Parameters
    ----------
    rhs : callable
        ``F(x) -> dxdt``, shape ``(n,) -> (n,)``.
    jacobian : callable or None
        Analytic ``J(x) -> (n, n)``.  If ``None``, finite differences used.
    n : int
        State dimension.
    jacobian_eps : float
        Step size for finite-difference Jacobian when ``jacobian`` is ``None``.

    Returns
    -------
    G : callable
        ``G(Y) -> dYdt`` where ``Y`` is the packed extended state.
    """
    _jac = jacobian if jacobian is not None else (
        lambda x: finite_difference_jacobian(rhs, x, eps=jacobian_eps)
    )

    def G(Y: np.ndarray) -> np.ndarray:
        X, Phi = unpack_extended_state(Y, n)
        dX = np.asarray(rhs(X), dtype=float)
        J = np.asarray(_jac(X), dtype=float)
        dPhi = J @ Phi
        return pack_extended_state(dX, dPhi)

    return G


# ---------------------------------------------------------------------------
# History-aware QR transform
# ---------------------------------------------------------------------------

def apply_history_aware_qr_transform(
    states_history: list[np.ndarray],
    rhs_history: list[np.ndarray],
    rhs_ext: Callable[[np.ndarray], np.ndarray],
    n: int,
    current_index: int,
    qr_epsilon: float = 1e-300,
    memory_start_index: int = 0,
) -> tuple[np.ndarray, float, str]:
    """Apply a history-aware QR transform to the variational block.

    Computes ``Q, R = QR(Φ_current)`` then:

    1. For every stored state in ``[memory_start_index, current_index]``:
       - Extracts ``Φⱼ`` from the packed state.
       - Applies ``Φⱼ ← Φⱼ · R⁻¹``.
       - Repacks and updates ``states_history[j]``.
    2. Recomputes ``rhs_history[j] = rhs_ext(states_history[j])`` for
       all transformed indices so that the ABM corrector sums remain
       coherent with the new basis.

    Parameters
    ----------
    states_history : list[np.ndarray]
        Mutable list; each entry is a packed extended state Y of shape
        ``(n + n*n,)``.
    rhs_history : list[np.ndarray]
        Mutable list; each entry is ``G(Y)`` of same shape.
    rhs_ext : callable
        Extended RHS ``G(Y) -> dYdt``.
    n : int
        State dimension.
    current_index : int
        Index of the current step in ``states_history``.
    qr_epsilon : float
        Floor for |diag(R)| to avoid log(0).
    memory_start_index : int
        First index of the memory window.

    Returns
    -------
    log_diag : np.ndarray, shape (n,)
        ``log|diag(R)|`` (floored at ``qr_epsilon``).
    cond_R : float
        Condition number of ``R``; large values indicate ill-conditioning.
    qr_status : str
        ``'ok'`` or ``'qr_ill_conditioned'``.
    """
    Y_k = states_history[current_index]
    _, Phi_k = unpack_extended_state(Y_k, n)

    Q, R = np.linalg.qr(Phi_k)

    # Log-diagonal accumulation
    diag_abs = np.abs(np.diag(R)).copy()
    diag_abs[diag_abs <= qr_epsilon] = qr_epsilon
    log_diag = np.log(diag_abs)

    # Condition number
    try:
        cond_R = float(np.linalg.cond(R))
    except Exception:
        cond_R = float("inf")

    qr_status = "ok"
    if cond_R > 1e12:
        qr_status = "qr_ill_conditioned"
        warnings.warn(
            f"fractional_variational_abm_qr: R ill-conditioned (cond={cond_R:.2e})"
            " at QR step; results may be unreliable.",
            RuntimeWarning,
            stacklevel=4,
        )

    # R_inv: use lstsq for robustness when R is nearly singular
    try:
        R_inv = np.linalg.solve(R, np.eye(n, dtype=float))
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)
        qr_status = "qr_ill_conditioned"

    # Transform all history in [memory_start_index, current_index]
    for j in range(memory_start_index, current_index + 1):
        X_j, Phi_j = unpack_extended_state(states_history[j], n)
        Phi_j_new = Phi_j @ R_inv
        states_history[j] = pack_extended_state(X_j, Phi_j_new)
        try:
            rhs_history[j] = np.asarray(rhs_ext(states_history[j]), dtype=float)
        except Exception:
            # If RHS fails on transformed state, keep old value
            pass

    return log_diag, cond_R, qr_status


# ---------------------------------------------------------------------------
# Extended ABM integrator (F2 local, mirrors official ABM weights)
# ---------------------------------------------------------------------------

def _caputo_abm_extended_stepwise(
    rhs_ext: Callable[[np.ndarray], np.ndarray],
    Y0: np.ndarray,
    q: float,
    h: float,
    n_steps: int,
    n: int,
    memory_mode: str = "full",
    memory_window: int | None = None,
    qr_callback: Callable | None = None,
    div_threshold: float | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], str]:
    """Caputo ABM step-by-step integrator for the extended variational system.

    Mirrors the weight formulation of
    ``hidden_attractors.integrations.abm._python_abm_integrate``:

        predictor scale = hᵠ / Γ(q+1)
        b_{j,n+1}      = (n+1-j)ᵠ − (n−j)ᵠ
        corrector scale = hᵠ / Γ(q+2)
        a_s            = (n-s)^(q+1) − (n-s-q)·(n-s+1)ᵠ
        a_j (j>s)      = (n-j+2)^(q+1) + (n-j)^(q+1) − 2·(n-j+1)^(q+1)

    Parameters
    ----------
    rhs_ext : callable
        Extended RHS ``G(Y) -> dYdt``.
    Y0 : np.ndarray
        Initial packed extended state.
    q, h, n_steps : float, float, int
        Caputo order, step size, total steps.
    n : int
        State dimension (for divergence check on X block).
    memory_mode : str
        ``'full'`` or ``'window'``.
    memory_window : int or None
        Window length in steps (required for ``'window'`` mode).
    qr_callback : callable or None
        Called as ``qr_callback(states, rhs_vals, current_idx)`` at each
        QR step; may mutate ``states`` and ``rhs_vals`` in place.
    div_threshold : float or None
        Stop if ``‖X‖ >= div_threshold``.

    Returns
    -------
    states : list[np.ndarray]
        All packed extended states (length n_steps+1).
    rhs_vals : list[np.ndarray]
        All RHS values (length n_steps+1).
    times : list[float]
        Times corresponding to each state.
    status : str
    """
    hq = h ** q
    pred_scale = hq / float(_gamma(q + 1.0))
    corr_scale = hq / float(_gamma(q + 2.0))

    dim = Y0.size
    states: list[np.ndarray] = [Y0.copy()]
    rhs_vals: list[np.ndarray] = [np.asarray(rhs_ext(Y0), dtype=float)]
    times: list[float] = [0.0]

    # Pre-compute power arrays large enough for all steps
    max_len = n_steps + 2
    pow_q = np.arange(max_len + 1, dtype=float) ** q
    pow_q1 = np.arange(max_len + 1, dtype=float) ** (q + 1.0)

    status = "ok"

    for step in range(n_steps):
        n_cur = step  # current last index in states/rhs_vals
        t_new = (step + 1) * h

        # Window start
        if memory_mode == "window" and memory_window is not None:
            s_idx = max(0, n_cur - int(memory_window) + 1)
        else:
            s_idx = 0

        # ── Predictor ──────────────────────────────────────────────────────
        j_range = np.arange(s_idx, n_cur + 1)
        b_weights = pow_q[n_cur + 1 - j_range] - pow_q[n_cur - j_range]

        F_block = np.stack([rhs_vals[j] for j in range(s_idx, n_cur + 1)], axis=0)
        predictor = states[s_idx] + pred_scale * (b_weights @ F_block)

        try:
            fp = np.asarray(rhs_ext(predictor), dtype=float)
        except Exception as exc:
            status = f"solver_exception:{exc}"
            break

        # ── Corrector ──────────────────────────────────────────────────────
        n_prime = n_cur - s_idx
        a0 = float(n_prime) ** (q + 1.0) - (float(n_prime) - q) * (float(n_prime) + 1.0) ** q

        if n_prime > 0:
            mid_idx = n_cur - np.arange(s_idx + 1, n_cur + 1)
            a_mid = pow_q1[mid_idx + 2] + pow_q1[mid_idx] - 2.0 * pow_q1[mid_idx + 1]
            a_weights = np.concatenate(([a0], a_mid))
        else:
            a_weights = np.array([a0])

        corrected = states[s_idx] + corr_scale * ((a_weights @ F_block) + fp)

        if not np.all(np.isfinite(corrected)):
            status = "nonfinite_solution"
            break

        # Divergence check on state block
        X_new = corrected[:n]
        if div_threshold is not None and np.linalg.norm(X_new) >= float(div_threshold):
            status = "diverged"
            states.append(corrected)
            rhs_vals.append(fp)
            times.append(t_new)
            break

        states.append(corrected)
        times.append(t_new)

        try:
            f_new = np.asarray(rhs_ext(corrected), dtype=float)
        except Exception as exc:
            status = f"solver_exception:{exc}"
            break
        rhs_vals.append(f_new)

        # QR callback (history-aware transform)
        if qr_callback is not None:
            qr_callback(states, rhs_vals, len(states) - 1)

    return states, rhs_vals, times, status


# ---------------------------------------------------------------------------
# Main F2 estimator
# ---------------------------------------------------------------------------

_FRAC_REFS = (
    "Danca & Kuznetsov 2018 — Matlab Code for Lyapunov Exponents of"
    " Fractional-Order Systems (Int. J. Bifurcation Chaos 28(5))",
    "Benettin et al. 1980 — Lyapunov Characteristic Exponents (Meccanica 15)",
    "Wolf et al. 1985 — Determining Lyapunov Exponents from a Time Series (Physica D 16)",
)


def fractional_variational_abm_qr(
    rhs: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray] | None,
    x0: np.ndarray,
    *,
    q: float,
    h: float,
    t_final: float,
    t_burn: float = 0.0,
    reorthonormalization_time: float | None = None,
    reorthonormalize_every: int | None = None,
    memory_mode: str = "full",
    memory_window: int | None = None,
    jacobian_eps: float = 1e-6,
    div_threshold: float | None = None,
    history_aware_qr: bool = True,
    qr_epsilon: float = 1e-300,
) -> LyapunovResult:
    """Estimate Caputo fractional-order Lyapunov exponents (F2).

    **Method identifier: ``fractional_variational_abm_qr``**
    **Phase: F2 — implemented, not yet validated against published benchmarks**

    Integrates the extended original–variational Caputo system

        ᶜDᵗq X   = F(X)
        ᶜDᵗq Φ   = J(X) Φ,   Φ(0) = I

    with an Adams–Bashforth–Moulton (ABM) predictor-corrector and applies
    history-aware QR reorthonormalisation to maintain coherence with Caputo
    memory.

    **Scope**

    * Valid for **0 < q < 1** (Caputo fractional ODE).
    * NOT valid for q = 1 (use ``integer_qr_benettin``).
    * Results are finite-time local Lyapunov exponents.
    * Does NOT certify chaos or hiddenness.
    * ``validated_against_published_benchmarks: false`` (F2 pending).

    **Methodological warning**

    This routine is **not yet validated against published benchmarks**.
    Results are finite-time local Lyapunov exponent estimates.
    Caputo memory requires transforming the entire stored variational history
    at each QR step (``history_aware_qr=True``).  If ``history_aware_qr=False``
    only the current Φ is reset (block-restart, not full-memory Caputo-aware).

    Parameters
    ----------
    rhs : callable
        Vector field ``F(x) -> dxdt``, shape ``(n,) -> (n,)``.
    jacobian : callable or None
        Analytic Jacobian ``J(x) -> (n, n)``.  ``None`` → finite differences.
    x0 : np.ndarray, shape (n,)
        Initial state.
    q : float
        Caputo order.  Must satisfy 0 < q < 1.
    h : float
        Step size (positive).
    t_final : float
        Total integration time (burn-in excluded).
    t_burn : float, default 0.0
        Burn-in integration time.  The extended system is integrated for
        ``t_burn`` before accumulating exponents.  At the burn-in/accumulation
        boundary a history-aware QR reset is applied without accumulation.
    reorthonormalization_time : float or None, default None
        Physical time between QR steps; converted to step count.
    reorthonormalize_every : int or None, default None
        Steps between QR steps.  Default 10 if both ``None``.
    memory_mode : str, default ``'full'``
        ``'full'`` — full Caputo history (recommended).
        ``'window'`` — finite-memory window (NOT equivalent to full Caputo).
    memory_window : int or None, default None
        Window size in steps (required if ``memory_mode='window'``).
    jacobian_eps : float, default 1e-6
        Finite-difference step.
    div_threshold : float or None, default None
        Divergence threshold on ``‖X‖``.
    history_aware_qr : bool, default True
        Apply QR transform to all stored variational history (Caputo-coherent).
        If ``False``, only Φ_current is reset (block-restart approximation).
    qr_epsilon : float, default 1e-300
        Floor for |diag(R)|.

    Returns
    -------
    result : LyapunovResult
        Exponent estimates with full F2 metadata.

    Raises
    ------
    ValueError
        If q not in (0, 1), h ≤ 0, or t_final ≤ 0.

    References
    ----------
    .. [Danca2018] M.-F. Danca & N. Kuznetsov, Int. J. Bifurcation Chaos
       28(5), 2018.
    """
    # ── Input validation ────────────────────────────────────────────────────
    q = float(q)
    if not (0.0 < q < 1.0):
        raise ValueError(
            f"fractional_variational_abm_qr is valid only for 0 < q < 1; got q={q}. "
            "For q=1, use integer_qr_benettin."
        )
    h = float(h)
    if h <= 0.0:
        raise ValueError("h must be positive.")
    t_final = float(t_final)
    if t_final <= 0.0:
        raise ValueError("t_final must be positive.")
    t_burn = float(t_burn)
    if t_burn < 0.0:
        raise ValueError("t_burn must be non-negative.")
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be one-dimensional.")
    n = x0.size

    # ── Resolve reorthonormalize_every ──────────────────────────────────────
    if reorthonormalize_every is not None and reorthonormalization_time is not None:
        # both given: use reorthonormalize_every
        interval = int(reorthonormalize_every)
    elif reorthonormalize_every is not None:
        interval = int(reorthonormalize_every)
    elif reorthonormalization_time is not None:
        interval = max(1, round(float(reorthonormalization_time) / h))
    else:
        interval = 10

    # ── Build extended system ───────────────────────────────────────────────
    rhs_ext = build_extended_variational_rhs(rhs, jacobian, n, jacobian_eps)

    Phi0 = np.eye(n, dtype=float)
    Y0 = pack_extended_state(x0, Phi0)

    # ── Burn-in phase ───────────────────────────────────────────────────────
    # Integrate the extended system for t_burn without accumulating exponents.
    # At the end of burn-in, apply a history-aware QR reset (no accumulation)
    # so the basis Φ starts orthonormal for the accumulation phase.
    burn_steps = int(max(0, round(t_burn / h)))
    accu_steps = int(max(0, round(t_final / h)))
    t_burn_effective = burn_steps * h

    sums = np.zeros(n, dtype=float)
    times_out: list[float] = []
    convergence_out: list[np.ndarray] = []
    last_accumulation_time = t_burn_effective
    elapsed = 0.0
    status = "ok"
    qr_ill_count = 0

    # ── Combined integration (burn + accumulation) ──────────────────────────
    # Strategy: integrate burn_steps + accu_steps in one pass.
    # Track whether we are in the accumulation phase.
    total_steps = burn_steps + accu_steps

    # Shared mutable history (will be built by _caputo_abm_extended_stepwise
    # but we need the QR callback to have access to the live lists).
    live_states: list[np.ndarray] = [Y0.copy()]
    live_rhs: list[np.ndarray] = [np.asarray(rhs_ext(Y0), dtype=float)]
    live_times: list[float] = [0.0]

    # Precompute powers
    max_len = total_steps + 2
    pow_q_arr = np.arange(max_len + 1, dtype=float) ** q
    pow_q1_arr = np.arange(max_len + 1, dtype=float) ** (q + 1.0)

    hq = h ** q
    pred_scale = hq / float(_gamma(q + 1.0))
    corr_scale = hq / float(_gamma(q + 2.0))

    step_global = 0  # counts steps integrated so far (0-indexed)

    for step in range(total_steps):
        n_cur = step  # index of current last element (= len(live_states)-1)
        t_new = (step + 1) * h
        in_burn = step < burn_steps

        # Window start index
        if memory_mode == "window" and memory_window is not None:
            s_idx = max(0, n_cur - int(memory_window) + 1)
        else:
            s_idx = 0

        # ── Predictor ──────────────────────────────────────────────────────
        j_range = np.arange(s_idx, n_cur + 1)
        b_weights = pow_q_arr[n_cur + 1 - j_range] - pow_q_arr[n_cur - j_range]
        F_block = np.stack([live_rhs[j] for j in range(s_idx, n_cur + 1)], axis=0)
        predictor = live_states[s_idx] + pred_scale * (b_weights @ F_block)

        try:
            fp = np.asarray(rhs_ext(predictor), dtype=float)
        except Exception as exc:
            status = f"solver_exception:{exc}"
            break

        # ── Corrector ──────────────────────────────────────────────────────
        n_prime = n_cur - s_idx
        a0 = float(n_prime) ** (q + 1.0) - (float(n_prime) - q) * (float(n_prime) + 1.0) ** q

        if n_prime > 0:
            mid_idx = n_cur - np.arange(s_idx + 1, n_cur + 1)
            a_mid = (pow_q1_arr[mid_idx + 2]
                     + pow_q1_arr[mid_idx]
                     - 2.0 * pow_q1_arr[mid_idx + 1])
            a_weights = np.concatenate(([a0], a_mid))
        else:
            a_weights = np.array([a0])

        corrected = live_states[s_idx] + corr_scale * ((a_weights @ F_block) + fp)

        if not np.all(np.isfinite(corrected)):
            status = "nonfinite_solution"
            break

        X_new = corrected[:n]
        if div_threshold is not None and np.linalg.norm(X_new) >= float(div_threshold):
            status = "diverged"
            live_states.append(corrected)
            live_rhs.append(fp)
            live_times.append(t_new)
            break

        live_states.append(corrected)
        live_times.append(t_new)

        try:
            f_new = np.asarray(rhs_ext(corrected), dtype=float)
        except Exception as exc:
            status = f"solver_exception:{exc}"
            break
        live_rhs.append(f_new)

        step_global = step + 1

        # ── QR step decision ───────────────────────────────────────────────
        # Trigger QR if at interval boundary, OR at the last burn-in step.
        at_qr_boundary = (step_global % interval == 0)
        at_burn_end = (burn_steps > 0 and step_global == burn_steps)

        if at_qr_boundary or at_burn_end:
            cur_idx = len(live_states) - 1
            mem_start = max(0, s_idx)  # use current window start for history transform

            if history_aware_qr:
                log_diag, cond_R, qr_st = apply_history_aware_qr_transform(
                    live_states,
                    live_rhs,
                    rhs_ext,
                    n,
                    cur_idx,
                    qr_epsilon=qr_epsilon,
                    memory_start_index=mem_start,
                )
            else:
                # Block-restart: only transform current Phi
                Y_k = live_states[cur_idx]
                X_k, Phi_k = unpack_extended_state(Y_k, n)
                Q, R = np.linalg.qr(Phi_k)
                diag_abs = np.abs(np.diag(R)).copy()
                diag_abs[diag_abs <= qr_epsilon] = qr_epsilon
                log_diag = np.log(diag_abs)
                live_states[cur_idx] = pack_extended_state(X_k, Q)
                live_rhs[cur_idx] = np.asarray(rhs_ext(live_states[cur_idx]), dtype=float)
                cond_R = 1.0
                qr_st = "ok"

            if qr_st == "qr_ill_conditioned":
                qr_ill_count += 1

            # Accumulate only during the accumulation phase
            if not in_burn and not at_burn_end:
                delta_t = t_new - last_accumulation_time
                if delta_t > 0:
                    elapsed += delta_t
                    sums += log_diag
                    convergence_out.append(sums / elapsed)
                    times_out.append(elapsed)
                    last_accumulation_time = t_new

    # Final elapsed correction: count QR steps that actually fired
    # (simpler: elapsed = total reorthonormalisation intervals × (interval × h))
    # Recompute from times_out
    if len(times_out) > 0:
        elapsed = times_out[-1]

    if elapsed > 0.0:
        exponents = sums / elapsed
    else:
        exponents = np.full(n, np.nan)

    extra_warnings = list(_FRACTIONAL_VARIATIONAL_ABM_QR_WARNINGS)
    if qr_ill_count > 0:
        extra_warnings.append(
            f"qr_ill_conditioned_count={qr_ill_count}: R was ill-conditioned at"
            f" {qr_ill_count} QR step(s); results may be unreliable."
        )
    if not history_aware_qr:
        extra_warnings.append(
            "history_aware_qr=False: block-restart used; NOT full-memory Caputo-aware."
        )

    return LyapunovResult(
        exponents=np.asarray(exponents, dtype=float),
        times=np.asarray(times_out, dtype=float),
        convergence=(
            np.asarray(convergence_out, dtype=float)
            if convergence_out
            else np.empty((0, n), dtype=float)
        ),
        status=status,
        method_id="fractional_variational_abm_qr",
        derivative_model="caputo",
        q=q,
        finite_time_local=True,
        jacobian_required=True,
        orthonormalization="qr",
        reference_ids=_FRACTIONAL_VARIATIONAL_ABM_QR_REFS,
        methodological_warnings=tuple(extra_warnings),
    )


__all__ = [
    "FractionalVariationalQRConfig",
    "pack_extended_state",
    "unpack_extended_state",
    "build_extended_variational_rhs",
    "apply_history_aware_qr_transform",
    "fractional_variational_abm_qr",
]
