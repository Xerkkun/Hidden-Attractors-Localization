"""EFORK-3 Caputo fractional integrator.

Implements the explicit three-stage fractional Runge-Kutta method
(Ghoreishi, Ghaffari & Saad, 2023) exactly as published, with the following
additions required by the /src workflow:

  - Optional external prehistory (history_times / history_states arrays),
    which shifts the effective t=0 and augments the history sum.
  - Early-stopping support (divergence and equilibrium convergence).
  - Memory-window truncation (``memory_mode='window'``).
  - Transparent fall-through to the native C fractional_integrate backend
    when ``use_c_backend=True`` and the system is registered; Python EFORK-3
    is used otherwise or when the C backend is unavailable.

The three-stage coefficients depend only on ``q`` (the Caputo order) and are
computed once per integration call via :func:`efork3_coefficients`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import gamma as _gamma

from .fractional_c import fractional_integrate


# ---------------------------------------------------------------------------
# Coefficient computation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EFORK3Coefficients:
    """Coefficients of the explicit three-stage fractional RK method."""
    alpha: float
    c2: float
    c3: float
    a21: float
    a31: float
    a32: float
    w1: float
    w2: float
    w3: float


def efork3_coefficients(alpha: float) -> EFORK3Coefficients:
    """Return three-stage EFORK coefficients for ``0 < alpha < 1``.

    Follows exactly the formulas in Ghoreishi et al. (2023).
    """
    q = float(alpha)
    if not 0.0 < q < 1.0:
        raise ValueError(
            f"EFORK-3 Caputo requires 0 < alpha < 1, got alpha={q}. "
            "For integer order (q=1) use a standard integrator."
        )
    g1 = math.gamma(1.0 + q)
    g2 = math.gamma(1.0 + 2.0 * q)
    g3 = math.gamma(1.0 + 3.0 * q)
    denom = 2.0 * g2 * g2 - g3
    return EFORK3Coefficients(
        alpha=q,
        c2=(1.0 / (2.0 * g1)) ** (1.0 / q),
        c3=(1.0 / (4.0 * g1)) ** (1.0 / q),
        a21=1.0 / (2.0 * g1 * g1),
        a31=(g1 * g1 * g2 + 2.0 * g2 * g2 - g3) / (4.0 * g1 * g1 * denom),
        a32=-g2 / (4.0 * denom),
        w1=(8.0 * g1**3 * g2**2 - 6.0 * g1**3 * g3 + g2 * g3) / (g1 * g2 * g3),
        w2=2.0 * g1 * g1 * (4.0 * g2 * g2 - g3) / (g2 * g3),
        w3=-8.0 * g1 * g1 * denom / (g2 * g3),
    )


# ---------------------------------------------------------------------------
# History term  (kernel sum that represents the fractional memory)
# ---------------------------------------------------------------------------

def _history_term(
    t_eval: float,
    times: np.ndarray,
    states: np.ndarray,
    n_local: int,
    alpha: float,
    h: float,
    s_idx: int = 0,
) -> np.ndarray:
    """Evaluate the discrete Caputo history kernel at ``t_eval``.

    Parameters
    ----------
    t_eval    : Evaluation time (absolute).
    times     : Full time array (concatenation of prehistory + current).
    states    : Full state array (same shape).
    n_local   : Index in *times/states* of the current step base point.
    alpha     : Caputo order.
    h         : Step size.
    s_idx     : Start index for window truncation (memory_mode='window').
    """
    if n_local <= s_idx:
        return np.zeros(states.shape[1], dtype=float)
    increments = states[s_idx + 1: n_local + 1] - states[s_idx: n_local]
    t_local = times[s_idx: n_local]
    t_local_next = times[s_idx + 1: n_local + 1]
    powers = (t_eval - t_local) ** (1.0 - alpha) - (t_eval - t_local_next) ** (1.0 - alpha)
    return (increments.T @ powers) / (h * math.gamma(2.0 - alpha))


# ---------------------------------------------------------------------------
# Pure Python EFORK-3  (used when C backend is unavailable)
# ---------------------------------------------------------------------------

def _python_efork3_integrate(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    x0: np.ndarray,
    q: float,
    h: float,
    t_final: float,
    divergence_norm: Optional[float] = 120.0,
    history_times: Optional[np.ndarray] = None,
    history_states: Optional[np.ndarray] = None,
    memory_mode: str = "full",
    memory_window_length: Optional[int] = None,
    early_stop_config: Optional[dict] = None,
    equilibria: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Pure-Python EFORK-3 Caputo integration (exact published algorithm)."""
    q = float(q)
    h = float(h)
    t_final = float(t_final)
    n_steps = int(np.ceil(t_final / h))

    x0_arr = np.asarray(x0, dtype=float)
    dim = x0_arr.size

    coeff = efork3_coefficients(q)
    h_alpha = h ** q

    # --- Build combined time / state arrays (prehistory + new integration) ---
    if history_times is not None and history_states is not None:
        pre_t = np.asarray(history_times, dtype=float)
        pre_x = np.asarray(history_states, dtype=float)
        K = len(pre_t)
    else:
        K = 1
        pre_t = np.array([0.0])
        pre_x = x0_arr.reshape(1, dim)

    total = K + n_steps
    times = np.zeros(total + 1, dtype=float)
    states = np.zeros((total + 1, dim), dtype=float)

    times[:K] = pre_t
    states[:K] = pre_x

    # Set times for new steps (relative to last prehistory point = 0)
    t_start = pre_t[-1]  # typically 0.0 after shift
    for step_idx in range(n_steps + 1):
        times[K - 1 + step_idx] = t_start + step_idx * h

    states[K - 1] = x0_arr

    # --- Early stop config parsing ---
    esc = early_stop_config if early_stop_config is not None else {}
    es_enabled = esc.get("enabled", True)
    div_enabled = esc.get("divergence_enabled", True)
    div_norm_es = esc.get("divergence_norm", 80.0)
    div_consec = esc.get("divergence_consecutive_steps", 5)
    div_growth = esc.get("divergence_growth_factor", 1.25)
    eq_enabled = esc.get("equilibrium_enabled", True)
    eq_t = esc.get("equilibrium_tol", 1e-3)
    eq_deriv = esc.get("equilibrium_derivative_tol", 1e-4)
    eq_consec = esc.get("equilibrium_consecutive_steps", 200)
    eq_min_t = esc.get("equilibrium_min_time", 5.0)

    div_consec_count = 0
    growth_consec_count = 0
    prev_norm = -1.0
    eq_consec_counts = [0] * len(equilibria) if equilibria else []

    status = "ok"
    last_local = 0  # index in the NEW integration part (0-indexed at x0)

    for n in range(n_steps):
        # n_abs: absolute index in times/states arrays
        n_abs = K - 1 + n
        tn = times[n_abs]
        yn = states[n_abs]

        # Window start index for memory truncation
        if memory_mode == "window" and memory_window_length is not None:
            s_idx = max(0, n_abs - int(memory_window_length))
        else:
            s_idx = 0

        def _mrhs(t_eval: float, state: np.ndarray, _n_abs=n_abs, _s_idx=s_idx) -> np.ndarray:
            force = np.asarray(rhs(t_eval, state), dtype=float)
            hist = _history_term(t_eval, times, states, _n_abs, q, h, _s_idx)
            return force - hist

        # Three-stage EFORK-3
        k1 = h_alpha * _mrhs(tn, yn)
        k2 = h_alpha * _mrhs(tn + coeff.c2 * h, yn + coeff.a21 * k1)
        k3 = h_alpha * _mrhs(tn + coeff.c3 * h, yn + coeff.a31 * k1 + coeff.a32 * k2)
        x_next = yn + coeff.w1 * k1 + coeff.w2 * k2 + coeff.w3 * k3

        norm = np.linalg.norm(x_next)

        if divergence_norm is not None and norm > divergence_norm:
            states[n_abs + 1] = x_next
            last_local = n + 1
            status = "diverged"
            break

        if not np.all(np.isfinite(x_next)):
            status = "nonfinite_solution"
            break

        states[n_abs + 1] = x_next
        last_local = n + 1
        t_next = times[n_abs + 1]

        # Early stop checks
        if es_enabled:
            if div_enabled:
                if norm > div_norm_es:
                    div_consec_count += 1
                else:
                    div_consec_count = 0
                if prev_norm >= 0.0:
                    if norm > div_growth * prev_norm:
                        growth_consec_count += 1
                    else:
                        growth_consec_count = 0
                prev_norm = norm
                if div_consec_count >= div_consec or growth_consec_count >= div_consec:
                    status = "diverged_early"
                    break
            else:
                prev_norm = norm

            if eq_enabled and equilibria and t_next >= eq_min_t:
                for ki, eq in enumerate(equilibria):
                    diff_n = np.linalg.norm(x_next - eq)
                    try:
                        deriv_n = np.linalg.norm(rhs(t_next, x_next))
                    except Exception:
                        deriv_n = 9999.0
                    if diff_n < eq_t and deriv_n < eq_deriv:
                        eq_consec_counts[ki] += 1
                    else:
                        eq_consec_counts[ki] = 0
                    if eq_consec_counts[ki] >= eq_consec:
                        status = "converged_equilibrium_early"
                        break
                if status == "converged_equilibrium_early":
                    break
        else:
            prev_norm = norm

    # Return only the NEW integration portion (skip prehistory)
    new_start = K - 1
    new_end = K - 1 + last_local + 1
    return (
        times[new_start:new_end],
        states[new_start:new_end],
        status,
    )


# ---------------------------------------------------------------------------
# Public API  (matches the signature used by the rest of /src)
# ---------------------------------------------------------------------------

def efork_integrate(
    system: Any,
    x0: np.ndarray,
    q: float,
    h: float,
    t_final: float,
    memory_mode: str = "full",
    memory_window_length: Optional[int] = None,
    k: float = 0.0,
    eps: float = 1.0,
    use_c_backend: bool = True,
    divergence_norm: Optional[float] = None,
    early_stop_config: Optional[dict] = None,
    equilibria: Optional[List[np.ndarray]] = None,
    history_times: Optional[np.ndarray] = None,
    history_states: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Integrate a Lur'e system using EFORK-3 (Caputo, 0 < q < 1) or Euler for q = 1.

    For ``q == 1.0`` the system is integer-order and is integrated with the
    forward-Euler / trapezoidal predictor-corrector (not EFORK-3, which is
    only defined for 0 < q < 1).

    For ``0 < q < 1`` the published three-stage EFORK Caputo method is used,
    either via the native C backend (fast) or the pure Python reference
    implementation above.

    Parameters
    ----------
    system : Lur'e system with attributes P, b, r, q, psi.
    x0     : Initial condition.
    q      : Caputo fractional order.
    h      : Step size.
    t_final: Integration end time.
    memory_mode : "full" or "window".
    memory_window_length : Steps to keep (window mode only).
    k      : Linearisation gain (used in the deformed RHS).
    eps    : Continuation parameter eta (0 → 1).
    use_c_backend : Attempt to use native C EFORK backend (recommended).
    divergence_norm : Hard-stop norm threshold.
    early_stop_config : Early-stop configuration dict.
    equilibria : List of equilibrium arrays for convergence detection.
    history_times  : External prehistory time array (shape (L,)).
    history_states : External prehistory state array (shape (L, dim)).

    Returns
    -------
    t_arr, x_arr, status
    """
    x0_arr = np.asarray(x0, dtype=float)

    # ── Integer order: standard predictor-corrector (Heun), q=1 not EFORK-3 ──
    if q == 1.0:
        p0 = system.P + k * np.outer(system.b, system.r)

        def rhs_int(x: np.ndarray) -> np.ndarray:
            sigma = float(system.r @ x)
            delta = float(system.psi(sigma)) - k * sigma
            return p0 @ x + eps * system.b * delta

        from ._q1_coefficients import (
            EFORK_Q1_A21,
            EFORK_Q1_A31,
            EFORK_Q1_A32,
            EFORK_Q1_W1,
            EFORK_Q1_W2,
            EFORK_Q1_W3,
        )

        n_steps = int(np.ceil(t_final / h))
        dim = x0_arr.size
        t_arr = np.zeros(n_steps + 1, dtype=float)
        x_arr = np.zeros((n_steps + 1, dim), dtype=float)
        t_arr[0] = 0.0
        x_arr[0] = x0_arr

        x = x0_arr.copy()
        status = "ok"
        last_idx = 0

        esc = early_stop_config or {}
        es_enabled = esc.get("enabled", True)
        div_enabled = esc.get("divergence_enabled", True)
        div_norm_es = esc.get("divergence_norm", 80.0)
        div_consec = esc.get("divergence_consecutive_steps", 5)
        div_growth = esc.get("divergence_growth_factor", 1.25)
        eq_enabled = esc.get("equilibrium_enabled", True)
        eq_t = esc.get("equilibrium_tol", 1e-3)
        eq_deriv = esc.get("equilibrium_derivative_tol", 1e-4)
        eq_consec = esc.get("equilibrium_consecutive_steps", 200)
        eq_min_t = esc.get("equilibrium_min_time", 5.0)

        div_consec_count = 0
        growth_consec_count = 0
        prev_norm = -1.0
        eq_consec_counts = [0] * len(equilibria) if equilibria else []

        for n in range(n_steps):
            t_next = (n + 1) * h
            try:
                k1 = h * rhs_int(x)
                k2 = h * rhs_int(x + EFORK_Q1_A21 * k1)
                k3 = h * rhs_int(x + EFORK_Q1_A31 * k1 + EFORK_Q1_A32 * k2)
                x_next = x + EFORK_Q1_W1 * k1 + EFORK_Q1_W2 * k2 + EFORK_Q1_W3 * k3
            except Exception as exc:
                status = f"solver_exception:{exc}"
                break

            norm = np.linalg.norm(x_next)

            if divergence_norm is not None and norm > divergence_norm:
                status = "diverged"
                x_arr[n + 1] = x_next
                t_arr[n + 1] = t_next
                last_idx = n + 1
                break

            if not np.all(np.isfinite(x_next)):
                status = "nonfinite_solution"
                break

            x = x_next
            x_arr[n + 1] = x
            t_arr[n + 1] = t_next
            last_idx = n + 1

            if es_enabled:
                if div_enabled:
                    if norm > div_norm_es:
                        div_consec_count += 1
                    else:
                        div_consec_count = 0
                    if prev_norm >= 0.0:
                        growth_consec_count = (growth_consec_count + 1) if norm > div_growth * prev_norm else 0
                    prev_norm = norm
                    if div_consec_count >= div_consec or growth_consec_count >= div_consec:
                        status = "diverged_early"
                        break
                else:
                    prev_norm = norm

                if eq_enabled and equilibria and t_next >= eq_min_t:
                    for ki, eq in enumerate(equilibria):
                        diff_n = np.linalg.norm(x_next - eq)
                        try:
                            deriv_n = np.linalg.norm(rhs_int(x_next))
                        except Exception:
                            deriv_n = 9999.0
                        if diff_n < eq_t and deriv_n < eq_deriv:
                            eq_consec_counts[ki] += 1
                        else:
                            eq_consec_counts[ki] = 0
                        if eq_consec_counts[ki] >= eq_consec:
                            status = "converged_equilibrium_early"
                            break
                    if status == "converged_equilibrium_early":
                        break
            else:
                prev_norm = norm

        return t_arr[:last_idx + 1], x_arr[:last_idx + 1], status

    # ── Fractional order 0 < q < 1 ────────────────────────────────────────────
    p0 = system.P + k * np.outer(system.b, system.r)

    def rhs_deformed(t_val: float, x_val: np.ndarray) -> np.ndarray:
        sigma = float(system.r @ x_val)
        delta = float(system.psi(sigma)) - k * sigma
        return p0 @ x_val + eps * system.b * delta

    # Pass the registered system only for pure nonlinear (no deformation)
    sys_to_pass = system if (abs(k) < 1e-12 and abs(eps - 1.0) < 1e-12) else None

    if use_c_backend:
        try:
            t_arr, x_arr, status, info = fractional_integrate(
                rhs=rhs_deformed,
                x0=x0_arr,
                q=q,
                h=h,
                t_final=t_final,
                method="efork",
                memory_mode=memory_mode,
                memory_window_length=memory_window_length,
                history_times=history_times,
                history_states=history_states,
                system=sys_to_pass,
                use_c_backend=True,
                divergence_norm=divergence_norm,
                return_history=True,
                allow_python_fallback=False,
                early_stop_config=early_stop_config,
                equilibria=equilibria,
            )
            return t_arr, x_arr, status
        except Exception:
            pass  # Fall through to pure Python EFORK-3

    # Pure Python EFORK-3 (exact published algorithm)
    return _python_efork3_integrate(
        rhs=rhs_deformed,
        x0=x0_arr,
        q=q,
        h=h,
        t_final=t_final,
        divergence_norm=divergence_norm,
        history_times=history_times,
        history_states=history_states,
        memory_mode=memory_mode,
        memory_window_length=memory_window_length,
        early_stop_config=early_stop_config,
        equilibria=equilibria,
    )
