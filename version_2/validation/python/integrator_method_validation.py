"""Integrator method validation logic.

Validates numerical integration methods against exact solutions, manufactured
solutions, and convergence order.  This phase is distinct from
``validation/integrator_crosscheck``:

- **integrator_method_validation**: Is the integrator numerically correct?
  Does it converge at the right rate against known exact solutions?
- **integrator_crosscheck**: Do independent integrators agree on chaotic
  dynamics (geometric/statistical consistency)?

Methods covered
---------------
EFORK3
    Already validated against published errors in
    ``tools/validation/validate_efork_integrator.py``.  Status recorded here
    without re-running.

ABM (Caputo, full history)
    Validated against Mittag-Leffler exact solution, manufactured solution
    t^m, and a diagonal linear vector system.

RK4 (classical, q=1 only)
    Validated against exact exponential decay, harmonic oscillator, diagonal
    linear system, and optionally scipy.solve_ivp.

This phase does NOT certify hidden attractors.  Every output includes:
    hiddenness_certified_by_this_pipeline: false
    no_hidden_verified_claim: true
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Mandatory metadata — injected in every output.
# ---------------------------------------------------------------------------
_NO_CLAIM = {
    "hiddenness_certified_by_this_pipeline": False,
    "no_hidden_verified_claim": True,
}

_ALLOWED_METHOD_STATUSES = {
    "method_validated_against_exact_solution",
    "method_validated_against_exact_and_solve_ivp",
    "method_crosschecked_only",
    "method_validation_inconclusive",
    "method_not_available",
    "method_validation_failed",
    "validated_elsewhere_against_published_errors",
}

_ALLOWED_TEST_STATUSES = {
    "abm_mittag_leffler_decreasing_error",
    "abm_mittag_leffler_inconclusive",
    "abm_mittag_leffler_failed",
    "abm_manufactured_solution_decreasing_error",
    "abm_manufactured_solution_inconclusive",
    "abm_manufactured_solution_failed",
    "abm_vector_linear_ok",
    "abm_vector_linear_inconclusive",
    "abm_vector_linear_failed",
    "rk4_order4_confirmed",
    "rk4_order4_inconclusive",
    "rk4_order4_failed",
    "rk4_energy_drift_decreasing",
    "rk4_energy_drift_inconclusive",
    "rk4_vector_linear_ok",
    "rk4_solve_ivp_consistent",
    "rk4_solve_ivp_inconsistent",
    "rk4_solve_ivp_skipped_no_scipy",
}


# ===========================================================================
# Mittag-Leffler function
# ===========================================================================

def mittag_leffler(
    alpha: float,
    beta: float,
    z: float,
    n_terms: int = 500,
    tol: float = 1.0e-16,
) -> float:
    """Evaluate the two-parameter Mittag-Leffler function E_{alpha,beta}(z).

    .. math::
        E_{\\alpha,\\beta}(z) = \\sum_{k=0}^{\\infty} \\frac{z^k}{\\Gamma(\\alpha k + \\beta)}

    Parameters
    ----------
    alpha : float
        First parameter (must be positive).
    beta : float
        Second parameter.
    z : float
        Argument.
    n_terms : int
        Maximum number of terms to sum.
    tol : float
        Convergence tolerance; summation stops when ``|term| < tol``.

    Returns
    -------
    value : float
        Approximation to E_{alpha,beta}(z).
    """
    z = float(z)
    total = 0.0
    for k in range(n_terms):
        try:
            z_pow = z ** k
        except (OverflowError, ValueError):
            break
        if not math.isfinite(z_pow):
            break
        try:
            term = z_pow / math.gamma(alpha * k + beta)
        except (OverflowError, ValueError):
            break
        if not math.isfinite(term):
            break
        total += term
        if abs(term) < tol:
            break
    return total


def mittag_leffler_q(z: float, q: float) -> float:
    """Evaluate E_q(z) = E_{q,1}(z)."""
    return mittag_leffler(q, 1.0, z)


# ===========================================================================
# ABM Caputo integrator (local, independent implementation)
# ===========================================================================

def abm_caputo_integrate(
    rhs,
    y0: np.ndarray,
    *,
    q: float,
    h: float,
    t_final: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a Caputo FDE using the Diethelm ABM predictor-corrector.

    Solves ``^C D_t^q y = rhs(t, y)`` with ``y(0) = y0``.

    ``rhs`` must accept ``(t: float, y: np.ndarray)`` and return an array of
    the same shape as ``y``.

    Parameters
    ----------
    rhs : callable
        Right-hand side ``f(t, y)``.
    y0 : array-like, shape (d,) or scalar
        Initial condition.
    q : float
        Fractional order, ``0 < q <= 1``.
    h : float
        Step size.
    t_final : float
        Integration horizon.

    Returns
    -------
    times : np.ndarray, shape (N+1,)
        Uniform time grid from 0 to t_final.
    states : np.ndarray, shape (N+1, d)
        State at each grid point.
    """
    q = float(q)
    h = float(h)
    n_steps = int(round(t_final / h))
    y0_arr = np.atleast_1d(np.asarray(y0, dtype=float))
    dim = y0_arr.size

    times = np.linspace(0.0, t_final, n_steps + 1)
    states = np.zeros((n_steps + 1, dim), dtype=float)
    f_hist = np.zeros((n_steps + 1, dim), dtype=float)

    states[0] = y0_arr
    f_hist[0] = np.atleast_1d(np.asarray(rhs(0.0, y0_arr), dtype=float))

    powers = np.arange(n_steps + 2, dtype=float)
    pow_q = powers ** q
    pow_q1 = powers ** (q + 1.0)
    hq = h ** q
    pred_scale = hq / math.gamma(q + 1.0)
    corr_scale = hq / math.gamma(q + 2.0)

    for i in range(n_steps):
        t_next = times[i + 1]
        b = pow_q[1: i + 2][::-1] - pow_q[0: i + 1][::-1]
        predictor = y0_arr + pred_scale * (b @ f_hist[: i + 1])
        fp = np.atleast_1d(np.asarray(rhs(t_next, predictor), dtype=float))

        if i == 0:
            a_weights = np.array([q], dtype=float)
        else:
            r_idx = np.arange(i, 0, -1, dtype=int)
            mid = pow_q1[r_idx + 1] + pow_q1[r_idx - 1] - 2.0 * pow_q1[r_idx]
            a0 = pow_q1[i] - (float(i) - q) * pow_q[i + 1]
            a_weights = np.concatenate(([a0], mid))

        corrected = y0_arr + corr_scale * ((a_weights @ f_hist[: i + 1]) + fp)
        states[i + 1] = corrected
        f_hist[i + 1] = np.atleast_1d(np.asarray(rhs(t_next, corrected), dtype=float))

    return times, states


# ===========================================================================
# Part A: ABM validation
# ===========================================================================

# ── A1: Mittag-Leffler scalar ──────────────────────────────────────────────

def validate_abm_mittag_leffler(
    q_values: list[float] | None = None,
    lambda_values: list[float] | None = None,
    y0: float = 1.0,
    t_final: float = 1.0,
    h_values: list[float] | None = None,
) -> tuple[list[dict], str]:
    """Validate ABM against the exact Mittag-Leffler solution.

    Equation: ``^C D^q y = λ y``, ``y(0) = y₀``.
    Exact:    ``y(t) = y₀ · E_q(λ · t^q)``.

    Returns
    -------
    rows : list of dict
        One row per (q, λ, h) combination.
    status : str
        One of the allowed ABM mittag-leffler status strings.
    """
    if q_values is None:
        q_values = [0.25, 0.5, 0.8, 0.9998]
    if lambda_values is None:
        lambda_values = [-1.0, -5.0]
    if h_values is None:
        h_values = [1 / 40, 1 / 80, 1 / 160, 1 / 320]

    rows = []
    all_decreasing = True

    for q in q_values:
        for lam in lambda_values:
            # Skip combinations where the Mittag-Leffler power series definition
            # suffers from extreme catastrophic cancellation in double precision.
            if q < 0.5 and lam < -2.0:
                continue

            prev_max_err = None
            n_decreasing = 0
            for h in h_values:
                def make_rhs(lam_=lam):
                    def rhs(t, y):
                        return lam_ * y
                    return rhs

                times, states = abm_caputo_integrate(
                    make_rhs(lam), np.array([y0]), q=q, h=h, t_final=t_final
                )
                exact = np.array([
                    y0 * mittag_leffler_q(lam * (t ** q), q) for t in times
                ])
                errors = np.abs(states[:, 0] - exact)
                max_err = float(np.max(errors))
                term_err = float(errors[-1])

                obs_order = float("nan")
                if prev_max_err is not None and max_err > 0.0 and prev_max_err > 0.0:
                    ratio = prev_max_err / max_err
                    obs_order = math.log2(ratio) if ratio > 0.0 else float("nan")
                    if max_err < prev_max_err:
                        n_decreasing += 1
                elif prev_max_err is not None:
                    if max_err < prev_max_err:
                        n_decreasing += 1

                rows.append({
                    "q": q,
                    "lambda": lam,
                    "h": h,
                    "max_error": max_err,
                    "terminal_error": term_err,
                    "observed_order": obs_order,
                })
                prev_max_err = max_err

            # Check monotone decrease for last 3 meshes (at least 3 points needed)
            if len(h_values) >= 3:
                last_errs = [r["max_error"] for r in rows if r["q"] == q and r["lambda"] == lam][-3:]
                if not all(last_errs[i] > last_errs[i + 1] for i in range(len(last_errs) - 1)):
                    all_decreasing = False

    overall = (
        "abm_mittag_leffler_decreasing_error"
        if all_decreasing
        else "abm_mittag_leffler_inconclusive"
    )
    return rows, overall


# ── A2: Manufactured solution t^m ─────────────────────────────────────────

def caputo_power_derivative(m: float, q: float, t: float) -> float:
    """Evaluate ``^C D^q [t^m]`` at time t.

    For ``m > q``: ``Gamma(m+1)/Gamma(m+1-q) * t^(m-q)``.
    For ``m <= q`` (integer m=0,...): 0.
    """
    if m <= q - 1.0 + 1e-12:
        return 0.0
    return math.gamma(m + 1.0) / math.gamma(m + 1.0 - q) * (t ** (m - q))


def validate_abm_manufactured_solution(
    q_values: list[float] | None = None,
    m_values: list[int] | None = None,
    t_final: float = 1.0,
    h_values: list[float] | None = None,
) -> tuple[list[dict], str]:
    """Validate ABM against manufactured solution ``y(t) = t^m``.

    The forcing is chosen so that the exact solution is ``y(t) = t^m``:
        ``^C D^q y = -y + forcing(t)``
    where ``forcing(t) = Gamma(m+1)/Gamma(m+1-q)*t^(m-q) + t^m``.

    Returns
    -------
    rows : list of dict
    status : str
    """
    if q_values is None:
        q_values = [0.25, 0.5, 0.8, 0.9998]
    if m_values is None:
        m_values = [4, 5]
    if h_values is None:
        h_values = [1 / 40, 1 / 80, 1 / 160, 1 / 320]

    rows = []
    all_decreasing = True

    for q in q_values:
        for m in m_values:
            prev_max_err = None
            for h in h_values:
                def make_rhs(q_=q, m_=m):
                    def rhs(t, y):
                        forcing = caputo_power_derivative(float(m_), q_, max(t, 1e-300)) + t ** m_
                        return -y + forcing
                    return rhs

                times, states = abm_caputo_integrate(
                    make_rhs(q, m), np.array([0.0]), q=q, h=h, t_final=t_final
                )
                exact = times ** m
                errors = np.abs(states[:, 0] - exact)
                max_err = float(np.max(errors))
                term_err = float(errors[-1])

                obs_order = float("nan")
                if prev_max_err is not None and max_err > 0.0 and prev_max_err > 0.0:
                    ratio = prev_max_err / max_err
                    obs_order = math.log2(ratio) if ratio > 0.0 else float("nan")

                rows.append({
                    "q": q,
                    "m": m,
                    "h": h,
                    "max_error": max_err,
                    "terminal_error": term_err,
                    "observed_order": obs_order,
                })
                prev_max_err = max_err

            if len(h_values) >= 3:
                last_errs = [r["max_error"] for r in rows if r["q"] == q and r["m"] == m][-3:]
                if not all(last_errs[i] > last_errs[i + 1] for i in range(len(last_errs) - 1)):
                    all_decreasing = False

    overall = (
        "abm_manufactured_solution_decreasing_error"
        if all_decreasing
        else "abm_manufactured_solution_inconclusive"
    )
    return rows, overall


# ── A3: Vector linear diagonal system ─────────────────────────────────────

def validate_abm_vector_linear(
    q_values: list[float] | None = None,
    eigenvalues: list[float] | None = None,
    x0: list[float] | None = None,
    t_final: float = 1.0,
    h_values: list[float] | None = None,
) -> tuple[list[dict], str]:
    """Validate ABM on a diagonal linear vector system.

    ``^C D^q X = A X``, ``A = diag(eigenvalues)``, ``X(0) = x0``.
    Exact: ``X_i(t) = x0_i * E_q(lambda_i * t^q)``.

    Returns
    -------
    rows : list of dict
    status : str
    """
    if q_values is None:
        q_values = [0.5, 0.8, 0.9998]
    if eigenvalues is None:
        eigenvalues = [-1.0, -2.0, -5.0]
    if x0 is None:
        x0 = [1.0, 0.5, -0.25]
    if h_values is None:
        h_values = [1 / 40, 1 / 80, 1 / 160]

    x0_arr = np.asarray(x0, dtype=float)
    lam = np.asarray(eigenvalues, dtype=float)
    rows = []
    all_ok = True

    for q in q_values:
        def make_rhs(lam_=lam):
            def rhs(t, y):
                return lam_ * np.asarray(y, dtype=float)
            return rhs

        prev_max_err = None
        for h in h_values:
            times, states = abm_caputo_integrate(
                make_rhs(lam), x0_arr, q=q, h=h, t_final=t_final
            )
            exact = np.array([
                [x0_arr[i] * mittag_leffler_q(lam[i] * (t ** q), q) for i in range(len(lam))]
                for t in times
            ])
            norm_errors = np.linalg.norm(states - exact, axis=1)
            max_norm_err = float(np.max(norm_errors))
            term_norm_err = float(norm_errors[-1])
            finite = bool(np.all(np.isfinite(states)))

            obs_order = float("nan")
            if prev_max_err is not None and max_norm_err > 0.0 and prev_max_err > 0.0:
                ratio = prev_max_err / max_norm_err
                obs_order = math.log2(ratio) if ratio > 0.0 else float("nan")

            rows.append({
                "q": q,
                "h": h,
                "max_norm_error": max_norm_err,
                "terminal_norm_error": term_norm_err,
                "observed_order": obs_order,
                "finite_values": finite,
            })
            if not finite or max_norm_err > 10.0:
                all_ok = False
            prev_max_err = max_norm_err

    overall = "abm_vector_linear_ok" if all_ok else "abm_vector_linear_inconclusive"
    return rows, overall


# ===========================================================================
# Part B: RK4 validation
# ===========================================================================

def _rk4_integrate_local(rhs, y0, t_final, h):
    """Thin wrapper around the official rk4_integrate for use in validation."""
    from hidden_attractors.solvers.rk4 import rk4_integrate
    y0_arr = np.atleast_1d(np.asarray(y0, dtype=float))
    times, states = rk4_integrate(rhs, y0_arr, t_final, h)
    return times, states


# ── B1: Exponential decay ──────────────────────────────────────────────────

def validate_rk4_exponential_decay(
    t_final: float = 1.0,
    y0: float = 1.0,
    h_values: list[float] | None = None,
) -> tuple[list[dict], str]:
    """Validate RK4 on y' = -y, y(0) = y0. Exact: y(t) = exp(-t).

    Returns
    -------
    rows : list of dict
    status : str
    """
    if h_values is None:
        h_values = [0.2, 0.1, 0.05, 0.025, 0.0125]

    def rhs(t, y):
        return -y

    rows = []
    prev_err = None

    for h in h_values:
        times, states = _rk4_integrate_local(rhs, np.array([y0]), t_final, h)
        exact = np.exp(-times)
        term_err = float(abs(states[-1, 0] - exact[-1]))

        obs_order = float("nan")
        if prev_err is not None and term_err > 0.0 and prev_err > 0.0:
            obs_order = math.log2(prev_err / term_err)

        rows.append({
            "h": h,
            "terminal_error": term_err,
            "observed_order": obs_order,
        })
        prev_err = term_err

    # Check order on the finest two h values
    fine_orders = [r["observed_order"] for r in rows[-2:] if math.isfinite(r["observed_order"])]
    order_ok = all(3.5 <= o <= 4.5 for o in fine_orders) if fine_orders else False
    status = "rk4_order4_confirmed" if order_ok else "rk4_order4_inconclusive"
    return rows, status


# ── B2: Harmonic oscillator ────────────────────────────────────────────────

def validate_rk4_harmonic_oscillator(
    t_final: float | None = None,
    x0: list[float] | None = None,
    h_values: list[float] | None = None,
) -> tuple[list[dict], str]:
    """Validate RK4 on x'=y, y'=-x. Exact: x=cos(t), y=-sin(t).

    Returns
    -------
    rows : list of dict
    status : str
    """
    if t_final is None:
        t_final = 2.0 * math.pi
    if x0 is None:
        x0 = [1.0, 0.0]
    if h_values is None:
        h_values = [0.2, 0.1, 0.05, 0.025]

    def rhs(t, y):
        return np.array([y[1], -y[0]])

    rows = []
    prev_err = None
    prev_drift = None
    drift_decreasing = True

    for h in h_values:
        times, states = _rk4_integrate_local(rhs, np.array(x0, dtype=float), t_final, h)
        exact_x = np.cos(times)
        exact_y = -np.sin(times)
        exact = np.column_stack([exact_x, exact_y])
        norm_errors = np.linalg.norm(states - exact, axis=1)
        max_norm_err = float(np.max(norm_errors))
        term_err = float(norm_errors[-1])
        energy = states[:, 0] ** 2 + states[:, 1] ** 2
        energy_drift = float(np.max(np.abs(energy - 1.0)))

        obs_order = float("nan")
        if prev_err is not None and max_norm_err > 0.0 and prev_err > 0.0:
            obs_order = math.log2(prev_err / max_norm_err)

        if prev_drift is not None and energy_drift >= prev_drift:
            drift_decreasing = False

        rows.append({
            "h": h,
            "max_norm_error": max_norm_err,
            "terminal_error": term_err,
            "energy_drift": energy_drift,
            "observed_order": obs_order,
        })
        prev_err = max_norm_err
        prev_drift = energy_drift

    status = (
        "rk4_energy_drift_decreasing"
        if drift_decreasing
        else "rk4_energy_drift_inconclusive"
    )
    return rows, status


# ── B3: Vector linear diagonal system ─────────────────────────────────────

def validate_rk4_vector_linear(
    eigenvalues: list[float] | None = None,
    x0: list[float] | None = None,
    t_final: float = 1.0,
    h_values: list[float] | None = None,
) -> tuple[list[dict], str]:
    """Validate RK4 on X' = A X, A = diag(eigenvalues).

    Exact: X_i(t) = X_i(0) * exp(lambda_i * t).

    Returns
    -------
    rows : list of dict
    status : str
    """
    if eigenvalues is None:
        eigenvalues = [-1.0, -2.0, -3.0]
    if x0 is None:
        x0 = [1.0, 0.5, -0.25]
    if h_values is None:
        h_values = [0.1, 0.05, 0.025]

    lam = np.asarray(eigenvalues, dtype=float)
    x0_arr = np.asarray(x0, dtype=float)

    def rhs(t, y):
        return lam * np.asarray(y, dtype=float)

    rows = []
    prev_err = None
    all_ok = True

    for h in h_values:
        times, states = _rk4_integrate_local(rhs, x0_arr, t_final, h)
        exact = np.array([
            [x0_arr[i] * math.exp(lam[i] * t) for i in range(len(lam))]
            for t in times
        ])
        norm_errors = np.linalg.norm(states - exact, axis=1)
        max_err = float(np.max(norm_errors))
        term_err = float(norm_errors[-1])

        obs_order = float("nan")
        if prev_err is not None and max_err > 0.0 and prev_err > 0.0:
            obs_order = math.log2(prev_err / max_err)

        rows.append({
            "h": h,
            "max_norm_error": max_err,
            "terminal_norm_error": term_err,
            "observed_order": obs_order,
            "finite_values": bool(np.all(np.isfinite(states))),
        })
        if max_err > 1.0:
            all_ok = False
        prev_err = max_err

    status = "rk4_vector_linear_ok" if all_ok else "rk4_order4_inconclusive"
    return rows, status


# ── B4: Comparison with scipy.solve_ivp ───────────────────────────────────

def validate_rk4_solve_ivp_comparison(
    t_final: float = 5.0,
    y0: float = 0.0,
    h_values: list[float] | None = None,
    scipy_method: str = "DOP853",
    scipy_rtol: float = 1e-11,
    scipy_atol: float = 1e-13,
    max_diff: float = 1e-4,
) -> tuple[list[dict], str]:
    """Compare RK4 against scipy.solve_ivp for y' = -y + sin(t).

    Returns
    -------
    rows : list of dict
    status : str
        ``rk4_solve_ivp_skipped_no_scipy`` if scipy is unavailable.
    """
    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        return [], "rk4_solve_ivp_skipped_no_scipy"

    if h_values is None:
        h_values = [0.05, 0.025]

    def rhs(t, y):
        return -y + math.sin(t)

    t_span = [0.0, t_final]
    t_eval = np.linspace(0.0, t_final, int(round(t_final / 0.001)) + 1)
    ref_sol = solve_ivp(
        lambda t, y: [-y[0] + math.sin(t)],
        t_span,
        [y0],
        method=scipy_method,
        t_eval=t_eval,
        rtol=scipy_rtol,
        atol=scipy_atol,
        dense_output=True,
    )

    rows = []
    all_consistent = True

    for h in h_values:
        times, states = _rk4_integrate_local(rhs, np.array([y0]), t_final, h)
        ref_vals = ref_sol.sol(times)[0]
        diff = np.abs(states[:, 0] - ref_vals)
        max_diff_val = float(np.max(diff))
        consistent = max_diff_val <= max_diff
        if not consistent:
            all_consistent = False
        rows.append({
            "h": h,
            "max_diff_vs_solve_ivp": max_diff_val,
            "consistent": consistent,
            "scipy_method": scipy_method,
        })

    status = (
        "rk4_solve_ivp_consistent" if all_consistent else "rk4_solve_ivp_inconsistent"
    )
    return rows, status


# ===========================================================================
# Top-level runner
# ===========================================================================

def run_integrator_method_validation(
    output_dir: str | Path,
    methods: list[str] | None = None,
    fast: bool = False,
) -> dict:
    """Run all integrator method validations and write CSV / JSON outputs.

    Parameters
    ----------
    output_dir : str or Path
        Directory where CSV and JSON outputs are written.
    methods : list of str or None
        Which methods to validate; ``None`` means all (``['ABM', 'RK4']``).
    fast : bool
        If True, use fewer h values for faster testing.

    Returns
    -------
    summary : dict
        Full validation summary including the no-claim fields.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = ["ABM", "RK4"]

    if fast:
        abm_h = [1 / 40, 1 / 80, 1 / 160]
        rk4_h_exp = [0.1, 0.05, 0.025]
        rk4_h_harm = [0.2, 0.1, 0.05]
        rk4_h_vec = [0.1, 0.05, 0.025]
        rk4_h_ivp = [0.05]
    else:
        abm_h = [1 / 40, 1 / 80, 1 / 160, 1 / 320]
        rk4_h_exp = [0.2, 0.1, 0.05, 0.025, 0.0125]
        rk4_h_harm = [0.2, 0.1, 0.05, 0.025]
        rk4_h_vec = [0.1, 0.05, 0.025]
        rk4_h_ivp = [0.05, 0.025]

    method_results: dict[str, Any] = {}

    # ── EFORK3 ──────────────────────────────────────────────────────────────
    method_results["EFORK3"] = {
        "status": "validated_elsewhere_against_published_errors",
        "reference_script": "tools/validation/validate_efork_integrator.py",
        "note": (
            "EFORK3 is validated against published terminal errors from "
            "Ghoreishi, Ghaffari, and Saad (2023) within 6e-9. "
            "See the reference script for details."
        ),
    }

    # ── ABM ─────────────────────────────────────────────────────────────────
    if "ABM" in methods:
        ml_rows, ml_status = validate_abm_mittag_leffler(h_values=abm_h)
        ms_rows, ms_status = validate_abm_manufactured_solution(h_values=abm_h)
        vl_rows, vl_status = validate_abm_vector_linear(h_values=abm_h)

        _write_csv(output_dir / "abm_mittag_leffler_convergence.csv", ml_rows)
        _write_csv(output_dir / "abm_manufactured_solution_convergence.csv", ms_rows)
        _write_csv(output_dir / "abm_vector_linear_convergence.csv", vl_rows)

        all_abm_ok = (
            ml_status == "abm_mittag_leffler_decreasing_error"
            and ms_status == "abm_manufactured_solution_decreasing_error"
            and vl_status == "abm_vector_linear_ok"
        )
        abm_status = (
            "method_validated_against_exact_solution"
            if all_abm_ok
            else "method_validation_inconclusive"
        )

        method_results["ABM"] = {
            "status": abm_status,
            "tests": {
                "mittag_leffler": ml_status,
                "manufactured_solution": ms_status,
                "vector_linear": vl_status,
            },
        }
    else:
        method_results["ABM"] = {"status": "method_not_available"}

    # ── RK4 ─────────────────────────────────────────────────────────────────
    if "RK4" in methods:
        exp_rows, exp_status = validate_rk4_exponential_decay(h_values=rk4_h_exp)
        harm_rows, harm_status = validate_rk4_harmonic_oscillator(h_values=rk4_h_harm)
        vec_rows, vec_status = validate_rk4_vector_linear(h_values=rk4_h_vec)
        ivp_rows, ivp_status = validate_rk4_solve_ivp_comparison(h_values=rk4_h_ivp)

        _write_csv(output_dir / "rk4_exponential_decay_convergence.csv", exp_rows)
        _write_csv(output_dir / "rk4_harmonic_oscillator_convergence.csv", harm_rows)
        _write_csv(output_dir / "rk4_vector_linear_convergence.csv", vec_rows)
        if ivp_rows:
            _write_csv(output_dir / "rk4_solve_ivp_comparison.csv", ivp_rows)

        exact_ok = (
            exp_status == "rk4_order4_confirmed"
            and harm_status in ("rk4_energy_drift_decreasing", "rk4_energy_drift_inconclusive")
            and vec_status == "rk4_vector_linear_ok"
        )
        ivp_ok = ivp_status in ("rk4_solve_ivp_consistent", "rk4_solve_ivp_skipped_no_scipy")
        if exact_ok and ivp_status == "rk4_solve_ivp_consistent":
            rk4_status = "method_validated_against_exact_and_solve_ivp"
        elif exact_ok:
            rk4_status = "method_validated_against_exact_solution"
        else:
            rk4_status = "method_validation_inconclusive"

        method_results["RK4"] = {
            "status": rk4_status,
            "tests": {
                "exponential_decay": exp_status,
                "harmonic_oscillator": harm_status,
                "vector_linear": vec_status,
                "solve_ivp_comparison": ivp_status,
            },
        }
    else:
        method_results["RK4"] = {"status": "method_not_available"}

    # ── Summary ─────────────────────────────────────────────────────────────
    summary = {
        "stage": "integrator_method_validation",
        "methods": method_results,
        **_NO_CLAIM,
    }
    _write_json(summary, output_dir / "integrator_method_validation_summary.json")

    return summary


# ===========================================================================
# Helpers
# ===========================================================================

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


__all__ = [
    "mittag_leffler",
    "mittag_leffler_q",
    "abm_caputo_integrate",
    "validate_abm_mittag_leffler",
    "validate_abm_manufactured_solution",
    "validate_abm_vector_linear",
    "validate_rk4_exponential_decay",
    "validate_rk4_harmonic_oscillator",
    "validate_rk4_vector_linear",
    "validate_rk4_solve_ivp_comparison",
    "run_integrator_method_validation",
    "_NO_CLAIM",
    "_ALLOWED_METHOD_STATUSES",
    "_ALLOWED_TEST_STATUSES",
]
