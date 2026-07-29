"""Numerical helpers used only by the completed paper07 non-smooth validation.

Only the fixed numerical primitives required to reproduce the recorded
validation case are exposed here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.special import gamma as gamma_func

from hidden_attractors.models.chua import ChuaParameters
from hidden_attractors.seed_generation.chua import (
    chua_gain,
    chua_matrices,
)
from hidden_attractors.seed_generation.core import (
    complex_dtype,
    fractional_iomega_power,
    real_dtype,
    validate_fractional_order,
)


def _historical_transfer(
    omega: float,
    q: float,
    pmat: np.ndarray,
    qvec: np.ndarray,
    rvec: np.ndarray,
) -> complex:
    """Return the transfer value in the sign convention of the fixed record."""

    matrix = (
        pmat.astype(complex_dtype)
        - fractional_iomega_power(omega, q)
        * np.eye(3, dtype=complex_dtype)
    )
    return complex_dtype(
        (
            rvec.astype(complex_dtype).reshape(1, -1)
            @ np.linalg.inv(matrix)
            @ qvec.astype(complex_dtype).reshape(-1, 1)
        )[0, 0]
    )


def biased_saturation_df(
    A: float,
    c: float,
    g: float,
    n_theta: int = 8192,
) -> tuple[float, float]:
    """Return the DC and first-harmonic terms of the fixed saturation model."""

    if A < 1e-6:
        return 0.0, 0.0
    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        n_theta,
        endpoint=False,
        dtype=real_dtype,
    )
    sigma = c + A * np.cos(theta)
    psi = g * np.clip(sigma, -1.0, 1.0)
    psi0 = float(np.mean(psi))
    psi1 = 2.0 * float(np.mean(psi * np.cos(theta)))
    return psi0, psi1 / A


def _biased_saturation_residual(
    A: float,
    c: float,
    omega: float,
    params: ChuaParameters,
    q: float,
    n_theta: int = 8192,
) -> np.ndarray:
    if A < 1e-6:
        return np.array([c, 1e2, 1e2], dtype=float)

    g = chua_gain(params)
    psi0, N1 = biased_saturation_df(A, c, g, n_theta)
    pmat, qvec, rvec = chua_matrices(params)

    try:
        x_bar = np.linalg.solve(pmat, -qvec * psi0)
        F0 = c - float(rvec @ x_bar)
    except np.linalg.LinAlgError:
        F0 = 1e3

    try:
        Wq = _historical_transfer(omega, q, pmat, qvec, rvec)
    except np.linalg.LinAlgError:
        Wq = 0.0

    term = 1.0 + Wq * N1
    return np.array([F0, float(term.real), float(term.imag)], dtype=float)


def find_biased_branches(
    params: ChuaParameters,
    q: float,
    s2_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reproduce the fixed biased-root calculation used by paper07."""

    q_val = validate_fractional_order(q)
    n_theta = int(s2_cfg.get("n_theta", 8192))

    A_grid = np.linspace(*s2_cfg["A_range"], int(s2_cfg["n_A"]))
    c_grid = np.linspace(*s2_cfg["c_range"], int(s2_cfg["n_c"]))
    omega_grid = np.linspace(*s2_cfg["omega_range"], int(s2_cfg["n_omega"]))
    tol = float(s2_cfg["residual_tol"])

    def residual(x: np.ndarray) -> np.ndarray:
        return _biased_saturation_residual(
            x[0],
            x[1],
            x[2],
            params,
            q_val,
            n_theta,
        )

    raw: list[tuple[float, float, float, float]] = []
    for A0 in A_grid:
        for c0 in c_grid:
            for w0 in omega_grid:
                try:
                    result = least_squares(
                        residual,
                        x0=[A0, c0, w0],
                        bounds=([1e-6, -12.0, 0.1], [25.0, 12.0, 8.0]),
                        method="trf",
                        ftol=1e-10,
                        xtol=1e-10,
                        gtol=1e-8,
                        max_nfev=300,
                    )
                    norm = float(np.linalg.norm(residual(result.x)))
                    if result.success and norm < tol:
                        raw.append((*result.x, norm))
                except Exception:
                    continue

    A_tol = c_tol = w_tol = 1e-3
    unique: list[dict[str, Any]] = []
    for A_c, c_c, w_c, res_c in raw:
        if A_c < 0.5 or not (0.5 <= w_c <= 6.0):
            continue
        duplicate = False
        for existing in unique:
            if (
                abs(A_c - existing["A"]) < A_tol
                and abs(c_c - existing["c"]) < c_tol
                and abs(w_c - existing["omega"]) < w_tol
            ):
                if res_c < existing["residual_norm"]:
                    existing.update(
                        {
                            "A": A_c,
                            "c": c_c,
                            "omega": w_c,
                            "residual_norm": res_c,
                        }
                    )
                duplicate = True
                break
        if not duplicate:
            unique.append(
                {
                    "A": A_c,
                    "c": c_c,
                    "omega": w_c,
                    "residual_norm": res_c,
                }
            )

    unique.sort(key=lambda row: row["residual_norm"])
    return unique


def build_biased_seed(
    params: ChuaParameters,
    q: float,
    A: float,
    c: float,
    omega: float,
    psi0: float,
    N1: float,
) -> dict[str, Any]:
    """Reconstruct the fixed paper07 biased harmonic seed."""

    q_val = validate_fractional_order(q)
    pmat, qvec, _rvec = chua_matrices(params)
    x_bar = np.linalg.solve(pmat, -qvec * psi0)
    lam = fractional_iomega_power(omega, q_val)
    matrix = (
        lam * np.eye(3, dtype=complex_dtype)
        - pmat.astype(complex_dtype)
    )
    X1 = np.linalg.solve(matrix, qvec.astype(complex_dtype)) * N1 * A
    return {
        "seed": x_bar + np.real(X1),
        "x_bar": x_bar,
        "Re_X1": np.real(X1),
        "Im_X1": np.imag(X1),
    }


def run_affine_continuation(
    params: ChuaParameters,
    q: float,
    h: float,
    seed_x0: np.ndarray,
    A: float,
    c: float,
    psi0: float,
    N1: float,
    lambda_values: list[float],
    t_transient: float,
    t_keep: float,
    div_threshold: float,
) -> list[dict[str, Any]]:
    """Reproduce the full-history affine continuation in the fixed record."""

    dim = 3
    h = float(h)
    q = float(q)
    nsteps_tr = int(np.ceil(t_transient / h))
    nsteps_kp = int(np.ceil(t_keep / h))
    steps_per_stage = nsteps_tr + nsteps_kp
    total_new = len(lambda_values) * steps_per_stage

    t_arr = np.zeros(1 + total_new, dtype=float)
    x_arr = np.zeros((1 + total_new, dim), dtype=float)
    f_arr = np.zeros((1 + total_new, dim), dtype=float)
    t_arr[0] = 0.0
    x_arr[0] = seed_x0

    pmat, qvec, rvec = chua_matrices(params)
    P_aff = pmat + N1 * np.outer(qvec, rvec)
    c_aff = qvec * (psi0 - N1 * c)
    g = params.m0 - params.m1

    def rhs(x: np.ndarray, eta: float) -> np.ndarray:
        sigma = float(rvec @ x)
        psi_val = g * np.clip(sigma, -1.0, 1.0)
        return (
            P_aff @ x
            + c_aff
            + eta * qvec * (psi_val - psi0 - N1 * (sigma - c))
        )

    f_arr[0] = rhs(x_arr[0], lambda_values[0])
    powers = np.arange(total_new + 3, dtype=float)
    pow_q = powers**q
    pow_q1 = powers ** (q + 1.0)
    hq = h**q
    pred_sc = hq / float(gamma_func(q + 1.0))
    gq2 = float(gamma_func(q + 2.0))
    corr_sc = hq / gq2 if abs(gq2) > 1e-15 else 0.0

    records: list[dict[str, Any]] = []
    curr_n = 0
    diverged = False

    for eta in lambda_values:
        if diverged:
            break
        x_in = x_arr[curr_n].copy()
        stage_ok = True

        for local_step in range(steps_per_stage):
            n = curr_n + local_step
            j_r = np.arange(0, n + 1)
            b_w = pow_q[n + 1 - j_r] - pow_q[n - j_r]
            pred = x_arr[0] + pred_sc * (b_w @ f_arr[0 : n + 1])

            fp = rhs(pred, eta)
            n_p = n
            a0 = (
                float(n_p) ** (q + 1)
                - (float(n_p) - q) * (float(n_p) + 1) ** q
            )
            if n_p > 0:
                mid = n - np.arange(1, n + 1)
                a_mid = (
                    pow_q1[mid + 2]
                    + pow_q1[mid]
                    - 2.0 * pow_q1[mid + 1]
                )
                a_w = np.concatenate(([a0], a_mid))
            else:
                a_w = np.array([a0])

            corrected = (
                x_arr[0]
                + corr_sc * ((a_w @ f_arr[0 : n + 1]) + fp)
            )
            norm = np.linalg.norm(corrected)

            if norm > div_threshold or not np.all(np.isfinite(corrected)):
                diverged = True
                stage_ok = False
                x_arr[n + 1] = (
                    corrected
                    if np.all(np.isfinite(corrected))
                    else x_arr[n]
                )
                t_arr[n + 1] = t_arr[n] + h
                f_arr[n + 1] = f_arr[n]
                break

            x_arr[n + 1] = corrected
            t_arr[n + 1] = t_arr[n] + h
            f_arr[n + 1] = rhs(corrected, eta)

        keep_start = curr_n + nsteps_tr + 1
        keep_end = curr_n + steps_per_stage
        keep_times = t_arr[keep_start : keep_end + 1]
        keep_states = x_arr[keep_start : keep_end + 1]
        x_out = (
            x_arr[keep_end]
            if stage_ok
            else x_arr[curr_n + steps_per_stage]
        )

        trajectory = (
            np.column_stack((keep_times, keep_states))
            if len(keep_times) > 0
            else np.empty((0, 4))
        )
        records.append(
            {
                "lambda_value": float(eta),
                "x_in": x_in,
                "x_out": x_out,
                "trajectory": trajectory,
                "status": "ok" if stage_ok else "diverged",
                "x_out_norm": float(np.linalg.norm(x_out)),
            }
        )
        curr_n = keep_end

    return records


def sample_ball(
    eq_point: np.ndarray,
    radius: float,
    n: int,
    seed: int,
) -> np.ndarray:
    """Sample the deterministic equilibrium ball used by the fixed contract."""

    rng = np.random.default_rng(seed)
    dim = len(eq_point)
    points: list[np.ndarray] = []
    while len(points) < n:
        batch = rng.normal(0.0, 1.0, (n * 3, dim))
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        radial_values = rng.uniform(0.0, 1.0, (n * 3, 1)) ** (1.0 / dim)
        ball = eq_point + radius * radial_values * batch / norms
        for point in ball:
            if np.linalg.norm(point - eq_point) <= radius:
                points.append(point)
            if len(points) >= n:
                break
    return np.array(points[:n])


__all__ = [
    "biased_saturation_df",
    "build_biased_seed",
    "find_biased_branches",
    "run_affine_continuation",
    "sample_ball",
]
