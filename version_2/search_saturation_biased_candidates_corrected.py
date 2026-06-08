#!/usr/bin/env python3
"""Corrected search for seeds of the non-smooth fractional Chua system using a biased describing function.

This script implements:
1. Auditing the sign convention (1 + Wq*N1 = 0 vs 1 - Wq*N1 = 0) and standardizing to 1 + Wq*N1 = 0.
2. Direct seed reconstruction with consistency checks (DC error, harmonic amplitude error, residual norms).
3. Affine homotopy continuation ensuring DC state preservation.
4. Homotopy identity verification at eta = 1.
5. Re-running cases with comparison before/after.
6. Exporting CSV files and plots to a new directory: outputs/biased_saturation_search_q09998_corrected/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Evitar popups GUI de matplotlib en sistemas Windows sin servidor de ventanas activo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import gamma

# Inyectamos version_2 en sys.path para asegurar que la librería hidden_attractors se importe correctamente
_file_dir = os.path.dirname(os.path.abspath(__file__))
if _file_dir not in sys.path:
    sys.path.insert(0, _file_dir)

from hidden_attractors.analysis.spectral import fft_spectrum
from hidden_attractors.continuation.continuation_fractional import run_fractional_continuation
from hidden_attractors.diagnostics.periodicity import classify_post_transient_periodicity
from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.models.chua import (
    ChuaParameters,
    chua_parameters,
    nonlinearity_nonsmooth,
)
from hidden_attractors.seed_generation.chua import (
    chua_gain,
    chua_matrices,
    find_harmonic_seed,
    find_omega_gain_candidates,
)
from hidden_attractors.seed_generation.core import (
    complex_dtype,
    fractional_iomega_power,
    real_dtype,
    validate_fractional_order,
)
from hidden_attractors.systems import get_system


# ── CLI BLOCKING ──────────────────────────────────────────────────────────────

for arg in sys.argv:
    if ("hiddenness" in arg and "no-hiddenness" not in arg) or "probe" in arg or "equilibria" in arg:
        print("Error: Hiddenness tests are intentionally disabled in the biased DF correction phase.")
        sys.exit(1)


# ── A. Sign Audit & Transfer Function ─────────────────────────────────────────

def get_Wq(omega: float, q: float, pmat: np.ndarray, qvec: np.ndarray, rvec: np.ndarray) -> complex:
    """Evaluate the transfer function matching the library sign convention: W = r^T (P - s^q I)^(-1) b.

    This yields a negative real part for the centered candidate, so that 1 + W*N = 0 holds.
    """
    matrix = pmat.astype(complex_dtype) - fractional_iomega_power(omega, q) * np.eye(3, dtype=complex_dtype)
    val = (
        rvec.astype(complex_dtype).reshape(1, -1)
        @ np.linalg.inv(matrix)
        @ qvec.astype(complex_dtype).reshape(-1, 1)
    )[0, 0]
    return complex_dtype(val)


def harmonic_residual_sign_audit(W: complex, N: float) -> Dict[str, float]:
    """Audit the absolute residuals for both 1 + WN and 1 - WN conventions."""
    R_plus = 1.0 + W * N
    R_minus = 1.0 - W * N
    return {
        "R_plus_real": float(R_plus.real),
        "R_plus_imag": float(R_plus.imag),
        "R_plus_abs": float(abs(R_plus)),
        "R_minus_real": float(R_minus.real),
        "R_minus_imag": float(R_minus.imag),
        "R_minus_abs": float(abs(R_minus)),
    }


# ── B. Biased Describing Function & Corrected Residual ────────────────────────

def biased_saturation_describing_function(
    A: float, c: float, g: float, n_theta: int = 8192
) -> Tuple[float, float]:
    """Compute the biased describing function values psi0 and N1 by numerical quadrature."""
    if A < 1e-6:
        return 0.0, 0.0

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=real_dtype)
    sigma = c + A * np.cos(theta)
    psi = g * np.clip(sigma, -1.0, 1.0)

    psi0 = float(np.mean(psi))
    psi1 = 2.0 * float(np.mean(psi * np.cos(theta)))
    N1 = psi1 / A
    return psi0, N1


def biased_saturation_residual(
    A: float, c: float, omega: float, params: ChuaParameters, q: float, sign_convention: str = "plus"
) -> np.ndarray:
    """Compute the 3D residual vector of the biased system under the specified convention."""
    if A < 1e-6:
        return np.array([c, 1e2, 1e2], dtype=float)

    g = chua_gain(params)
    psi0, N1 = biased_saturation_describing_function(A, c, g)
    pmat, qvec, rvec = chua_matrices(params)

    # F0 = c - r^T x_bar
    try:
        x_bar = np.linalg.solve(pmat, -qvec * psi0)
        F0 = c - float(rvec @ x_bar)
    except np.linalg.LinAlgError:
        F0 = 1e3

    # Transfer function Wq
    try:
        Wq = get_Wq(omega, q, pmat, qvec, rvec)
    except np.linalg.LinAlgError:
        Wq = 0.0

    # Choose sign convention
    if sign_convention == "plus":
        term = 1.0 + Wq * N1
    else:
        term = 1.0 - Wq * N1

    F1 = float(np.real(term))
    F2 = float(np.imag(term))
    return np.array([F0, F1, F2], dtype=float)


# ── C. Multi-Start Grid Search ────────────────────────────────────────────────

def find_biased_saturation_branches(
    params: ChuaParameters,
    q: float,
    A_range: Tuple[float, float] = (0.5, 10.0),
    c_range: Tuple[float, float] = (-8.0, 8.0),
    omega_range: Tuple[float, float] = (0.5, 6.0),
    n_A: int = 5,
    n_c: int = 9,
    n_omega: int = 6,
    residual_tol: float = 1e-4,
    A_tol: float = 1e-3,
    c_tol: float = 1e-3,
    omega_tol: float = 1e-3,
    sign_convention: str = "plus",
) -> List[Dict[str, Any]]:
    """Perform multi-start grid search for biased describing function roots under sign_convention."""
    q_val = validate_fractional_order(q)
    A_grid = np.linspace(A_range[0], A_range[1], n_A)
    c_grid = np.linspace(c_range[0], c_range[1], n_c)
    omega_grid = np.linspace(omega_range[0], omega_range[1], n_omega)

    raw_candidates = []

    def residual_func(x):
        return biased_saturation_residual(x[0], x[1], x[2], params, q_val, sign_convention)

    for A0 in A_grid:
        for c0 in c_grid:
            for w0 in omega_grid:
                try:
                    res = least_squares(
                        residual_func,
                        x0=[A0, c0, w0],
                        bounds=([1e-6, -12.0, 0.1], [25.0, 12.0, 8.0]),
                        ftol=1e-10,
                        xtol=1e-10,
                        max_nfev=300,
                    )
                    if res.success:
                        A_opt, c_opt, w_opt = res.x
                        residuals = residual_func(res.x)
                        norm_res = np.linalg.norm(residuals)
                        if norm_res < residual_tol:
                            raw_candidates.append((A_opt, c_opt, w_opt, norm_res))
                except Exception:
                    continue

    unique_candidates: List[Dict[str, Any]] = []
    for A_c, c_c, w_c, res_c in raw_candidates:
        if A_c < 0.5 or w_c < 0.5 or w_c > 6.0:
            continue
        dup = False
        for existing in unique_candidates:
            dA = abs(A_c - existing["A"])
            dc = abs(c_c - existing["c"])
            dw = abs(w_c - existing["omega"])
            if dA < A_tol and dc < c_tol and dw < omega_tol:
                if res_c < existing["residual_norm"]:
                    existing["A"] = A_c
                    existing["c"] = c_c
                    existing["omega"] = w_c
                    existing["residual_norm"] = res_c
                dup = True
                break
        if not dup:
            unique_candidates.append(
                {
                    "A": A_c,
                    "c": c_c,
                    "omega": w_c,
                    "residual_norm": res_c,
                }
            )

    unique_candidates.sort(key=lambda x: x["residual_norm"])
    return unique_candidates


# ── D. Seed Reconstruction & Consistency Checks ───────────────────────────────

def build_biased_fractional_seed(
    params: ChuaParameters, q: float, A: float, c: float, omega: float, psi0: float, N1: float
) -> Dict[str, Any]:
    """Reconstruct the biased Lur'e seed state vector and constituent components."""
    q_val = validate_fractional_order(q)
    pmat, qvec, rvec = chua_matrices(params)

    # DC state: x_bar = -P^(-1) b psi0
    x_bar = np.linalg.solve(pmat, -qvec * psi0)

    # First harmonic component: X1 = (lambda I - P)^(-1) b * N1 * A
    lam = fractional_iomega_power(omega, q_val)
    matrix = lam * np.eye(3, dtype=complex_dtype) - pmat.astype(complex_dtype)
    inv_qvec = np.linalg.solve(matrix, qvec.astype(complex_dtype))
    X1 = inv_qvec * N1 * A

    # Seed = x_bar + Re(X1)
    X_seed = x_bar + np.real(X1)

    return {
        "seed": X_seed,
        "x_bar": x_bar,
        "Re_X1": np.real(X1),
        "Im_X1": np.imag(X1),
    }


# ── E. Affine Homotopy Identity Verification ──────────────────────────────────

def check_affine_homotopy_identity(
    X: np.ndarray, params: ChuaParameters, A: float, c: float, psi0: float, k_eff: float
) -> float:
    """Verify that at eta = 1, the affine homotopy matches the original Chua vector field."""
    pmat, qvec, rvec = chua_matrices(params)
    g = params.m0 - params.m1

    # Original Chua vector field
    sigma = float(rvec @ X)
    psi_val = g * np.clip(sigma, -1.0, 1.0)
    f_original = pmat @ X + qvec * psi_val

    # Affine homotopy field at eta = 1
    P_aff = pmat + k_eff * np.outer(qvec, rvec)
    const_aff = qvec * (psi0 - k_eff * c)
    f_eta1 = P_aff @ X + const_aff + 1.0 * qvec * (psi_val - psi0 - k_eff * (sigma - c))

    return float(np.linalg.norm(f_eta1 - f_original))


# ── F. Monolithic Affine Continuation ─────────────────────────────────────────

def run_biased_affine_fractional_continuation(
    params: ChuaParameters,
    q: float,
    h: float,
    seed_x0: np.ndarray,
    A: float,
    c: float,
    psi0: float,
    N1: float,
    lambda_values: List[float],
    t_transient: float = 30.0,
    t_keep: float = 30.0,
    div_threshold: float = 120.0,
    memory_mode: str = "full",
    memory_window_length: int | None = None,
) -> List[Dict[str, Any]]:
    """Execute monolithic Caputo ABM continuation with the bias-preserving affine homotopy."""
    dim = 3
    h = float(h)
    q = float(q)

    nsteps_tr = int(np.ceil(t_transient / h))
    nsteps_kp = int(np.ceil(t_keep / h))
    steps_per_stage = nsteps_tr + nsteps_kp
    num_stages = len(lambda_values)
    total_new_steps = num_stages * steps_per_stage

    t_arr = np.zeros(1 + total_new_steps, dtype=float)
    x_arr = np.zeros((1 + total_new_steps, dim), dtype=float)
    f_arr = np.zeros((1 + total_new_steps, dim), dtype=float)

    t_arr[0] = 0.0
    x_arr[0] = seed_x0

    pmat, qvec, rvec = chua_matrices(params)
    P_aff = pmat + N1 * np.outer(qvec, rvec)
    const_aff = qvec * (psi0 - N1 * c)
    g = params.m0 - params.m1

    def eval_rhs_deformed(x: np.ndarray, eta_val: float) -> np.ndarray:
        sigma = float(rvec @ x)
        psi_val = g * np.clip(sigma, -1.0, 1.0)
        return P_aff @ x + const_aff + eta_val * qvec * (psi_val - psi0 - N1 * (sigma - c))

    f_arr[0] = eval_rhs_deformed(x_arr[0], lambda_values[0])

    powers = np.arange(total_new_steps + 3, dtype=float)
    pow_q = powers ** q
    pow_q1 = powers ** (q + 1.0)
    hq = h ** q
    pred_scale = hq / float(gamma(q + 1.0))
    val_gq2 = float(gamma(q + 2.0))
    corr_scale = hq / val_gq2 if abs(val_gq2) > 1e-15 else 0.0

    steps_records = []
    curr_n = 0
    diverged = False
    stop_reason = ""

    for stage_idx, eta in enumerate(lambda_values):
        eta_f = float(eta)
        if diverged:
            break

        x_in = x_arr[curr_n].copy()
        x_in_norm = float(np.linalg.norm(x_in))

        stage_diverged = False
        stage_steps = 0

        transient_start_idx = curr_n + 1
        transient_end_idx = curr_n + nsteps_tr
        keep_start_idx = transient_end_idx + 1
        keep_end_idx = transient_end_idx + nsteps_kp

        for local_step in range(steps_per_stage):
            n = curr_n + local_step
            t_n1 = t_arr[n] + h

            if memory_mode == "window" and memory_window_length is not None:
                s_idx = max(0, n - int(memory_window_length) + 1)
            else:
                s_idx = 0

            # Predictor
            j_range = np.arange(s_idx, n + 1)
            b_weights = pow_q[n + 1 - j_range] - pow_q[n - j_range]
            predictor = x_arr[s_idx] + pred_scale * (b_weights @ f_arr[s_idx: n + 1])

            try:
                fp = eval_rhs_deformed(predictor, eta_f)
            except Exception as exc:
                diverged = True
                stage_diverged = True
                stop_reason = f"solver_exception:{exc}"
                break

            # Corrector
            n_prime = n - s_idx
            a0 = float(n_prime) ** (q + 1.0) - (float(n_prime) - q) * (float(n_prime) + 1.0) ** q
            if n_prime > 0:
                mid_indices = n - np.arange(s_idx + 1, n + 1)
                a_mid = (pow_q1[mid_indices + 2]
                         + pow_q1[mid_indices]
                         - 2.0 * pow_q1[mid_indices + 1])
                a_weights = np.concatenate(([a0], a_mid))
            else:
                a_weights = np.array([a0])

            corrected = x_arr[s_idx] + corr_scale * ((a_weights @ f_arr[s_idx: n + 1]) + fp)
            norm = np.linalg.norm(corrected)

            if norm > div_threshold:
                diverged = True
                stage_diverged = True
                stop_reason = "diverged"
                break
            if not np.all(np.isfinite(corrected)):
                diverged = True
                stage_diverged = True
                stop_reason = "nonfinite_solution"
                break

            x_arr[n + 1] = corrected
            t_arr[n + 1] = t_n1
            f_arr[n + 1] = eval_rhs_deformed(corrected, eta_f)
            stage_steps += 1

        if stage_diverged:
            last_idx = curr_n + stage_steps
            stage_times = t_arr[transient_start_idx: last_idx + 1]
            stage_states = x_arr[transient_start_idx: last_idx + 1]
            traj = np.column_stack((stage_times, stage_states)) if len(stage_times) > 0 else np.empty((0, 1 + dim))
            x_out = x_arr[last_idx] if stage_steps > 0 else x_in

            steps_records.append({
                "lambda_value": eta_f,
                "x_in": x_in,
                "x_out": x_out,
                "trajectory": traj,
                "status": stop_reason,
                "n_steps": stage_steps,
                "t_end": float(stage_times[-1]) if len(stage_times) > 0 else 0.0,
                "max_norm": float(np.max(np.linalg.norm(stage_states, axis=1))) if len(stage_states) > 0 else x_in_norm,
                "x_in_norm": x_in_norm,
                "x_out_norm": float(np.linalg.norm(x_out)),
            })
            break

        keep_times = t_arr[keep_start_idx: keep_end_idx + 1]
        keep_states = x_arr[keep_start_idx: keep_end_idx + 1]
        x_out = x_arr[keep_end_idx]
        all_stage_states = x_arr[transient_start_idx: keep_end_idx + 1]
        max_norm = float(np.max(np.linalg.norm(all_stage_states, axis=1)))
        traj = np.column_stack((keep_times, keep_states))

        steps_records.append({
            "lambda_value": eta_f,
            "x_in": x_in,
            "x_out": x_out,
            "trajectory": traj,
            "status": "ok",
            "n_steps": steps_per_stage,
            "t_end": float(keep_times[-1]),
            "max_norm": max_norm,
            "x_in_norm": x_in_norm,
            "x_out_norm": float(np.linalg.norm(x_out)),
        })

        curr_n = keep_end_idx

    return steps_records


# ── J. Plotting Corrected Figures ─────────────────────────────────────────────

def plot_sign_audit(sign_audit_rows: List[Dict[str, Any]], outdir: Path) -> Path:
    """Plot absolute residual values for both 1 + WN and 1 - WN conventions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.arange(len(sign_audit_rows))
    bar_width = 0.35

    abs_plus = [row["abs_1_plus_Wq_N1"] for row in sign_audit_rows]
    abs_minus = [row["abs_1_minus_Wq_N1"] for row in sign_audit_rows]
    labels = [f"B{row['branch_id']}\nc={row['c']:.2f}" for row in sign_audit_rows]

    ax.bar(indices - bar_width/2, abs_plus, bar_width, label="|1 + Wq*N1| (Oficial)", color="blue")
    ax.bar(indices + bar_width/2, abs_minus, bar_width, label="|1 - Wq*N1|", color="orange")

    ax.set_ylabel("Residuo Armónico Absoluto")
    ax.set_title("Auditoría de Signo Armónico para cada Raíz Encontrada")
    ax.set_xticks(indices)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = outdir / "figures" / "sign_audit" / "sign_audit_comparison.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def plot_seed_consistency(consistency_rows: List[Dict[str, Any]], outdir: Path) -> Path:
    """Plot DC output error and first-harmonic amplitude error for each seed."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    indices = np.arange(len(consistency_rows))
    labels = [f"B{row['branch_id']}" for row in consistency_rows]

    dc_errs = [row["dc_output_error"] for row in consistency_rows]
    amp_errs = [row["harmonic_amp_error"] for row in consistency_rows]

    ax1.bar(indices, dc_errs, color="teal", alpha=0.8)
    ax1.set_ylabel("Error en Bias DC |r^T x_bar - c|")
    ax1.set_title("Error del Componente DC de la Semilla")
    ax1.set_xticks(indices)
    ax1.set_xticklabels(labels)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    ax2.bar(indices, amp_errs, color="purple", alpha=0.8)
    ax2.set_ylabel("Error de Amplitud Fasorial |harmonic_amp - A|")
    ax2.set_title("Error de Amplitud del Primer Armónico")
    ax2.set_xticks(indices)
    ax2.set_xticklabels(labels)
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = outdir / "figures" / "seed_consistency" / "seed_consistency_errors.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def plot_continuation_affine(steps: List[Dict[str, Any]], outdir: Path, prefix: str) -> Path:
    """Plot affine homotopy continuation metrics: norm, states, and RMS vs eta."""
    etas = [s["lambda_value"] for s in steps]
    norms = [s["x_out_norm"] for s in steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Norm vs eta
    ax1.plot(etas, norms, "b-o", linewidth=2)
    ax1.set_xlabel("eta")
    ax1.set_ylabel("||X_final||")
    ax1.set_title("Continuación Afín: Norma vs eta")
    ax1.grid(True)

    # Coordinates vs eta
    coords = np.array([s["x_out"] for s in steps])
    ax2.plot(etas, coords[:, 0], "-o", label="x")
    ax2.plot(etas, coords[:, 1], "-o", label="y")
    ax2.plot(etas, coords[:, 2], "-o", label="z")
    ax2.set_xlabel("eta")
    ax2.set_ylabel("Coordenadas")
    ax2.set_title("Coordenadas Finales vs eta")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    fig_path = outdir / "figures" / "continuation_affine" / f"{prefix}_continuation_affine.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def plot_final_attractor(trajectory: np.ndarray, outdir: Path, prefix: str) -> Path:
    """Plot final 3D attractor and xy, xz, yz projections."""
    states = trajectory[:, 1:4]
    centroid = np.mean(states, axis=0)

    fig = plt.figure(figsize=(15, 10))

    # 3D
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(states[:, 0], states[:, 1], states[:, 2], "b-", alpha=0.7, linewidth=0.8)
    ax3d.scatter(centroid[0], centroid[1], centroid[2], color="red", s=100, label="Centroide")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("Atractor 3D Post-Transitorio")
    ax3d.legend()

    # xy
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.plot(states[:, 0], states[:, 1], "g-", alpha=0.7, linewidth=0.8)
    ax_xy.scatter(centroid[0], centroid[1], color="red", s=50)
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_title("xy Projection")
    ax_xy.grid(True)

    # xz
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.plot(states[:, 0], states[:, 2], "r-", alpha=0.7, linewidth=0.8)
    ax_xz.scatter(centroid[0], centroid[2], color="red", s=50)
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.set_title("xz Projection")
    ax_xz.grid(True)

    # yz
    ax_yz = fig.add_subplot(2, 2, 4)
    ax_yz.plot(states[:, 1], states[:, 2], "c-", alpha=0.7, linewidth=0.8)
    ax_yz.scatter(centroid[1], centroid[2], color="red", s=50)
    ax_yz.set_xlabel("y")
    ax_yz.set_ylabel("z")
    ax_yz.set_title("yz Projection")
    ax_yz.grid(True)

    plt.tight_layout()
    fig_path = outdir / "figures" / "attractors" / f"{prefix}_attractor.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def plot_time_series(trajectory: np.ndarray, outdir: Path, prefix: str) -> Path:
    """Plot time series x(t), y(t), z(t) for the post-transient segment."""
    times = trajectory[:, 0]
    states = trajectory[:, 1:4]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax1.plot(times, states[:, 0], "b-", linewidth=1.0)
    ax1.set_ylabel("x(t)")
    ax1.grid(True)
    ax1.set_title("Series Temporales de Estados")

    ax2.plot(times, states[:, 1], "g-", linewidth=1.0)
    ax2.set_ylabel("y(t)")
    ax2.grid(True)

    ax3.plot(times, states[:, 2], "r-", linewidth=1.0)
    ax3.set_ylabel("z(t)")
    ax3.set_xlabel("Tiempo t (s)")
    ax3.grid(True)

    plt.tight_layout()
    fig_path = outdir / "figures" / "timeseries" / f"{prefix}_timeseries.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def plot_fft_psd(trajectory: np.ndarray, h: float, outdir: Path, prefix: str) -> Tuple[Path, float]:
    """Plot FFT amplitude spectrum of x(t) and mark the dominant frequency."""
    x_val = trajectory[:, 1]
    spec = fft_spectrum(x_val, h)

    fig, ax = plt.subplots(figsize=(10, 5))
    if spec.frequency_hz.size > 0:
        ax.plot(spec.frequency_rad_s, spec.values, "b-", linewidth=1.2)
        dom_idx = np.argmax(spec.values)
        dom_freq_rad = spec.frequency_rad_s[dom_idx]
        ax.axvline(dom_freq_rad, color="red", linestyle="--", alpha=0.7)
        ax.plot(dom_freq_rad, spec.values[dom_idx], "ro", label=f"Frecuencia Dominante: {dom_freq_rad:.3f} rad/s")
    else:
        dom_freq_rad = 0.0
        ax.text(0.5, 0.5, "Espectro vacío", transform=ax.transAxes, ha="center")

    ax.set_xlabel("Frecuencia angular (rad/s)")
    ax.set_ylabel("Amplitud FFT")
    ax.set_title("Espectro de Amplitud FFT")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    fig_path = outdir / "figures" / "fft" / f"{prefix}_fft.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path, dom_freq_rad


def plot_centered_comparison(
    biased_traj: np.ndarray,
    centered_traj: np.ndarray | None,
    outdir: Path,
    prefix: str,
) -> Path:
    """Generate comparison overlay figures (3D state space and FFT) with centered reference."""
    fig = plt.figure(figsize=(16, 7))

    # 3D
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(biased_traj[:, 1], biased_traj[:, 2], biased_traj[:, 3], "b-", alpha=0.7, linewidth=0.8, label="Sesgado Corregido")
    
    biased_centroid = np.mean(biased_traj[:, 1:4], axis=0)
    ax3d.scatter(biased_centroid[0], biased_centroid[1], biased_centroid[2], color="darkblue", s=100, marker="o")

    if centered_traj is not None and len(centered_traj) > 0:
        ax3d.plot(centered_traj[:, 1], centered_traj[:, 2], centered_traj[:, 3], "r--", alpha=0.6, linewidth=0.8, label="Centrado de Referencia")
        centered_centroid = np.mean(centered_traj[:, 1:4], axis=0)
        ax3d.scatter(centered_centroid[0], centered_centroid[1], centered_centroid[2], color="darkred", s=100, marker="X")

    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("Comparación Atractores 3D")
    ax3d.legend()

    # FFT
    ax_fft = fig.add_subplot(1, 2, 2)
    h_b = float(np.median(np.diff(biased_traj[:, 0])))
    spec_b = fft_spectrum(biased_traj[:, 1], h_b)
    if spec_b.frequency_hz.size > 0:
        ax_fft.plot(spec_b.frequency_rad_s, spec_b.values, "b-", alpha=0.8, label="Sesgado Corregido")

    if centered_traj is not None and len(centered_traj) > 0:
        h_c = float(np.median(np.diff(centered_traj[:, 0])))
        spec_c = fft_spectrum(centered_traj[:, 1], h_c)
        if spec_c.frequency_hz.size > 0:
            ax_fft.plot(spec_c.frequency_rad_s, spec_c.values, "r--", alpha=0.7, label="Centrado de Referencia")

    ax_fft.set_xlabel("Frecuencia angular (rad/s)")
    ax_fft.set_ylabel("Amplitud")
    ax_fft.set_title("Comparación de Espectros FFT")
    ax_fft.grid(True)
    ax_fft.legend()

    plt.tight_layout()
    fig_path = outdir / "figures" / "comparisons" / f"{prefix}_comparison.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


# ── H. Legacy Comparer ────────────────────────────────────────────────────────

def load_legacy_by_case(legacy_dir: Path) -> Dict[Tuple[float, float, int], Dict[str, Any]]:
    """Load legacy search results from the previous outputs directory."""
    roots_path = legacy_dir / "roots.csv"
    summary_path = legacy_dir / "summary.csv"
    if not (roots_path.exists() and summary_path.exists()):
        return {}

    try:
        roots_df = pd.read_csv(roots_path)
        summary_df = pd.read_csv(summary_path)

        legacy_data = {}
        for _, root_row in roots_df.iterrows():
            m1 = float(root_row["m1"])
            m0 = float(root_row["m0"])
            branch = int(root_row["branch"])
            prefix = str(root_row["prefix"])

            sum_rows = summary_df[summary_df["prefix"] == prefix]
            if not sum_rows.empty:
                sum_row = sum_rows.iloc[0]
                status = str(sum_row["status"])
                classification = str(sum_row["classification"])
            else:
                status = "unknown"
                classification = "unknown"

            legacy_data[(m1, m0, branch)] = {
                "A": float(root_row["A"]),
                "c": float(root_row["c"]),
                "omega": float(root_row["omega"]),
                "N1": float(root_row["N1"]),
                "psi0": float(root_row["psi0"]),
                "residual_norm": float(root_row["residual_norm"]),
                "status": status,
                "classification": classification,
            }
        return legacy_data
    except Exception as e:
        print(f"Warning: could not load legacy data: {e}")
        return {}


# ── MAIN RUNNER ───────────────────────────────────────────────────────────────

def run_corrected_workflow(
    q: float,
    h: float,
    sign_convention: str,
    eta_step: float,
    memory_mode: str,
    outdir: Path,
    m1_grid: List[float],
    m0_grid: List[float],
    t_transient: float,
    t_keep: float,
    t_sim_final: float,
    t_sim_transient: float,
    rerun_priority: bool,
    compare_with_prev: bool,
) -> None:
    """Run search, sign audit, seed consistency, affine continuation, simulation, and comparison."""
    alpha = 8.4562
    beta = 12.0732
    gamma = 0.0052

    # Create directories
    outdir.mkdir(parents=True, exist_ok=True)
    for sub in ["figures", "trajectories", "continuation_steps"]:
        (outdir / sub).mkdir(parents=True, exist_ok=True)
    for fig_sub in ["sign_audit", "seed_consistency", "continuation_affine", "attractors", "timeseries", "fft", "comparisons"]:
        (outdir / "figures" / fig_sub).mkdir(parents=True, exist_ok=True)

    # Initialize CSV lists
    roots_corrected_rows = []
    roots_sign_audit_rows = []
    seed_consistency_rows = []
    affine_continuation_rows = []
    final_classification_rows = []
    comparison_rows = []
    manifest_figures = []
    affine_identity_rows = []

    # Counters
    total_recalc_roots = 0
    total_sign_audit_1_plus_wins = 0
    total_seeds_passed_consistency = 0
    total_affine_continuation_ok = 0
    total_periodic = 0
    total_nonperiodic = 0
    total_chaotic = 0

    legacy_data = {}
    if compare_with_prev:
        legacy_dir = Path("outputs/biased_saturation_search_q09998")
        legacy_data = load_legacy_by_case(legacy_dir)

    lambda_values = list(np.arange(0.0, 1.0 + 1e-9, eta_step))

    # Grid loop
    for m1 in m1_grid:
        for m0 in m0_grid:
            print(f"\n========================================")
            print(f"Analizando parámetros: m1={m1}, m0={m0}")
            print(f"========================================")
            params = chua_parameters(
                model="nonsmooth", alpha=alpha, beta=beta, gamma=gamma, m0=m0, m1=m1
            )
            g = chua_gain(params)
            pmat, qvec, rvec = chua_matrices(params)

            # 1. Centered Reference
            centered_seeds = []
            try:
                pairs = find_omega_gain_candidates(q, params, wmin=0.5, wmax=6.0, nscan=5000)
                for idx, (w, gain) in enumerate(pairs):
                    try:
                        seed_data = find_harmonic_seed(q=q, params=params, branch_index=idx, wmin=0.5, wmax=6.0)
                        centered_seeds.append(seed_data)
                    except Exception as e:
                        print(f"  [Centered] Error al resolver semilla centered para la rama {idx}: {e}")
            except Exception as e:
                print(f"  [Centered] Error al buscar candidatos centered: {e}")

            print(f"  Encontradas {len(centered_seeds)} ramas centradas.")

            # 2. Biased Search
            biased_roots = find_biased_saturation_branches(
                params=params,
                q=q,
                A_range=(0.5, 10.0),
                c_range=(-8.0, 8.0),
                omega_range=(0.5, 6.0),
                n_A=5,
                n_c=9,
                n_omega=6,
                residual_tol=1e-4,
                sign_convention=sign_convention,
            )
            print(f"  Búsqueda sesgada encontró {len(biased_roots)} raíces.")

            for root_idx, r in enumerate(biased_roots[:2]):
                total_recalc_roots += 1
                A = r["A"]
                c = r["c"]
                omega = r["omega"]
                res_norm = r["residual_norm"]

                psi0, N1 = biased_saturation_describing_function(A, c, g)
                is_centered_duplicate = abs(c) <= 0.05
                prefix = f"biased_q{int(q*10000)}_m1_{m1:.4f}_m0_{m0:.4f}_branch_{root_idx}"
                if is_centered_duplicate:
                    prefix += f"_c_centered_like"
                else:
                    prefix += f"_c_{c:.3f}"
                prefix = prefix.replace(".", "p").replace("-", "m")

                print(f"  -- Procesando candidato {root_idx}: A={A:.3f}, c={c:.3f}, w={omega:.3f}")

                # Audit sign on this root
                try:
                    Wq = get_Wq(omega, q, pmat, qvec, rvec)
                except np.linalg.LinAlgError:
                    Wq = 0.0

                audit_res = harmonic_residual_sign_audit(Wq, N1)
                
                # Check if 1 + Wq*N1 dominates
                if audit_res["R_plus_abs"] < audit_res["R_minus_abs"]:
                    total_sign_audit_1_plus_wins += 1

                roots_sign_audit_rows.append({
                    "case_id": f"m1_{m1:.4f}_m0_{m0:.4f}",
                    "branch_id": root_idx,
                    "A": A,
                    "c": c,
                    "omega": omega,
                    "N1": N1,
                    "psi0": psi0,
                    "Re_Wq": Wq.real,
                    "Im_Wq": Wq.imag,
                    "abs_1_plus_Wq_N1": audit_res["R_plus_abs"],
                    "abs_1_minus_Wq_N1": audit_res["R_minus_abs"],
                    "selected_sign_convention": sign_convention,
                    "sign_consistent_with_centered": audit_res["R_plus_abs"] < 1e-3,
                })

                # Reconstruct seed
                seed_info = build_biased_fractional_seed(params, q, A, c, omega, psi0, N1)
                X_seed = seed_info["seed"]
                x_bar = seed_info["x_bar"]
                Re_X1 = seed_info["Re_X1"]
                Im_X1 = seed_info["Im_X1"]

                # Algebraic tests
                dc_output_error = abs(float(rvec @ x_bar) - c)
                harmonic_output_re = float(rvec @ Re_X1)
                harmonic_output_im = float(rvec @ Im_X1)
                harmonic_amp = np.sqrt(harmonic_output_re**2 + harmonic_output_im**2)
                harmonic_amp_error = abs(harmonic_amp - A)

                # Dictum
                residual_norm_plus = np.sqrt(biased_saturation_residual(A, c, omega, params, q, "plus")**2).sum()
                residual_norm_minus = np.sqrt(biased_saturation_residual(A, c, omega, params, q, "minus")**2).sum()

                # dictamen
                if dc_output_error >= 1e-5:
                    status_consistency = "dc_mismatch"
                elif harmonic_amp_error >= 1e-4:
                    status_consistency = "harmonic_amplitude_mismatch"
                elif residual_norm_minus < 1e-4 and residual_norm_minus < residual_norm_plus:
                    status_consistency = "sign_mismatch"
                elif residual_norm_plus < 1e-4:
                    status_consistency = "ok"
                    total_seeds_passed_consistency += 1
                else:
                    status_consistency = "rejected"

                seed_consistency_rows.append({
                    "case_id": f"m1_{m1:.4f}_m0_{m0:.4f}",
                    "branch_id": root_idx,
                    "dc_output_error": dc_output_error,
                    "harmonic_output_re": harmonic_output_re,
                    "harmonic_output_im": harmonic_output_im,
                    "harmonic_amp": harmonic_amp,
                    "harmonic_amp_error": harmonic_amp_error,
                    "R_plus_abs": audit_res["R_plus_abs"],
                    "R_minus_abs": audit_res["R_minus_abs"],
                    "residual_norm_plus": residual_norm_plus,
                    "residual_norm_minus": residual_norm_minus,
                    "seed_consistency_status": status_consistency,
                })

                # Homotopy identity check
                max_identity_err = 0.0
                # Over random points
                for _ in range(10):
                    X_rand = np.random.uniform(-5.0, 5.0, size=3)
                    err = check_affine_homotopy_identity(X_rand, params, A, c, psi0, N1)
                    if err > max_identity_err:
                        max_identity_err = err
                # Over X_seed
                err_seed = check_affine_homotopy_identity(X_seed, params, A, c, psi0, N1)
                if err_seed > max_identity_err:
                    max_identity_err = err_seed

                affine_identity_rows.append({
                    "case_id": f"m1_{m1:.4f}_m0_{m0:.4f}",
                    "branch_id": root_idx,
                    "max_identity_error_eta1": max_identity_err,
                    "status": "ok" if max_identity_err < 1e-10 else "identity_failed",
                })

                # Write roots corrected CSV info
                roots_corrected_rows.append({
                    "m1": m1,
                    "m0": m0,
                    "q": q,
                    "branch": root_idx,
                    "A": A,
                    "c": c,
                    "omega": omega,
                    "N1": N1,
                    "psi0": psi0,
                    "residual_norm": res_norm,
                    "X_seed_x": X_seed[0],
                    "X_seed_y": X_seed[1],
                    "X_seed_z": X_seed[2],
                })

                # 3. Continuation
                print(f"    Iniciando continuación afín Caputo ABM...")
                cont_steps = run_biased_affine_fractional_continuation(
                    params=params,
                    q=q,
                    h=h,
                    seed_x0=X_seed,
                    A=A,
                    c=c,
                    psi0=psi0,
                    N1=N1,
                    lambda_values=lambda_values,
                    t_transient=t_transient,
                    t_keep=t_keep,
                    memory_mode=memory_mode,
                )

                cont_path_file = outdir / "continuation_steps" / f"{prefix}_path.csv"
                pd.DataFrame([{
                    "lambda": s["lambda_value"],
                    "status": s["status"],
                    "x_out_norm": s["x_out_norm"],
                    "x_out": s["x_out"].tolist()
                } for s in cont_steps]).to_csv(cont_path_file, index=False)

                survived = len(cont_steps) == len(lambda_values) and all(s["status"] == "ok" for s in cont_steps)
                print(f"    Estatus continuación afín: {'SOBREVIVIÓ' if survived else 'FALLÓ'}")

                eta_fail = np.nan
                if not survived:
                    for s in cont_steps:
                        if s["status"] != "ok":
                            eta_fail = s["lambda_value"]
                            break

                # Write continuation summary row
                affine_continuation_rows.append({
                    "case_id": f"m1_{m1:.4f}_m0_{m0:.4f}",
                    "branch_id": root_idx,
                    "status": "ok" if survived else "failed",
                    "final_state_norm": cont_steps[-1]["x_out_norm"] if len(cont_steps) > 0 else np.nan,
                    "max_norm": max(s["max_norm"] for s in cont_steps) if len(cont_steps) > 0 else np.nan,
                    "rms_observed": np.nan, # can be updated with keep segment RMS if needed
                    "eta_failure": eta_fail,
                })

                if not survived:
                    final_classification_rows.append({
                        "case_id": f"m1_{m1:.4f}_m0_{m0:.4f}",
                        "branch_id": root_idx,
                        "centroid_x": np.nan,
                        "centroid_y": np.nan,
                        "centroid_z": np.nan,
                        "dominant_frequency": np.nan,
                        "classification": "biased_corrected_continuation_failed",
                    })
                    
                    # Store comparison row if compare_with_prev
                    if compare_with_prev:
                        legacy_key = (m1, m0, root_idx)
                        legacy = legacy_data.get(legacy_key, {})
                        comparison_rows.append({
                            "case_id": f"m1_{m1:.4f}_m0_{m0:.4f}",
                            "m1": m1,
                            "m0": m0,
                            "branch_id": root_idx,
                            "old_A": legacy.get("A", np.nan),
                            "old_c": legacy.get("c", np.nan),
                            "old_omega": legacy.get("omega", np.nan),
                            "new_A": A,
                            "new_c": c,
                            "new_omega": omega,
                            "old_harmonic_sign": "minus",
                            "new_harmonic_sign": "plus",
                            "old_residual_norm": legacy.get("residual_norm", np.nan),
                            "new_residual_norm": res_norm,
                            "old_continuation_status": legacy.get("status", "unknown"),
                            "new_continuation_status": "failed",
                            "old_final_classification": legacy.get("classification", "unknown"),
                            "new_final_classification": "biased_corrected_continuation_failed",
                            "changed_due_to_sign_correction": "No" if legacy else "unknown",
                            "changed_due_to_affine_homotopy": "Yes" if legacy and legacy.get("status") == "ok" else "No",
                            "notes": "Failed under affine continuation",
                        })
                    continue

                total_affine_continuation_ok += 1
                x_final_cont = cont_steps[-1]["x_out"]

                # 4. Final simulation
                print(f"    Corriendo simulación final del sistema original (eta=1)...")
                system = get_system("chua-nonsmooth")
                system.parameters["m0"] = m0
                system.parameters["m1"] = m1

                sim_t, sim_x, sim_status, sim_info = fractional_integrate(
                    rhs=lambda t, x: system.rhs(x, system.parameters),
                    x0=x_final_cont,
                    q=q,
                    h=h,
                    t_final=t_sim_final,
                    method="abm",
                    memory_mode=memory_mode,
                    system=system,
                    use_c_backend=True,
                )

                full_traj = np.column_stack((sim_t, sim_x))
                post_traj = full_traj[sim_t >= t_sim_transient]

                # Save trajectory
                traj_path = outdir / "trajectories" / f"{prefix}_trajectory.csv"
                pd.DataFrame(post_traj, columns=["t", "x", "y", "z"]).to_csv(traj_path, index=False)

                # Periodicity and attractor classification
                classification = "biased_corrected_collapsed_to_equilibrium"
                final_centroid = np.zeros(3)
                dom_freq_rad = 0.0

                if len(post_traj) > 10:
                    final_centroid = np.mean(post_traj[:, 1:4], axis=0)
                    p_res = classify_post_transient_periodicity(post_traj, h=h)
                    lbl = p_res["candidate_label"]

                    max_range = float(np.max(np.ptp(post_traj[:, 1:4], axis=0)))
                    if max_range < 0.05:
                        classification = "biased_corrected_collapsed_to_equilibrium"
                    elif lbl == "regular_periodic_rejected":
                        classification = "biased_corrected_regular_periodic_rejected"
                        total_periodic += 1
                    elif lbl == "chaotic_candidate_pending_robustness":
                        classification = "biased_corrected_chaotic_candidate_pending_robustness"
                        total_chaotic += 1
                    else:
                        classification = "biased_corrected_nonperiodic_candidate"
                        total_nonperiodic += 1
                else:
                    classification = "biased_corrected_collapsed_to_equilibrium"

                # 5. Centered reference simulation
                centered_traj = None
                if centered_seeds:
                    c_seed = min(centered_seeds, key=lambda s: abs(s.omega - omega))
                    print(f"    Corriendo continuación de referencia centrada...")
                    try:
                        c_steps = run_fractional_continuation(
                            system=system,
                            seed_x0=c_seed.seed,
                            k_gain=c_seed.gain,
                            lambda_values=lambda_values,
                            h=h,
                            memory_mode=memory_mode,
                            integrator="abm",
                            t_transient=t_transient,
                            t_keep=t_keep,
                            q=q
                        )
                        c_final_cont = c_steps[-1]["x_out"]
                        c_sim_t, c_sim_x, _, _ = fractional_integrate(
                            rhs=lambda t, x: system.rhs(x, system.parameters),
                            x0=c_final_cont,
                            q=q,
                            h=h,
                            t_final=t_sim_final,
                            method="abm",
                            memory_mode=memory_mode,
                            system=system,
                            use_c_backend=True,
                        )
                        centered_full_traj = np.column_stack((c_sim_t, c_sim_x))
                        centered_traj = centered_full_traj[c_sim_t >= t_sim_transient]
                    except Exception as e:
                        print(f"      [Centered Reference] Error al correr continuación/simulación de referencia: {e}")

                # Save final classification row
                final_classification_rows.append({
                    "case_id": f"m1_{m1:.4f}_m0_{m0:.4f}",
                    "branch_id": root_idx,
                    "centroid_x": final_centroid[0],
                    "centroid_y": final_centroid[1],
                    "centroid_z": final_centroid[2],
                    "dominant_frequency": dom_freq_rad,
                    "classification": classification,
                })

                # Write comparisons if needed
                if compare_with_prev:
                    legacy_key = (m1, m0, root_idx)
                    legacy = legacy_data.get(legacy_key, {})
                    
                    changed_homotopy = "No"
                    if legacy:
                        if legacy.get("status") != "ok" or legacy.get("classification") != classification:
                            changed_homotopy = "Yes"

                    comparison_rows.append({
                        "case_id": f"m1_{m1:.4f}_m0_{m0:.4f}",
                        "m1": m1,
                        "m0": m0,
                        "branch_id": root_idx,
                        "old_A": legacy.get("A", np.nan),
                        "old_c": legacy.get("c", np.nan),
                        "old_omega": legacy.get("omega", np.nan),
                        "new_A": A,
                        "new_c": c,
                        "new_omega": omega,
                        "old_harmonic_sign": "minus",
                        "new_harmonic_sign": "plus",
                        "old_residual_norm": legacy.get("residual_norm", np.nan),
                        "new_residual_norm": res_norm,
                        "old_continuation_status": legacy.get("status", "unknown"),
                        "new_continuation_status": "ok",
                        "old_final_classification": legacy.get("classification", "unknown"),
                        "new_final_classification": classification,
                        "changed_due_to_sign_correction": "No" if legacy else "unknown",
                        "changed_due_to_affine_homotopy": changed_homotopy,
                        "notes": "Successfully ran with corrected homotopy",
                    })

                # 6. Generate Figures
                print(f"    Generando figuras...")
                cont_fig = plot_continuation_affine(cont_steps, outdir, prefix)
                att_fig = plot_final_attractor(post_traj, outdir, prefix)
                ts_fig = plot_time_series(post_traj, outdir, prefix)
                fft_fig, dom_freq_rad = plot_fft_psd(post_traj, h, outdir, prefix)
                comp_fig = plot_centered_comparison(post_traj, centered_traj, outdir, prefix)

                manifest_figures.extend([
                    str(cont_fig.relative_to(outdir)),
                    str(att_fig.relative_to(outdir)),
                    str(ts_fig.relative_to(outdir)),
                    str(fft_fig.relative_to(outdir)),
                    str(comp_fig.relative_to(outdir)),
                ])

    # Generate multi-branch global figures
    if roots_sign_audit_rows:
        sa_fig = plot_sign_audit(roots_sign_audit_rows, outdir)
        manifest_figures.append(str(sa_fig.relative_to(outdir)))
    if seed_consistency_rows:
        sc_fig = plot_seed_consistency(seed_consistency_rows, outdir)
        manifest_figures.append(str(sc_fig.relative_to(outdir)))

    # Save CSVs
    pd.DataFrame(roots_corrected_rows).to_csv(outdir / "roots_corrected.csv", index=False)
    pd.DataFrame(roots_sign_audit_rows).to_csv(outdir / "roots_sign_audit.csv", index=False)
    pd.DataFrame(seed_consistency_rows).to_csv(outdir / "seed_consistency_checks.csv", index=False)
    pd.DataFrame(affine_continuation_rows).to_csv(outdir / "affine_continuation_summary.csv", index=False)
    pd.DataFrame(final_classification_rows).to_csv(outdir / "final_classification_corrected.csv", index=False)
    pd.DataFrame(affine_identity_rows).to_csv(outdir / "affine_homotopy_identity_checks.csv", index=False)

    if compare_with_prev:
        pd.DataFrame(comparison_rows).to_csv(outdir / "centered_vs_biased_corrected_comparison.csv", index=False)

    # Write manifest.json
    manifest = {
        "search_parameters": {
            "q": q,
            "h": h,
            "sign_convention": sign_convention,
            "eta_step": eta_step,
            "memory_mode": memory_mode,
            "m1_grid": m1_grid,
            "m0_grid": m0_grid,
            "t_transient": t_transient,
            "t_keep": t_keep,
            "t_sim_final": t_sim_final,
            "t_sim_transient": t_sim_transient,
        },
        "statistics": {
            "total_recalc_roots": total_recalc_roots,
            "total_sign_audit_1_plus_wins": total_sign_audit_1_plus_wins,
            "total_seeds_passed_consistency": total_seeds_passed_consistency,
            "total_affine_continuation_ok": total_affine_continuation_ok,
            "total_periodic": total_periodic,
            "total_nonperiodic": total_nonperiodic,
            "total_chaotic": total_chaotic,
        },
        "figures": manifest_figures,
        "outputs": [
            "roots_corrected.csv",
            "roots_sign_audit.csv",
            "seed_consistency_checks.csv",
            "affine_continuation_summary.csv",
            "final_classification_corrected.csv",
            "affine_homotopy_identity_checks.csv",
        ]
    }
    if compare_with_prev:
        manifest["outputs"].append("centered_vs_biased_corrected_comparison.csv")

    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n========================================")
    print(f"Búsqueda Corregida Finalizada.")
    print(f"Raíces recalculadas: {total_recalc_roots}")
    print(f"Raíces dominadas por 1+WN: {total_sign_audit_1_plus_wins}")
    print(f"Semillas consistentes DC/fasor: {total_seeds_passed_consistency}")
    print(f"Continuaciones afines ok: {total_affine_continuation_ok}")
    print(f"Candidatos periódicos: {total_periodic}")
    print(f"Candidatos no periódicos: {total_nonperiodic}")
    print(f"Candidatos caóticos: {total_chaotic}")
    print(f"Resultados guardados en: {outdir}")
    print(f"========================================")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Búsqueda corregida de semillas para Chua fraccionario no suave."
    )
    parser.add_argument("--q", type=float, default=0.9998)
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--harmonic-sign", choices=["plus", "minus"], default="plus")
    parser.add_argument("--eta-step", type=float, default=0.05)
    parser.add_argument("--memory-mode", choices=["full", "window"], default="full")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--outdir", type=str, default="outputs/biased_saturation_search_q09998_corrected")
    parser.add_argument("--rerun-priority-cases", action="store_true")
    parser.add_argument("--compare-with-previous", action="store_true")
    parser.add_argument("--no-hiddenness-tests", action="store_true")

    args = parser.parse_args()

    # Define grids
    if args.rerun_priority_cases or args.quick:
        # Just run the cases of interest
        m1_vals = [-1.1468, -1.20, -1.25]
        m0_vals = [-0.1768, -0.20, -0.24]
    else:
        m1_vals = [-1.1468, -1.20, -1.25]
        m0_vals = [-0.1768, -0.20, -0.24]

    if args.quick:
        t_transient = 10.0
        t_keep = 10.0
        t_sim_final = 50.0
        t_sim_transient = 10.0
    else:
        t_transient = 30.0
        t_keep = 30.0
        t_sim_final = 300.0
        t_sim_transient = 100.0

    run_corrected_workflow(
        q=args.q,
        h=args.h,
        sign_convention=args.harmonic_sign,
        eta_step=args.eta_step,
        memory_mode=args.memory_mode,
        outdir=Path(args.outdir),
        m1_grid=m1_vals,
        m0_grid=m0_vals,
        t_transient=t_transient,
        t_keep=t_keep,
        t_sim_final=t_sim_final,
        t_sim_transient=t_sim_transient,
        rerun_priority=args.rerun_priority_cases,
        compare_with_prev=args.compare_with_previous,
    )
