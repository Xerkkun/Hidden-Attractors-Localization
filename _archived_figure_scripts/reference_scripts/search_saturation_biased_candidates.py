#!/usr/bin/env python3
"""Search for seeds of the non-smooth fractional Chua system using a biased describing function.

This script implements:
1. Biased describing function computation by numerical quadrature.
2. Algebraic residual evaluation of the biased harmonic balance equations.
3. Multi-start grid-based root search for biased candidates.
4. Monolithic Caputo ABM continuation preserving the DC bias.
5. Final Caputo ABM simulations of the original system from continued seeds.
6. Diagnostic classification using FFT/PSD and trajectory periodicity checks.
7. Detailed visual plotting of DF surfaces, residues, continuation paths, attractors,
   time series, power spectra, and comparison against centered branches.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, root
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


# ── 1. Biased Describing Function ─────────────────────────────────────────────

def biased_saturation_describing_function(
    A: float, c: float, g: float, n_theta: int = 8192
) -> Tuple[float, float]:
    """Compute the biased describing function values psi0 and N1 by numerical quadrature.

    Parameters
    ----------
    A : float
        Oscillation amplitude. Must satisfy A >= 1e-6.
    c : float
        DC bias of the feedback coordinate.
    g : float
        Saturation gain (m0 - m1).
    n_theta : int, default 8192
        Number of quadrature points over [0, 2pi].

    Returns
    -------
    psi0 : float
        DC output component.
    N1 : float
        First-harmonic equivalent gain (psi1 / A).
    """
    if A < 1e-6:
        return 0.0, 0.0

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=real_dtype)
    sigma = c + A * np.cos(theta)
    psi = g * np.clip(sigma, -1.0, 1.0)

    psi0 = float(np.mean(psi))
    psi1 = 2.0 * float(np.mean(psi * np.cos(theta)))
    N1 = psi1 / A
    return psi0, N1


# ── 2. Algebraic Residual ─────────────────────────────────────────────────────

def biased_saturation_residual(
    A: float, c: float, omega: float, params: ChuaParameters, q: float
) -> np.ndarray:
    """Evaluate the 3D algebraic residual of the biased harmonic balance system.

    Parameters
    ----------
    A : float
        Oscillation amplitude.
    c : float
        DC bias.
    omega : float
        Angular frequency.
    params : ChuaParameters
        Chua circuit parameters.
    q : float
        Caputo fractional order.

    Returns
    -------
    residual : np.ndarray, shape (3,)
        [F0, F1, F2] representing DC and complex first-harmonic balance.
    """
    if A < 1e-6:
        # Penalize small amplitudes to push solver into valid regime
        return np.array([c, 1e2, 1e2], dtype=float)

    g = chua_gain(params)
    psi0, N1 = biased_saturation_describing_function(A, c, g)

    pmat, qvec, rvec = chua_matrices(params)

    # F0 = c - r^T (-P^(-1) b psi0)
    try:
        x_bar = np.linalg.solve(pmat, -qvec * psi0)
        F0 = c - float(rvec @ x_bar)
    except np.linalg.LinAlgError:
        F0 = 1e3

    # Transfer function: Wq(omega) = r^T (lambda I - P)^(-1) b
    try:
        lam = fractional_iomega_power(omega, q)
        matrix = lam * np.eye(3, dtype=complex_dtype) - pmat.astype(complex_dtype)
        inv_qvec = np.linalg.solve(matrix, qvec.astype(complex_dtype))
        Wq = complex_dtype(rvec.astype(complex_dtype) @ inv_qvec)
    except np.linalg.LinAlgError:
        Wq = 0.0

    # F1, F2 = Re, Im of (1 - Wq * N1)
    term = 1.0 - Wq * N1
    F1 = float(np.real(term))
    F2 = float(np.imag(term))

    return np.array([F0, F1, F2], dtype=float)


# ── 3. Multi-Start Grid Search ────────────────────────────────────────────────

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
) -> List[Dict[str, Any]]:
    """Perform multi-start grid search to find unique biased describing function roots.

    Parameters
    ----------
    params : ChuaParameters
        Chua system parameters.
    q : float
        Fractional order.
    A_range : tuple of float
        Range of amplitude starting grid.
    c_range : tuple of float
        Range of bias starting grid.
    omega_range : tuple of float
        Range of frequency starting grid.
    n_A, n_c, n_omega : int
        Number of grid divisions.
    residual_tol : float
        Tolerance for accepting a candidate as a root.
    A_tol, c_tol, omega_tol : float
        Thresholds for duplicate grouping.

    Returns
    -------
    roots : list of dict
        Unique root definitions.
    """
    q_val = validate_fractional_order(q)
    A_grid = np.linspace(A_range[0], A_range[1], n_A)
    c_grid = np.linspace(c_range[0], c_range[1], n_c)
    omega_grid = np.linspace(omega_range[0], omega_range[1], n_omega)

    raw_candidates = []

    def residual_func(x):
        return biased_saturation_residual(x[0], x[1], x[2], params, q_val)

    # Multi-start loop
    for A0 in A_grid:
        for c0 in c_grid:
            for w0 in omega_grid:
                # Solve using least_squares with bounds to enforce A > 0 and omega > 0
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

    # Filter out duplicates
    unique_candidates: List[Dict[str, Any]] = []
    for A_c, c_c, w_c, res_c in raw_candidates:
        # Ensure roots are physical
        if A_c < 0.5 or w_c < 0.5 or w_c > 6.0:
            continue
        # Check against duplicates
        dup = False
        for existing in unique_candidates:
            dA = abs(A_c - existing["A"])
            dc = abs(c_c - existing["c"])
            dw = abs(w_c - existing["omega"])
            if dA < A_tol and dc < c_tol and dw < omega_tol:
                # Keep the one with the smaller residual
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

    # Sort by residual norm
    unique_candidates.sort(key=lambda x: x["residual_norm"])
    return unique_candidates


# ── 4. Seed Construction ──────────────────────────────────────────────────────

def build_biased_fractional_seed(
    params: ChuaParameters, q: float, A: float, c: float, omega: float, psi0: float, N1: float
) -> Dict[str, Any]:
    """Reconstruct a biased Lur'e seed from DC and first-harmonic balance equations.

    Parameters
    ----------
    params : ChuaParameters
        Chua circuit parameters.
    q : float
        Fractional order.
    A : float
        Oscillation amplitude.
    c : float
        DC bias of the feedback coordinate.
    omega : float
        Angular frequency.
    psi0 : float
        DC output component.
    N1 : float
        First-harmonic equivalent gain.

    Returns
    -------
    seed_report : dict
         Dataclass-like dict containing the seed state and constituent vectors.
    """
    q_val = validate_fractional_order(q)
    pmat, qvec, rvec = chua_matrices(params)

    # DC component: x_bar = -P^(-1) b psi0
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
        "A": A,
        "c": c,
        "omega": omega,
        "N1": N1,
        "psi0": psi0,
    }


# ── 5. Monolithic Biased Continuation ──────────────────────────────────────────

def run_biased_fractional_continuation_monolithic(
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
    """Execute monolithic Caputo ABM continuation preserving the DC bias state.

    The deformed field is:
    f_eta(X) = P0 X + eta * b * [ psi(r^T X) - N1 * r^T X ] + (1 - eta) * b * [ psi0 - N1 * c ]
    where P0 = P + N1 * b * r^T.
    """
    dim = 3
    h = float(h)
    q = float(q)

    nsteps_tr = int(np.ceil(t_transient / h))
    nsteps_kp = int(np.ceil(t_keep / h))
    steps_per_stage = nsteps_tr + nsteps_kp
    num_stages = len(lambda_values)
    total_new_steps = num_stages * steps_per_stage

    # Pre-allocate history
    t_arr = np.zeros(1 + total_new_steps, dtype=float)
    x_arr = np.zeros((1 + total_new_steps, dim), dtype=float)
    f_arr = np.zeros((1 + total_new_steps, dim), dtype=float)

    t_arr[0] = 0.0
    x_arr[0] = seed_x0

    pmat, qvec, rvec = chua_matrices(params)
    P0 = pmat + N1 * np.outer(qvec, rvec)
    bias_coef = qvec * (psi0 - N1 * c)
    g = chua_gain(params)

    def eval_rhs_deformed(x: np.ndarray, eta_val: float) -> np.ndarray:
        sigma = float(rvec @ x)
        psi_val = g * np.clip(sigma, -1.0, 1.0)
        return P0 @ x + eta_val * qvec * (psi_val - N1 * sigma) + (1.0 - eta_val) * bias_coef

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
            # Save incomplete stage info
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

        # Save successful stage info
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


# ── 6. Plotting Utilities ─────────────────────────────────────────────────────

def plot_df_surfaces(
    params: ChuaParameters,
    root_A: float,
    root_c: float,
    root_N1: float,
    root_psi0: float,
    outdir: Path,
    prefix: str,
) -> Path:
    """Generate surface plots of describing function terms and mark the root."""
    g = chua_gain(params)
    A_vals = np.linspace(0.1, 10.0, 100)
    c_vals = np.linspace(-6.0, 6.0, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: N1(A, c) vs A for various c
    for c_select in [-4.0, -2.0, 0.0, 2.0, 4.0]:
        n1_curve = [biased_saturation_describing_function(a, c_select, g)[1] for a in A_vals]
        ax1.plot(A_vals, n1_curve, label=f"c = {c_select:.1f}")
    # Mark root
    ax1.plot(root_A, root_N1, "ro", markersize=8, label=f"Raíz (A={root_A:.3f}, N1={root_N1:.3f})")
    ax1.set_xlabel("Amplitud A")
    ax1.set_ylabel("Ganancia equivalente N1(A, c)")
    ax1.set_title("Primer armónico N1 vs Amplitud")
    ax1.grid(True)
    ax1.legend()

    # Panel 2: psi0(A, c) vs c for various A
    for A_select in [1.0, 3.0, 5.0, 8.0]:
        psi0_curve = [biased_saturation_describing_function(A_select, c_val, g)[0] for c_val in c_vals]
        ax2.plot(c_vals, psi0_curve, label=f"A = {A_select:.1f}")
    # Mark root
    ax2.plot(root_c, root_psi0, "ro", markersize=8, label=f"Raíz (c={root_c:.3f}, psi0={root_psi0:.3f})")
    ax2.set_xlabel("DC Bias c")
    ax2.set_ylabel("DC Output component psi0(A, c)")
    ax2.set_title("Componente DC psi0 vs Bias")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    fig_path = outdir / "df_surfaces" / f"{prefix}_df_surfaces.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def plot_harmonic_residuals(
    params: ChuaParameters,
    q: float,
    root_A: float,
    root_c: float,
    root_omega: float,
    outdir: Path,
    prefix: str,
) -> Path:
    """Generate a residue heatmap in the (A, c) plane and mark the root."""
    A_grid = np.linspace(0.5, 10.0, 60)
    c_grid = np.linspace(-8.0, 8.0, 60)
    residuals = np.zeros((len(c_grid), len(A_grid)))

    for i, cv in enumerate(c_grid):
        for j, Av in enumerate(A_grid):
            res = biased_saturation_residual(Av, cv, root_omega, params, q)
            residuals[i, j] = np.log10(np.linalg.norm(res) + 1e-15)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(A_grid, c_grid, residuals, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="log10(Norma Residual)")
    ax.plot(root_A, root_c, "ro", markersize=9, label="Raíz solucionada")
    ax.set_xlabel("Amplitud A")
    ax.set_ylabel("Bias c")
    ax.set_title(f"Residual Map at omega = {root_omega:.3f} (q = {q:.4f})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    fig_path = outdir / "harmonic_residuals" / f"{prefix}_residuals.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def plot_continuation_path(steps: List[Dict[str, Any]], outdir: Path, prefix: str) -> Path:
    """Plot continuation metrics: norm, states, and RMS vs eta."""
    etas = [s["lambda_value"] for s in steps]
    norms = [s["x_out_norm"] for s in steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Norm vs eta
    ax1.plot(etas, norms, "b-o", linewidth=2)
    ax1.set_xlabel("Parámetro de Deformación eta")
    ax1.set_ylabel("Norma del estado final ||X_final||")
    ax1.set_title("Camino de Continuación: Norma del Estado vs eta")
    ax1.grid(True)

    # Individual coordinates vs eta
    coords = np.array([s["x_out"] for s in steps])
    ax2.plot(etas, coords[:, 0], "-o", label="x")
    ax2.plot(etas, coords[:, 1], "-o", label="y")
    ax2.plot(etas, coords[:, 2], "-o", label="z")
    ax2.set_xlabel("eta")
    ax2.set_ylabel("Coordenadas de x_out")
    ax2.set_title("Coordenadas del Estado Final vs eta")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    fig_path = outdir / "continuation" / f"{prefix}_continuation.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def plot_final_attractor(trajectory: np.ndarray, outdir: Path, prefix: str) -> Path:
    """Plot the final 3D attractor and its xy, xz, yz projections."""
    states = trajectory[:, 1:4]
    centroid = np.mean(states, axis=0)

    fig = plt.figure(figsize=(15, 10))

    # 3D plot
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(states[:, 0], states[:, 1], states[:, 2], "b-", alpha=0.7, linewidth=0.8)
    ax3d.scatter(centroid[0], centroid[1], centroid[2], color="red", s=100, label="Centroide")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("Atractor Final 3D Post-Transitorio")
    ax3d.legend()

    # Projections
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.plot(states[:, 0], states[:, 1], "g-", alpha=0.7, linewidth=0.8)
    ax_xy.scatter(centroid[0], centroid[1], color="red", s=50)
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_title("Proyección xy")
    ax_xy.grid(True)

    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.plot(states[:, 0], states[:, 2], "r-", alpha=0.7, linewidth=0.8)
    ax_xz.scatter(centroid[0], centroid[2], color="red", s=50)
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.set_title("Proyección xz")
    ax_xz.grid(True)

    ax_yz = fig.add_subplot(2, 2, 4)
    ax_yz.plot(states[:, 1], states[:, 2], "c-", alpha=0.7, linewidth=0.8)
    ax_yz.scatter(centroid[1], centroid[2], color="red", s=50)
    ax_yz.set_xlabel("y")
    ax_yz.set_ylabel("z")
    ax_yz.set_title("Proyección yz")
    ax_yz.grid(True)

    plt.tight_layout()
    fig_path = outdir / "attractors" / f"{prefix}_attractor.png"
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
    ax1.set_title("Series Temporales del Atractor Final")

    ax2.plot(times, states[:, 1], "g-", linewidth=1.0)
    ax2.set_ylabel("y(t)")
    ax2.grid(True)

    ax3.plot(times, states[:, 2], "r-", linewidth=1.0)
    ax3.set_ylabel("z(t)")
    ax3.set_xlabel("Tiempo t (s)")
    ax3.grid(True)

    plt.tight_layout()
    fig_path = outdir / "timeseries" / f"{prefix}_timeseries.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def plot_fft_psd(trajectory: np.ndarray, h: float, outdir: Path, prefix: str) -> Tuple[Path, float]:
    """Plot FFT amplitude spectrum of x(t) and mark the dominant frequency."""
    times = trajectory[:, 0]
    x_val = trajectory[:, 1]
    
    spec = fft_spectrum(x_val, h)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    if spec.frequency_hz.size > 0:
        ax.plot(spec.frequency_rad_s, spec.values, "b-", linewidth=1.2)
        dom_idx = np.argmax(spec.values)
        dom_freq_rad = spec.frequency_rad_s[dom_idx]
        dom_freq_hz = spec.frequency_hz[dom_idx]
        
        ax.axvline(dom_freq_rad, color="red", linestyle="--", alpha=0.7)
        ax.plot(dom_freq_rad, spec.values[dom_idx], "ro", label=f"Dominante: {dom_freq_rad:.3f} rad/s ({dom_freq_hz:.3f} Hz)")
    else:
        dom_freq_rad = 0.0
        ax.text(0.5, 0.5, "Espectro vacío o datos insuficientes", transform=ax.transAxes, ha="center")

    ax.set_xlabel("Frecuencia angular (rad/s)")
    ax.set_ylabel("Amplitud FFT (normalizada)")
    ax.set_title("Espectro de Amplitud FFT para x(t)")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    fig_path = outdir / "fft" / f"{prefix}_fft.png"
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
    """Generate comparison figures (3D state space and FFT overlay) with centered candidates."""
    fig = plt.figure(figsize=(16, 7))

    # Panel 1: 3D Trajectory overlay
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(biased_traj[:, 1], biased_traj[:, 2], biased_traj[:, 3], "b-", alpha=0.7, linewidth=0.8, label="Sesgado")
    
    biased_centroid = np.mean(biased_traj[:, 1:4], axis=0)
    ax3d.scatter(biased_centroid[0], biased_centroid[1], biased_centroid[2], color="darkblue", s=100, marker="o")
    
    if centered_traj is not None and len(centered_traj) > 0:
        ax3d.plot(centered_traj[:, 1], centered_traj[:, 2], centered_traj[:, 3], "r--", alpha=0.6, linewidth=0.8, label="Centrado")
        centered_centroid = np.mean(centered_traj[:, 1:4], axis=0)
        ax3d.scatter(centered_centroid[0], centered_centroid[1], centered_centroid[2], color="darkred", s=100, marker="X")
    
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("Comparación de Atractores 3D")
    ax3d.legend()

    # Panel 2: FFT overlay
    ax_fft = fig.add_subplot(1, 2, 2)
    h_b = float(np.median(np.diff(biased_traj[:, 0])))
    spec_b = fft_spectrum(biased_traj[:, 1], h_b)
    if spec_b.frequency_hz.size > 0:
        ax_fft.plot(spec_b.frequency_rad_s, spec_b.values, "b-", alpha=0.8, label="Sesgado")
    
    if centered_traj is not None and len(centered_traj) > 0:
        h_c = float(np.median(np.diff(centered_traj[:, 0])))
        spec_c = fft_spectrum(centered_traj[:, 1], h_c)
        if spec_c.frequency_hz.size > 0:
            ax_fft.plot(spec_c.frequency_rad_s, spec_c.values, "r--", alpha=0.7, label="Centrado")
            
    ax_fft.set_xlabel("Frecuencia angular (rad/s)")
    ax_fft.set_ylabel("Amplitud")
    ax_fft.set_title("Comparación de Espectros FFT")
    ax_fft.grid(True)
    ax_fft.legend()

    plt.tight_layout()
    fig_path = outdir / "comparisons" / f"{prefix}_comparison.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


# ── 7. Main Runner ────────────────────────────────────────────────────────────

def run_biased_saturation_search_workflow(
    q: float,
    h: float,
    memory_mode: str,
    outdir: Path,
    m1_grid: List[float],
    m0_grid: List[float],
    t_transient: float,
    t_keep: float,
    t_sim_final: float,
    t_sim_transient: float,
    max_candidates_per_param: int,
) -> None:
    """Run search, continuation, simulation, plotting, and comparison loops."""
    # Base circuit parameters
    alpha = 8.4562
    beta = 12.0732
    gamma = 0.0052

    # Create directories
    outdir.mkdir(parents=True, exist_ok=True)
    for sub in ["figures", "trajectories", "continuation_steps"]:
        (outdir / sub).mkdir(parents=True, exist_ok=True)
    for fig_sub in ["df_surfaces", "harmonic_residuals", "continuation", "attractors", "timeseries", "fft", "comparisons"]:
        (outdir / "figures" / fig_sub).mkdir(parents=True, exist_ok=True)

    # Outputs lists
    summary_rows: List[Dict[str, Any]] = []
    roots_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    manifest_figures: List[str] = []

    total_biased_roots_found = 0
    total_survived_continuation = 0
    total_nonperiodic = 0

    lambda_values = list(np.linspace(0.0, 1.0, 11))

    # Grid search loop
    for m1 in m1_grid:
        for m0 in m0_grid:
            print(f"\n========================================")
            print(f"Analizando parámetros: m1={m1}, m0={m0}")
            print(f"========================================")
            params = chua_parameters(
                model="nonsmooth",
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                m0=m0,
                m1=m1
            )
            g = chua_gain(params)

            # 1. Centered Branch Solving (Reference)
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
                residual_tol=1e-4
            )
            print(f"  Búsqueda sesgada encontró {len(biased_roots)} raíces.")

            # Keep only unique, non-trivial roots up to max limit
            for root_idx, r in enumerate(biased_roots[:max_candidates_per_param]):
                total_biased_roots_found += 1
                A = r["A"]
                c = r["c"]
                omega = r["omega"]
                res_norm = r["residual_norm"]

                psi0, N1 = biased_saturation_describing_function(A, c, g)
                
                # Check closeness to centered
                is_centered_duplicate = abs(c) <= 0.05
                prefix = f"biased_q{int(q*10000)}_m1_{m1:.4f}_m0_{m0:.4f}_branch_{root_idx}"
                if is_centered_duplicate:
                    prefix += f"_c_centered_like"
                else:
                    prefix += f"_c_{c:.3f}"

                prefix = prefix.replace(".", "p").replace("-", "m")
                print(f"  -- Procesando candidato sesgado {root_idx}: A={A:.3f}, c={c:.3f}, w={omega:.3f}")

                # Save root details to roots.csv
                roots_rows.append({
                    "m1": m1,
                    "m0": m0,
                    "q": q,
                    "branch": root_idx,
                    "prefix": prefix,
                    "A": A,
                    "c": c,
                    "omega": omega,
                    "N1": N1,
                    "psi0": psi0,
                    "residual_norm": res_norm,
                    "is_centered_duplicate": is_centered_duplicate,
                })

                # Build seed state
                seed_info = build_biased_fractional_seed(params, q, A, c, omega, psi0, N1)
                X_seed = seed_info["seed"]

                # 3. Continuation
                print(f"    Iniciando continuación numérica Caputo ABM...")
                cont_steps = run_biased_fractional_continuation_monolithic(
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
                print(f"    Estatus continuación: {'SOBREVIVIÓ' if survived else 'FALLÓ'}")

                if not survived:
                    summary_rows.append({
                        "prefix": prefix,
                        "m1": m1,
                        "m0": m0,
                        "q": q,
                        "A_root": A,
                        "c_root": c,
                        "omega_root": omega,
                        "N1_root": N1,
                        "psi0_root": psi0,
                        "residual_norm": res_norm,
                        "status": "biased_continuation_failed",
                        "final_centroid_x": np.nan,
                        "final_centroid_y": np.nan,
                        "final_centroid_z": np.nan,
                        "final_dominant_freq": np.nan,
                        "classification": "biased_continuation_failed",
                    })
                    continue

                total_survived_continuation += 1
                x_final_cont = cont_steps[-1]["x_out"]

                # 4. Final simulation of original system (eta = 1) from continuation end state
                print(f"    Corriendo simulación final del sistema original (eta=1)...")
                # Get the system object
                system = get_system("chua-nonsmooth")
                # Update parameters dynamically to match current grid case
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

                # Keep post-transient trajectory for analysis
                full_traj = np.column_stack((sim_t, sim_x))
                post_traj = full_traj[sim_t >= t_sim_transient]

                # Save trajectory
                traj_path = outdir / "trajectories" / f"{prefix}_trajectory.csv"
                pd.DataFrame(post_traj, columns=["t", "x", "y", "z"]).to_csv(traj_path, index=False)

                # Periodicity and attractor classification
                classification = "biased_collapsed_to_equilibrium"
                final_centroid = np.zeros(3)
                dom_freq_rad = 0.0

                if len(post_traj) > 10:
                    final_centroid = np.mean(post_traj[:, 1:4], axis=0)
                    # Use periodicity classifier
                    p_res = classify_post_transient_periodicity(post_traj, h=h)
                    lbl = p_res["candidate_label"]
                    p_status = p_res["periodicity_status"]

                    # Check if collapsed to equilibrium (norm of derivative near 0 or ptp range very small)
                    max_range = float(np.max(np.ptp(post_traj[:, 1:4], axis=0)))
                    if max_range < 0.05:
                        classification = "biased_collapsed_to_equilibrium"
                    elif lbl == "regular_periodic_rejected":
                        classification = "biased_regular_periodic_rejected"
                    elif lbl == "chaotic_candidate_pending_robustness":
                        classification = "biased_chaotic_candidate_pending_robustness"
                        total_nonperiodic += 1
                    else:
                        classification = "biased_nonperiodic_candidate"
                        total_nonperiodic += 1
                else:
                    classification = "biased_collapsed_to_equilibrium"

                # 5. Generate Centered Reference for Comparison if available
                # Use the closest centered candidate
                centered_traj = None
                centered_seed_rep = None
                if centered_seeds:
                    # Choose centered seed closest in omega/A
                    c_seed = min(centered_seeds, key=lambda s: abs(s.omega - omega))
                    # Run centered continuation (using monolithic ABM via library)
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
                        # Run final simulation for centered
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
                        centered_seed_rep = c_seed
                    except Exception as e:
                        print(f"      [Centered Reference] Error al correr continuación/simulación de referencia: {e}")

                # 6. Compute Comparison Metrics
                trajectory_distance_to_centered = np.nan
                seed_distance_to_centered = np.nan
                diff_A = np.nan
                diff_omega = np.nan
                diff_k = np.nan
                same_attractor = "n/a"

                if centered_seed_rep is not None:
                    seed_distance_to_centered = float(np.linalg.norm(X_seed - centered_seed_rep.seed))
                    diff_A = float(A - centered_seed_rep.amplitude)
                    diff_omega = float(omega - centered_seed_rep.omega)
                    diff_k = float(N1 - centered_seed_rep.gain)

                if centered_traj is not None and len(centered_traj) > 0 and len(post_traj) > 0:
                    centered_centroid = np.mean(centered_traj[:, 1:4], axis=0)
                    trajectory_distance_to_centered = float(np.linalg.norm(final_centroid - centered_centroid))

                    # Bounding boxes
                    b_min = np.min(post_traj[:, 1:4], axis=0)
                    b_max = np.max(post_traj[:, 1:4], axis=0)
                    c_min = np.min(centered_traj[:, 1:4], axis=0)
                    c_max = np.max(centered_traj[:, 1:4], axis=0)

                    # Simple heuristic for same attractor (centroid distance < 0.2 and box overlap)
                    box_dist = np.linalg.norm((b_min - c_min) + (b_max - c_max))
                    if trajectory_distance_to_centered < 0.2 and box_dist < 0.4:
                        same_attractor = "yes"
                    else:
                        same_attractor = "no"

                # Save comparison row
                comparison_rows.append({
                    "prefix": prefix,
                    "m1": m1,
                    "m0": m0,
                    "seed_distance_to_centered": seed_distance_to_centered,
                    "trajectory_distance_to_centered": trajectory_distance_to_centered,
                    "diff_A": diff_A,
                    "diff_omega": diff_omega,
                    "diff_k": diff_k,
                    "same_attractor": same_attractor,
                })

                # 7. Generate Figures
                print(f"    Generando figuras del candidato...")
                df_fig = plot_df_surfaces(params, A, c, N1, psi0, outdir, prefix)
                res_fig = plot_harmonic_residuals(params, q, A, c, omega, outdir, prefix)
                cont_fig = plot_continuation_path(cont_steps, outdir, prefix)
                att_fig = plot_final_attractor(post_traj, outdir, prefix)
                ts_fig = plot_time_series(post_traj, outdir, prefix)
                fft_fig, dom_freq_rad = plot_fft_psd(post_traj, h, outdir, prefix)
                comp_fig = plot_centered_comparison(post_traj, centered_traj, outdir, prefix)

                manifest_figures.extend([
                    str(df_fig.relative_to(outdir)),
                    str(res_fig.relative_to(outdir)),
                    str(cont_fig.relative_to(outdir)),
                    str(att_fig.relative_to(outdir)),
                    str(ts_fig.relative_to(outdir)),
                    str(fft_fig.relative_to(outdir)),
                    str(comp_fig.relative_to(outdir)),
                ])

                # Save final summary details
                summary_rows.append({
                    "prefix": prefix,
                    "m1": m1,
                    "m0": m0,
                    "q": q,
                    "A_root": A,
                    "c_root": c,
                    "omega_root": omega,
                    "N1_root": N1,
                    "psi0_root": psi0,
                    "residual_norm": res_norm,
                    "status": "ok",
                    "final_centroid_x": final_centroid[0],
                    "final_centroid_y": final_centroid[1],
                    "final_centroid_z": final_centroid[2],
                    "final_dominant_freq": dom_freq_rad,
                    "classification": classification,
                })

    # Write summary files
    pd.DataFrame(summary_rows).to_csv(outdir / "summary.csv", index=False)
    pd.DataFrame(roots_rows).to_csv(outdir / "roots.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(outdir / "centered_comparison.csv", index=False)

    # Write manifest.json
    manifest = {
        "search_parameters": {
            "q": q,
            "h": h,
            "memory_mode": memory_mode,
            "m1_grid": m1_grid,
            "m0_grid": m0_grid,
            "t_transient": t_transient,
            "t_keep": t_keep,
            "t_sim_final": t_sim_final,
            "t_sim_transient": t_sim_transient,
        },
        "statistics": {
            "total_biased_roots_found": total_biased_roots_found,
            "total_survived_continuation": total_survived_continuation,
            "total_nonperiodic": total_nonperiodic,
        },
        "figures": manifest_figures,
        "outputs": [
            "summary.csv",
            "roots.csv",
            "centered_comparison.csv",
        ]
    }
    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n========================================")
    print(f"Búsqueda finalizada.")
    print(f"Total raíces encontradas: {total_biased_roots_found}")
    print(f"Total continuadas con éxito: {total_survived_continuation}")
    print(f"Total candidatos no periódicos/caóticos: {total_nonperiodic}")
    print(f"Resultados guardados en: {outdir}")
    print(f"========================================")


# ── 8. Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Búsqueda de semillas para Chua fraccionario no suave usando función descriptiva sesgada."
    )
    parser.add_argument("--quick", action="store_true", help="Usa una simulación y grilla reducidas para verificación rápida.")
    parser.add_argument("--q", type=float, default=0.9998, help="Orden Caputo fraccionario.")
    parser.add_argument("--h", type=float, default=0.01, help="Paso de tiempo de integración.")
    parser.add_argument("--memory-mode", choices=["full", "window"], default="full", help="Modo de memoria Caputo.")
    parser.add_argument("--outdir", type=str, default="outputs/biased_saturation_search_q09998", help="Directorio de salida.")
    parser.add_argument("--m1-grid", type=float, nargs="+", default=None, help="Malla de valores para m1.")
    parser.add_argument("--m0-grid", type=float, nargs="+", default=None, help="Malla de valores para m0.")
    parser.add_argument("--max-candidates-per-parameter", type=int, default=2, help="Límite de candidatos sesgados por combinación de parámetros.")

    args = parser.parse_args()

    # Define grids based on mode
    if args.quick:
        # Quick grid
        m1_vals = args.m1_grid or [-1.20]
        m0_vals = args.m0_grid or [-0.20]
        t_transient = 10.0
        t_keep = 10.0
        t_sim_final = 50.0
        t_sim_transient = 10.0
    else:
        # Full grid
        m1_vals = args.m1_grid or [-1.1468, -1.20, -1.25]
        m0_vals = args.m0_grid or [-0.1768, -0.20, -0.24]
        t_transient = 30.0
        t_keep = 30.0
        t_sim_final = 300.0
        t_sim_transient = 100.0

    run_biased_saturation_search_workflow(
        q=args.q,
        h=args.h,
        memory_mode=args.memory_mode,
        outdir=Path(args.outdir),
        m1_grid=m1_vals,
        m0_grid=m0_vals,
        t_transient=t_transient,
        t_keep=t_keep,
        t_sim_final=t_sim_final,
        t_sim_transient=t_sim_transient,
        max_candidates_per_param=args.max_candidates_per_parameter,
    )
