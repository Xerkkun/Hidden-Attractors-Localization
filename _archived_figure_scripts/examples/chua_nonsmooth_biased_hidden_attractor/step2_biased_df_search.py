#!/usr/bin/env python3
"""
Paso 2: Búsqueda con Función Descriptiva Sesgada (DF Sesgada Corregida)
========================================================================
Implementa la búsqueda de raíces de la DF sesgada para el sistema de Chua
fraccionario no suave, usando la convención de signo corregida:

    1 + Wq(jω) · N₁(A, c) = 0

donde:
  - Wq(jω) = transferencia fraccionaria del Lur'e lineal
  - N₁(A, c) = primer coeficiente de Fourier de la DF sesgada (función de A y c)
  - c = bias DC de la señal de entrada a la no linealidad

El proceso completo es:
  1. Grid search multi-arranque en (A, c, ω) → raíces de la DF sesgada
  2. Auditoría de convención de signo (1+WN vs 1-WN)
  3. Reconstrucción algebraica de la semilla sesgada con verificación:
       - Error DC: |r^T x̄ - c| < 1e-5
       - Error fasorial: ||X₁|| ≈ A
  4. Verificación de identidad homotópica en η=1
  5. Continuación afín Caputo ABM desde η=0 (sistema linealizado) a η=1 (sistema real)
  6. Simulación final y clasificación de periodicidad
  7. Comparación con la referencia centrada del Paso 1

Salidas generadas:
  - roots_corrected.csv             : todas las raíces encontradas
  - roots_sign_audit.csv            : auditoría de convención de signo
  - seed_consistency_checks.csv     : errores de consistencia algebraica
  - affine_homotopy_identity.csv    : verificación de identidad η=1
  - affine_continuation_summary.csv : resumen de continuación por candidato
  - final_classification.csv        : clasificación final de cada candidato
  - trajectories/                   : CSV por candidato que sobrevive
  - plots/                          : figuras del reporte
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import gamma as gamma_func

# ── Path setup ────────────────────────────────────────────────────────────────
EXAMPLE_DIR = Path(__file__).resolve().parent
VERSION2    = EXAMPLE_DIR.parents[1]
ROOT        = VERSION2.parent

for p in [str(VERSION2), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml

from hidden_attractors.analysis.spectral import fft_spectrum
from hidden_attractors.diagnostics.periodicity import classify_post_transient_periodicity
from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.models.chua import ChuaParameters, chua_parameters
from hidden_attractors.seed_generation.chua import chua_matrices, chua_gain
from hidden_attractors.seed_generation.core import (
    complex_dtype,
    fractional_iomega_power,
    real_dtype,
    validate_fractional_order,
)
from hidden_attractors.systems import get_system

CFG_PATH = VERSION2 / "configs" / "examples" / "chua_nonsmooth_biased_df_search.yaml"


def load_config() -> Dict[str, Any]:
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════════════
# A. Función de Transferencia Fraccionaria
# ══════════════════════════════════════════════════════════════════════════════

def get_Wq(omega: float, q: float, pmat: np.ndarray,
           qvec: np.ndarray, rvec: np.ndarray) -> complex:
    """Evalúa W_q(jω) = r^T (P - (jω)^q I)^{-1} b.

    Convención: 1 + Wq·N₁ = 0  → Re(Wq) < 0 para ramas estables.
    """
    matrix = pmat.astype(complex_dtype) - fractional_iomega_power(omega, q) * np.eye(3, dtype=complex_dtype)
    return complex_dtype(
        (rvec.astype(complex_dtype).reshape(1, -1)
         @ np.linalg.inv(matrix)
         @ qvec.astype(complex_dtype).reshape(-1, 1))[0, 0]
    )


def harmonic_residual_sign_audit(W: complex, N1: float) -> Dict[str, float]:
    """Residuos absolutos para ambas convenciones de signo."""
    R_plus  = 1.0 + W * N1
    R_minus = 1.0 - W * N1
    return {
        "R_plus_real":  float(R_plus.real),
        "R_plus_imag":  float(R_plus.imag),
        "R_plus_abs":   float(abs(R_plus)),
        "R_minus_real": float(R_minus.real),
        "R_minus_imag": float(R_minus.imag),
        "R_minus_abs":  float(abs(R_minus)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# B. Función Descriptiva Sesgada
# ══════════════════════════════════════════════════════════════════════════════

def biased_saturation_df(A: float, c: float, g: float,
                          n_theta: int = 8192) -> Tuple[float, float]:
    """Calcula ψ₀ y N₁ por cuadratura numérica.

    Para la saturación bilineal f(σ) = g·sat(σ):
      ψ₀ = (1/2π) ∫₀²π f(c + A·cos θ) dθ   (componente DC)
      N₁  = (2/π) ∫₀²π f(c + A·cos θ)·cos θ dθ / A

    Args:
        A      : amplitud del armónico fundamental
        c      : bias DC de la señal de entrada
        g      : ganancia de la no linealidad (m0 - m1)
        n_theta: puntos de cuadratura

    Returns:
        (ψ₀, N₁)
    """
    if A < 1e-6:
        return 0.0, 0.0
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=real_dtype)
    sigma = c + A * np.cos(theta)
    psi   = g * np.clip(sigma, -1.0, 1.0)
    psi0  = float(np.mean(psi))
    psi1  = 2.0 * float(np.mean(psi * np.cos(theta)))
    return psi0, psi1 / A


def biased_saturation_residual(A: float, c: float, omega: float,
                                params: ChuaParameters, q: float,
                                n_theta: int = 8192) -> np.ndarray:
    """Residuo 3D del sistema sesgado bajo la convención 1 + Wq·N₁ = 0.

    Componentes:
        F₀ = c - r^T x̄        (sesgo DC)
        F₁ = Re(1 + Wq·N₁)    (balance armónico real)
        F₂ = Im(1 + Wq·N₁)    (balance armónico imaginario)
    """
    if A < 1e-6:
        return np.array([c, 1e2, 1e2], dtype=float)

    g    = chua_gain(params)
    psi0, N1 = biased_saturation_df(A, c, g, n_theta)
    pmat, qvec, rvec = chua_matrices(params)

    try:
        x_bar = np.linalg.solve(pmat, -qvec * psi0)
        F0 = c - float(rvec @ x_bar)
    except np.linalg.LinAlgError:
        F0 = 1e3

    try:
        Wq = get_Wq(omega, q, pmat, qvec, rvec)
    except np.linalg.LinAlgError:
        Wq = 0.0

    term = 1.0 + Wq * N1
    return np.array([F0, float(term.real), float(term.imag)], dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# C. Búsqueda Grid Multi-arranque
# ══════════════════════════════════════════════════════════════════════════════

def find_biased_branches(params: ChuaParameters, q: float,
                          s2_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Grid search multi-arranque en (A, c, ω) para raíces de la DF sesgada."""
    q_val = validate_fractional_order(q)
    n_theta = int(s2_cfg.get("n_theta", 8192))

    A_grid     = np.linspace(*s2_cfg["A_range"],     int(s2_cfg["n_A"]))
    c_grid     = np.linspace(*s2_cfg["c_range"],     int(s2_cfg["n_c"]))
    omega_grid = np.linspace(*s2_cfg["omega_range"], int(s2_cfg["n_omega"]))
    tol        = float(s2_cfg["residual_tol"])

    def residual(x):
        return biased_saturation_residual(x[0], x[1], x[2], params, q_val, n_theta)

    raw = []
    for A0 in A_grid:
        for c0 in c_grid:
            for w0 in omega_grid:
                try:
                    res = least_squares(
                        residual, x0=[A0, c0, w0],
                        bounds=([1e-6, -12.0, 0.1], [25.0, 12.0, 8.0]),
                        ftol=1e-10, xtol=1e-10, max_nfev=300,
                    )
                    if res.success and np.linalg.norm(residual(res.x)) < tol:
                        raw.append((*res.x, np.linalg.norm(residual(res.x))))
                except Exception:
                    continue

    # Deduplicación
    A_tol = c_tol = w_tol = 1e-3
    unique: List[Dict[str, Any]] = []
    for A_c, c_c, w_c, res_c in raw:
        if A_c < 0.5 or not (0.5 <= w_c <= 6.0):
            continue
        dup = False
        for ex in unique:
            if abs(A_c - ex["A"]) < A_tol and abs(c_c - ex["c"]) < c_tol and abs(w_c - ex["omega"]) < w_tol:
                if res_c < ex["residual_norm"]:
                    ex.update({"A": A_c, "c": c_c, "omega": w_c, "residual_norm": res_c})
                dup = True
                break
        if not dup:
            unique.append({"A": A_c, "c": c_c, "omega": w_c, "residual_norm": res_c})

    unique.sort(key=lambda x: x["residual_norm"])
    return unique


# ══════════════════════════════════════════════════════════════════════════════
# D. Reconstrucción de Semilla Sesgada
# ══════════════════════════════════════════════════════════════════════════════

def build_biased_seed(params: ChuaParameters, q: float,
                       A: float, c: float, omega: float,
                       psi0: float, N1: float) -> Dict[str, Any]:
    """Reconstruye el vector de estado semilla del sistema Lur'e sesgado.

    La semilla se construye como:
        X_seed = x̄ + Re(X₁)

    donde:
        x̄  = −P⁻¹ b ψ₀          (estado de equilibrio con bias)
        X₁  = (λI − P)⁻¹ b N₁ A  (componente del primer armónico)
        λ   = (jω)^q
    """
    q_val = validate_fractional_order(q)
    pmat, qvec, rvec = chua_matrices(params)
    x_bar = np.linalg.solve(pmat, -qvec * psi0)
    lam   = fractional_iomega_power(omega, q_val)
    matrix = lam * np.eye(3, dtype=complex_dtype) - pmat.astype(complex_dtype)
    X1 = np.linalg.solve(matrix, qvec.astype(complex_dtype)) * N1 * A
    return {
        "seed":   x_bar + np.real(X1),
        "x_bar":  x_bar,
        "Re_X1":  np.real(X1),
        "Im_X1":  np.imag(X1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# E. Continuación Afín Caputo ABM
# ══════════════════════════════════════════════════════════════════════════════

def run_affine_continuation(params: ChuaParameters, q: float, h: float,
                              seed_x0: np.ndarray, A: float, c: float,
                              psi0: float, N1: float,
                              lambda_values: List[float],
                              t_transient: float, t_keep: float,
                              div_threshold: float) -> List[Dict[str, Any]]:
    """Continuación afín Caputo ABM desde η=0 hasta η=1.

    El campo deformado es:
        f_η(x) = (P + N₁ br^T)x + b(ψ₀ − N₁c) + η·b·[f(r^T x) − ψ₀ − N₁(r^T x − c)]

    En η=0: sistema linealizado alrededor del bias (fácil de integrar desde la semilla).
    En η=1: sistema original de Chua no suave.
    """
    dim = 3
    h   = float(h)
    q   = float(q)
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
    P_aff   = pmat + N1 * np.outer(qvec, rvec)
    c_aff   = qvec * (psi0 - N1 * c)
    g       = params.m0 - params.m1

    def rhs(x: np.ndarray, eta: float) -> np.ndarray:
        sigma   = float(rvec @ x)
        psi_val = g * np.clip(sigma, -1.0, 1.0)
        return P_aff @ x + c_aff + eta * qvec * (psi_val - psi0 - N1 * (sigma - c))

    f_arr[0] = rhs(x_arr[0], lambda_values[0])
    powers   = np.arange(total_new + 3, dtype=float)
    pow_q    = powers ** q
    pow_q1   = powers ** (q + 1.0)
    hq       = h ** q
    pred_sc  = hq / float(gamma_func(q + 1.0))
    gq2      = float(gamma_func(q + 2.0))
    corr_sc  = hq / gq2 if abs(gq2) > 1e-15 else 0.0

    records, curr_n, diverged = [], 0, False

    for eta in lambda_values:
        if diverged:
            break
        x_in     = x_arr[curr_n].copy()
        stage_ok = True

        for local_step in range(steps_per_stage):
            n    = curr_n + local_step
            j_r  = np.arange(0, n + 1)
            b_w  = pow_q[n + 1 - j_r] - pow_q[n - j_r]
            pred = x_arr[0] + pred_sc * (b_w @ f_arr[0: n + 1])

            fp   = rhs(pred, eta)
            n_p  = n
            a0   = float(n_p) ** (q + 1) - (float(n_p) - q) * (float(n_p) + 1) ** q
            if n_p > 0:
                mid  = n - np.arange(1, n + 1)
                a_mid = pow_q1[mid + 2] + pow_q1[mid] - 2.0 * pow_q1[mid + 1]
                a_w  = np.concatenate(([a0], a_mid))
            else:
                a_w = np.array([a0])

            corrected = x_arr[0] + corr_sc * ((a_w @ f_arr[0: n + 1]) + fp)
            nrm = np.linalg.norm(corrected)

            if nrm > div_threshold or not np.all(np.isfinite(corrected)):
                diverged  = True
                stage_ok  = False
                x_arr[n + 1] = corrected if np.all(np.isfinite(corrected)) else x_arr[n]
                t_arr[n + 1] = t_arr[n] + h
                f_arr[n + 1] = f_arr[n]
                break

            x_arr[n + 1] = corrected
            t_arr[n + 1] = t_arr[n] + h
            f_arr[n + 1] = rhs(corrected, eta)

        keep_start = curr_n + nsteps_tr + 1
        keep_end   = curr_n + steps_per_stage
        keep_times = t_arr[keep_start: keep_end + 1]
        keep_states = x_arr[keep_start: keep_end + 1]
        x_out = x_arr[keep_end] if stage_ok else x_arr[curr_n + steps_per_stage]

        traj = np.column_stack((keep_times, keep_states)) if len(keep_times) > 0 else np.empty((0, 4))
        records.append({
            "lambda_value": float(eta),
            "x_in":  x_in,
            "x_out": x_out,
            "trajectory": traj,
            "status": "ok" if stage_ok else "diverged",
            "x_out_norm": float(np.linalg.norm(x_out)),
        })
        curr_n = keep_end

    return records


# ══════════════════════════════════════════════════════════════════════════════
# F. Plots del Reporte
# ══════════════════════════════════════════════════════════════════════════════

LW = 0.7
CMAP = plt.cm.plasma


def _ax_style(ax, grid=True):
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")


def _ax3d_style(ax):
    ax.grid(True, color="#cbd5e1", lw=0.3, ls="--", alpha=0.5)


def plot_sign_audit(audit_rows: List[Dict], outpath: Path) -> None:
    """Barras de auditoría de convención de signo (|1+WN| vs |1-WN|)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    idx = np.arange(len(audit_rows))
    w   = 0.35
    ab_plus  = [r["R_plus_abs"] for r in audit_rows]
    ab_minus = [r["R_minus_abs"] for r in audit_rows]
    lbls     = [f"m1={r['m1']:.3f}\nm0={r['m0']:.3f}\nc={r['c']:.2f}" for r in audit_rows]
    ax.bar(idx - w/2, ab_plus,  w, label="|1 + Wq·N₁| (convención oficial)", color="#4DBBD5")
    ax.bar(idx + w/2, ab_minus, w, label="|1 − Wq·N₁|",                      color="#E64B35", alpha=0.7)
    ax.set_xticks(idx); ax.set_xticklabels(lbls, fontsize=6)
    ax.set_ylabel("Residuo armónico absoluto")
    ax.set_title("Auditoría de Convención de Signo Armónico", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7)
    _ax_style(ax)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_attractor_report(post_traj: np.ndarray, prefix: str,
                           params_str: str, verdict: str, outpath: Path) -> None:
    """Figura de 7 paneles igual al reporte: 3D, proyecciones, series, FFT, info."""
    states = post_traj[:, 1:4]
    times  = post_traj[:, 0]
    h_est  = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    n      = len(states)
    if n < 10:
        return

    fig = plt.figure(figsize=(17, 9.5), dpi=300)
    fig.suptitle(
        f"[Chua Fraccionario No Suave  |  DF Sesgada  |  q=0.9998]\n{params_str}",
        fontsize=11, fontweight="bold", y=0.99, va="top",
    )

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 4, figure=fig, left=0.055, right=0.97, top=0.90,
                  bottom=0.07, hspace=0.55, wspace=0.38)

    # 3D (ocupa 2 filas, 2 columnas)
    ax3 = fig.add_subplot(gs[:2, :2], projection="3d")
    for i in range(n - 1):
        ax3.plot(states[i:i+2, 0], states[i:i+2, 1], states[i:i+2, 2],
                 lw=0.35, color=CMAP(i / n), alpha=0.75)
    _ax3d_style(ax3)
    ax3.set_xlabel("x", labelpad=2); ax3.set_ylabel("y", labelpad=2); ax3.set_zlabel("z", labelpad=2)
    ax3.set_title("Espacio de fase 3D", fontsize=11, fontweight='bold', pad=5)

    # Proyecciones 2D
    pairs = [("x", "y", 0, 1, gs[0, 2]), ("x", "z", 0, 2, gs[0, 3]), ("y", "z", 1, 2, gs[1, 2])]
    for xl, yl, ix, iy, gss in pairs:
        ax2 = fig.add_subplot(gss)
        ax2.plot(states[:, ix], states[:, iy], lw=0.3, color="#E64B35", alpha=0.8)
        _ax_style(ax2)
        ax2.set_xlabel(xl, fontsize=8); ax2.set_ylabel(yl, fontsize=8)
        ax2.set_title(f"Proy. {xl}-{yl}", fontsize=8)

    # FFT de x(t)
    ax_fft = fig.add_subplot(gs[1, 3])
    try:
        spec = fft_spectrum(states[:, 0], h_est)
        if spec.frequency_hz.size > 0:
            ax_fft.plot(spec.frequency_rad_s, spec.values, lw=1.0, color="#4DBBD5")
            dom_idx = np.argmax(spec.values)
            ax_fft.axvline(spec.frequency_rad_s[dom_idx], color="#E64B35", ls="--", alpha=0.7)
    except Exception:
        pass
    _ax_style(ax_fft)
    ax_fft.set_xlabel("ω (rad/s)", fontsize=8); ax_fft.set_ylabel("Amplitud", fontsize=8)
    ax_fft.set_title("Espectro FFT x(t)", fontsize=8)

    # Series de tiempo
    ts_cfg = [("x(t)", states[:, 0], "#E64B35"), ("y(t)", states[:, 1], "#4DBBD5"), ("z(t)", states[:, 2], "#00A087")]
    for col_i, (lbl, sig, clr) in enumerate(ts_cfg):
        ax_t = fig.add_subplot(gs[2, col_i])
        ax_t.plot(times, sig, lw=0.35, color=clr)
        _ax_style(ax_t)
        ax_t.set_xlabel("t [s]", fontsize=8); ax_t.set_ylabel(lbl, fontsize=8)
        ax_t.set_title(f"Serie {lbl}", fontsize=8)

    # Panel de información
    ax_i = fig.add_subplot(gs[2, 3])
    ax_i.axis("off")
    centroid = np.mean(states, axis=0)
    info = [
        "Modelo: Chua No Suave (Saturación)",
        "Orden fraccionario q = 0.9998",
        "Integrador: Caputo ABM memoria completa",
        "",
        *[f"  {p.strip()}" for p in params_str.split("|")],
        "",
        f"Veredicto: {verdict}",
        f"Puntos post-trans.: {n:,}",
        f"Centroide: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})",
    ]
    ax_i.text(0.04, 0.97, "\n".join(info), transform=ax_i.transAxes,
              va="top", ha="left", fontsize=7, fontfamily="monospace", linespacing=1.5)
    ax_i.set_title("Parámetros", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_continuation_metrics(cont_steps: List[Dict], prefix: str, outpath: Path) -> None:
    """Evolución de ||x_out|| y coordenadas vs η durante la continuación afín."""
    etas  = [s["lambda_value"] for s in cont_steps]
    norms = [s["x_out_norm"] for s in cont_steps]
    coords = np.array([s["x_out"] for s in cont_steps])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(etas, norms, "o-", color="#4DBBD5", lw=2)
    _ax_style(ax1)
    ax1.set_xlabel("η (lambda)", fontsize=9)
    ax1.set_ylabel("||x_final||", fontsize=9)
    ax1.set_title("Norma vs η — Continuación Afín", fontsize=9, fontweight='bold')

    for ci, (lbl, clr) in enumerate(zip(["x", "y", "z"], ["#E64B35", "#4DBBD5", "#00A087"])):
        ax2.plot(etas, coords[:, ci], "o-", label=lbl, color=clr, lw=1.5)
    _ax_style(ax2)
    ax2.set_xlabel("η (lambda)", fontsize=9)
    ax2.set_ylabel("Coordenada", fontsize=9)
    ax2.set_title("Coordenadas vs η", fontsize=9, fontweight='bold')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# G. Runner Principal
# ══════════════════════════════════════════════════════════════════════════════

def run_biased_df_search(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ejecuta el pipeline completo de búsqueda DF sesgada."""
    sys_cfg  = cfg["system"]
    grid_cfg = cfg["parameter_grid"]
    int_cfg  = cfg["integrator"]
    s2_cfg   = cfg["step2_biased_df_search"]
    plot_cfg = cfg["plots"]

    q           = float(sys_cfg["q"])
    h           = float(int_cfg["h"])
    memory_mode = int_cfg["memory_mode"]
    alpha  = float(sys_cfg["parameters"]["alpha"])
    beta   = float(sys_cfg["parameters"]["beta"])
    gamma  = float(sys_cfg["parameters"]["gamma"])

    t_sim_final  = float(s2_cfg["t_sim_final"])
    t_sim_trans  = float(s2_cfg["t_sim_transient"])
    t_cont_trans = float(s2_cfg["t_transient"])
    t_cont_keep  = float(s2_cfg["t_keep"])
    eta_step     = float(s2_cfg["eta_step"])
    div_thr      = float(s2_cfg["div_threshold"])
    max_br       = int(s2_cfg.get("max_branches_per_case", 2))
    lambda_vals  = list(np.arange(0.0, 1.0 + 1e-9, eta_step))

    m1_values = [float(v) for v in grid_cfg["m1_values"]]
    m0_values = [float(v) for v in grid_cfg["m0_values"]]

    out_root   = ROOT / cfg["experiment"]["output_dir"]
    out_s2     = out_root / "step2_biased_df"
    for sub in ["trajectories", "plots", "plots/sign_audit", "plots/continuation",
                "plots/attractors", "continuation_steps"]:
        (out_s2 / sub).mkdir(parents=True, exist_ok=True)

    # Acumuladores CSV
    roots_rows    : List[Dict] = []
    audit_rows    : List[Dict] = []
    seed_rows     : List[Dict] = []
    identity_rows : List[Dict] = []
    cont_rows     : List[Dict] = []
    classif_rows  : List[Dict] = []
    all_results   : List[Dict] = []

    print("=" * 65)
    print("PASO 2 - Busqueda DF Sesgada Corregida")
    print(f"  q = {q},  h = {h},  eta_step = {eta_step}")
    print(f"  Convencion de signo: 1 + Wq*N1 = 0")
    print(f"  Grilla: {len(m1_values)} x {len(m0_values)}")
    print("=" * 65)

    for m1 in m1_values:
        for m0 in m0_values:
            params = chua_parameters(
                model="nonsmooth", alpha=alpha, beta=beta, gamma=gamma, m0=m0, m1=m1,
            )
            g    = chua_gain(params)
            pmat, qvec, rvec = chua_matrices(params)

            print(f"\n  m1={m1}, m0={m0} ...")

            biased_roots = find_biased_branches(params, q, s2_cfg)
            print(f"    Raíces sesgadas: {len(biased_roots)}")

            for root_idx, r in enumerate(biased_roots[:max_br]):
                A, c, omega, res_norm = r["A"], r["c"], r["omega"], r["residual_norm"]
                psi0, N1 = biased_saturation_df(A, c, g)
                is_centered = abs(c) <= 0.05

                tag = (
                    f"biased_q{int(q*10000)}_m1_{m1:.4f}_m0_{m0:.4f}"
                    f"_branch_{root_idx}_c_{'centered_like' if is_centered else f'{c:.3f}'}"
                )
                tag = tag.replace(".", "p").replace("-", "m")

                print(f"    br{root_idx}: A={A:.3f}  c={c:.3f}  w={omega:.3f}  res={res_norm:.2e}"
                      + ("  [centrada, skip]" if is_centered else ""))

                # ── Auditoría de signo ─────────────────────────────────────
                Wq = get_Wq(omega, q, pmat, qvec, rvec)
                audit = harmonic_residual_sign_audit(Wq, N1)
                audit_rows.append({
                    "m1": m1, "m0": m0, "branch": root_idx, "A": A, "c": c,
                    "omega": omega, "N1": N1, "psi0": psi0,
                    "Re_Wq": Wq.real, "Im_Wq": Wq.imag,
                    **{k: v for k, v in audit.items()},
                    "sign_consistent": audit["R_plus_abs"] < 1e-3,
                })

                # ── Reconstrucción de semilla ──────────────────────────────
                seed_info = build_biased_seed(params, q, A, c, omega, psi0, N1)
                X_seed, x_bar, Re_X1 = seed_info["seed"], seed_info["x_bar"], seed_info["Re_X1"]

                dc_err   = abs(float(rvec @ x_bar) - c)
                harm_re  = float(rvec @ Re_X1)
                harm_im  = float(rvec @ seed_info["Im_X1"])
                harm_amp = np.sqrt(harm_re**2 + harm_im**2)
                amp_err  = abs(harm_amp - A)
                R_plus   = float(np.linalg.norm(biased_saturation_residual(A, c, omega, params, q)))

                if dc_err >= 1e-5:
                    seed_status = "dc_mismatch"
                elif amp_err >= 1e-4:
                    seed_status = "harmonic_amplitude_mismatch"
                elif R_plus >= 1e-4:
                    seed_status = "residual_too_large"
                else:
                    seed_status = "ok"

                seed_rows.append({
                    "m1": m1, "m0": m0, "branch": root_idx,
                    "dc_err": dc_err, "amp_err": amp_err, "R_plus": R_plus,
                    "seed_status": seed_status,
                })
                roots_rows.append({
                    "m1": m1, "m0": m0, "q": q, "branch": root_idx,
                    "A": A, "c": c, "omega": omega, "N1": N1, "psi0": psi0,
                    "residual_norm": res_norm,
                    "X_seed_x": X_seed[0], "X_seed_y": X_seed[1], "X_seed_z": X_seed[2],
                    "prefix": tag,
                })

                # ── Verificación de identidad homotópica ──────────────────
                max_id_err = 0.0
                for _ in range(5):
                    X_rand = np.random.uniform(-5, 5, 3)
                    # f_{η=1}(x) = f_original(x) → error debe ser ~0
                    sigma  = float(rvec @ X_rand)
                    psi_v  = (params.m0 - params.m1) * np.clip(sigma, -1, 1)
                    f_orig = pmat @ X_rand + qvec * psi_v
                    P_aff  = pmat + N1 * np.outer(qvec, rvec)
                    c_aff  = qvec * (psi0 - N1 * c)
                    f_eta1 = P_aff @ X_rand + c_aff + 1.0 * qvec * (psi_v - psi0 - N1 * (sigma - c))
                    max_id_err = max(max_id_err, float(np.linalg.norm(f_eta1 - f_orig)))
                identity_rows.append({
                    "m1": m1, "m0": m0, "branch": root_idx,
                    "max_identity_error": max_id_err,
                    "status": "ok" if max_id_err < 1e-10 else "failed",
                })

                # ── Continuación afín ─────────────────────────────────────
                print(f"      Continuacion afin ...")
                cont_steps = run_affine_continuation(
                    params, q, h, X_seed, A, c, psi0, N1, lambda_vals,
                    t_cont_trans, t_cont_keep, div_thr,
                )
                survived = (len(cont_steps) == len(lambda_vals)
                            and all(s["status"] == "ok" for s in cont_steps))
                print(f"      Continuacion: {'OK' if survived else 'FALLO'}")

                cont_rows.append({
                    "m1": m1, "m0": m0, "branch": root_idx,
                    "status": "ok" if survived else "failed",
                    "final_norm": cont_steps[-1]["x_out_norm"] if cont_steps else float("nan"),
                })

                if not survived:
                    classif_rows.append({
                        "m1": m1, "m0": m0, "branch": root_idx,
                        "classification": "continuation_failed", "prefix": tag,
                    })
                    all_results.append({**roots_rows[-1], "cont_status": "failed",
                                        "verdict": "continuation_failed"})
                    continue

                # ── Plots de continuación ─────────────────────────────────
                if plot_cfg["save_figures"] and plot_cfg.get("continuation_norm", True):
                    plot_continuation_metrics(
                        cont_steps, tag,
                        out_s2 / "plots" / "continuation" / f"{tag}_continuation.png",
                    )

                x_final_cont = cont_steps[-1]["x_out"].copy()

                # ── Simulación final ───────────────────────────────────────
                system = get_system("chua-nonsmooth")
                system.parameters.update({"m1": m1, "m0": m0, "alpha": alpha, "beta": beta, "gamma": gamma})
                print(f"      Simulacion final ...")
                sim_t, sim_x, sim_status, _ = fractional_integrate(
                    rhs=lambda t, x: system.rhs(x, system.parameters),
                    x0=x_final_cont, q=q, h=h, t_final=t_sim_final,
                    method="abm", memory_mode=memory_mode,
                    system=system, use_c_backend=True,
                )

                full_traj = np.column_stack((sim_t, sim_x))
                post_traj = full_traj[sim_t >= t_sim_trans]

                # ── Clasificación ─────────────────────────────────────────
                if len(post_traj) > 10:
                    max_range = float(np.max(np.ptp(post_traj[:, 1:4], axis=0)))
                    if max_range < 0.05:
                        verdict = "collapsed_to_equilibrium"
                    else:
                        diag = classify_post_transient_periodicity(post_traj, h=h)
                        verdict = diag["candidate_label"]
                else:
                    verdict = "too_short"

                print(f"      Veredicto: {verdict}")

                # ── Guardar trayectoria ───────────────────────────────────
                traj_path = out_s2 / "trajectories" / f"{tag}_trajectory.csv"
                pd.DataFrame(post_traj, columns=["t", "x", "y", "z"]).to_csv(traj_path, index=False)

                # ── Figura del reporte ────────────────────────────────────
                if plot_cfg["save_figures"]:
                    params_str = f"m1={m1} | m0={m0} | A={A:.3f} | c={c:.3f} | ω={omega:.3f}"
                    plot_attractor_report(
                        post_traj, tag, params_str, verdict,
                        out_s2 / "plots" / "attractors" / f"{tag}_report.png",
                    )

                classif_rows.append({
                    "m1": m1, "m0": m0, "branch": root_idx,
                    "classification": verdict, "prefix": tag,
                    "centroid_x": float(np.mean(post_traj[:, 1])) if len(post_traj) > 0 else float("nan"),
                    "centroid_y": float(np.mean(post_traj[:, 2])) if len(post_traj) > 0 else float("nan"),
                    "centroid_z": float(np.mean(post_traj[:, 3])) if len(post_traj) > 0 else float("nan"),
                })
                all_results.append({**roots_rows[-1], "cont_status": "ok", "verdict": verdict})

    # ── Guardar CSVs ──────────────────────────────────────────────────────────
    pd.DataFrame(roots_rows).to_csv(out_s2 / "roots_corrected.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out_s2 / "roots_sign_audit.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out_s2 / "seed_consistency_checks.csv", index=False)
    pd.DataFrame(identity_rows).to_csv(out_s2 / "affine_homotopy_identity.csv", index=False)
    pd.DataFrame(cont_rows).to_csv(out_s2 / "affine_continuation_summary.csv", index=False)
    pd.DataFrame(classif_rows).to_csv(out_s2 / "final_classification.csv", index=False)

    # ── Figura de auditoría de signo ──────────────────────────────────────────
    if plot_cfg["save_figures"] and audit_rows:
        plot_sign_audit(audit_rows, out_s2 / "plots" / "sign_audit" / "sign_audit_comparison.png")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print("\n[PASO 2 COMPLETADO]")
    survived = [r for r in classif_rows if "failed" not in r["classification"]]
    print(f"  Raices procesadas       : {len(roots_rows)}")
    print(f"  Continuaciones OK       : {len(cont_rows)}")
    print(f"  Candidatos supervivientes: {len(survived)}")
    for r in survived:
        print(f"    m1={r['m1']} m0={r['m0']} br={r['branch']}  c~{r.get('centroid_x', '?'):.3f}"
              f"  -> {r['classification']}")

    return all_results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    run_biased_df_search(cfg)
