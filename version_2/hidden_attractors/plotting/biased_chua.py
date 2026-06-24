# -*- coding: utf-8 -*-
"""Plotting functions for the biased Chua fractional-order hidden attractor example.

Enforces:
  - White background
  - No internal titles (plt.title, set_title, suptitle are prohibited)
  - Axis labels present
  - Routes exports through export_figure / intercept_and_export_path
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from .style import apply_library_style, apply_axes_style, get_figsize
from .export import intercept_and_export_path

# Apply style globally
apply_library_style()

def plot_centered_trajectory(traj: np.ndarray, outpath: Path, t_burn: float, h: float) -> None:
    """4-panel figure: 3D phase space + 2D projections + time series."""
    n_burn = int(t_burn / h)
    states = traj[n_burn:, 1:4]
    times  = traj[n_burn:, 0]
    n = len(states)
    if n < 10:
        return

    fig = plt.figure(figsize=(16, 9))

    # 3D
    ax3 = fig.add_subplot(2, 3, (1, 4), projection="3d")
    ax3.plot(states[:, 0], states[:, 1], states[:, 2], lw=0.4, color='#10b981', alpha=0.85)
    apply_axes_style(ax3, is_3d=True)
    ax3.set_xlabel("x", labelpad=2)
    ax3.set_ylabel("y", labelpad=2)
    ax3.set_zlabel("z", labelpad=2)

    # Projections
    pairs = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]
    for sub_i, (xl, yl, ix, iy) in enumerate(pairs):
        ax = fig.add_subplot(2, 3, sub_i + 2 + (1 if sub_i >= 1 else 0))
        ax.plot(states[:, ix], states[:, iy], lw=0.3, color="#E64B35", alpha=0.8)
        apply_axes_style(ax, grid=True)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

    # Time series (x, y, z)
    for sub_i, (lbl, sig, clr) in enumerate(
        [("x(t)", states[:, 0], "#E64B35"),
         ("y(t)", states[:, 1], "#4DBBD5"),
         ("z(t)", states[:, 2], "#00A087")]
    ):
        ax = fig.add_subplot(2, 3, sub_i + 4 if sub_i < 2 else 6)
        ax.plot(times, sig, lw=0.35, color=clr)
        apply_axes_style(ax, grid=True)
        ax.set_xlabel("t [s]")
        ax.set_ylabel(lbl)

    fig.tight_layout()
    intercept_and_export_path(fig, outpath, 'attractor')
    plt.close(fig)

def plot_sign_audit(first_arg: Union[List[Dict[str, Any]], np.ndarray],
                    second_arg: Optional[Union[Path, np.ndarray]] = None,
                    third_arg: Optional[np.ndarray] = None,
                    fourth_arg: Optional[Path] = None) -> None:
    """Plots sign audit. Supports both audit_rows (Step 2) and c_grid line plots (legacy)."""
    if isinstance(first_arg, list):
        # Step 2 style (bar chart)
        audit_rows = first_arg
        outpath = Path(second_arg)
        fig, ax = plt.subplots(figsize=(10, 5))
        idx = np.arange(len(audit_rows))
        w   = 0.35
        ab_plus  = [r["R_plus_abs"] for r in audit_rows]
        ab_minus = [r["R_minus_abs"] for r in audit_rows]
        lbls     = [f"m1={r['m1']:.3f}\nm0={r['m0']:.3f}\nc={r['c']:.2f}" for r in audit_rows]
        ax.bar(idx - w/2, ab_plus,  w, label="|1 + Wq·N₁| (convención)", color="#4DBBD5")
        ax.bar(idx + w/2, ab_minus, w, label="|1 − Wq·N₁|",            color="#E64B35", alpha=0.7)
        ax.set_xticks(idx)
        ax.set_xticklabels(lbls, fontsize=6)
        ax.set_ylabel("Residuo armónico absoluto")
        ax.legend(fontsize=7)
        apply_axes_style(ax, grid=True)
        fig.tight_layout()
        intercept_and_export_path(fig, outpath, 'transfer')
        plt.close(fig)
    else:
        # Legacy style (line plot)
        c_grid = first_arg
        R_plus = second_arg
        R_minus = third_arg
        outpath = Path(fourth_arg)
        fig, ax = plt.subplots(figsize=get_figsize("2d"))
        ax.plot(c_grid, R_plus, label="1 + Wq(jω) N₁", lw=1.2, color="#E64B35")
        ax.plot(c_grid, R_minus, label="1 - Wq(jω) N₁", lw=1.2, linestyle="--", color="#4DBBD5")
        apply_axes_style(ax, grid=True)
        ax.set_xlabel("c (bias DC)")
        ax.set_ylabel("Residuo Armónico")
        ax.legend(loc="best")
        fig.tight_layout()
        intercept_and_export_path(fig, outpath, 'transfer')
        plt.close(fig)

def plot_attractor_report(states: np.ndarray, info: Union[List[str], str], outpath: Path) -> None:
    """Legacy or 2x2 format chaos attractor report."""
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 3D
    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_3d.plot(states[:, 0], states[:, 1], states[:, 2], lw=0.4, color="#E64B35")
    apply_axes_style(ax_3d, is_3d=True)
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")

    # Projection x-y
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xy.plot(states[:, 0], states[:, 1], lw=0.3, color="#4DBBD5")
    apply_axes_style(ax_xy, grid=True)
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")

    # Time series
    ax_t = fig.add_subplot(gs[1, 0])
    ax_t.plot(states[:, 0], lw=0.4, color="#00A087")
    apply_axes_style(ax_t, grid=True)
    ax_t.set_xlabel("Puntos")
    ax_t.set_ylabel("x(t)")

    # Info text
    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.axis("off")
    info_list = info if isinstance(info, list) else [info]
    ax_info.text(0.05, 0.95, "\n".join(info_list), transform=ax_info.transAxes,
                 va="top", ha="left", fontsize=9, fontfamily="monospace")

    fig.tight_layout()
    intercept_and_export_path(fig, outpath, 'attractor')
    plt.close(fig)

def plot_continuation_metrics(history: Union[Dict[str, List[float]], List[Dict[str, Any]]],
                              outpath_or_prefix: Union[Path, str],
                              outpath: Optional[Path] = None) -> None:
    """Plots continuation metrics (A_obs/A_theo, norm/coords vs eta)."""
    if isinstance(history, list):
        # Step 2 style
        cont_steps = history
        real_outpath = Path(outpath if outpath else outpath_or_prefix)
        etas  = [s["lambda_value"] for s in cont_steps]
        norms = [s["x_out_norm"] for s in cont_steps]
        coords = np.array([s["x_out"] for s in cont_steps])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(etas, norms, "o-", color="#4DBBD5", lw=2)
        apply_axes_style(ax1, grid=True)
        ax1.set_xlabel("η (lambda)", fontsize=9)
        ax1.set_ylabel("||x_final||", fontsize=9)

        for ci, (lbl, clr) in enumerate(zip(["x", "y", "z"], ["#E64B35", "#4DBBD5", "#00A087"])):
            ax2.plot(etas, coords[:, ci], "o-", label=lbl, color=clr, lw=1.5)
        apply_axes_style(ax2, grid=True)
        ax2.set_xlabel("η (lambda)", fontsize=9)
        ax2.set_ylabel("Coordenada", fontsize=9)
        ax2.legend(fontsize=8)

        fig.tight_layout()
        intercept_and_export_path(fig, real_outpath, 'continuation')
        plt.close(fig)
    else:
        # Legacy/Dict style
        hist = history
        real_outpath = Path(outpath_or_prefix)
        fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
        
        axes[0].plot(hist["eta"], hist["A_obs"], label="A observada", color="#E64B35")
        axes[0].plot(hist["eta"], hist["A_theo"], label="A teórica", linestyle="--", color="#4DBBD5")
        apply_axes_style(axes[0], grid=True)
        axes[0].set_ylabel("Amplitud")
        axes[0].legend(loc="best")
        
        axes[1].plot(hist["eta"], hist["omega_obs"], label="ω observada", color="#00A087")
        axes[1].plot(hist["eta"], hist["omega_theo"], label="ω teórica", linestyle="--", color="#ff7f0e")
        apply_axes_style(axes[1], grid=True)
        axes[1].set_ylabel("Frecuencia (rad/s)")
        axes[1].set_xlabel("η (Homotopía)")
        axes[1].legend(loc="best")

        fig.tight_layout()
        intercept_and_export_path(fig, real_outpath, 'continuation')
        plt.close(fig)

def _get_colour_map() -> Dict[str, str]:
    return {
        "target_attractor":   "#ef4444",
        "stable_equilibrium": "#3b82f6",
        "divergence":         "#f59e0b",
        "other_attractor":    "#8b5cf6",
        "numerical_failure":  "#94a3b8",
    }

def plot_sphere_summary(eq_name: str, eq_pt: np.ndarray, radius: float,
                         runs: List[Dict], pts: np.ndarray, outpath: Path) -> None:
    """3D figure: probe points colored by destination."""
    cmap = _get_colour_map()
    fig = plt.figure(figsize=get_figsize("3d"), dpi=300)
    ax  = fig.add_subplot(111, projection="3d")
    for res, pt in zip(runs, pts):
        ax.scatter(*pt, color=cmap.get(res["destination"], "#9ca3af"), s=8, alpha=0.6)
    ax.scatter(*eq_pt, color="black", marker="x", s=60, zorder=10)
    apply_axes_style(ax, is_3d=True)
    ax.set_xlabel("x", labelpad=2)
    ax.set_ylabel("y", labelpad=2)
    ax.set_zlabel("z", labelpad=2)
    
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=c, label=d, markersize=7)
               for d, c in cmap.items()]
    ax.legend(handles=handles, fontsize=6, loc="best")
    fig.tight_layout()
    intercept_and_export_path(fig, outpath, 'sphere_test')
    plt.close(fig)

def build_hiddenness_heatmap(
    records: List[Dict],
    radii: List[float],
) -> tuple[plt.Figure, plt.Axes]:
    """Build the canonical TARGET-contact heatmap used by every report."""
    radius_values = sorted({float(value) for value in radii})
    present = {str(row["equilibrium"]) for row in records}
    preferred_order = ["E0", "E+", "E-"]
    eq_names = [name for name in preferred_order if name in present]
    eq_names.extend(sorted(present - set(eq_names)))

    fractions = np.full((len(eq_names), len(radius_values)), np.nan, dtype=float)
    hits = np.zeros_like(fractions, dtype=int)
    totals = np.zeros_like(fractions, dtype=int)
    for row in records:
        eq_idx = eq_names.index(str(row["equilibrium"]))
        radius = float(row["radius"])
        r_idx = int(np.argmin(np.abs(np.asarray(radius_values) - radius)))
        total = int(row.get("samples", row.get("total", 0)))
        hit = int(row.get("TARGET", row.get("target_hits", 0)))
        totals[eq_idx, r_idx] += total
        hits[eq_idx, r_idx] += hit

    mask = totals > 0
    fractions[mask] = hits[mask] / totals[mask]
    plot_values = np.nan_to_num(fractions, nan=0.0)

    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=300)
    image = ax.imshow(plot_values, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=1.0)
    apply_axes_style(ax)
    ax.set_xticks(np.arange(len(radius_values)))
    ax.set_xticklabels([f"{radius:.0e}" for radius in radius_values])
    ax.set_yticks(np.arange(len(eq_names)))
    eq_labels = {"E0": r"$E_0$", "E+": r"$E_+$", "E-": r"$E_-$"}
    ax.set_yticklabels([eq_labels.get(name, name) for name in eq_names])
    ax.set_xlabel(r"Radio $r$")
    ax.set_ylabel("Equilibrio")

    for i in range(len(eq_names)):
        for j in range(len(radius_values)):
            label = f"{hits[i, j]}/{totals[i, j]}" if totals[i, j] else "--"
            color = "white" if plot_values[i, j] >= 0.55 else "#111827"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=8)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("Fracción TARGET")
    fig.tight_layout()
    return fig, ax


def plot_heatmap_hiddenness(records: List[Dict], radii: List[float], outpath: Path) -> None:
    """Export the canonical TARGET-contact heatmap."""
    fig, _ = build_hiddenness_heatmap(records, radii)
    intercept_and_export_path(fig, outpath, "heatmap")
    plt.close(fig)

# Aliases
plot_sphere_3d = plot_sphere_summary
plot_heatmap = plot_heatmap_hiddenness

def plot_candidate_report(traj: np.ndarray, params_str: str, verdict: str,
                           outpath: Path, h: float = 0.01) -> None:
    """7-panel report figure: 3D + 2D projections + time series + FFT + parameters."""
    from hidden_attractors.analysis.spectral import fft_spectrum

    states = traj[:, 1:4]
    times  = traj[:, 0]
    n      = len(states)
    if n < 10:
        return

    h_est = float(np.median(np.diff(times))) if len(times) > 1 else h

    fig = plt.figure(figsize=(17, 9.5), dpi=300)
    gs = GridSpec(3, 4, figure=fig, left=0.055, right=0.97, top=0.90,
                  bottom=0.07, hspace=0.55, wspace=0.38)

    # 3D
    ax3 = fig.add_subplot(gs[:2, :2], projection="3d")
    cmap = plt.cm.plasma
    for i in range(n - 1):
        ax3.plot(states[i:i+2, 0], states[i:i+2, 1], states[i:i+2, 2],
                 lw=0.4, color=cmap(i / n), alpha=0.85)
    apply_axes_style(ax3, is_3d=True)
    ax3.set_xlabel("x", labelpad=2)
    ax3.set_ylabel("y", labelpad=2)
    ax3.set_zlabel("z", labelpad=2)

    # Projections
    pairs = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]
    for sub_i, (xl, yl, ix, iy) in enumerate(pairs):
        row = sub_i // 2
        col = 2 + (sub_i % 2)
        ax2 = fig.add_subplot(gs[row, col])
        ax2.plot(states[:, ix], states[:, iy], lw=0.35, color="#ef4444", alpha=0.85)
        apply_axes_style(ax2, grid=True)
        ax2.set_xlabel(xl)
        ax2.set_ylabel(yl)

    # FFT
    ax_fft = fig.add_subplot(gs[2, 0])
    try:
        spec = fft_spectrum(states[:, 0], h_est)
        if spec.frequency_rad_s.size > 0:
            ax_fft.plot(spec.frequency_rad_s, spec.values, lw=0.5, color="#10b981")
            ax_fft.set_xlim(0, 10.0)
    except Exception:
        pass
    apply_axes_style(ax_fft, grid=True)
    ax_fft.set_xlabel("Frecuencia (rad/s)")
    ax_fft.set_ylabel("Espectro de Amplitud")


    # Time series (x, y, z)
    ax_t = fig.add_subplot(gs[2, 1:3])
    colors = ["#3b82f6", "#f59e0b", "#8b5cf6"]
    for i in range(3):
        ax_t.plot(times, states[:, i], lw=0.4, color=colors[i], label=f"x{i+1}(t)")
    apply_axes_style(ax_t, grid=True)
    ax_t.set_xlabel("t (s)")
    ax_t.set_ylabel("Estado")
    ax_t.legend(loc="upper right", fontsize=6)

    # Info panel
    ax_i = fig.add_subplot(gs[2, 3])
    ax_i.axis("off")
    info = [
        "Metricas Dinamicas",
        "------------------",
        f"Parametros: {params_str}",
        f"Clasificacion: {verdict}",
        f"Puntos: {n:,}",
        f"Paso efectivo h: {h_est:.4f}",
    ]
    ax_i.text(0.05, 0.95, "\n".join(info), transform=ax_i.transAxes,
              va="top", ha="left", fontsize=8, fontfamily="monospace")

    fig.tight_layout()
    intercept_and_export_path(fig, outpath, 'attractor')
    plt.close(fig)

def plot_biased_vs_centered(biased_traj: np.ndarray, centered_traj: Optional[np.ndarray],
                             params_str: str, outpath: Path) -> None:
    """3D comparison and FFT of the biased attractor vs centered reference."""
    fig = plt.figure(figsize=(16, 7), dpi=300)
    
    # 3D
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.plot(biased_traj[:, 1], biased_traj[:, 2], biased_traj[:, 3],
             lw=0.4, color="#ef4444", label="Sesgado (BDF)")
    if centered_traj is not None:
        ax3.plot(centered_traj[:, 1], centered_traj[:, 2], centered_traj[:, 3],
                 lw=0.4, color="#3b82f6", label="Centrado (Base)")
    apply_axes_style(ax3, is_3d=True)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.legend(loc="best")

    # FFT comparison
    from hidden_attractors.analysis.spectral import fft_spectrum
    ax_fft = fig.add_subplot(1, 2, 2)
    
    try:
        spec_b = fft_spectrum(biased_traj[:, 1], 0.01)
        if spec_b.frequency_rad_s.size > 0:
            ax_fft.plot(spec_b.frequency_rad_s, spec_b.values, lw=0.6, color="#ef4444", label="Sesgado (BDF)")
    except Exception:
        pass
    
    if centered_traj is not None:
        try:
            spec_c = fft_spectrum(centered_traj[:, 1], 0.01)
            if spec_c.frequency_rad_s.size > 0:
                ax_fft.plot(spec_c.frequency_rad_s, spec_c.values, lw=0.6, color="#3b82f6", label="Centrado (Base)")
        except Exception:
            pass
        
    apply_axes_style(ax_fft, grid=True)
    ax_fft.set_xlim(0, 10.0)
    ax_fft.set_xlabel("Frecuencia (rad/s)")
    ax_fft.set_ylabel("Espectro de Amplitud")
    ax_fft.legend(loc="best")

    fig.tight_layout()
    intercept_and_export_path(fig, outpath, 'attractor')
    plt.close(fig)

def plot_mega_summary(candidates: List[Dict], outpath: Path) -> None:
    """3D mosaic of all candidates that survived."""
    valid = [(c, c["traj"]) for c in candidates if c["traj"] is not None]
    if not valid:
        return

    n_all  = len(valid)
    ncols  = min(3, n_all)
    nrows  = (n_all + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 5, nrows * 4.5 + 1), dpi=300)
    cmap = plt.cm.plasma

    for idx, (cand, traj) in enumerate(valid):
        states = traj[:, 1:4]
        n = len(states)
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        for i in range(n - 1):
            ax.plot(states[i:i+2, 0], states[i:i+2, 1], states[i:i+2, 2],
                    lw=0.3, color=cmap(i / n), alpha=0.8)
        apply_axes_style(ax, is_3d=True)
        ax.set_xlabel("x", labelpad=0)
        ax.set_ylabel("y", labelpad=0)
        ax.set_zlabel("z", labelpad=0)

    fig.tight_layout()
    intercept_and_export_path(fig, outpath, 'attractor')
    plt.close(fig)
