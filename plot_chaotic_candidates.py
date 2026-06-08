"""
plot_chaotic_candidates.py
==========================
Genera figuras de alta calidad para todos los candidatos caóticos encontrados
en la exploración del sistema de Chua fraccionario:
  - Chua No Suave (saturación) – modo memoria completa y truncada
  - Chua Arctan fraccionario – modo memoria completa (cuando esté disponible)

Cada figura incluye:
  1. Espacio de fase 3D (x-y-z)
  2. Proyecciones 2D (x-y, x-z, y-z)
  3. Series de tiempo (x, y, z)
  4. Parámetros del candidato en el título

Las figuras se guardan en outputs/chaotic_candidates_plots/
"""

import sys
import csv
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

ROOT = Path(r"c:\Users\moren\Desktop\Codes\Hidden Attractors Fractional Order")
OUT_DIR = ROOT / "outputs" / "chaotic_candidates_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Color palette ────────────────────────────────────────────────────────────
COLORS = {
    "full":   "#E64B35",   # rojo
    "window": "#4DBBD5",   # azul-cyan
    "arctan_full":   "#00A087",  # verde
    "arctan_window": "#F39B7F",  # naranja
}
BG_COLOR = "#0F0F1A"
PANEL_COLOR = "#1A1A2E"
TICK_COLOR = "#CCCCDD"
LABEL_COLOR = "#EEEEFF"

def style_ax(ax, grid=True):
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TICK_COLOR, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.xaxis.label.set_color(LABEL_COLOR)
    ax.yaxis.label.set_color(LABEL_COLOR)
    if grid:
        ax.grid(True, color="#222244", lw=0.4, linestyle="--", alpha=0.6)

def style_ax3d(ax):
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TICK_COLOR, labelsize=6.5)
    ax.xaxis.label.set_color(LABEL_COLOR)
    ax.yaxis.label.set_color(LABEL_COLOR)
    ax.zaxis.label.set_color(LABEL_COLOR)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#222244")
    ax.yaxis.pane.set_edgecolor("#222244")
    ax.zaxis.pane.set_edgecolor("#222244")
    ax.grid(True, color="#222244", lw=0.3, linestyle="--", alpha=0.5)


def make_figure(traj_path: Path, case_id: str, model_label: str, memory_label: str,
                params_str: str, color: str, t_transient: float = 100.0, h: float = 0.01):
    """
    Crea figura de 4 paneles:
      - Izquierda arriba: espacio de fase 3D
      - Derecha arriba: proyecciones 2D (3 subpaneles)
      - Abajo: series de tiempo (x, y, z)
    """
    data = np.loadtxt(traj_path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 4:
        print(f"  [WARN] Formato inesperado en {traj_path.name}, se omite.")
        return None

    t = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]

    n_burn = int(t_transient / h)
    t_  = t[n_burn:]
    x_  = x[n_burn:]
    y_  = y[n_burn:]
    z_  = z[n_burn:]

    # Subsample para no saturar el plot (máx 15 000 puntos)
    max_pts = 15_000
    stride = max(1, len(t_) // max_pts)
    t_ = t_[::stride]
    x_ = x_[::stride]
    y_ = y_[::stride]
    z_ = z_[::stride]

    fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)
    fig.suptitle(
        f"[{model_label} | {memory_label}]   {params_str}",
        color=LABEL_COLOR, fontsize=11, fontweight="bold", y=0.98
    )

    gs = GridSpec(
        3, 4,
        figure=fig,
        left=0.06, right=0.97, top=0.93, bottom=0.07,
        hspace=0.45, wspace=0.35
    )

    # ── Panel 1: 3D Phase Space ────────────────────────────────────────────
    ax3d = fig.add_subplot(gs[:2, :2], projection="3d")
    pts = np.c_[x_, y_, z_]
    n_seg = len(pts) - 1
    cmap = plt.cm.plasma
    for i in range(n_seg):
        frac = i / n_seg
        seg_color = cmap(frac)
        ax3d.plot(pts[i:i+2, 0], pts[i:i+2, 1], pts[i:i+2, 2],
                  lw=0.4, color=seg_color, alpha=0.7)
    style_ax3d(ax3d)
    ax3d.set_xlabel("x", labelpad=3)
    ax3d.set_ylabel("y", labelpad=3)
    ax3d.set_zlabel("z", labelpad=3)
    ax3d.set_title("Espacio de fase 3D", color=LABEL_COLOR, fontsize=9, pad=4)

    # ── Paneles 2D ────────────────────────────────────────────────────────
    proj_specs = [
        ("x", "y", x_, y_),
        ("x", "z", x_, z_),
        ("y", "z", y_, z_),
    ]
    for col_idx, (xl, yl, xd, yd) in enumerate(proj_specs):
        ax2 = fig.add_subplot(gs[0, 2 + col_idx % 2] if col_idx < 2 else gs[1, 2])
        # Adjust layout for 3 2D plots
        row = 0 if col_idx == 0 else (0 if col_idx == 1 else 1)
        col = 2 if col_idx == 0 else 3
        if col_idx == 2:
            col = 2
            row = 1
        ax2 = fig.add_subplot(gs[row, col])
        ax2.plot(xd, yd, lw=0.35, color=color, alpha=0.75)
        style_ax(ax2)
        ax2.set_xlabel(xl, fontsize=8)
        ax2.set_ylabel(yl, fontsize=8)
        ax2.set_title(f"Proyección {xl}-{yl}", color=LABEL_COLOR, fontsize=8)

    # Extra 2D: yz in gs[1,3]
    ax_yz = fig.add_subplot(gs[1, 3])
    ax_yz.plot(y_, z_, lw=0.35, color=color, alpha=0.75)
    style_ax(ax_yz)
    ax_yz.set_xlabel("y", fontsize=8)
    ax_yz.set_ylabel("z", fontsize=8)
    ax_yz.set_title("Proyección y-z", color=LABEL_COLOR, fontsize=8)

    # ── Series de tiempo ─────────────────────────────────────────────────
    ts_colors = ["#E64B35", "#4DBBD5", "#00A087"]
    for row_i, (sig, lbl, clr) in enumerate(zip([x_, y_, z_], ["x(t)", "y(t)", "z(t)"], ts_colors)):
        ax_t = fig.add_subplot(gs[2, row_i])
        ax_t.plot(t_, sig, lw=0.4, color=clr)
        style_ax(ax_t)
        ax_t.set_xlabel("t", fontsize=8)
        ax_t.set_ylabel(lbl, fontsize=8)
        ax_t.set_title(f"Serie {lbl}", color=LABEL_COLOR, fontsize=8)

    # Waterfall info panel
    ax_info = fig.add_subplot(gs[2, 3])
    ax_info.set_facecolor(PANEL_COLOR)
    ax_info.axis("off")
    info_lines = [
        f"Modelo:  {model_label}",
        f"Memoria: {memory_label}",
        "",
        f"Candidato: {case_id}",
        "",
        *[f"  {p}" for p in params_str.split("|")],
        "",
        f"Transiente descartado: {t_transient}s",
        f"Puntos post-trans.: {len(t_):,}",
        f"Subsampleo: 1/{stride}",
    ]
    ax_info.text(
        0.05, 0.95, "\n".join(info_lines),
        transform=ax_info.transAxes,
        va="top", ha="left",
        color=LABEL_COLOR, fontsize=7.5,
        fontfamily="monospace",
        linespacing=1.5
    )
    ax_info.set_title("Info", color=LABEL_COLOR, fontsize=8)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Candidatos catalogados
# ══════════════════════════════════════════════════════════════════════════════

CANDIDATES = [
    # ── Chua No Suave (Saturación) ─────────────────────────────────────────
    # Único candidato chaotic_candidate_pending_robustness en ambos modos de memoria
    {
        "case_id":      "m1_m1p2000_m0_m0p2000_branch_0",
        "model":        "chua_saturation",
        "model_label":  "Chua No Suave (Saturación)",
        "memory":       "full",
        "memory_label": "Memoria Completa (q=0.9998)",
        "color":        COLORS["full"],
        "traj_path":    ROOT / "outputs" / "saturation_search_seed1_mem_full_sweep" / "m1_m1p2000_m0_m0p2000_branch_0_trajectory.csv",
        "params_str":   "m1=-1.2 | m0=-0.2 | ω₀=2.039 | k=0.263 | A₀=4.80",
    },
    {
        "case_id":      "m1_m1p2000_m0_m0p2000_branch_0_window",
        "model":        "chua_saturation",
        "model_label":  "Chua No Suave (Saturación)",
        "memory":       "window",
        "memory_label": "Memoria Truncada Lm=10s (q=0.9998)",
        "color":        COLORS["window"],
        "traj_path":    ROOT / "outputs" / "saturation_search_seed0p9998_mem_window_sweep" / "m1_m1p2000_m0_m0p2000_branch_0_trajectory.csv",
        "params_str":   "m1=-1.2 | m0=-0.2 | ω₀=2.040 | k=0.263 | A₀=4.80",
    },
    # Candidatos nonperiodic del barrido q=1.0 (seed entero)
    {
        "case_id":      "m1_m0p8000_m0_m0p1000_branch_0_full",
        "model":        "chua_saturation",
        "model_label":  "Chua No Suave (Saturación)",
        "memory":       "full",
        "memory_label": "Memoria Completa (q=0.9998)",
        "color":        COLORS["full"],
        "traj_path":    ROOT / "outputs" / "saturation_search_seed1_mem_full_sweep" / "m1_m0p8000_m0_m0p1000_branch_0_trajectory.csv",
        "params_str":   "m1=-0.8 | m0=-0.1 | ω₀=3.245 | k=0.613 | A₀=1.29",
    },
    {
        "case_id":      "m1_m1p0000_m0_m0p1000_branch_1_full",
        "model":        "chua_saturation",
        "model_label":  "Chua No Suave (Saturación)",
        "memory":       "full",
        "memory_label": "Memoria Completa (q=0.9998)",
        "color":        COLORS["full"],
        "traj_path":    ROOT / "outputs" / "saturation_search_seed1_mem_full_sweep" / "m1_m1p0000_m0_m0p1000_branch_1_trajectory.csv",
        "params_str":   "m1=-1.0 | m0=-0.1 | ω₀=3.245 | k=0.813 | A₀=1.24",
    },
    {
        "case_id":      "m1_m1p2000_m0_m0p1000_branch_1_full",
        "model":        "chua_saturation",
        "model_label":  "Chua No Suave (Saturación)",
        "memory":       "full",
        "memory_label": "Memoria Completa (q=0.9998)",
        "color":        COLORS["full"],
        "traj_path":    ROOT / "outputs" / "saturation_search_seed1_mem_full_sweep" / "m1_m1p2000_m0_m0p1000_branch_1_trajectory.csv",
        "params_str":   "m1=-1.2 | m0=-0.1 | ω₀=3.245 | k=1.013 | A₀=1.20",
    },
    {
        "case_id":      "m1_m1p4000_m0_m0p1000_branch_1_full",
        "model":        "chua_saturation",
        "model_label":  "Chua No Suave (Saturación)",
        "memory":       "full",
        "memory_label": "Memoria Completa (q=0.9998)",
        "color":        COLORS["full"],
        "traj_path":    ROOT / "outputs" / "saturation_search_seed1_mem_full_sweep" / "m1_m1p4000_m0_m0p1000_branch_1_trajectory.csv",
        "params_str":   "m1=-1.4 | m0=-0.1 | ω₀=3.245 | k=1.213 | A₀=1.17",
    },
    {
        "case_id":      "m1_m1p6000_m0_m0p1000_branch_1_full",
        "model":        "chua_saturation",
        "model_label":  "Chua No Suave (Saturación)",
        "memory":       "full",
        "memory_label": "Memoria Completa (q=0.9998)",
        "color":        COLORS["full"],
        "traj_path":    ROOT / "outputs" / "saturation_search_seed1_mem_full_sweep" / "m1_m1p6000_m0_m0p1000_branch_1_trajectory.csv",
        "params_str":   "m1=-1.6 | m0=-0.1 | ω₀=3.245 | k=1.413 | A₀=1.16",
    },
]

# ── Candidatos Arctan (se agregan si el archivo existe) ─────────────────────
ARCTAN_CANDIDATES_SPEC = [
    # Basados en la exploración anterior (version_2) con ABM/ADM entero
    # Se agregarán dinámicamente si los archivos de trayectoria existen
    {
        "case_id":      "a1_0p10_a2_m1p2000_rho_1p00_branch_0",
        "model":        "chua_arctan",
        "model_label":  "Chua Arctan Fraccionario",
        "memory":       "full",
        "memory_label": "Memoria Completa (q=0.99)",
        "color":        COLORS["arctan_full"],
        "traj_path":    ROOT / "outputs" / "arctan_search_seed0p99_mem_full" / "a1_0p10_a2_m1p2000_rho_1p00_branch_0_trajectory.csv",
        "params_str":   "a1=0.1 | a2=-1.2 | rho=1.0 | α=8.4562 | β=12.0732",
    },
    {
        "case_id":      "a1_0p10_a2_m1p5585_rho_1p25_branch_0",
        "model":        "chua_arctan",
        "model_label":  "Chua Arctan Fraccionario",
        "memory":       "full",
        "memory_label": "Memoria Completa (q=0.99)",
        "color":        COLORS["arctan_full"],
        "traj_path":    ROOT / "outputs" / "arctan_search_seed0p99_mem_full" / "a1_0p10_a2_m1p5585_rho_1p25_branch_0_trajectory.csv",
        "params_str":   "a1=0.1 | a2=-1.5585 | rho=1.25 | α=8.4562 | β=12.0732",
    },
]

# Agregar candidatos arctan que existen
for spec in ARCTAN_CANDIDATES_SPEC:
    if spec["traj_path"].exists():
        CANDIDATES.append(spec)

# ══════════════════════════════════════════════════════════════════════════════
# Generar figuras individuales
# ══════════════════════════════════════════════════════════════════════════════

generated = []
skipped = []

for cand in CANDIDATES:
    traj = cand["traj_path"]
    if not traj.exists():
        print(f"[SKIP] {cand['case_id']} - trayectoria no encontrada: {traj.name}")
        skipped.append(cand["case_id"])
        continue

    print(f"[PLOT] {cand['case_id']} ({cand['memory_label']})...")
    fig = make_figure(
        traj_path=traj,
        case_id=cand["case_id"],
        model_label=cand["model_label"],
        memory_label=cand["memory_label"],
        params_str=cand["params_str"],
        color=cand["color"],
        t_transient=100.0,
        h=0.01,
    )
    if fig is None:
        skipped.append(cand["case_id"])
        continue

    out_file = OUT_DIR / f"{cand['case_id']}_detailed.png"
    fig.savefig(out_file, dpi=160, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Guardado: {out_file.name}")
    generated.append((cand["case_id"], cand["model_label"], cand["memory_label"], out_file))

# ══════════════════════════════════════════════════════════════════════════════
# Figura resumen: panel con miniaturas de TODOS los candidatos
# ══════════════════════════════════════════════════════════════════════════════

print("\n[SUMMARY] Generando figura resumen de todos los candidatos...")

# Leer y plotear espacio de fase 3D de todos los candidatos en un solo lienzo
valid_cands = [c for c in CANDIDATES if c["traj_path"].exists()]
n_cands = len(valid_cands)

if n_cands > 0:
    ncols = min(4, n_cands)
    nrows = (n_cands + ncols - 1) // ncols

    fig_sum = plt.figure(figsize=(ncols * 4, nrows * 3.5), facecolor=BG_COLOR)
    fig_sum.suptitle(
        "Resumen: Candidatos Caóticos – Chua Fraccionario No Suave",
        color=LABEL_COLOR, fontsize=13, fontweight="bold", y=0.98
    )

    for idx, cand in enumerate(valid_cands):
        data = np.loadtxt(cand["traj_path"], delimiter=",", skiprows=1)
        t = data[:, 0]
        x = data[:, 1]
        y = data[:, 2]
        z = data[:, 3]
        n_burn = int(100.0 / 0.01)
        stride = max(1, (len(t) - n_burn) // 8000)
        x_ = x[n_burn::stride]
        y_ = y[n_burn::stride]
        z_ = z[n_burn::stride]

        ax = fig_sum.add_subplot(nrows, ncols, idx + 1, projection="3d")
        pts = np.c_[x_, y_, z_]
        cmap = plt.cm.plasma
        for i in range(len(pts) - 1):
            frac = i / len(pts)
            ax.plot(pts[i:i+2, 0], pts[i:i+2, 1], pts[i:i+2, 2],
                     lw=0.35, color=cmap(frac), alpha=0.8)

        style_ax3d(ax)
        ax.set_xlabel("x", fontsize=6, labelpad=1)
        ax.set_ylabel("y", fontsize=6, labelpad=1)
        ax.set_zlabel("z", fontsize=6, labelpad=1)
        short_id = cand["case_id"].replace("_branch_", "\nb")
        model_short = "Sat" if "saturation" in cand["model"] else "Arctan"
        mem_short = "Full" if cand["memory"] == "full" else f"Win{10}"
        ax.set_title(
            f"[{model_short}|{mem_short}]\n{short_id}",
            color=LABEL_COLOR, fontsize=6.5, pad=2
        )

    fig_sum.tight_layout(rect=[0, 0, 1, 0.96])
    summary_file = OUT_DIR / "all_candidates_summary_phase3d.png"
    fig_sum.savefig(summary_file, dpi=160, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig_sum)
    print(f"  -> Figura resumen guardada: {summary_file.name}")

# ══════════════════════════════════════════════════════════════════════════════
# Reporte en terminal
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("REPORTE FINAL - plot_chaotic_candidates.py")
print("=" * 70)
print(f"\n  Candidatos procesados: {len(generated)}")
print(f"  Candidatos omitidos  : {len(skipped)}")
print(f"\n  Figuras generadas en: {OUT_DIR}")
print()
for case_id, model, mem, path in generated:
    print(f"  [OK]  {case_id}")
    print(f"        [{model} | {mem}]")
    print(f"        -> {path.name}")
    print()
