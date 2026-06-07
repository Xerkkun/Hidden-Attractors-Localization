"""
generate_all_plots_and_summary.py
==================================
Genera TODAS las figuras de candidatos caoticos encontrados en la exploracion
del circuito de Chua fraccionario No Suave (saturacion) y Arctan.

Para cada candidato produce:
  - Figura individual 4 paneles (3D, proyecciones 2D, series de tiempo)
  - Una figura resumen colectiva con espacios de fase 3D de todos los candidatos

Uso:
    .venv\\Scripts\\python generate_all_plots_and_summary.py
"""
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "chaotic_candidates_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Estilos ──────────────────────────────────────────────────────────────────
BG    = "#0D0D1A"
PANEL = "#141428"
TEXT  = "#E8E8FF"
TICK  = "#9999BB"
GRID  = "#1E1E3A"

CMAP_TRAJ = plt.cm.plasma

def ax_style(ax, grid=True):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TICK, labelsize=7)
    for s in ax.spines.values():
        s.set_edgecolor("#2A2A50")
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    if grid:
        ax.grid(True, color=GRID, lw=0.4, ls="--", alpha=0.7)

def ax3d_style(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TICK, labelsize=6)
    for attr in ["xaxis","yaxis","zaxis"]:
        getattr(ax, attr).label.set_color(TEXT)
        getattr(ax, attr).pane.fill = False
        getattr(ax, attr).pane.set_edgecolor("#1E1E3A")
    ax.grid(True, color=GRID, lw=0.3, ls="--", alpha=0.5)


def load_traj(path, t_transient=100.0, h=0.01, max_pts=12000):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    assert data.ndim == 2 and data.shape[1] >= 4, f"Formato inesperado: {path}"
    t, x, y, z = data[:,0], data[:,1], data[:,2], data[:,3]
    n_burn = int(t_transient / h)
    t, x, y, z = t[n_burn:], x[n_burn:], y[n_burn:], z[n_burn:]
    stride = max(1, len(t) // max_pts)
    return t[::stride], x[::stride], y[::stride], z[::stride], stride


def plot_candidate(traj_path, case_id, model_label, memory_label,
                   params_str, color, t_transient=100.0, h=0.01):
    t, x, y, z, stride = load_traj(traj_path, t_transient, h)
    n = len(t)

    fig = plt.figure(figsize=(17, 9.5), facecolor=BG)
    fig.suptitle(
        f"[{model_label}  |  {memory_label}]\n{params_str}",
        color=TEXT, fontsize=10.5, fontweight="bold", y=0.99, va="top"
    )

    gs = GridSpec(3, 4, figure=fig,
                  left=0.055, right=0.97, top=0.90, bottom=0.07,
                  hspace=0.55, wspace=0.38)

    # ── 3D phase space ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[:2, :2], projection="3d")
    pts = np.c_[x, y, z]
    for i in range(n-1):
        frac = i / n
        ax3.plot(pts[i:i+2,0], pts[i:i+2,1], pts[i:i+2,2],
                 lw=0.35, color=CMAP_TRAJ(frac), alpha=0.75)
    ax3d_style(ax3)
    ax3.set_xlabel("x", labelpad=2); ax3.set_ylabel("y", labelpad=2)
    ax3.set_zlabel("z", labelpad=2)
    ax3.set_title("Espacio de fase 3D", color=TEXT, fontsize=9, pad=5)

    # ── 2D projections ──────────────────────────────────────────────────────
    projs = [("x","y", x, y, gs[0,2]), ("x","z", x, z, gs[0,3]),
             ("y","z", y, z, gs[1,2])]
    for xl, yl, xd, yd, gss in projs:
        ax2 = fig.add_subplot(gss)
        ax2.plot(xd, yd, lw=0.3, color=color, alpha=0.8)
        ax_style(ax2)
        ax2.set_xlabel(xl, fontsize=8); ax2.set_ylabel(yl, fontsize=8)
        ax2.set_title(f"Proy. {xl}-{yl}", color=TEXT, fontsize=8)

    # detail x-y with colormap
    ax_xy2 = fig.add_subplot(gs[1,3])
    for i in range(n-1):
        ax_xy2.plot(x[i:i+2], z[i:i+2], lw=0.25,
                    color=CMAP_TRAJ(i/n), alpha=0.7)
    ax_style(ax_xy2)
    ax_xy2.set_xlabel("x", fontsize=8); ax_xy2.set_ylabel("z", fontsize=8)
    ax_xy2.set_title("Proy. x-z (color=tiempo)", color=TEXT, fontsize=7.5)

    # ── Time series ─────────────────────────────────────────────────────────
    ts_cfg = [("x(t)", x, "#E64B35"), ("y(t)", y, "#4DBBD5"), ("z(t)", z, "#00A087")]
    for col_i, (lbl, sig, clr) in enumerate(ts_cfg):
        ax_t = fig.add_subplot(gs[2, col_i])
        ax_t.plot(t, sig, lw=0.35, color=clr)
        ax_style(ax_t)
        ax_t.set_xlabel("t [s]", fontsize=8); ax_t.set_ylabel(lbl, fontsize=8)
        ax_t.set_title(f"Serie {lbl}", color=TEXT, fontsize=8)

    # ── Info panel ──────────────────────────────────────────────────────────
    ax_i = fig.add_subplot(gs[2, 3])
    ax_i.set_facecolor(PANEL); ax_i.axis("off")
    info = [
        f"Modelo:  {model_label}",
        f"Memoria: {memory_label}",
        "",
        f"ID: {case_id}",
        "",
        *[f"  {p.strip()}" for p in params_str.split("|")],
        "",
        f"Transiente: {t_transient} s",
        f"Puntos: {n:,}  (1/{stride})",
    ]
    ax_i.text(0.04, 0.97, "\n".join(info), transform=ax_i.transAxes,
              va="top", ha="left", color=TEXT, fontsize=7.2,
              fontfamily="monospace", linespacing=1.5)
    ax_i.set_title("Parámetros", color=TEXT, fontsize=8)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Catálogo de candidatos a graficar
# ══════════════════════════════════════════════════════════════════════════════
# Colores por modo/modelo
C = {
    "sat_full":    "#E64B35",   # rojo       - Saturación Memoria Completa
    "sat_win":     "#4DBBD5",   # cyan       - Saturación Memoria Truncada
    "arctan_full": "#00A087",   # verde      - Arctan Memoria Completa
    "arctan_win":  "#F39B7F",   # naranja    - Arctan Memoria Truncada
}

# ── Chua No Suave (Saturacion) – q=0.9998 ────────────────────────────────────
SAT_FULL_DIR = ROOT / "outputs" / "saturation_search_seed1_mem_full_sweep"
SAT_WIN_DIR  = ROOT / "outputs" / "saturation_search_seed0p9998_mem_window_sweep"

CANDIDATES_SAT_FULL = [
    # chaotic_candidate_pending_robustness
    dict(case_id="m1_m1p2000_m0_m0p2000_branch_0",
         traj=SAT_FULL_DIR / "m1_m1p2000_m0_m0p2000_branch_0_trajectory.csv",
         params="m1=-1.2 | m0=-0.2 | ω₀=2.039 | k=0.263 | A₀=4.80",
         verdict="chaotic"),
    # nonperiodic_candidate
    dict(case_id="m1_m0p8000_m0_m0p1000_branch_0",
         traj=SAT_FULL_DIR / "m1_m0p8000_m0_m0p1000_branch_0_trajectory.csv",
         params="m1=-0.8 | m0=-0.1 | ω₀=3.245 | k=0.613 | A₀=1.29",
         verdict="nonperiodic"),
    dict(case_id="m1_m1p0000_m0_m0p1000_branch_1",
         traj=SAT_FULL_DIR / "m1_m1p0000_m0_m0p1000_branch_1_trajectory.csv",
         params="m1=-1.0 | m0=-0.1 | ω₀=3.245 | k=0.813 | A₀=1.24",
         verdict="nonperiodic"),
    dict(case_id="m1_m1p2000_m0_m0p1000_branch_1",
         traj=SAT_FULL_DIR / "m1_m1p2000_m0_m0p1000_branch_1_trajectory.csv",
         params="m1=-1.2 | m0=-0.1 | ω₀=3.245 | k=1.013 | A₀=1.20",
         verdict="nonperiodic"),
    dict(case_id="m1_m1p4000_m0_m0p1000_branch_1",
         traj=SAT_FULL_DIR / "m1_m1p4000_m0_m0p1000_branch_1_trajectory.csv",
         params="m1=-1.4 | m0=-0.1 | ω₀=3.245 | k=1.213 | A₀=1.17",
         verdict="nonperiodic"),
    dict(case_id="m1_m1p6000_m0_m0p1000_branch_1",
         traj=SAT_FULL_DIR / "m1_m1p6000_m0_m0p1000_branch_1_trajectory.csv",
         params="m1=-1.6 | m0=-0.1 | ω₀=3.245 | k=1.413 | A₀=1.16",
         verdict="nonperiodic"),
]

CANDIDATES_SAT_WIN = [
    # chaotic_candidate_pending_robustness
    dict(case_id="m1_m1p2000_m0_m0p2000_branch_0_window",
         traj=SAT_WIN_DIR / "m1_m1p2000_m0_m0p2000_branch_0_trajectory.csv",
         params="m1=-1.2 | m0=-0.2 | ω₀=2.040 | k=0.263 | A₀=4.80 | Lm=10s",
         verdict="chaotic"),
]

# ── Chua Arctan Fraccionario – q=0.99 memoria completa ───────────────────────
ARC_FULL_DIR = ROOT / "outputs" / "arctan_search_seed0p99_mem_full"

# Solo los chaotic_candidate_pending_robustness del summary
CANDIDATES_ARC_FULL = [
    dict(case_id="a1_0p10_a2_m1p2000_rho_1p00_branch_0",
         traj=ARC_FULL_DIR / "a1_0p10_a2_m1p2000_rho_1p00_branch_0_trajectory.csv",
         params="a1=0.10 | a2=-1.20 | ρ=1.00 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p10_a2_m1p2000_rho_1p25_branch_0",
         traj=ARC_FULL_DIR / "a1_0p10_a2_m1p2000_rho_1p25_branch_0_trajectory.csv",
         params="a1=0.10 | a2=-1.20 | ρ=1.25 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p10_a2_m1p2000_rho_2p00_branch_0",
         traj=ARC_FULL_DIR / "a1_0p10_a2_m1p2000_rho_2p00_branch_0_trajectory.csv",
         params="a1=0.10 | a2=-1.20 | ρ=2.00 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p10_a2_m1p2000_rho_3p00_branch_0",
         traj=ARC_FULL_DIR / "a1_0p10_a2_m1p2000_rho_3p00_branch_0_trajectory.csv",
         params="a1=0.10 | a2=-1.20 | ρ=3.00 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p10_a2_m1p2000_rho_4p00_branch_0",
         traj=ARC_FULL_DIR / "a1_0p10_a2_m1p2000_rho_4p00_branch_0_trajectory.csv",
         params="a1=0.10 | a2=-1.20 | ρ=4.00 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p10_a2_m1p5585_rho_0p75_branch_0",
         traj=ARC_FULL_DIR / "a1_0p10_a2_m1p5585_rho_0p75_branch_0_trajectory.csv",
         params="a1=0.10 | a2=-1.5585 | ρ=0.75 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p10_a2_m1p5585_rho_1p00_branch_0",
         traj=ARC_FULL_DIR / "a1_0p10_a2_m1p5585_rho_1p00_branch_0_trajectory.csv",
         params="a1=0.10 | a2=-1.5585 | ρ=1.00 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p10_a2_m1p5585_rho_1p25_branch_0",
         traj=ARC_FULL_DIR / "a1_0p10_a2_m1p5585_rho_1p25_branch_0_trajectory.csv",
         params="a1=0.10 | a2=-1.5585 | ρ=1.25 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p10_a2_m1p5585_rho_1p50_branch_0",
         traj=ARC_FULL_DIR / "a1_0p10_a2_m1p5585_rho_1p50_branch_0_trajectory.csv",
         params="a1=0.10 | a2=-1.5585 | ρ=1.50 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p10_a2_m1p5585_rho_2p00_branch_0",
         traj=ARC_FULL_DIR / "a1_0p10_a2_m1p5585_rho_2p00_branch_0_trajectory.csv",
         params="a1=0.10 | a2=-1.5585 | ρ=2.00 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p10_a2_m1p5585_rho_3p00_branch_0",
         traj=ARC_FULL_DIR / "a1_0p10_a2_m1p5585_rho_3p00_branch_0_trajectory.csv",
         params="a1=0.10 | a2=-1.5585 | ρ=3.00 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p10_a2_m1p5585_rho_4p00_branch_0",
         traj=ARC_FULL_DIR / "a1_0p10_a2_m1p5585_rho_4p00_branch_0_trajectory.csv",
         params="a1=0.10 | a2=-1.5585 | ρ=4.00 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p10_a2_m1p5585_rho_4p00_branch_1",
         traj=ARC_FULL_DIR / "a1_0p10_a2_m1p5585_rho_4p00_branch_1_trajectory.csv",
         params="a1=0.10 | a2=-1.5585 | ρ=4.00 br1 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p20_a2_m1p2000_rho_1p25_branch_0",
         traj=ARC_FULL_DIR / "a1_0p20_a2_m1p2000_rho_1p25_branch_0_trajectory.csv",
         params="a1=0.20 | a2=-1.20 | ρ=1.25 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p20_a2_m1p2000_rho_2p00_branch_0",
         traj=ARC_FULL_DIR / "a1_0p20_a2_m1p2000_rho_2p00_branch_0_trajectory.csv",
         params="a1=0.20 | a2=-1.20 | ρ=2.00 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p20_a2_m1p2000_rho_3p00_branch_0",
         traj=ARC_FULL_DIR / "a1_0p20_a2_m1p2000_rho_3p00_branch_0_trajectory.csv",
         params="a1=0.20 | a2=-1.20 | ρ=3.00 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p20_a2_m1p2000_rho_4p00_branch_0",
         traj=ARC_FULL_DIR / "a1_0p20_a2_m1p2000_rho_4p00_branch_0_trajectory.csv",
         params="a1=0.20 | a2=-1.20 | ρ=4.00 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p20_a2_m1p2000_rho_4p00_branch_1",
         traj=ARC_FULL_DIR / "a1_0p20_a2_m1p2000_rho_4p00_branch_1_trajectory.csv",
         params="a1=0.20 | a2=-1.20 | ρ=4.00 br1 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p20_a2_m1p5585_rho_1p00_branch_0",
         traj=ARC_FULL_DIR / "a1_0p20_a2_m1p5585_rho_1p00_branch_0_trajectory.csv",
         params="a1=0.20 | a2=-1.5585 | ρ=1.00 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p20_a2_m1p5585_rho_1p25_branch_0",
         traj=ARC_FULL_DIR / "a1_0p20_a2_m1p5585_rho_1p25_branch_0_trajectory.csv",
         params="a1=0.20 | a2=-1.5585 | ρ=1.25 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p20_a2_m1p5585_rho_1p50_branch_0",
         traj=ARC_FULL_DIR / "a1_0p20_a2_m1p5585_rho_1p50_branch_0_trajectory.csv",
         params="a1=0.20 | a2=-1.5585 | ρ=1.50 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p20_a2_m1p5585_rho_2p00_branch_0",
         traj=ARC_FULL_DIR / "a1_0p20_a2_m1p5585_rho_2p00_branch_0_trajectory.csv",
         params="a1=0.20 | a2=-1.5585 | ρ=2.00 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p20_a2_m1p5585_rho_3p00_branch_0",
         traj=ARC_FULL_DIR / "a1_0p20_a2_m1p5585_rho_3p00_branch_0_trajectory.csv",
         params="a1=0.20 | a2=-1.5585 | ρ=3.00 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p20_a2_m1p5585_rho_3p00_branch_1",
         traj=ARC_FULL_DIR / "a1_0p20_a2_m1p5585_rho_3p00_branch_1_trajectory.csv",
         params="a1=0.20 | a2=-1.5585 | ρ=3.00 br1 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p20_a2_m1p5585_rho_4p00_branch_0",
         traj=ARC_FULL_DIR / "a1_0p20_a2_m1p5585_rho_4p00_branch_0_trajectory.csv",
         params="a1=0.20 | a2=-1.5585 | ρ=4.00 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
    dict(case_id="a1_0p20_a2_m1p5585_rho_4p00_branch_1",
         traj=ARC_FULL_DIR / "a1_0p20_a2_m1p5585_rho_4p00_branch_1_trajectory.csv",
         params="a1=0.20 | a2=-1.5585 | ρ=4.00 br1 | α=8.4562 | β=12.0732 | γ=0.0052",
         verdict="chaotic"),
]

# Arctan ventana: NINGUN candidato sobrevive la continuacion -> no se grafica
# (resultado es en si mismo importante: se reporta en el resumen)


# ══════════════════════════════════════════════════════════════════════════════
# Generar figuras individuales
# ══════════════════════════════════════════════════════════════════════════════

GROUPS = [
    (CANDIDATES_SAT_FULL, C["sat_full"],    "Chua No Suave (Saturacion)", "Mem. Completa q=0.9998", "sat_full"),
    (CANDIDATES_SAT_WIN,  C["sat_win"],     "Chua No Suave (Saturacion)", "Mem. Truncada Lm=10s q=0.9998","sat_win"),
    (CANDIDATES_ARC_FULL, C["arctan_full"], "Chua Arctan Fraccionario",  "Mem. Completa q=0.99",  "arc_full"),
]

generated = []

for cands, color, model_label, mem_label, group_key in GROUPS:
    sub_dir = OUT_DIR / group_key
    sub_dir.mkdir(exist_ok=True)
    for c in cands:
        traj = c["traj"]
        if not traj.exists():
            print(f"[SKIP] {c['case_id']} -- no existe {traj.name}")
            continue
        print(f"[PLOT] {group_key}/{c['case_id']}")
        try:
            fig = plot_candidate(
                traj_path=traj,
                case_id=c["case_id"],
                model_label=model_label,
                memory_label=mem_label,
                params_str=c["params"],
                color=color,
            )
            out = sub_dir / f"{c['case_id']}_detailed.png"
            fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
            plt.close(fig)
            generated.append((group_key, c["case_id"], c["verdict"], out))
            print(f"  -> {out.name}")
        except Exception as e:
            print(f"  [ERR] {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Figura resumen por grupo (espacios de fase 3D en mosaico)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[SUMMARY FIGURES]")

for cands, color, model_label, mem_label, group_key in GROUPS:
    valid = [c for c in cands if c["traj"].exists()]
    if not valid:
        continue
    n = len(valid)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols

    fig_s = plt.figure(figsize=(ncols * 3.8, nrows * 3.3 + 0.8), facecolor=BG)
    fig_s.suptitle(
        f"{model_label}  —  {mem_label}\n{n} candidatos",
        color=TEXT, fontsize=11, fontweight="bold", y=0.99
    )

    for idx, c in enumerate(valid):
        try:
            t, x, y, z, stride = load_traj(c["traj"], max_pts=6000)
        except:
            continue
        ax = fig_s.add_subplot(nrows, ncols, idx+1, projection="3d")
        pts = np.c_[x, y, z]
        for i in range(len(pts)-1):
            ax.plot(pts[i:i+2,0], pts[i:i+2,1], pts[i:i+2,2],
                    lw=0.3, color=CMAP_TRAJ(i/len(pts)), alpha=0.8)
        ax3d_style(ax)
        ax.set_xlabel("x", fontsize=5, labelpad=0)
        ax.set_ylabel("y", fontsize=5, labelpad=0)
        ax.set_zlabel("z", fontsize=5, labelpad=0)
        short = c["case_id"].replace("branch","b")
        label = "★" if c["verdict"]=="chaotic" else "○"
        ax.set_title(f"{label} {short}", color=TEXT, fontsize=5.5, pad=2,
                     wrap=True)

    fig_s.tight_layout(rect=[0, 0, 1, 0.95])
    out_s = OUT_DIR / f"summary_{group_key}.png"
    fig_s.savefig(out_s, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig_s)
    print(f"  -> {out_s.name}  ({n} paneles)")


# ══════════════════════════════════════════════════════════════════════════════
# Figura mega-resumen: todos los grupos juntos
# ══════════════════════════════════════════════════════════════════════════════

print("\n[MEGA SUMMARY]")

all_valid = []
for cands, color, model_label, mem_label, group_key in GROUPS:
    for c in cands:
        if c["traj"].exists():
            all_valid.append((c, color, f"{model_label}\n{mem_label}", group_key))

n_all = len(all_valid)
ncols = min(6, n_all)
nrows = (n_all + ncols - 1) // ncols

fig_mega = plt.figure(figsize=(ncols*3.5, nrows*3.2+1), facecolor=BG)
fig_mega.suptitle(
    "Todos los candidatos caóticos — Chua Fraccionario No Suave + Arctan",
    color=TEXT, fontsize=12, fontweight="bold", y=0.99
)

for idx, (c, color, title, group_key) in enumerate(all_valid):
    try:
        t, x, y, z, stride = load_traj(c["traj"], max_pts=5000)
    except:
        continue
    ax = fig_mega.add_subplot(nrows, ncols, idx+1, projection="3d")
    pts = np.c_[x, y, z]
    for i in range(len(pts)-1):
        ax.plot(pts[i:i+2,0], pts[i:i+2,1], pts[i:i+2,2],
                lw=0.3, color=CMAP_TRAJ(i/len(pts)), alpha=0.75)
    ax3d_style(ax)
    ax.set_xlabel("x",fontsize=4.5,labelpad=0)
    ax.set_ylabel("y",fontsize=4.5,labelpad=0)
    ax.set_zlabel("z",fontsize=4.5,labelpad=0)
    short = c["case_id"].replace("branch","b")[:28]
    label = "★" if c["verdict"]=="chaotic" else "○"
    grp_short = {"sat_full":"Sat-F","sat_win":"Sat-W","arc_full":"Arc-F"}[group_key]
    ax.set_title(f"[{grp_short}] {label}\n{short}", color=TEXT,
                 fontsize=4.8, pad=1)

fig_mega.tight_layout(rect=[0, 0, 1, 0.97])
out_mega = OUT_DIR / "MEGA_all_candidates.png"
fig_mega.savefig(out_mega, dpi=130, facecolor=BG, bbox_inches="tight")
plt.close(fig_mega)
print(f"  -> {out_mega.name}  ({n_all} paneles)")

# -- Reporte terminal ----------------------------------------------------------
print("\n" + "="*65)
print("REPORTE — generate_all_plots_and_summary.py")
print("="*65)
print(f"\nFiguras generadas : {len(generated)}")
print(f"Directorio salida : {OUT_DIR}")
print(f"\nDesglose por grupo:")
for gk in ["sat_full","sat_win","arc_full"]:
    cnt = sum(1 for g,_,_,_ in generated if g==gk)
    print(f"  {gk:12s}: {cnt} figuras")
print()
