"""
plot_sphere_closeup.py
======================
Genera figuras de sondeos esféricos con ACERCAMIENTO a cada equilibrio.
Exporta cada figura en tres formatos y dos temas:

  Figs/
  ├── png/   (fondo oscuro, 180 dpi  — ya existentes, para la presentación)
  ├── pdf/   (fondo claro, vectorial  — para incluir en LaTeX/publicación)
  └── eps/   (fondo claro, vectorial  — para revistas que requieran EPS)

Figuras generadas por cada modo (light/dark):
  chua_frac_ns_sphere_closeup_E0
  chua_frac_ns_sphere_closeup_Ep
  chua_frac_ns_sphere_closeup_Em
  chua_frac_ns_fig13_hiddenness_spherical_probes
  chua_frac_ns_fig14_hiddenness_contact_heatmap
"""

import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
DATA_DIR = (
    BASE / "version_2" / "validation" / "outputs"
    / "candidate_chaos_hiddenness"
    / "danca2017_chua_fractional_saturation_candidate" / "report"
)
FIGS_BASE = BASE / "DF y NC Chua entero y fraccionario copy" / "Figs"

# Directorios de salida por formato
DIR_PNG = FIGS_BASE / "png"
DIR_PDF = FIGS_BASE / "pdf"
DIR_EPS = FIGS_BASE / "eps"
for d in (FIGS_BASE, DIR_PNG, DIR_PDF, DIR_EPS):
    d.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON = DATA_DIR / "hiddenness_run_summary.json"
RESULTS_CSV  = DATA_DIR / "ball_sampling_results.csv"

# ── Paletas de colores por tema ────────────────────────────────────────────
THEMES = {
    "dark": {
        "fig_bg":      "#0f172a",
        "ax_bg":       "#1e293b",
        "pane_edge":   "#334155",
        "tick_color":  "#94a3b8",
        "label_color": "#94a3b8",
        "title_color": "#f1f5f9",
        "text_color":  "#e2e8f0",
        "grid_color":  "#1e293b",
        "legend_bg":   "#1e293b",
        "legend_edge": "#475569",
        "legend_text": "#e2e8f0",
        "eq_edge":     "white",
        "sphere_edge": "#475569",
        # Colores de datos (oscuro — saturados y brillantes)
        "target":      "#ef4444",
        "equilibrium": "#3b82f6",
        "diverged":    "#64748b",
        "no_match":    "#94a3b8",
        "other":       "#f59e0b",
        "eq_star":     {
            "E0": "#3b82f6",
            "E+": "#ef4444",
            "E-": "#f59e0b",
        },
    },
    "light": {
        "fig_bg":      "white",
        "ax_bg":       "#f8fafc",
        "pane_edge":   "#cbd5e1",
        "tick_color":  "#334155",
        "label_color": "#475569",
        "title_color": "#0f172a",
        "text_color":  "#1e293b",
        "grid_color":  "#e2e8f0",
        "legend_bg":   "white",
        "legend_edge": "#94a3b8",
        "legend_text": "#1e293b",
        "eq_edge":     "#1e293b",
        "sphere_edge": "#94a3b8",
        # Colores de datos (claro — más oscuros para contraste sobre blanco)
        "target":      "#dc2626",
        "equilibrium": "#2563eb",
        "diverged":    "#64748b",
        "no_match":    "#94a3b8",
        "other":       "#d97706",
        "eq_star":     {
            "E0": "#2563eb",
            "E+": "#dc2626",
            "E-": "#d97706",
        },
    },
}

# Mapeo de etiquetas CSV → (clave de color en tema, leyenda, marcador)
LABEL_MAP = {
    "target_candidate":     ("target",      "TARGET (alcanza atractor)", "o"),
    "target_attractor":     ("target",      "TARGET (alcanza atractor)", "o"),
    "equilibrium":          ("equilibrium", "Converge al equilibrio",    "s"),
    "stable_equilibrium":   ("equilibrium", "Converge al equilibrio",    "s"),
    "converged_equilibrium":("equilibrium", "Converge al equilibrio",    "s"),
    "diverged":             ("diverged",    "Divergencia",               "^"),
    "divergence":           ("diverged",    "Divergencia",               "^"),
    "no_target_match":      ("no_match",    "Sin match al target",       "D"),
    "numerical_failure":    ("no_match",    "Fallo numérico",            "x"),
    "unclassified":         ("other",       "No clasificado",            "P"),
    "other_attractor":      ("other",       "Otro atractor",             "D"),
}


# ── Carga de datos ─────────────────────────────────────────────────────────
def load_summary():
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def load_results():
    rows = []
    with open(RESULTS_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def parse_row(row):
    return {
        "eq":     row["equilibrium"],
        "radius": float(row["radius"]),
        "x0":     np.array([float(row["x0"]), float(row["y0"]), float(row["z0"])]),
        "label":  row["label"],
    }

def make_sphere_surface(center, r, n=28):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    xs = r * np.outer(np.cos(u), np.sin(v)) + center[0]
    ys = r * np.outer(np.sin(u), np.sin(v)) + center[1]
    zs = r * np.outer(np.ones(n), np.cos(v)) + center[2]
    return xs, ys, zs


# ── Utilidades de estilo ───────────────────────────────────────────────────
def apply_ax_theme(ax, theme):
    """Aplica el tema de color a un Axes3D."""
    t = THEMES[theme]
    ax.set_facecolor(t["ax_bg"])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(t["pane_edge"])
    ax.yaxis.pane.set_edgecolor(t["pane_edge"])
    ax.zaxis.pane.set_edgecolor(t["pane_edge"])
    ax.tick_params(colors=t["tick_color"], labelsize=6.5)
    ax.xaxis.label.set_color(t["label_color"])
    ax.yaxis.label.set_color(t["label_color"])
    ax.zaxis.label.set_color(t["label_color"])
    ax.grid(True, color=t["grid_color"], linewidth=0.3, alpha=0.5)


def save_figure(fig, stem, theme, out_dir_png, out_dir_pdf, out_dir_eps, dpi=180):
    """Guarda la figura en PNG (oscuro) o PDF+EPS (claro)."""
    if theme == "dark":
        p = out_dir_png / f"{stem}.png"
        fig.savefig(str(p), dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [PNG] {p.name}")
    else:
        # PDF
        p = out_dir_pdf / f"{stem}.pdf"
        fig.savefig(str(p), format="pdf", bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [PDF] {p.name}")
        # EPS (sin transparencia — se aplana a blanco)
        p = out_dir_eps / f"{stem}.eps"
        fig.savefig(str(p), format="eps", bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [EPS] {p.name}")


# ── Figura por equilibrio ──────────────────────────────────────────────────
def plot_equilibrium_closeup(eq_name, eq_pt, all_rows, radii, summary_decisions,
                             out_dir_png, out_dir_pdf, out_dir_eps, theme):
    t = THEMES[theme]
    n_radii = len(radii)
    ncols = 3
    nrows = (n_radii + ncols - 1) // ncols

    fig = plt.figure(figsize=(5.5 * ncols, 5.0 * nrows),
                     dpi=180 if theme == "dark" else 150)
    fig.patch.set_facecolor(t["fig_bg"])

    eq_star_color = t["eq_star"].get(eq_name, t["title_color"])

    rows_by_radius = {}
    for row in all_rows:
        if row["eq"] == eq_name:
            rows_by_radius.setdefault(row["radius"], []).append(row)

    hits_by_radius = {}
    for d in summary_decisions:
        if d["equilibrium"] == eq_name:
            hits_by_radius[d["radius"]] = (d["target_hits"], d["samples"])

    plotted_labels = set()

    for idx, radius in enumerate(radii):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        apply_ax_theme(ax, theme)

        # Esfera semitransparente
        xs, ys, zs = make_sphere_surface(eq_pt, radius)
        ax.plot_surface(xs, ys, zs, color=eq_star_color, alpha=0.06,
                        edgecolor=t["sphere_edge"], linewidth=0.2)

        # Equilibrio (estrella)
        lbl_star = f"Eq. {eq_name}" if "eq_star" not in plotted_labels else ""
        ax.scatter([eq_pt[0]], [eq_pt[1]], [eq_pt[2]],
                   color=eq_star_color, marker="*", s=200,
                   edgecolors=t["eq_edge"], linewidths=0.7, zorder=15,
                   label=lbl_star)
        plotted_labels.add("eq_star")

        # Condiciones iniciales por destino
        subset = rows_by_radius.get(radius, [])
        dest_groups = {}
        for row in subset:
            dest_groups.setdefault(row["label"], []).append(row["x0"])

        for dest, pts_list in dest_groups.items():
            color_key, lbl, mrk = LABEL_MAP.get(dest, ("other", dest, "o"))
            col = t[color_key]
            pts_arr = np.array(pts_list)
            ax.scatter(pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2],
                       color=col, marker=mrk, s=20, alpha=0.88,
                       edgecolors="none", zorder=10,
                       label=lbl if lbl not in plotted_labels else "")
            plotted_labels.add(lbl)

        # Zoom local
        span = radius * 1.35
        ax.set_xlim(eq_pt[0] - span, eq_pt[0] + span)
        ax.set_ylim(eq_pt[1] - span, eq_pt[1] + span)
        ax.set_zlim(eq_pt[2] - span, eq_pt[2] + span)

        ax.set_xlabel("x", fontsize=7, labelpad=2)
        ax.set_ylabel("y", fontsize=7, labelpad=2)
        ax.set_zlabel("z", fontsize=7, labelpad=2)

        hits, samples = hits_by_radius.get(radius, (0, 0))
        pct = 100.0 * hits / samples if samples > 0 else 0.0
        ax.set_title(
            f"$r = {radius:.0e}$   →   {hits}/{samples} TARGET ({pct:.0f}%)",
            color=t["title_color"], fontsize=9.5, fontweight="bold", pad=8
        )
        ax.view_init(elev=22, azim=225 - idx * 10)

    # Título global
    decision_text = {
        "E0": r"SIN CONTACTOS — no autoexcitado desde $E_0$",
        "E+": r"CONTACTOS DETECTADOS — autoexcitado desde $E_+$",
        "E-": r"CONTACTOS DETECTADOS — autoexcitado desde $E_-$",
    }.get(eq_name, "")
    fig.suptitle(
        f"Sondeos esféricos: equilibrio {eq_name}   |   {decision_text}",
        color=t["title_color"], fontsize=13, fontweight="bold", y=1.01
    )

    # Leyenda global
    handles, labels = [], []
    seen = set()
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen and l:
                handles.append(h); labels.append(l); seen.add(l)
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center", ncol=min(len(handles), 4),
            fontsize=9, framealpha=0.85,
            facecolor=t["legend_bg"], edgecolor=t["legend_edge"],
            labelcolor=t["legend_text"],
            bbox_to_anchor=(0.5, -0.04)
        )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    stem = f"chua_frac_ns_sphere_closeup_{eq_name.replace('+', 'p').replace('-', 'm')}"
    save_figure(fig, stem, theme, out_dir_png, out_dir_pdf, out_dir_eps)
    plt.close(fig)


# ── Figura combinada (fig13) ───────────────────────────────────────────────
def plot_combined_summary(equilibria, all_rows, radii, summary_decisions,
                          out_dir_png, out_dir_pdf, out_dir_eps, theme):
    t = THEMES[theme]
    sel_radii = [radii[0], radii[len(radii) // 2], radii[-1]]
    n_eq = len(equilibria)
    n_r  = len(sel_radii)

    fig = plt.figure(figsize=(5.5 * n_eq, 5.0 * n_r),
                     dpi=180 if theme == "dark" else 150)
    fig.patch.set_facecolor(t["fig_bg"])

    rows_by_eq_r = {}
    for row in all_rows:
        key = (row["eq"], row["radius"])
        rows_by_eq_r.setdefault(key, []).append(row)

    hits_map = {}
    for d in summary_decisions:
        hits_map[(d["equilibrium"], d["radius"])] = (d["target_hits"], d["samples"])

    plotted_labels = set()
    subplot_idx = 1

    for col_idx, (eq_name, eq_pt) in enumerate(equilibria.items()):
        eq_star_color = t["eq_star"].get(eq_name, t["title_color"])
        for row_idx, radius in enumerate(sel_radii):
            ax = fig.add_subplot(n_r, n_eq, subplot_idx, projection="3d")
            apply_ax_theme(ax, theme)

            xs, ys, zs = make_sphere_surface(eq_pt, radius)
            ax.plot_surface(xs, ys, zs, color=eq_star_color, alpha=0.06,
                            edgecolor=t["sphere_edge"], linewidth=0.15)

            ax.scatter([eq_pt[0]], [eq_pt[1]], [eq_pt[2]],
                       color=eq_star_color, marker="*", s=170,
                       edgecolors=t["eq_edge"], linewidths=0.5, zorder=15)

            subset = rows_by_eq_r.get((eq_name, radius), [])
            dest_groups = {}
            for row in subset:
                dest_groups.setdefault(row["label"], []).append(row["x0"])

            for dest, pts_list in dest_groups.items():
                color_key, lbl, mrk = LABEL_MAP.get(dest, ("other", dest, "o"))
                col = t[color_key]
                pts_arr = np.array(pts_list)
                ax.scatter(pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2],
                           color=col, marker=mrk, s=14, alpha=0.88,
                           edgecolors="none", zorder=10,
                           label=lbl if lbl not in plotted_labels else "")
                plotted_labels.add(lbl)

            span = radius * 1.35
            ax.set_xlim(eq_pt[0] - span, eq_pt[0] + span)
            ax.set_ylim(eq_pt[1] - span, eq_pt[1] + span)
            ax.set_zlim(eq_pt[2] - span, eq_pt[2] + span)

            hits, samples = hits_map.get((eq_name, radius), (0, 0))
            pct = 100.0 * hits / samples if samples > 0 else 0.0

            if row_idx == 0:
                ax.set_title(
                    f"Eq. {eq_name}\n$r = {radius:.0e}$  {hits}/{samples} ({pct:.0f}%)",
                    color=t["title_color"], fontsize=9, fontweight="bold", pad=6
                )
            else:
                ax.set_title(
                    f"$r = {radius:.0e}$  {hits}/{samples} ({pct:.0f}%)",
                    color=t["title_color"], fontsize=8.5, pad=5
                )

            ax.set_xlabel("x", fontsize=6.5, labelpad=1)
            ax.set_ylabel("y", fontsize=6.5, labelpad=1)
            ax.set_zlabel("z", fontsize=6.5, labelpad=1)
            ax.view_init(elev=20, azim=220)
            subplot_idx += 1

    fig.suptitle(
        "Sondeos esféricos — acercamiento local a cada equilibrio\n"
        r"Chua fraccionario no suave  $q = 0.9998$,  $(m_0, m_1) = (-0.2,\ -1.2)$",
        color=t["title_color"], fontsize=13, fontweight="bold", y=1.02
    )

    handles, labels = [], []
    seen = set()
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen and l:
                handles.append(h); labels.append(l); seen.add(l)
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center", ncol=min(len(handles), 5),
            fontsize=9, framealpha=0.85,
            facecolor=t["legend_bg"], edgecolor=t["legend_edge"],
            labelcolor=t["legend_text"],
            bbox_to_anchor=(0.5, -0.04)
        )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    stem = "chua_frac_ns_fig13_hiddenness_spherical_probes"
    save_figure(fig, stem, theme, out_dir_png, out_dir_pdf, out_dir_eps)
    plt.close(fig)


# ── Mapa de calor (fig14) ──────────────────────────────────────────────────
def plot_heatmap(equilibria, radii, summary_decisions,
                 out_dir_png, out_dir_pdf, out_dir_eps, theme):
    t = THEMES[theme]

    hits_map = {}
    for d in summary_decisions:
        hits_map[(d["equilibrium"], d["radius"])] = (d["target_hits"], d["samples"])

    eq_names = list(equilibria.keys())
    n_eq = len(eq_names)
    n_r  = len(radii)

    mat_hits = np.zeros((n_eq, n_r))
    mat_pct  = np.zeros((n_eq, n_r))
    for i, eq in enumerate(eq_names):
        for j, r in enumerate(radii):
            h, s = hits_map.get((eq, r), (0, 1))
            mat_hits[i, j] = h
            mat_pct[i, j]  = 100.0 * h / s if s > 0 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8),
                             dpi=180 if theme == "dark" else 150)
    fig.patch.set_facecolor(t["fig_bg"])

    radius_labels = [f"${r:.0e}$" for r in radii]

    cmap_hits = "YlOrRd"
    cmap_pct  = "RdYlGn_r" if theme == "dark" else "RdYlGn_r"

    for ax, mat, title, fmt_str, cmap in [
        (axes[0], mat_hits, "Contactos TARGET (absolutos)", ".0f", cmap_hits),
        (axes[1], mat_pct,  "Tasa de contacto TARGET (%)",  ".0f", cmap_pct),
    ]:
        ax.set_facecolor(t["ax_bg"])
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0,
                       vmax=mat.max() if mat.max() > 0 else 1)

        for i in range(n_eq):
            for j in range(n_r):
                val = mat[i, j]
                txt_color = "white" if val > mat.max() * 0.65 else "#111827"
                ax.text(j, i, f"{val:{fmt_str}}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=txt_color)

        ax.set_xticks(range(n_r))
        ax.set_xticklabels(radius_labels, rotation=30, ha="right",
                           fontsize=9, color=t["text_color"])
        ax.set_yticks(range(n_eq))
        ax.set_yticklabels(eq_names, fontsize=11, color=t["text_color"],
                           fontweight="bold")
        ax.set_xlabel("Radio de sondeo", color=t["label_color"], fontsize=10)
        ax.set_title(title, color=t["title_color"], fontsize=11,
                     fontweight="bold", pad=10)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=t["tick_color"], labelsize=8)

    fig.suptitle(
        "Mapa de calor — sondeos esféricos por equilibrio y radio",
        color=t["title_color"], fontsize=13, fontweight="bold", y=1.03
    )
    plt.tight_layout()

    stem = "chua_frac_ns_fig14_hiddenness_contact_heatmap"
    save_figure(fig, stem, theme, out_dir_png, out_dir_pdf, out_dir_eps)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("[plot_sphere_closeup] Cargando datos...")
    summary  = load_summary()
    rows_raw = load_results()

    equilibria = {k: np.array(v) for k, v in summary["equilibria"].items()}
    decisions  = summary["decisions"]
    radii      = sorted(set(float(r["radius"]) for r in rows_raw
                            if r["profile"] == "report"))
    rows       = [parse_row(r) for r in rows_raw if r["profile"] == "report"]

    print(f"[info] Equilibrios : {list(equilibria.keys())}")
    print(f"[info] Radios      : {radii}")
    print(f"[info] Total filas : {len(rows)}")

    for theme in ("dark", "light"):
        tag = "PNG (oscuro)" if theme == "dark" else "PDF+EPS (claro)"
        print("\n" + "-"*60)
        print(f"  TEMA: {tag}")
        print("-"*60)

        # Figuras individuales por equilibrio
        for eq_name, eq_pt in equilibria.items():
            print(f"\n  [eq={eq_name}] Acercamiento esférico...")
            plot_equilibrium_closeup(
                eq_name, eq_pt, rows, radii, decisions,
                DIR_PNG, DIR_PDF, DIR_EPS, theme
            )

        # Figura combinada
        print("\n  [fig13] Figura combinada...")
        plot_combined_summary(
            equilibria, rows, radii, decisions,
            DIR_PNG, DIR_PDF, DIR_EPS, theme
        )

        # Mapa de calor
        print("\n  [fig14] Mapa de calor...")
        plot_heatmap(
            equilibria, radii, decisions,
            DIR_PNG, DIR_PDF, DIR_EPS, theme
        )

    print("\n" + "="*60)
    print("  DONE -- Archivos guardados en:")
    print(f"    PNG  -> {DIR_PNG}")
    print(f"    PDF  -> {DIR_PDF}")
    print(f"    EPS  -> {DIR_EPS}")
    print("="*60)


if __name__ == "__main__":
    main()
