#!/usr/bin/env python3
"""
Paso 5: Resumen Final y Galería de Figuras
==========================================
Lee los resultados generados por los pasos anteriores y produce:

  1. Tabla Markdown de resumen de todos los candidatos con su veredicto
  2. Galería completa de atractores (figura de 4 paneles por candidato)
  3. Figura comparativa: atractor sesgado vs referencia centrada
  4. Figura MEGA-resumen: todos los candidatos en mosaico 3D
  5. Reporte final de ocultedad en Markdown + JSON

Usa el mismo estilo visual oscuro (tema plasma) del reporte.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
EXAMPLE_DIR = Path(__file__).resolve().parent
VERSION2    = EXAMPLE_DIR.parents[1]
ROOT        = VERSION2.parent

for p in [str(VERSION2), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml

CFG_PATH = VERSION2 / "configs" / "examples" / "chua_nonsmooth_biased_df_search.yaml"


def load_config() -> Dict[str, Any]:
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Estilo visual del reporte ──────────────────────────────────────────────────
BG    = "#0D0D1A"
PANEL = "#141428"
TEXT  = "#E8E8FF"
TICK  = "#9999BB"
GRID  = "#1E1E3A"
CMAP  = plt.cm.plasma


def _ax_dark(ax, grid=True):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TICK, labelsize=7)
    for s in ax.spines.values():
        s.set_edgecolor("#2A2A50")
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    if grid:
        ax.grid(True, color=GRID, lw=0.4, ls="--", alpha=0.7)


def _ax3d_dark(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TICK, labelsize=6)
    for attr in ["xaxis", "yaxis", "zaxis"]:
        getattr(ax, attr).label.set_color(TEXT)
        getattr(ax, attr).pane.fill = False
        getattr(ax, attr).pane.set_edgecolor("#1E1E3A")
    ax.grid(True, color=GRID, lw=0.3, ls="--", alpha=0.5)


# ── Carga de trayectoria ──────────────────────────────────────────────────────

def load_traj(traj_path: Path, t_burn: float = 0.0,
              max_pts: int = 15000) -> Optional[np.ndarray]:
    """Carga trayectoria CSV (t, x, y, z) y descarta el transiente."""
    if not traj_path.exists():
        return None
    df = pd.read_csv(traj_path)
    if t_burn > 0:
        df = df[df["t"] >= t_burn]
    data = df[["t", "x", "y", "z"]].values
    stride = max(1, len(data) // max_pts)
    return data[::stride]


# ── Figura de 7 paneles (estilo reporte) ─────────────────────────────────────

def plot_candidate_report(traj: np.ndarray, title: str,
                           params_str: str, verdict: str,
                           outpath: Path, h: float = 0.01) -> None:
    """Figura completa: 3D + 2D proyecciones + series de tiempo + FFT + info."""
    from hidden_attractors.analysis.spectral import fft_spectrum

    states = traj[:, 1:4]
    times  = traj[:, 0]
    n      = len(states)
    if n < 10:
        return

    h_est = float(np.median(np.diff(times))) if len(times) > 1 else h

    fig = plt.figure(figsize=(17, 9.5), facecolor=BG)
    fig.suptitle(title, color=TEXT, fontsize=10.5, fontweight="bold", y=0.99, va="top")
    gs = GridSpec(3, 4, figure=fig, left=0.055, right=0.97, top=0.90,
                  bottom=0.07, hspace=0.55, wspace=0.38)

    # 3D
    ax3 = fig.add_subplot(gs[:2, :2], projection="3d")
    for i in range(n - 1):
        ax3.plot(states[i:i+2, 0], states[i:i+2, 1], states[i:i+2, 2],
                 lw=0.35, color=CMAP(i / n), alpha=0.75)
    _ax3d_dark(ax3)
    ax3.set_xlabel("x", labelpad=2); ax3.set_ylabel("y", labelpad=2); ax3.set_zlabel("z", labelpad=2)
    ax3.set_title("Espacio de fase 3D", color=TEXT, fontsize=9, pad=5)

    # Proyecciones
    for xl, yl, ix, iy, gss in [("x", "y", 0, 1, gs[0, 2]), ("x", "z", 0, 2, gs[0, 3]),
                                  ("y", "z", 1, 2, gs[1, 2])]:
        ax2 = fig.add_subplot(gss)
        ax2.plot(states[:, ix], states[:, iy], lw=0.3, color="#E64B35", alpha=0.8)
        _ax_dark(ax2)
        ax2.set_xlabel(xl, fontsize=8); ax2.set_ylabel(yl, fontsize=8)
        ax2.set_title(f"Proy. {xl}-{yl}", color=TEXT, fontsize=8)

    # FFT
    ax_fft = fig.add_subplot(gs[1, 3])
    try:
        spec = fft_spectrum(states[:, 0], h_est)
        if spec.frequency_hz.size > 0:
            ax_fft.plot(spec.frequency_rad_s, spec.values, lw=1.0, color="#4DBBD5")
            dom_idx = np.argmax(spec.values)
            ax_fft.axvline(spec.frequency_rad_s[dom_idx], color="#E64B35", ls="--", alpha=0.7,
                           label=f"ω₀={spec.frequency_rad_s[dom_idx]:.3f}")
            ax_fft.legend(fontsize=7)
    except Exception:
        pass
    _ax_dark(ax_fft)
    ax_fft.set_xlabel("ω (rad/s)", fontsize=8); ax_fft.set_ylabel("Amplitud", fontsize=8)
    ax_fft.set_title("Espectro FFT x(t)", color=TEXT, fontsize=8)

    # Series de tiempo
    for col_i, (lbl, sig, clr) in enumerate(
        [("x(t)", states[:, 0], "#E64B35"),
         ("y(t)", states[:, 1], "#4DBBD5"),
         ("z(t)", states[:, 2], "#00A087")]
    ):
        ax_t = fig.add_subplot(gs[2, col_i])
        ax_t.plot(times, sig, lw=0.35, color=clr)
        _ax_dark(ax_t)
        ax_t.set_xlabel("t [s]", fontsize=8); ax_t.set_ylabel(lbl, fontsize=8)
        ax_t.set_title(f"Serie {lbl}", color=TEXT, fontsize=8)

    # Panel de info
    ax_i = fig.add_subplot(gs[2, 3])
    ax_i.set_facecolor(PANEL); ax_i.axis("off")
    centroid = np.mean(states, axis=0)
    info = [
        "Modelo: Chua No Suave (Saturación)",
        "q = 0.9998  |  Caputo ABM",
        "",
        *[f"  {p.strip()}" for p in params_str.split("|")],
        "",
        f"Veredicto: {verdict}",
        f"Puntos: {n:,}",
        f"Centroide: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})",
    ]
    ax_i.text(0.04, 0.97, "\n".join(info), transform=ax_i.transAxes,
              va="top", ha="left", color=TEXT, fontsize=7, fontfamily="monospace", linespacing=1.5)
    ax_i.set_title("Parámetros", color=TEXT, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# ── Figura comparativa sesgado vs centrado ────────────────────────────────────

def plot_biased_vs_centered(biased_traj: np.ndarray,
                             centered_traj: Optional[np.ndarray],
                             params_str: str, outpath: Path) -> None:
    """Overlay 3D y FFT del atractor sesgado vs referencia centrada."""
    fig = plt.figure(figsize=(16, 7), facecolor=BG)
    fig.suptitle(
        f"Comparación: Atractor Sesgado vs Referencia Centrada\n{params_str}",
        color=TEXT, fontsize=10, fontweight="bold", y=0.98,
    )

    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.set_facecolor(PANEL)
    states_b = biased_traj[:, 1:4]
    n = len(states_b)
    for i in range(n - 1):
        ax3.plot(states_b[i:i+2, 0], states_b[i:i+2, 1], states_b[i:i+2, 2],
                 lw=0.3, color="#4DBBD5", alpha=0.7)
    cent_b = np.mean(states_b, axis=0)
    ax3.scatter(*cent_b, color="#4DBBD5", s=80, marker="o", label="Sesgado (c≠0)")

    if centered_traj is not None and len(centered_traj) > 0:
        states_c = centered_traj[:, 1:4]
        nc = len(states_c)
        for i in range(nc - 1):
            ax3.plot(states_c[i:i+2, 0], states_c[i:i+2, 1], states_c[i:i+2, 2],
                     lw=0.3, color="#E64B35", alpha=0.6, linestyle="--")
        cent_c = np.mean(states_c, axis=0)
        ax3.scatter(*cent_c, color="#E64B35", s=80, marker="X", label="Centrado (c=0)")

    _ax3d_dark(ax3)
    ax3.set_xlabel("x", labelpad=2); ax3.set_ylabel("y", labelpad=2); ax3.set_zlabel("z", labelpad=2)
    ax3.set_title("Espacio de fase 3D", color=TEXT, fontsize=9)
    ax3.legend(fontsize=8)

    # FFT comparativa
    ax_fft = fig.add_subplot(1, 2, 2)
    ax_fft.set_facecolor(PANEL)
    h_est = float(np.median(np.diff(biased_traj[:, 0]))) if len(biased_traj) > 1 else 0.01
    try:
        from hidden_attractors.analysis.spectral import fft_spectrum
        spec_b = fft_spectrum(biased_traj[:, 1], h_est)
        if spec_b.frequency_hz.size > 0:
            ax_fft.plot(spec_b.frequency_rad_s, spec_b.values, color="#4DBBD5", lw=1.0, label="Sesgado")
        if centered_traj is not None and len(centered_traj) > 0:
            h_c = float(np.median(np.diff(centered_traj[:, 0])))
            spec_c = fft_spectrum(centered_traj[:, 1], h_c)
            if spec_c.frequency_hz.size > 0:
                ax_fft.plot(spec_c.frequency_rad_s, spec_c.values, color="#E64B35",
                            lw=1.0, linestyle="--", label="Centrado")
    except Exception:
        pass
    _ax_dark(ax_fft)
    ax_fft.set_xlabel("ω (rad/s)", fontsize=9, color=TEXT)
    ax_fft.set_ylabel("Amplitud", fontsize=9, color=TEXT)
    ax_fft.set_title("Comparación de Espectros FFT", color=TEXT, fontsize=9)
    ax_fft.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# ── Figura MEGA-resumen: todos los candidatos ─────────────────────────────────

def plot_mega_summary(candidates: List[Dict], outpath: Path) -> None:
    """Mosaico 3D de todos los candidatos que sobrevivieron."""
    valid = [(c, c["traj"]) for c in candidates if c["traj"] is not None]
    if not valid:
        return

    n_all  = len(valid)
    ncols  = min(3, n_all)
    nrows  = (n_all + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 5, nrows * 4.5 + 1), facecolor=BG)
    fig.suptitle(
        "Atractores Ocultos — Chua Fraccionario No Suave (q=0.9998)  |  DF Sesgada",
        color=TEXT, fontsize=12, fontweight="bold", y=0.99,
    )

    for idx, (cand, traj) in enumerate(valid):
        states = traj[:, 1:4]
        n = len(states)
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        for i in range(n - 1):
            ax.plot(states[i:i+2, 0], states[i:i+2, 1], states[i:i+2, 2],
                    lw=0.3, color=CMAP(i / n), alpha=0.8)
        _ax3d_dark(ax)
        ax.set_xlabel("x", fontsize=5, labelpad=0)
        ax.set_ylabel("y", fontsize=5, labelpad=0)
        ax.set_zlabel("z", fontsize=5, labelpad=0)
        ax.set_title(
            f"★ m1={cand['m1']} | m0={cand['m0']}\nc={cand.get('c', 0):.3f}  {cand.get('verdict', '')}",
            color=TEXT, fontsize=7, pad=2,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# ── Reporte Markdown ──────────────────────────────────────────────────────────

def write_markdown_report(classif_rows: List[Dict], hid_results: List[Dict],
                           outpath: Path) -> None:
    """Escribe el reporte Markdown del experimento completo."""
    lines = [
        "# Atractor Oculto en Chua Fraccionario No Suave — Reporte Final\n",
        "> [!NOTE]",
        "> Primer ejemplo exitoso de la librería `hidden_attractors_fo`.",
        "> Los resultados son reproducibles ejecutando `run_example.py`.\n",
        "## Parámetros del Sistema\n",
        "| Parámetro | Valor |",
        "|---|---|",
        "| Sistema | Chua No Suave (Saturación bilineal) |",
        "| Orden fraccionario q | 0.9998 |",
        "| α (alpha) | 8.4562 |",
        "| β (beta) | 12.0732 |",
        "| γ (gamma) | 0.0052 |",
        "| Integrador | Caputo ABM memoria completa |",
        "| h | 0.01 s |",
        "",
        "## Candidatos Encontrados (Paso 2)\n",
        "| m1 | m0 | branch | c (bias DC) | Clasificación |",
        "|---|---|---|---|---|",
    ]

    for r in classif_rows:
        lines.append(
            f"| {r.get('m1','?')} | {r.get('m0','?')} | {r.get('branch','?')}"
            f"| {r.get('c', r.get('centroid_x','?'))} | **{r.get('classification','?')}** |"
        )

    lines += [
        "",
        "## Resultados de Ocultedad (Paso 3)\n",
        "> [!WARNING]",
        "> La ausencia de contacto con las vecindades ensayadas **NO constituye**",
        "> prueba matemática global de ocultedad, sino verificación numérica finita.\n",
        "| Candidato | m1 | m0 | c | Estado | Compatible | TARGET hits | Muestras |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in hid_results:
        icon = "✅" if r.get("hidden_compatible") else "❌"
        lines.append(
            f"| `{r.get('prefix','?')}` | {r.get('m1','?')} | {r.get('m0','?')}"
            f"| {r.get('c', 0):.3f} | `{r.get('hiddenness_status','?')}` | {icon}"
            f"| {r.get('target_hits', '?')} | {r.get('samples_total', '?')} |"
        )

    lines += [
        "",
        "## Proceso Metodológico\n",
        "1. **Función Descriptiva Centrada (c=0)**: Búsqueda de ramas de la DF estándar.",
        "   Solo produce atractores periódicos (autoexcitados o sin interés).",
        "2. **Función Descriptiva Sesgada (c≠0)**: Extensión al caso con bias DC.",
        "   Convención de signo: `1 + Wq(jω) · N₁(A,c) = 0`.",
        "3. **Reconstrucción algebraica de semilla**: x̄ = −P⁻¹bψ₀, X₁ fasorial.",
        "4. **Verificación de identidad homotópica**: f_{η=1} ≡ f_original.",
        "5. **Continuación afín Caputo ABM**: deforma gradualmente el sistema desde",
        "   el linealizado (η=0) hasta el original (η=1).",
        "6. **Simulación final**: integración larga con el sistema original.",
        "7. **Verificación de ocultedad**: barrido de esferas alrededor de todos",
        "   los equilibrios con 225 muestras/equilibrio × 6 radios.",
        "8. **Test extendido**: verificación masiva hasta r=2.0 con multiprocessing.",
    ]

    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Reporte Markdown: {outpath}")


# ── Runner principal ──────────────────────────────────────────────────────────

def run_summarize_and_plot(cfg: Dict[str, Any]) -> None:
    """Lee todos los resultados previos y genera la galería y el reporte."""
    plot_cfg = cfg["plots"]
    dpi      = int(plot_cfg.get("dpi", 150))
    out_root = ROOT / cfg["experiment"]["output_dir"]
    out_s5   = out_root / "step5_summary"
    out_s5.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("PASO 5 — Resumen y Galería de Figuras")
    print("=" * 65)

    # ── Leer clasificación del Paso 2 ────────────────────────────────────────
    classif_csv = out_root / "step2_biased_df" / "final_classification.csv"
    classif_rows = pd.read_csv(classif_csv).to_dict("records") if classif_csv.exists() else []

    # ── Leer resultados de ocultedad del Paso 3 ───────────────────────────────
    hid_json = out_root / "step3_hiddenness" / "hiddenness_global_summary.json"
    hid_results = json.loads(hid_json.read_text(encoding="utf-8")) if hid_json.exists() else []

    # ── Cargar trayectorias del Paso 2 ────────────────────────────────────────
    traj_dir_s2 = out_root / "step2_biased_df" / "trajectories"
    candidates = []
    for row in classif_rows:
        if "failed" in row.get("classification", ""):
            continue
        prefix    = row.get("prefix", "")
        traj_path = traj_dir_s2 / f"{prefix}_trajectory.csv"
        traj      = load_traj(traj_path)
        candidates.append({
            **row,
            "traj":      traj,
            "traj_path": traj_path,
        })

    # ── Cargar trayectoria centrada del Paso 1 ────────────────────────────────
    centered_dir = out_root / "step1_centered" / "trajectories"
    centered_traj = None
    if centered_dir.exists():
        c_files = list(centered_dir.glob("*_trajectory.csv"))
        if c_files:
            centered_traj = load_traj(c_files[0])

    # ── Galería: figura de reporte por candidato ──────────────────────────────
    if plot_cfg["save_figures"]:
        print("\n  Generando galería de figuras …")
        for cand in candidates:
            if cand["traj"] is None:
                print(f"    [SKIP] {cand['prefix']} — trayectoria no encontrada")
                continue
            m1      = cand.get("m1", "?")
            m0      = cand.get("m0", "?")
            c_val   = cand.get("c", cand.get("centroid_x", "?"))
            verdict = cand.get("classification", "?")
            p_str   = f"m1={m1} | m0={m0} | c≈{c_val:.3f}" if isinstance(c_val, float) else f"m1={m1} | m0={m0}"
            title   = f"[Chua No Suave  |  DF Sesgada  |  q=0.9998]\n{p_str}"
            outpath = out_s5 / "gallery" / f"{cand['prefix']}_detailed.png"
            plot_candidate_report(cand["traj"], title, p_str, verdict, outpath)
            print(f"    {outpath.name}")

    # ── Comparativa sesgado vs centrado ───────────────────────────────────────
    if plot_cfg["save_figures"] and plot_cfg.get("comparison_biased_centered", True):
        for cand in candidates:
            if cand["traj"] is None:
                continue
            m1 = cand.get("m1", "?")
            m0 = cand.get("m0", "?")
            p_str = f"m1={m1} | m0={m0}"
            plot_biased_vs_centered(
                biased_traj=cand["traj"],
                centered_traj=centered_traj,
                params_str=p_str,
                outpath=out_s5 / "comparisons" / f"{cand['prefix']}_vs_centered.png",
            )

    # ── MEGA-resumen ──────────────────────────────────────────────────────────
    if plot_cfg["save_figures"]:
        plot_mega_summary(candidates, out_s5 / "MEGA_all_candidates.png")
        print(f"\n  MEGA-resumen: {out_s5 / 'MEGA_all_candidates.png'}")

    # ── Reporte Markdown ──────────────────────────────────────────────────────
    write_markdown_report(classif_rows, hid_results, out_s5 / "final_report.md")

    # ── JSON completo ─────────────────────────────────────────────────────────
    summary = {
        "experiment": cfg["experiment"],
        "candidates_found":    len(classif_rows),
        "candidates_survived": len(candidates),
        "hiddenness_results":  hid_results,
    }
    (out_s5 / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n[PASO 5 COMPLETADO]")
    print(f"  Directorio: {out_s5}")
    print(f"  Candidatos graficados: {len(candidates)}")
    print(f"  Reporte: {out_s5 / 'final_report.md'}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    run_summarize_and_plot(cfg)
