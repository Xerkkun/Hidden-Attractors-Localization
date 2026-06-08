#!/usr/bin/env python3
"""
Paso 3: Verificación de Ocultedad — Protocolo Estándar
=======================================================
Verifica la ocultedad de los candidatos que sobrevivieron la continuación
afín usando el contrato formal de la librería.

Protocolo:
  - Para cada candidato sesgado: cargar la trayectoria de referencia del Paso 2
  - Resolver los equilibrios E₀, E₊, E₋ del sistema
  - Para cada equilibrio, para cada radio de prueba r ∈ {1e-5, …, 1e-2}:
      * Generar n_samples puntos aleatorios en la superficie de la esfera ||x - Eᵢ|| = r
      * Integrar cada punto durante t_final_probe segundos con Caputo ABM
      * Clasificar el destino: atractor objetivo / equilibrio / divergencia / otro
  - Evaluar el contrato: B(A) ∩ U_ε(Eᵢ) = ∅ para todo i y todo r ensayado
  - Exportar JSON del contrato + CSV de resultados + figuras 3D de esferas

Nota sobre la interpretación:
  HIDDEN_COMPATIBLE significa que ninguna de las muestras ensayadas alcanzó
  el atractor objetivo partiendo de vecindades del equilibrio. Esto es una
  verificación numérica finita bajo los parámetros declarados, NO una prueba
  matemática global de ocultedad.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.systems import get_system
from hidden_attractors.verification.equilibria import solve_equilibria
from hidden_attractors.verification.hiddenness import (
    evaluate_target_match,
    generate_neighborhood_points,
)
from hidden_attractors.verification.hiddenness_contract import verify_hiddenness_contract

CFG_PATH = VERSION2 / "configs" / "examples" / "chua_nonsmooth_biased_df_search.yaml"


def load_config() -> Dict[str, Any]:
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Plotting (estilo de la librería) ─────────────────────────────────────────
COLOUR_MAP = {
    "target_attractor":   "#ef4444",
    "stable_equilibrium": "#3b82f6",
    "divergence":         "#f59e0b",
    "other_attractor":    "#8b5cf6",
    "numerical_failure":  "#94a3b8",
}


def plot_sphere_summary(eq_name: str, eq_pt: np.ndarray, radius: float,
                         runs: List[Dict], pts: np.ndarray, outpath: Path) -> None:
    """Figura 3D: puntos de sonda coloreados por destino."""
    fig = plt.figure(figsize=(7, 6), dpi=300)
    ax  = fig.add_subplot(111, projection="3d")
    for res, pt in zip(runs, pts):
        ax.scatter(*pt, color=COLOUR_MAP.get(res["destination"], "#9ca3af"),
                   s=8, alpha=0.6)
    ax.scatter(*eq_pt, color="black", marker="x", s=60, zorder=10, label=eq_name)
    ax.set_xlabel("x", fontsize=10); ax.set_ylabel("y", fontsize=10)
    ax.set_zlabel("z", fontsize=10)
    ax.set_title(f"{eq_name}  r={radius:.0e}", fontsize=11, fontweight='bold')
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=c, label=d, markersize=7)
               for d, c in COLOUR_MAP.items()]
    ax.legend(handles=handles, fontsize=6, loc="best")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_heatmap_hiddenness(records: List[Dict], radii: List[float],
                             outpath: Path) -> None:
    """Heatmap: fracción de TARGET hits por equilibrio × radio."""
    eq_names  = sorted({r["equilibrium"] for r in records})
    rad_strs  = [f"{r:.0e}" for r in radii]
    data      = np.full((len(eq_names), len(radii)), np.nan)
    counts    = np.zeros_like(data, dtype=int)

    for rec in records:
        try:
            i = eq_names.index(rec["equilibrium"])
        except ValueError:
            continue
        for j, rad in enumerate(radii):
            if abs(rec["radius"] - rad) < rad * 0.01:
                data[i, j]   = rec["TARGET"] / max(rec["samples"], 1)
                counts[i, j] = rec["samples"]
                break

    fig, ax = plt.subplots(figsize=(12, max(3, len(eq_names) * 1.2)))
    masked = np.ma.array(data, mask=np.isnan(data))
    im = ax.imshow(masked, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(radii))); ax.set_xticklabels(rad_strs, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(eq_names))); ax.set_yticklabels(eq_names, fontsize=9)
    ax.set_xlabel("Radio de esfera", fontsize=9)
    ax.set_ylabel("Equilibrio", fontsize=9)
    ax.set_title("Fracción TARGET hits\nverde=0 (compatible con ocultedad)  rojo=1 (autoexcitado)",
                 fontsize=9)
    plt.colorbar(im, ax=ax, label="fracción TARGET")
    for i in range(len(eq_names)):
        for j in range(len(radii)):
            if not np.isnan(data[i, j]):
                t_txt = int(data[i, j] * counts[i, j])
                txt   = f"{data[i, j]:.2f}\n({t_txt}/{counts[i,j]})"
                col   = "white" if data[i, j] > 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=col)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


# ── Integración de sonda ──────────────────────────────────────────────────────

def run_probe(system: Any, x0: np.ndarray, ref_tail: np.ndarray,
              stable_eqs: List[np.ndarray], h3_cfg: Dict) -> Dict[str, Any]:
    """Integra una sonda y clasifica su destino."""
    q      = float(h3_cfg["_q"])
    h      = float(h3_cfg["_h"])
    t_fin  = float(h3_cfg["t_final_probe"])
    t_burn = float(h3_cfg["t_burn_probe"])
    eq_tol = float(h3_cfg["equilibrium_tol"])
    metric = h3_cfg["match_metric"]
    m_tol  = float(h3_cfg["match_tol"])

    try:
        t_arr, x_arr, status, _ = fractional_integrate(
            rhs=lambda t, x: system.rhs(x, system.parameters),
            x0=x0, q=q, h=h, t_final=t_fin,
            method="abm", memory_mode="full",
            system=system, use_c_backend=True,
        )
    except Exception as exc:
        return {"destination": "numerical_failure", "status": f"exception:{exc}"}

    if status in ("diverged", "nonfinite_solution"):
        return {"destination": "divergence", "status": status}

    n_burn = int(np.ceil(t_burn / h))
    tail   = x_arr[n_burn:] if len(x_arr) > n_burn else x_arr
    final  = x_arr[-1] if len(x_arr) > 0 else x0

    for eq in stable_eqs:
        if np.linalg.norm(final - eq) <= eq_tol:
            return {"destination": "stable_equilibrium", "status": "ok"}

    if evaluate_target_match(tail, ref_tail, metric=metric, tolerance=m_tol):
        return {"destination": "target_attractor", "status": "ok"}

    return {"destination": "other_attractor", "status": "ok"}


# ── Runner por candidato ──────────────────────────────────────────────────────

def run_hiddenness_for_candidate(candidate: Dict[str, Any],
                                  cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ejecuta el protocolo completo de ocultedad para un candidato."""
    sys_cfg = cfg["system"]
    int_cfg = cfg["integrator"]
    h3_cfg  = cfg["step3_hiddenness"].copy()
    h3_cfg["_q"] = float(sys_cfg["q"])
    h3_cfg["_h"] = float(int_cfg["h"])
    plot_cfg = cfg["plots"]

    alpha = float(sys_cfg["parameters"]["alpha"])
    beta  = float(sys_cfg["parameters"]["beta"])
    gamma = float(sys_cfg["parameters"]["gamma"])
    m1    = float(candidate["m1"])
    m0    = float(candidate["m0"])
    prefix = str(candidate["prefix"])
    traj_path = Path(candidate["traj_path"])

    out_root = ROOT / cfg["experiment"]["output_dir"]
    outdir   = out_root / "step3_hiddenness" / prefix
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [{prefix}]  m1={m1}  m0={m0}  c={candidate.get('c','?')}")

    # 1. Referencia
    df_ref   = pd.read_csv(traj_path)
    t_burn   = float(h3_cfg["t_burn_probe"])
    ref_tail = df_ref[df_ref["t"] >= t_burn][["x", "y", "z"]].values
    print(f"    Referencia: {len(ref_tail)} puntos post-transitorio")
    if len(ref_tail) < int(h3_cfg.get("min_ref_tail_points", 200)):
        print("    [WARN] Trayectoria de referencia muy corta")

    # 2. Sistema y equilibrios
    system = get_system("chua-nonsmooth")
    system.parameters.update({"m1": m1, "m0": m0, "alpha": alpha, "beta": beta, "gamma": gamma})
    equilibria = solve_equilibria(system)
    stable_eqs = list(equilibria.values())
    print(f"    Equilibrios: {list(equilibria.keys())}")

    radii          = [float(r) for r in h3_cfg["radii"]]
    samples_per_r  = [int(s) for s in h3_cfg["samples_per_radius"]]
    random_seed    = int(cfg["experiment"]["random_seed"])

    all_runs     : List[Dict] = []
    sphere_recs  : List[Dict] = []

    for eq_name, eq_pt in equilibria.items():
        for r_idx, (radius, n_samples) in enumerate(zip(radii, samples_per_r)):
            print(f"    [{eq_name}]  r={radius:.0e}  n={n_samples}", end=" … ", flush=True)
            pts = generate_neighborhood_points(
                eq_point=eq_pt, radius=radius, num_samples=n_samples,
                mode=h3_cfg["sampling_mode"], seed=random_seed + r_idx,
            )
            stats = {k: 0 for k in
                     ["target_attractor", "stable_equilibrium", "divergence",
                      "other_attractor", "numerical_failure"]}
            radius_runs: List[Dict] = []

            for pt in pts:
                res = run_probe(system, pt, ref_tail, stable_eqs, h3_cfg)
                radius_runs.append(res)
                stats[res["destination"]] = stats.get(res["destination"], 0) + 1

                # Parada temprana: si hay un hit al objetivo, está autoexcitado
                if stats["target_attractor"] > 0:
                    break

            print(f"TARGET={stats['target_attractor']}  EQ={stats['stable_equilibrium']}  "
                  f"DIV={stats['divergence']}")

            # Figura 3D de esfera
            if plot_cfg["save_figures"]:
                safe_r = f"{radius:.0e}".replace("-", "m")
                plot_sphere_summary(
                    eq_name, eq_pt, radius, radius_runs, pts[:len(radius_runs)],
                    outdir / f"sphere_{eq_name}_{safe_r}.png",
                )

            for res in radius_runs:
                all_runs.append({
                    "equilibrium": eq_name, "radius": float(radius), **res,
                })
            sphere_recs.append({
                "system_id": prefix, "equilibrium": eq_name, "radius": float(radius),
                "samples": len(radius_runs), "TARGET": stats["target_attractor"],
                "EQ": stats["stable_equilibrium"], "OTHER": stats["other_attractor"],
                "DIV": stats["divergence"], "FAIL": stats["numerical_failure"],
            })

    # 3. Contrato de ocultedad
    contract = verify_hiddenness_contract(
        equilibria=equilibria,
        sphere_summary_records=sphere_recs,
        probe_runs=all_runs,
        required_radii=radii,
        require_all_equilibria=bool(h3_cfg.get("require_all_equilibria", True)),
        allow_numerical_failures=bool(h3_cfg.get("allow_numerical_failures", False)),
        ref_tail_size=len(ref_tail),
        min_ref_tail_points=int(h3_cfg.get("min_ref_tail_points", 200)),
        target_match_metric=h3_cfg["match_metric"],
        target_match_tol=float(h3_cfg["match_tol"]),
        target_match_nn_percentile=float(h3_cfg["match_percentile"]),
        seed_reached_attractor=True,
    )

    # 4. Guardar
    pd.DataFrame(sphere_recs).to_csv(outdir / "hiddenness_summary.csv", index=False)
    pd.DataFrame([{"equilibrium": r["equilibrium"], "radius": r["radius"],
                   "destination": r["destination"], "status": r["status"]}
                  for r in all_runs]).to_csv(outdir / "probe_runs.csv", index=False)

    contract_safe = {k: v for k, v in contract.items() if k != "run_metadata"}
    with open(outdir / "hiddenness_contract.json", "w", encoding="utf-8") as f:
        json.dump(contract_safe, f, indent=2)

    # 5. Heatmap
    if plot_cfg["save_figures"] and plot_cfg.get("heatmap_hiddenness", True):
        plot_heatmap_hiddenness(sphere_recs, radii, outdir / "heatmap_hiddenness.png")

    # 6. Veredicto
    status = contract["hiddenness_status"]
    print(f"\n    VEREDICTO: {status}")
    print(f"    hidden_verified:    {contract['hidden_verified']}")
    print(f"    hidden_compatible:  {contract['hidden_compatible']}")
    print(f"    target_hits_total:  {contract['target_hits_total']}")

    return {
        "prefix": prefix,
        "m1": m1, "m0": m0, "c": candidate.get("c", 0.0),
        "hiddenness_status":  contract["hiddenness_status"],
        "hidden_verified":    contract["hidden_verified"],
        "hidden_compatible":  contract["hidden_compatible"],
        "self_excited":       contract["self_excited_contact_detected"],
        "target_hits":        contract["target_hits_total"],
        "samples_total":      contract["samples_total"],
        "equilibria_tested":  contract["equilibria_tested"],
    }


# ── Runner global ─────────────────────────────────────────────────────────────

def run_hiddenness_verification(candidates: List[Dict[str, Any]],
                                 cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ejecuta la verificación de ocultedad para todos los candidatos."""
    print("=" * 65)
    print("PASO 3 — Verificación de Ocultedad (Protocolo Estándar)")
    print(f"  Radios: {cfg['step3_hiddenness']['radii']}")
    print(f"  Muestras/radio: {cfg['step3_hiddenness']['samples_per_radius']}")
    print(f"  Candidatos a verificar: {len(candidates)}")
    print("=" * 65)
    print(
        "\n  ADVERTENCIA: La ausencia de contacto con las vecindades ensayadas\n"
        "  NO constituye prueba matemática global de ocultedad, sino una\n"
        "  verificación numérica finita bajo los radios y tiempos declarados.\n"
    )

    results = []
    for cand in candidates:
        result = run_hiddenness_for_candidate(cand, cfg)
        results.append(result)

    # Resumen global
    out_root = ROOT / cfg["experiment"]["output_dir"] / "step3_hiddenness"
    out_root.mkdir(parents=True, exist_ok=True)

    with open(out_root / "hiddenness_global_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n[PASO 3 COMPLETADO]")
    for r in results:
        icon = "✓" if r["hidden_compatible"] else "✗"
        print(f"  {icon} m1={r['m1']}  m0={r['m0']}  c={r['c']:.3f}"
              f"  → {r['hiddenness_status']}  (hits={r['target_hits']})")

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()

    # Los candidatos por defecto son los 3 encontrados en el proceso histórico.
    # En el uso del orquestador run_example.py, esta lista viene del Paso 2.
    out_root = ROOT / cfg["experiment"]["output_dir"] / "step2_biased_df"
    traj_dir = out_root / "trajectories"

    KNOWN_CANDIDATES = [
        {
            "m1": -1.1468, "m0": -0.1768, "branch": 1, "c": 2.776,
            "prefix": "biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776",
            "traj_path": str(traj_dir / "biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776_trajectory.csv"),
        },
        {
            "m1": -1.1468, "m0": -0.200, "branch": 1, "c": -2.705,
            "prefix": "biased_q9998_m1_m1p1468_m0_m0p2000_branch_1_c_m2p705",
            "traj_path": str(traj_dir / "biased_q9998_m1_m1p1468_m0_m0p2000_branch_1_c_m2p705_trajectory.csv"),
        },
        {
            "m1": -1.1468, "m0": -0.240, "branch": 1, "c": -2.581,
            "prefix": "biased_q9998_m1_m1p1468_m0_m0p2400_branch_1_c_m2p581",
            "traj_path": str(traj_dir / "biased_q9998_m1_m1p1468_m0_m0p2400_branch_1_c_m2p581_trajectory.csv"),
        },
    ]

    valid = [c for c in KNOWN_CANDIDATES if Path(c["traj_path"]).exists()]
    if not valid:
        print("No se encontraron trayectorias del Paso 2. Ejecute step2_biased_df_search.py primero.")
        sys.exit(1)

    run_hiddenness_verification(valid, cfg)
