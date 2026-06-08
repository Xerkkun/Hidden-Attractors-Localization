#!/usr/bin/env python3
"""Hiddenness tests for genuine biased (|c| > 0.05) candidates that survived
continuation in the corrected biased DF sweep.

Targets (3 candidates):
  - m1=-1.1468, m0=-0.1768, branch 1, c=+2.776  (periodic)
  - m1=-1.1468, m0=-0.2000, branch 1, c=-2.705   (periodic)
  - m1=-1.1468, m0=-0.2400, branch 1, c=-2.581   (periodic)

Protocol:
  - Equilibria:   E0, E+, E- (solved analytically per parameters)
  - Radii:        1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2
  - Samples/rad:  20, 25, 30, 40, 50, 60  (225 per equilibrium, ~675 per candidate)
  - Integrator:   Caputo ABM full-memory  (same as continuation)
  - t_final:      200 s,  t_burn: 50 s
  - Match metric: nn_percentile (90th perc, tol=0.5)

Output: outputs/biased_saturation_search_q09998_corrected/hiddenness/<prefix>/
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.models.chua import chua_parameters
from hidden_attractors.systems import get_system
from hidden_attractors.verification.equilibria import solve_equilibria
from hidden_attractors.verification.hiddenness import (
    evaluate_target_match,
    generate_neighborhood_points,
)
from hidden_attractors.verification.hiddenness_contract import (
    verify_hiddenness_contract,
)

# ── Parameters ─────────────────────────────────────────────────────────────────

Q = 0.9998
H = 0.01
ALPHA = 8.4562
BETA  = 12.0732
GAMMA = 0.0052

CORRECTED_OUTDIR = Path("outputs/biased_saturation_search_q09998_corrected")
TRAJ_DIR = CORRECTED_OUTDIR / "trajectories"
HID_BASE  = CORRECTED_OUTDIR / "hiddenness"

RADII           = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2]
SAMPLES_PER_R   = [20,   25,   30,   40,   50,   60  ]  # 225 per equilibrium
T_FINAL_PROBE   = 200.0
T_BURN_PROBE    = 50.0
DIVERGENCE_NORM = 120.0
EQUILIBRIUM_TOL = 0.5
MATCH_METRIC    = "nn_percentile"
MATCH_TOL       = 0.5
MATCH_PERC      = 90.0
RANDOM_SEED     = 42

# Candidates: (m1, m0, branch, traj_filename)
CANDIDATES = [
    {
        "m1": -1.1468, "m0": -0.1768, "branch": 1, "c": 2.776,
        "prefix": "biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776",
        "traj_file": "biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776_trajectory.csv",
    },
    {
        "m1": -1.1468, "m0": -0.200, "branch": 1, "c": -2.705,
        "prefix": "biased_q9998_m1_m1p1468_m0_m0p2000_branch_1_c_m2p705",
        "traj_file": "biased_q9998_m1_m1p1468_m0_m0p2000_branch_1_c_m2p705_trajectory.csv",
    },
    {
        "m1": -1.1468, "m0": -0.240, "branch": 1, "c": -2.581,
        "prefix": "biased_q9998_m1_m1p1468_m0_m0p2400_branch_1_c_m2p581",
        "traj_file": "biased_q9998_m1_m1p1468_m0_m0p2400_branch_1_c_m2p581_trajectory.csv",
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_ref_tail(traj_file: Path, t_burn: float) -> np.ndarray:
    """Load post-transient portion of a saved trajectory CSV."""
    df = pd.read_csv(traj_file)
    post = df[df["t"] >= t_burn]
    return post[["x", "y", "z"]].values


def solve_chua_equilibria(m1: float, m0: float) -> Dict[str, np.ndarray]:
    """Solve Chua non-smooth equilibria analytically."""
    system = get_system("chua-nonsmooth")
    system.parameters["m1"] = m1
    system.parameters["m0"] = m0
    system.parameters["alpha"] = ALPHA
    system.parameters["beta"]  = BETA
    system.parameters["gamma"] = GAMMA
    return solve_equilibria(system)


def run_probe(
    system: Any,
    x0: np.ndarray,
    ref_tail: np.ndarray,
    stable_eqs: List[np.ndarray],
) -> Dict[str, Any]:
    """Integrate a single neighborhood probe and classify its destination."""
    try:
        t_arr, x_arr, status, _ = fractional_integrate(
            rhs=lambda t, x: system.rhs(x, system.parameters),
            x0=x0,
            q=Q,
            h=H,
            t_final=T_FINAL_PROBE,
            method="abm",
            memory_mode="full",
            system=system,
            use_c_backend=True,
        )
    except Exception as exc:
        return {"destination": "numerical_failure", "status": f"exception:{exc}", "trajectory": np.empty((0, 3))}

    if status in ("diverged", "nonfinite_solution"):
        return {"destination": "divergence", "status": status, "trajectory": x_arr}

    n_burn = int(np.ceil(T_BURN_PROBE / H))
    tail = x_arr[n_burn:] if len(x_arr) > n_burn else x_arr

    final = x_arr[-1] if len(x_arr) > 0 else x0
    for eq in stable_eqs:
        if np.linalg.norm(final - eq) <= EQUILIBRIUM_TOL:
            return {"destination": "stable_equilibrium", "status": "ok", "trajectory": x_arr}

    if evaluate_target_match(tail, ref_tail, metric=MATCH_METRIC, tolerance=MATCH_TOL):
        return {"destination": "target_attractor", "status": "ok", "trajectory": x_arr}

    return {"destination": "other_attractor", "status": "ok", "trajectory": x_arr}


def plot_sphere_summary(
    eq_name: str,
    eq_pt: np.ndarray,
    radius: float,
    runs: List[Dict[str, Any]],
    pts: np.ndarray,
    outdir: Path,
    prefix: str,
) -> None:
    """3D scatter plot of probe initial conditions coloured by destination."""
    colour_map = {
        "target_attractor":  "#dc2626",
        "stable_equilibrium": "#111827",
        "divergence":         "#f59e0b",
        "other_attractor":    "#2563eb",
        "numerical_failure":  "#6b7280",
    }
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for run, pt in zip(runs, pts):
        dest = run["destination"]
        c = colour_map.get(dest, "#9ca3af")
        ax.scatter(*pt, color=c, s=8, alpha=0.6)

    ax.scatter(*eq_pt, color="black", marker="x", s=60, zorder=10, label=eq_name)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"{prefix}\n{eq_name} r={radius:.0e}")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                           label=d, markersize=7)
               for d, c in colour_map.items()]
    ax.legend(handles=handles, fontsize=6, loc="best")
    fig.tight_layout()
    safe_r = f"{radius:.0e}".replace("-", "m")
    fig_path = outdir / f"sphere_{eq_name}_{safe_r}.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


# ── Main per-candidate runner ─────────────────────────────────────────────────

def run_hiddenness_for_candidate(cand: Dict[str, Any]) -> Dict[str, Any]:
    m1     = cand["m1"]
    m0     = cand["m0"]
    prefix = cand["prefix"]
    traj_path = TRAJ_DIR / cand["traj_file"]

    print(f"\n{'='*70}")
    print(f"HIDDENNESS TEST: {prefix}")
    print(f"  m1={m1}, m0={m0}, c={cand['c']}")
    print(f"{'='*70}")

    outdir = HID_BASE / prefix
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load reference attractor tail
    ref_tail = load_ref_tail(traj_path, T_BURN_PROBE)
    print(f"  Referencia: {len(ref_tail)} puntos post-transitorio cargados.")
    if len(ref_tail) < 200:
        print("  [WARN] Trayectoria de referencia muy corta — resultados no confiables.")

    # 2. Set up system
    system = get_system("chua-nonsmooth")
    system.parameters["m1"] = m1
    system.parameters["m0"] = m0
    system.parameters["alpha"] = ALPHA
    system.parameters["beta"]  = BETA
    system.parameters["gamma"] = GAMMA

    # 3. Solve equilibria
    equilibria = solve_chua_equilibria(m1, m0)
    stable_eqs = list(equilibria.values())
    print(f"  Equilibrios: {list(equilibria.keys())}")
    for name, pt in equilibria.items():
        print(f"    {name}: {pt}")

    # 4. Run sphere sweeps
    all_probe_runs: List[Dict[str, Any]] = []
    sphere_summary_records: List[Dict[str, Any]] = []
    rng = np.random.default_rng(RANDOM_SEED)

    for eq_name, eq_pt in equilibria.items():
        for r_idx, (radius, n_samples) in enumerate(zip(RADII, SAMPLES_PER_R)):
            print(f"\n  [{eq_name}] radio={radius:.0e}  n={n_samples}")
            pts = generate_neighborhood_points(
                eq_point=eq_pt,
                radius=radius,
                num_samples=n_samples,
                mode="sphere_random",
                seed=RANDOM_SEED + r_idx,
            )
            stats = {"target_attractor": 0, "stable_equilibrium": 0,
                     "divergence": 0, "other_attractor": 0, "numerical_failure": 0}
            radius_runs: List[Dict[str, Any]] = []

            for k, pt in enumerate(pts):
                res = run_probe(system, pt, ref_tail, stable_eqs)
                radius_runs.append(res)
                dest = res["destination"]
                stats[dest] = stats.get(dest, 0) + 1
                if (k + 1) % max(1, n_samples // 4) == 0:
                    print(f"    {k+1}/{n_samples}  TARGET={stats['target_attractor']}  "
                          f"EQ={stats['stable_equilibrium']}  "
                          f"DIV={stats['divergence']}  "
                          f"OTHER={stats['other_attractor']}")

                # Early stop: if any target hit found, no need to continue this radius
                if stats["target_attractor"] > 0:
                    remaining = n_samples - (k + 1)
                    print(f"    *** TARGET HIT — parando pronto (faltan {remaining} muestras)")
                    break

            print(f"    Resumen: {stats}")
            for res in radius_runs:
                all_probe_runs.append({
                    "equilibrium": eq_name,
                    "radius": float(radius),
                    "destination": res["destination"],
                    "status": res["status"],
                    "trajectory": res["trajectory"],
                })
            sphere_summary_records.append({
                "system_id": prefix,
                "equilibrium": eq_name,
                "radius": float(radius),
                "samples": len(radius_runs),
                "TARGET": stats["target_attractor"],
                "EQ":     stats["stable_equilibrium"],
                "OTHER":  stats["other_attractor"],
                "DIV":    stats["divergence"],
                "FAIL":   stats["numerical_failure"],
            })

            # Plot
            plot_sphere_summary(eq_name, eq_pt, radius, radius_runs, pts[:len(radius_runs)], outdir, prefix)

    # 5. Verify hiddenness contract
    contract = verify_hiddenness_contract(
        equilibria=equilibria,
        sphere_summary_records=sphere_summary_records,
        probe_runs=all_probe_runs,
        required_radii=RADII,
        require_all_equilibria=True,
        allow_numerical_failures=False,
        ref_tail_size=len(ref_tail),
        min_ref_tail_points=200,
        target_match_metric=MATCH_METRIC,
        target_match_tol=MATCH_TOL,
        target_match_nn_percentile=MATCH_PERC,
        seed_reached_attractor=True,
    )

    # 6. Save CSV summary
    df_summary = pd.DataFrame(sphere_summary_records)
    df_summary.to_csv(outdir / "hiddenness_summary.csv", index=False)

    # Save probe run destinations (no trajectories)
    probe_rows = [
        {"equilibrium": r["equilibrium"], "radius": r["radius"],
         "destination": r["destination"], "status": r["status"]}
        for r in all_probe_runs
    ]
    pd.DataFrame(probe_rows).to_csv(outdir / "probe_runs.csv", index=False)

    # 7. Save contract JSON
    contract_jsonable = {k: v for k, v in contract.items() if k != "run_metadata"}
    with open(outdir / "hiddenness_contract.json", "w", encoding="utf-8") as f:
        json.dump(contract_jsonable, f, indent=2)

    # 8. Print verdict
    print(f"\n  VEREDICTO: {contract['hiddenness_status']}")
    print(f"  hidden_verified:              {contract['hidden_verified']}")
    print(f"  hidden_compatible:            {contract['hidden_compatible']}")
    print(f"  self_excited_contact:         {contract['self_excited_contact_detected']}")
    print(f"  target_hits_total:            {contract['target_hits_total']}")
    print(f"  samples_total:                {contract['samples_total']}")
    if contract["failed_requirements"]:
        print(f"  failed_requirements:")
        for req in contract["failed_requirements"]:
            print(f"    - {req}")

    return {
        "prefix": prefix,
        "m1": m1, "m0": m0, "c": cand["c"],
        "hiddenness_status": contract["hiddenness_status"],
        "hidden_verified": contract["hidden_verified"],
        "hidden_compatible": contract["hidden_compatible"],
        "self_excited_contact": contract["self_excited_contact_detected"],
        "target_hits": contract["target_hits_total"],
        "samples": contract["samples_total"],
        "equilibria_tested": contract["equilibria_tested"],
    }


# ── Global summary ─────────────────────────────────────────────────────────────

def write_global_summary(results: List[Dict[str, Any]]) -> None:
    outdir = HID_BASE
    outdir.mkdir(parents=True, exist_ok=True)

    # Markdown report
    lines = [
        "# Pruebas de Ocultedad — Candidatos Sesgados No Centrados\n",
        "> [!WARNING]",
        "> **ADVERTENCIA:** La ausencia de contacto con las vecindades ensayadas **no constituye prueba matemática global de ocultedad**, sino una verificación numérica bajo los radios, integrador y tiempos declarados.\n",
        "## Parámetros de la Prueba\n",
        f"- **q** = {Q},  **h** = {H}",
        f"- **Radios:** {RADII}",
        f"- **Muestras por radio:** {SAMPLES_PER_R}",
        f"- **t_final:** {T_FINAL_PROBE} s,  **t_burn:** {T_BURN_PROBE} s",
        f"- **Integrador:** Caputo ABM memoria completa",
        f"- **Métrica de coincidencia:** {MATCH_METRIC} (perc={MATCH_PERC}%, tol={MATCH_TOL})",
        "",
        "## Resultados\n",
        "| Candidato | m1 | m0 | c | Equilibrios | TARGET hits | Total muestras | Estado Contrato | Compatible |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in results:
        eq_str = ", ".join(r["equilibria_tested"])
        lines.append(
            f"| `{r['prefix']}` | {r['m1']} | {r['m0']} | {r['c']:.3f} | "
            f"{eq_str} | **{r['target_hits']}** | {r['samples']} | "
            f"`{r['hiddenness_status']}` | {'✅' if r['hidden_compatible'] else '❌'} |"
        )

    lines += [
        "",
        "## Conclusión Operacional\n",
    ]
    any_hidden = any(r["hidden_compatible"] for r in results)
    any_self_excited = any(r["self_excited_contact"] for r in results)

    if any_self_excited:
        lines.append("Al menos un candidato sesgado mostró **contacto autoexcitado** con la vecindad de un equilibrio.")
    elif any_hidden:
        lines.append("Ningún candidato sesgado mostró contacto con la vecindad de los equilibrios en los radios ensayados. Son **compatibles con la ocultedad** bajo el contrato declarado.")
    else:
        lines.append("Todos los candidatos fallaron los requerimientos del contrato de verificación.")

    report_path = HID_BASE / "hiddenness_global_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Reporte global guardado en: {report_path}")

    # JSON
    json_path = HID_BASE / "hiddenness_global_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON global guardado en: {json_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("PRUEBAS DE OCULTEDAD — CANDIDATOS SESGADOS NO CENTRADOS")
    print("q=0.9998, h=0.01, Caputo ABM memoria completa")
    print("=" * 70)

    HID_BASE.mkdir(parents=True, exist_ok=True)
    all_results = []

    for cand in CANDIDATES:
        result = run_hiddenness_for_candidate(cand)
        all_results.append(result)

    write_global_summary(all_results)

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    for r in all_results:
        status = r["hiddenness_status"]
        hits   = r["target_hits"]
        print(f"  [{r['m1']}, {r['m0']}, c={r['c']:.3f}]  "
              f"TARGET_HITS={hits}  STATUS={status}")

    print("\n[ADVERTENCIA] Estos resultados no constituyen prueba matemática global de ocultedad.")
