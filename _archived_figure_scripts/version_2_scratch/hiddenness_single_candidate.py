#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prueba de ocultedad para UN candidato sesgado (|c| > 0.05).

Uso:
    python hiddenness_single_candidate.py <candidate_index>

donde candidate_index ∈ {0, 1, 2}:
  0 → m1=-1.1468, m0=-0.1768, branch 1, c=+2.776
  1 → m1=-1.1468, m0=-0.2000, branch 1, c=-2.705
  2 → m1=-1.1468, m0=-0.2400, branch 1, c=-2.581

Nota: NO usa parada temprana — todas las muestras de cada radio se integran completas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Force UTF-8 stdout/stderr on Windows (avoids cp1252 UnicodeEncodeError)
import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(_sys.stderr, 'reconfigure'):
    _sys.stderr.reconfigure(encoding='utf-8', errors='replace')

_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.systems import get_system
from hidden_attractors.verification.equilibria import solve_equilibria
from hidden_attractors.verification.hiddenness import (
    evaluate_target_match,
    generate_neighborhood_points,
)
from hidden_attractors.verification.hiddenness_contract import (
    verify_hiddenness_contract,
)

# ── Configuración global ───────────────────────────────────────────────────────

Q     = 0.9998
H     = 0.01
ALPHA = 8.4562
BETA  = 12.0732
GAMMA = 0.0052

CORRECTED_OUTDIR = Path("outputs/biased_saturation_search_q09998_corrected")
TRAJ_DIR         = CORRECTED_OUTDIR / "trajectories"
HID_BASE         = CORRECTED_OUTDIR / "hiddenness"

RADII           = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2]
SAMPLES_PER_R   = [20,   25,   30,   40,   50,   60  ]   # 225 por equilibrio
T_FINAL_PROBE   = 200.0
T_BURN_PROBE    = 50.0
EQUILIBRIUM_TOL = 0.5
MATCH_METRIC    = "nn_percentile"
MATCH_TOL       = 0.5
MATCH_PERC      = 90.0
RANDOM_SEED     = 42

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


# ── Utilidades ─────────────────────────────────────────────────────────────────

def _tag(m1, m0, c):
    return f"[m1={m1} m0={m0} c={c:+.3f}]"


def load_ref_tail(traj_file: Path, t_burn: float) -> np.ndarray:
    df = pd.read_csv(traj_file)
    post = df[df["t"] >= t_burn]
    return post[["x", "y", "z"]].values


def setup_system(m1: float, m0: float):
    sys_ = get_system("chua-nonsmooth")
    sys_.parameters.update({
        "m1": m1, "m0": m0,
        "alpha": ALPHA, "beta": BETA, "gamma": GAMMA,
    })
    return sys_


def get_equilibria(system) -> Dict[str, np.ndarray]:
    return solve_equilibria(system)


def run_probe(
    system,
    x0: np.ndarray,
    ref_tail: np.ndarray,
    stable_eqs: List[np.ndarray],
) -> Dict[str, Any]:
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
        return {"destination": "numerical_failure", "status": f"exception:{exc}"}

    if status in ("diverged", "nonfinite_solution"):
        return {"destination": "divergence", "status": status}

    n_burn = int(np.ceil(T_BURN_PROBE / H))
    tail = x_arr[n_burn:] if len(x_arr) > n_burn else x_arr
    final = x_arr[-1] if len(x_arr) > 0 else x0

    for eq in stable_eqs:
        if np.linalg.norm(final - eq) <= EQUILIBRIUM_TOL:
            return {"destination": "stable_equilibrium", "status": "ok"}

    if evaluate_target_match(tail, ref_tail, metric=MATCH_METRIC, tolerance=MATCH_TOL):
        return {"destination": "target_attractor", "status": "ok"}

    return {"destination": "other_attractor", "status": "ok"}


def plot_sphere_summary(
    eq_name: str,
    eq_pt: np.ndarray,
    radius: float,
    runs: List[Dict[str, Any]],
    pts: np.ndarray,
    outdir: Path,
    prefix: str,
) -> None:
    colour_map = {
        "target_attractor":   "#dc2626",   # rojo  → autoexcitado si desde vecindad del eq
        "stable_equilibrium": "#111827",   # negro
        "divergence":         "#f59e0b",   # naranja
        "other_attractor":    "#2563eb",   # azul
        "numerical_failure":  "#6b7280",   # gris
    }
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for run_res, pt in zip(runs, pts):
        dest = run_res["destination"]
        ax.scatter(*pt, color=colour_map.get(dest, "#9ca3af"), s=8, alpha=0.6)

    ax.scatter(*eq_pt, color="black", marker="x", s=60, zorder=10, label=eq_name)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"{prefix}\n{eq_name}  r={radius:.0e}")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, label=d, markersize=7)
        for d, c in colour_map.items()
    ]
    ax.legend(handles=handles, fontsize=6, loc="best")
    fig.tight_layout()

    safe_r = f"{radius:.0e}".replace("-", "m")
    fig.savefig(outdir / f"sphere_{eq_name}_{safe_r}.png", dpi=130)
    plt.close(fig)


# ── Runner principal ───────────────────────────────────────────────────────────

def run_candidate(cand: Dict[str, Any]) -> Dict[str, Any]:
    m1, m0, c = cand["m1"], cand["m0"], cand["c"]
    prefix     = cand["prefix"]
    traj_path  = TRAJ_DIR / cand["traj_file"]
    tag        = _tag(m1, m0, c)
    outdir     = HID_BASE / prefix
    outdir.mkdir(parents=True, exist_ok=True)

    # Log file por candidato
    log_path = outdir / "run.log"
    logf = open(log_path, "w", encoding="utf-8", buffering=1)

    def log(msg: str):
        print(msg)
        logf.write(msg + "\n")

    log("=" * 70)
    log(f"HIDDENNESS TEST: {prefix}")
    log(f"  m1={m1}, m0={m0}, c={c:+.3f}")
    log("  Muestras COMPLETAS — sin parada temprana")
    log("=" * 70)

    # 1. Referencia
    ref_tail = load_ref_tail(traj_path, T_BURN_PROBE)
    log(f"  Referencia: {len(ref_tail)} puntos post-transitorio cargados.")
    if len(ref_tail) < 200:
        log("  [WARN] Trayectoria muy corta — resultados no confiables.")

    # 2. Sistema y equilibrios
    system     = setup_system(m1, m0)
    equilibria = get_equilibria(system)
    stable_eqs = list(equilibria.values())
    log(f"  Equilibrios encontrados: {list(equilibria.keys())}")
    for name, pt in equilibria.items():
        log(f"    {name}: {pt}")

    # 3. Sweep de esferas
    all_probe_runs: List[Dict[str, Any]] = []
    sphere_summary_records: List[Dict[str, Any]] = []

    for eq_name, eq_pt in equilibria.items():
        for r_idx, (radius, n_samples) in enumerate(zip(RADII, SAMPLES_PER_R)):
            log(f"\n  {tag} [{eq_name}]  radio={radius:.0e}  n={n_samples}")

            pts = generate_neighborhood_points(
                eq_point=eq_pt,
                radius=radius,
                num_samples=n_samples,
                mode="sphere_random",
                seed=RANDOM_SEED + r_idx,
            )

            stats = {
                "target_attractor": 0, "stable_equilibrium": 0,
                "divergence": 0, "other_attractor": 0, "numerical_failure": 0,
            }
            radius_runs: List[Dict[str, Any]] = []

            for k, pt in enumerate(pts):
                res = run_probe(system, pt, ref_tail, stable_eqs)
                radius_runs.append(res)
                stats[res["destination"]] = stats.get(res["destination"], 0) + 1

                # Progreso cada 25 %
                if (k + 1) % max(1, n_samples // 4) == 0:
                    log(f"    {k+1}/{n_samples}  "
                        f"TARGET={stats['target_attractor']}  "
                        f"EQ={stats['stable_equilibrium']}  "
                        f"DIV={stats['divergence']}  "
                        f"OTHER={stats['other_attractor']}")

            # ── Resumen de este radio ──────────────────────────────────────
            log(f"    FINAL ({n_samples} muestras): {stats}")

            # Guardar resultados y plots
            for res in radius_runs:
                all_probe_runs.append({
                    "equilibrium": eq_name,
                    "radius": float(radius),
                    **res,
                })

            sphere_summary_records.append({
                "system_id":  prefix,
                "equilibrium": eq_name,
                "radius":     float(radius),
                "samples":    len(radius_runs),
                "TARGET":     stats["target_attractor"],
                "EQ":         stats["stable_equilibrium"],
                "OTHER":      stats["other_attractor"],
                "DIV":        stats["divergence"],
                "FAIL":       stats["numerical_failure"],
            })

            plot_sphere_summary(
                eq_name, eq_pt, radius, radius_runs, pts, outdir, prefix
            )

    # 4. Contrato de ocultedad
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

    # 5. Guardar CSV
    df_summary = pd.DataFrame(sphere_summary_records)
    df_summary.to_csv(outdir / "hiddenness_summary.csv", index=False)

    probe_rows = [
        {"equilibrium": r["equilibrium"], "radius": r["radius"],
         "destination": r["destination"], "status": r["status"]}
        for r in all_probe_runs
    ]
    pd.DataFrame(probe_rows).to_csv(outdir / "probe_runs.csv", index=False)

    # 6. Guardar contrato JSON
    contract_jsonable = {k: v for k, v in contract.items() if k != "run_metadata"}
    with open(outdir / "hiddenness_contract.json", "w", encoding="utf-8") as f:
        json.dump(contract_jsonable, f, indent=2)

    # 7. Imprimir veredicto
    log(f"\n{'─'*50}".replace('─', '-'))
    log(f"  VEREDICTO: {contract['hiddenness_status']}")
    log(f"  hidden_verified:          {contract['hidden_verified']}")
    log(f"  hidden_compatible:        {contract['hidden_compatible']}")
    log(f"  self_excited_contact:     {contract['self_excited_contact_detected']}")
    log(f"  target_hits_total:        {contract['target_hits_total']}")
    log(f"  samples_total:            {contract['samples_total']}")
    if contract.get("failed_requirements"):
        log("  failed_requirements:")
        for req in contract["failed_requirements"]:
            log(f"    - {req}")

    logf.close()

    return {
        "prefix":              prefix,
        "m1": m1, "m0": m0, "c": c,
        "hiddenness_status":   contract["hiddenness_status"],
        "hidden_verified":     contract["hidden_verified"],
        "hidden_compatible":   contract["hidden_compatible"],
        "self_excited_contact": contract["self_excited_contact_detected"],
        "target_hits":         contract["target_hits_total"],
        "samples":             contract["samples_total"],
        "equilibria_tested":   contract["equilibria_tested"],
        "log_path":            str(log_path),
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python hiddenness_single_candidate.py <candidate_index>")
        print("  candidate_index ∈ {0, 1, 2}")
        sys.exit(1)

    idx = int(sys.argv[1])
    if idx not in (0, 1, 2):
        print(f"Error: candidate_index debe ser 0, 1 o 2 (recibido: {idx})")
        sys.exit(1)

    cand = CANDIDATES[idx]
    result = run_candidate(cand)

    # Guardar resultado JSON individual (para que el lanzador lo recoja)
    HID_BASE.mkdir(parents=True, exist_ok=True)
    result_path = HID_BASE / cand["prefix"] / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n[DONE] Resultado guardado en {result_path}")
    sys.exit(0)
