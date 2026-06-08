#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test de ocultedad extendido para todos los equilibrios (E0, E+, E-):
   m1=-1.1468, m0=-0.1768, branch=1, c=+2.776

Protocolo:  Muestreo ball (volumen completo) alrededor de E0, E+ y E-.
            Aumento masivo de la densidad de muestras y expansion a radios lejanos (hasta r=2.0).
            Ejecutado en paralelo usando multiprocessing.
"""

from __future__ import annotations
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Windows UTF-8 fix
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from hidden_attractors.systems import get_system
from hidden_attractors.verification.equilibria import solve_equilibria

# ── Parametros Globales ────────────────────────────────────────────────────────

M1, M0, C_BIAS = -1.1468, -0.1768, 2.776
Q     = 0.9998
H     = 0.01
ALPHA = 8.4562
BETA  = 12.0732
GAMMA = 0.0052

PREFIX    = "biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776"
TRAJ_FILE = (Path("outputs/biased_saturation_search_q09998_corrected/trajectories")
             / "biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776_trajectory.csv")
OUTDIR    = (Path("outputs/biased_saturation_search_q09998_corrected/hiddenness")
             / (PREFIX + "_all_extended"))
OUTDIR.mkdir(parents=True, exist_ok=True)

# Plan de radios para cada equilibrio: (radio, n_muestras)
RADIUS_PLAN = [
    (1e-5,   80),
    (3e-5,   120),
    (1e-4,   160),
    (3e-4,   240),
    (1e-3,   400),
    (3e-3,   600),
    (1e-2,   800),
    (3e-2,   1000),
    (1e-1,   1200),
    (3e-1,   1200),
    (1.0,    1500),
    (2.0,    1500),
]

RADII         = [r for r, n in RADIUS_PLAN]
SAMPLES_PER_R = [n for r, n in RADIUS_PLAN]

T_FINAL_PROBE   = 200.0
T_BURN_PROBE    = 50.0
EQUILIBRIUM_TOL = 0.5
MATCH_METRIC    = "nn_percentile"
MATCH_TOL       = 0.5
MATCH_PERC      = 90.0
RANDOM_SEED     = 54321

# ── Variables Globales del Worker (Inicializadas por init_worker) ─────────────

_worker_system = None
_worker_ref_tail = None
_worker_stable_eqs = None

def init_worker(m1, m0, alpha, beta, gamma, ref_tail, stable_eqs):
    global _worker_system, _worker_ref_tail, _worker_stable_eqs
    from hidden_attractors.systems import get_system
    _worker_system = get_system("chua-nonsmooth")
    _worker_system.parameters.update({
        "m1": m1, "m0": m0,
        "alpha": alpha, "beta": beta, "gamma": gamma
    })
    _worker_ref_tail = ref_tail
    _worker_stable_eqs = stable_eqs

def worker_run_probe(x0) -> Dict[str, Any]:
    global _worker_system, _worker_ref_tail, _worker_stable_eqs
    from hidden_attractors.integrations.fractional_c import fractional_integrate
    from hidden_attractors.verification.hiddenness import evaluate_target_match

    try:
        _, x_arr, status, _ = fractional_integrate(
            rhs=lambda t, x: _worker_system.rhs(x, _worker_system.parameters),
            x0=x0, q=Q, h=H, t_final=T_FINAL_PROBE,
            method="abm", memory_mode="full",
            system=_worker_system, use_c_backend=True,
        )
    except Exception as exc:
        return {"destination": "numerical_failure", "status": f"exception:{exc}"}

    if status in ("diverged", "nonfinite_solution"):
        return {"destination": "divergence", "status": status}

    n_burn = int(np.ceil(T_BURN_PROBE / H))
    tail   = x_arr[n_burn:] if len(x_arr) > n_burn else x_arr
    final  = x_arr[-1] if len(x_arr) > 0 else x0

    for eq in _worker_stable_eqs:
        if np.linalg.norm(final - eq) <= EQUILIBRIUM_TOL:
            return {"destination": "stable_equilibrium", "status": "ok"}

    if evaluate_target_match(tail, _worker_ref_tail, metric=MATCH_METRIC, tolerance=MATCH_TOL):
        return {"destination": "target_attractor", "status": "ok"}

    return {"destination": "other_attractor", "status": "ok"}

# ── Utilidades de Muestreo y Log ────────────────────────────────────────────────

def sample_ball(eq_point: np.ndarray, radius: float, n: int, seed: int) -> np.ndarray:
    rng  = np.random.default_rng(seed)
    dim  = len(eq_point)
    pts  = []
    while len(pts) < n:
        batch = rng.normal(0.0, 1.0, (n * 3, dim))
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        r_vals = rng.uniform(0.0, 1.0, (n * 3, 1)) ** (1.0 / dim)
        ball_pts = eq_point + radius * r_vals * batch / norms
        for pt in ball_pts:
            if np.linalg.norm(pt - eq_point) <= radius:
                pts.append(pt)
            if len(pts) >= n:
                break
    return np.array(pts[:n])

def log(msg: str, logf=None):
    print(msg, flush=True)
    if logf:
        logf.write(msg + "\n")
        logf.flush()

def load_ref_tail(t_burn: float) -> np.ndarray:
    df = pd.read_csv(TRAJ_FILE)
    return df[df["t"] >= t_burn][["x", "y", "z"]].values

# ── Graficos ───────────────────────────────────────────────────────────────────

def plot_sphere(eq_name, eq_pt, radius, n, runs, pts, outdir):
    cmap = {
        "target_attractor":   "#dc2626",
        "stable_equilibrium": "#111827",
        "divergence":         "#f59e0b",
        "other_attractor":    "#2563eb",
        "numerical_failure":  "#6b7280",
    }
    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")
    for res, pt in zip(runs, pts):
        ax.scatter(*pt, color=cmap.get(res["destination"], "#9ca3af"), s=10, alpha=0.65)
    ax.scatter(*eq_pt, color="black", marker="x", s=80, zorder=10, label=eq_name)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"c=+2.776  {eq_name}  r={radius:.0e}  n={n}")
    handles = [plt.Line2D([0],[0], marker="o", color="w",
                           markerfacecolor=c, label=d, markersize=7)
               for d, c in cmap.items()]
    ax.legend(handles=handles, fontsize=6)
    fig.tight_layout()
    sr = f"{radius:.0e}".replace("-", "m")
    fig.savefig(outdir / f"sphere_{eq_name}_{sr}.png", dpi=130)
    plt.close(fig)

def plot_heatmap(records, outdir):
    eq_names  = ["E0", "E+", "E-"]
    radii_str = [f"{r:.0e}" for r in RADII]

    data       = np.full((len(eq_names), len(RADII)), np.nan)
    count_data = np.zeros((len(eq_names), len(RADII)), dtype=int)

    for rec in records:
        try:
            i = eq_names.index(rec["equilibrium"])
        except ValueError:
            continue
        j = -1
        for jj, rad in enumerate(RADII):
            if abs(rec["radius"] - rad) < rad * 0.01:
                j = jj; break
        if j >= 0:
            data[i, j]       = rec["TARGET"] / max(rec["samples"], 1)
            count_data[i, j] = rec["samples"]

    fig, ax = plt.subplots(figsize=(12, 4))
    masked = np.ma.array(data, mask=np.isnan(data))
    im = ax.imshow(masked, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(RADII)))
    ax.set_xticklabels(radii_str, rotation=30, ha="right")
    ax.set_yticks(range(len(eq_names)))
    ax.set_yticklabels(eq_names)
    ax.set_xlabel("Radio de esfera (ball sampling progresivo)")
    ax.set_ylabel("Equilibrio")
    ax.set_title("Fraccion TARGET hits (Todos los Equilibrios, Radios Extendidos)\n"
                 "verde=0 TARGET (compatible con ocultedad)   rojo=todos TARGET (autoexcitado)")
    plt.colorbar(im, ax=ax, label="fraccion TARGET")

    for i in range(len(eq_names)):
        for j in range(len(RADII)):
            if not np.isnan(data[i, j]):
                n_txt = count_data[i, j]
                t_txt = int(data[i, j] * n_txt)
                txt   = f"{data[i,j]:.2f}\n({t_txt}/{n_txt})"
                col   = "white" if data[i,j] > 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=col)

    fig.tight_layout()
    fig.savefig(outdir / "heatmap_target_fraction.png", dpi=150)
    plt.close(fig)

# ── Main Runner ────────────────────────────────────────────────────────────────

def main():
    logf = open(OUTDIR / "run.log", "w", encoding="utf-8", buffering=1)

    log("=" * 80, logf)
    log(f"TEST EXTENDIDO COMPLETO (Multiprocessing): m1={M1}, m0={M0}, c={C_BIAS:+.3f}", logf)
    log(f"  Radios: {RADII}", logf)
    log(f"  N/radio: {SAMPLES_PER_R}", logf)
    log(f"  Total/equilibrio: {sum(SAMPLES_PER_R)}   Total (3 eq): {sum(SAMPLES_PER_R)*3}", logf)
    log(f"  t_final={T_FINAL_PROBE}s  t_burn={T_BURN_PROBE}s", logf)
    log("=" * 80, logf)

    ref_tail = load_ref_tail(T_BURN_PROBE)
    log(f"  Referencia: {len(ref_tail)} puntos post-transitorio cargados.", logf)

    system     = get_system("chua-nonsmooth")
    system.parameters.update({"m1": M1, "m0": M0, "alpha": ALPHA, "beta": BETA, "gamma": GAMMA})
    equilibria = solve_equilibria(system)
    stable_eqs = list(equilibria.values())

    log("  Equilibrios detectados:", logf)
    for name, pt in equilibria.items():
        log(f"    {name}: {np.array2string(pt, precision=6)}", logf)

    # Cores
    n_workers = min(12, multiprocessing.cpu_count() - 2)
    n_workers = max(1, n_workers)
    log(f"\n  Iniciando Pool de multiprocessing con {n_workers} workers...", logf)

    all_records: List[Dict[str, Any]] = []
    all_runs:    List[Dict[str, Any]] = []
    t0_global   = time.time()
    total_probes = sum(SAMPLES_PER_R) * len(equilibria)
    probe_count = 0

    pool = multiprocessing.Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(M1, M0, ALPHA, BETA, GAMMA, ref_tail, stable_eqs)
    )

    for eq_idx, (eq_name, eq_pt) in enumerate(equilibria.items()):
        log(f"\n--- EVALUANDO EQUILIBRIO: {eq_name} ---", logf)
        for r_idx, (radius, n_samples) in enumerate(RADIUS_PLAN):
            t0_radio = time.time()
            log(f"  [{eq_name}]  radio={radius:.0e}  n={n_samples}", logf)

            # Muestreo
            pts = sample_ball(
                eq_point=eq_pt,
                radius=radius,
                n=n_samples,
                seed=RANDOM_SEED + eq_idx * 100 + r_idx * 10,
            )

            # Lanzamiento asíncrono
            async_results = [pool.apply_async(worker_run_probe, (pt,)) for pt in pts]

            stats = {k: 0 for k in
                     ["target_attractor","stable_equilibrium","other_attractor",
                      "divergence","numerical_failure"]}
            radius_runs: List[Dict[str, Any]] = []

            # Recolección y reporte de progreso
            report_step = max(1, n_samples // 4)
            for k, ares in enumerate(async_results):
                res = ares.get()
                radius_runs.append(res)
                stats[res["destination"]] = stats.get(res["destination"], 0) + 1
                probe_count += 1

                if (k + 1) % report_step == 0 or (k + 1) == n_samples:
                    elapsed = time.time() - t0_global
                    rate    = probe_count / elapsed if elapsed > 0 else 0
                    eta     = (total_probes - probe_count) / rate if rate > 0 else 0
                    log(f"    {k+1:4d}/{n_samples:4d}  "
                        f"TARGET={stats['target_attractor']:4d}  "
                        f"EQ={stats['stable_equilibrium']:4d}  "
                        f"OTHER={stats['other_attractor']:4d}  "
                        f"DIV={stats['divergence']}  "
                        f"[{elapsed:.0f}s total, ETA~{eta:.0f}s]", logf)

            dt_radio = time.time() - t0_radio
            log(f"    FINAL ({n_samples} muestras, {dt_radio:.1f}s): {stats}", logf)

            rec = {
                "equilibrium": eq_name, "radius": float(radius),
                "samples": n_samples,
                "TARGET": stats["target_attractor"],
                "EQ":     stats["stable_equilibrium"],
                "OTHER":  stats["other_attractor"],
                "DIV":    stats["divergence"],
                "FAIL":   stats["numerical_failure"],
            }
            all_records.append(rec)
            for res in radius_runs:
                all_runs.append({"equilibrium": eq_name, "radius": float(radius), **res})

            # Ploteo de esfera 3D
            plot_sphere(eq_name, eq_pt, radius, n_samples, radius_runs, pts, OUTDIR)

    pool.close()
    pool.join()

    # Veredicto
    target_hits  = sum(r["TARGET"] for r in all_records)
    samples_tot  = sum(r["samples"] for r in all_records)
    self_excited = target_hits > 0
    status       = "SELF_EXCITED_DETECTED" if self_excited else "HIDDEN_COMPATIBLE"

    log(f"\n{'='*80}", logf)
    log("TABLA RESUMEN COMPLETA POR EQUILIBRIO x RADIO (Radios Extendidos)", logf)
    log(f"{'Eq':<5} {'Radio':<10} {'N':>6} {'TARGET':>8} {'EQ':>6} {'OTHER':>7} {'DIV':>5} {'Fraccion':>10}", logf)
    log("-"*80, logf)
    for rec in all_records:
        frac = rec["TARGET"] / max(rec["samples"], 1)
        log(f"{rec['equilibrium']:<5} {rec['radius']:<10.0e} {rec['samples']:>6d} {rec['TARGET']:>8d} "
            f"{rec['EQ']:>6d} {rec['OTHER']:>7d} {rec['DIV']:>5d} {frac:>10.3f}", logf)
    log("-"*80, logf)

    log(f"\n  VEREDICTO FINAL: {status}", logf)
    log(f"  TARGET hits totales:      {target_hits} / {samples_tot}", logf)
    log(f"  Tiempo total:             {time.time()-t0_global:.0f}s", logf)
    log(f"{'='*80}", logf)

    # Guardar
    pd.DataFrame(all_records).to_csv(OUTDIR / "hiddenness_summary.csv", index=False)
    pd.DataFrame([{"equilibrium": r["equilibrium"], "radius": r["radius"],
                   "destination": r["destination"], "status": r["status"]}
                  for r in all_runs]).to_csv(OUTDIR / "probe_runs.csv", index=False)

    result = {
        "prefix": PREFIX,
        "m1": M1, "m0": M0, "c": C_BIAS,
        "sampling_mode": "ball_random",
        "protocol": {
            "radii": RADII,
            "n_per_radius": SAMPLES_PER_R,
            "total_probes": total_probes,
            "t_final": T_FINAL_PROBE,
            "t_burn": T_BURN_PROBE,
        },
        "hiddenness_status": status,
        "target_hits_total": target_hits,
        "samples_total": samples_tot,
        "records": all_records,
    }

    result_path = OUTDIR / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    plot_heatmap(all_records, OUTDIR)
    log(f"  Resultados en: {result_path}", logf)
    logf.close()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
