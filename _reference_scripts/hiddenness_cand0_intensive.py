#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test de ocultedad intensivo para el candidato 0:
   m1=-1.1468, m0=-0.1768, branch=1, c=+2.776

Protocolo:  Identico al del repositorio oficial (ball sampling, escalado progresivo),
            pero con muchos mas radios y muestras.

Radios y muestras progresivas (mismo escalado que el oficial: n aumenta con r):
  r=1e-5  ->  12  muestras   (radio mas pequeño, vecindad muy local)
  r=3e-5  ->  18  muestras
  r=1e-4  ->  24  muestras   (radio oficial minimo)
  r=3e-4  ->  36  muestras
  r=1e-3  ->  60  muestras   (radio oficial maximo, x2.5 vs oficial)
  r=3e-3  ->  100 muestras
  r=1e-2  ->  150 muestras   (radio grande, mayor cobertura)

Total por equilibrio: 400 muestras
Total (3 equilibrios x 7 radios): 1200 integraciones Caputo ABM
Muestreo: ball (volumen completo, igual al protocolo oficial)
"""

from __future__ import annotations
import json, sys, time
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

from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.systems import get_system
from hidden_attractors.verification.equilibria import solve_equilibria
from hidden_attractors.verification.hiddenness import evaluate_target_match


def sample_ball(eq_point: np.ndarray, radius: float, n: int, seed: int) -> np.ndarray:
    """Puntos uniformes dentro de la bola de radio r centrada en eq_point.
    Usa el metodo de rechazo con muestras gaussianas normalizadas y radio variable,
    equivalente al protocolo ball del repositorio oficial.
    """
    rng  = np.random.default_rng(seed)
    dim  = len(eq_point)
    pts  = []
    while len(pts) < n:
        batch = rng.normal(0.0, 1.0, (n * 3, dim))
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        # radios uniformes dentro de la bola via r^(1/dim)
        r_vals = rng.uniform(0.0, 1.0, (n * 3, 1)) ** (1.0 / dim)
        ball_pts = eq_point + radius * r_vals * batch / norms
        for pt in ball_pts:
            if np.linalg.norm(pt - eq_point) <= radius:
                pts.append(pt)
            if len(pts) >= n:
                break
    return np.array(pts[:n])

# ── Parametros ─────────────────────────────────────────────────────────────────

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
             / (PREFIX + "_intensive"))
OUTDIR.mkdir(parents=True, exist_ok=True)

# Protocolo progresivo: (radio, n_muestras)
# n escala aproximadamente con r^0.5 para cubrir el volumen de la bola de forma proporcional
RADIUS_PLAN = [
    (1e-5,   12),   # radio muy local
    (3e-5,   18),
    (1e-4,   24),   # radio minimo del protocolo oficial
    (3e-4,   36),
    (1e-3,   60),   # radio maximo del protocolo oficial (x2.5 muestras)
    (3e-3,  100),
    (1e-2,  150),   # radio grande, maxima cobertura
]
RADII         = [r for r, n in RADIUS_PLAN]
SAMPLES_PER_R = [n for r, n in RADIUS_PLAN]

T_FINAL_PROBE   = 200.0
T_BURN_PROBE    = 50.0
EQUILIBRIUM_TOL = 0.5
MATCH_METRIC    = "nn_percentile"
MATCH_TOL       = 0.5
MATCH_PERC      = 90.0
RANDOM_SEED     = 42

# ── Utilidades ─────────────────────────────────────────────────────────────────

def log(msg: str, logf=None):
    print(msg, flush=True)
    if logf:
        logf.write(msg + "\n")
        logf.flush()


def load_ref_tail(t_burn: float) -> np.ndarray:
    df = pd.read_csv(TRAJ_FILE)
    return df[df["t"] >= t_burn][["x", "y", "z"]].values


def setup_system():
    s = get_system("chua-nonsmooth")
    s.parameters.update({"m1": M1, "m0": M0,
                          "alpha": ALPHA, "beta": BETA, "gamma": GAMMA})
    return s


def run_probe(system, x0, ref_tail, stable_eqs) -> Dict[str, Any]:
    try:
        _, x_arr, status, _ = fractional_integrate(
            rhs=lambda t, x: system.rhs(x, system.parameters),
            x0=x0, q=Q, h=H, t_final=T_FINAL_PROBE,
            method="abm", memory_mode="full",
            system=system, use_c_backend=True,
        )
    except Exception as exc:
        return {"destination": "numerical_failure", "status": f"exception:{exc}"}

    if status in ("diverged", "nonfinite_solution"):
        return {"destination": "divergence", "status": status}

    n_burn = int(np.ceil(T_BURN_PROBE / H))
    tail   = x_arr[n_burn:] if len(x_arr) > n_burn else x_arr
    final  = x_arr[-1] if len(x_arr) > 0 else x0

    for eq in stable_eqs:
        if np.linalg.norm(final - eq) <= EQUILIBRIUM_TOL:
            return {"destination": "stable_equilibrium", "status": "ok"}

    if evaluate_target_match(tail, ref_tail, metric=MATCH_METRIC, tolerance=MATCH_TOL):
        return {"destination": "target_attractor", "status": "ok"}

    return {"destination": "other_attractor", "status": "ok"}


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
    """Fraccion de TARGET hits por (equilibrio x radio)."""
    eq_names  = ["E0", "E+", "E-"]
    radii_str = [f"{r:.0e}" for r in RADII]

    data       = np.full((len(eq_names), len(RADII)), np.nan)
    count_data = np.zeros((len(eq_names), len(RADII)), dtype=int)

    for rec in records:
        try:
            i = eq_names.index(rec["equilibrium"])
        except ValueError:
            continue
        # match radius con tolerancia flotante
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
    ax.set_title("Fraccion TARGET hits  [candidato c=+2.776]\n"
                 "verde=0 TARGET (compatible con ocultedad)   rojo=todos TARGET (autoexcitado)")
    plt.colorbar(im, ax=ax, label="fraccion TARGET")

    for i in range(len(eq_names)):
        for j in range(len(RADII)):
            if not np.isnan(data[i, j]):
                n_txt = count_data[i, j]
                t_txt = int(data[i, j] * n_txt)
                txt   = f"{data[i,j]:.2f}\n({t_txt}/{n_txt})"
                col   = "white" if data[i,j] > 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=col)

    fig.tight_layout()
    fig.savefig(outdir / "heatmap_target_fraction.png", dpi=150)
    plt.close(fig)
    print(f"  Heatmap: {outdir / 'heatmap_target_fraction.png'}", flush=True)


def print_plan():
    total = sum(n for _, n in RADIUS_PLAN) * 3   # 3 equilibrios
    print("Protocolo de muestreo (ball, progresivo):")
    print(f"  {'Radio':<10} {'N muestras':>12}  {'Acum./eq':>10}")
    acum = 0
    for r, n in RADIUS_PLAN:
        acum += n
        print(f"  {r:<10.0e} {n:>12d}  {acum:>10d}")
    print(f"  Total/equilibrio: {acum}   Total (3 eq): {acum*3}")
    print()


# ── Runner ─────────────────────────────────────────────────────────────────────

def main():
    logf = open(OUTDIR / "run.log", "w", encoding="utf-8", buffering=1)

    log("=" * 70, logf)
    log(f"TEST INTENSIVO (ball progressivo): m1={M1}, m0={M0}, c={C_BIAS:+.3f}", logf)
    log(f"  Radios: {RADII}", logf)
    log(f"  N/radio: {SAMPLES_PER_R}", logf)
    log(f"  Total integraciones: {sum(SAMPLES_PER_R)*3}", logf)
    log(f"  t_final={T_FINAL_PROBE}s  t_burn={T_BURN_PROBE}s", logf)
    log(f"  Muestreo: ball (volumen completo, protocolo oficial)", logf)
    log("=" * 70, logf)

    ref_tail = load_ref_tail(T_BURN_PROBE)
    log(f"  Referencia: {len(ref_tail)} puntos post-transitorio cargados.", logf)

    system     = setup_system()
    equilibria = solve_equilibria(system)
    stable_eqs = list(equilibria.values())

    log(f"  Equilibrios:", logf)
    for name, pt in equilibria.items():
        log(f"    {name}: {np.array2string(pt, precision=6)}", logf)
    log("", logf)

    all_records: List[Dict[str, Any]] = []
    all_runs:    List[Dict[str, Any]] = []
    t0_global   = time.time()
    probe_count = 0
    total_probes = sum(SAMPLES_PER_R) * len(equilibria)

    for eq_idx, (eq_name, eq_pt) in enumerate(equilibria.items()):
        for r_idx, (radius, n_samples) in enumerate(RADIUS_PLAN):
            t0_radio = time.time()
            log(f"  [{eq_name}]  radio={radius:.0e}  n={n_samples}", logf)

            # ball sampling (volumen completo, protocolo oficial)
            pts = sample_ball(
                eq_point=eq_pt,
                radius=radius,
                n=n_samples,
                seed=RANDOM_SEED + r_idx * 10 + eq_idx * 100,
            )

            stats = {k: 0 for k in
                     ["target_attractor","stable_equilibrium","other_attractor",
                      "divergence","numerical_failure"]}
            radius_runs: List[Dict[str, Any]] = []

            for k, pt in enumerate(pts):
                res = run_probe(system, pt, ref_tail, stable_eqs)
                radius_runs.append(res)
                stats[res["destination"]] = stats.get(res["destination"], 0) + 1
                probe_count += 1

                # Progreso cada 25% de las muestras de este radio
                report_step = max(1, n_samples // 4)
                if (k + 1) % report_step == 0 or (k + 1) == n_samples:
                    elapsed = time.time() - t0_global
                    rate    = probe_count / elapsed if elapsed > 0 else 0
                    eta     = (total_probes - probe_count) / rate if rate > 0 else 0
                    log(f"    {k+1:3d}/{n_samples:3d}  "
                        f"TARGET={stats['target_attractor']:3d}  "
                        f"EQ={stats['stable_equilibrium']:3d}  "
                        f"OTHER={stats['other_attractor']:3d}  "
                        f"DIV={stats['divergence']}  "
                        f"[{elapsed:.0f}s elaps, ETA~{eta:.0f}s]", logf)

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

            plot_sphere(eq_name, eq_pt, radius, n_samples, radius_runs, pts, OUTDIR)
            log("", logf)

    # ── Veredicto ──────────────────────────────────────────────────────────────
    target_hits  = sum(r["TARGET"] for r in all_records)
    samples_tot  = sum(r["samples"] for r in all_records)
    self_excited = target_hits > 0

    status           = "SELF_EXCITED_DETECTED" if self_excited else "HIDDEN_COMPATIBLE"
    hidden_compatible = not self_excited

    # Tabla de resultados por equilibrio
    log(f"{'='*70}", logf)
    log("TABLA RESUMEN POR EQUILIBRIO x RADIO", logf)
    log(f"{'Eq':<5} {'Radio':<10} {'N':>5} {'TARGET':>8} {'EQ':>6} {'OTHER':>7} {'Fraccion':>10}", logf)
    log("-"*60, logf)
    for rec in all_records:
        frac = rec["TARGET"] / max(rec["samples"], 1)
        log(f"{rec['equilibrium']:<5} {rec['radius']:<10.0e} "
            f"{rec['samples']:>5d} {rec['TARGET']:>8d} "
            f"{rec['EQ']:>6d} {rec['OTHER']:>7d} {frac:>10.3f}", logf)

    log("-"*60, logf)
    log(f"\n  VEREDICTO FINAL: {status}", logf)
    log(f"  hidden_compatible:        {hidden_compatible}", logf)
    log(f"  self_excited_contact:     {self_excited}", logf)
    log(f"  TARGET hits totales:      {target_hits} / {samples_tot}", logf)
    log(f"  Tiempo total:             {time.time()-t0_global:.0f}s", logf)
    log(f"\n  [ADVERTENCIA] No prueba global de ocultedad matematica.", logf)

    # Guardar
    pd.DataFrame(all_records).to_csv(OUTDIR / "hiddenness_summary.csv", index=False)
    pd.DataFrame([{"equilibrium": r["equilibrium"], "radius": r["radius"],
                   "destination": r["destination"], "status": r["status"]}
                  for r in all_runs]).to_csv(OUTDIR / "probe_runs.csv", index=False)

    result = {
        "prefix":                        PREFIX,
        "m1": M1, "m0": M0, "c": C_BIAS,
        "sampling_mode":                 "ball_random",
        "protocol": {
            "radii":   RADII,
            "n_per_radius": SAMPLES_PER_R,
            "total_per_equilibrium": sum(SAMPLES_PER_R),
            "total_probes": sum(SAMPLES_PER_R) * len(equilibria),
            "t_final": T_FINAL_PROBE,
            "t_burn":  T_BURN_PROBE,
        },
        "hiddenness_status":             status,
        "hidden_compatible":             hidden_compatible,
        "hidden_verified":               False,
        "self_excited_contact_detected": self_excited,
        "target_hits_total":             target_hits,
        "samples_total":                 samples_tot,
        "equilibria_tested":             list(equilibria.keys()),
        "records":                       all_records,
    }

    result_path = OUTDIR / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    plot_heatmap(all_records, OUTDIR)
    log(f"  Resultado en: {result_path}", logf)
    logf.close()

    return result


if __name__ == "__main__":
    print_plan()
    result = main()
    sys.exit(0)
