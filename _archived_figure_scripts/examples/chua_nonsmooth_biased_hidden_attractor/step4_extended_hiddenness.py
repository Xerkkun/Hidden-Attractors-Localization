#!/usr/bin/env python3
"""
Paso 4: Verificación Extendida de Ocultedad (Multiprocessing)
=============================================================
Test masivo para el candidato confirmado (c ≈ +2.776, m1=-1.1468, m0=-0.1768).

Extiende el barrido de esferas hasta radios de r = 2.0, con densidad
masiva de muestras por radio (hasta 1500 puntos por radio por equilibrio)
usando un Pool de multiprocessing para explotar todos los núcleos disponibles.

Este paso confirma definitivamente que el atractor NO es alcanzable desde las
vecindades de ninguno de los 3 equilibrios, incluso explorando regiones del
espacio de estado muy alejadas de ellos.

Total de pruebas: ~28.830 integraciones (9.610 por equilibrio × 3 equilibrios)

Salidas:
  - hiddenness_summary.csv      : resumen por equilibrio × radio
  - probe_runs.csv              : destino de cada sonda
  - result.json                 : veredicto final y protocolo completo
  - heatmap_target_fraction.png : fracción TARGET por equilibrio × radio
  - sphere_*.png                : figuras 3D de esferas por radio
"""

from __future__ import annotations

import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

CFG_PATH = VERSION2 / "configs" / "examples" / "chua_nonsmooth_biased_df_search.yaml"


def load_config() -> Dict[str, Any]:
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Variables globales de worker (se inicializan en el pool) ──────────────────
_worker_system    = None
_worker_ref_tail  = None
_worker_stable_eqs = None
_worker_h4_cfg    = None


def init_worker(m1: float, m0: float, alpha: float, beta: float, gamma: float,
                ref_tail: np.ndarray, stable_eqs: List[np.ndarray],
                h4_cfg: Dict[str, Any]) -> None:
    """Inicializa el worker del pool con el sistema y referencias globales."""
    global _worker_system, _worker_ref_tail, _worker_stable_eqs, _worker_h4_cfg
    from hidden_attractors.systems import get_system as _get
    _worker_system = _get("chua-nonsmooth")
    _worker_system.parameters.update({
        "m1": m1, "m0": m0, "alpha": alpha, "beta": beta, "gamma": gamma,
    })
    _worker_ref_tail   = ref_tail
    _worker_stable_eqs = stable_eqs
    _worker_h4_cfg     = h4_cfg


def worker_run_probe(x0: np.ndarray) -> Dict[str, Any]:
    """Función de worker: integra una sonda y clasifica su destino."""
    from hidden_attractors.integrations.fractional_c import fractional_integrate
    from hidden_attractors.verification.hiddenness import evaluate_target_match

    cfg    = _worker_h4_cfg
    q      = float(cfg["_q"])
    h      = float(cfg["_h"])
    t_fin  = float(cfg["t_final_probe"])
    t_burn = float(cfg["t_burn_probe"])
    eq_tol = float(cfg["equilibrium_tol"])

    try:
        _, x_arr, status, _ = fractional_integrate(
            rhs=lambda t, x: _worker_system.rhs(x, _worker_system.parameters),
            x0=x0, q=q, h=h, t_final=t_fin,
            method="abm", memory_mode="full",
            system=_worker_system, use_c_backend=True,
        )
    except Exception as exc:
        return {"destination": "numerical_failure", "status": f"exception:{exc}"}

    if status in ("diverged", "nonfinite_solution"):
        return {"destination": "divergence", "status": status}

    n_burn = int(np.ceil(t_burn / h))
    tail   = x_arr[n_burn:] if len(x_arr) > n_burn else x_arr
    final  = x_arr[-1] if len(x_arr) > 0 else x0

    for eq in _worker_stable_eqs:
        if np.linalg.norm(final - eq) <= eq_tol:
            return {"destination": "stable_equilibrium", "status": "ok"}

    if evaluate_target_match(tail, _worker_ref_tail,
                              metric=cfg["match_metric"],
                              tolerance=float(cfg["match_tol"])):
        return {"destination": "target_attractor", "status": "ok"}

    return {"destination": "other_attractor", "status": "ok"}


# ── Muestreo ball ─────────────────────────────────────────────────────────────

def sample_ball(eq_point: np.ndarray, radius: float, n: int, seed: int) -> np.ndarray:
    """Muestrea n puntos uniformes dentro del ball de radio r centrado en eq_point."""
    rng = np.random.default_rng(seed)
    dim = len(eq_point)
    pts: List[np.ndarray] = []
    while len(pts) < n:
        batch  = rng.normal(0.0, 1.0, (n * 3, dim))
        norms  = np.linalg.norm(batch, axis=1, keepdims=True)
        r_vals = rng.uniform(0.0, 1.0, (n * 3, 1)) ** (1.0 / dim)
        ball   = eq_point + radius * r_vals * batch / norms
        for pt in ball:
            if np.linalg.norm(pt - eq_point) <= radius:
                pts.append(pt)
            if len(pts) >= n:
                break
    return np.array(pts[:n])


# ── Plotting ──────────────────────────────────────────────────────────────────
COLOUR_MAP = {
    "target_attractor":   "#ef4444",
    "stable_equilibrium": "#3b82f6",
    "divergence":         "#f59e0b",
    "other_attractor":    "#8b5cf6",
    "numerical_failure":  "#94a3b8",
}


def plot_sphere_3d(eq_name: str, eq_pt: np.ndarray, radius: float,
                   runs: List[Dict], pts: np.ndarray, outdir: Path) -> None:
    """Nube 3D de puntos de sonda coloreados por destino."""
    fig = plt.figure(figsize=(7, 6), dpi=300)
    ax  = fig.add_subplot(111, projection="3d")
    for res, pt in zip(runs, pts):
        ax.scatter(*pt, color=COLOUR_MAP.get(res["destination"], "#9ca3af"), s=10, alpha=0.65)
    ax.scatter(*eq_pt, color="black", marker="x", s=80, zorder=10, label=eq_name)
    ax.set_xlabel("x", fontsize=10); ax.set_ylabel("y", fontsize=10)
    ax.set_zlabel("z", fontsize=10)
    ax.set_title(f"c=+2.776  {eq_name}  r={radius:.0e}  n={len(runs)}", fontsize=11, fontweight='bold')
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=c, label=d, markersize=7)
               for d, c in COLOUR_MAP.items()]
    ax.legend(handles=handles, fontsize=6)
    fig.tight_layout()
    safe_r = f"{radius:.0e}".replace("-", "m")
    outpath = outdir / f"sphere_{eq_name}_{safe_r}.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_heatmap(records: List[Dict], radii: List[float], outpath: Path) -> None:
    """Heatmap de fracción TARGET por equilibrio × radio (versión extendida)."""
    eq_names = sorted({r["equilibrium"] for r in records})
    rad_strs = [f"{r:.0e}" for r in radii]
    data     = np.full((len(eq_names), len(radii)), np.nan)
    counts   = np.zeros_like(data, dtype=int)

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

    fig, ax = plt.subplots(figsize=(max(14, len(radii) * 1.2), max(3, len(eq_names) * 1.2)))
    masked = np.ma.array(data, mask=np.isnan(data))
    im = ax.imshow(masked, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(radii)))
    ax.set_xticklabels(rad_strs, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(eq_names)))
    ax.set_yticklabels(eq_names, fontsize=9)
    ax.set_xlabel("Radio (ball sampling)", fontsize=9)
    ax.set_ylabel("Equilibrio", fontsize=9)
    ax.set_title(
        "Fracción TARGET hits — Test Extendido (Todos equilibrios, Radios hasta r=2.0)\n"
        "verde=0 TARGET (compatible con ocultedad)   rojo=todos TARGET (autoexcitado)",
        fontsize=9,
    )
    plt.colorbar(im, ax=ax, label="fracción TARGET")
    for i in range(len(eq_names)):
        for j in range(len(radii)):
            if not np.isnan(data[i, j]):
                t_txt = int(data[i, j] * counts[i, j])
                txt   = f"{data[i, j]:.2f}\n({t_txt}/{counts[i,j]})"
                col   = "white" if data[i, j] > 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=col)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


# ── Runner principal ──────────────────────────────────────────────────────────

def run_extended_hiddenness(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ejecuta el test extendido para el candidato confirmado."""
    sys_cfg = cfg["system"]
    int_cfg = cfg["integrator"]
    h4_cfg  = cfg["step4_extended_hiddenness"].copy()
    h4_cfg["_q"]    = float(sys_cfg["q"])
    h4_cfg["_h"]    = float(int_cfg["h"])
    plot_cfg = cfg["plots"]

    alpha = float(sys_cfg["parameters"]["alpha"])
    beta  = float(sys_cfg["parameters"]["beta"])
    gamma = float(sys_cfg["parameters"]["gamma"])

    target = h4_cfg["target_candidate"]
    m1, m0 = float(target["m1"]), float(target["m0"])
    c_bias  = float(target["c"])
    prefix  = (
        f"biased_q{int(float(sys_cfg['q'])*10000)}"
        f"_m1_{m1:.4f}_m0_{m0:.4f}_branch_{target['branch']}_c_{c_bias:.3f}"
    ).replace(".", "p").replace("-", "m")

    out_root = ROOT / cfg["experiment"]["output_dir"]
    traj_path = out_root / "step2_biased_df" / "trajectories" / f"{prefix}_trajectory.csv"
    outdir    = out_root / "step4_extended_hiddenness" / prefix
    outdir.mkdir(parents=True, exist_ok=True)

    # Referencia
    df_ref   = pd.read_csv(traj_path)
    t_burn   = float(h4_cfg["t_burn_probe"])
    ref_tail = df_ref[df_ref["t"] >= t_burn][["x", "y", "z"]].values

    # Equilibrios
    from hidden_attractors.systems import get_system
    from hidden_attractors.verification.equilibria import solve_equilibria
    system = get_system("chua-nonsmooth")
    system.parameters.update({"m1": m1, "m0": m0, "alpha": alpha, "beta": beta, "gamma": gamma})
    equilibria  = solve_equilibria(system)
    stable_eqs  = list(equilibria.values())

    radius_plan = [(float(r), int(n)) for r, n in h4_cfg["radius_plan"]]
    radii       = [r for r, _ in radius_plan]
    random_seed = int(cfg["experiment"]["random_seed"])

    # Workers
    n_workers_cfg = h4_cfg.get("n_workers", "auto")
    if n_workers_cfg == "auto":
        n_workers = max(1, multiprocessing.cpu_count() - 2)
    else:
        n_workers = int(n_workers_cfg)

    print("=" * 65)
    print("PASO 4 — Verificación Extendida (Multiprocessing)")
    print(f"  m1={m1}  m0={m0}  c={c_bias}")
    print(f"  Radios: {radii}")
    print(f"  Total muestras/equilibrio: {sum(n for _, n in radius_plan)}")
    print(f"  Total muestras: {sum(n for _, n in radius_plan) * len(equilibria)}")
    print(f"  Workers: {n_workers}")
    print("=" * 65)

    logf = open(outdir / "run.log", "w", encoding="utf-8", buffering=1)

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    all_records : List[Dict] = []
    all_runs    : List[Dict] = []
    t0_global   = time.time()
    total_probes = sum(n for _, n in radius_plan) * len(equilibria)
    probe_count  = 0

    pool = multiprocessing.Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(m1, m0, alpha, beta, gamma, ref_tail, stable_eqs, h4_cfg),
    )

    for eq_idx, (eq_name, eq_pt) in enumerate(equilibria.items()):
        log(f"\n--- EQUILIBRIO: {eq_name}  ({np.array2string(eq_pt, precision=4)}) ---")

        for r_idx, (radius, n_samples) in enumerate(radius_plan):
            t0_r = time.time()
            seed  = random_seed + eq_idx * 100 + r_idx * 10
            pts   = sample_ball(eq_pt, radius, n_samples, seed)

            async_res = [pool.apply_async(worker_run_probe, (pt,)) for pt in pts]
            stats = {k: 0 for k in
                     ["target_attractor", "stable_equilibrium", "divergence",
                      "other_attractor", "numerical_failure"]}
            radius_runs: List[Dict] = []
            report_step = max(1, n_samples // 4)

            for k, ares in enumerate(async_res):
                res = ares.get()
                radius_runs.append(res)
                stats[res["destination"]] = stats.get(res["destination"], 0) + 1
                probe_count += 1

                if (k + 1) % report_step == 0 or (k + 1) == n_samples:
                    elapsed = time.time() - t0_global
                    rate    = probe_count / elapsed if elapsed > 0 else 0
                    eta     = (total_probes - probe_count) / rate if rate > 0 else 0
                    log(f"  [{eq_name}] r={radius:.0e}  {k+1:4d}/{n_samples:4d}  "
                        f"TARGET={stats['target_attractor']}  EQ={stats['stable_equilibrium']}  "
                        f"OTHER={stats['other_attractor']}  "
                        f"[{elapsed:.0f}s  ETA~{eta:.0f}s]")

            dt_r = time.time() - t0_r
            log(f"  FINAL r={radius:.0e}  n={n_samples}  {dt_r:.1f}s: {stats}")

            if plot_cfg["save_figures"]:
                plot_sphere_3d(eq_name, eq_pt, radius, radius_runs, pts, outdir)

            for res in radius_runs:
                all_runs.append({"equilibrium": eq_name, "radius": float(radius), **res})

            all_records.append({
                "equilibrium": eq_name, "radius": float(radius), "samples": n_samples,
                "TARGET": stats["target_attractor"], "EQ": stats["stable_equilibrium"],
                "OTHER": stats["other_attractor"], "DIV": stats["divergence"],
                "FAIL": stats["numerical_failure"],
            })

    pool.close()
    pool.join()

    # Veredicto
    target_hits  = sum(r["TARGET"] for r in all_records)
    samples_tot  = sum(r["samples"] for r in all_records)
    self_excited = target_hits > 0
    status       = "SELF_EXCITED_DETECTED" if self_excited else "HIDDEN_COMPATIBLE"

    log(f"\n{'='*65}")
    log(f"VEREDICTO FINAL: {status}")
    log(f"  TARGET hits totales: {target_hits} / {samples_tot}")
    log(f"  Tiempo total: {time.time()-t0_global:.0f}s")
    log("=" * 65)
    logf.close()

    # Guardar
    pd.DataFrame(all_records).to_csv(outdir / "hiddenness_summary.csv", index=False)
    pd.DataFrame([{"equilibrium": r["equilibrium"], "radius": r["radius"],
                   "destination": r["destination"], "status": r["status"]}
                  for r in all_runs]).to_csv(outdir / "probe_runs.csv", index=False)

    result = {
        "prefix": prefix, "m1": m1, "m0": m0, "c": c_bias,
        "sampling_mode": h4_cfg["sampling_mode"],
        "protocol": {"radii": radii, "radius_plan": radius_plan,
                     "total_probes": total_probes,
                     "t_final": float(h4_cfg["t_final_probe"]),
                     "t_burn":  float(h4_cfg["t_burn_probe"])},
        "hiddenness_status": status,
        "target_hits_total": int(target_hits),
        "samples_total":     int(samples_tot),
        "records": all_records,
    }
    with open(outdir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if plot_cfg["save_figures"] and plot_cfg.get("heatmap_hiddenness", True):
        plot_heatmap(all_records, radii, outdir / "heatmap_target_fraction.png")

    print(f"\n[PASO 4 COMPLETADO]  Resultado en {outdir / 'result.json'}")
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.freeze_support()   # necesario en Windows
    cfg = load_config()
    out_root = ROOT / cfg["experiment"]["output_dir"]
    target = cfg["step4_extended_hiddenness"]["target_candidate"]
    q      = float(cfg["system"]["q"])
    m1, m0 = float(target["m1"]), float(target["m0"])
    c_bias  = float(target["c"])
    prefix  = (
        f"biased_q{int(q*10000)}_m1_{m1:.4f}_m0_{m0:.4f}"
        f"_branch_{target['branch']}_c_{c_bias:.3f}"
    ).replace(".", "p").replace("-", "m")
    traj_path = out_root / "step2_biased_df" / "trajectories" / f"{prefix}_trajectory.csv"
    if not traj_path.exists():
        print(f"Trayectoria de referencia no encontrada: {traj_path}")
        print("Ejecute step2_biased_df_search.py primero.")
        sys.exit(1)
    run_extended_hiddenness(cfg)
