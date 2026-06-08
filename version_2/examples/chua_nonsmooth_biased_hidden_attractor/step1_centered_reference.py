#!/usr/bin/env python3
"""
Paso 1: Búsqueda Centrada de Referencia
========================================
Encuentra las ramas de la Función Descriptiva (DF) centrada (bias DC = 0)
del sistema de Chua fraccionario no suave como línea base.

Esta búsqueda centrada motivó el desarrollo de la DF sesgada: la rama
centrada sobrevive la continuación pero produce atractores periódicos
(no caóticos ocultos). El bias DC c ≠ 0 abre nuevas ramas inaccesibles
desde la DF centrada.

Salidas generadas:
  - centered_branches.csv      : ramas encontradas (omega, k, A, semilla)
  - centered_continuation.csv  : estado al final de continuación
  - centered_trajectories/     : CSV por candidato que sobrevive
  - plots/                     : espacio de fase 3D, series de tiempo
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
EXAMPLE_DIR = Path(__file__).resolve().parent
VERSION2    = EXAMPLE_DIR.parents[1]   # version_2/
ROOT        = VERSION2.parent          # raíz del repo

for p in [str(VERSION2), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml

from hidden_attractors.continuation.continuation_fractional import run_fractional_continuation
from hidden_attractors.diagnostics.periodicity import classify_post_transient_periodicity
from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.models.chua import chua_parameters
from hidden_attractors.seed_generation.chua import find_harmonic_seed, find_omega_gain_candidates
from hidden_attractors.systems import get_system

# ── Carga de configuración ─────────────────────────────────────────────────────
CFG_PATH = VERSION2 / "configs" / "examples" / "chua_nonsmooth_biased_df_search.yaml"


def load_config() -> Dict[str, Any]:
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Plotting (estilo de la librería) ─────────────────────────────────────────
LW = 0.7   # line width estándar de la librería


def _ax_style(ax, grid=True):
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")


def _ax3d_style(ax):
    ax.grid(True, color="#cbd5e1", lw=0.3, ls="--", alpha=0.5)


def plot_trajectory(traj: np.ndarray, title: str, outpath: Path, t_burn: float, h: float) -> None:
    """Figura de 4 paneles: espacio de fase 3D + proyecciones + series de tiempo."""
    n_burn = int(t_burn / h)
    states = traj[n_burn:, 1:4]
    times  = traj[n_burn:, 0]
    n = len(states)
    if n < 10:
        return

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    # 3D
    ax3 = fig.add_subplot(2, 3, (1, 4), projection="3d")
    ax3.plot(states[:, 0], states[:, 1], states[:, 2], lw=0.4, color='#10b981', alpha=0.85)
    _ax3d_style(ax3)
    ax3.set_xlabel("x", labelpad=2); ax3.set_ylabel("y", labelpad=2); ax3.set_zlabel("z", labelpad=2)
    ax3.set_title("Espacio de fase 3D", fontsize=11, fontweight='bold')

    # Proyecciones
    pairs = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]
    for sub_i, (xl, yl, ix, iy) in enumerate(pairs):
        ax = fig.add_subplot(2, 3, sub_i + 2 + (1 if sub_i >= 1 else 0))
        ax.plot(states[:, ix], states[:, iy], lw=0.3, color="#E64B35", alpha=0.8)
        _ax_style(ax)
        ax.set_xlabel(xl, fontsize=10); ax.set_ylabel(yl, fontsize=10)
        ax.set_title(f"Proyección {xl}-{yl}", fontsize=8)

    # Series de tiempo (x, y, z)
    for sub_i, (lbl, sig, clr) in enumerate(
        [("x(t)", states[:, 0], "#E64B35"),
         ("y(t)", states[:, 1], "#4DBBD5"),
         ("z(t)", states[:, 2], "#00A087")]
    ):
        ax = fig.add_subplot(2, 3, sub_i + 4 if sub_i < 2 else 6)
        ax.plot(times, sig, lw=0.35, color=clr)
        _ax_style(ax)
        ax.set_xlabel("t [s]", fontsize=10); ax.set_ylabel(lbl, fontsize=10)
        ax.set_title(f"Serie {lbl}", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


# ── Runner principal ──────────────────────────────────────────────────────────

def run_centered_reference(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ejecuta búsqueda centrada sobre todos los parámetros de la grilla."""
    sys_cfg   = cfg["system"]
    grid_cfg  = cfg["parameter_grid"]
    int_cfg   = cfg["integrator"]
    s1_cfg    = cfg["step1_centered_reference"]
    plot_cfg  = cfg["plots"]

    q = float(sys_cfg["q"])
    h = float(int_cfg["h"])
    memory_mode = int_cfg["memory_mode"]
    alpha = float(sys_cfg["parameters"]["alpha"])
    beta  = float(sys_cfg["parameters"]["beta"])
    gamma = float(sys_cfg["parameters"]["gamma"])

    out_root = ROOT / cfg["experiment"]["output_dir"]
    out_centered = out_root / "step1_centered"
    out_centered.mkdir(parents=True, exist_ok=True)
    (out_centered / "trajectories").mkdir(exist_ok=True)
    (out_centered / "plots").mkdir(exist_ok=True)

    t_sim_final    = float(s1_cfg["t_sim_final"])
    t_sim_trans    = float(s1_cfg["t_sim_transient"])
    t_cont_trans   = float(s1_cfg["t_transient"])
    t_cont_keep    = float(s1_cfg["t_keep"])
    eta_steps      = int(s1_cfg["eta_steps"])
    omega_min      = float(s1_cfg["omega_min"])
    omega_max      = float(s1_cfg["omega_max"])
    nscan          = int(s1_cfg["nscan"])
    lambda_values  = list(np.linspace(0.0, 1.0, eta_steps))

    m1_values = [float(v) for v in grid_cfg["m1_values"]]
    m0_values = [float(v) for v in grid_cfg["m0_values"]]

    all_results: List[Dict[str, Any]] = []

    print("=" * 65)
    print("PASO 1 — Búsqueda Centrada de Referencia")
    print(f"  q = {q},  h = {h},  memoria = {memory_mode}")
    print(f"  Grilla: {len(m1_values)} × {len(m0_values)} = {len(m1_values)*len(m0_values)} casos")
    print("=" * 65)

    for m1 in m1_values:
        for m0 in m0_values:
            params = chua_parameters(
                model="nonsmooth",
                alpha=alpha, beta=beta, gamma=gamma, m0=m0, m1=m1,
            )
            prefix = (
                f"centered_q{int(q*10000)}_m1_{m1:.4f}_m0_{m0:.4f}"
                .replace(".", "p").replace("-", "m")
            )

            print(f"\n  m1={m1}, m0={m0} …")

            # --- DF centrada: scan de frecuencias ---
            try:
                pairs = find_omega_gain_candidates(
                    q=q, params=params,
                    wmin=omega_min, wmax=omega_max, nscan=nscan,
                )
            except Exception as exc:
                print(f"    [DF SCAN] error: {exc}")
                continue

            print(f"    Ramas centradas encontradas: {len(pairs)}")

            for branch_idx, (omega0, k_gain) in enumerate(pairs):
                row: Dict[str, Any] = {
                    "m1": m1, "m0": m0, "branch": branch_idx,
                    "omega0": float(omega0), "k_gain": float(k_gain),
                    "cont_status": "not_run", "sim_status": "not_run",
                    "verdict": "not_run", "prefix": prefix + f"_br{branch_idx}",
                }

                # --- Reconstruir semilla ---
                try:
                    seed_data = find_harmonic_seed(
                        q=q, params=params, branch_index=branch_idx,
                        wmin=omega_min, wmax=omega_max,
                    )
                    x_seed = seed_data.seed
                except Exception as exc:
                    print(f"    [SEED br{branch_idx}] error: {exc}")
                    row["cont_status"] = "seed_error"
                    all_results.append(row)
                    continue

                # --- Continuación ABM ---
                system = get_system("chua-nonsmooth")
                system.parameters.update({
                    "m1": m1, "m0": m0,
                    "alpha": alpha, "beta": beta, "gamma": gamma,
                })

                try:
                    steps = run_fractional_continuation(
                        system=system,
                        seed_x0=x_seed,
                        k_gain=k_gain,
                        lambda_values=lambda_values,
                        h=h,
                        memory_mode=memory_mode,
                        integrator="abm",
                        t_transient=t_cont_trans,
                        t_keep=t_cont_keep,
                        q=q,
                    )
                    final_step  = steps[-1]
                    cont_status = final_step["status"]
                except Exception as exc:
                    print(f"    [CONT br{branch_idx}] error: {exc}")
                    row["cont_status"] = f"error:{exc}"
                    all_results.append(row)
                    continue

                row["cont_status"] = cont_status
                print(f"    br{branch_idx}: omega={omega0:.3f}  k={k_gain:.4f}  cont={cont_status}")

                if cont_status != "ok":
                    all_results.append(row)
                    continue

                x_final = final_step["x_out"].copy()

                # --- Simulación larga ---
                try:
                    sim_t, sim_x, sim_status, _ = fractional_integrate(
                        rhs=lambda t, x: system.rhs(x, system.parameters),
                        x0=x_final, q=q, h=h, t_final=t_sim_final,
                        method="abm", memory_mode=memory_mode,
                        system=system, use_c_backend=True,
                    )
                except Exception as exc:
                    row["sim_status"] = f"error:{exc}"
                    all_results.append(row)
                    continue

                row["sim_status"] = sim_status

                # --- Clasificación ---
                full_traj = np.column_stack((sim_t, sim_x))
                post_traj = full_traj[sim_t >= t_sim_trans]

                if len(post_traj) > 10:
                    diag = classify_post_transient_periodicity(post_traj, h=h)
                    verdict = diag["candidate_label"]
                else:
                    verdict = "too_short"

                row["verdict"] = verdict
                print(f"           sim_status={sim_status}  verdict={verdict}")

                # --- Guardar trayectoria ---
                traj_path = out_centered / "trajectories" / f"{row['prefix']}_trajectory.csv"
                np.savetxt(
                    traj_path, post_traj, delimiter=",",
                    header="t,x,y,z", comments="",
                )

                # --- Figura (estilo reporte) ---
                if plot_cfg["save_figures"]:
                    params_str = f"m1={m1} | m0={m0} | ω₀={omega0:.3f} | k={k_gain:.4f}"
                    plot_trajectory(
                        traj=post_traj,
                        title=f"[Referencia Centrada] {params_str}",
                        outpath=out_centered / "plots" / f"{row['prefix']}_phase.png",
                        t_burn=0.0,   # ya viene post-transiente
                        h=h,
                    )

                all_results.append(row)

    # --- CSV resumen ---
    csv_path = out_centered / "centered_branches.csv"
    if all_results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)

    print(f"\n[PASO 1 COMPLETADO] Resultados: {csv_path}")
    survived = [r for r in all_results if r["cont_status"] == "ok"]
    print(f"  Ramas que sobreviven continuación: {len(survived)} / {len(all_results)}")
    verdicts = {}
    for r in survived:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    for v, cnt in verdicts.items():
        print(f"    {v}: {cnt}")

    return all_results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    run_centered_reference(cfg)
