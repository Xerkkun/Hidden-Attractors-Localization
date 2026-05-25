#!/usr/bin/env python3
"""
Crea la matriz de experimentos para comparar:
1) semilla entera vs semilla fraccionaria;
2) continuacion tipo entera vs continuacion fraccionaria;
3) memoria Caputo completa vs memoria truncada;
4) ABM vs EFORK.

Este script NO modifica el workflow principal de la libreria.
Solo genera manifiestos, tareas, carpetas, llaves de cache y comandos base.
Los scripts ejecutores posteriores deben consumir los CSV/JSON generados aqui.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NumericalContract:
    model: str = "chua_nonsmooth"
    alpha: float = 8.4562
    beta: float = 12.0732
    gamma: float = 0.0052
    m0: float = -0.1768
    m1: float = -1.1468
    q_target: float = 0.9998
    h: float = 0.01
    t_final: float = 1500.0
    t_burn: float = 100.0
    memory_length: float = 40.0
    eta_steps: int = 101
    radii: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2)
    samples_per_radius: int = 64
    seed_phases: tuple[float, ...] = (0.0,)


@dataclass(frozen=True)
class ExperimentSpec:
    exp_id: str
    label: str
    seed_family: str
    transfer_power: str
    df_family: str
    continuation_family: str
    continuation_memory: str
    continuation_ic_policy: str
    hiddenness_memory_cases: tuple[str, ...]
    hiddenness_integrators: tuple[str, ...] = ("abm", "efork")
    continuation_integrators: tuple[str, ...] = ("abm", "efork")


def stable_key(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def experiments() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            exp_id="E01_integer_seed_integer_cont_full_tests",
            label="Semilla entera; continuacion tipo entera q=1; ocultedad full-history",
            seed_family="integer_like",
            transfer_power="s",
            df_family="classic",
            continuation_family="integer_like_q1",
            continuation_memory="none_q1_last_point",
            continuation_ic_policy="last_point_only",
            hiddenness_memory_cases=("full",),
        ),
        ExperimentSpec(
            exp_id="E02_integer_seed_integer_cont_truncated_tests",
            label="Semilla entera; continuacion tipo entera q=1; ocultedad truncada",
            seed_family="integer_like",
            transfer_power="s",
            df_family="classic",
            continuation_family="integer_like_q1",
            continuation_memory="none_q1_last_point",
            continuation_ic_policy="last_point_only",
            hiddenness_memory_cases=("truncated",),
        ),
        ExperimentSpec(
            exp_id="E03_integer_seed_fractional_cont_full_full_tests",
            label="Semilla entera; continuacion fraccionaria full-history; ocultedad full-history",
            seed_family="integer_like",
            transfer_power="s",
            df_family="classic",
            continuation_family="fractional_caputo",
            continuation_memory="full",
            continuation_ic_policy="history_window",
            hiddenness_memory_cases=("full",),
        ),
        ExperimentSpec(
            exp_id="E04_integer_seed_fractional_cont_truncated_truncated_tests",
            label="Semilla entera; continuacion fraccionaria truncada; ocultedad truncada",
            seed_family="integer_like",
            transfer_power="s",
            df_family="classic",
            continuation_family="fractional_caputo",
            continuation_memory="truncated",
            continuation_ic_policy="history_window_truncated",
            hiddenness_memory_cases=("truncated",),
        ),
        ExperimentSpec(
            exp_id="E05_fractional_seed_fractional_cont_full_all_tests",
            label="Semilla fraccionaria W_q; DF clasica; continuacion full; ocultedad full y truncada",
            seed_family="fractional",
            transfer_power="s^q",
            df_family="classic",
            continuation_family="fractional_caputo",
            continuation_memory="full",
            continuation_ic_policy="history_window",
            hiddenness_memory_cases=("full", "truncated"),
        ),
        ExperimentSpec(
            exp_id="E06_fractional_seed_fractional_cont_truncated_all_tests",
            label="Semilla fraccionaria W_q; DF clasica; continuacion truncada; ocultedad truncada y full",
            seed_family="fractional",
            transfer_power="s^q",
            df_family="classic",
            continuation_family="fractional_caputo",
            continuation_memory="truncated",
            continuation_ic_policy="history_window_truncated",
            hiddenness_memory_cases=("truncated", "full"),
        ),
    ]


def build_manifest(contract: NumericalContract, root: Path) -> dict[str, Any]:
    exp_list = experiments()
    shared = {
        "algebra": {
            "cache_key": stable_key({"stage": "algebra", "contract": asdict(contract)}),
            "outputs": [
                "shared/algebra/equilibria.json",
                "shared/algebra/jacobians.json",
                "shared/algebra/eigenvalues_matignon.json",
                "shared/algebra/lure_split.json",
            ],
        },
        "seed_integer_like": {
            "cache_key": stable_key({"stage": "seed", "family": "integer_like", "transfer": "s"}),
            "outputs": [
                "shared/seeds/integer_like_seed.json",
                "figures/shared/nyquist_integer_like.png",
            ],
        },
        "seed_fractional": {
            "cache_key": stable_key({"stage": "seed", "family": "fractional", "transfer": "s^q", "q": contract.q_target}),
            "outputs": [
                "shared/seeds/fractional_seed.json",
                "figures/shared/nyquist_fractional.png",
            ],
        },
        "equilibrium_clouds": {
            "cache_key": stable_key({"stage": "equilibrium_clouds", "radii": contract.radii, "n": contract.samples_per_radius}),
            "outputs": [
                "shared/equilibrium_clouds/E0.csv",
                "shared/equilibrium_clouds/Eplus.csv",
                "shared/equilibrium_clouds/Eminus.csv",
            ],
        },
    }

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Comparacion aislada de semillas, continuacion e integracion fraccionaria con memoria completa/truncada.",
        "scope": "No modificar el workflow principal hasta analizar resultados.",
        "root": str(root),
        "contract": asdict(contract),
        "shared_caches": shared,
        "experiments": [asdict(e) for e in exp_list],
    }


def build_tasks(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    contract = manifest["contract"]
    shared = manifest["shared_caches"]

    cache_tasks = []
    for name, meta in shared.items():
        cache_tasks.append({
            "task_id": f"cache_{name}",
            "stage": "shared_cache",
            "cache_name": name,
            "cache_key": meta["cache_key"],
            "outputs": ";".join(meta["outputs"]),
            "status": "pending",
        })

    continuation_tasks = []
    hiddenness_tasks = []
    figure_tasks = []

    for exp in manifest["experiments"]:
        seed_cache = "seed_fractional" if exp["seed_family"] == "fractional" else "seed_integer_like"
        for cont_solver in exp["continuation_integrators"]:
            cont_payload = {
                "exp_id": exp["exp_id"],
                "seed_cache": seed_cache,
                "solver": cont_solver,
                "continuation_family": exp["continuation_family"],
                "continuation_memory": exp["continuation_memory"],
                "q": 1.0 if exp["continuation_family"] == "integer_like_q1" else contract["q_target"],
                "eta_steps": contract["eta_steps"],
                "memory_length": None if exp["continuation_memory"] in ("full", "none_q1_last_point") else contract["memory_length"],
            }
            cont_key = stable_key(cont_payload)
            continuation_tasks.append({
                "task_id": f"cont_{exp['exp_id']}_{cont_solver}",
                "stage": "continuation",
                "exp_id": exp["exp_id"],
                "seed_cache": seed_cache,
                "continuation_solver": cont_solver,
                "continuation_family": exp["continuation_family"],
                "continuation_memory": exp["continuation_memory"],
                "continuation_ic_policy": exp["continuation_ic_policy"],
                "q_continuation": cont_payload["q"],
                "memory_length": cont_payload["memory_length"],
                "cache_key": cont_key,
                "output_dir": f"experiments/{exp['exp_id']}/continuation/{cont_solver}",
                "status": "pending",
            })

            for integ in exp["hiddenness_integrators"]:
                for mem in exp["hiddenness_memory_cases"]:
                    for sign in ("plus", "minus"):
                        hid_payload = {
                            "exp_id": exp["exp_id"],
                            "continuation_key": cont_key,
                            "hiddenness_integrator": integ,
                            "hiddenness_memory": mem,
                            "sign": sign,
                            "q": contract["q_target"],
                            "h": contract["h"],
                            "t_final": contract["t_final"],
                            "memory_length": None if mem == "full" else contract["memory_length"],
                        }
                        hid_key = stable_key(hid_payload)
                        hiddenness_tasks.append({
                            "task_id": f"hid_{exp['exp_id']}_{cont_solver}_{integ}_{mem}_{sign}",
                            "stage": "hiddenness",
                            "exp_id": exp["exp_id"],
                            "continuation_solver": cont_solver,
                            "continuation_cache_key": cont_key,
                            "hiddenness_integrator": integ,
                            "hiddenness_memory": mem,
                            "sign": sign,
                            "q_hiddenness": contract["q_target"],
                            "memory_length": hid_payload["memory_length"],
                            "equilibrium_cloud_cache": shared["equilibrium_clouds"]["cache_key"],
                            "cache_key": hid_key,
                            "output_dir": f"experiments/{exp['exp_id']}/hiddenness/{cont_solver}/{integ}_{mem}_{sign}",
                            "status": "pending",
                        })

        figure_tasks.append({
            "task_id": f"fig_{exp['exp_id']}",
            "stage": "figures",
            "exp_id": exp["exp_id"],
            "required_inputs": "continuation;hiddenness",
            "outputs": (
                f"figures/{exp['exp_id']}/seed_nyquist.png;"
                f"figures/{exp['exp_id']}/continuation_path.png;"
                f"figures/{exp['exp_id']}/phase3d.png;"
                f"figures/{exp['exp_id']}/basin_slices_xy_xz.png;"
                f"figures/{exp['exp_id']}/hiddenness_summary.png"
            ),
            "status": "pending",
        })

    return {
        "shared_cache_tasks": cache_tasks,
        "continuation_tasks": continuation_tasks,
        "hiddenness_tasks": hiddenness_tasks,
        "figure_tasks": figure_tasks,
    }


def write_readme(root: Path, manifest: dict[str, Any], workers: int) -> None:
    text = f"""# Experimentos Chua fraccionario no suave: matriz semilla/continuacion/memoria

Esta carpeta contiene una matriz experimental independiente del workflow principal.

## Contrato numerico base

- modelo: `{manifest['contract']['model']}`
- parametros: alpha={manifest['contract']['alpha']}, beta={manifest['contract']['beta']}, gamma={manifest['contract']['gamma']}, m0={manifest['contract']['m0']}, m1={manifest['contract']['m1']}
- q objetivo: {manifest['contract']['q_target']}
- h={manifest['contract']['h']}, t_final={manifest['contract']['t_final']}, t_burn={manifest['contract']['t_burn']}
- memoria truncada Lm={manifest['contract']['memory_length']}

## Orden obligatorio

1. `tasks/shared_cache_tasks.csv`
2. `tasks/continuation_tasks.csv`
3. `tasks/hiddenness_tasks.csv`
4. `tasks/figure_tasks.csv`
5. `reports/aggregate_comparison.csv` y `reports/experiment_matrix_summary.md`

## Paralelizacion

Usar maximo {workers} workers de Python. Forzar `OMP_NUM_THREADS=1` por proceso si los backends C usan OpenMP.
No ejecutar dos procesos escribiendo el mismo `cache_key`.

## Regla metodologica

La semilla por funcion descriptiva es solo aproximacion de primer armonico.
En Caputo, la continuacion fraccionaria debe propagar historia completa o ventana truncada etiquetada.
No declarar ocultedad si alguna cuenca desde vecindad de equilibrio alcanza el atractor candidato.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def write_run_commands(root: Path, workers: int) -> None:
    commands = f"""$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

# 1. Construir caches compartidos
python experiments/chua_nonsmooth_memory_matrix/run_shared_cache_tasks.py --tasks "{root / 'tasks' / 'shared_cache_tasks.csv'}" --workers {workers}

# 2. Ejecutar continuaciones
python experiments/chua_nonsmooth_memory_matrix/run_continuation_tasks.py --tasks "{root / 'tasks' / 'continuation_tasks.csv'}" --workers {workers}

# 3. Ejecutar pruebas de ocultedad
python experiments/chua_nonsmooth_memory_matrix/run_hiddenness_tasks.py --tasks "{root / 'tasks' / 'hiddenness_tasks.csv'}" --workers {workers}

# 4. Generar figuras y reporte comparativo
python experiments/chua_nonsmooth_memory_matrix/run_figure_tasks.py --tasks "{root / 'tasks' / 'figure_tasks.csv'}" --workers {max(1, workers // 2)}
python experiments/chua_nonsmooth_memory_matrix/aggregate_results.py --root "{root}"
"""
    (root / "run_experiments.ps1").write_text(commands, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/chua_nonsmooth_fractional_memory_matrix")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--q", type=float, default=0.9998)
    p.add_argument("--h", type=float, default=0.01)
    p.add_argument("--t-final", type=float, default=1500.0)
    p.add_argument("--t-burn", type=float, default=100.0)
    p.add_argument("--memory-length", type=float, default=40.0)
    p.add_argument("--samples-per-radius", type=int, default=64)
    args = p.parse_args()

    root = Path(args.out).resolve()
    contract = NumericalContract(
        q_target=args.q,
        h=args.h,
        t_final=args.t_final,
        t_burn=args.t_burn,
        memory_length=args.memory_length,
        samples_per_radius=args.samples_per_radius,
    )

    manifest = build_manifest(contract, root)
    tasks = build_tasks(manifest)

    write_json(root / "experiment_matrix.json", manifest)
    write_json(root / "cache_registry.json", manifest["shared_caches"])
    for name, rows in tasks.items():
        write_csv(root / "tasks" / f"{name}.csv", rows)

    for sub in ["shared", "experiments", "figures", "reports", "logs"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

    write_readme(root, manifest, args.workers)
    write_run_commands(root, args.workers)

    print(f"Matriz creada en: {root}")
    print(f"Experimentos: {len(manifest['experiments'])}")
    print(f"Tareas de cache: {len(tasks['shared_cache_tasks'])}")
    print(f"Tareas de continuacion: {len(tasks['continuation_tasks'])}")
    print(f"Tareas de ocultedad: {len(tasks['hiddenness_tasks'])}")
    print(f"Tareas de figuras: {len(tasks['figure_tasks'])}")


if __name__ == "__main__":
    main()
