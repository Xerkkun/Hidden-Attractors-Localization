#!/usr/bin/env python3
"""Orquestador Principal — Primer Atractor Oculto en Chua Fraccionario
====================================================================
Ejecuta el pipeline completo en orden, o pasos individuales.

Uso rápido (prueba de humo, simulaciones cortas):
    python run_example.py --quick

Ejecución completa (puede tomar horas en el Paso 4):
    python run_example.py

Pasos individuales:
    python run_example.py --steps 1 2 3
    python run_example.py --steps 4
    python run_example.py --steps 5

Pasos disponibles:
  1 – Búsqueda centrada de referencia (DF c=0)
  2 – Búsqueda DF sesgada corregida (c != 0, continuación afín)
  3 – Verificación de ocultedad estándar (225 muestras/equilibrio)
  4 – Verificación extendida con multiprocessing (hasta r=2.0)
  5 – Resumen final y galería de figuras

Configuración: configs/examples/chua_nonsmooth_biased_df_search.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
EXAMPLE_DIR = Path(__file__).resolve().parent
VERSION2    = EXAMPLE_DIR.parents[1]   # version_2/
ROOT        = VERSION2.parent          # raíz del repo

for p in [str(VERSION2), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml

from hidden_attractors.workflows.biased_chua import (
    run_centered_reference,
    run_biased_df_search,
    run_hiddenness_verification,
    run_extended_hiddenness,
    run_summarize_and_plot,
)

CFG_PATH = VERSION2 / "configs" / "examples" / "chua_nonsmooth_biased_df_search.yaml"


def load_config(quick: bool = False) -> Dict[str, Any]:
    with open(CFG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if quick:
        cfg["step1_centered_reference"]["t_sim_final"]    = 50.0
        cfg["step1_centered_reference"]["t_sim_transient"] = 10.0
        cfg["step1_centered_reference"]["t_transient"]    = 10.0
        cfg["step1_centered_reference"]["t_keep"]         = 10.0
        cfg["step2_biased_df_search"]["t_sim_final"]      = 50.0
        cfg["step2_biased_df_search"]["t_sim_transient"]  = 10.0
        cfg["step2_biased_df_search"]["t_transient"]      = 10.0
        cfg["step2_biased_df_search"]["t_keep"]           = 10.0
        cfg["step3_hiddenness"]["t_final_probe"]          = 30.0
        cfg["step3_hiddenness"]["t_burn_probe"]           = 10.0
        cfg["step3_hiddenness"]["samples_per_radius"]     = [5, 5, 5, 5, 5, 5]
        cfg["step4_extended_hiddenness"]["t_final_probe"] = 30.0
        cfg["step4_extended_hiddenness"]["t_burn_probe"]  = 10.0
        cfg["step4_extended_hiddenness"]["radius_plan"]   = [
            [1e-5, 10], [1e-4, 10], [1e-3, 10], [1e-2, 10],
        ]
        print("[MODO QUICK] Tiempos de simulación reducidos para prueba de humo.\n")

    return cfg


def banner(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65 + "\n")


def run_step1(cfg: Dict[str, Any]) -> List[Dict]:
    banner("PASO 1 — Búsqueda Centrada de Referencia (DF c=0)")
    return run_centered_reference(cfg)


def run_step2(cfg: Dict[str, Any]) -> List[Dict]:
    banner("PASO 2 — Búsqueda DF Sesgada Corregida")
    return run_biased_df_search(cfg)


def run_step3(cfg: Dict[str, Any], step2_results: Optional[List[Dict]] = None) -> List[Dict]:
    banner("PASO 3 — Verificación de Ocultedad (Protocolo Estándar)")

    if step2_results:
        traj_dir = ROOT / cfg["experiment"]["output_dir"] / "step2_biased_df" / "trajectories"
        candidates = []
        for r in step2_results:
            if "failed" in r.get("verdict", ""):
                continue
            prefix    = r.get("prefix", "")
            traj_path = traj_dir / f"{prefix}_trajectory.csv"
            if traj_path.exists():
                candidates.append({
                    "m1": r["m1"], "m0": r["m0"],
                    "branch": r.get("branch", 0),
                    "c": r.get("c", 0.0),
                    "prefix": prefix,
                    "traj_path": str(traj_path),
                })
    else:
        traj_dir = ROOT / cfg["experiment"]["output_dir"] / "step2_biased_df" / "trajectories"
        candidates = [
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

    valid = [c for c in candidates if Path(c["traj_path"]).exists()]
    if not valid:
        print("[PASO 3] No se encontraron trayectorias. Ejecute el Paso 2 primero.")
        return []

    return run_hiddenness_verification(valid, cfg)


def run_step4(cfg: Dict[str, Any]) -> Dict:
    banner("PASO 4 — Verificación Extendida Multiprocessing (r hasta 2.0)")
    import multiprocessing
    multiprocessing.freeze_support()
    traj_dir  = ROOT / cfg["experiment"]["output_dir"] / "step2_biased_df" / "trajectories"
    target    = cfg["step4_extended_hiddenness"]["target_candidate"]
    q         = float(cfg["system"]["q"])
    m1, m0    = float(target["m1"]), float(target["m0"])
    c_bias    = float(target["c"])
    prefix    = (
        f"biased_q{int(q*10000)}_m1_{m1:.4f}_m0_{m0:.4f}"
        f"_branch_{target['branch']}_c_{c_bias:.3f}"
    ).replace(".", "p").replace("-", "m")
    traj_path = traj_dir / f"{prefix}_trajectory.csv"
    if not traj_path.exists():
        print(f"[PASO 4] Trayectoria no encontrada: {traj_path}")
        print("  Ejecute el Paso 2 primero.")
        return {}
    return run_extended_hiddenness(cfg)


def run_step5(cfg: Dict[str, Any]) -> None:
    banner("PASO 5 — Resumen Final y Galería de Figuras")
    run_summarize_and_plot(cfg)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, metavar="N",
        choices=[1, 2, 3, 4, 5],
        help="Pasos a ejecutar (por defecto: todos excepto el 4 extendido)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Modo rápido: tiempos reducidos para prueba de humo",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Incluir el Paso 4 (verificación extendida — puede tomar horas)",
    )
    args = parser.parse_args()

    cfg = load_config(quick=args.quick)

    if args.steps:
        steps = sorted(set(args.steps))
    elif args.all:
        steps = [1, 2, 3, 4, 5]
    else:
        steps = [1, 2, 3, 5]

    print(f"\n{'='*65}")
    print("  PRIMER ATRACTOR OCULTO — CHUA FRACCIONARIO NO SUAVE")
    print(f"  Librería: hidden_attractors_fo")
    print(f"  Pasos a ejecutar: {steps}")
    print(f"  Salida: {ROOT / cfg['experiment']['output_dir']}")
    print(f"{'='*65}\n")

    t0        = time.time()
    s2_results: Optional[List[Dict]] = None
    s3_results: Optional[List[Dict]] = None

    if 1 in steps:
        t1 = time.time()
        run_step1(cfg)
        print(f"  [Paso 1 completado en {time.time()-t1:.0f}s]")

    if 2 in steps:
        t1 = time.time()
        s2_results = run_step2(cfg)
        print(f"  [Paso 2 completado en {time.time()-t1:.0f}s]")

    if 3 in steps:
        t1 = time.time()
        s3_results = run_step3(cfg, s2_results)
        print(f"  [Paso 3 completado en {time.time()-t1:.0f}s]")

    if 4 in steps:
        t1 = time.time()
        run_step4(cfg)
        print(f"  [Paso 4 completado en {time.time()-t1:.0f}s]")

    if 5 in steps:
        t1 = time.time()
        run_step5(cfg)
        print(f"  [Paso 5 completado en {time.time()-t1:.0f}s]")

    total = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Pipeline completado en {total:.0f}s ({total/60:.1f} min)")
    print(f"  Resultados: {ROOT / cfg['experiment']['output_dir']}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
