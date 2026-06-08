"""
PLANTILLA DE EXPLORACION – Chua Fraccionario (Atractores Ocultos)
==================================================================
Plantilla genérica para buscar atractores caoticos ocultos en sistemas
de tipo Chua fraccionario usando Funcion Descriptiva (DF) + Continuacion
Numerica fraccionaria (ABM) + diagnostico de periodicidad.

FLUJO:
  1. Funcion Descriptiva (DF) fraccionaria  → semilla x0, omega0, k
  2. Continuacion numerica ABM fraccionaria → refinamiento de la semilla
  3. Simulacion larga con ABM fraccionario  → trayectoria candidata
  4. Diagnostico de periodicidad            → clasificacion del candidato
  5. (Opcional) Comparacion EFORK full/trunc para candidatos robustos

PARAMETROS AJUSTABLES (ver seccion "CONFIGURACION"):
  - modelo      : "arctan" | "saturation"
  - q_dynamics  : orden fraccionario (0 < q <= 1)
  - memory_mode : "full" | "window"  (Lm en segundos si es window)
  - grid params : rangos de a1/a2/rho (arctan) o m1/m0 (saturation)
  - t_final, t_transient, h
  - Lm          : longitud de memoria truncada en segundos

DEPENDENCIAS:
  - version_2/hidden_attractors/ (paquete del proyecto)
  - .venv con numpy, matplotlib, scipy

USO:
  .venv\\Scripts\\python exploration_template.py --model arctan --memory-mode full
  .venv\\Scripts\\python exploration_template.py --model saturation --memory-mode window --Lm 10.0
"""

import sys
import csv
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "version_2"))

from hidden_attractors.models.chua import ChuaParameters
from hidden_attractors.continuation.continuation_fractional import run_fractional_continuation
from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.systems.builtins import _chua_lure_system
from hidden_attractors.diagnostics.periodicity import classify_post_transient_periodicity

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTAR FUNCION DESCRIPTIVA SEGUN MODELO
# ──────────────────────────────────────────────────────────────────────────────
# Descomenta la que corresponda al modelo que quieras explorar:

# Para Arctan (Wu 2023):
from hidden_attractors.seed_generation.chua_arctan_wu2023 import find_centered_arctan_wu2023_branches as find_branches

# Para Saturacion (No Suave):
# from hidden_attractors.seed_generation.chua_saturation import find_saturation_branches as find_branches


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACION — AJUSTA ESTOS PARAMETROS
# ══════════════════════════════════════════════════════════════════════════════

def get_config(args):
    cfg = SimpleNamespace()

    # ── Integrador fraccionario ───────────────────────────────────────────────
    cfg.model       = args.model          # "arctan" | "saturation"
    cfg.q_dynamics  = 0.99                # orden fraccionario Caputo
    cfg.h           = 0.01                # paso de tiempo
    cfg.t_final     = 300.0              # tiempo total de simulacion
    cfg.t_transient = 100.0              # tiempo de transiente a descartar
    cfg.memory_mode = args.memory_mode    # "full" | "window"
    cfg.Lm_seconds  = args.Lm            # longitud de ventana en segundos (solo para window)

    if cfg.memory_mode == "window":
        cfg.memory_window_length = int(cfg.Lm_seconds / cfg.h)
    else:
        cfg.memory_window_length = None

    # ── Parametros de la rejilla de busqueda ─────────────────────────────────
    # MODELO ARCTAN (descomentar y ajustar):
    cfg.a1_values  = [0.1, 0.2]
    cfg.a2_values  = [-1.0, -1.2, -1.5585, -2.0, -2.5, -3.0]
    cfg.rho_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]

    # MODELO SATURACION (descomentar y ajustar):
    # cfg.m1_values = [-0.8, -1.0, -1.2, -1.4, -1.6]
    # cfg.m0_values = [-0.1, -0.2, -0.3, -0.4]

    # Parametros fijos del sistema Chua
    cfg.chua_fixed = dict(
        alpha=8.4562,
        beta=12.0732,
        gamma=0.0052,
    )

    # ── Continuacion ─────────────────────────────────────────────────────────
    cfg.eta_values = np.linspace(0.0, 1.0, 11)   # lambda de continuacion
    cfg.t_cont_trans = 30.0                        # transiente en cada paso
    cfg.t_cont_keep  = 30.0                        # ventana que se conserva

    # ── DF scan ──────────────────────────────────────────────────────────────
    cfg.nscan = 5000           # puntos en el scan de frecuencia
    cfg.df_transfer_mode = "fractional_spectral"

    # ── Salida ───────────────────────────────────────────────────────────────
    cfg.output_dir = ROOT / "outputs" / args.output_dir
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def build_system(model, chua_fixed, **nonlin_params):
    """Construye el namespace del sistema para el integrador."""
    params_dict = {**chua_fixed, "model": model, **nonlin_params}
    params = ChuaParameters(model=model, **{k: v for k,v in params_dict.items() if k!="model"})
    lure = _chua_lure_system(params_dict)
    return SimpleNamespace(
        system_id=f"chua_fractional_{model}",
        name=f"chua_fractional_{model}",
        parameters=params_dict,
        lure=lure,
    )


def run_continuation(system, branch, cfg):
    """Ejecuta la continuacion ABM fraccionaria."""
    steps = run_fractional_continuation(
        system=system,
        seed_x0=branch.seed,
        k_gain=branch.gain,
        lambda_values=cfg.eta_values,
        h=cfg.h,
        memory_mode=cfg.memory_mode,
        memory_window_length=cfg.memory_window_length,
        integrator="abm",
        use_c_backend=True,
        require_c_backend=False,
        allow_python_fallback=True,
        t_transient=cfg.t_cont_trans,
        t_keep=cfg.t_cont_keep,
        q=cfg.q_dynamics,
    )
    return steps


def run_simulation(system, x0, cfg):
    """Ejecuta la simulacion larga con ABM fraccionario."""
    def rhs(t, v):
        L = system.lure
        return L.matrix @ v + L.input_vector * L.nonlinearity(L.output_vector @ v)

    times, states, status, info = fractional_integrate(
        rhs=rhs,
        x0=x0,
        q=cfg.q_dynamics,
        h=cfg.h,
        t_final=cfg.t_final,
        method="abm",
        memory_mode=cfg.memory_mode,
        memory_window_length=cfg.memory_window_length,
        system=system,
        use_c_backend=True,
        allow_python_fallback=True,
    )
    return times, states, status


def classify(times, states, cfg):
    """Ejecuta el diagnostico de periodicidad post-transiente."""
    traj_data = np.column_stack((times, states))
    diag = classify_post_transient_periodicity(
        traj_data,
        h=cfg.h,
        config={"t_transient": cfg.t_transient}
    )
    return diag["candidate_label"]


def save_plot(states, times, case_id, verdict, cfg):
    """Guarda figura 3D del espacio de fase."""
    n_burn = int(cfg.t_transient / cfg.h)
    tail = states[n_burn:]
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tail[:,0], tail[:,1], tail[:,2], lw=0.4, color="crimson", alpha=0.8)
    ax.set_title(f"{case_id}\n{verdict}", fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    out = cfg.output_dir / f"{case_id}_phase3d.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def save_traj(times, states, case_id, cfg):
    """Guarda la trayectoria como CSV."""
    data = np.column_stack((times, states))
    out = cfg.output_dir / f"{case_id}_trajectory.csv"
    np.savetxt(out, data, delimiter=",", header="t,x,y,z", comments="")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# GENERADOR DE REJILLA DE PARAMETROS
# ══════════════════════════════════════════════════════════════════════════════

def param_grid(cfg):
    """
    Genera los pares (params_dict, nonlin_params) para iterar.
    AJUSTA ESTA FUNCION SEGUN EL MODELO QUE EXPLORES.
    """
    if cfg.model == "arctan":
        for a1 in cfg.a1_values:
            for a2 in cfg.a2_values:
                for rho in cfg.rho_values:
                    nonlin = dict(a1=a1, a2=a2, rho=rho)
                    yield nonlin
    elif cfg.model == "saturation":
        for m1 in cfg.m1_values:
            for m0 in cfg.m0_values:
                nonlin = dict(m1=m1, m0=m0)
                yield nonlin
    else:
        raise ValueError(f"Modelo desconocido: {cfg.model}")


def nonlin_to_case_id(nonlin):
    """Convierte dict de parametros a un identificador de caso limpio."""
    parts = []
    for k, v in sorted(nonlin.items()):
        v_str = f"{abs(v):.4f}".replace(".", "p")
        sign = "m" if v < 0 else ""
        parts.append(f"{k}_{sign}{v_str}")
    return "_".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# BUCLE PRINCIPAL DE EXPLORACION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Plantilla de exploración de atractores ocultos en Chua fraccionario."
    )
    parser.add_argument("--model", default="arctan",
                        choices=["arctan","saturation"],
                        help="Modelo de no linealidad")
    parser.add_argument("--memory-mode", default="full",
                        choices=["full","window"],
                        help="Modo de memoria Caputo")
    parser.add_argument("--Lm", type=float, default=10.0,
                        help="Longitud de ventana en segundos (solo para window)")
    parser.add_argument("--output-dir", default="outputs/exploration",
                        help="Directorio de salida")
    args = parser.parse_args()

    cfg = get_config(args)

    print(f"\n{'='*65}")
    print(f"EXPLORACION: {cfg.model.upper()}  |  memoria={cfg.memory_mode}  |  q={cfg.q_dynamics}")
    print(f"Salida: {cfg.output_dir}")
    print(f"{'='*65}\n")

    results = []
    case_idx = 0

    for nonlin in param_grid(cfg):

        # 1. Construir sistema
        system = build_system(cfg.model, cfg.chua_fixed, **nonlin)

        # 2. Funcion Descriptiva fraccionaria → semillas
        try:
            # ⚠️  Ajusta el llamado segun el modelo:
            #   - arctan : find_branches(q, params, transfer_mode, nscan)
            #   - saturation: find_branches(q, params, ...)
            params_obj = ChuaParameters(model=cfg.model,
                                        **cfg.chua_fixed, **nonlin)
            branches = find_branches(
                q=cfg.q_dynamics,
                params=params_obj,
                transfer_mode=cfg.df_transfer_mode,
                nscan=cfg.nscan,
            )
        except Exception as e:
            key = nonlin_to_case_id(nonlin)
            print(f"[DF ERROR] {key}: {e}")
            continue

        for branch in branches:
            case_idx += 1
            base_id  = nonlin_to_case_id(nonlin)
            case_id  = f"{base_id}_branch_{branch.branch_index}"
            nonlin_str = " | ".join(f"{k}={v:.4f}" for k,v in sorted(nonlin.items()))
            print(f"\n[{case_idx}] {case_id}")
            print(f"      omega0={branch.omega:.4f}  k={branch.gain:.4f}  A0={branch.amplitude:.4f}")

            row = dict(case_id=case_id, **nonlin,
                       omega0=branch.omega, k=branch.gain, A0=branch.amplitude,
                       cont_status="", sim_status="", verdict="", final_state="")

            try:
                # 3. Continuacion ABM
                steps = run_continuation(system, branch, cfg)
                final = steps[-1]
                row["cont_status"] = final["status"]
                print(f"      continuacion: {final['status']}")

                if final["status"] != "ok":
                    row["verdict"] = "continuation_failed"
                    results.append(row)
                    continue

                x0 = final["x_out"].copy()
                row["final_state"] = x0.tolist()

                # 4. Simulacion larga
                times, states, sim_status = run_simulation(system, x0, cfg)
                row["sim_status"] = sim_status
                print(f"      simulacion:   {sim_status}")

                # 5. Diagnostico
                verdict = classify(times, states, cfg)
                row["verdict"] = verdict
                print(f"      veredicto:    {verdict}")

                # 6. Guardar si es candidato
                if verdict in ["chaotic_candidate_pending_robustness",
                               "nonperiodic_candidate"]:
                    save_traj(times, states, case_id, cfg)
                    out_fig = save_plot(states, times, case_id, verdict, cfg)
                    print(f"      -> Guardado: {out_fig.name}")

            except Exception as e:
                row["verdict"] = f"error: {e}"
                print(f"      [ERR] {e}")

            results.append(row)

    # 7. Escribir resumen CSV
    if results:
        fieldnames = list(results[0].keys())
        csv_path = cfg.output_dir / "summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f"\n[DONE] summary escrito en {csv_path}")

    # 8. Estadisticas finales
    chaotic  = [r for r in results if "chaotic" in r.get("verdict","")]
    nonper   = [r for r in results if "nonperiodic" in r.get("verdict","")]
    print(f"\n{'='*55}")
    print(f"Total casos evaluados    : {len(results)}")
    print(f"Caoticos (chaotic_cand.) : {len(chaotic)}")
    print(f"No periodicos            : {len(nonper)}")
    print(f"Rechazados               : {len(results)-len(chaotic)-len(nonper)}")
    print(f"{'='*55}")
    for r in chaotic + nonper:
        print(f"  * {r['case_id']}  [{r['verdict']}]")


if __name__ == "__main__":
    main()
