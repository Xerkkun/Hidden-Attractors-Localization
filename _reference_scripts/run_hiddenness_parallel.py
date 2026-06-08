#!/usr/bin/env python3
"""Lanzador paralelo: ejecuta los 3 candidatos sesgados como subprocesos independientes.

Lanza hiddenness_single_candidate.py 0, 1 y 2 en paralelo y espera a que
todos terminen antes de construir el reporte global.

Uso:
    python run_hiddenness_parallel.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Configuración ──────────────────────────────────────────────────────────────

SCRIPT = Path(__file__).resolve().parent / "hiddenness_single_candidate.py"
HID_BASE = Path("outputs/biased_saturation_search_q09998_corrected/hiddenness")

CANDIDATE_PREFIXES = [
    "biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776",
    "biased_q9998_m1_m1p1468_m0_m0p2000_branch_1_c_m2p705",
    "biased_q9998_m1_m1p1468_m0_m0p2400_branch_1_c_m2p581",
]

CANDIDATE_LABELS = [
    "m1=-1.1468  m0=-0.1768  c=+2.776",
    "m1=-1.1468  m0=-0.2000  c=-2.705",
    "m1=-1.1468  m0=-0.2400  c=-2.581",
]


# ── Lanzamiento ────────────────────────────────────────────────────────────────

def launch_processes():
    python = sys.executable
    processes = []

    print("=" * 70)
    print("LANZADOR PARALELO — PRUEBAS DE OCULTEDAD")
    print("=" * 70)
    print(f"Script worker: {SCRIPT}")
    print()

    for idx, label in enumerate(CANDIDATE_LABELS):
        log_path = HID_BASE / CANDIDATE_PREFIXES[idx] / "stdout.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logf = open(log_path, "w", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            [python, str(SCRIPT), str(idx)],
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        processes.append({"proc": proc, "logf": logf, "label": label,
                          "log_path": log_path, "idx": idx})
        print(f"  [PID {proc.pid:6d}] Candidato {idx}: {label}")

    print()
    print(f"  3 procesos corriendo.  Esperando...")
    print(f"  Logs en:  {HID_BASE}/<prefix>/stdout.log")
    print()
    return processes


def wait_all(processes):
    t0 = time.time()
    pending = set(range(len(processes)))

    while pending:
        for i in list(pending):
            p = processes[i]
            ret = p["proc"].poll()
            if ret is not None:
                elapsed = time.time() - t0
                status = "OK" if ret == 0 else f"ERROR(rc={ret})"
                print(f"  [{elapsed:6.0f}s] Candidato {p['idx']} terminó: {status}  — {p['label']}")
                p["logf"].close()
                pending.discard(i)

        if pending:
            time.sleep(10)  # polling cada 10 s

    print(f"\n  Todos los procesos terminaron en {time.time()-t0:.0f}s")


# ── Reporte global ─────────────────────────────────────────────────────────────

def collect_results():
    results = []
    for prefix in CANDIDATE_PREFIXES:
        result_path = HID_BASE / prefix / "result.json"
        if result_path.exists():
            with open(result_path, encoding="utf-8") as f:
                results.append(json.load(f))
        else:
            results.append({
                "prefix": prefix,
                "hiddenness_status": "ERROR_NO_RESULT",
                "hidden_verified": False,
                "hidden_compatible": False,
                "self_excited_contact": None,
                "target_hits": -1,
                "samples": -1,
                "equilibria_tested": [],
            })
    return results


def write_global_report(results):
    lines = [
        "# Pruebas de Ocultedad — Candidatos Sesgados No Centrados\n",
        "> [!WARNING]",
        "> La ausencia de contacto con las vecindades ensayadas **no es prueba matemática",
        "> global de ocultedad**, sino verificación numérica bajo los radios, integrador",
        "> y tiempos declarados.\n",
        "## Parámetros del Test\n",
        "| Parámetro | Valor |",
        "|:---|:---|",
        "| q | 0.9998 |",
        "| h | 0.01 |",
        "| Integrador | Caputo ABM memoria completa |",
        "| Radios | 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2 |",
        "| Muestras/radio | 20, 25, 30, 40, 50, 60 (225/equilibrio) |",
        "| t_final | 200 s |",
        "| t_burn | 50 s |",
        "| Parada temprana | NO — muestras completas |",
        "| Métrica match | nn_percentile (perc=90%, tol=0.5) |",
        "",
        "## Resultados\n",
        "| Candidato | m1 | m0 | c | TARGET hits | Total muestras | Status | Compatible |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for r in results:
        eq_str = ", ".join(r.get("equilibria_tested", []))
        hits   = r["target_hits"]
        samp   = r["samples"]
        compat = "✅" if r["hidden_compatible"] else "❌"
        status = r["hiddenness_status"]
        m1, m0, c = r.get("m1","?"), r.get("m0","?"), r.get("c","?")
        lines.append(
            f"| `{r['prefix']}` | {m1} | {m0} | {c} | "
            f"**{hits}** | {samp} | `{status}` | {compat} |"
        )

    lines += ["", "## Detalle por Candidato\n"]
    for r in results:
        lines += [
            f"### `{r['prefix']}`\n",
            f"- **hidden_verified:** {r['hidden_verified']}",
            f"- **hidden_compatible:** {r['hidden_compatible']}",
            f"- **self_excited_contact:** {r.get('self_excited_contact', '?')}",
            f"- **target_hits / samples:** {r['target_hits']} / {r['samples']}",
            f"- **equilibria_tested:** {', '.join(r.get('equilibria_tested',[]))}",
            f"- **log:** `{HID_BASE / r['prefix'] / 'run.log'}`",
            "",
        ]

    lines += [
        "## Conclusión Operacional\n",
    ]
    any_compat = any(r["hidden_compatible"] for r in results)
    any_sex    = any(r.get("self_excited_contact", False) for r in results)

    if any_sex:
        lines.append(
            "⚠️ Al menos un candidato sesgado mostró **contacto autoexcitado** con la "
            "vecindad de un equilibrio."
        )
    elif any_compat:
        lines.append(
            "✅ Ningún candidato sesgado mostró contacto con la vecindad de los "
            "equilibrios en los radios ensayados. Son **compatibles con la ocultedad** "
            "bajo el contrato declarado."
        )
    else:
        lines.append(
            "❌ Todos los candidatos fallaron los requerimientos del contrato de "
            "verificación."
        )

    report_path = HID_BASE / "hiddenness_global_summary.md"
    HID_BASE.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = HID_BASE / "hiddenness_global_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Reporte global: {report_path}")
    print(f"  JSON global:    {json_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    procs = launch_processes()
    wait_all(procs)

    results = collect_results()
    write_global_report(results)

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    for r in results:
        print(f"  [{r.get('m1','?')}, {r.get('m0','?')}, c={r.get('c','?')}]  "
              f"TARGET_HITS={r['target_hits']}  STATUS={r['hiddenness_status']}")

    print()
    print("[ADVERTENCIA] Estos resultados no constituyen prueba matemática global de ocultedad.")
