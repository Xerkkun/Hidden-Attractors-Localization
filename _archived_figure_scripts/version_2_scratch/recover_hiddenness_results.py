#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruye result.json y reporte global a partir de los run.log ya existentes.

Las integraciones Caputo ya terminaron correctamente; solo falló el print
de un carácter Unicode. Este script parsea los run.log y produce los mismos
artefactos que habría generado el script original si hubiera terminado bien.
"""

from __future__ import annotations
import json, re, sys
from pathlib import Path

HID_BASE = Path("outputs/biased_saturation_search_q09998_corrected/hiddenness")

CANDIDATES_META = [
    {"m1": -1.1468, "m0": -0.1768, "c":  2.776,
     "prefix": "biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776"},
    {"m1": -1.1468, "m0": -0.200,  "c": -2.705,
     "prefix": "biased_q9998_m1_m1p1468_m0_m0p2000_branch_1_c_m2p705"},
    {"m1": -1.1468, "m0": -0.240,  "c": -2.581,
     "prefix": "biased_q9998_m1_m1p1468_m0_m0p2400_branch_1_c_m2p581"},
]

RADII         = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2]
SAMPLES_PER_R = [20, 25, 30, 40, 50, 60]
EQUILIBRIA_ORDER = ["E0", "E+", "E-"]

# ── Parser de run.log ─────────────────────────────────────────────────────────

def parse_run_log(log_path: Path):
    """
    Extrae las filas FINAL por (equilibrio, radio) y el veredicto si existe.
    Devuelve: List[dict], List[str] equilibria_found
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
    records = []
    eq_order = []

    # Cada bloque empieza con:  [m1=... m0=... c=...] [EQ]  radio=R  n=N
    block_re = re.compile(
        r'\[m1=[\d\.\-]+ m0=[\d\.\-]+ c=[+\-\d\.]+\] \[(\S+)\]\s+radio=([\deE\.\-\+]+)\s+n=(\d+)'
    )
    final_re = re.compile(
        r"FINAL \(\d+ muestras\): \{([^}]+)\}"
    )

    blocks = list(block_re.finditer(text))
    finals = list(final_re.finditer(text))

    for i, blk in enumerate(blocks):
        eq_name = blk.group(1)
        radius  = float(blk.group(2))
        n_plan  = int(blk.group(3))
        if eq_name not in eq_order:
            eq_order.append(eq_name)

        # Buscar el FINAL que sigue a este bloque (antes del siguiente bloque)
        start = blk.end()
        end   = blocks[i+1].start() if i+1 < len(blocks) else len(text)
        segment = text[start:end]
        fm = final_re.search(segment)
        if fm:
            # Parsear el dict-string
            kv_re = re.compile(r"'(\w+)':\s*(\d+)")
            stats = {m.group(1): int(m.group(2)) for m in kv_re.finditer(fm.group(1))}
            records.append({
                "equilibrium": eq_name,
                "radius":      radius,
                "samples":     sum(stats.values()),
                "TARGET":      stats.get("target_attractor", 0),
                "EQ":          stats.get("stable_equilibrium", 0),
                "OTHER":       stats.get("other_attractor", 0),
                "DIV":         stats.get("divergence", 0),
                "FAIL":        stats.get("numerical_failure", 0),
            })

    return records, eq_order


def derive_contract(records, eq_order, prefix, m1, m0, c):
    """Evalúa el contrato de ocultedad sobre los registros parseados."""
    target_hits_total = sum(r["TARGET"] for r in records)
    samples_total     = sum(r["samples"] for r in records)
    self_excited      = target_hits_total > 0
    all_eqs_covered   = all(eq in [r["equilibrium"] for r in records] for eq in eq_order)
    all_radii_covered = all(
        any(abs(r["radius"] - rad) < 1e-12 for r in records)
        for rad in RADII
    )

    # Si hay target hits → autoexcitado; de lo contrario → compatible con ocultedad
    if self_excited:
        status = "SELF_EXCITED_DETECTED"
        hidden_compatible = False
        hidden_verified   = False
    elif all_eqs_covered and all_radii_covered:
        status = "HIDDEN_COMPATIBLE"
        hidden_compatible = True
        hidden_verified   = False   # la verificación numérica no constituye prueba
    else:
        status = "INCOMPLETE_COVERAGE"
        hidden_compatible = False
        hidden_verified   = False

    failed = []
    if not all_radii_covered:
        failed.append("not all required radii covered")
    if not all_eqs_covered:
        failed.append("not all equilibria covered")

    return {
        "prefix":                      prefix,
        "m1": m1, "m0": m0, "c": c,
        "hiddenness_status":            status,
        "hidden_verified":              hidden_verified,
        "hidden_compatible":            hidden_compatible,
        "self_excited_contact_detected": self_excited,
        "target_hits_total":            target_hits_total,
        "samples_total":                samples_total,
        "equilibria_tested":            eq_order,
        "failed_requirements":          failed,
        "records":                      records,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

all_results = []

for meta in CANDIDATES_META:
    prefix   = meta["prefix"]
    log_path = HID_BASE / prefix / "run.log"
    print(f"\n{'='*60}")
    print(f"Procesando: {prefix}")

    if not log_path.exists():
        print(f"  [ERROR] run.log no encontrado: {log_path}")
        continue

    records, eq_order = parse_run_log(log_path)
    print(f"  Equilibrios encontrados: {eq_order}")
    print(f"  Bloques parseados:       {len(records)}")
    for r in records:
        print(f"    [{r['equilibrium']}] r={r['radius']:.0e}  "
              f"n={r['samples']}  TARGET={r['TARGET']}  "
              f"EQ={r['EQ']}  OTHER={r['OTHER']}  DIV={r['DIV']}")

    contract = derive_contract(records, eq_order, prefix, meta["m1"], meta["m0"], meta["c"])

    # Guardar result.json
    result_path = HID_BASE / prefix / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)
    print(f"  result.json guardado en: {result_path}")

    print(f"\n  VEREDICTO: {contract['hiddenness_status']}")
    print(f"  hidden_compatible:        {contract['hidden_compatible']}")
    print(f"  self_excited_contact:     {contract['self_excited_contact_detected']}")
    print(f"  TARGET hits totales:      {contract['target_hits_total']} / {contract['samples_total']}")

    all_results.append(contract)

# ── Reporte global ─────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("REPORTE GLOBAL")
print(f"{'='*60}")

lines = [
    "# Pruebas de Ocultedad — Candidatos Sesgados No Centrados\n",
    "> [!WARNING]",
    "> La ausencia de contacto con las vecindades ensayadas **no es prueba matematica",
    "> global de ocultedad**, sino verificacion numerica bajo los radios, integrador",
    "> y tiempos declarados.\n",
    "## Parametros del Test\n",
    "| Parametro | Valor |",
    "|:---|:---|",
    "| q | 0.9998 |",
    "| h | 0.01 |",
    "| Integrador | Caputo ABM memoria completa |",
    "| Radios | 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2 |",
    "| Muestras/radio | 20, 25, 30, 40, 50, 60 (225/equilibrio) |",
    "| t_final | 200 s |",
    "| t_burn | 50 s |",
    "| Parada temprana | NO — muestras completas |",
    "| Metrica match | nn_percentile (perc=90%, tol=0.5) |",
    "",
    "## Resultados\n",
    "| Candidato | m1 | m0 | c | TARGET hits | Total muestras | Status | Compatible |",
    "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
]

for r in all_results:
    compat = "SI" if r["hidden_compatible"] else "NO"
    lines.append(
        f"| `{r['prefix']}` | {r['m1']} | {r['m0']} | {r['c']:.3f} | "
        f"**{r['target_hits_total']}** | {r['samples_total']} | "
        f"`{r['hiddenness_status']}` | {compat} |"
    )

lines += ["", "## Detalle por Equilibrio\n"]
for r in all_results:
    lines += [f"### `{r['prefix']}`\n"]
    lines.append("| Equilibrio | Radio | Muestras | TARGET | EQ | OTHER | DIV |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for rec in r["records"]:
        lines.append(
            f"| {rec['equilibrium']} | {rec['radius']:.0e} | {rec['samples']} | "
            f"**{rec['TARGET']}** | {rec['EQ']} | {rec['OTHER']} | {rec['DIV']} |"
        )
    lines += [
        "",
        f"- **TARGET hits totales:** {r['target_hits_total']} / {r['samples_total']}",
        f"- **self_excited_contact:** {r['self_excited_contact_detected']}",
        f"- **Veredicto:** `{r['hiddenness_status']}`",
        "",
    ]

lines += ["## Conclusion Operacional\n"]
any_compat = any(r["hidden_compatible"] for r in all_results)
any_sex    = any(r["self_excited_contact_detected"] for r in all_results)

if any_sex:
    # Cuales son autoexcitados
    sexcited = [r["prefix"] for r in all_results if r["self_excited_contact_detected"]]
    hidden   = [r["prefix"] for r in all_results if r["hidden_compatible"]]
    lines.append(f"Candidatos con contacto autoexcitado detectado: {sexcited}")
    if hidden:
        lines.append(f"Candidatos compatibles con ocultedad: {hidden}")
elif any_compat:
    lines.append(
        "Ningun candidato sesgado mostro contacto con la vecindad de los "
        "equilibrios en los radios ensayados. Son compatibles con la ocultedad "
        "bajo el contrato declarado."
    )
else:
    lines.append("Todos los candidatos fallaron los requerimientos del contrato.")

report_path = HID_BASE / "hiddenness_global_summary.md"
report_path.write_text("\n".join(lines), encoding="utf-8")

json_path = HID_BASE / "hiddenness_global_summary.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print(f"  Reporte global: {report_path}")
print(f"  JSON global:    {json_path}")

print(f"\n{'='*60}")
print("RESUMEN FINAL")
print(f"{'='*60}")
for r in all_results:
    print(f"  [{r['m1']}, {r['m0']}, c={r['c']:+.3f}]  "
          f"TARGET={r['target_hits_total']}/{r['samples_total']}  "
          f"STATUS={r['hiddenness_status']}")
