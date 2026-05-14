#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parent
TARGET_Q = 0.9998
OUTDIR = ROOT / "outputs" / "q_audit"
CONFIG_OUT = ROOT / "configs" / "lure_biased_multiparam_q09998.yaml"

SEARCH_SPECS = [
    "configs",
    "outputs",
    "outputs/lure_route",
    "outputs/extended_search",
    "runs*",
    "chua_piecewise",
    "df_seed_comparison",
    "q_order_sweep",
]
EXTENSIONS = {".json", ".csv", ".yaml", ".yml", ".txt", ".md"}
KEY_RE = re.compile(
    r"(?i)\b(q|frac_order|HIDDEN_ATTRACTORS_FRAC_ORDER)\b\s*(?:[:=,]\s*)?[\"']?"
    r"([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"
)
KNOWN_Q_RE = re.compile(r"(?<![0-9.])(0\.9998|0\.999|0\.99)(?![0-9.])")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def iter_roots() -> List[Path]:
    roots: List[Path] = []
    seen = set()
    for spec in SEARCH_SPECS:
        matches = list(ROOT.glob(spec)) if "*" in spec else [ROOT / spec]
        for path in matches:
            if not path.exists():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            roots.append(path)
    return roots


def iter_files() -> Iterable[Path]:
    seen = set()
    for root in iter_roots():
        if root.is_file():
            candidates = [root]
        else:
            candidates = root.rglob("*")
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return f"__READ_ERROR__ {exc}"


def collect_q_values(text: str) -> List[float]:
    values: List[float] = []
    for match in KEY_RE.finditer(text):
        try:
            values.append(float(match.group(2)))
        except ValueError:
            pass
    for match in KNOWN_Q_RE.finditer(text):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            pass
    out: List[float] = []
    for value in values:
        if not any(abs(value - old) <= 5e-10 for old in out):
            out.append(value)
    return out


def classify(values: Sequence[float]) -> str:
    if not values:
        return "q_missing"
    rounded = {round(float(v), 10) for v in values}
    target = round(TARGET_Q, 10)
    if len(rounded) > 1:
        return "q_mixed"
    if target in rounded:
        return "q_0p9998_correct_for_new_exploration"
    return "q_not_0p9998_historical_do_not_use_as_input"


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_clean_config() -> None:
    CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    text = """# Clean q=0.9998 configuration for biased Lure multiparameter exploration.
model: piecewise
q: 0.9998
frac_order: 0.9998
enforce_q_consistency: true

params:
  alpha_chua: 8.4562
  beta: 12.0732
  gamma_chua: 0.0052
  m0: -0.1768
  m1: -1.1468
  a1: 0.4
  a2: -1.5585
  rho: 1.0

outputs:
  root: outputs/lure_biased_multiparam_q09998

search:
  A_min: 0.5
  A_max: 8.0
  omega_min: 1.2
  omega_max: 3.0
  sigma0_min: -4.0
  sigma0_max: 4.0
  n_samples: 5000
  quadrature_points: 4096
  K_rhoH: 20
  local_refine_top: 100
  residual_keep: 0.05
  rhoH_keep: 0.30
  residual_priority: 0.02
  rhoH_priority: 0.15
  random_seed: 20260514
  source_hint_manifest: outputs/lure_route/lure_candidates_manifest.csv

seeds:
  phases:
    - 0.0
    - 0.7853981633974483
    - 1.5707963267948966
    - 2.356194490192345
    - 3.141592653589793
    - 3.9269908169872414
    - 4.71238898038469
    - 5.497787143782138

continuation:
  enabled: false
  routes:
    - C1
  max_candidates: 6
  max_seeds_per_candidate: 1
  eta_start: 0.0
  eta_target: 1.0
  eta_steps: 9
  q_fixed: 0.9998
  h: 0.01
  memory_length: 20
  t_block: 200
  n_blocks: 8
  survivor_h: 0.01
  survivor_memory_length: 40
  survivor_t_final: 1500
  smooth_width: 0.2
  divergence_norm: 120
  equilibrium_tol: 0.001
  nontrivial_range: 0.01

early_equilibrium_filter:
  enabled: false
  rho: 1.0e-5
  h: 0.01
  memory_length: 40
  t_final: 1500

robustness:
  enabled: false
  cases:
    R0:
      h: 0.01
      memory_length: 40
      t_final: 3000
    R1:
      h: 0.005
      memory_length: 40
      t_final: 3000
    R2:
      h: 0.005
      memory_length: 80
      t_final: 6000

cost_guard:
  max_simulations_without_force: 2000
  max_estimated_hours_without_force: 12
"""
    CONFIG_OUT.write_text(text, encoding="utf-8")


def main() -> None:
    rows: List[Dict[str, Any]] = []
    for path in sorted(iter_files(), key=lambda p: rel(p).lower()):
        text = load_text(path)
        values = collect_q_values(text)
        status = classify(values)
        rows.append(
            {
                "path": rel(path),
                "extension": path.suffix.lower(),
                "classification": status,
                "q_values": ";".join(f"{v:.10g}" for v in values),
                "q_value_count": len(values),
                "size_bytes": path.stat().st_size if path.exists() else "",
            }
        )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fields = ["path", "extension", "classification", "q_values", "q_value_count", "size_bytes"]
    write_csv(OUTDIR / "q_audit_table.csv", rows, fields)

    counts = Counter(row["classification"] for row in rows)
    summary = {
        "target_q": TARGET_Q,
        "files_scanned": len(rows),
        "classification_counts": dict(sorted(counts.items())),
        "search_specs": SEARCH_SPECS,
        "config_written": rel(CONFIG_OUT),
        "notes": [
            "Historical files are not modified.",
            "Files classified as q_not_0p9998_historical_do_not_use_as_input must not feed the new q=0.9998 exploration except as source_hint_only_q_mismatch.",
            "Files classified as q_mixed need manual review before being used as numerical input.",
        ],
    }
    (OUTDIR / "q_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# q audit for biased Lure multiparameter exploration",
        "",
        f"Target q for the new exploration: `{TARGET_Q}`.",
        "",
        "Historical files were not modified.",
        "",
        "## Classification counts",
        "",
    ]
    for key, count in sorted(counts.items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(
        [
            "",
            "## Classification policy",
            "",
            "- `q_0p9998_correct_for_new_exploration`: explicit q is 0.9998.",
            "- `q_not_0p9998_historical_do_not_use_as_input`: explicit q exists but is not 0.9998; use only as `source_hint_only_q_mismatch` if needed.",
            "- `q_missing`: no explicit q token was found.",
            "- `q_mixed`: multiple distinct q values were found.",
            "",
            "## Files requiring caution",
            "",
        ]
    )
    cautious = [row for row in rows if row["classification"] != "q_0p9998_correct_for_new_exploration"]
    for row in cautious[:200]:
        lines.append(f"- `{row['classification']}` `{row['path']}` q_values=`{row['q_values']}`")
    if len(cautious) > 200:
        lines.append(f"- ... {len(cautious) - 200} more rows in `q_audit_table.csv`")
    (OUTDIR / "q_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_clean_config()
    print(f"q_audit_report={rel(OUTDIR / 'q_audit_report.md')}")
    print(f"q_audit_table={rel(OUTDIR / 'q_audit_table.csv')}")
    print(f"q_audit_summary={rel(OUTDIR / 'q_audit_summary.json')}")
    print(f"clean_config={rel(CONFIG_OUT)}")


if __name__ == "__main__":
    main()
