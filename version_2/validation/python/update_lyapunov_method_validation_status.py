#!/usr/bin/env python3
"""Promote DK2018 published-value evidence only after both native cases pass."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_CASES = {
    "published_dk2018_rabinovich_fabrikant_q0999",
    "published_dk2018_lorenz_q0985",
}
METHOD_ID = "fractional_variational_dk2018_block_restart_abm_gs"


def _load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def promote(runtime_output_dir: Path, official_dir: Path) -> dict[str, object]:
    cases_csv = runtime_output_dir / "benchmark_cases.csv"
    rows = _load_rows(cases_csv)
    missing = sorted(REQUIRED_CASES - rows.keys())
    if missing:
        raise RuntimeError(f"Cannot promote DK2018 evidence; missing cases: {missing}")
    for case_id in sorted(REQUIRED_CASES):
        row = rows[case_id]
        if row.get("status") != "published_benchmark_passed_quantitative":
            raise RuntimeError(f"Cannot promote {case_id}; status is {row.get('status')!r}.")
        if row.get("numerical_route") != "native_c":
            raise RuntimeError(f"Cannot promote {case_id}; numerical_route must be native_c.")
        if row.get("execution_contract") != "dk2018_block_restart_abm_gs":
            raise RuntimeError(f"Cannot promote {case_id}; execution contract does not match DK2018.")
    convergence_sources = {
        case_id: runtime_output_dir / "convergence" / f"{case_id}.csv"
        for case_id in sorted(REQUIRED_CASES)
    }
    for case_id, source in convergence_sources.items():
        if not source.exists():
            raise RuntimeError(f"Cannot promote {case_id}; convergence CSV is missing.")

    official_dir.mkdir(parents=True, exist_ok=True)
    convergence_dir = official_dir / "convergence"
    convergence_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cases_csv, official_dir / "published_benchmark_results.csv")
    for case_id, source in convergence_sources.items():
        shutil.copy2(source, convergence_dir / source.name)

    summary = {
        "schema_version": "1.0",
        "method_id": METHOD_ID,
        "status": "published_quantitative_validated",
        "required_cases": sorted(REQUIRED_CASES),
        "certifications": {
            "chaos_certified_by_this_pipeline": False,
            "hiddenness_certified_by_this_pipeline": False,
        },
        "notes": [
            "This promotes only the DK2018 block-restart ABM-GS reproduction lane.",
            "It does not validate fractional_variational_abm_qr.",
        ],
    }
    manifest = {
        "schema_version": "1.0",
        "method_id": METHOD_ID,
        "execution_contract": "dk2018_block_restart_abm_gs",
        "status": "published_quantitative_validated",
        "required_cases": sorted(REQUIRED_CASES),
        "numeric_outputs_policy": "promoted_after_real_native_quantitative_passes",
    }
    (official_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (official_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-output-dir",
        type=Path,
        default=ROOT / "validation" / "outputs" / "lyapunov_benchmarks" / "fractional_variational_abm_qr",
    )
    parser.add_argument(
        "--official-dir",
        type=Path,
        default=ROOT / "validation" / "chaos_validation" / "lyapunov_methods" / f"{METHOD_ID}_published",
    )
    args = parser.parse_args()
    print(json.dumps(promote(args.runtime_output_dir.resolve(), args.official_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
