#!/usr/bin/env python3
"""Record a real long DK2018 native run without promoting failed evidence."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METHOD_ID = "fractional_variational_dk2018_block_restart_abm_gs"
REQUIRED_CASES = {
    "published_dk2018_rabinovich_fabrikant_q0999",
    "published_dk2018_lorenz_q0985",
}


def _list_value(row: dict[str, str], key: str) -> list[object]:
    raw = row.get(key, "")
    return list(ast.literal_eval(raw)) if raw else []


def record(runtime_output_dir: Path, official_dir: Path) -> dict[str, object]:
    with (runtime_output_dir / "benchmark_cases.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["case_id"]: row for row in csv.DictReader(handle)}
    missing = sorted(REQUIRED_CASES - rows.keys())
    if missing:
        raise RuntimeError(f"Cannot record DK2018 long run; missing cases: {missing}")

    verdicts = []
    for case_id in sorted(REQUIRED_CASES):
        row = rows[case_id]
        verdicts.append(
            {
                "case_id": case_id,
                "status": row["status"],
                "validation_run_class": row.get("validation_run_class"),
                "numerical_route": row.get("numerical_route"),
                "execution_contract": row.get("execution_contract"),
                "computed_exponents": _list_value(row, "computed_exponents"),
                "expected_exponents": _list_value(row, "expected_exponents"),
                "absolute_differences": _list_value(row, "absolute_differences"),
                "absolute_tolerance": float(row["absolute_tolerance"]),
                "failing_components": _list_value(row, "failing_components"),
            }
        )

    passed = all(row["status"] == "published_benchmark_passed_quantitative" for row in rows.values() if row["case_id"] in REQUIRED_CASES)
    status = "published_quantitative_validated" if passed else "published_benchmarks_pending_reproduced_discrepancy"
    summary = {
        "schema_version": "1.0",
        "method_id": METHOD_ID,
        "status": status,
        "validation_run_class": "published_quantitative_long",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_cases": sorted(REQUIRED_CASES),
        "published_case_verdicts": verdicts,
        "quick_ci_guarantee": "native_smoke_only_not_published_quantitative_validation",
        "published_quantitative_opt_in": "RUN_PUBLISHED_LYAPUNOV=1",
        "local_full_history_qr_status": "published_benchmarks_pending_separate_contract_required",
        "certifications": {
            "chaos_certified_by_this_pipeline": False,
            "hiddenness_certified_by_this_pipeline": False,
        },
        "notes": [
            "This records a real long native run for the DK2018 block-restart ABM-GS lane.",
            "The RF discrepancy is independently reproduced by the MATLAB fde12 oracle; see validation/matlab/README.md.",
            "Fast CI and smoke tests do not establish published quantitative validation.",
            "This lane does not validate fractional_variational_abm_qr.",
        ],
    }
    official_dir.mkdir(parents=True, exist_ok=True)
    with (official_dir / "validation_summary.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "schema_version": "1.0",
        "method_id": METHOD_ID,
        "execution_contract": "dk2018_block_restart_abm_gs",
        "status": status,
        "required_cases": sorted(REQUIRED_CASES),
        "numeric_outputs_policy": "summary_only_until_all_real_native_quantitative_cases_pass",
    }
    with (official_dir / "manifest.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-output-dir", type=Path, required=True)
    parser.add_argument(
        "--official-dir",
        type=Path,
        default=ROOT / "validation" / "chaos_validation" / "lyapunov_methods" / f"{METHOD_ID}_published",
    )
    args = parser.parse_args()
    print(json.dumps(record(args.runtime_output_dir.resolve(), args.official_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
