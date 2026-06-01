#!/usr/bin/env python3
"""Run bounded opt-in F3 sensitivity sweeps for selected Fischer 2020 rows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from cloned_dynamics_benchmarks import load_benchmark_case  # noqa: E402
from diagnose_cloned_dynamics_discrepancies import (  # noqa: E402
    DIAGNOSTICS_DIR,
    SENSITIVITY_FIELDS,
    classify_row,
    generate_diagnostics,
)
from hidden_attractors.analysis import compute_cloned_dynamics_spectrum  # noqa: E402
from hidden_attractors.systems.fischer_benchmarks import get_fischer_benchmark  # noqa: E402


BENCHMARKS_DIR = (
    PROJECT_ROOT
    / "validation"
    / "lyapunov_benchmarks"
    / "fractional_cloned_dynamics_abm_gs_published"
)
PRIORITY_ROWS = {
    "fischer2020_jerk_commensurate.yaml": [0, 1, 2, 3],
    "fischer2020_jerk_incommensurate.yaml": [1, 2, 3],
    "fischer2020_four_wing_incommensurate.yaml": [1, 2, 3],
    "fischer2020_financial_incommensurate.yaml": [0],
}
OUTPUT_NAMES = {
    "delta": "sensitivity_delta.csv",
    "t_clone": "sensitivity_t_clone.csv",
    "h": "sensitivity_h.csv",
    "k": "sensitivity_k.csv",
    "gs_policy": "sensitivity_gs_policy.csv",
    "q1_mode": "sensitivity_q1_mode.csv",
}


def _unique(values: list[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def _variants(axis: str, case: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    integration = case["integration"]
    if axis == "delta":
        return [{"variant": f"delta={value:g}", "delta": value} for value in (1e-2, 1e-3, 1e-4, 1e-5)]
    if axis == "t_clone":
        base = float(integration["t_clone"])
        values = [0.5 * base, base, 2.0 * base]
        if case["system"]["kind"] == "jerk":
            values += [1.0, 5.0, 10.0]
        return [{"variant": f"t_clone={value:g}", "t_clone": value} for value in _unique(values)]
    if axis == "h":
        base = float(integration["h_clone"])
        return [{"variant": f"h_clone={value:g}", "h_clone": value} for value in (base, 0.5 * base, 2.0 * base)]
    if axis == "k":
        base = int(integration["K"])
        return [{"variant": f"K={value}", "K": value} for value in _unique([base, max(1, base // 2), 2 * base])]
    if axis == "gs_policy":
        return [
            {"variant": f"orthonormalization={value}", "orthonormalization": value}
            for value in ("gs_modified", "gs_classical", "qr")
        ]
    if axis == "q1_mode" and np.allclose(row["orders"], 1.0):
        return [
            {"variant": "integration_mode=fractional_abm_q1", "integration_mode": "fractional_abm"},
            {"variant": "integration_mode=integer_rk4_reference", "integration_mode": "integer_rk4_reference"},
        ]
    return []


def _run(axis: str, case_file: str, row_index: int, case: dict[str, Any], row: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    integration = deepcopy(case["integration"])
    integration.update({key: value for key, value in variant.items() if key in {"delta", "t_clone", "h_clone", "K"}})
    rhs, x0, _ = get_fischer_benchmark(case["system"]["kind"])
    orthonormalization = variant.get("orthonormalization", "gs_modified")
    integration_mode = variant.get("integration_mode", "fractional_abm")
    result = compute_cloned_dynamics_spectrum(
        rhs,
        x0,
        orders=row["orders"],
        h=float(integration["h_clone"]),
        t_clone=float(integration["t_clone"]),
        n_clones=len(x0),
        k_blocks=int(integration["K"]),
        delta=float(integration["delta"]),
        method=orthonormalization,
        memory_protocol=integration["memory_protocol"],
        system_id=case["system"]["kind"],
        parameters=case["system"]["parameters"],
        integration_mode=integration_mode,
    )
    computed = np.asarray(result.exponents, dtype=float)
    published = np.asarray(row["lyapunov"], dtype=float)
    errors = np.abs(computed - published)
    official_shape = {
        "case_file": case_file,
        "system": case["system"]["kind"],
        "row_index": row_index,
        "type": row["type"],
        "orders": row["orders"],
        "computed_LE": computed.tolist(),
        "published_LE": published.tolist(),
        "abs_error": errors.tolist(),
        "sign_match": bool(np.array_equal(np.sign(computed), np.sign(published))),
        "K01_published": row["K01"],
        "status": "published_benchmark_inconclusive" if result.status != "ok" else "diagnostic_sensitivity_result",
    }
    classified = classify_row(official_shape)
    return {
        "axis": axis,
        "case_file": case_file,
        "system": case["system"]["kind"],
        "row_index": row_index,
        "type": row["type"],
        "orders": row["orders"],
        "variant": variant["variant"],
        "delta": integration["delta"],
        "t_clone": integration["t_clone"],
        "h_clone": integration["h_clone"],
        "K": integration["K"],
        "orthonormalization": orthonormalization,
        "integration_mode": integration_mode,
        "computed_LE": computed.tolist(),
        "published_LE": published.tolist(),
        "abs_error": errors.tolist(),
        "max_abs_error": float(np.max(errors)),
        "strict_sign_match": classified["strict_sign_match"],
        "tolerant_sign_match": classified["tolerant_sign_match"],
        "near_zero_sign_boundary": classified["near_zero_sign_boundary"],
        "discrepancy_class": classified["discrepancy_class"],
        "status": result.status,
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SENSITIVITY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", action="append", choices=sorted(OUTPUT_NAMES), help="Sweep axis; repeat to select several.")
    parser.add_argument("--case-file", action="append", help="Restrict execution to selected Fischer YAML files.")
    parser.add_argument("--max-runs", type=int, default=12, help="Bound costly executions; use 0 for no limit.")
    parser.add_argument("--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR)
    args = parser.parse_args()
    generate_diagnostics(diagnostics_dir=args.diagnostics_dir)
    if os.environ.get("RUN_F3_DISCREPANCY_SWEEPS") != "1":
        print("Sensitivity plan generated. Set RUN_F3_DISCREPANCY_SWEEPS=1 to execute bounded sweeps.")
        return
    axes = args.axis or ["delta", "t_clone", "gs_policy", "q1_mode"]
    selected_files = args.case_file or list(PRIORITY_ROWS)
    all_rows: list[dict[str, Any]] = []
    limit_reached = False
    for case_file in selected_files:
        case = load_benchmark_case(BENCHMARKS_DIR / case_file)
        for row_index in PRIORITY_ROWS.get(case_file, []):
            row = case["expected"]["rows"][row_index]
            for axis in axes:
                for variant in _variants(axis, case, row):
                    if args.max_runs and len(all_rows) >= args.max_runs:
                        limit_reached = True
                        break
                    all_rows.append(_run(axis, case_file, row_index, case, row, variant))
                if limit_reached:
                    break
            if limit_reached:
                break
        if limit_reached:
            break
    for axis, filename in OUTPUT_NAMES.items():
        rows = [row for row in all_rows if row["axis"] == axis]
        _write(args.diagnostics_dir / filename, rows)
    summary = {
        "status": "partial_sweeps_executed_not_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_executed": len(all_rows),
        "max_runs": args.max_runs,
        "limit_reached": limit_reached,
        "axes_requested": axes,
        "validated_after_sensitivity": False,
        "notes": ["Sensitivity results are diagnostic only and do not promote F3."],
    }
    (args.diagnostics_dir / "sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Recorded {len(all_rows)} bounded sensitivity rows; validation remains false.")


if __name__ == "__main__":
    main()
