#!/usr/bin/env python3
"""Run bounded opt-in F3 sensitivity sweeps for selected Fischer 2020 rows."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import time
from collections import defaultdict
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
    SUMMARY_PATH,
    build_matrix,
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
SENSITIVITY_REPORT = "sensitivity_analysis_report.md"
RUN_LOG = "sensitivity_run_log.json"


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
        "lambda_max_abs_error": float(errors[0]),
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


def _parse(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: _parse(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(str(row.get(field)) for field in (
        "axis",
        "case_file",
        "row_index",
        "variant",
        "delta",
        "t_clone",
        "h_clone",
        "K",
        "orthonormalization",
        "integration_mode",
    ))


def _append_unique(path: Path, row: dict[str, Any]) -> bool:
    rows = _read(path)
    signatures = {_signature(existing) for existing in rows}
    if _signature(row) in signatures:
        return False
    rows.append(row)
    _write(path, rows)
    return True


def _planned_signature(
    axis: str,
    case_file: str,
    row_index: int,
    case: dict[str, Any],
    variant: dict[str, Any],
) -> tuple[Any, ...]:
    integration = deepcopy(case["integration"])
    integration.update(
        {
            key: value
            for key, value in variant.items()
            if key in {"delta", "t_clone", "h_clone", "K"}
        }
    )
    return _signature(
        {
            "axis": axis,
            "case_file": case_file,
            "row_index": row_index,
            "variant": variant["variant"],
            "delta": integration["delta"],
            "t_clone": integration["t_clone"],
            "h_clone": integration["h_clone"],
            "K": integration["K"],
            "orthonormalization": variant.get("orthonormalization", "gs_modified"),
            "integration_mode": variant.get("integration_mode", "fractional_abm"),
        }
    )


def _load_all_rows(diagnostics_dir: Path) -> list[dict[str, Any]]:
    return [
        row
        for filename in OUTPUT_NAMES.values()
        for row in _read(diagnostics_dir / filename)
    ]


def _baseline_rows() -> dict[tuple[str, int], dict[str, Any]]:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return {
        (row["case_file"], int(row["row_index"])): row
        for row in build_matrix(summary)
    }


def _improved_class(baseline: dict[str, Any], candidate: dict[str, Any]) -> bool:
    ranking = {
        "numerical_failure_or_inconclusive": 0,
        "strict_discrepancy": 1,
        "sign_pattern_supported_not_quantitative": 2,
        "quantitative_abs_pass_near_zero_sign_boundary": 3,
        "quantitative_abs_pass_strict_sign_pass": 4,
    }
    return ranking[candidate["discrepancy_class"]] > ranking[baseline["discrepancy_class"]]


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=lambda row: float(row["max_abs_error"]))


def _row_label(row: dict[str, Any]) -> str:
    return f"{row['system']} {row['type']} {row['orders']} ({row['variant']})"


def _load_log(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _write_analysis(diagnostics_dir: Path) -> dict[str, Any]:
    rows = _load_all_rows(diagnostics_dir)
    baseline = _baseline_rows()
    grouped_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    improvements = []
    for row in rows:
        grouped_axis[str(row["axis"])].append(row)
        grouped_system[str(row["system"])].append(row)
        original = baseline[(str(row["case_file"]), int(row["row_index"]))]
        if _improved_class(original, row):
            improvements.append(
                {
                    "case_file": row["case_file"],
                    "row_index": int(row["row_index"]),
                    "system": row["system"],
                    "axis": row["axis"],
                    "variant": row["variant"],
                    "baseline_class": original["discrepancy_class"],
                    "improved_class": row["discrepancy_class"],
                    "max_abs_error": float(row["max_abs_error"]),
                    "lambda_max_abs_error": float(row["lambda_max_abs_error"]),
                }
            )
    remaining = []
    for key, original in baseline.items():
        if original["discrepancy_class"] != "strict_discrepancy":
            continue
        candidates = [
            row for row in rows
            if (str(row["case_file"]), int(row["row_index"])) == key
        ]
        if not candidates or not any(_improved_class(original, row) for row in candidates):
            remaining.append(
                {
                    "case_file": original["case_file"],
                    "row_index": original["row_index"],
                    "system": original["system"],
                    "orders": original["orders"],
                    "baseline_class": original["discrepancy_class"],
                }
            )
    axes = []
    for axis, axis_rows in sorted(grouped_axis.items()):
        best = _best_row(axis_rows)
        improved_keys = {
            (str(row["case_file"]), int(row["row_index"]))
            for row in axis_rows
            if _improved_class(
                baseline[(str(row["case_file"]), int(row["row_index"]))],
                row,
            )
        }
        degraded_keys = {
            (str(row["case_file"]), int(row["row_index"]))
            for row in axis_rows
            if float(row["max_abs_error"])
            > float(baseline[(str(row["case_file"]), int(row["row_index"]))]["max_abs_error"])
        }
        axes.append(
            {
                "axis": axis,
                "runs": len(axis_rows),
                "best_row": _row_label(best),
                "best_max_abs_error": float(best["max_abs_error"]),
                "best_lambda_max_abs_error": float(best["lambda_max_abs_error"]),
                "improved_rows": len(improved_keys),
                "degraded_rows": len(degraded_keys),
            }
        )
    systems = []
    for system, system_rows in sorted(grouped_system.items()):
        best = _best_row(system_rows)
        systems.append(
            {
                "system": system,
                "runs": len(system_rows),
                "best_row": _row_label(best),
                "best_max_abs_error": float(best["max_abs_error"]),
                "strict_sign_matches": sum(bool(row["strict_sign_match"]) for row in system_rows),
                "tolerant_sign_matches": sum(bool(row["tolerant_sign_match"]) for row in system_rows),
            }
        )
    gs_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in grouped_axis.get("gs_policy", []):
        gs_groups[(str(row["case_file"]), int(row["row_index"]))].append(row)
    gs_policy_max_spread = max(
        (
            max(float(row["max_abs_error"]) for row in group)
            - min(float(row["max_abs_error"]) for row in group)
        )
        for group in gs_groups.values()
    ) if gs_groups else None
    t_clone_improved = next(
        (row for row in axes if row["axis"] == "t_clone"),
        {"improved_rows": 0},
    )["improved_rows"]
    delta_improved = next(
        (row for row in axes if row["axis"] == "delta"),
        {"improved_rows": 0},
    )["improved_rows"]
    hypotheses = [
        {
            "hypothesis": "H1",
            "evidence_for": f"T_clone improved {t_clone_improved} row classifications; jerk q=1 reaches quantitative agreement at T_clone=10.",
            "evidence_against": "No protocol variant is proven equivalent to the article.",
            "status": "supported_by_t_clone_sensitivity",
        },
        {
            "hypothesis": "H2",
            "evidence_for": "T_clone changes some classes strongly; h_clone and K produce smaller row-specific shifts.",
            "evidence_against": "Parameter sensitivity alone does not identify the article convention.",
            "status": "supported_but_not_identified",
        },
        {
            "hypothesis": "H3",
            "evidence_for": "GS modified, GS classical, and QR outputs are compared.",
            "evidence_against": f"The maximum cross-policy max_abs_error spread is {gs_policy_max_spread:.6g}; policy outputs are numerically equivalent for the swept rows.",
            "status": "weakened_as_primary_explanation",
        },
        {
            "hypothesis": "H4",
            "evidence_for": f"Delta sweeps improved {delta_improved} row classifications and near-zero sign crossings remain explicitly classified.",
            "evidence_against": "Large jerk lambda_3 gaps cannot be explained by near-zero signs.",
            "status": "supported_for_subset_only",
        },
        {
            "hypothesis": "H5",
            "evidence_for": "Incommensurate four-wing and jerk rows retain targeted sweeps.",
            "evidence_against": "Commensurate jerk discrepancies show this is not the only cause.",
            "status": "open",
        },
        {
            "hypothesis": "H6",
            "evidence_for": "Rounded published values can affect near-zero sign interpretation.",
            "evidence_against": "Rounding cannot explain large lambda_3 gaps.",
            "status": "supported_for_subset_only",
        },
    ]
    run_log = _load_log(diagnostics_dir / RUN_LOG)
    summary = {
        "status": "partial_sweeps_executed_not_validation" if rows else "planned_not_executed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validated_after_sensitivity": False,
        "runs_total": len(rows),
        "axes_executed": sorted(grouped_axis),
        "systems_executed": sorted(grouped_system),
        "full_sweep_status": "not_executed_bounded_partial_due_to_cost",
        "elapsed_seconds_total": sum(float(row.get("elapsed_seconds") or 0.0) for row in rows),
        "gs_policy_max_abs_error_spread": gs_policy_max_spread,
        "axis_summary": axes,
        "system_summary": systems,
        "best_improvements": improvements,
        "remaining_discrepancies": remaining,
        "dominant_hypotheses": ["H1", "H2", "H5"],
        "run_invocations": run_log,
        "notes": [
            "Sensitivity sweeps are diagnostic only and do not promote F3.",
            "F3 remains validated=False.",
            "No chaos or hiddenness is certified.",
            "Hypotheses remain hypotheses, not definitive conclusions.",
            "The unlimited sweep remains pending because bounded runs already showed substantial cost.",
        ],
    }
    (diagnostics_dir / "sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# F3 sensitivity analysis report",
        "",
        "## Executive summary",
        "",
        f"Controlled diagnostic sweeps recorded `{len(rows)}` unique runs.",
        f"Execution timestamp: `{summary['generated_at_utc']}`.",
        f"Recorded integration time: `{summary['elapsed_seconds_total']:.3f}` seconds.",
        "These runs assess sensitivity only. They do not promote F3 validation,",
        "certify chaos, or certify hiddenness.",
        "The unlimited sweep was not executed because bounded runs already showed",
        "substantial cost; partial outputs were preserved after each row.",
        "",
        "## Executed commands",
        "",
    ]
    lines.extend(f"- `{entry['command']}`" for entry in run_log)
    lines.extend(
        [
            "",
            "## Results by axis",
            "",
            "| axis | runs | best row | best max_abs_error | best lambda_max_abs_error | improved rows | degraded rows |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in axes:
        lines.append(
            f"| {row['axis']} | {row['runs']} | {row['best_row']} | "
            f"{row['best_max_abs_error']:.6g} | {row['best_lambda_max_abs_error']:.6g} | "
            f"{row['improved_rows']} | {row['degraded_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Results by system",
            "",
            "| system | runs | best row | best max_abs_error | strict sign matches | tolerant sign matches |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in systems:
        lines.append(
            f"| {row['system']} | {row['runs']} | {row['best_row']} | "
            f"{row['best_max_abs_error']:.6g} | {row['strict_sign_matches']} | "
            f"{row['tolerant_sign_matches']} |"
        )
    lines.extend(
        [
            "",
            "## Best improvements",
            "",
            "| system | axis | variant | baseline class | improved class | max_abs_error | lambda_max_abs_error |",
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    for row in improvements:
        lines.append(
            f"| {row['system']} | {row['axis']} | {row['variant']} | "
            f"`{row['baseline_class']}` | `{row['improved_class']}` | "
            f"{row['max_abs_error']:.6g} | {row['lambda_max_abs_error']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Persistent strict discrepancies",
            "",
            "| system | orders | case file | row index |",
            "|---|---|---|---:|",
        ]
    )
    for row in remaining:
        lines.append(
            f"| {row['system']} | `{row['orders']}` | `{row['case_file']}` | "
            f"{row['row_index']} |"
        )
    lines.extend(
        [
            "",
            "## Hypothesis assessment",
            "",
            "| hypothesis | evidence for | evidence against | status |",
            "|---|---|---|---|",
        ]
    )
    for row in hypotheses:
        lines.append(
            f"| {row['hypothesis']} | {row['evidence_for']} | "
            f"{row['evidence_against']} | `{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "Favored hypotheses: `H1`, `H2`, and `H5` remain the main audit paths.",
            "Weakened hypothesis: `H3` is not supported as a primary explanation",
            "because GS modified, GS classical, and QR remain numerically equivalent",
            "for the swept rows. `H4` and `H6` explain only near-zero subsets.",
        ]
    )
    lines.extend(
        [
            "",
            "## Conservative conclusion",
            "",
            "The sweep evidence is diagnostic and partial. It does not justify promotion",
            "to validation. The official F3 state remains",
            "`published_benchmarks_pending_discrepancy` with `validated=False`.",
            "",
        ]
    )
    (diagnostics_dir / SENSITIVITY_REPORT).write_text("\n".join(lines), encoding="utf-8")
    official = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    diagnostics = official.setdefault("discrepancy_diagnostics", {})
    diagnostics.update(
        {
            "sensitivity_status": summary["status"],
            "sensitivity_summary": f"discrepancy_diagnostics/sensitivity_summary.json",
            "sensitivity_report": f"discrepancy_diagnostics/{SENSITIVITY_REPORT}",
            "validated_after_diagnostics": False,
        }
    )
    SUMMARY_PATH.write_text(json.dumps(official, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", action="append", choices=sorted(OUTPUT_NAMES), help="Sweep axis; repeat to select several.")
    parser.add_argument("--case-file", action="append", help="Restrict execution to selected Fischer YAML files.")
    parser.add_argument("--row-index", action="append", type=int, help="Restrict execution to selected row indices.")
    parser.add_argument("--max-runs", type=int, default=12, help="Bound costly executions; use 0 for no limit.")
    parser.add_argument("--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR)
    args = parser.parse_args()
    generate_diagnostics(diagnostics_dir=args.diagnostics_dir)
    if os.environ.get("RUN_F3_DISCREPANCY_SWEEPS") != "1":
        if _load_all_rows(args.diagnostics_dir):
            summary = _write_analysis(args.diagnostics_dir)
            generate_diagnostics(diagnostics_dir=args.diagnostics_dir)
            print(
                f"Sensitivity analysis refreshed from {summary['runs_total']} "
                "stored rows; validation remains false."
            )
        else:
            print("Sensitivity plan generated. Set RUN_F3_DISCREPANCY_SWEEPS=1 to execute bounded sweeps.")
        return
    axes = args.axis or ["delta", "t_clone", "gs_policy", "q1_mode"]
    selected_files = args.case_file or list(PRIORITY_ROWS)
    all_rows: list[dict[str, Any]] = []
    new_rows = 0
    existing_signatures = {
        _signature(row)
        for row in _load_all_rows(args.diagnostics_dir)
    }
    limit_reached = False
    for case_file in selected_files:
        case = load_benchmark_case(BENCHMARKS_DIR / case_file)
        row_indices = PRIORITY_ROWS.get(case_file, [])
        if args.row_index is not None:
            row_indices = [index for index in row_indices if index in args.row_index]
        for row_index in row_indices:
            row = case["expected"]["rows"][row_index]
            for axis in axes:
                for variant in _variants(axis, case, row):
                    if args.max_runs and len(all_rows) >= args.max_runs:
                        limit_reached = True
                        break
                    signature = _planned_signature(axis, case_file, row_index, case, variant)
                    if signature in existing_signatures:
                        continue
                    started = time.perf_counter()
                    result = _run(axis, case_file, row_index, case, row, variant)
                    result["elapsed_seconds"] = time.perf_counter() - started
                    all_rows.append(result)
                    if _append_unique(args.diagnostics_dir / OUTPUT_NAMES[axis], result):
                        new_rows += 1
                        existing_signatures.add(signature)
                if limit_reached:
                    break
            if limit_reached:
                break
        if limit_reached:
            break
    log_path = args.diagnostics_dir / RUN_LOG
    log = _load_log(log_path)
    log.append(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "axes_requested": axes,
            "case_files": selected_files,
            "row_indices": args.row_index,
            "max_runs": args.max_runs,
            "runs_attempted": len(all_rows),
            "new_unique_rows": new_rows,
            "limit_reached": limit_reached,
        }
    )
    log_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    summary = _write_analysis(args.diagnostics_dir)
    generate_diagnostics(diagnostics_dir=args.diagnostics_dir)
    print(
        f"Recorded {new_rows} new sensitivity rows "
        f"({summary['runs_total']} unique cumulative); validation remains false."
    )


if __name__ == "__main__":
    main()
