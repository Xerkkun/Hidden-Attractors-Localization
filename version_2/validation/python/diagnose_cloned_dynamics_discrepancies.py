#!/usr/bin/env python3
"""Generate auditable discrepancy diagnostics for the Fischer 2020 F3 lane."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD_DIR = (
    PROJECT_ROOT
    / "validation"
    / "chaos_validation"
    / "lyapunov_methods"
    / "fractional_cloned_dynamics_abm_gs_published"
)
SUMMARY_PATH = METHOD_DIR / "validation_summary.json"
OUTPUT_RESULTS_PATH = (
    PROJECT_ROOT
    / "validation"
    / "outputs"
    / "lyapunov_benchmarks"
    / "fractional_cloned_dynamics_abm_gs_published"
    / "published_benchmark_results.csv"
)
DIAGNOSTICS_DIR = METHOD_DIR / "discrepancy_diagnostics"
ZERO_TOL = 0.02
TARGET_ABS_TOL = 0.05

MATRIX_FIELDS = [
    "case_file",
    "system",
    "row_index",
    "type",
    "orders",
    "computed_lambda_1",
    "computed_lambda_2",
    "computed_lambda_3",
    "published_lambda_1",
    "published_lambda_2",
    "published_lambda_3",
    "abs_error_1",
    "abs_error_2",
    "abs_error_3",
    "max_abs_error",
    "lambda_max_abs_error",
    "sign_match",
    "status",
    "K01_published",
    "discrepancy_class",
    "near_zero_sign_boundary",
    "strict_sign_match",
    "tolerant_sign_match",
    "sign_component_statuses",
    "quantitative_abs_pass",
    "interpretation",
    "notes",
]

SENSITIVITY_FIELDS = [
    "axis",
    "case_file",
    "system",
    "row_index",
    "type",
    "orders",
    "variant",
    "delta",
    "t_clone",
    "h_clone",
    "K",
    "orthonormalization",
    "integration_mode",
    "computed_LE",
    "published_LE",
    "abs_error",
    "max_abs_error",
    "strict_sign_match",
    "tolerant_sign_match",
    "near_zero_sign_boundary",
    "discrepancy_class",
    "status",
]


def _sign(value: float) -> int:
    return 0 if value == 0 else (1 if value > 0 else -1)


def classify_sign_match(
    computed: Iterable[float],
    published: Iterable[float],
    zero_tol: float = ZERO_TOL,
) -> dict[str, Any]:
    """Classify strict and near-zero-compatible sign agreement component-wise."""

    statuses = []
    computed_values = [float(value) for value in computed]
    published_values = [float(value) for value in published]
    for calculated, expected in zip(computed_values, published_values, strict=True):
        if abs(expected) < zero_tol and abs(calculated) < zero_tol:
            statuses.append("near_zero_compatible")
        elif _sign(calculated) == _sign(expected):
            statuses.append("strict_match")
        elif abs(expected) < zero_tol or abs(calculated) < zero_tol:
            statuses.append("near_zero_boundary_crossing")
        else:
            statuses.append("strict_mismatch")
    strict = all(
        _sign(calculated) == _sign(expected)
        for calculated, expected in zip(computed_values, published_values, strict=True)
    )
    tolerant = all(status != "strict_mismatch" for status in statuses)
    return {
        "sign_component_statuses": statuses,
        "strict_sign_match": strict,
        "tolerant_sign_match": tolerant,
        "near_zero_sign_boundary": not strict and tolerant,
    }


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def classify_row(row: dict[str, Any]) -> dict[str, Any]:
    """Add the fine diagnostic interpretation without changing official status."""

    computed = [float(value) for value in row["computed_LE"]]
    published = [float(value) for value in row["published_LE"]]
    errors = [float(value) for value in row["abs_error"]]
    signs = classify_sign_match(computed, published)
    quantitative_abs_pass = _finite(errors) and max(errors) < TARGET_ABS_TOL
    if not _finite(computed) or row["status"] == "published_benchmark_inconclusive":
        discrepancy_class = "numerical_failure_or_inconclusive"
        interpretation = "numerical_failure_or_inconclusive"
    elif quantitative_abs_pass and signs["strict_sign_match"]:
        discrepancy_class = "quantitative_abs_pass_strict_sign_pass"
        interpretation = "strict_quantitative_agreement"
    elif quantitative_abs_pass and signs["near_zero_sign_boundary"]:
        discrepancy_class = "quantitative_abs_pass_near_zero_sign_boundary"
        interpretation = "near_zero_boundary"
    elif signs["strict_sign_match"]:
        discrepancy_class = "sign_pattern_supported_not_quantitative"
        interpretation = "sign_pattern_only"
    else:
        discrepancy_class = "strict_discrepancy"
        interpretation = (
            "near_zero_boundary_with_quantitative_discrepancy"
            if signs["near_zero_sign_boundary"]
            else "strict_sign_discrepancy"
        )
    notes = []
    if signs["near_zero_sign_boundary"]:
        notes.append("strict sign crossing occurs only at the explicit near-zero boundary")
    if row["status"] == "published_benchmark_passed_quantitative" and not signs["strict_sign_match"]:
        notes.append("historical quantitative absolute-error pass crosses strict sign boundary")
    if row["system"] == "financial":
        notes.append("financial RHS contains abs(x)")
    if row["system"] == "jerk" and max(errors) >= TARGET_ABS_TOL:
        notes.append("review jerk exponential nonlinearity scale and clone protocol")
    return {
        "case_file": row["case_file"],
        "system": row["system"],
        "row_index": int(row["row_index"]),
        "type": row["type"],
        "orders": row["orders"],
        "computed_lambda_1": computed[0],
        "computed_lambda_2": computed[1],
        "computed_lambda_3": computed[2],
        "published_lambda_1": published[0],
        "published_lambda_2": published[1],
        "published_lambda_3": published[2],
        "abs_error_1": errors[0],
        "abs_error_2": errors[1],
        "abs_error_3": errors[2],
        "max_abs_error": max(errors),
        "lambda_max_abs_error": errors[0],
        "sign_match": bool(row["sign_match"]),
        "status": row["status"],
        "K01_published": row["K01_published"],
        "discrepancy_class": discrepancy_class,
        "near_zero_sign_boundary": signs["near_zero_sign_boundary"],
        "strict_sign_match": signs["strict_sign_match"],
        "tolerant_sign_match": signs["tolerant_sign_match"],
        "sign_component_statuses": signs["sign_component_statuses"],
        "quantitative_abs_pass": quantitative_abs_pass,
        "interpretation": interpretation,
        "notes": "; ".join(notes),
    }


def build_matrix(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one diagnostic record per official benchmark result."""

    return [classify_row(row) for row in summary["results"]]


def _parse_csv_value(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def check_outputs_csv_consistency(
    summary_rows: list[dict[str, Any]],
    outputs_csv: Path,
) -> dict[str, Any]:
    """Compare optional ignored runtime CSV rows against the tracked summary."""

    if not outputs_csv.exists():
        return {"outputs_csv_present": False, "outputs_csv_consistent": None, "differences": []}
    with outputs_csv.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    differences = []
    if len(csv_rows) != len(summary_rows):
        differences.append(f"row_count:{len(csv_rows)}!={len(summary_rows)}")
    for index, (runtime, official) in enumerate(zip(csv_rows, summary_rows)):
        for field in ("case_file", "system", "status"):
            if runtime.get(field) != str(official[field]):
                differences.append(f"row_{index}:{field}")
        for field in ("row_index", "computed_LE", "published_LE", "abs_error", "sign_match"):
            if _parse_csv_value(runtime.get(field, "")) != official[field]:
                differences.append(f"row_{index}:{field}")
    return {
        "outputs_csv_present": True,
        "outputs_csv_consistent": not differences,
        "differences": differences,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _sensitivity_plan() -> dict[str, Any]:
    return {
        "status": "planned_or_partial",
        "purpose": "diagnose protocol and numerical sensitivity without promoting validation",
        "opt_in": "RUN_F3_DISCREPANCY_SWEEPS=1",
        "bounded_default": "run_cloned_dynamics_sensitivity.py limits active runs unless --max-runs 0 is supplied",
        "priority_rows": [
            {"case_file": "fischer2020_jerk_commensurate.yaml", "row_indices": [0, 1, 2, 3]},
            {"case_file": "fischer2020_jerk_incommensurate.yaml", "row_indices": [1, 2, 3]},
            {"case_file": "fischer2020_four_wing_incommensurate.yaml", "row_indices": [1, 2, 3]},
            {"case_file": "fischer2020_financial_incommensurate.yaml", "row_indices": [0]},
        ],
        "axes": {
            "delta": [1e-2, 1e-3, 1e-4, 1e-5],
            "t_clone": {
                "multipliers": [0.5, 1.0, 2.0],
                "jerk_explicit_values": [1.0, 5.0, 10.0],
            },
            "h_clone": {"multipliers": [0.5, 1.0, 2.0], "double_only_if_stable": True},
            "K": {"multipliers": [0.5, 1.0, 2.0], "double_only_if_cost_allows": True},
            "orthonormalization": ["gs_modified", "gs_classical", "qr"],
            "abm_q1_mode": ["fractional_abm_q1", "integer_rk4_reference"],
        },
        "hypotheses": [
            "H1: cloning protocol is not identical to the published protocol",
            "H2: T_C, N_C, h_C, or K interpretation differs",
            "H3: classical GS, modified GS, QR, or accumulated norm convention differs",
            "H4: near-zero exponents are strongly sensitive",
            "H5: incommensurate ABM handling requires a specific audit",
            "H6: published rounded values may retain ambiguity",
        ],
    }


def _format_vector(values: Iterable[Any]) -> str:
    return "[" + ", ".join(f"{float(value):.6g}" for value in values) + "]"


def _global_rows(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matrix:
        grouped[row["system"]].append(row)
    result = []
    for system in ("financial", "four_wing", "jerk"):
        rows = grouped[system]
        result.append(
            {
                "system": system,
                "commensurate": sum(row["type"] == "Comm" for row in rows),
                "incommensurate": sum(row["type"] == "Incomm" for row in rows),
                "quantitative": sum(row["quantitative_abs_pass"] for row in rows),
                "sign_pattern": sum(
                    row["discrepancy_class"] == "sign_pattern_supported_not_quantitative"
                    for row in rows
                ),
                "failed": sum(row["discrepancy_class"] == "strict_discrepancy" for row in rows),
                "sign_mismatches": sum(not row["strict_sign_match"] for row in rows),
            }
        )
    return result


def _render_report(matrix: list[dict[str, Any]], consistency: dict[str, Any]) -> str:
    global_rows = _global_rows(matrix)
    lines = [
        "# Fischer 2020 F3 discrepancy report",
        "",
        "## Executive summary",
        "",
        "F3 was executed against 24 published Fischer 2020 rows. It produced 10",
        "quantitative passes, 6 sign-pattern passes, and 8 failures. Therefore the",
        "method remains `validated=False`.",
        "",
        "The official status remains `published_benchmarks_pending_discrepancy`.",
        "This diagnostic layer classifies the recorded results; it does not tune",
        "parameters to force agreement and it does not promote validation.",
        "",
        "## Runtime CSV consistency",
        "",
        f"- `outputs_csv_present = {str(consistency['outputs_csv_present']).lower()}`",
        f"- `outputs_csv_consistent = {str(consistency['outputs_csv_consistent']).lower() if consistency['outputs_csv_consistent'] is not None else 'not_checked'}`",
        "",
        "## Global table",
        "",
        "| system | commensurate rows | incommensurate rows | quantitative | sign-pattern | failed | sign mismatches |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_rows:
        lines.append(
            f"| {row['system']} | {row['commensurate']} | {row['incommensurate']} | "
            f"{row['quantitative']} | {row['sign_pattern']} | {row['failed']} | "
            f"{row['sign_mismatches']} |"
        )
    lines.extend(
        [
            "",
            "## Row-level classification",
            "",
            "| system | type | orders | computed LE | published LE | abs error | diagnostic class |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in matrix:
        lines.append(
            f"| {row['system']} | {row['type']} | `{row['orders']}` | "
            f"`{_format_vector([row[f'computed_lambda_{index}'] for index in (1, 2, 3)])}` | "
            f"`{_format_vector([row[f'published_lambda_{index}'] for index in (1, 2, 3)])}` | "
            f"`{_format_vector([row[f'abs_error_{index}'] for index in (1, 2, 3)])}` | "
            f"`{row['discrepancy_class']}` |"
        )
    lines.extend(
        [
            "",
            "## Diagnosis by system",
            "",
            "### Financial",
            "",
            "- Results are mostly close to the published rows.",
            "- The `[0.9, 1, 1]` incommensurate row crosses the sign boundary for an exponent close to zero.",
            "- The RHS is nonsmooth because it contains `abs(x)`.",
            "",
            "### Four-wing",
            "",
            "- Several commensurate rows reproduce closely.",
            "- Incommensurate rows retain discrepancies, including second-exponent sign changes.",
            "",
            "### Jerk",
            "",
            "- Several rows retain large `lambda_3` discrepancies.",
            "- Review the exponential nonlinearity scale, `T_clone`, and ABM protocol interpretation.",
            "",
            "## Ordered hypotheses",
            "",
            "1. H1: cloning protocol is not identical to the published protocol.",
            "2. H2: interpretation of `T_C`, `N_C`, `h_C`, or `K` differs.",
            "3. H3: classical GS, modified GS, QR, or accumulated norm convention differs.",
            "4. H4: near-zero exponents are strongly sensitive.",
            "5. H5: incommensurate ABM handling requires a specific audit.",
            "6. H6: published rounded values may retain ambiguity.",
            "",
            "These are hypotheses, not conclusions. The current evidence does not justify",
            "claiming that the article is incorrect.",
            "",
            "## Near-zero sign policy",
            "",
            "A quantitative absolute-error pass may still cross the strict sign boundary",
            "for near-zero exponents. Use the additional diagnostic field",
            "`near_zero_sign_boundary` before interpreting sign failures.",
            "",
            "## Recommendations",
            "",
            "- Do not promote F3.",
            "- Execute `T_clone` and `delta` sensitivity sweeps for discrepant rows.",
            "- Validate ABM `q=1` against an exact solution.",
            "- Compare integer jerk against the independent reusable RK4 integrator.",
            "- Review whether the article uses a transient before clone accumulation.",
            "- Review whether clones restart on a long fiducial trajectory or restart block-locally.",
            "",
            "## Methodological declaration",
            "",
            "Passing or failing F3 does not validate or invalidate F2.",
            "F3 does not certify chaos. F3 does not certify hiddenness.",
            "The current state is `published_benchmarks_pending_discrepancy`.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_diagnostics(
    summary_path: Path = SUMMARY_PATH,
    outputs_csv_path: Path = OUTPUT_RESULTS_PATH,
    diagnostics_dir: Path = DIAGNOSTICS_DIR,
) -> dict[str, Any]:
    """Generate tracked diagnostics from the official historical summary."""

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "published_benchmarks_pending_discrepancy":
        raise ValueError("F3 diagnostics require the conservative discrepancy-pending summary.")
    if summary["validated"] or summary["validated_against_published_benchmarks"]:
        raise ValueError("F3 diagnostics must not run against a promoted summary.")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix(summary)
    consistency = check_outputs_csv_consistency(summary["results"], outputs_csv_path)
    _write_csv(diagnostics_dir / "fischer2020_discrepancy_matrix.csv", matrix, MATRIX_FIELDS)
    _write_csv(diagnostics_dir / "fischer2020_row_classification.csv", matrix, MATRIX_FIELDS)
    plan = _sensitivity_plan()
    (diagnostics_dir / "sensitivity_plan.yaml").write_text(
        yaml.safe_dump(plan, sort_keys=False),
        encoding="utf-8",
    )
    policy = {
        "policy_id": "near_zero_sign_policy",
        "zero_tol": ZERO_TOL,
        "rules": [
            {"when": "abs(published_i) < zero_tol and abs(computed_i) < zero_tol", "status": "near_zero_compatible"},
            {"when": "sign(computed_i) == sign(published_i)", "status": "strict_match"},
            {"when": "abs(published_i) < zero_tol or abs(computed_i) < zero_tol", "status": "near_zero_boundary_crossing"},
            {"when": "otherwise", "status": "strict_mismatch"},
        ],
        "recommendation": (
            "A quantitative absolute-error pass may still cross the strict sign boundary "
            "for near-zero exponents. Use near_zero_sign_boundary before interpreting sign failures."
        ),
    }
    (diagnostics_dir / "near_zero_sign_policy.json").write_text(
        json.dumps(policy, indent=2) + "\n",
        encoding="utf-8",
    )
    sensitivity_summary = {
        "status": "planned_not_executed",
        "opt_in": "RUN_F3_DISCREPANCY_SWEEPS=1",
        "validated_after_sensitivity": False,
        "notes": ["Sensitivity sweeps are diagnostic only and do not promote F3."],
    }
    sensitivity_summary_path = diagnostics_dir / "sensitivity_summary.json"
    if not sensitivity_summary_path.exists():
        sensitivity_summary_path.write_text(
            json.dumps(sensitivity_summary, indent=2) + "\n",
            encoding="utf-8",
        )
    for filename in (
        "sensitivity_delta.csv",
        "sensitivity_t_clone.csv",
        "sensitivity_h.csv",
        "sensitivity_k.csv",
        "sensitivity_gs_policy.csv",
        "sensitivity_q1_mode.csv",
    ):
        path = diagnostics_dir / filename
        if not path.exists():
            _write_csv(path, [], SENSITIVITY_FIELDS)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "source_summary": _display_path(summary_path),
        "outputs_csv": _display_path(outputs_csv_path),
        **consistency,
        "rows_total": len(matrix),
        "diagnostic_classes": dict(Counter(row["discrepancy_class"] for row in matrix)),
        "validated_after_diagnostics": False,
        "certifications": {
            "chaos_certified_by_this_pipeline": False,
            "hiddenness_certified_by_this_pipeline": False,
        },
    }
    (diagnostics_dir / "diagnostic_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    row_notes = {
        f"{row['case_file']}#{row['row_index']}": {
            "discrepancy_class": row["discrepancy_class"],
            "interpretation": row["interpretation"],
            "notes": row["notes"],
        }
        for row in matrix
        if row["notes"]
    }
    (diagnostics_dir / "row_level_notes.json").write_text(
        json.dumps(row_notes, indent=2) + "\n",
        encoding="utf-8",
    )
    (diagnostics_dir / "fischer2020_discrepancy_report.md").write_text(
        _render_report(matrix, consistency),
        encoding="utf-8",
    )
    (diagnostics_dir / "README.md").write_text(
        "# F3 Fischer 2020 discrepancy diagnostics\n\n"
        "This directory classifies the recorded F3 discrepancies without promoting\n"
        "validation. See [the report](fischer2020_discrepancy_report.md) and the\n"
        "reproducible [sensitivity plan](sensitivity_plan.yaml).\n",
        encoding="utf-8",
    )
    summary["discrepancy_diagnostics"] = {
        "status": "diagnostics_added",
        "report": "discrepancy_diagnostics/fischer2020_discrepancy_report.md",
        "matrix": "discrepancy_diagnostics/fischer2020_discrepancy_matrix.csv",
        "near_zero_sign_policy": "discrepancy_diagnostics/near_zero_sign_policy.json",
        "sensitivity_status": "planned_or_partial",
        "validated_after_diagnostics": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"summary": summary, "matrix": matrix, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--outputs-csv", type=Path, default=OUTPUT_RESULTS_PATH)
    parser.add_argument("--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR)
    args = parser.parse_args()
    generated = generate_diagnostics(args.summary, args.outputs_csv, args.diagnostics_dir)
    metadata = generated["metadata"]
    print(
        f"Recorded {metadata['rows_total']} F3 diagnostic rows; "
        f"outputs_csv_present={metadata['outputs_csv_present']}; "
        f"outputs_csv_consistent={metadata['outputs_csv_consistent']}"
    )


if __name__ == "__main__":
    main()
