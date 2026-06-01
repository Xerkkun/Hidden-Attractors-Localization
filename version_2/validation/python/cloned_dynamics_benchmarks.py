"""Execution helpers for Fischer 2020 cloned-dynamics benchmarks."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hidden_attractors.analysis import compute_cloned_dynamics_spectrum
from hidden_attractors.systems.fischer_benchmarks import get_fischer_benchmark


ALLOWED_STATUSES = {
    "published_benchmark_passed_quantitative",
    "published_benchmark_passed_sign_pattern",
    "published_benchmark_failed_quantitative",
    "published_benchmark_inconclusive",
    "published_benchmark_smoke_passed",
}
METHOD_ID = "fractional_cloned_dynamics_abm_gs_published"
REFERENCE = {
    "authors": ["C. Fischer", "K. L. A. Zourmba", "A. Mohamadou"],
    "year": 2020,
    "doi": "10.1016/j.apnum.2020.03.027",
}
NOTES = [
    "Finite-time local Lyapunov indicators only.",
    "Published_block_restart is not full-memory Caputo-aware.",
    "Passing F3 does not validate fractional_variational_abm_qr.",
    "Passing F3 does not certify chaos or hiddenness.",
]


def load_benchmark_case(path: str | Path) -> dict[str, Any]:
    """Load one Fischer YAML specification."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _sign_match(computed: np.ndarray, published: np.ndarray) -> bool:
    return bool(np.array_equal(np.sign(computed), np.sign(published)))


def run_benchmark_row(
    case: dict[str, Any],
    row: dict[str, Any],
    *,
    fast: bool = False,
    case_file: str | None = None,
    row_index: int | None = None,
) -> tuple[dict[str, Any], object]:
    """Execute one published row or a reduced smoke version of it."""

    rhs, x0, _ = get_fischer_benchmark(case["system"]["kind"])
    integration = case["integration"]
    h = float(integration["h_clone"])
    t_clone = float(integration["t_clone"])
    k_blocks = int(integration["K"])
    if fast:
        t_clone = min(t_clone, 5 * h)
        k_blocks = min(k_blocks, 2)
    method = "qr" if case["method_id"].endswith("_qr") else "gs"
    result = compute_cloned_dynamics_spectrum(
        rhs,
        x0,
        orders=row["orders"],
        h=h,
        t_clone=t_clone,
        n_clones=len(x0),
        k_blocks=k_blocks,
        delta=float(integration["delta"]),
        method=method,
        memory_protocol=integration["memory_protocol"],
        system_id=case["system"]["kind"],
        parameters=case["system"]["parameters"],
        return_history=True,
    )
    published = np.asarray(row["lyapunov"], dtype=float)
    computed = np.asarray(result.exponents, dtype=float)
    errors = np.abs(computed - published)
    signs_match = _sign_match(computed, published) if np.all(np.isfinite(computed)) else False
    if result.status != "ok" or not np.all(np.isfinite(computed)):
        status = "published_benchmark_inconclusive"
    elif fast:
        status = "published_benchmark_smoke_passed"
    elif np.all(errors < float(case["expected"]["tolerance_abs_target"])):
        status = "published_benchmark_passed_quantitative"
    elif signs_match and errors[0] < float(case["expected"]["tolerance_abs_initial"]):
        status = "published_benchmark_passed_sign_pattern"
    else:
        status = "published_benchmark_failed_quantitative"
    return {
        "method_id": case["method_id"],
        "case_file": case_file,
        "row_index": row_index,
        "system": case["system"]["kind"],
        "orders": row["orders"],
        "type": row["type"],
        "computed_LE": computed.tolist(),
        "published_LE": published.tolist(),
        "abs_error": errors.tolist(),
        "sign_match": signs_match,
        "K01_published": row["K01"],
        "status": status,
    }, result


def _git_metadata() -> dict[str, object]:
    """Capture the source revision without failing outside a Git checkout."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {"commit": commit, "working_tree_dirty": dirty}


def _summary_status(rows: list[dict[str, Any]], *, fast: bool) -> str:
    if not rows:
        return "published_benchmarks_not_run"
    if fast:
        return "published_benchmark_smoke_only_not_quantitative_validation"
    statuses = {row["status"] for row in rows}
    if "published_benchmark_inconclusive" in statuses:
        return "published_benchmarks_inconclusive_numerical_failure"
    if "published_benchmark_failed_quantitative" in statuses:
        return "published_benchmarks_pending_discrepancy"
    if statuses == {"published_benchmark_passed_quantitative"}:
        return "published_quantitative_validated"
    if statuses <= {
        "published_benchmark_passed_quantitative",
        "published_benchmark_passed_sign_pattern",
    }:
        return "published_sign_pattern_supported_not_quantitatively_validated"
    return "published_benchmarks_pending_discrepancy"


def _write_convergence(
    output: Path,
    executions: list[tuple[dict[str, Any], object]],
) -> None:
    """Store representative first-row commensurate convergence histories."""

    selected = []
    seen_systems: set[str] = set()
    for row, result in executions:
        if row["type"] != "Comm" or row["row_index"] != 0 or row["system"] in seen_systems:
            continue
        seen_systems.add(row["system"])
        history = result.method_metadata.get("history") or []
        for block in history:
            selected.append(
                {
                    "system": row["system"],
                    "orders": row["orders"],
                    "block": block["block"],
                    "time": block["time"],
                    "lambda_1": block["exponents"][0],
                    "lambda_2": block["exponents"][1],
                    "lambda_3": block["exponents"][2],
                }
            )
    path = output / "convergence_by_block.csv"
    fields = ["system", "orders", "block", "time", "lambda_1", "lambda_2", "lambda_3"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)


def write_results(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    *,
    executions: list[tuple[dict[str, Any], object]] | None = None,
    fast: bool = False,
    command: str | None = None,
    global_parameters: list[dict[str, Any]] | None = None,
    official_summary_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write auditable validation outputs and optionally the official verdict."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "method_id", "case_file", "row_index", "system", "orders", "type",
        "computed_LE", "published_LE", "abs_error", "sign_match", "K01_published",
        "status",
    ]
    for filename in ("published_benchmark_results.csv", "published_benchmark_all_rows_results.csv"):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    if executions is not None:
        _write_convergence(output, executions)

    status = _summary_status(rows, fast=fast)
    quantitative = sum(row["status"] == "published_benchmark_passed_quantitative" for row in rows)
    sign_pattern = sum(row["status"] == "published_benchmark_passed_sign_pattern" for row in rows)
    failures = sum(row["status"] == "published_benchmark_failed_quantitative" for row in rows)
    inconclusive = sum(row["status"] == "published_benchmark_inconclusive" for row in rows)
    sign_mismatches = sum(not row["sign_match"] for row in rows)
    strict_initial_gate_failures = sum(
        not row["sign_match"] or row["abs_error"][0] >= 0.15
        for row in rows
    )
    validated = status == "published_quantitative_validated"
    execution = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_class": "published_smoke_fast" if fast else "published_quantitative_long",
        "command": command,
        "python_version": sys.version,
        "platform": platform.platform(),
        "global_parameters": global_parameters or [],
        **_git_metadata(),
    }
    summary = {
        "schema_version": "1.0",
        "method_id": METHOD_ID,
        "reference": REFERENCE,
        "status": status,
        "validated": validated,
        "validated_against_published_benchmarks": validated,
        "memory_protocol": "published_block_restart",
        "benchmarks_run": bool(rows) and not fast,
        "smoke_only": fast,
        "rows_total": len(rows),
        "rows_passed_quantitative": quantitative,
        "rows_passed_sign_pattern": sign_pattern,
        "rows_failed": failures,
        "rows_inconclusive": inconclusive,
        "rows_sign_mismatch": sign_mismatches,
        "rows_strict_initial_gate_failures": strict_initial_gate_failures,
        "quick_ci_guarantee": "smoke_only_not_published_quantitative_validation",
        "published_quantitative_opt_in": "RUN_PUBLISHED_CLONED=1",
        "published_all_rows_opt_in": "RUN_PUBLISHED_CLONED_ALL=1",
        "execution": execution,
        "certifications": {
            "chaos_certified_by_this_pipeline": False,
            "hiddenness_certified_by_this_pipeline": False,
        },
        "notes": NOTES,
        "results": rows,
    }
    (output / "run_metadata.json").write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if official_summary_dir is not None:
        official = Path(official_summary_dir)
        official.mkdir(parents=True, exist_ok=True)
        (official / "validation_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


__all__ = ["ALLOWED_STATUSES", "load_benchmark_case", "run_benchmark_row", "write_results"]
