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
from hidden_attractors.analysis.lyapunov_cloned import (
    CLONED_EXECUTION_PROTOCOLS, LEGACY_EXECUTION_PROTOCOL, REFERENCE_RAIL_PROTOCOL,
)
from hidden_attractors.systems.fischer_benchmarks import get_fischer_benchmark


ALLOWED_STATUSES = {
    "published_benchmark_passed_quantitative",
    "published_benchmark_passed_sign_pattern",
    "published_benchmark_failed_quantitative",
    "published_benchmark_inconclusive",
    "published_benchmark_smoke_passed",
}
METHOD_ID = "fractional_cloned_dynamics_abm_gs_published"
LEGACY_CRITERIA_VERSION = "published_tolerance_first_v1"
CURRENT_CRITERIA_VERSION = "published_tolerance_and_required_sign_v2"
EVALUATION_CRITERIA_VERSIONS = (LEGACY_CRITERIA_VERSION, CURRENT_CRITERIA_VERSION)
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


def evaluate_benchmark_spectrum(
    computed: list[float] | np.ndarray,
    published: list[float] | np.ndarray,
    expected: dict[str, Any],
    *,
    solver_status: str = "ok",
    fast: bool = False,
    criteria_version: str | None = None,
) -> dict[str, Any]:
    """Evaluate an existing spectrum without integrating or relaxing its signs.

    V2 enforces ``sign_pattern_required`` also for a quantitative pass. The
    explicit V1 option preserves the tolerance-first historical evaluator.
    Near-zero crossings remain diagnostic; they never change a strict sign.
    """

    version = criteria_version or expected.get("evaluation_criteria_version", CURRENT_CRITERIA_VERSION)
    if version not in EVALUATION_CRITERIA_VERSIONS:
        raise ValueError(f"unknown benchmark evaluation criteria: {version}")
    actual = np.asarray(computed, dtype=float)
    reference = np.asarray(published, dtype=float)
    if actual.ndim != 1 or actual.shape != reference.shape or not actual.size:
        raise ValueError("computed and published spectra must have the same nonempty vector shape")
    finite = bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)))
    errors = np.abs(actual - reference)
    signs_match = _sign_match(actual, reference) if finite else False
    target = float(expected["tolerance_abs_target"])
    initial = float(expected["tolerance_abs_initial"])
    if not np.isfinite(target) or not np.isfinite(initial) or min(target, initial) <= 0:
        raise ValueError("benchmark tolerances must be finite and positive")
    required = expected.get("sign_pattern_required", False)
    if not isinstance(required, bool):
        raise ValueError("sign_pattern_required must be a YAML boolean")
    quantitative_tolerance_met = bool(finite and np.all(errors < target))
    if solver_status != "ok" or not finite:
        legacy_status = status = "published_benchmark_inconclusive"
    elif fast:
        legacy_status = status = "published_benchmark_smoke_passed"
    else:
        if quantitative_tolerance_met:
            legacy_status = "published_benchmark_passed_quantitative"
        elif signs_match and errors[0] < initial:
            legacy_status = "published_benchmark_passed_sign_pattern"
        else:
            legacy_status = "published_benchmark_failed_quantitative"
        status = legacy_status
        if (version == CURRENT_CRITERIA_VERSION and quantitative_tolerance_met
                and required and not signs_match):
            status = "published_benchmark_failed_quantitative"
    zero_tol = float(expected.get("near_zero_diagnostic_tolerance", 0.02))
    if not np.isfinite(zero_tol) or zero_tol < 0:
        raise ValueError("near-zero diagnostic tolerance must be finite and nonnegative")
    component_diagnostics = []
    for value, baseline in zip(actual, reference):
        if not np.isfinite(value) or not np.isfinite(baseline):
            label = "nonfinite"
        elif np.sign(value) == np.sign(baseline):
            label = "strict_match"
        elif abs(value) < zero_tol or abs(baseline) < zero_tol:
            label = "near_zero_boundary_crossing"
        else:
            label = "strict_mismatch"
        component_diagnostics.append(label)
    return {
        "status": status,
        "abs_error": errors.tolist(),
        "sign_match": signs_match,
        "evaluation_criteria_version": version,
        "legacy_status": legacy_status,
        "quantitative_tolerance_met": quantitative_tolerance_met,
        "tolerance_abs_target": target,
        "tolerance_abs_initial": initial,
        "sign_pattern_required": required,
        "required_sign_gate_met": not required or signs_match,
        "near_zero_diagnostic_tolerance": zero_tol,
        "component_sign_diagnostics": component_diagnostics,
        "near_zero_diagnostic_changes_verdict": False,
        "solver_status": solver_status,
    }


def run_benchmark_row(
    case: dict[str, Any],
    row: dict[str, Any],
    *,
    fast: bool = False,
    case_file: str | None = None,
    row_index: int | None = None,
    criteria_version: str | None = None,
    execution_protocol: str | None = None,
) -> tuple[dict[str, Any], object]:
    """Execute one published row or a reduced smoke version of it."""

    legacy_rhs, x0, _ = get_fischer_benchmark(case["system"]["kind"])
    x0 = np.asarray(case["system"].get("x0", x0), dtype=float)

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        """Adapt the state/parameters model to the canonical time/state API."""
        return legacy_rhs(state, case["system"]["parameters"])

    integration = case["integration"]
    has_reference_contract = all(key in integration for key in ("h", "T_ref", "N"))
    protocol = execution_protocol or integration.get(
        "execution_protocol", REFERENCE_RAIL_PROTOCOL if has_reference_contract else LEGACY_EXECUTION_PROTOCOL,
    )
    if protocol not in CLONED_EXECUTION_PROTOCOLS:
        raise ValueError(f"unknown cloned execution protocol: {protocol}")
    reference_arguments = {}
    if protocol == REFERENCE_RAIL_PROTOCOL:
        if not has_reference_contract:
            raise ValueError("reference-rail benchmark needs h, T_ref and N")
        if not np.isclose(float(integration["h"]) * int(integration["N"]), float(integration["T_ref"]), rtol=1e-12, atol=0):
            raise ValueError("reference h*N must equal T_ref")
        if not np.isclose(float(integration["t_clone"]) * int(integration["K"]), float(integration["T_ref"]), rtol=1e-12, atol=0):
            raise ValueError("clone K*t_clone must equal T_ref")
        reference_arguments = {
            "reference_h": float(integration["h"]),
            "reference_sampling": integration.get("reference_sampling", "grid_exact"),
            "reference_convolution_block_size": int(integration.get("reference_convolution_block_size", 256)),
        }
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
        execution_protocol=protocol,
        **reference_arguments,
    )
    published = np.asarray(row["lyapunov"], dtype=float)
    computed = np.asarray(result.exponents, dtype=float)
    evaluation = evaluate_benchmark_spectrum(
        computed, published, case["expected"], solver_status=result.status,
        fast=fast, criteria_version=criteria_version,
    )
    return {
        "method_id": getattr(result, "method_id", case["method_id"]),
        "requested_method_id": case["method_id"],
        "execution_protocol": protocol,
        "reference_rail": getattr(result, "method_metadata", {}).get("reference_rail"),
        "case_file": case_file,
        "row_index": row_index,
        "system": case["system"]["kind"],
        "orders": row["orders"],
        "type": row["type"],
        "computed_LE": computed.tolist(),
        "published_LE": published.tolist(),
        "abs_error": evaluation["abs_error"],
        "sign_match": evaluation["sign_match"],
        "K01_published": row["K01"],
        **evaluation,
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
        not row["sign_match"] or row["abs_error"][0] >= row.get("tolerance_abs_initial", 0.15)
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
        "schema_version": "1.1",
        "evaluation_criteria_versions": sorted({
            row.get("evaluation_criteria_version", LEGACY_CRITERIA_VERSION) for row in rows
        }),
        "execution_protocols": sorted({
            row.get("execution_protocol", LEGACY_EXECUTION_PROTOCOL) for row in rows
        }),
        "method_ids": sorted({row["method_id"] for row in rows}),
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


__all__ = [
    "ALLOWED_STATUSES", "CURRENT_CRITERIA_VERSION", "LEGACY_CRITERIA_VERSION",
    "EVALUATION_CRITERIA_VERSIONS", "evaluate_benchmark_spectrum",
    "CLONED_EXECUTION_PROTOCOLS", "LEGACY_EXECUTION_PROTOCOL", "REFERENCE_RAIL_PROTOCOL",
    "load_benchmark_case", "run_benchmark_row", "write_results",
]
