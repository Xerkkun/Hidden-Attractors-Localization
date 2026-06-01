"""Execution helpers for Fischer 2020 cloned-dynamics benchmarks."""

from __future__ import annotations

import csv
import json
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


def write_results(output_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write conservative validation outputs without promotion."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "published_benchmark_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "method_id", "system", "orders", "type", "computed_LE", "published_LE",
            "abs_error", "sign_match", "K01_published", "status",
        ])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method_id": "fractional_cloned_dynamics_abm_gs_published",
        "status": "published_benchmarks_not_run" if not rows else "published_benchmark_results_recorded_without_automatic_promotion",
        "validated": False,
        "validated_against_published_benchmarks": False,
        "memory_protocol": "published_block_restart",
        "certifications": {
            "chaos_certified_by_this_pipeline": False,
            "hiddenness_certified_by_this_pipeline": False,
        },
        "results": rows,
    }
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


__all__ = ["ALLOWED_STATUSES", "load_benchmark_case", "run_benchmark_row", "write_results"]
