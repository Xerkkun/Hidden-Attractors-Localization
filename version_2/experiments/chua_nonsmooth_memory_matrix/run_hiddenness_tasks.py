#!/usr/bin/env python3
"""Test continued candidate seeds from fixed equilibrium-neighborhood clouds.

Describing-function and continuation outputs remain candidate generators only.
This script integrates the target Caputo system at ``q_target`` and records
quantitative finite-radius evidence; it never emits ``hidden_verified``.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from _common import (
    CLASS_LABELS,
    VERSION2_ROOT,
    assert_unique_cache_keys,
    full_history_horizon,
    is_ok_status,
    json_result,
    load_matrix,
    metadata,
    optional_float,
    read_csv_rows,
    run_process_pool,
    status_counts,
    write_status,
    write_trajectory,
)

from hidden_attractors.analysis import classify_trajectory_against_equilibria, trajectory_metrics
from hidden_attractors.diagnostics import classify_post_transient_periodicity
from hidden_attractors.io import read_json, write_csv
from hidden_attractors.models import chua_nonsmooth_parameters, equilibria_nonsmooth
from hidden_attractors.native.backends import FractionalChuaBackend, FullHistoryABMBackend


def _continuation_dir(root: Path, cache_key: str) -> Path:
    """Find the continuation output directory named by a hiddenness task."""

    for row in read_csv_rows(root / "tasks" / "continuation_tasks.csv"):
        if row["cache_key"] == cache_key:
            return root / row["output_dir"]
    raise KeyError(f"continuation cache key {cache_key} was not found.")


def _cloud_rows(root: Path) -> list[dict[str, str]]:
    """Load the three shared, identical initial-condition cloud artifacts."""

    rows: list[dict[str, str]] = []
    for name in ("E0", "Eplus", "Eminus"):
        rows.extend(read_csv_rows(root / "shared" / "equilibrium_clouds" / f"{name}.csv"))
    return rows


def _integrator(row: dict[str, str], contract: dict[str, Any]) -> tuple[Callable[[np.ndarray], np.ndarray], str, float | None]:
    """Construct an existing target-system integrator for one solver/memory cell."""

    q = float(contract["q_target"])
    h = float(contract["h"])
    t_final = float(contract["t_final"])
    if row["hiddenness_memory"] == "full":
        lm = full_history_horizon(t_final, h)
        policy = "full_caputo_history_no_finite_memory_truncation"
    else:
        lm = float(optional_float(row.get("memory_length")) or contract["memory_length"])
        policy = "finite_caputo_history_window"
    params = chua_nonsmooth_parameters()
    output_name = f"memory_matrix_hidden_{row['cache_key']}_{os.getpid()}"
    if row["hiddenness_integrator"] == "abm":
        backend = FullHistoryABMBackend.build(output_name=output_name)
        backend.set_nonsmooth_params(params)
        if row["hiddenness_memory"] == "full":
            return lambda seed: backend.integrate(seed, q=q, h=h, t_final=t_final), policy, None
        return lambda seed: backend.integrate_truncated(seed, q=q, h=h, Lm=lm, t_final=t_final), policy, lm
    if row["hiddenness_integrator"] == "efork":
        backend = FractionalChuaBackend.build(output_name=output_name)
        backend.set_nonsmooth_params(params)
        return lambda seed: backend.integrate_efork3(seed, q=q, h=h, Lm=lm, t_final=t_final), policy, lm
    raise ValueError(f"unknown hiddenness_integrator {row['hiddenness_integrator']}")


def _candidate_class(
    trajectory: np.ndarray,
    *,
    sign: str,
    q: float,
    h: float,
    t_burn: float,
    equilibria: dict[str, np.ndarray],
    reference: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Assign an operational destination label against a candidate reference."""

    coarse = classify_trajectory_against_equilibria(trajectory, equilibria, t_start=t_burn)
    metrics, _payload = trajectory_metrics(trajectory, h=h, t_start=t_burn, reference=reference)
    periodicity = classify_post_transient_periodicity(
        trajectory,
        h=h,
        config={"t_transient": t_burn},
    )
    if bool(coarse["diverged"]) or bool(metrics["diverged"]):
        return "infinity", metrics, periodicity
    if bool(coarse["equilibrium_hit"]):
        mapping = {"E0": "equilibrium_E0", "E+": "equilibrium_Eplus", "E-": "equilibrium_Eminus"}
        return mapping[str(coarse["closest_equilibrium"])], metrics, periodicity
    cloud = float(metrics.get("cloud_median_distance_norm", float("nan")))
    ranges = float(metrics.get("range_relative_distance", float("nan")))
    target_hit = bool(
        metrics["bounded"]
        and metrics["noncollapsed_variance"]
        and math.isfinite(cloud)
        and math.isfinite(ranges)
        and cloud <= 0.35
        and ranges <= 0.60
    )
    if target_hit:
        return f"target_candidate_{sign}", metrics, periodicity
    if bool(periodicity.get("periodic_post_transient")):
        return "periodic_or_quasiperiodic", metrics, periodicity
    return "bounded_other", metrics, periodicity


def _failed_output(outdir: Path, meta: dict[str, Any], reason: str) -> None:
    """Write required hiddenness artifacts when prerequisites or integration fail."""

    write_csv(outdir / "hiddenness_raw.csv", [], ["class_label", "status"])
    json_result(outdir / "hiddenness_summary.json", {"status": "numerical_failure", "hiddenness_status": "numerical_failure", "reason": reason, "metadata": meta})
    write_status(outdir / "status.json", status="failed", meta=meta, outputs=["hiddenness_raw.csv", "hiddenness_summary.json", "representative_trajectories/"], reason=reason)


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one sign-specific target-system hiddenness experiment."""

    root = Path(job["root"])
    manifest = job["manifest"]
    contract = dict(manifest["contract"])
    row = job["row"]
    outdir = root / row["output_dir"]
    status_path = outdir / "status.json"
    required = [outdir / "hiddenness_raw.csv", outdir / "hiddenness_summary.json", outdir / "representative_trajectories"]
    if not bool(job["force"]) and is_ok_status(status_path, required):
        return {"task_id": row["task_id"], "status": "skipped_ok"}
    outdir.mkdir(parents=True, exist_ok=True)
    q = float(contract["q_target"])
    memory_policy = "full" if row["hiddenness_memory"] == "full" else "truncated"
    meta = metadata(
        manifest,
        row,
        stage="hiddenness",
        q=q,
        integrator=row["hiddenness_integrator"],
        memory_policy=memory_policy,
        workers=int(job["workers"]),
        extra={"sign": row["sign"], "classification_labels": list(CLASS_LABELS)},
    )
    try:
        cont_dir = _continuation_dir(root, row["continuation_cache_key"])
        cont_status = read_json(cont_dir / "status.json")
        if cont_status.get("status") != "ok":
            reason = "prerequisite continuation did not complete successfully."
            _failed_output(outdir, meta, reason)
            return {"task_id": row["task_id"], "status": "failed", "reason": reason}
        candidate = read_json(cont_dir / f"final_seed_{row['sign']}.json")
        seed = np.asarray(candidate["seed"], dtype=float)
        integrate, policy, effective_lm = _integrator(row, contract)
        meta["memory_policy"] = policy
        meta["Lm"] = effective_lm
        reference_trajectory = integrate(seed)
        write_trajectory(outdir / "representative_trajectories" / f"target_candidate_{row['sign']}.csv", reference_trajectory, max_rows=5000)
        reference_metrics, reference_payload = trajectory_metrics(
            reference_trajectory,
            h=float(contract["h"]),
            t_start=float(contract["t_burn"]),
        )
        periodicity = classify_post_transient_periodicity(
            reference_trajectory,
            h=float(contract["h"]),
            config={"t_transient": float(contract["t_burn"])},
        )
        cloud_rows = _cloud_rows(root)
        max_probes = int(job["max_probes_per_cloud"])
        if max_probes > 0:
            selected: list[dict[str, str]] = []
            for eq_name in ("E0", "E+", "E-"):
                eq_rows = [item for item in cloud_rows if item["equilibrium"] == eq_name]
                selected.extend(eq_rows[:max_probes])
            cloud_rows = selected
        equilibria = equilibria_nonsmooth(chua_nonsmooth_parameters())
        raw: list[dict[str, Any]] = []
        representative_saved = {f"target_candidate_{row['sign']}"}
        failures = 0
        for item in cloud_rows:
            initial = np.asarray([float(item["x0"]), float(item["y0"]), float(item["z0"])], dtype=float)
            try:
                trajectory = integrate(initial)
                label, metrics, probe_periodicity = _candidate_class(
                    trajectory,
                    sign=row["sign"],
                    q=q,
                    h=float(contract["h"]),
                    t_burn=float(contract["t_burn"]),
                    equilibria=equilibria,
                    reference=reference_payload,
                )
            except Exception as exc:
                label = "numerical_failure"
                metrics = {"exception": str(exc)}
                probe_periodicity = {"periodicity_status": "numerical_failure"}
                trajectory = np.empty((0, 4), dtype=float)
                failures += 1
            if label not in representative_saved and trajectory.size:
                write_trajectory(outdir / "representative_trajectories" / f"{label}.csv", trajectory, max_rows=5000)
                representative_saved.add(label)
            raw.append(
                {
                    **item,
                    "exp_id": row["exp_id"],
                    "cache_key": row["cache_key"],
                    "sign": row["sign"],
                    "hiddenness_integrator": row["hiddenness_integrator"],
                    "hiddenness_memory": row["hiddenness_memory"],
                    "class_label": label,
                    "target_hit": label == f"target_candidate_{row['sign']}",
                    "bounded": metrics.get("bounded", False),
                    "max_norm": metrics.get("max_norm", float("nan")),
                    "cloud_median_distance_norm": metrics.get("cloud_median_distance_norm", float("nan")),
                    "range_relative_distance": metrics.get("range_relative_distance", float("nan")),
                    "fft_relative_delta": metrics.get("fft_relative_delta", float("nan")),
                    "periodicity_status": probe_periodicity.get("periodicity_status", ""),
                }
            )
        write_csv(outdir / "hiddenness_raw.csv", raw)
        counts = status_counts(raw, "class_label")
        target_hits = int(counts.get(f"target_candidate_{row['sign']}", 0))
        sampled_fraction_is_complete = max_probes <= 0
        if failures:
            verdict = "numerical_failure"
        elif target_hits:
            verdict = "not_hidden_under_tested_radii"
        elif bool(periodicity.get("periodic_post_transient")):
            verdict = "inconclusive"
        elif not sampled_fraction_is_complete:
            verdict = "inconclusive"
        else:
            verdict = "compatible_with_hiddenness_under_tested_radii"
        summary = {
            "status": "ok" if not failures else "numerical_failure",
            "hiddenness_status": verdict,
            "scientific_warning": "Zero contact is finite-radius numerical evidence only; periodic candidates remain inconclusive and no outcome here proves global hiddenness.",
            "candidate": f"target_candidate_{row['sign']}",
            "reached_candidate_from_seed": bool(reference_metrics["bounded"] and reference_metrics["noncollapsed_variance"]),
            "reference_periodicity_status": periodicity["periodicity_status"],
            "reference_periodicity_gate": "inconclusive_if_periodic_post_transient",
            "reference_boundedness_status": "bounded" if reference_metrics["bounded"] else "infinity",
            "tested_initial_conditions": len(raw),
            "complete_shared_clouds_used": sampled_fraction_is_complete,
            "any_equilibrium_ball_hit": bool(target_hits > 0),
            "target_hits": target_hits,
            "failure_rate": float(failures / len(raw)) if raw else 1.0,
            "class_counts": counts,
            "metadata": meta,
        }
        json_result(outdir / "hiddenness_summary.json", summary)
        status = "ok" if not failures else "failed"
        write_status(status_path, status=status, meta=meta, outputs=["hiddenness_raw.csv", "hiddenness_summary.json", "representative_trajectories/"])
        return {"task_id": row["task_id"], "status": status}
    except Exception as exc:
        _failed_output(outdir, meta, str(exc))
        return {"task_id": row["task_id"], "status": "failed", "reason": str(exc)}


def main() -> None:
    """Dispatch hiddenness task rows with process-level parallelism only."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(VERSION2_ROOT / "outputs/chua_nonsmooth_fractional_memory_matrix/tasks/hiddenness_tasks.csv"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-probes-per-cloud", type=int, default=0, help="Diagnostic partial run only; a positive cap forces an inconclusive no-hit verdict.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root, manifest, rows = load_matrix(args.tasks)
    assert_unique_cache_keys(rows)
    jobs = [
        {
            "root": str(root),
            "manifest": manifest,
            "row": row,
            "workers": args.workers,
            "max_probes_per_cloud": args.max_probes_per_cloud,
            "force": args.force,
        }
        for row in rows
    ]
    for result in run_process_pool(run_one, jobs, workers=args.workers):
        print(f"{result['task_id']}: {result['status']}")


if __name__ == "__main__":
    main()
