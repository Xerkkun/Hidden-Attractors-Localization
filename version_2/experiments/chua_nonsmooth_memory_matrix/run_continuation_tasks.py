#!/usr/bin/env python3
"""Execute independent continuation cells for the non-smooth Chua matrix.

The script compares EFORK and ABM using explicit continuation contracts.  The
integer-like route uses ``q=1`` and transmits only the preceding endpoint.
The fractional Caputo route transports either its complete ABM history or its
declared finite window; EFORK transports the effective window exposed by its
existing native continuation API.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from _common import (
    VERSION2_ROOT,
    assert_unique_cache_keys,
    experiment_spec,
    full_history_horizon,
    harmonic_seed_from_payload,
    is_ok_status,
    json_result,
    load_matrix,
    metadata,
    optional_float,
    read_csv_rows,
    run_process_pool,
    write_status,
    write_trajectory,
)

from hidden_attractors.io import read_json, write_csv
from hidden_attractors.native.backends import FractionalChuaBackend, FullHistoryABMBackend
from hidden_attractors.solvers import FractionalHistory
from hidden_attractors.systems import get_system
from hidden_attractors.workflows.integer_lure import continue_integer_lure_seed
from hidden_attractors.workflows.protocol import ContinuationPlan


def _seed_path(root: Path, row: dict[str, str]) -> Path:
    """Resolve the cached classical-DF seed selected by a continuation row."""

    family = "fractional" if row["seed_cache"] == "seed_fractional" else "integer_like"
    return root / "shared" / "seeds" / f"{family}_seed.json"


def _empty_outputs(outdir: Path, meta: dict[str, Any], reason: str) -> list[str]:
    """Write the required empty artifacts for an explicitly failed continuation."""

    outputs = [
        "continuation_summary.json",
        "continuation_path.csv",
        "final_history_window.npz",
        "final_seed_plus.json",
        "final_seed_minus.json",
        "trajectory_tail.csv",
        "status.json",
    ]
    json_result(outdir / "continuation_summary.json", {"status": "failed", "reason": reason, "metadata": meta})
    write_csv(outdir / "continuation_path.csv", [], ["lambda", "eta", "status"])
    np.savez_compressed(outdir / "final_history_window.npz", t_window=np.empty(0), x_window=np.empty((0, 3)), metadata=np.array(str(meta)))
    json_result(outdir / "final_seed_plus.json", {"status": "failed", "reason": reason, "metadata": meta})
    json_result(outdir / "final_seed_minus.json", {"status": "failed", "reason": reason, "metadata": meta})
    write_csv(outdir / "trajectory_tail.csv", [], ["t", "x", "y", "z"])
    write_status(outdir / "status.json", status="failed", meta=meta, outputs=outputs, reason=reason)
    return outputs


def _store_final_seeds(outdir: Path, final_state: np.ndarray, meta: dict[str, Any]) -> list[str]:
    """Store symmetric candidate seeds without claiming either is an attractor."""

    state = np.asarray(final_state, dtype=float)
    plus = state if state[0] >= 0.0 else -state
    minus = -plus
    common = {
        "status": "ok",
        "interpretation": "continued_candidate_seed_pending_Caputo_hiddenness_test",
        "metadata": meta,
    }
    json_result(outdir / "final_seed_plus.json", {**common, "candidate_label": "target_candidate_plus", "seed": plus})
    json_result(outdir / "final_seed_minus.json", {**common, "candidate_label": "target_candidate_minus", "seed": minus})
    return ["final_seed_plus.json", "final_seed_minus.json"]


def _integer_efork(
    root: Path,
    outdir: Path,
    manifest: dict[str, Any],
    row: dict[str, str],
    meta: dict[str, Any],
    *,
    stage_transient: float,
    stage_keep: float,
) -> dict[str, Any]:
    """Run integer Lur'e continuation with the required last-point policy."""

    seed = harmonic_seed_from_payload(read_json(_seed_path(root, row)))
    system = get_system("chua-nonsmooth")
    plan = ContinuationPlan.uniform(int(manifest["contract"]["eta_steps"]), internal_parameter="eta")
    steps = continue_integer_lure_seed(
        system,
        seed,
        plan=plan,
        t_transient=stage_transient,
        t_keep=stage_keep,
        h=float(manifest["contract"]["h"]),
        div_threshold=120.0,
    )
    path_rows: list[dict[str, Any]] = []
    for step in steps:
        path_rows.append(
            {
                "lambda": step.lambda_value,
                "eta": step.lambda_value,
                "x_in": step.x_in[0],
                "y_in": step.x_in[1],
                "z_in": step.x_in[2],
                "x_out": step.x_out[0],
                "y_out": step.x_out[1],
                "z_out": step.x_out[2],
                "history_points_in": 1,
                "history_points_out": 1,
                "continuation_ic_policy": "last_point_only",
                "status": step.status,
                "cache_key": row["cache_key"],
            }
        )
    write_csv(outdir / "continuation_path.csv", path_rows)
    survived = bool(steps and len(steps) == len(plan.lambda_values) and all(step.status == "ok" for step in steps))
    if not steps:
        raise RuntimeError("integer continuation returned no stages.")
    final = steps[-1].x_out
    tail = steps[-1].trajectory
    np.savez_compressed(
        outdir / "final_history_window.npz",
        t_window=np.array([0.0]),
        x_window=final.reshape(1, 3),
        q=np.array([1.0]),
        h=np.array([float(manifest["contract"]["h"])]),
        memory_policy=np.array(["last_point_only"]),
    )
    _store_final_seeds(outdir, final, meta)
    write_trajectory(outdir / "trajectory_tail.csv", tail)
    summary = {
        "status": "ok" if survived else "failed",
        "survived": survived,
        "continuation_family": "integer_like_q1",
        "continuation_solver": "efork",
        "continuation_ic_policy": "last_point_only",
        "memory_interpretation": "integer-order continuation propagates only the last endpoint; no Caputo history applies",
        "stages_completed": len(steps),
        "stages_requested": len(plan.lambda_values),
        "final_state": final,
        "metadata": meta,
    }
    json_result(outdir / "continuation_summary.json", summary)
    return summary


def _integer_abm(
    root: Path,
    outdir: Path,
    manifest: dict[str, Any],
    row: dict[str, str],
    meta: dict[str, Any],
    *,
    stage_transient: float,
    stage_keep: float,
) -> dict[str, Any]:
    """Run q=1 ABM continuation, discarding each previous stage history.

    At integer order the ABM PECE discretization is used as an ODE
    continuation integrator.  Only ``x_out`` seeds the next eta step, as
    required by the ``last_point_only`` comparison contract.
    """

    seed = harmonic_seed_from_payload(read_json(_seed_path(root, row)))
    h = float(manifest["contract"]["h"])
    lambda_values = np.linspace(0.0, 1.0, int(manifest["contract"]["eta_steps"]))
    backend = FullHistoryABMBackend.build(output_name=f"memory_matrix_cont_{row['cache_key']}")
    x_current = np.asarray(seed.seed, dtype=float)
    path_rows: list[dict[str, Any]] = []
    tail = np.empty((0, 4), dtype=float)
    for lam in lambda_values:
        stage = backend.continue_full_history(
            x_current,
            lambda_values=[float(lam)],
            q=1.0,
            k=float(seed.gain),
            h=h,
            t_transient=stage_transient,
            t_keep=stage_keep,
        )
        x_out = np.asarray(stage["x_out"][0], dtype=float)
        if not np.all(np.isfinite(x_out)):
            raise RuntimeError(f"integer ABM continuation diverged at eta={lam}.")
        path_rows.append(
            {
                "lambda": float(lam),
                "eta": float(lam),
                "x_in": x_current[0],
                "y_in": x_current[1],
                "z_in": x_current[2],
                "x_out": x_out[0],
                "y_out": x_out[1],
                "z_out": x_out[2],
                "history_points_in": 1,
                "history_points_out": 1,
                "continuation_ic_policy": "last_point_only",
                "status": "ok",
                "cache_key": row["cache_key"],
            }
        )
        x_current = x_out
        tail = stage["trajectories"][0]
    write_csv(outdir / "continuation_path.csv", path_rows)
    np.savez_compressed(
        outdir / "final_history_window.npz",
        t_window=np.array([0.0]),
        x_window=x_current.reshape(1, 3),
        q=np.array([1.0]),
        h=np.array([h]),
        memory_policy=np.array(["last_point_only"]),
        exact_transported_window=np.array([True]),
    )
    _store_final_seeds(outdir, x_current, meta)
    write_trajectory(outdir / "trajectory_tail.csv", tail)
    summary = {
        "status": "ok",
        "survived": True,
        "continuation_family": "integer_like_q1",
        "continuation_solver": "abm",
        "continuation_ic_policy": "last_point_only",
        "memory_interpretation": "q=1 ABM stages restart from the preceding endpoint only",
        "stages_completed": int(lambda_values.size),
        "stages_requested": int(lambda_values.size),
        "final_state": x_current,
        "metadata": meta,
    }
    json_result(outdir / "continuation_summary.json", summary)
    return summary


def _observed_history(trajectories: np.ndarray, *, q: float, h: float, memory_length: float) -> FractionalHistory:
    """Assemble the transported observed tails and extract the recorded window."""

    joined: list[np.ndarray] = []
    offset = 0.0
    for index, segment in enumerate(np.asarray(trajectories, dtype=float)):
        values = segment.copy()
        if index:
            values = values[1:]
        values[:, 0] += offset
        if values.shape[0]:
            offset = float(values[-1, 0])
            joined.append(values)
    if not joined:
        raise RuntimeError("fractional continuation returned no observed trajectory.")
    return FractionalHistory.from_trajectory(
        np.vstack(joined),
        q=q,
        h=h,
        memory_length=memory_length,
    )


def _fractional_efork(
    root: Path,
    outdir: Path,
    manifest: dict[str, Any],
    row: dict[str, str],
    meta: dict[str, Any],
    *,
    stage_transient: float,
    stage_keep: float,
) -> dict[str, Any]:
    """Run Caputo EFORK continuation while carrying effective numerical history."""

    contract = manifest["contract"]
    q = float(contract["q_target"])
    h = float(contract["h"])
    seed = harmonic_seed_from_payload(read_json(_seed_path(root, row)))
    lambda_values = np.linspace(0.0, 1.0, int(contract["eta_steps"]))
    if row["continuation_memory"] == "full":
        lm = full_history_horizon(lambda_values.size * (stage_transient + stage_keep), h)
        policy = "full_history_via_nontruncating_EFORK_horizon"
    elif row["continuation_memory"] == "truncated":
        lm = float(optional_float(row.get("memory_length")) or contract["memory_length"])
        policy = "truncated_history_window"
        if stage_keep < lm:
            raise ValueError(
                "stage_keep must be at least Lm for truncated continuation so "
                "final_history_window.npz exactly represents the transported window."
            )
    else:
        raise ValueError("fractional_caputo requires continuation_memory full or truncated.")
    backend = FractionalChuaBackend.build(output_name=f"memory_matrix_cont_{row['cache_key']}")
    result = backend.continue_efork3(
        seed.seed,
        lambda_values=lambda_values,
        q=q,
        k=float(seed.gain),
        h=h,
        Lm=lm,
        t_transient=stage_transient,
        t_keep=stage_keep,
        t_observe=0.0,
        carry_memory=True,
    )
    path_rows = []
    for index, lam in enumerate(result["lambda"]):
        path_rows.append(
            {
                "lambda": float(lam),
                "eta": float(lam),
                "x_in": result["x_in"][index, 0],
                "y_in": result["x_in"][index, 1],
                "z_in": result["x_in"][index, 2],
                "x_out": result["x_out"][index, 0],
                "y_out": result["x_out"][index, 1],
                "z_out": result["x_out"][index, 2],
                "history_points_in": int(result["history_in_counts"][index]),
                "history_points_out": int(result["history_out_counts"][index]),
                "continuation_ic_policy": row["continuation_ic_policy"],
                "status": "ok",
                "cache_key": row["cache_key"],
            }
        )
    write_csv(outdir / "continuation_path.csv", path_rows)
    history = _observed_history(result["trajectories"], q=q, h=h, memory_length=lm)
    backend_history_points = int(result["history_out_counts"][-1])
    history_is_exact = bool(row["continuation_memory"] == "truncated")
    representation = (
        "exact_transported_truncated_window"
        if history_is_exact
        else "observed_segments_only_backend_transports_additional_internal_history_values_not_exposed_by_API"
    )
    np.savez_compressed(
        outdir / "final_history_window.npz",
        t_window=history.t_window,
        x_window=history.x_window,
        q=np.array([q]),
        h=np.array([h]),
        memory_length=np.array([lm]),
        memory_policy=np.array([policy]),
        representation=np.array([representation]),
        exact_transported_window=np.array([history_is_exact]),
        backend_history_points_out=np.array([backend_history_points]),
    )
    final = np.asarray(result["x_out"][-1], dtype=float)
    _store_final_seeds(outdir, final, meta)
    write_trajectory(outdir / "trajectory_tail.csv", result["trajectories"][-1])
    summary = {
        "status": "ok",
        "survived": True,
        "continuation_family": "fractional_caputo",
        "continuation_solver": "efork",
        "continuation_memory": row["continuation_memory"],
        "continuation_ic_policy": row["continuation_ic_policy"],
        "carry_memory": True,
        "effective_Lm": lm,
        "history_points_saved": history.memory_points,
        "backend_history_points_final": backend_history_points,
        "final_history_window_exact": history_is_exact,
        "final_history_window_representation": representation,
        "full_history_implementation": policy if row["continuation_memory"] == "full" else None,
        "stages_completed": int(lambda_values.size),
        "stages_requested": int(lambda_values.size),
        "final_state": final,
        "metadata": {**meta, "Lm": lm, "memory_policy": policy},
    }
    json_result(outdir / "continuation_summary.json", summary)
    return summary


def _fractional_abm(
    root: Path,
    outdir: Path,
    manifest: dict[str, Any],
    row: dict[str, str],
    meta: dict[str, Any],
    *,
    stage_transient: float,
    stage_keep: float,
) -> dict[str, Any]:
    """Run causal Caputo ABM continuation with exact reported history.

    Full mode retains the chronological Volterra history over all eta stages.
    Truncated mode retains the explicitly declared restarted window ``Lm``.
    Neither mode interprets the continued oscillation as an exact autonomous
    periodic solution of a Caputo system.
    """

    contract = manifest["contract"]
    q = float(contract["q_target"])
    h = float(contract["h"])
    seed = harmonic_seed_from_payload(read_json(_seed_path(root, row)))
    lambda_values = np.linspace(0.0, 1.0, int(contract["eta_steps"]))
    backend = FullHistoryABMBackend.build(output_name=f"memory_matrix_cont_{row['cache_key']}")
    if row["continuation_memory"] == "full":
        result = backend.continue_full_history(
            seed.seed,
            lambda_values=lambda_values,
            q=q,
            k=float(seed.gain),
            h=h,
            t_transient=stage_transient,
            t_keep=stage_keep,
        )
        lm: float | None = None
        policy = "full_caputo_history_ABM_causal_eta_chain"
    elif row["continuation_memory"] == "truncated":
        lm = float(optional_float(row.get("memory_length")) or contract["memory_length"])
        result = backend.continue_truncated_history(
            seed.seed,
            lambda_values=lambda_values,
            q=q,
            k=float(seed.gain),
            h=h,
            Lm=lm,
            t_transient=stage_transient,
            t_keep=stage_keep,
        )
        policy = "truncated_caputo_history_ABM_restarted_window"
    else:
        raise ValueError("fractional_caputo requires continuation_memory full or truncated.")
    path_rows: list[dict[str, Any]] = []
    for index, lam in enumerate(result["lambda"]):
        path_rows.append(
            {
                "lambda": float(lam),
                "eta": float(lam),
                "x_in": result["x_in"][index, 0],
                "y_in": result["x_in"][index, 1],
                "z_in": result["x_in"][index, 2],
                "x_out": result["x_out"][index, 0],
                "y_out": result["x_out"][index, 1],
                "z_out": result["x_out"][index, 2],
                "history_points_in": int(result["history_in_counts"][index]),
                "history_points_out": int(result["history_out_counts"][index]),
                "continuation_ic_policy": row["continuation_ic_policy"],
                "status": "ok",
                "cache_key": row["cache_key"],
            }
        )
    write_csv(outdir / "continuation_path.csv", path_rows)
    history = np.asarray(result["final_history"], dtype=float)
    np.savez_compressed(
        outdir / "final_history_window.npz",
        t_window=history[:, 0],
        x_window=history[:, 1:4],
        q=np.array([q]),
        h=np.array([h]),
        memory_length=np.array([np.nan if lm is None else lm]),
        memory_policy=np.array([policy]),
        representation=np.array(["exact_transported_abm_history"]),
        exact_transported_window=np.array([True]),
        backend_history_points_out=np.array([int(result["history_out_counts"][-1])]),
    )
    final = np.asarray(result["x_out"][-1], dtype=float)
    if not np.all(np.isfinite(final)):
        raise RuntimeError("fractional ABM continuation produced a non-finite final state.")
    _store_final_seeds(outdir, final, meta)
    write_trajectory(outdir / "trajectory_tail.csv", result["trajectories"][-1])
    summary = {
        "status": "ok",
        "survived": True,
        "continuation_family": "fractional_caputo",
        "continuation_solver": "abm",
        "continuation_memory": row["continuation_memory"],
        "continuation_ic_policy": row["continuation_ic_policy"],
        "carry_memory": True,
        "effective_Lm": lm,
        "history_points_saved": int(history.shape[0]),
        "backend_history_points_final": int(result["history_out_counts"][-1]),
        "final_history_window_exact": True,
        "final_history_window_representation": "exact_transported_abm_history",
        "full_history_implementation": policy if row["continuation_memory"] == "full" else None,
        "stages_completed": int(lambda_values.size),
        "stages_requested": int(lambda_values.size),
        "final_state": final,
        "metadata": {**meta, "Lm": lm, "memory_policy": policy},
    }
    json_result(outdir / "continuation_summary.json", summary)
    return summary


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one continuation task and isolate failures to its own output."""

    root = Path(job["root"])
    manifest = job["manifest"]
    row = job["row"]
    outdir = root / row["output_dir"]
    status_path = outdir / "status.json"
    required = [
        outdir / name
        for name in (
            "continuation_summary.json",
            "continuation_path.csv",
            "final_history_window.npz",
            "final_seed_plus.json",
            "final_seed_minus.json",
            "trajectory_tail.csv",
        )
    ]
    if not bool(job["force"]) and is_ok_status(status_path, required):
        return {"task_id": row["task_id"], "status": "skipped_ok"}
    outdir.mkdir(parents=True, exist_ok=True)
    q = float(row["q_continuation"])
    policy = "last_point_only" if row["continuation_family"] == "integer_like_q1" else row["continuation_memory"]
    meta = metadata(
        manifest,
        row,
        stage="continuation",
        q=q,
        integrator=row["continuation_solver"],
        memory_policy=policy,
        workers=int(job["workers"]),
        extra={
            "continuation_family": row["continuation_family"],
            "continuation_ic_policy": row["continuation_ic_policy"],
            "stage_transient": float(job["stage_transient"]),
            "stage_keep": float(job["stage_keep"]),
        },
    )
    try:
        if row["continuation_family"] == "integer_like_q1" and row["continuation_solver"] == "abm":
            summary = _integer_abm(root, outdir, manifest, row, meta, stage_transient=float(job["stage_transient"]), stage_keep=float(job["stage_keep"]))
        elif row["continuation_family"] == "integer_like_q1":
            summary = _integer_efork(root, outdir, manifest, row, meta, stage_transient=float(job["stage_transient"]), stage_keep=float(job["stage_keep"]))
        elif row["continuation_family"] == "fractional_caputo" and row["continuation_solver"] == "abm":
            summary = _fractional_abm(root, outdir, manifest, row, meta, stage_transient=float(job["stage_transient"]), stage_keep=float(job["stage_keep"]))
        elif row["continuation_family"] == "fractional_caputo":
            summary = _fractional_efork(root, outdir, manifest, row, meta, stage_transient=float(job["stage_transient"]), stage_keep=float(job["stage_keep"]))
        else:
            raise ValueError(f"unknown continuation_family {row['continuation_family']}")
        status = "ok" if summary["status"] == "ok" else "failed"
        if "effective_Lm" in summary and summary["effective_Lm"] is not None:
            meta["Lm"] = summary["effective_Lm"]
            meta["memory_policy"] = summary["metadata"]["memory_policy"]
        elif "metadata" in summary and "memory_policy" in summary["metadata"]:
            meta["Lm"] = summary["metadata"].get("Lm")
            meta["memory_policy"] = summary["metadata"]["memory_policy"]
        write_status(
            status_path,
            status=status,
            meta=meta,
            outputs=["continuation_summary.json", "continuation_path.csv", "final_history_window.npz", "final_seed_plus.json", "final_seed_minus.json", "trajectory_tail.csv"],
            reason="" if status == "ok" else "continuation did not reach every eta stage",
        )
        return {"task_id": row["task_id"], "status": status}
    except Exception as exc:
        _empty_outputs(outdir, meta, str(exc))
        return {"task_id": row["task_id"], "status": "failed", "reason": str(exc)}


def main() -> None:
    """Dispatch continuation rows with explicit per-stage integration horizons."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(VERSION2_ROOT / "outputs/chua_nonsmooth_fractional_memory_matrix/tasks/continuation_tasks.csv"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stage-transient", type=float, default=None, help="Per-eta transient; defaults to matrix t_burn.")
    parser.add_argument("--stage-keep", type=float, default=None, help="Per-eta observed segment; defaults to matrix t_burn.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root, manifest, rows = load_matrix(args.tasks)
    assert_unique_cache_keys(rows)
    transient = float(args.stage_transient if args.stage_transient is not None else manifest["contract"]["t_burn"])
    keep = float(args.stage_keep if args.stage_keep is not None else manifest["contract"]["t_burn"])
    if transient <= 0.0 or keep <= 0.0:
        raise ValueError("stage transient and keep horizons must be positive.")
    jobs = [
        {
            "root": str(root),
            "manifest": manifest,
            "row": row,
            "workers": args.workers,
            "stage_transient": transient,
            "stage_keep": keep,
            "force": args.force,
        }
        for row in rows
    ]
    for result in run_process_pool(run_one, jobs, workers=args.workers):
        print(f"{result['task_id']}: {result['status']}")


if __name__ == "__main__":
    main()
