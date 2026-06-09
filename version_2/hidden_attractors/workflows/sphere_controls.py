"""Equilibrium-neighborhood controls and coarse robustness workflow.

This workflow probes whether trajectories initialized within balls around Chua
equilibria are classified into the target-attractor basin. A target hit from
an equilibrium neighborhood is evidence against the tested hiddenness claim.
The workflow also runs a coarse backend-classifier robustness pass from each
candidate continuation endpoint.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Tuple

import numpy as np

from ..basins import class_label, is_target_class
from ..candidates import load_final_candidate_records
from ..io import append_csv, read_csv_rows, read_json, timestamp, write_csv, write_json
from ..native.backends import BasinBackend
from ..parallel import force_single_openmp_thread_current_process, force_single_openmp_thread_env
from ..paths import OUTPUTS, PROJECT_ROOT
from ..reproducibility import (
    collect_lure_metadata,
    collect_run_metadata,
    collect_seed_metadata,
    write_run_metadata,
)
from ..systems import get_system
from .protocol import sample_uniform_ball


DEFAULT_SOURCE_DIR = PROJECT_ROOT / "validation" / "06_post_continuation_filter"

RAW_FIELDS = [
    "candidate_id",
    "candidate_rank",
    "q",
    "A",
    "sigma0",
    "omega",
    "rho_H",
    "equilibrium_id",
    "radius",
    "sample_id",
    "batch_100",
    "x0",
    "y0",
    "z0",
    "h",
    "memory_length",
    "t_final",
    "t_burn",
    "class_id",
    "class_label",
    "target_hit",
]

SUMMARY_FIELDS = [
    "candidate_id",
    "equilibrium_id",
    "radius",
    "cumulative_samples",
    "n_executed",
    "n_target_hits",
    "target_hit_fraction",
    "n_equilibrium",
    "n_target_positive",
    "n_target_negative",
    "n_infinity",
    "n_unknown",
    "n_numerical_failure",
]

ROBUST_RAW_FIELDS = [
    "candidate_id",
    "case_id",
    "q",
    "h",
    "memory_length",
    "t_final",
    "t_burn",
    "start_x",
    "start_y",
    "start_z",
    "class_id",
    "class_label",
    "target_hit",
]

ROBUST_SUMMARY_FIELDS = [
    "candidate_id",
    "executed_cases",
    "target_cases",
    "non_target_cases",
    "robust_attractor",
    "robustness_status",
    "notes",
]


def sphere_fields() -> list[str]:
    """Return stable CSV fields for raw sphere-control rows."""

    return RAW_FIELDS + ["route", "mu", "theta", "target_class_id", "target_class_label"]


def robust_raw_fields() -> list[str]:
    """Return stable CSV fields for raw robustness rows."""

    return ROBUST_RAW_FIELDS + ["route", "mu", "theta", "target_class_id", "target_class_label"]


def as_float(value: Any, default: float = float("nan")) -> float:
    """Parse a numeric artifact value, returning ``default`` on failure."""

    try:
        return float(value)
    except Exception:
        return default


def load_requested_candidates(source_dir: str | Path = DEFAULT_SOURCE_DIR) -> list[dict[str, Any]]:
    """Load exactly the three final candidates used by this project stage."""

    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(load_final_candidate_records(source_dir)):
        row = record.to_dict()
        row["candidate_rank"] = idx
        row["survivor_source"] = record.source
        row["survivor_final_class"] = ""
        rows.append(row)
    return rows


def load_equilibria_from_c() -> dict[str, np.ndarray]:
    """Load equilibria from the same C basin backend used for classification."""

    return BasinBackend.build(output_name="chua_basin_sphere_controls").equilibria()


def robustness_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return the classifier-based robustness contracts."""

    base = {
        "q": float(args.q),
        "divergence_norm": float(args.divergence_norm),
        "r_bound": float(args.r_bound),
        "equilibrium_tol": float(args.equilibrium_tol),
        "mean_x_gap": float(args.mean_x_gap),
        "cap_win": int(args.cap_win),
    }
    return [
        {**base, "case_id": "R0_baseline", "h": float(args.h), "memory_length": float(args.memory_length), "t_final": float(args.t_final), "t_burn": float(args.t_burn)},
        {**base, "case_id": "R1_longer", "h": float(args.h), "memory_length": float(args.memory_length), "t_final": float(args.robust_long_t_final), "t_burn": float(args.robust_long_t_burn)},
        {**base, "case_id": "R2_shorter_memory", "h": float(args.h), "memory_length": float(args.robust_short_memory), "t_final": float(args.t_final), "t_burn": float(args.t_burn)},
        {**base, "case_id": "R3_refined_h", "h": float(args.robust_refined_h), "memory_length": float(args.memory_length), "t_final": float(args.robust_refined_t_final), "t_burn": float(args.robust_refined_t_burn)},
    ]


def make_plan(outdir: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    """Create ball-neighborhood seeds and persist the run configuration.

    Mathematical purpose:
        Generate initial conditions inside balls around every equilibrium. These
        are the local probes used to test whether the candidate target basin
        intersects equilibrium neighborhoods under the tested numerical
        contract.
    """

    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    eqs = load_equilibria_from_c()
    radii = [float(x) for x in str(args.radii).split(",") if str(x).strip()]
    sphere_contract = {
        "q": float(args.q),
        "h": float(args.h),
        "memory_length": float(args.memory_length),
        "t_final": float(args.t_final),
        "t_burn": float(args.t_burn),
        "divergence_norm": float(args.divergence_norm),
        "r_bound": float(args.r_bound),
        "equilibrium_tol": float(args.equilibrium_tol),
        "mean_x_gap": float(args.mean_x_gap),
        "cap_win": int(args.cap_win),
    }
    candidates = load_requested_candidates(Path(args.source_dir).resolve())[: int(args.top_n)]
    for cand in candidates:
        cand["target_class_id"] = -1
        cand["target_class_label"] = "target_positive_or_negative"
    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, Any]] = []
    index = 0
    samples_by_radius = {
        str(radius): int(args.samples_per_radius) + idx * int(args.sample_growth_per_radius)
        for idx, radius in enumerate(radii)
    }
    for cand in candidates:
        for eq_id, center in eqs.items():
            for radius in radii:
                points = sample_uniform_ball(center, radius, samples_by_radius[str(radius)], rng)
                for sample_id, x0 in enumerate(points):
                    rows.append(
                        {
                            "case_index": index,
                            "candidate_id": cand["candidate_id"],
                            "candidate_rank": cand["candidate_rank"],
                            "route": cand["route"],
                            "q": cand["q"],
                            "mu": cand["mu"],
                            "theta": cand["theta"],
                            "A": cand["A"],
                            "sigma0": cand["sigma0"],
                            "omega": cand["omega"],
                            "rho_H": cand["rho_H"],
                            "target_class_id": cand["target_class_id"],
                            "target_class_label": cand["target_class_label"],
                            "equilibrium_id": eq_id,
                            "radius": radius,
                            "sample_id": sample_id,
                            "batch_100": int(sample_id // 100 + 1),
                            "x0": float(x0[0]),
                            "y0": float(x0[1]),
                            "z0": float(x0[2]),
                        }
                    )
                    index += 1
    cfg = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "ball_neighborhood_controls_plus_c_efork_robustness",
        "classification": "target_hit means class_id in {1, 2}, the EFORK nontrivial bounded target class used by the basin backend",
        "candidates": candidates,
        "equilibria": {k: v.tolist() for k, v in eqs.items()},
        "radii": radii,
        "samples_per_radius": int(args.samples_per_radius),
        "sample_growth_per_radius": int(args.sample_growth_per_radius),
        "samples_by_radius": samples_by_radius,
        "sampling_mode": "ball",
        "cumulative_batches": sorted(set(samples_by_radius.values())),
        "sphere_contract": sphere_contract,
        "robustness_cases": robustness_cases(args),
        "chunks": int(args.chunks),
        "random_seed": int(args.seed),
    }
    lure = get_system("chua-nonsmooth").lure
    seed = candidates[0] if candidates else None
    if seed is not None:
        seed = {**seed, "x0": seed.get("seed", seed.get("robust_start"))}
    run_metadata = collect_run_metadata(
        run_id=root.name,
        workflow="sphere_controls",
        system="fractional_nonsmooth_chua",
        q=float(args.q),
        h=float(args.h),
        t_final=float(args.t_final),
        t_burn=float(args.t_burn),
        memory_mode="finite_window",
        M=int(round(float(args.memory_length) / float(args.h))),
        memory_window_steps=int(round(float(args.memory_length) / float(args.h))),
        memory_window_time=float(args.memory_length),
        is_full_caputo=False,
        integrator_name="efork3",
        integrator_backend="native",
        caputo=True,
        parameters=get_system("chua-nonsmooth").parameters,
        lure=collect_lure_metadata(
            lure,
            transfer_convention="c^T(A - sI)^(-1)b",
            harmonic_condition="equilibrium-ball controls only",
        ),
        seed=collect_seed_metadata(seed, source=str(Path(args.source_dir).resolve())),
        random_seed=int(args.seed),
        random_seed_policy="fixed_reproducible",
        extra={"robustness_cases": cfg["robustness_cases"]},
    )
    cfg["run_metadata"] = write_run_metadata(root / "run_metadata.json", run_metadata)
    write_csv(root / "sphere_plan.csv", rows)
    write_json(root / "top3_sphere_robustness_config.json", cfg)
    return cfg


def _classify_point(backend: BasinBackend, x0: float, y0: float, z0: float, contract: dict[str, Any]) -> int:
    return backend.classify_point(
        [x0, y0, z0],
        q=float(contract["q"]),
        h=float(contract["h"]),
        Lm=float(contract["memory_length"]),
        t_final=float(contract["t_final"]),
        t_burn=float(contract["t_burn"]),
        divergence_norm=float(contract["divergence_norm"]),
        r_bound=float(contract["r_bound"]),
        equilibrium_tol=float(contract["equilibrium_tol"]),
        cap_win=int(contract["cap_win"]),
        mean_x_gap=float(contract["mean_x_gap"]),
    )


def run_sphere_chunk(outdir: str | Path, chunk_id: int, chunks: int) -> Path:
    """Classify one independent chunk of the sphere-control plan."""

    force_single_openmp_thread_current_process()
    root = Path(outdir)
    cfg = read_json(root / "top3_sphere_robustness_config.json")
    plan = read_csv_rows(root / "sphere_plan.csv")
    backend = BasinBackend.build(output_name="chua_basin_sphere_controls")
    path = root / f"sphere_raw_chunk_{chunk_id:03d}.csv"
    if path.exists():
        path.unlink()
    done_path = root / f"sphere_raw_chunk_{chunk_id:03d}.done"
    if done_path.exists():
        done_path.unlink()
    rows_done = 0
    for item in plan:
        idx = int(item["case_index"])
        if idx % int(chunks) != int(chunk_id):
            continue
        cid = _classify_point(backend, float(item["x0"]), float(item["y0"]), float(item["z0"]), cfg["sphere_contract"])
        target_class_id = int(float(item.get("target_class_id", -999)))
        row = {
            **{k: item.get(k, "") for k in sphere_fields()},
            "h": cfg["sphere_contract"]["h"],
            "memory_length": cfg["sphere_contract"]["memory_length"],
            "t_final": cfg["sphere_contract"]["t_final"],
            "t_burn": cfg["sphere_contract"]["t_burn"],
            "class_id": cid,
            "class_label": class_label(cid),
            "target_hit": bool(cid == target_class_id) if target_class_id in (1, 2) else is_target_class(cid),
        }
        append_csv(path, row, sphere_fields())
        rows_done += 1
        if rows_done % 50 == 0:
            print(f"sphere chunk {chunk_id}: {rows_done} rows", flush=True)
    write_json(done_path, {"chunk_id": int(chunk_id), "rows": rows_done, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return path


def run_robustness(outdir: str | Path) -> Path:
    """Run coarse target-class robustness tests from continuation endpoints."""

    force_single_openmp_thread_current_process()
    root = Path(outdir)
    cfg = read_json(root / "top3_sphere_robustness_config.json")
    backend = BasinBackend.build(output_name="chua_basin_sphere_controls")
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for cand in cfg["candidates"]:
        start = cand["robust_start"]
        for case in cfg["robustness_cases"]:
            cid = _classify_point(backend, float(start[0]), float(start[1]), float(start[2]), case)
            target_class_id = int(cand.get("target_class_id", -999))
            raw_rows.append(
                {
                    "candidate_id": cand["candidate_id"],
                    "case_id": case["case_id"],
                    "q": case["q"],
                    "h": case["h"],
                    "memory_length": case["memory_length"],
                    "t_final": case["t_final"],
                    "t_burn": case["t_burn"],
                    "start_x": float(start[0]),
                    "start_y": float(start[1]),
                    "start_z": float(start[2]),
                    "class_id": cid,
                    "class_label": class_label(cid),
                    "target_class_id": cand.get("target_class_id", ""),
                    "target_class_label": cand.get("target_class_label", ""),
                    "route": cand.get("route", ""),
                    "mu": cand.get("mu", ""),
                    "theta": cand.get("theta", ""),
                    "target_hit": bool(cid == target_class_id) if target_class_id in (1, 2) else is_target_class(cid),
                }
            )
        cand_rows = [r for r in raw_rows if r["candidate_id"] == cand["candidate_id"]]
        target_cases = sum(1 for r in cand_rows if bool(r["target_hit"]))
        robust = target_cases == len(cand_rows) and len(cand_rows) > 0
        summary_rows.append(
            {
                "candidate_id": cand["candidate_id"],
                "executed_cases": len(cand_rows),
                "target_cases": target_cases,
                "non_target_cases": len(cand_rows) - target_cases,
                "robust_attractor": robust,
                "robustness_status": "robust_under_tested_contracts" if robust else "not_robust_under_tested_contracts",
                "notes": "EFORK finite-memory robustness from continuation final state.",
            }
        )
    write_csv(root / "attractor_robustness_raw.csv", raw_rows, robust_raw_fields())
    write_csv(root / "attractor_robustness_summary.csv", summary_rows, ROBUST_SUMMARY_FIELDS)
    write_json(root / "attractor_robustness.done", {"rows": len(raw_rows), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return root / "attractor_robustness_summary.csv"


def aggregate(outdir: str | Path, *, wait: bool = False, poll_sec: float = 30.0) -> Path:
    """Aggregate all sphere chunks into summary and decision artifacts."""

    root = Path(outdir)
    cfg = read_json(root / "top3_sphere_robustness_config.json")
    chunks = int(cfg["chunks"])
    while wait:
        if all((root / f"sphere_raw_chunk_{idx:03d}.done").exists() for idx in range(chunks)) and (root / "attractor_robustness.done").exists():
            break
        time.sleep(float(poll_sec))
    raw_rows: list[dict[str, str]] = []
    for idx in range(chunks):
        raw_rows.extend(read_csv_rows(root / f"sphere_raw_chunk_{idx:03d}.csv"))
    write_csv(root / "sphere_raw.csv", raw_rows, sphere_fields())
    summary_rows: list[dict[str, Any]] = []
    grouped: dict[Tuple[str, str, float], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["candidate_id"], row["equilibrium_id"], as_float(row["radius"]))].append(row)
    for key, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: int(float(r["sample_id"])))
        for cumulative in cfg["cumulative_batches"]:
            sub = [r for r in rows_sorted if int(float(r["sample_id"])) < int(cumulative)]
            counts = Counter(str(r.get("class_label", "")) for r in sub)
            targets = sum(1 for r in sub if str(r.get("target_hit", "")).lower() == "true")
            n = len(sub)
            summary_rows.append(
                {
                    "candidate_id": key[0],
                    "equilibrium_id": key[1],
                    "radius": key[2],
                    "cumulative_samples": cumulative,
                    "n_executed": n,
                    "n_target_hits": targets,
                    "target_hit_fraction": float(targets / max(n, 1)),
                    "n_equilibrium": counts.get("equilibrium", 0),
                    "n_target_positive": counts.get("target_positive", 0),
                    "n_target_negative": counts.get("target_negative", 0),
                    "n_infinity": counts.get("infinity", 0),
                    "n_unknown": counts.get("unknown", 0),
                    "n_numerical_failure": counts.get("numerical_failure", 0),
                }
            )
    summary_rows.sort(key=lambda r: (r["candidate_id"], r["equilibrium_id"], float(r["radius"]), int(r["cumulative_samples"])))
    write_csv(root / "sphere_cumulative_summary.csv", summary_rows, SUMMARY_FIELDS)
    decisions = []
    for cand in cfg["candidates"]:
        rows = [
            r for r in summary_rows
            if r["candidate_id"] == cand["candidate_id"]
            and int(r["cumulative_samples"]) == int(cfg["samples_by_radius"][str(float(r["radius"]))])
        ]
        target_total = sum(int(r["n_target_hits"]) for r in rows)
        smallest = min([float(r["radius"]) for r in rows if int(r["n_target_hits"]) > 0], default=float("nan"))
        decisions.append(
            {
                "candidate_id": cand["candidate_id"],
                "total_sphere_trajectories": sum(int(r["n_executed"]) for r in rows),
                "total_neighborhood_trajectories": sum(int(r["n_executed"]) for r in rows),
                "total_target_hits": target_total,
                "smallest_radius_with_target_hit": "" if math.isnan(smallest) else smallest,
                "hiddenness_status": "self_excited_contact_detected" if target_total > 0 else "compatible_with_hiddenness_under_tested_radii",
                "notes": "Ball-neighborhood test uses EFORK target class; it is sufficient only within the declared tested radii.",
            }
        )
    write_csv(root / "sphere_decision.csv", decisions)
    summary = {
        "status": "ok" if len(raw_rows) == len(read_csv_rows(root / "sphere_plan.csv")) else "partial",
        "raw_rows": len(raw_rows),
        "planned_rows": len(read_csv_rows(root / "sphere_plan.csv")),
        "sphere_summary_csv": str(root / "sphere_cumulative_summary.csv"),
        "sphere_decision_csv": str(root / "sphere_decision.csv"),
        "robustness_summary_csv": str(root / "attractor_robustness_summary.csv"),
        "decisions": decisions,
    }
    write_json(root / "top3_sphere_robustness_summary.json", summary)
    return root / "top3_sphere_robustness_summary.json"


def launch(outdir: str | Path, args: argparse.Namespace) -> Path:
    """Launch independent OS processes for sphere chunks, robustness and aggregation."""

    root = Path(outdir)
    cfg = make_plan(root, args)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = force_single_openmp_thread_env(os.environ.copy())
    env["PYTHONUNBUFFERED"] = "1"
    script = Path(args.script_path).resolve()
    launched: list[dict[str, Any]] = []
    for idx in range(int(args.chunks)):
        cmd = [sys.executable, str(script), "--job", "sphere-chunk", "--output-dir", str(root), "--chunk-id", str(idx), "--chunks", str(args.chunks)]
        stdout = (logs / f"sphere_chunk_{idx:03d}.out").open("ab")
        stderr = (logs / f"sphere_chunk_{idx:03d}.err").open("ab")
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        launched.append({"job": f"sphere_chunk_{idx:03d}", "pid": proc.pid, "cmd": cmd})
    cmd = [sys.executable, str(script), "--job", "robustness", "--output-dir", str(root)]
    stdout = (logs / "robustness.out").open("ab")
    stderr = (logs / "robustness.err").open("ab")
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    launched.append({"job": "robustness", "pid": proc.pid, "cmd": cmd})
    cmd = [sys.executable, str(script), "--job", "aggregate", "--output-dir", str(root), "--wait"]
    stdout = (logs / "aggregate.out").open("ab")
    stderr = (logs / "aggregate.err").open("ab")
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    launched.append({"job": "aggregate", "pid": proc.pid, "cmd": cmd})
    manifest = {"output_dir": str(root), "chunks": int(cfg["chunks"]), "launched": launched}
    write_json(root / "launch_manifest.json", manifest)
    print(manifest, flush=True)
    return root / "launch_manifest.json"


def make_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the compatibility script."""

    parser = argparse.ArgumentParser(description="Equilibrium-ball hiddenness controls and classifier robustness.")
    parser.add_argument("--job", choices=["launch", "sphere-chunk", "robustness", "aggregate"], default="launch")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--radii", default="1e-5,3e-5,1e-4,3e-4,1e-3,1e-2")
    parser.add_argument("--samples-per-radius", type=int, default=100)
    parser.add_argument("--samples-growth-per-radius", dest="sample_growth_per_radius", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--q", type=float, default=0.9998)
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--memory-length", type=float, default=10.0)
    parser.add_argument("--t-final", type=float, default=1500.0)
    parser.add_argument("--t-burn", type=float, default=100.0)
    parser.add_argument("--robust-long-t-final", type=float, default=3000.0)
    parser.add_argument("--robust-long-t-burn", type=float, default=200.0)
    parser.add_argument("--robust-short-memory", type=float, default=5.0)
    parser.add_argument("--robust-refined-h", type=float, default=0.005)
    parser.add_argument("--robust-refined-t-final", type=float, default=1000.0)
    parser.add_argument("--robust-refined-t-burn", type=float, default=100.0)
    parser.add_argument("--divergence-norm", type=float, default=120.0)
    parser.add_argument("--r-bound", type=float, default=60.0)
    parser.add_argument("--equilibrium-tol", type=float, default=1e-3)
    parser.add_argument("--mean-x-gap", type=float, default=0.75)
    parser.add_argument("--cap-win", type=int, default=150)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--script-path", default=str(PROJECT_ROOT / "tools" / "cli" / "lure_top3_sphere_robustness.py"))
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    import warnings
    warnings.warn(
        "Deprecated: use 'hidden-attractors hiddenness sphere-controls ...'",
        DeprecationWarning,
        stacklevel=2
    )
    print("Deprecated: use 'hidden-attractors hiddenness sphere-controls ...'")
    args = make_parser().parse_args(argv)
    outdir = Path(args.output_dir).resolve() if args.output_dir else OUTPUTS / f"lure_top3_sphere_robustness_{timestamp()}"
    if args.job == "launch":
        launch(outdir, args)
    elif args.job == "sphere-chunk":
        run_sphere_chunk(outdir, int(args.chunk_id), int(args.chunks))
    elif args.job == "robustness":
        run_robustness(outdir)
    elif args.job == "aggregate":
        aggregate(outdir, wait=bool(args.wait))
