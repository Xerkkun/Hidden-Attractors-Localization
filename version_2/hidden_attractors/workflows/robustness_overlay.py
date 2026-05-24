"""Workflow for C/EFORK robustness-overlay trajectory comparisons.

The functions here keep the command-line script small while preserving a
reusable API for examples and future experiments.  The workflow compares
geometric and spectral features under changes in ``h``, finite-memory length
``Lm``, and total integration time.  It intentionally does not classify
hiddenness: hiddenness requires separate equilibrium-neighborhood basin tests.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..analysis.trajectory import RobustnessCase, default_robustness_cases, sample_rows, trajectory_metrics
from ..candidates import load_final_candidate_records
from ..io import read_csv_rows, read_json, safe_name, timestamp, write_csv, write_json
from ..native.backends import FractionalChuaBackend
from ..parallel import force_single_openmp_thread_current_process, force_single_openmp_thread_env
from ..paths import OUTPUTS, PROJECT_ROOT, RUNTIME_CACHE


DEFAULT_SOURCE_DIR = PROJECT_ROOT / "validation" / "04_candidates"

METRIC_FIELDS = [
    "candidate_id",
    "route",
    "case_id",
    "q",
    "h",
    "Lm",
    "t_final",
    "t_burn",
    "h_change_pct",
    "Lm_change_pct",
    "t_final_change_pct",
    "rows",
    "stored_rows",
    "analysis_start",
    "bounded",
    "diverged",
    "equilibrium_like",
    "noncollapsed_variance",
    "final_norm",
    "max_norm",
    "range_x",
    "range_y",
    "range_z",
    "var_x_tail",
    "var_y_tail",
    "var_z_tail",
    "fft_peak",
    "psd_entropy",
    "section_points",
    "range_relative_distance",
    "fft_relative_delta",
    "cloud_median_distance",
    "cloud_median_distance_norm",
    "section_median_distance",
    "section_median_distance_norm",
    "trajectory_csv",
]


def _cache_environment() -> None:
    """Set writable Matplotlib/font cache folders before importing pyplot."""

    (RUNTIME_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
    (RUNTIME_CACHE / "xdg_cache").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE / "xdg_cache"))


def _case_dicts(cases: Sequence[RobustnessCase]) -> list[dict[str, Any]]:
    baseline = cases[0]
    return [case.as_dict(baseline) for case in cases]


def _candidate_dicts(source_dir: str | Path, q: float) -> list[dict[str, Any]]:
    records = load_final_candidate_records(source_dir)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row = record.to_dict()
        row["candidate_rank"] = index
        row["q"] = float(q)
        rows.append(row)
    return rows


def make_config(
    outdir: str | Path,
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    q: float = 0.9998,
    divergence_norm: float = 120.0,
    equilibrium_tol: float = 1.0e-3,
    max_store_points: int = 6000,
    max_metric_points: int = 1000,
    max_section_points: int = 300,
    tail_fraction_start: float = 0.5,
) -> dict[str, Any]:
    """Create and persist the robustness-overlay workflow configuration.

    Mathematical purpose:
        Define the numerical contracts that will be compared against the same
        candidate initial states.  The baseline is the first case returned by
        :func:`default_robustness_cases`.

    Validity warning:
        A configuration here evaluates persistence of observed trajectory
        geometry.  It is not a proof of chaos and not a hiddenness test.
    """

    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    cases = _case_dicts(default_robustness_cases(q=float(q)))
    cfg = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "C EFORK trajectory overlay; no target or hiddenness classification",
        "params": {
            "alpha": 8.4562,
            "beta": 12.0732,
            "gamma": 0.0052,
            "m0": -0.1768,
            "m1": -1.1468,
        },
        "analysis": {
            "divergence_norm": float(divergence_norm),
            "equilibrium_tol": float(equilibrium_tol),
            "max_store_points": int(max_store_points),
            "max_metric_points": int(max_metric_points),
            "max_section_points": int(max_section_points),
            "tail_fraction_start": float(tail_fraction_start),
            "cloud_note": "Distances compare subsampled tail point clouds, normalized by baseline range norm.",
            "section_note": "Poincare section uses x=0 upward crossings after analysis_start.",
        },
        "candidates": _candidate_dicts(source_dir, q=float(q)),
        "cases": cases,
        "chunks": 3,
    }
    write_json(root / "robustness_overlay_config.json", cfg)
    return cfg


def save_sampled_trajectory(path: str | Path, traj: np.ndarray, max_points: int) -> int:
    """Store an evenly subsampled ``t,x,y,z`` trajectory CSV."""

    sampled = sample_rows(np.asarray(traj, dtype=float), int(max_points))
    rows = [{"t": float(r[0]), "x": float(r[1]), "y": float(r[2]), "z": float(r[3])} for r in sampled]
    write_csv(path, rows, ["t", "x", "y", "z"])
    return int(sampled.shape[0])


def load_trajectory_csv(path: str | Path) -> np.ndarray:
    """Load a sampled trajectory CSV written by :func:`save_sampled_trajectory`."""

    rows = read_csv_rows(path)
    return np.asarray([[float(r["t"]), float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)


def run_candidate(outdir: str | Path, candidate_index: int) -> Path:
    """Run all robustness cases for one candidate and write metrics CSV.

    Numerical model:
        Integration is delegated to the existing C EFORK backend.  Python does
        not implement the fractional stepping here.
    """

    force_single_openmp_thread_current_process()
    root = Path(outdir)
    cfg = read_json(root / "robustness_overlay_config.json")
    cand = cfg["candidates"][int(candidate_index)]
    backend = FractionalChuaBackend.build(output_name="chua_frac_overlay")
    cid_safe = safe_name(cand["candidate_id"])
    traj_dir = root / "trajectories" / cid_safe
    traj_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    ref_payload: dict[str, Any] | None = None
    for case in cfg["cases"]:
        started = time.time()
        traj = backend.integrate_efork3(
            cand["robust_start"],
            q=float(case["q"]),
            h=float(case["h"]),
            Lm=float(case["Lm"]),
            t_final=float(case["t_final"]),
        )
        traj_path = traj_dir / f"{case['case_id']}.csv"
        stored = save_sampled_trajectory(traj_path, traj, int(cfg["analysis"]["max_store_points"]))
        analysis_start = max(float(case["t_burn"]), float(cfg["analysis"]["tail_fraction_start"]) * float(case["t_final"]))
        metric, payload = trajectory_metrics(
            traj,
            h=float(case["h"]),
            t_start=analysis_start,
            divergence_norm=float(cfg["analysis"]["divergence_norm"]),
            equilibrium_tol=float(cfg["analysis"]["equilibrium_tol"]),
            max_section_points=int(cfg["analysis"]["max_section_points"]),
            max_cloud_points=int(cfg["analysis"]["max_metric_points"]),
            reference=ref_payload,
        )
        metric.update(
            {
                "candidate_id": cand["candidate_id"],
                "route": cand.get("route", ""),
                "case_id": case["case_id"],
                "q": case["q"],
                "h": case["h"],
                "Lm": case["Lm"],
                "t_final": case["t_final"],
                "t_burn": case["t_burn"],
                "h_change_pct": case["h_change_pct"],
                "Lm_change_pct": case["Lm_change_pct"],
                "t_final_change_pct": case["t_final_change_pct"],
                "rows": int(traj.shape[0]),
                "stored_rows": int(stored),
                "analysis_start": analysis_start,
                "trajectory_csv": str(traj_path),
                "elapsed_sec": time.time() - started,
            }
        )
        rows.append(metric)
        if case["case_id"] == "R0_base":
            ref_payload = payload
        print(f"{cand['candidate_id']} {case['case_id']} rows={traj.shape[0]} elapsed={metric['elapsed_sec']:.2f}", flush=True)
    path = root / f"metrics_{cid_safe}.csv"
    write_csv(path, rows, METRIC_FIELDS + ["elapsed_sec"])
    write_json(
        root / f"metrics_{cid_safe}.done",
        {"candidate_id": cand["candidate_id"], "rows": len(rows), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
    )
    return path


def plot_candidate(outdir: str | Path, cand: dict[str, Any], metric_rows: Sequence[dict[str, str]]) -> str:
    """Generate the overlay figure for one candidate from saved trajectory CSVs."""

    _cache_environment()
    from ..plotting.overlays import plot_trajectory_overlay

    trajectories: list[np.ndarray] = []
    labels: list[str] = []
    for row in metric_rows:
        X = load_trajectory_csv(row["trajectory_csv"])
        analysis_start = max(float(row["t_burn"]), 0.5 * float(row["t_final"]))
        trajectories.append(X[X[:, 0] >= analysis_start])
        labels.append(f"{row['case_id']} h={float(row['h']):g} Lm={float(row['Lm']):g} T={float(row['t_final']):g}")
    plot_dir = Path(outdir) / "plots"
    return plot_trajectory_overlay(
        trajectories,
        labels,
        title=cand["candidate_id"],
        output_path=plot_dir / f"overlay_{safe_name(cand['candidate_id'])}.png",
    )


def aggregate(outdir: str | Path, *, wait: bool = False) -> Path:
    """Aggregate candidate metric files and build overlay plots."""

    root = Path(outdir)
    cfg = read_json(root / "robustness_overlay_config.json")
    while wait:
        done = [(root / f"metrics_{safe_name(c['candidate_id'])}.done").exists() for c in cfg["candidates"]]
        if all(done):
            break
        time.sleep(30.0)
    all_rows: list[dict[str, str]] = []
    plots: list[str] = []
    for cand in cfg["candidates"]:
        path = root / f"metrics_{safe_name(cand['candidate_id'])}.csv"
        rows = read_csv_rows(path)
        all_rows.extend(rows)
        if rows:
            plots.append(plot_candidate(root, cand, rows))
    write_csv(root / "robustness_overlay_metrics.csv", all_rows, METRIC_FIELDS + ["elapsed_sec"])
    summary = {
        "status": "ok" if len(all_rows) == len(cfg["candidates"]) * len(cfg["cases"]) else "partial",
        "metric_rows": len(all_rows),
        "metrics_csv": str(root / "robustness_overlay_metrics.csv"),
        "plots": plots,
        "notes": [
            "No hiddenness or target classification is made here.",
            "Baseline is R0_base per candidate; relative distances compare each case against that baseline.",
            "Overlay plots use the post-transient analysis tail only.",
        ],
    }
    write_json(root / "robustness_overlay_summary.json", summary)
    return root / "robustness_overlay_summary.json"


def launch_independent_jobs(outdir: str | Path, args: argparse.Namespace) -> None:
    """Launch one independent OS process per candidate plus one aggregator."""

    root = Path(outdir)
    cfg = make_config(
        root,
        source_dir=args.source_dir,
        q=float(args.q),
        divergence_norm=float(args.divergence_norm),
        equilibrium_tol=float(args.equilibrium_tol),
        max_store_points=int(args.max_store_points),
        max_metric_points=int(args.max_metric_points),
        max_section_points=int(args.max_section_points),
        tail_fraction_start=float(args.tail_fraction_start),
    )
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = force_single_openmp_thread_env(os.environ.copy())
    env["PYTHONUNBUFFERED"] = "1"
    launched: list[dict[str, Any]] = []
    script = Path(args.script_path).resolve()
    for idx, cand in enumerate(cfg["candidates"]):
        cmd = [sys.executable, str(script), "--job", "candidate", "--output-dir", str(root), "--candidate-index", str(idx)]
        stdout = (logs / f"candidate_{idx}_{safe_name(cand['candidate_id'])}.out").open("ab")
        stderr = (logs / f"candidate_{idx}_{safe_name(cand['candidate_id'])}.err").open("ab")
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        launched.append({"job": "candidate", "candidate_id": cand["candidate_id"], "pid": proc.pid, "cmd": cmd})
    cmd = [sys.executable, str(script), "--job", "aggregate", "--output-dir", str(root), "--wait"]
    stdout = (logs / "aggregate.out").open("ab")
    stderr = (logs / "aggregate.err").open("ab")
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    launched.append({"job": "aggregate", "pid": proc.pid, "cmd": cmd})
    manifest = {"output_dir": str(root), "launched": launched}
    write_json(root / "launch_manifest.json", manifest)
    print(manifest, flush=True)


def make_parser() -> argparse.ArgumentParser:
    """Return the CLI parser used by the thin compatibility script."""

    parser = argparse.ArgumentParser(description="Overlay C/EFORK trajectories under h/Lm/t_final changes.")
    parser.add_argument("--job", choices=["launch", "candidate", "aggregate"], default="launch")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--q", type=float, default=0.9998)
    parser.add_argument("--divergence-norm", type=float, default=120.0)
    parser.add_argument("--equilibrium-tol", type=float, default=1e-3)
    parser.add_argument("--max-store-points", type=int, default=6000)
    parser.add_argument("--max-metric-points", type=int, default=1000)
    parser.add_argument("--max-section-points", type=int, default=300)
    parser.add_argument("--tail-fraction-start", type=float, default=0.5)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--script-path", default=str(PROJECT_ROOT / "tools" / "cli" / "robustness_overlay_c_trajectories.py"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint."""

    args = make_parser().parse_args(argv)
    outdir = Path(args.output_dir).resolve() if args.output_dir else OUTPUTS / f"robustness_overlay_c_trajectories_{timestamp()}"
    if args.job == "launch":
        launch_independent_jobs(outdir, args)
    elif args.job == "candidate":
        run_candidate(outdir, int(args.candidate_index))
    elif args.job == "aggregate":
        aggregate(outdir, wait=bool(args.wait))
