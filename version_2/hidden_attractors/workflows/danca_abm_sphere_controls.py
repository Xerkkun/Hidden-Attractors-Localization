"""Danca ABM full-history sphere controls plus strict unknown refinement.

This workflow reproduces the sphere-neighborhood control geometry used for the
project EFORK candidate, but integrates every point with the Danca
Caputo Adams-Bashforth-Moulton solver using full memory.  It then optionally
launches the strict target-reference refinement only for rows that remain
``unknown`` under the coarse Danca sphere classifier.

Mathematical warning:
    The coarse labels are finite-time basin diagnostics.  A target hit from an
    equilibrium sphere is evidence against hiddenness under the tested
    numerical contract; absence of such hits is compatibility evidence, not a
    proof.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..basins import class_label
from ..io import append_csv, read_csv_rows, read_json, write_csv, write_json
from ..parallel import force_single_openmp_thread_current_process, force_single_openmp_thread_env
from ..paths import PROJECT_ROOT


LEGACY_ROOT = PROJECT_ROOT / "tools" / "legacy"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from danca2017_chua_abm_replication import (  # noqa: E402
    DancaChuaConfig,
    caputo_abm_integrate,
    chua_rhs_factory,
    classify_trajectory,
)
from equilibria_analysis import solve_equilibria  # noqa: E402


ROOT_OUTPUTS = PROJECT_ROOT.parent / "outputs"
DEFAULT_DANCA_SOURCE = ROOT_OUTPUTS / "danca2017_chua_abm_20260515_182354"

RAW_FIELDS = [
    "case_index",
    "candidate_id",
    "q",
    "equilibrium_id",
    "radius",
    "sample_id",
    "batch_100",
    "x0",
    "y0",
    "z0",
    "h",
    "t_final",
    "t_burn",
    "class_id",
    "class_label",
    "target_hit",
    "danca_class",
    "bounded",
    "final_norm",
    "closest_equilibrium",
    "closest_equilibrium_distance",
    "range_norm_tail",
    "range_x_tail",
    "range_y_tail",
    "range_z_tail",
    "mean_x_tail",
    "mean_y_tail",
    "mean_z_tail",
    "elapsed_sec",
    "status",
]

SUMMARY_FIELDS = [
    "candidate_id",
    "equilibrium_id",
    "radius",
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


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _unit_direction(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vec / norm


def _danca_config(source_dir: Path) -> DancaChuaConfig:
    raw = read_json(source_dir / "run_config.json")
    return DancaChuaConfig(
        q=float(raw["q"]),
        h=float(raw["h"]),
        t_final=float(raw["t_final"]),
        transient=float(raw["transient"]),
        alpha=float(raw["alpha"]),
        beta=float(raw["beta"]),
        gamma_chua=float(raw["gamma_chua"]),
        m0=float(raw["m0"]),
        m1=float(raw["m1"]),
        delta=float(raw["delta"]),
        equilibrium_tol=float(raw["equilibrium_tol"]),
        divergence_norm=float(raw["divergence_norm"]),
        nontrivial_range_tol=float(raw["nontrivial_range_tol"]),
        local_samples_per_unstable_eq=int(raw["local_samples_per_unstable_eq"]),
        figure_local_trajectories=int(raw["figure_local_trajectories"]),
        rng_seed=int(raw["rng_seed"]),
        store_stride=int(raw.get("store_stride", 1)),
    )


def _coarse_danca_class(row: dict[str, Any], *, mean_x_gap: float) -> int:
    klass = str(row.get("class", ""))
    if str(row.get("status", "")) != "ok":
        return 5
    if klass == "infinity":
        return 3
    if klass.startswith("equilibrium_"):
        return 0
    if klass == "bounded_nontrivial":
        mean_x = _float(row.get("mean_x_tail", 0.0), 0.0)
        if mean_x > float(mean_x_gap):
            return 1
        if mean_x < -float(mean_x_gap):
            return 2
        return 4
    return 4


def make_plan(outdir: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.danca_source_dir).resolve()
    dcfg = _danca_config(source_dir)
    eqs = solve_equilibria(dcfg.params())
    radii = [float(item) for item in str(args.radii).split(",") if item.strip()]
    eq_names = [item for item in str(args.equilibria).split(",") if item.strip()]
    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, Any]] = []
    case_index = 0
    for eq_id in eq_names:
        center = np.asarray(eqs[eq_id], dtype=float)
        for radius in radii:
            for sample_id in range(int(args.samples_per_radius)):
                x0 = center + float(radius) * _unit_direction(rng)
                rows.append(
                    {
                        "case_index": case_index,
                        "candidate_id": "danca2017_reference",
                        "q": dcfg.q,
                        "equilibrium_id": eq_id,
                        "radius": float(radius),
                        "sample_id": sample_id,
                        "batch_100": int(sample_id // 100 + 1),
                        "x0": float(x0[0]),
                        "y0": float(x0[1]),
                        "z0": float(x0[2]),
                    }
                )
                case_index += 1
    write_csv(root / "danca_sphere_plan.csv", rows)
    for name in ("run_config.json", "danca_reference_summary.json"):
        src = source_dir / name
        if src.exists():
            write_json(root / name, read_json(src))
    cfg = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "danca_abm_full_memory_sphere_controls",
        "danca_source_dir": str(source_dir),
        "candidate_id": "danca2017_reference",
        "equilibria": {key: val.tolist() for key, val in eqs.items()},
        "tested_equilibria": eq_names,
        "radii": radii,
        "samples_per_radius": int(args.samples_per_radius),
        "chunks": int(args.chunks),
        "random_seed": int(args.seed),
        "classification": {
            "mean_x_gap": float(args.mean_x_gap),
            "target_hit": "class_id in {1,2}; bounded_nontrivial with |mean_x_tail| <= mean_x_gap is unknown",
        },
        "contract": {
            "q": dcfg.q,
            "h": dcfg.h,
            "t_final": dcfg.t_final,
            "t_burn": dcfg.transient,
            "history_policy": "full_caputo_history_no_finite_memory_truncation",
            "equilibrium_tol": dcfg.equilibrium_tol,
            "divergence_norm": dcfg.divergence_norm,
            "nontrivial_range_tol": dcfg.nontrivial_range_tol,
            "store_stride": dcfg.store_stride,
        },
        "planned_rows": len(rows),
        "chain_strict_unknown_refinement": bool(args.chain_strict_unknown_refinement),
        "strict_refine_chunks": int(args.strict_refine_chunks),
    }
    write_json(root / "danca_abm_sphere_config.json", cfg)
    return cfg


def run_chunk(outdir: str | Path, chunk_id: int, chunks: int) -> Path:
    force_single_openmp_thread_current_process()
    root = Path(outdir)
    cfg = read_json(root / "danca_abm_sphere_config.json")
    source_dir = Path(cfg["danca_source_dir"])
    dcfg = _danca_config(source_dir)
    eqs = solve_equilibria(dcfg.params())
    rhs = chua_rhs_factory(dcfg)
    plan = read_csv_rows(root / "danca_sphere_plan.csv")
    path = root / f"danca_sphere_raw_chunk_{int(chunk_id):03d}.csv"
    if path.exists():
        path.unlink()
    done = root / f"danca_sphere_raw_chunk_{int(chunk_id):03d}.done"
    if done.exists():
        done.unlink()
    rows = 0
    for item in plan:
        idx = int(float(item["case_index"]))
        if idx % int(chunks) != int(chunk_id):
            continue
        seed = np.array([_float(item["x0"]), _float(item["y0"]), _float(item["z0"])], dtype=float)
        started = time.time()
        try:
            traj, meta = caputo_abm_integrate(
                rhs,
                seed,
                q=dcfg.q,
                h=dcfg.h,
                t_final=dcfg.t_final,
                divergence_norm=dcfg.divergence_norm,
                store_stride=dcfg.store_stride,
            )
            cls = classify_trajectory(traj, dcfg, eqs)
            tail = traj[traj[:, 0] >= dcfg.transient, 1:4]
            mean_tail = np.mean(tail, axis=0) if tail.size else np.array([np.nan, np.nan, np.nan])
            row = {
                **item,
                **meta,
                **cls,
                "danca_class": cls.get("class", ""),
                "mean_x_tail": float(mean_tail[0]),
                "mean_y_tail": float(mean_tail[1]),
                "mean_z_tail": float(mean_tail[2]),
                "status": "ok",
                "elapsed_sec": time.time() - started,
                "h": dcfg.h,
                "t_final": dcfg.t_final,
                "t_burn": dcfg.transient,
            }
            cid = _coarse_danca_class(row, mean_x_gap=float(cfg["classification"]["mean_x_gap"]))
            row["class_id"] = cid
            row["class_label"] = class_label(cid)
            row["target_hit"] = cid in (1, 2)
        except Exception as exc:
            row = {
                **item,
                "status": "exception",
                "error": repr(exc),
                "elapsed_sec": time.time() - started,
                "class_id": 5,
                "class_label": class_label(5),
                "target_hit": False,
                "danca_class": "numerical_failure",
                "h": dcfg.h,
                "t_final": dcfg.t_final,
                "t_burn": dcfg.transient,
            }
        append_csv(path, row, RAW_FIELDS)
        rows += 1
        if rows % 25 == 0:
            print(f"danca ABM sphere chunk {chunk_id}: {rows} rows", flush=True)
    write_json(done, {"chunk_id": int(chunk_id), "rows": rows, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return path


def aggregate(outdir: str | Path, *, wait: bool = False, poll_sec: float = 60.0) -> Path:
    root = Path(outdir)
    cfg = read_json(root / "danca_abm_sphere_config.json")
    chunks = int(cfg["chunks"])
    while wait:
        if all((root / f"danca_sphere_raw_chunk_{idx:03d}.done").exists() for idx in range(chunks)):
            break
        time.sleep(float(poll_sec))
    rows: list[dict[str, str]] = []
    for idx in range(chunks):
        rows.extend(read_csv_rows(root / f"danca_sphere_raw_chunk_{idx:03d}.csv"))
    rows.sort(key=lambda row: int(float(row.get("case_index", 0))))
    write_csv(root / "danca_sphere_raw.csv", rows, RAW_FIELDS)

    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("equilibrium_id", "")), _float(row.get("radius")))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (eq_id, radius), sub in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        counts = Counter(row.get("class_label", "") for row in sub)
        targets = counts.get("target_positive", 0) + counts.get("target_negative", 0)
        n = len(sub)
        summary_rows.append(
            {
                "candidate_id": cfg["candidate_id"],
                "equilibrium_id": eq_id,
                "radius": radius,
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
    write_csv(root / "danca_sphere_cumulative_summary.csv", summary_rows, SUMMARY_FIELDS)
    total_targets = sum(int(row["n_target_hits"]) for row in summary_rows)
    total_unknown = sum(int(row["n_unknown"]) for row in summary_rows)
    decision = {
        "candidate_id": cfg["candidate_id"],
        "total_sphere_trajectories": len(rows),
        "planned_rows": int(cfg["planned_rows"]),
        "total_target_hits": total_targets,
        "total_unknown": total_unknown,
        "hiddenness_status": "not_supported_by_danca_abm_sphere_test" if total_targets > 0 else "compatible_with_hiddenness_under_tested_spheres",
        "notes": "Danca ABM full-history sphere test; unknown rows should be passed to strict target-reference refinement.",
    }
    write_json(root / "danca_sphere_decision.json", decision)
    summary = {
        "status": "ok" if len(rows) == int(cfg["planned_rows"]) else "partial",
        "raw_rows": len(rows),
        "planned_rows": int(cfg["planned_rows"]),
        "summary_csv": str(root / "danca_sphere_cumulative_summary.csv"),
        "decision_json": str(root / "danca_sphere_decision.json"),
        "decision": decision,
    }
    write_json(root / "danca_abm_sphere_summary.json", summary)
    return root / "danca_abm_sphere_summary.json"


def refine_after_aggregate(outdir: str | Path, *, wait: bool = True, poll_sec: float = 60.0) -> Path:
    root = Path(outdir)
    cfg = read_json(root / "danca_abm_sphere_config.json")
    while wait and not (root / "danca_abm_sphere_summary.json").exists():
        time.sleep(float(poll_sec))
    if not bool(cfg.get("chain_strict_unknown_refinement", True)):
        write_json(root / "strict_unknown_refinement_launch.json", {"status": "skipped"})
        return root / "strict_unknown_refinement_launch.json"
    refined_dir = root / "strict_unknown_refinement"
    script = PROJECT_ROOT / "tools" / "cli" / "strict_target_refinement.py"
    cmd = [
        sys.executable,
        str(script),
        "--job",
        "launch",
        "--mode",
        "sphere-danca",
        "--source-dir",
        str(root),
        "--source-csv",
        str(root / "danca_sphere_raw.csv"),
        "--source-labels",
        "unknown",
        "--output-dir",
        str(refined_dir),
        "--chunks",
        str(int(cfg.get("strict_refine_chunks", 4))),
    ]
    logs = root / "logs"
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=force_single_openmp_thread_env(os.environ.copy()),
        stdout=(logs / "strict_unknown_refinement_launch.out").open("ab"),
        stderr=(logs / "strict_unknown_refinement_launch.err").open("ab"),
        start_new_session=True,
        close_fds=True,
    )
    manifest = {"status": "launched_strict_unknown_refinement", "pid": proc.pid, "cmd": cmd, "output_dir": str(refined_dir)}
    write_json(root / "strict_unknown_refinement_launch.json", manifest)
    return root / "strict_unknown_refinement_launch.json"


def launch(outdir: str | Path, args: argparse.Namespace) -> Path:
    root = Path(outdir)
    cfg = make_plan(root, args)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = force_single_openmp_thread_env(os.environ.copy())
    env["PYTHONUNBUFFERED"] = "1"
    script = Path(args.script_path).resolve()
    launched: list[dict[str, Any]] = []
    for idx in range(int(args.chunks)):
        cmd = [sys.executable, str(script), "--job", "chunk", "--output-dir", str(root), "--chunk-id", str(idx), "--chunks", str(args.chunks)]
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=(logs / f"chunk_{idx:03d}.out").open("ab"),
            stderr=(logs / f"chunk_{idx:03d}.err").open("ab"),
            start_new_session=True,
            close_fds=True,
        )
        launched.append({"job": f"chunk_{idx:03d}", "pid": proc.pid, "cmd": cmd})
    for job in ("aggregate", "refine-after-aggregate"):
        cmd = [sys.executable, str(script), "--job", job, "--output-dir", str(root), "--wait"]
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=(logs / f"{job}.out").open("ab"),
            stderr=(logs / f"{job}.err").open("ab"),
            start_new_session=True,
            close_fds=True,
        )
        launched.append({"job": job, "pid": proc.pid, "cmd": cmd})
    manifest = {"output_dir": str(root), "planned_rows": cfg["planned_rows"], "chunks": int(cfg["chunks"]), "launched": launched}
    write_json(root / "launch_manifest.json", manifest)
    print(manifest, flush=True)
    return root / "launch_manifest.json"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Danca ABM full-history sphere controls and strict unknown refinement.")
    parser.add_argument("--job", choices=["launch", "chunk", "aggregate", "refine-after-aggregate"], default="launch")
    parser.add_argument("--output-dir", default=str(ROOT_OUTPUTS / "danca_abm_sphere_controls_20260520"))
    parser.add_argument("--danca-source-dir", default=str(DEFAULT_DANCA_SOURCE))
    parser.add_argument("--equilibria", default="E0,E+,E-")
    parser.add_argument("--radii", default="1e-5,3e-5,1e-4,3e-4,1e-3")
    parser.add_argument("--samples-per-radius", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--mean-x-gap", type=float, default=0.75)
    parser.add_argument("--chain-strict-unknown-refinement", action="store_true", default=True)
    parser.add_argument("--no-chain-strict-unknown-refinement", dest="chain_strict_unknown_refinement", action="store_false")
    parser.add_argument("--strict-refine-chunks", type=int, default=4)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--script-path", default=str(PROJECT_ROOT / "tools" / "cli" / "danca_abm_sphere_controls.py"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    outdir = Path(args.output_dir).resolve()
    if args.job == "launch":
        launch(outdir, args)
    elif args.job == "chunk":
        run_chunk(outdir, int(args.chunk_id), int(args.chunks))
    elif args.job == "aggregate":
        aggregate(outdir, wait=bool(args.wait))
    elif args.job == "refine-after-aggregate":
        refine_after_aggregate(outdir, wait=bool(args.wait))


if __name__ == "__main__":
    main()
