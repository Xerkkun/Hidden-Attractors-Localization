#!/usr/bin/env python3
"""Compare basin cuts for Danca 2017 Chua and the best project candidate.

The Danca side uses Caputo ABM with full history and h=0.01.  The project side
uses the repository basin backend for the best available candidate, keeping the
project's EFORK finite-memory numerical contract explicit rather than mixing it
with the ABM replication.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
_CACHE_ROOT = ROOT / ".runtime_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg_cache"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

import chua_initial_cond as chua
from danca2017_chua_abm_replication import (
    DancaChuaConfig,
    caputo_abm_integrate,
    chua_rhs_factory,
    classify_trajectory,
    load_trajectory,
)
from equilibria_analysis import solve_equilibria
from extended_search_utils import json_safe, trajectory_ranges, write_csv
from parallel_policy import compile_c_target, force_single_openmp_thread_env, force_single_openmp_thread_current_process


CLASS_LABELS = {
    0: "equilibrium",
    1: "target_positive",
    2: "target_negative",
    3: "infinity",
    4: "unknown",
    5: "numerical_failure",
}
CLASS_COLORS = ["#111827", "#16a34a", "#2563eb", "#dc2626", "#9ca3af", "#f59e0b"]
CLASS_CMAP = ListedColormap(CLASS_COLORS)
CLASS_NORM = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), CLASS_CMAP.N)


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def project_best_seed() -> Tuple[str, np.ndarray, Path]:
    """Return the best project candidate seed used for basin comparison.

    Mathematical purpose:
        Select a concrete initial condition on the observed candidate attractor
        so basin cuts are not drawn with a silently projected seed.
    Equations:
        None; this is artifact selection from the existing continuation run.
    Parameters:
        Uses the biased Lur'e q=0.9998 summary and trajectory files.
    Output:
        Candidate id, seed vector, and source trajectory path.
    Validity warning:
        This selects the strongest current candidate, not a hidden-verified
        attractor.  Hiddenness remains controlled by equilibrium-neighborhood
        tests.
    """

    summary_path = ROOT / "outputs" / "lure_biased_multiparam_q09998" / "lure_biased_multiparam_summary.json"
    candidate_id = "lure_biased_q_0p99980_rank_0001"
    if summary_path.exists():
        data = read_json(summary_path)
        candidate_id = str(data.get("best_candidate_id", candidate_id))
    traj_path = (
        ROOT
        / "outputs"
        / "lure_biased_multiparam_q09998"
        / "trajectories"
        / f"{candidate_id}_{candidate_id}_phi_00_C1_continuation.csv"
    )
    if not traj_path.exists() and candidate_id == "lure_biased_q_0p99980_rank_0001":
        traj_path = (
            ROOT
            / "outputs"
            / "lure_biased_multiparam_q09998"
            / "trajectories"
            / "lure_biased_q_0p99980_rank_0001_lure_biased_q_0p99980_rank_0001_phi_00_C1_continuation.csv"
        )
    if not traj_path.exists():
        raise FileNotFoundError(f"No encontre trayectoria para {candidate_id}: {traj_path}")
    traj = load_trajectory(traj_path)
    seed = np.asarray(traj[0, 1:4], dtype=float)
    return candidate_id, seed, traj_path


def danca_seed_from_previous_replication() -> Tuple[str, np.ndarray, Path]:
    candidates = [
        path
        for path in (ROOT / "outputs").glob("danca2017_chua_abm_*/danca_reference_summary.json")
        if "smoke" not in str(path)
    ]
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No hay resumen Danca previo; ejecuta primero danca2017_chua_abm_replication.py.")
    path = candidates[-1]
    data = read_json(path)
    best = data.get("best_seed", {})
    x0 = np.asarray(best.get("x0", [np.nan, np.nan, np.nan]), dtype=float)
    if x0.shape != (3,) or not np.all(np.isfinite(x0)):
        raise ValueError(f"El resumen Danca no contiene x0 valido: {path}")
    return str(best.get("seed_id", "danca_reference_seed")), x0, path


def make_plan(outdir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    danca_seed_id, danca_seed, danca_source = danca_seed_from_previous_replication()
    project_id, project_seed, project_source = project_best_seed()
    xvals = np.linspace(float(args.xmin), float(args.xmax), int(args.grid))
    yvals = np.linspace(float(args.ymin), float(args.ymax), int(args.grid))
    plan_rows: List[Dict[str, Any]] = []
    for iy, y in enumerate(yvals):
        for ix, x in enumerate(xvals):
            plan_rows.append(
                {
                    "case_index": iy * len(xvals) + ix,
                    "ix": ix,
                    "iy": iy,
                    "x0": float(x),
                    "y0": float(y),
                    "z0": float(danca_seed[2]),
                }
            )
    write_csv(outdir / "danca_basin_plan.csv", plan_rows)
    cfg = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "grid": int(args.grid),
        "xlim": [float(args.xmin), float(args.xmax)],
        "ylim": [float(args.ymin), float(args.ymax)],
        "plane": "xy",
        "danca": {
            "seed_id": danca_seed_id,
            "seed": danca_seed.tolist(),
            "seed_source_summary": str(danca_source),
            "method": "Caputo ABM predictor-corrector with full history",
            "q": 0.9998,
            "h": float(args.danca_h),
            "t_final": float(args.danca_t_final),
            "transient": float(args.danca_transient),
            "z_plane": float(danca_seed[2]),
            "history_policy": "full_caputo_history_no_finite_memory_truncation",
        },
        "project": {
            "candidate_id": project_id,
            "seed": project_seed.tolist(),
            "seed_source_trajectory": str(project_source),
            "method": "C basin backend using EFORK finite-memory classifier",
            "q": 0.9998,
            "h": float(args.project_h),
            "Lm": float(args.project_lm),
            "t_final": float(args.project_t_final),
            "t_burn": float(args.project_t_burn),
            "z_plane": float(project_seed[2]),
        },
        "classification": {
            "equilibrium_tol": float(args.equilibrium_tol),
            "divergence_norm": float(args.divergence_norm),
            "r_bound": float(args.r_bound),
            "mean_x_gap": float(args.mean_x_gap),
        },
        "chunks": int(args.chunks),
    }
    write_json(outdir / "basin_comparison_config.json", cfg)
    return cfg


def danca_class_id(row: Dict[str, Any], cfg: Dict[str, Any]) -> int:
    if row.get("status") != "ok":
        return 5
    cls = str(row.get("final_class", ""))
    if cls == "infinity":
        return 3
    if cls.startswith("equilibrium_"):
        return 0
    if cls == "bounded_nontrivial":
        mean_x = float(row.get("mean_x_tail", 0.0))
        gap = float(cfg["classification"]["mean_x_gap"])
        if mean_x > gap:
            return 1
        if mean_x < -gap:
            return 2
        return 4
    return 4


def run_danca_chunk(outdir: Path, chunk_id: int, chunks: int) -> Path:
    force_single_openmp_thread_current_process()
    cfg = read_json(outdir / "basin_comparison_config.json")
    plan = read_csv_rows(outdir / "danca_basin_plan.csv")
    dcfg = DancaChuaConfig(
        q=float(cfg["danca"]["q"]),
        h=float(cfg["danca"]["h"]),
        t_final=float(cfg["danca"]["t_final"]),
        transient=float(cfg["danca"]["transient"]),
        equilibrium_tol=float(cfg["classification"]["equilibrium_tol"]),
        divergence_norm=float(cfg["classification"]["divergence_norm"]),
        store_stride=1,
    )
    eqs = solve_equilibria(dcfg.params())
    rhs = chua_rhs_factory(dcfg)
    rows: List[Dict[str, Any]] = []
    for item in plan:
        idx = int(item["case_index"])
        if idx % int(chunks) != int(chunk_id):
            continue
        x0 = np.array([float(item["x0"]), float(item["y0"]), float(item["z0"])], dtype=float)
        started = time.time()
        try:
            traj, meta = caputo_abm_integrate(
                rhs,
                x0,
                q=dcfg.q,
                h=dcfg.h,
                t_final=dcfg.t_final,
                divergence_norm=dcfg.divergence_norm,
                store_stride=max(1, int(math.ceil(dcfg.t_final / dcfg.h / 2000))),
            )
            cls = classify_trajectory(traj, dcfg, eqs)
            tail = traj[traj[:, 0] >= dcfg.transient, 1:4]
            mean_tail = np.mean(tail, axis=0) if tail.size else np.array([np.nan, np.nan, np.nan])
            row = {
                **item,
                **meta,
                "status": "ok",
                "elapsed_sec": time.time() - started,
                "final_class": cls.get("class", ""),
                "target_hit": cls.get("target_hit", False),
                "mean_x_tail": float(mean_tail[0]),
                "mean_y_tail": float(mean_tail[1]),
                "mean_z_tail": float(mean_tail[2]),
                **trajectory_ranges(traj),
            }
            row["class_id"] = danca_class_id(row, cfg)
            row["class_label"] = CLASS_LABELS[int(row["class_id"])]
        except Exception as exc:
            row = {
                **item,
                "status": "exception",
                "error": repr(exc),
                "elapsed_sec": time.time() - started,
                "final_class": "numerical_failure",
                "target_hit": False,
                "class_id": 5,
                "class_label": CLASS_LABELS[5],
            }
        rows.append(row)
        write_csv(outdir / f"danca_basin_chunk_{chunk_id:03d}.csv", rows)
    path = outdir / f"danca_basin_chunk_{chunk_id:03d}.csv"
    write_csv(path, rows)
    (outdir / f"danca_basin_chunk_{chunk_id:03d}.done").write_text(
        json.dumps({"chunk_id": int(chunk_id), "rows": len(rows), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}),
        encoding="utf-8",
    )
    return path


def load_project_basin_library() -> Any:
    native_dir = ROOT / ".runtime_native"
    native_dir.mkdir(exist_ok=True)
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    result = compile_c_target(
        ROOT / "chua_basin_lib.c",
        native_dir / f"chua_basin_compare{ext}",
        target_kind="shared",
        openmp=False,
    )
    lib = ctypes.CDLL(str(result.path.resolve()))
    lib.set_chua_params.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.set_chua_params.restype = None
    lib.set_chua_model.argtypes = [ctypes.c_int]
    lib.set_chua_model.restype = None
    lib.compute_basin_xy.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    ]
    lib.compute_basin_xy.restype = ctypes.c_int
    return lib


def run_project_basin(outdir: Path) -> Path:
    cfg = read_json(outdir / "basin_comparison_config.json")
    pcfg = cfg["project"]
    lib = load_project_basin_library()
    lib.set_chua_model(0)
    lib.set_chua_params(8.4562, 12.0732, 0.0052, -0.1768, -1.1468)
    grid = int(cfg["grid"])
    out = np.empty(grid * grid, dtype=np.int32)
    rc = lib.compute_basin_xy(
        grid,
        grid,
        float(cfg["xlim"][0]),
        float(cfg["xlim"][1]),
        float(cfg["ylim"][0]),
        float(cfg["ylim"][1]),
        float(pcfg["z_plane"]),
        float(pcfg["q"]),
        float(pcfg["h"]),
        float(pcfg["Lm"]),
        float(pcfg["t_final"]),
        float(pcfg["t_burn"]),
        float(cfg["classification"]["divergence_norm"]),
        float(cfg["classification"]["r_bound"]),
        float(cfg["classification"]["equilibrium_tol"]),
        150,
        float(cfg["classification"]["mean_x_gap"]),
        out,
    )
    if rc != 0:
        raise RuntimeError(f"compute_basin_xy returned {rc}")
    grid_arr = out.reshape((grid, grid))
    np.save(outdir / "project_best_basin_grid.npy", grid_arr)
    rows: List[Dict[str, Any]] = []
    xvals = np.linspace(float(cfg["xlim"][0]), float(cfg["xlim"][1]), grid)
    yvals = np.linspace(float(cfg["ylim"][0]), float(cfg["ylim"][1]), grid)
    for iy, y in enumerate(yvals):
        for ix, x in enumerate(xvals):
            cid = int(grid_arr[iy, ix])
            rows.append(
                {
                    "ix": ix,
                    "iy": iy,
                    "x0": float(x),
                    "y0": float(y),
                    "z0": float(pcfg["z_plane"]),
                    "class_id": cid,
                    "class_label": CLASS_LABELS.get(cid, f"class_{cid}"),
                }
            )
    write_csv(outdir / "project_best_basin_grid.csv", rows)
    plot_grid(
        grid_arr,
        cfg,
        outdir / "project_best_basin_xy.png",
        title=f"project {pcfg['candidate_id']} EFORK h={pcfg['h']}",
        seed=np.asarray(pcfg["seed"], dtype=float),
        z_plane=float(pcfg["z_plane"]),
    )
    (outdir / "project_best_basin_grid.done").write_text(
        json.dumps({"rows": len(rows), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}),
        encoding="utf-8",
    )
    return outdir / "project_best_basin_grid.csv"


def danca_grid_from_chunks(outdir: Path, cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(outdir.glob("danca_basin_chunk_*.csv")):
        rows.extend(read_csv_rows(path))
    grid = int(cfg["grid"])
    arr = np.full((grid, grid), 5, dtype=np.int32)
    for row in rows:
        arr[int(row["iy"]), int(row["ix"])] = int(row["class_id"])
    return arr, rows


def plot_grid(grid: np.ndarray, cfg: Dict[str, Any], path: Path, *, title: str, seed: np.ndarray, z_plane: float) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    ax.imshow(
        grid,
        origin="lower",
        extent=[cfg["xlim"][0], cfg["xlim"][1], cfg["ylim"][0], cfg["ylim"][1]],
        cmap=CLASS_CMAP,
        norm=CLASS_NORM,
        interpolation="nearest",
        aspect="auto",
    )
    ax.scatter([seed[0]], [seed[1]], marker="*", s=120, c="#fde047", edgecolors="black", linewidths=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.text(0.02, 0.98, f"z={z_plane:.5g}", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    handles = [
        Patch(facecolor=CLASS_COLORS[k], edgecolor="black", linewidth=0.3, label=CLASS_LABELS[k])
        for k in range(len(CLASS_COLORS))
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.88)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def aggregate(outdir: Path, *, wait: bool = False, poll_sec: float = 60.0) -> Path:
    cfg = read_json(outdir / "basin_comparison_config.json")
    chunks = int(cfg["chunks"])
    while wait:
        present = [outdir / f"danca_basin_chunk_{idx:03d}.done" for idx in range(chunks)]
        if all(path.exists() for path in present) and (outdir / "project_best_basin_grid.done").exists():
            break
        time.sleep(float(poll_sec))
    danca_grid, danca_rows = danca_grid_from_chunks(outdir, cfg)
    write_csv(outdir / "danca_basin_grid_raw.csv", danca_rows)
    np.save(outdir / "danca_basin_grid.npy", danca_grid)
    plot_grid(
        danca_grid,
        cfg,
        outdir / "danca_article_basin_xy_abm_h001.png",
        title=f"Danca ABM full history h={cfg['danca']['h']}",
        seed=np.asarray(cfg["danca"]["seed"], dtype=float),
        z_plane=float(cfg["danca"]["z_plane"]),
    )
    project_path = outdir / "project_best_basin_grid.csv"
    project_grid = np.load(outdir / "project_best_basin_grid.npy") if (outdir / "project_best_basin_grid.npy").exists() else None
    counts = {
        "danca": class_counts(danca_grid),
        "project": class_counts(project_grid) if project_grid is not None else {},
    }
    summary = {
        "status": "ok" if project_grid is not None and len(danca_rows) >= int(cfg["grid"]) ** 2 else "partial",
        "counts": counts,
        "outputs": {
            "danca_png": str(outdir / "danca_article_basin_xy_abm_h001.png"),
            "project_png": str(outdir / "project_best_basin_xy.png"),
            "danca_csv": str(outdir / "danca_basin_grid_raw.csv"),
            "project_csv": str(project_path),
        },
        "notes": [
            "Danca: ABM full Caputo history, no finite-memory truncation.",
            "Project: EFORK finite-memory C basin classifier, so the comparison is numerical-contract aware rather than method-identical.",
            "The xy cuts use each attractor seed's own z coordinate to avoid silent projection onto an inconsistent plane.",
        ],
    }
    write_json(outdir / "basin_comparison_summary.json", summary)
    return outdir / "basin_comparison_summary.json"


def class_counts(grid: np.ndarray | None) -> Dict[str, int]:
    if grid is None:
        return {}
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return {CLASS_LABELS.get(int(v), f"class_{int(v)}"): int(c) for v, c in zip(vals, counts)}


def launch(outdir: Path, args: argparse.Namespace) -> Path:
    cfg = make_plan(outdir, args)
    logs = outdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = force_single_openmp_thread_env(os.environ.copy())
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLCONFIGDIR"] = str(_CACHE_ROOT / "matplotlib")
    env["XDG_CACHE_HOME"] = str(_CACHE_ROOT / "xdg_cache")
    jobs: List[Dict[str, Any]] = []
    for chunk_id in range(int(args.chunks)):
        argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--job",
            "danca-chunk",
            "--output-dir",
            str(outdir),
            "--chunk-id",
            str(chunk_id),
            "--chunks",
            str(args.chunks),
        ]
        proc = subprocess.Popen(
            argv,
            cwd=str(ROOT),
            env=env,
            stdout=(logs / f"danca_chunk_{chunk_id:03d}.out").open("ab"),
            stderr=(logs / f"danca_chunk_{chunk_id:03d}.err").open("ab"),
            start_new_session=True,
            close_fds=True,
        )
        jobs.append({"name": f"danca_chunk_{chunk_id:03d}", "pid": proc.pid, "argv": argv})
    for job_name, extra in [
        ("project-basin", []),
        ("aggregate", ["--wait"]),
    ]:
        argv = [sys.executable, str(Path(__file__).resolve()), "--job", job_name, "--output-dir", str(outdir), *extra]
        proc = subprocess.Popen(
            argv,
            cwd=str(ROOT),
            env=env,
            stdout=(logs / f"{job_name}.out").open("ab"),
            stderr=(logs / f"{job_name}.err").open("ab"),
            start_new_session=True,
            close_fds=True,
        )
        jobs.append({"name": job_name, "pid": proc.pid, "argv": argv})
    manifest = {
        "status": "launched",
        "outdir": str(outdir),
        "config": cfg,
        "jobs": jobs,
    }
    write_json(outdir / "launch_manifest.json", manifest)
    print(json.dumps(json_safe(manifest), indent=2), flush=True)
    return outdir / "launch_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basin comparison: Danca ABM h=0.01 vs best project candidate.")
    parser.add_argument("--job", choices=["prepare", "launch", "danca-chunk", "project-basin", "aggregate"], default="launch")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / f"basin_compare_danca_project_h001_{timestamp()}"))
    parser.add_argument("--grid", type=int, default=21)
    parser.add_argument("--xmin", type=float, default=-9.0)
    parser.add_argument("--xmax", type=float, default=9.0)
    parser.add_argument("--ymin", type=float, default=-3.5)
    parser.add_argument("--ymax", type=float, default=3.5)
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--danca-h", type=float, default=0.01)
    parser.add_argument("--danca-t-final", type=float, default=500.0)
    parser.add_argument("--danca-transient", type=float, default=250.0)
    parser.add_argument("--project-h", type=float, default=0.01)
    parser.add_argument("--project-lm", type=float, default=40.0)
    parser.add_argument("--project-t-final", type=float, default=1500.0)
    parser.add_argument("--project-t-burn", type=float, default=100.0)
    parser.add_argument("--equilibrium-tol", type=float, default=0.01)
    parser.add_argument("--divergence-norm", type=float, default=120.0)
    parser.add_argument("--r-bound", type=float, default=30.0)
    parser.add_argument("--mean-x-gap", type=float, default=0.08)
    parser.add_argument("--wait", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    if args.job == "prepare":
        make_plan(outdir, args)
    elif args.job == "launch":
        launch(outdir, args)
    elif args.job == "danca-chunk":
        run_danca_chunk(outdir, int(args.chunk_id), int(args.chunks))
    elif args.job == "project-basin":
        run_project_basin(outdir)
    elif args.job == "aggregate":
        aggregate(outdir, wait=bool(args.wait))


if __name__ == "__main__":
    main()
