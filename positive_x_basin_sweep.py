#!/usr/bin/env python3
"""High-resolution positive-x basin sweep for Danca and project candidates.

Mathematical purpose:
    Focus the basin cut on the positive equilibrium neighborhood ``E+`` and on
    positive ``x`` initial conditions.  This directly supports the
    self-excited/hidden-attractor question: do initial conditions near an
    equilibrium fall into the candidate-attractor basin?

Numerical contracts:
    Danca side uses the full-history Caputo ABM replication.
    Project side uses the EFORK finite-memory C basin classifier with explicit
    ``Lm``.  A chained refinement job can subsequently reclassify project
    ``unknown`` cells by trajectory geometry against positive/negative target
    references.

Validity warning:
    A basin plot is numerical evidence under the recorded contract.  It is not
    a theorem of hiddenness.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
from matplotlib.patches import Circle, Patch
import numpy as np

from chua_basin_comparison_h001 import danca_seed_from_previous_replication, project_best_seed
from danca2017_chua_abm_replication import DancaChuaConfig, caputo_abm_integrate, chua_rhs_factory, classify_trajectory, load_trajectory
from equilibria_analysis import solve_equilibria
from extended_search_utils import json_safe, trajectory_ranges
from hidden_attractors.basins import CLASS_LABELS, class_label
from hidden_attractors.io import append_csv, read_csv_rows, timestamp, write_csv
from hidden_attractors.models import chua_piecewise_parameters, equilibria_piecewise
from hidden_attractors.native import BasinBackend
from parallel_policy import force_single_openmp_thread_current_process, force_single_openmp_thread_env


CLASS_COLORS = {
    0: "#111827",
    1: "#16a34a",
    2: "#2563eb",
    3: "#dc2626",
    4: "#9ca3af",
    5: "#f59e0b",
}
CLASS_CMAP = ListedColormap([CLASS_COLORS[k] for k in range(6)])
CLASS_NORM = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), CLASS_CMAP.N)

DANCA_FIELDS = [
    "case_index",
    "ix",
    "iy",
    "x0",
    "y0",
    "z0",
    "status",
    "elapsed_sec",
    "final_class",
    "target_hit",
    "mean_x_tail",
    "mean_y_tail",
    "mean_z_tail",
    "range_x",
    "range_y",
    "range_z",
    "class_id",
    "class_label",
]

PROJECT_FIELDS = [
    "case_index",
    "ix",
    "iy",
    "x0",
    "y0",
    "z0",
    "class_id",
    "class_label",
]


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _danca_class_id(row: Dict[str, Any], cfg: Dict[str, Any]) -> int:
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


def _equilibria_for_plot() -> dict[str, np.ndarray]:
    return equilibria_piecewise(chua_piecewise_parameters())


def _eq_slug(name: str) -> str:
    return {"E+": "Eplus", "E-": "Eminus"}.get(name, name)


def _plot_annotations(ax: plt.Axes, cfg: Dict[str, Any], seed: np.ndarray, seed_label: str) -> None:
    xlim = cfg["xlim"]
    ylim = cfg["ylim"]
    seed_visible = float(xlim[0]) <= seed[0] <= float(xlim[1]) and float(ylim[0]) <= seed[1] <= float(ylim[1])
    if seed_visible:
        ax.scatter([seed[0]], [seed[1]], marker="*", s=145, c="#fde047", edgecolors="black", linewidths=0.8, zorder=6, label=seed_label)
        ax.annotate(seed_label, xy=(seed[0], seed[1]), xytext=(8, 9), textcoords="offset points", fontsize=8, color="black")
    else:
        ax.text(0.02, 0.90, f"star/seed outside this local view: {seed_label}", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    for name, eq in _equilibria_for_plot().items():
        if float(xlim[0]) <= eq[0] <= float(xlim[1]) and float(ylim[0]) <= eq[1] <= float(ylim[1]):
            ax.scatter([eq[0]], [eq[1]], marker="o", s=56, facecolor="white", edgecolor="black", linewidths=1.0, zorder=7)
            ax.annotate(f"{name} equilibrium", xy=(eq[0], eq[1]), xytext=(8, -13), textcoords="offset points", fontsize=8, color="black")
            ax.add_patch(Circle((eq[0], eq[1]), 0.25, fill=False, ls="--", lw=1.0, ec="black", alpha=0.75))
            ax.text(eq[0] + 0.28, eq[1] + 0.22, f"visual neighborhood of {name}", fontsize=7, color="black")


def plot_grid(
    arr: np.ndarray,
    cfg: Dict[str, Any],
    path: Path,
    *,
    title: str,
    seed: np.ndarray,
    seed_label: str,
    z_plane: float,
    zoom: bool = False,
    zoom_equilibrium: str = "E+",
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.7))
    ax.imshow(
        arr,
        origin="lower",
        extent=[cfg["xlim"][0], cfg["xlim"][1], cfg["ylim"][0], cfg["ylim"][1]],
        cmap=CLASS_CMAP,
        norm=CLASS_NORM,
        interpolation="nearest",
        aspect="auto",
    )
    _plot_annotations(ax, cfg, seed, seed_label)
    ax.set_xlabel("x0")
    ax.set_ylabel("y0")
    ax.set_title(title)
    ax.text(0.02, 0.98, f"z0={z_plane:.5g}", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    if zoom:
        ep = _equilibria_for_plot()[zoom_equilibrium]
        ax.set_xlim(max(float(cfg["xlim"][0]), ep[0] - 0.9), min(float(cfg["xlim"][1]), ep[0] + 0.9))
        ax.set_ylim(max(float(cfg["ylim"][0]), ep[1] - 0.55), min(float(cfg["ylim"][1]), ep[1] + 0.55))
    handles = [
        Patch(facecolor=CLASS_COLORS[k], edgecolor="black", linewidth=0.3, label=CLASS_LABELS[k])
        for k in range(6)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def visible_equilibria(cfg: Dict[str, Any]) -> List[str]:
    xlim = cfg["xlim"]
    ylim = cfg["ylim"]
    out: List[str] = []
    for name, eq in _equilibria_for_plot().items():
        if float(xlim[0]) <= eq[0] <= float(xlim[1]) and float(ylim[0]) <= eq[1] <= float(ylim[1]):
            out.append(name)
    return out


def plot_target_explanation(outdir: Path, cfg: Dict[str, Any]) -> str:
    """Plot a visual explanation of target_positive and target_negative."""

    traj_path = Path(cfg["project"]["seed_source_trajectory"])
    traj = load_trajectory(traj_path)
    tail = traj[traj[:, 0] >= max(float(cfg["project"]["t_burn"]), 0.5 * float(cfg["project"]["t_final"]))]
    if tail.shape[0] < 4:
        tail = traj
    pos = tail[:, 1:4]
    neg = -pos
    pos = pos[np.linspace(0, pos.shape[0] - 1, min(1600, pos.shape[0])).astype(int)]
    neg = neg[np.linspace(0, neg.shape[0] - 1, min(1600, neg.shape[0])).astype(int)]
    eqs = _equilibria_for_plot()
    fig = plt.figure(figsize=(12.0, 5.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], lw=0.7, color=CLASS_COLORS[1], label="target_positive")
    ax.plot(neg[:, 0], neg[:, 1], neg[:, 2], lw=0.7, color=CLASS_COLORS[2], label="target_negative")
    for name, eq in eqs.items():
        ax.scatter([eq[0]], [eq[1]], [eq[2]], color="white", edgecolor="black", s=36)
        ax.text(eq[0], eq[1], eq[2], f" {name}", fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Two target labels: same basin notion, opposite phase-space side")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    text = (
        "How to read the basin labels\n\n"
        "target_positive: the initial condition reaches the bounded nontrivial\n"
        "target geometry on the positive side, typically mean x > 0.\n\n"
        "target_negative: the initial condition reaches the symmetric negative\n"
        "target geometry, typically mean x < 0.\n\n"
        "For hiddenness tests, both count as target hits. If points near an\n"
        "equilibrium fall into either target class, that is evidence against\n"
        "hiddenness under the tested numerical contract.\n\n"
        "The basin plot answers: which initial conditions (x0,y0,z0 fixed)\n"
        "fall into target_positive/target_negative/equilibrium/infinity/unknown?"
    )
    ax2.text(0.02, 0.98, text, va="top", ha="left", fontsize=11, linespacing=1.35)
    fig.tight_layout()
    path = outdir / "target_positive_negative_explanation.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def make_plan(outdir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    danca_seed_id, danca_seed, danca_source = danca_seed_from_previous_replication()
    project_id, project_seed, project_source = project_best_seed()
    xvals = np.linspace(float(args.xmin), float(args.xmax), int(args.nx))
    yvals = np.linspace(float(args.ymin), float(args.ymax), int(args.ny))
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
    write_csv(outdir / "positive_x_basin_plan.csv", plan_rows)
    cfg = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "nx": int(args.nx),
        "ny": int(args.ny),
        "grid_points": int(args.nx) * int(args.ny),
        "xlim": [float(args.xmin), float(args.xmax)],
        "ylim": [float(args.ymin), float(args.ymax)],
        "plane": "xy_positive_x_focus",
        "purpose": "High-resolution positive-x basin cut around E+ to inspect whether equilibrium neighborhoods intersect the target basin.",
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
            "method": "C basin backend using EFORK finite-memory classifier; project unknown cells can be refined by target-reference geometry",
            "q": 0.9998,
            "h": float(args.project_h),
            "Lm": float(args.project_lm),
            "t_final": float(args.project_t_final),
            "t_burn": float(args.project_t_burn),
            "z_plane": float(project_seed[2]),
        },
        "equilibria": {k: v.tolist() for k, v in _equilibria_for_plot().items()},
        "classification": {
            "equilibrium_tol": float(args.equilibrium_tol),
            "divergence_norm": float(args.divergence_norm),
            "r_bound": float(args.r_bound),
            "mean_x_gap": float(args.mean_x_gap),
        },
        "chunks": int(args.chunks),
        "project_chunks": int(args.project_chunks),
        "chain_project_refinement": bool(args.chain_project_refinement),
        "refine_chunks": int(args.refine_chunks),
    }
    write_json(outdir / "positive_x_basin_config.json", cfg)
    write_json(outdir / "basin_comparison_config.json", cfg)
    plot_target_explanation(outdir, cfg)
    return cfg


def run_danca_chunk(outdir: Path, chunk_id: int, chunks: int) -> Path:
    force_single_openmp_thread_current_process()
    cfg = read_json(outdir / "positive_x_basin_config.json")
    plan = read_csv_rows(outdir / "positive_x_basin_plan.csv")
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
    path = outdir / f"danca_basin_chunk_{chunk_id:03d}.csv"
    if path.exists():
        path.unlink()
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
            row["class_id"] = _danca_class_id(row, cfg)
            row["class_label"] = class_label(row["class_id"])
        except Exception as exc:
            row = {
                **item,
                "status": "exception",
                "error": repr(exc),
                "elapsed_sec": time.time() - started,
                "final_class": "numerical_failure",
                "target_hit": False,
                "class_id": 5,
                "class_label": class_label(5),
            }
        append_csv(path, row, DANCA_FIELDS)
    write_json(outdir / f"danca_basin_chunk_{chunk_id:03d}.done", {"chunk_id": int(chunk_id), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return path


def run_project_chunk(outdir: Path, chunk_id: int, chunks: int) -> Path:
    force_single_openmp_thread_current_process()
    cfg = read_json(outdir / "positive_x_basin_config.json")
    pcfg = cfg["project"]
    plan = read_csv_rows(outdir / "positive_x_basin_plan.csv")
    backend = BasinBackend.build(output_name=f"chua_basin_posx_{int(chunk_id):03d}")
    path = outdir / f"project_basin_chunk_{chunk_id:03d}.csv"
    if path.exists():
        path.unlink()
    for item in plan:
        idx = int(item["case_index"])
        if idx % int(chunks) != int(chunk_id):
            continue
        cid = backend.classify_point(
            [float(item["x0"]), float(item["y0"]), float(pcfg["z_plane"])],
            q=float(pcfg["q"]),
            h=float(pcfg["h"]),
            Lm=float(pcfg["Lm"]),
            t_final=float(pcfg["t_final"]),
            t_burn=float(pcfg["t_burn"]),
            divergence_norm=float(cfg["classification"]["divergence_norm"]),
            r_bound=float(cfg["classification"]["r_bound"]),
            equilibrium_tol=float(cfg["classification"]["equilibrium_tol"]),
            cap_win=150,
            mean_x_gap=float(cfg["classification"]["mean_x_gap"]),
        )
        row = {
            "case_index": item["case_index"],
            "ix": item["ix"],
            "iy": item["iy"],
            "x0": item["x0"],
            "y0": item["y0"],
            "z0": float(pcfg["z_plane"]),
            "class_id": cid,
            "class_label": class_label(cid),
        }
        append_csv(path, row, PROJECT_FIELDS)
    write_json(outdir / f"project_basin_chunk_{chunk_id:03d}.done", {"chunk_id": int(chunk_id), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return path


def _grid_from_rows(rows: Sequence[Dict[str, Any]], nx: int, ny: int) -> np.ndarray:
    arr = np.full((ny, nx), 5, dtype=np.int32)
    for row in rows:
        arr[int(row["iy"]), int(row["ix"])] = int(row["class_id"])
    return arr


def _load_chunk_rows(outdir: Path, prefix: str, chunks: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for idx in range(chunks):
        rows.extend(read_csv_rows(outdir / f"{prefix}_{idx:03d}.csv"))
    rows.sort(key=lambda r: (int(r["iy"]), int(r["ix"])))
    return rows


def _counts(arr: np.ndarray) -> Dict[str, int]:
    vals, counts = np.unique(arr.astype(int), return_counts=True)
    return {class_label(int(v)): int(c) for v, c in zip(vals, counts)}


def aggregate(outdir: Path, *, wait: bool = False, poll_sec: float = 60.0) -> Path:
    cfg = read_json(outdir / "positive_x_basin_config.json")
    while wait:
        danca_done = all((outdir / f"danca_basin_chunk_{idx:03d}.done").exists() for idx in range(int(cfg["chunks"])))
        project_done = all((outdir / f"project_basin_chunk_{idx:03d}.done").exists() for idx in range(int(cfg["project_chunks"])))
        if danca_done and project_done:
            break
        time.sleep(float(poll_sec))
    danca_rows = _load_chunk_rows(outdir, "danca_basin_chunk", int(cfg["chunks"]))
    project_rows = _load_chunk_rows(outdir, "project_basin_chunk", int(cfg["project_chunks"]))
    nx, ny = int(cfg["nx"]), int(cfg["ny"])
    danca_grid = _grid_from_rows(danca_rows, nx, ny)
    project_grid = _grid_from_rows(project_rows, nx, ny)
    write_csv(outdir / "danca_basin_grid_raw.csv", danca_rows)
    write_csv(outdir / "project_best_basin_grid.csv", project_rows)
    np.save(outdir / "danca_basin_grid.npy", danca_grid)
    np.save(outdir / "project_best_basin_grid.npy", project_grid)
    plot_grid(
        danca_grid,
        cfg,
        outdir / "danca_positive_x_basin_xy.png",
        title=f"Danca ABM full history, positive-x basin h={cfg['danca']['h']}",
        seed=np.asarray(cfg["danca"]["seed"], dtype=float),
        seed_label="Danca reference seed",
        z_plane=float(cfg["danca"]["z_plane"]),
    )
    plot_grid(
        danca_grid,
        cfg,
        outdir / "danca_positive_x_Eplus_zoom.png",
        title="Danca basin zoom around E+",
        seed=np.asarray(cfg["danca"]["seed"], dtype=float),
        seed_label="Danca reference seed",
        z_plane=float(cfg["danca"]["z_plane"]),
        zoom=True,
        zoom_equilibrium="E+",
    )
    plot_grid(
        project_grid,
        cfg,
        outdir / "project_positive_x_basin_xy.png",
        title=f"Project best EFORK, positive-x basin h={cfg['project']['h']} Lm={cfg['project']['Lm']}",
        seed=np.asarray(cfg["project"]["seed"], dtype=float),
        seed_label="project best candidate seed",
        z_plane=float(cfg["project"]["z_plane"]),
    )
    plot_grid(
        project_grid,
        cfg,
        outdir / "project_positive_x_Eplus_zoom.png",
        title="Project best basin zoom around E+",
        seed=np.asarray(cfg["project"]["seed"], dtype=float),
        seed_label="project best candidate seed",
        z_plane=float(cfg["project"]["z_plane"]),
        zoom=True,
        zoom_equilibrium="E+",
    )
    equilibrium_zoom_outputs: Dict[str, Dict[str, str]] = {}
    for eq_name in visible_equilibria(cfg):
        danca_path = outdir / f"danca_{_eq_slug(eq_name)}_zoom.png"
        project_path = outdir / f"project_{_eq_slug(eq_name)}_zoom.png"
        plot_grid(
            danca_grid,
            cfg,
            danca_path,
            title=f"Danca basin zoom around {eq_name}",
            seed=np.asarray(cfg["danca"]["seed"], dtype=float),
            seed_label="Danca reference seed",
            z_plane=float(cfg["danca"]["z_plane"]),
            zoom=True,
            zoom_equilibrium=eq_name,
        )
        plot_grid(
            project_grid,
            cfg,
            project_path,
            title=f"Project best basin zoom around {eq_name}",
            seed=np.asarray(cfg["project"]["seed"], dtype=float),
            seed_label="project best candidate seed",
            z_plane=float(cfg["project"]["z_plane"]),
            zoom=True,
            zoom_equilibrium=eq_name,
        )
        equilibrium_zoom_outputs[eq_name] = {"danca": str(danca_path), "project": str(project_path)}
    summary = {
        "status": "ok" if len(danca_rows) == nx * ny and len(project_rows) == nx * ny else "partial",
        "counts": {"danca": _counts(danca_grid), "project": _counts(project_grid)},
        "outputs": {
            "danca_png": str(outdir / "danca_positive_x_basin_xy.png"),
            "danca_zoom_png": str(outdir / "danca_positive_x_Eplus_zoom.png"),
            "project_png": str(outdir / "project_positive_x_basin_xy.png"),
            "project_zoom_png": str(outdir / "project_positive_x_Eplus_zoom.png"),
            "equilibrium_zoom_pngs": equilibrium_zoom_outputs,
            "target_explanation_png": str(outdir / "target_positive_negative_explanation.png"),
            "project_csv": str(outdir / "project_best_basin_grid.csv"),
            "danca_csv": str(outdir / "danca_basin_grid_raw.csv"),
        },
        "notes": [
            "The plotted star is explicitly labeled as the reference/candidate seed.",
            "E+ and any visible equilibria are marked and labeled in x-y projection.",
            "The dashed circle around E+ is a visual neighborhood marker, not a tested radius.",
            "Project unknown cells can be refined by the chained target-reference workflow.",
        ],
    }
    write_json(outdir / "positive_x_basin_summary.json", summary)
    return outdir / "positive_x_basin_summary.json"


def refine_after_aggregate(outdir: Path, *, wait: bool = True, poll_sec: float = 120.0) -> Path:
    cfg = read_json(outdir / "positive_x_basin_config.json")
    while wait and not (outdir / "positive_x_basin_summary.json").exists():
        time.sleep(float(poll_sec))
    if not bool(cfg.get("chain_project_refinement", False)):
        write_json(outdir / "refine_after_aggregate.done", {"status": "skipped"})
        return outdir / "refine_after_aggregate.done"
    refined_dir = outdir / "project_refined_unknowns"
    argv = [
        sys.executable,
        str(ROOT / "refine_project_basin_classification.py"),
        "--source-dir",
        str(outdir),
        "--output-dir",
        str(refined_dir),
        "--chunks",
        str(int(cfg.get("refine_chunks", 4))),
    ]
    logs = outdir / "logs"
    proc = subprocess.Popen(
        argv,
        cwd=str(ROOT),
        env=force_single_openmp_thread_env(os.environ.copy()),
        stdout=(logs / "project_refine_launch.out").open("ab"),
        stderr=(logs / "project_refine_launch.err").open("ab"),
        start_new_session=True,
        close_fds=True,
    )
    manifest = {"status": "launched_project_refinement", "pid": proc.pid, "argv": argv, "output_dir": str(refined_dir)}
    write_json(outdir / "project_refinement_launch.json", manifest)
    return outdir / "project_refinement_launch.json"


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
        argv = [sys.executable, str(Path(__file__).resolve()), "--job", "danca-chunk", "--output-dir", str(outdir), "--chunk-id", str(chunk_id), "--chunks", str(args.chunks)]
        proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=(logs / f"danca_chunk_{chunk_id:03d}.out").open("ab"), stderr=(logs / f"danca_chunk_{chunk_id:03d}.err").open("ab"), start_new_session=True, close_fds=True)
        jobs.append({"name": f"danca_chunk_{chunk_id:03d}", "pid": proc.pid, "argv": argv})
    for chunk_id in range(int(args.project_chunks)):
        argv = [sys.executable, str(Path(__file__).resolve()), "--job", "project-chunk", "--output-dir", str(outdir), "--chunk-id", str(chunk_id), "--chunks", str(args.project_chunks)]
        proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=(logs / f"project_chunk_{chunk_id:03d}.out").open("ab"), stderr=(logs / f"project_chunk_{chunk_id:03d}.err").open("ab"), start_new_session=True, close_fds=True)
        jobs.append({"name": f"project_chunk_{chunk_id:03d}", "pid": proc.pid, "argv": argv})
    for job in ["aggregate", "refine-after-aggregate"]:
        argv = [sys.executable, str(Path(__file__).resolve()), "--job", job, "--output-dir", str(outdir), "--wait"]
        proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=(logs / f"{job}.out").open("ab"), stderr=(logs / f"{job}.err").open("ab"), start_new_session=True, close_fds=True)
        jobs.append({"name": job, "pid": proc.pid, "argv": argv})
    manifest = {"status": "launched", "outdir": str(outdir), "config": cfg, "jobs": jobs}
    write_json(outdir / "launch_manifest.json", manifest)
    print(json.dumps(json_safe(manifest), indent=2), flush=True)
    return outdir / "launch_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-resolution x>0 basin sweep with equilibrium labels.")
    parser.add_argument("--job", choices=["prepare", "launch", "danca-chunk", "project-chunk", "aggregate", "refine-after-aggregate"], default="launch")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / f"positive_x_basin_sweep_{timestamp()}"))
    parser.add_argument("--nx", type=int, default=161)
    parser.add_argument("--ny", type=int, default=101)
    parser.add_argument("--xmin", type=float, default=0.0)
    parser.add_argument("--xmax", type=float, default=9.0)
    parser.add_argument("--ymin", type=float, default=-1.5)
    parser.add_argument("--ymax", type=float, default=1.5)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--project-chunks", type=int, default=4)
    parser.add_argument("--refine-chunks", type=int, default=4)
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--danca-h", type=float, default=0.01)
    parser.add_argument("--danca-t-final", type=float, default=500.0)
    parser.add_argument("--danca-transient", type=float, default=250.0)
    parser.add_argument("--project-h", type=float, default=0.01)
    parser.add_argument("--project-lm", type=float, default=10.0)
    parser.add_argument("--project-t-final", type=float, default=1500.0)
    parser.add_argument("--project-t-burn", type=float, default=100.0)
    parser.add_argument("--equilibrium-tol", type=float, default=0.01)
    parser.add_argument("--divergence-norm", type=float, default=120.0)
    parser.add_argument("--r-bound", type=float, default=30.0)
    parser.add_argument("--mean-x-gap", type=float, default=0.08)
    parser.add_argument("--chain-project-refinement", action="store_true", default=True)
    parser.add_argument("--no-chain-project-refinement", dest="chain_project_refinement", action="store_false")
    parser.add_argument("--wait", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir).resolve()
    if args.job == "prepare":
        make_plan(outdir, args)
    elif args.job == "launch":
        launch(outdir, args)
    elif args.job == "danca-chunk":
        run_danca_chunk(outdir, int(args.chunk_id), int(args.chunks))
    elif args.job == "project-chunk":
        run_project_chunk(outdir, int(args.chunk_id), int(args.chunks))
    elif args.job == "aggregate":
        aggregate(outdir, wait=bool(args.wait))
    elif args.job == "refine-after-aggregate":
        refine_after_aggregate(outdir, wait=bool(args.wait))


if __name__ == "__main__":
    main()
