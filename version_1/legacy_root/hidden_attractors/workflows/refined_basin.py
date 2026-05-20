"""Refined basin classification for unresolved EFORK basin cells.

This workflow revisits cells previously labeled ``unknown`` by the coarse C
basin classifier.  The coarse classifier only used boundedness, equilibrium
capture, and a tail ``mean_x`` sign rule.  Here each unresolved point is
integrated with the same EFORK numerical contract, then compared against
positive/negative reference attractor trajectories through tail-cloud distance,
coordinate ranges, dominant FFT frequency, and a Poincare-section cloud when
available.

The workflow is conservative: it does not invent a hiddenness claim.  It only
relabels cells when the trajectory geometry is close enough to a reference
target under the explicit numerical contract.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from parallel_policy import force_single_openmp_thread_current_process, force_single_openmp_thread_env

from ..analysis.trajectory import cloud_median_distance, trajectory_metrics
from ..basins import CLASS_LABELS, class_label
from ..io import append_csv, read_csv_rows, read_json, timestamp, write_csv, write_json
from ..native.backends import FractionalChuaBackend
from ..paths import OUTPUTS, PROJECT_ROOT, RUNTIME_CACHE


DEFAULT_SOURCE_DIR = OUTPUTS / "basin_compare_danca_project_h001_grid101_20260517"

REFINED_CLASS_LABELS = {
    **CLASS_LABELS,
    6: "bounded_other",
}
CLASS_COLORS = {
    0: "#111827",
    1: "#16a34a",
    2: "#2563eb",
    3: "#dc2626",
    4: "#9ca3af",
    5: "#f59e0b",
    6: "#7c3aed",
}

REFINED_FIELDS = [
    "case_index",
    "ix",
    "iy",
    "x0",
    "y0",
    "z0",
    "old_class_id",
    "old_class_label",
    "class_id",
    "class_label",
    "refined_status",
    "best_reference",
    "best_score",
    "score_positive",
    "score_negative",
    "cloud_norm_positive",
    "cloud_norm_negative",
    "section_norm_positive",
    "section_norm_negative",
    "range_rel_positive",
    "range_rel_negative",
    "fft_rel_positive",
    "fft_rel_negative",
    "bounded",
    "diverged",
    "equilibrium_like",
    "noncollapsed_variance",
    "range_x",
    "range_y",
    "range_z",
    "var_x_tail",
    "var_y_tail",
    "var_z_tail",
    "fft_peak",
    "section_points",
    "elapsed_sec",
]


def _cache_environment() -> None:
    (RUNTIME_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
    (RUNTIME_CACHE / "xdg_cache").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE / "xdg_cache"))


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _existing_refined_case_indices(path: Path) -> set[int]:
    """Return refined ``case_index`` values already written to a chunk CSV.

    Mathematical purpose:
        The refined basin test is a set of independent integrations from
        previously unresolved initial conditions.  Reusing completed rows after
        an interruption preserves the same numerical contract while avoiding a
        silent restart from scratch.

    Validity warning:
        This is only a resume guard.  It assumes the output directory still
        belongs to the same source grid and refined-basin configuration.
    """

    done: set[int] = set()
    for row in read_csv_rows(path):
        raw = row.get("case_index")
        if raw not in (None, ""):
            done.add(_int(raw))
    return done


def _load_reference_seed(source_cfg: dict[str, Any]) -> np.ndarray:
    project = source_cfg["project"]
    seed = np.asarray(project["seed"], dtype=float)
    if seed.shape != (3,) or not np.all(np.isfinite(seed)):
        raise ValueError("Project seed in source basin config is not a finite 3-vector.")
    return seed


def _analysis_start(cfg: dict[str, Any]) -> float:
    contract = cfg["project_contract"]
    analysis = cfg["analysis"]
    return max(float(contract["t_burn"]), float(analysis["tail_fraction_start"]) * float(contract["t_final"]))


def _score_against(payload: dict[str, Any], ref: dict[str, Any], weights: dict[str, float]) -> dict[str, float]:
    ref_range = np.asarray(ref["range_vec"], dtype=float)
    denom = max(float(np.linalg.norm(ref_range)), 1.0e-12)
    range_rel = float(np.linalg.norm(np.asarray(payload["range_vec"], dtype=float) - ref_range) / denom)
    ref_fft = float(ref.get("fft_peak", float("nan")))
    peak = float(payload.get("fft_peak", float("nan")))
    fft_rel = float(abs(peak - ref_fft) / max(abs(ref_fft), 1.0e-12)) if math.isfinite(ref_fft) and math.isfinite(peak) else float("nan")
    cloud = cloud_median_distance(np.asarray(payload["tail_sample"], dtype=float), np.asarray(ref["tail_sample"], dtype=float))
    cloud_norm = cloud / denom if math.isfinite(cloud) else float("nan")
    section = cloud_median_distance(np.asarray(payload["section"], dtype=float), np.asarray(ref["section"], dtype=float))
    section_norm = section / denom if math.isfinite(section) else float("nan")
    terms: list[tuple[float, float]] = []
    if math.isfinite(cloud_norm):
        terms.append((weights["cloud"], cloud_norm))
    if math.isfinite(range_rel):
        terms.append((weights["range"], range_rel))
    if math.isfinite(fft_rel):
        terms.append((weights["fft"], min(fft_rel, 2.0)))
    if math.isfinite(section_norm):
        terms.append((weights["section"], section_norm))
    score = sum(w * v for w, v in terms) / max(sum(w for w, _ in terms), 1.0e-12)
    return {
        "score": float(score),
        "cloud_norm": float(cloud_norm),
        "section_norm": float(section_norm),
        "range_rel": float(range_rel),
        "fft_rel": float(fft_rel),
    }


def _reference_payloads(cfg: dict[str, Any], backend: FractionalChuaBackend) -> dict[str, dict[str, Any]]:
    seed = np.asarray(cfg["reference"]["positive_seed"], dtype=float)
    contract = cfg["project_contract"]
    analysis_start = _analysis_start(cfg)
    refs: dict[str, dict[str, Any]] = {}
    for label, x0 in {"positive": seed, "negative": -seed}.items():
        traj = backend.integrate_efork3(
            x0,
            q=float(contract["q"]),
            h=float(contract["h"]),
            Lm=float(contract["Lm"]),
            t_final=float(contract["t_final"]),
        )
        metrics, payload = trajectory_metrics(
            traj,
            h=float(contract["h"]),
            t_start=analysis_start,
            divergence_norm=float(cfg["analysis"]["divergence_norm"]),
            equilibrium_tol=float(cfg["analysis"]["equilibrium_tol"]),
            max_section_points=int(cfg["analysis"]["max_section_points"]),
            max_cloud_points=int(cfg["analysis"]["max_cloud_points"]),
        )
        refs[label] = {**payload, "metrics": metrics}
    return refs


def make_config(outdir: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    """Prepare a refined-basin run from an existing coarse basin output."""

    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_dir).resolve()
    source_cfg = read_json(source_dir / "basin_comparison_config.json")
    nx = int(source_cfg.get("nx", source_cfg.get("grid")))
    ny = int(source_cfg.get("ny", source_cfg.get("grid")))
    source_rows = read_csv_rows(source_dir / "project_best_basin_grid.csv")
    unknown_rows = [row for row in source_rows if row.get("class_label") == "unknown"]
    if int(args.max_unknowns) > 0:
        unknown_rows = unknown_rows[: int(args.max_unknowns)]
    for row in unknown_rows:
        row["case_index"] = int(row["iy"]) * nx + int(row["ix"])
        row["old_class_id"] = row.get("class_id", "")
        row["old_class_label"] = row.get("class_label", "")
    write_csv(root / "unknown_refinement_plan.csv", unknown_rows)
    project = source_cfg["project"]
    classification = source_cfg["classification"]
    cfg = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "target-reference refined classification for previously unknown project basin cells",
        "source_dir": str(source_dir),
        "source_grid_csv": str(source_dir / "project_best_basin_grid.csv"),
        "source_png": str(source_dir / "project_best_basin_xy.png"),
        "nx": nx,
        "ny": ny,
        "grid": nx,
        "xlim": source_cfg["xlim"],
        "ylim": source_cfg["ylim"],
        "plane": source_cfg.get("plane", "xy"),
        "project_contract": {
            "candidate_id": project["candidate_id"],
            "q": float(project["q"]),
            "h": float(project["h"]),
            "Lm": float(project["Lm"]),
            "t_final": float(args.t_final) if float(args.t_final) > 0 else float(project["t_final"]),
            "t_burn": float(args.t_burn) if float(args.t_burn) >= 0 else float(project["t_burn"]),
            "z_plane": float(project["z_plane"]),
        },
        "reference": {
            "positive_seed": _load_reference_seed(source_cfg).tolist(),
            "negative_seed_policy": "symmetry: negative_seed = -positive_seed",
        },
        "analysis": {
            "divergence_norm": float(args.divergence_norm) if float(args.divergence_norm) > 0 else float(classification["divergence_norm"]),
            "equilibrium_tol": float(args.equilibrium_tol) if float(args.equilibrium_tol) > 0 else float(classification["equilibrium_tol"]),
            "tail_fraction_start": float(args.tail_fraction_start),
            "max_cloud_points": int(args.max_cloud_points),
            "max_section_points": int(args.max_section_points),
            "max_score": float(args.max_score),
            "max_cloud_norm": float(args.max_cloud_norm),
            "max_range_rel": float(args.max_range_rel),
            "max_fft_rel": float(args.max_fft_rel),
            "max_section_norm": float(args.max_section_norm),
            "score_weights": {
                "cloud": float(args.weight_cloud),
                "range": float(args.weight_range),
                "fft": float(args.weight_fft),
                "section": float(args.weight_section),
            },
        },
        "unknown_cells_planned": len(unknown_rows),
        "chunks": int(args.chunks),
    }
    write_json(root / "refined_basin_config.json", cfg)
    return cfg


def classify_refined(traj: np.ndarray, cfg: dict[str, Any], refs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Classify one trajectory using target-reference geometry."""

    contract = cfg["project_contract"]
    analysis = cfg["analysis"]
    metrics, payload = trajectory_metrics(
        traj,
        h=float(contract["h"]),
        t_start=_analysis_start(cfg),
        divergence_norm=float(analysis["divergence_norm"]),
        equilibrium_tol=float(analysis["equilibrium_tol"]),
        max_section_points=int(analysis["max_section_points"]),
        max_cloud_points=int(analysis["max_cloud_points"]),
    )
    if metrics["diverged"]:
        return {**metrics, "class_id": 3, "class_label": "infinity", "refined_status": "diverged", "best_reference": "", "best_score": float("nan")}
    if metrics["equilibrium_like"]:
        return {**metrics, "class_id": 0, "class_label": "equilibrium", "refined_status": "equilibrium_like", "best_reference": "", "best_score": float("nan")}
    if not metrics["bounded"]:
        return {**metrics, "class_id": 4, "class_label": "unknown", "refined_status": "not_bounded_not_diverged", "best_reference": "", "best_score": float("nan")}
    if not metrics["noncollapsed_variance"]:
        return {**metrics, "class_id": 4, "class_label": "unknown", "refined_status": "collapsed_or_low_variance", "best_reference": "", "best_score": float("nan")}

    weights = {k: float(v) for k, v in analysis["score_weights"].items()}
    pos = _score_against(payload, refs["positive"], weights)
    neg = _score_against(payload, refs["negative"], weights)
    best_label, best = ("positive", pos) if pos["score"] <= neg["score"] else ("negative", neg)
    score_ok = best["score"] <= float(analysis["max_score"])
    cloud_ok = best["cloud_norm"] <= float(analysis["max_cloud_norm"]) if math.isfinite(best["cloud_norm"]) else False
    range_ok = best["range_rel"] <= float(analysis["max_range_rel"]) if math.isfinite(best["range_rel"]) else False
    fft_ok = best["fft_rel"] <= float(analysis["max_fft_rel"]) if math.isfinite(best["fft_rel"]) else True
    section_ok = best["section_norm"] <= float(analysis["max_section_norm"]) if math.isfinite(best["section_norm"]) else True
    close_to_target = bool(score_ok and cloud_ok and range_ok and fft_ok and section_ok)
    class_id = 1 if best_label == "positive" else 2
    if close_to_target:
        refined_status = "matched_target_reference"
    else:
        class_id = 6
        refined_status = "bounded_noncollapsed_reference_mismatch"
    return {
        **metrics,
        "class_id": class_id,
        "class_label": REFINED_CLASS_LABELS[class_id],
        "refined_status": refined_status,
        "best_reference": best_label,
        "best_score": best["score"],
        "score_positive": pos["score"],
        "score_negative": neg["score"],
        "cloud_norm_positive": pos["cloud_norm"],
        "cloud_norm_negative": neg["cloud_norm"],
        "section_norm_positive": pos["section_norm"],
        "section_norm_negative": neg["section_norm"],
        "range_rel_positive": pos["range_rel"],
        "range_rel_negative": neg["range_rel"],
        "fft_rel_positive": pos["fft_rel"],
        "fft_rel_negative": neg["fft_rel"],
    }


def run_chunk(outdir: str | Path, chunk_id: int, chunks: int, *, skip_existing: bool = False) -> Path:
    """Reintegrate and refine one chunk of previously unknown cells."""

    force_single_openmp_thread_current_process()
    root = Path(outdir)
    cfg = read_json(root / "refined_basin_config.json")
    plan = read_csv_rows(root / "unknown_refinement_plan.csv")
    backend = FractionalChuaBackend.build(output_name=f"chua_frac_refined_basin_{int(chunk_id):03d}")
    refs = _reference_payloads(cfg, backend)
    path = root / f"unknown_refined_chunk_{chunk_id:03d}.csv"
    if path.exists() and not skip_existing:
        path.unlink()
    done = root / f"unknown_refined_chunk_{chunk_id:03d}.done"
    if done.exists():
        done.unlink()
    completed = _existing_refined_case_indices(path) if skip_existing else set()
    rows_done = 0
    contract = cfg["project_contract"]
    for row in plan:
        case_index = int(row["case_index"])
        if case_index % int(chunks) != int(chunk_id):
            continue
        if case_index in completed:
            continue
        started = time.time()
        out: dict[str, Any]
        try:
            traj = backend.integrate_efork3(
                [_float(row["x0"]), _float(row["y0"]), _float(row["z0"])],
                q=float(contract["q"]),
                h=float(contract["h"]),
                Lm=float(contract["Lm"]),
                t_final=float(contract["t_final"]),
            )
            out = classify_refined(traj, cfg, refs)
        except Exception as exc:
            out = {
                "class_id": 5,
                "class_label": "numerical_failure",
                "refined_status": repr(exc),
                "best_reference": "",
                "best_score": float("nan"),
            }
        final = {
            "case_index": row["case_index"],
            "ix": row["ix"],
            "iy": row["iy"],
            "x0": row["x0"],
            "y0": row["y0"],
            "z0": row["z0"],
            "old_class_id": row.get("old_class_id", row.get("class_id", "")),
            "old_class_label": row.get("old_class_label", row.get("class_label", "")),
            **out,
            "elapsed_sec": time.time() - started,
        }
        append_csv(path, final, REFINED_FIELDS)
        rows_done += 1
        if rows_done % 10 == 0:
            print(f"refined chunk {chunk_id}: {rows_done} rows", flush=True)
    write_json(
        done,
        {
            "chunk_id": int(chunk_id),
            "rows": len(completed) + rows_done,
            "rows_added": rows_done,
            "skip_existing": bool(skip_existing),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    return path


def _counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("class_label", "")) for row in rows))


def plot_grid(outdir: Path, cfg: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    """Plot the refined basin grid."""

    _cache_environment()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    nx = int(cfg.get("nx", cfg["grid"]))
    ny = int(cfg.get("ny", cfg["grid"]))
    arr = np.full((ny, nx), 5, dtype=np.int32)
    for row in rows:
        arr[_int(row["iy"]), _int(row["ix"])] = _int(row["class_id"], 5)
    np.save(outdir / "project_best_basin_refined_grid.npy", arr)
    color_list = [CLASS_COLORS[k] for k in sorted(CLASS_COLORS)]
    cmap = ListedColormap(color_list)
    norm = BoundaryNorm(np.arange(-0.5, len(color_list) + 0.5, 1.0), cmap.N)
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ax.imshow(
        arr,
        origin="lower",
        extent=[float(cfg["xlim"][0]), float(cfg["xlim"][1]), float(cfg["ylim"][0]), float(cfg["ylim"][1])],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="auto",
    )
    seed = np.asarray(cfg["reference"]["positive_seed"], dtype=float)
    ax.scatter([seed[0]], [seed[1]], s=42, marker="*", color="#ec4899", edgecolor="black", linewidth=0.4, label="candidate seed")
    ax.set_xlabel("x0")
    ax.set_ylabel("y0")
    ax.set_title(f"refined project basin, z={float(cfg['project_contract']['z_plane']):.4g}")
    handles = [
        Patch(facecolor=CLASS_COLORS[k], edgecolor="black", linewidth=0.3, label=REFINED_CLASS_LABELS[k])
        for k in sorted(CLASS_COLORS)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.88)
    fig.tight_layout()
    path = outdir / "project_best_basin_refined_xy.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def aggregate(outdir: str | Path, *, wait: bool = False, poll_sec: float = 30.0) -> Path:
    """Merge refined chunks with the original grid and write final artifacts."""

    root = Path(outdir)
    cfg = read_json(root / "refined_basin_config.json")
    chunks = int(cfg["chunks"])
    while wait:
        if all((root / f"unknown_refined_chunk_{idx:03d}.done").exists() for idx in range(chunks)):
            break
        time.sleep(float(poll_sec))
    source_rows = read_csv_rows(Path(cfg["source_grid_csv"]))
    by_key = {(int(row["iy"]), int(row["ix"])): dict(row) for row in source_rows}
    refined_rows: list[dict[str, str]] = []
    for idx in range(chunks):
        refined_rows.extend(read_csv_rows(root / f"unknown_refined_chunk_{idx:03d}.csv"))
    for row in refined_rows:
        key = (int(row["iy"]), int(row["ix"]))
        merged = by_key[key]
        merged.update({k: row.get(k, "") for k in row})
        by_key[key] = merged
    all_rows = [by_key[key] for key in sorted(by_key)]
    write_csv(root / "unknown_refined_rows.csv", refined_rows, REFINED_FIELDS)
    write_csv(root / "project_best_basin_refined_grid.csv", all_rows)
    plot_path = plot_grid(root, cfg, all_rows)
    summary = {
        "status": "ok" if len(refined_rows) == int(cfg["unknown_cells_planned"]) else "partial",
        "source_counts": _counts(source_rows),
        "refined_counts": _counts(all_rows),
        "unknown_cells_planned": int(cfg["unknown_cells_planned"]),
        "unknown_cells_refined": len(refined_rows),
        "refined_png": plot_path,
        "refined_csv": str(root / "project_best_basin_refined_grid.csv"),
        "unknown_refined_csv": str(root / "unknown_refined_rows.csv"),
        "notes": [
            "Only cells labeled unknown in the source grid were reintegrated with the enriched classifier.",
            "Non-unknown source labels were preserved.",
            "Target labels require bounded noncollapsed trajectories and closeness to the positive/negative reference geometry.",
        ],
    }
    write_json(root / "refined_basin_summary.json", summary)
    return root / "refined_basin_summary.json"


def launch(outdir: str | Path, args: argparse.Namespace) -> Path:
    """Launch independent OS processes for chunked refined classification."""

    root = Path(outdir)
    cfg = make_config(root, args)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = force_single_openmp_thread_env(os.environ.copy())
    env["PYTHONUNBUFFERED"] = "1"
    script = Path(args.script_path).resolve()
    launched: list[dict[str, Any]] = []
    for idx in range(int(args.chunks)):
        cmd = [sys.executable, str(script), "--job", "chunk", "--output-dir", str(root), "--chunk-id", str(idx), "--chunks", str(args.chunks)]
        if bool(args.skip_existing):
            cmd.append("--skip-existing")
        stdout = (logs / f"chunk_{idx:03d}.out").open("ab")
        stderr = (logs / f"chunk_{idx:03d}.err").open("ab")
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        launched.append({"job": f"chunk_{idx:03d}", "pid": proc.pid, "cmd": cmd})
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
    parser = argparse.ArgumentParser(description="Refine unknown cells in the project basin using target-reference trajectory geometry.")
    parser.add_argument("--job", choices=["launch", "chunk", "aggregate"], default="launch")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--t-final", type=float, default=0.0, help="Override project t_final; 0 keeps source contract.")
    parser.add_argument("--t-burn", type=float, default=-1.0, help="Override project t_burn; negative keeps source contract.")
    parser.add_argument("--divergence-norm", type=float, default=0.0, help="Override source divergence norm; 0 keeps source value.")
    parser.add_argument("--equilibrium-tol", type=float, default=0.0, help="Override source equilibrium tolerance; 0 keeps source value.")
    parser.add_argument("--tail-fraction-start", type=float, default=0.5)
    parser.add_argument("--max-cloud-points", type=int, default=700)
    parser.add_argument("--max-section-points", type=int, default=250)
    parser.add_argument("--max-score", type=float, default=0.95)
    parser.add_argument("--max-cloud-norm", type=float, default=0.85)
    parser.add_argument("--max-range-rel", type=float, default=1.50)
    parser.add_argument("--max-fft-rel", type=float, default=1.00)
    parser.add_argument("--max-section-norm", type=float, default=1.20)
    parser.add_argument("--weight-cloud", type=float, default=1.0)
    parser.add_argument("--weight-range", type=float, default=0.35)
    parser.add_argument("--weight-fft", type=float, default=0.20)
    parser.add_argument("--weight-section", type=float, default=0.35)
    parser.add_argument("--max-unknowns", type=int, default=0, help="Debug cap; 0 means all unknown cells.")
    parser.add_argument("--skip-existing", action="store_true", help="Keep existing refined chunk rows and only integrate missing case_index values.")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--script-path", default=str(PROJECT_ROOT / "refine_project_basin_classification.py"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    outdir = Path(args.output_dir).resolve() if args.output_dir else OUTPUTS / f"project_basin_refined_{timestamp()}"
    if args.job == "launch":
        launch(outdir, args)
    elif args.job == "chunk":
        run_chunk(outdir, int(args.chunk_id), int(args.chunks), skip_existing=bool(args.skip_existing))
    elif args.job == "aggregate":
        aggregate(outdir, wait=bool(args.wait))
