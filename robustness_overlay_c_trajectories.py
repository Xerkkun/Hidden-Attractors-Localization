#!/usr/bin/env python3
"""C/EFORK trajectory overlays for numerical robustness inspection.

Mathematical purpose:
    Compare whether an observed candidate attractor remains qualitatively
    similar when the numerical contract changes.  This script does not decide
    hiddenness and does not classify the result as target/non-target.  It
    records geometric and spectral diagnostics for visual inspection.

Numerical model:
    Caputo EFORK finite-memory integration through chua_frac_backend_lib.c.
    The finite memory length Lm is part of the tested numerical contract.

Validity warning:
    Chaotic trajectories are not expected to coincide pointwise.  Robustness is
    therefore inspected through tail ranges, dominant FFT frequency, tail point
    cloud distance, Poincare-section distance, non-collapse, no equilibrium
    convergence, and no divergence.
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
_CACHE_ROOT = ROOT / ".runtime_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg_cache"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import chua_initial_cond as chua
from extended_search_utils import fft_peak_and_entropy, json_safe, trajectory_ranges
from lure_top3_sphere_robustness import load_requested_candidates
from parallel_policy import compile_c_target, force_single_openmp_thread_env, force_single_openmp_thread_current_process


SOURCE_DIR = ROOT / "outputs" / "lure_biased_multiparam_q09998_20260515_195444"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs"

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


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def safe_name(text: str) -> str:
    out = []
    for ch in str(text):
        out.append(ch if ch.isalnum() or ch in {"_", "-"} else "_")
    return "".join(out)


def default_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    base = {"q": float(args.q), "h": 0.01, "Lm": 10.0, "t_final": 1500.0, "t_burn": 100.0}
    cases = [
        {"case_id": "R0_base", **base},
        {"case_id": "R1_h_finer", **base, "h": 0.005},
        {"case_id": "R2_h_coarser", **base, "h": 0.02},
        {"case_id": "R3_Lm_lower", **base, "Lm": 5.0},
        {"case_id": "R4_Lm_higher", **base, "Lm": 20.0},
        {"case_id": "R5_t_longer", **base, "t_final": 3000.0, "t_burn": 200.0},
    ]
    return cases


def make_config(outdir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    candidates = load_requested_candidates(Path(args.source_dir).resolve())
    cases = default_cases(args)
    base = cases[0]
    for case in cases:
        case["h_change_pct"] = 100.0 * (float(case["h"]) - float(base["h"])) / float(base["h"])
        case["Lm_change_pct"] = 100.0 * (float(case["Lm"]) - float(base["Lm"])) / float(base["Lm"])
        case["t_final_change_pct"] = 100.0 * (float(case["t_final"]) - float(base["t_final"])) / float(base["t_final"])
    cfg = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "C EFORK trajectory overlay; no target classification",
        "params": {
            "alpha": 8.4562,
            "beta": 12.0732,
            "gamma": 0.0052,
            "m0": -0.1768,
            "m1": -1.1468,
        },
        "analysis": {
            "divergence_norm": float(args.divergence_norm),
            "equilibrium_tol": float(args.equilibrium_tol),
            "max_store_points": int(args.max_store_points),
            "max_metric_points": int(args.max_metric_points),
            "max_section_points": int(args.max_section_points),
            "tail_fraction_start": float(args.tail_fraction_start),
            "cloud_note": "Distances compare subsampled tail point clouds, normalized by baseline range norm.",
            "section_note": "Poincare section uses x=0 upward crossings after analysis_start.",
        },
        "candidates": candidates,
        "cases": cases,
        "chunks": len(candidates),
    }
    write_json(outdir / "robustness_overlay_config.json", cfg)
    return cfg


def load_frac_lib() -> Any:
    native_dir = ROOT / ".runtime_native"
    native_dir.mkdir(exist_ok=True)
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    result = compile_c_target(
        ROOT / "chua_frac_backend_lib.c",
        native_dir / f"chua_frac_overlay{ext}",
        target_kind="shared",
        openmp=False,
    )
    lib = ctypes.CDLL(str(result.path.resolve()))
    lib.set_frac_chua_params.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.set_frac_chua_params.restype = None
    lib.set_frac_chua_model.argtypes = [ctypes.c_int]
    lib.set_frac_chua_model.restype = None
    lib.efork_rows.argtypes = [ctypes.c_double, ctypes.c_double]
    lib.efork_rows.restype = ctypes.c_int
    lib.integrate_chua_efork3.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ]
    lib.integrate_chua_efork3.restype = ctypes.c_int
    lib.set_frac_chua_model(0)
    lib.set_frac_chua_params(8.4562, 12.0732, 0.0052, -0.1768, -1.1468)
    return lib


def integrate_c(lib: Any, x0: Sequence[float], case: Dict[str, Any]) -> np.ndarray:
    rows = int(lib.efork_rows(float(case["t_final"]), float(case["h"])))
    if rows <= 0:
        raise RuntimeError(f"efork_rows returned {rows}")
    out = np.empty(rows * 4, dtype=np.float64)
    rc = int(
        lib.integrate_chua_efork3(
            float(x0[0]),
            float(x0[1]),
            float(x0[2]),
            float(case["q"]),
            float(case["h"]),
            float(case["Lm"]),
            float(case["t_final"]),
            0.0,
            1.0,
            out,
        )
    )
    if rc != 0:
        raise RuntimeError(f"integrate_chua_efork3 returned {rc}")
    return out.reshape((rows, 4))


def tail_slice(traj: np.ndarray, case: Dict[str, Any], cfg: Dict[str, Any]) -> np.ndarray:
    start_t = max(float(case["t_burn"]), float(cfg["analysis"]["tail_fraction_start"]) * float(case["t_final"]))
    return traj[traj[:, 0] >= start_t]


def sample_rows(arr: np.ndarray, max_points: int) -> np.ndarray:
    X = np.asarray(arr)
    if X.shape[0] <= int(max_points):
        return X
    idx = np.linspace(0, X.shape[0] - 1, int(max_points)).astype(int)
    return X[idx]


def save_sampled_trajectory(path: Path, traj: np.ndarray, cfg: Dict[str, Any]) -> int:
    sampled = sample_rows(traj, int(cfg["analysis"]["max_store_points"]))
    rows = [
        {"t": float(r[0]), "x": float(r[1]), "y": float(r[2]), "z": float(r[3])}
        for r in sampled
    ]
    write_csv(path, rows, ["t", "x", "y", "z"])
    return int(sampled.shape[0])


def equilibria() -> Dict[str, np.ndarray]:
    return {
        "E0": np.array([0.0, 0.0, 0.0], dtype=float),
        "E+": np.array([6.5883078865388685, 0.0028364022560936064, -6.585471484282775], dtype=float),
        "E-": np.array([-6.5883078865388685, -0.0028364022560936064, 6.585471484282775], dtype=float),
    }


def min_final_eq_dist(final: np.ndarray) -> float:
    return float(min(np.linalg.norm(final - eq) for eq in equilibria().values()))


def chua_xdot(state: np.ndarray) -> float:
    p = {"model": "piecewise", "alpha": np.float64(8.4562), "beta": np.float64(12.0732), "gamma": np.float64(0.0052), "m0": np.float64(-0.1768), "m1": np.float64(-1.1468)}
    return float(chua.rhs_original(np.asarray(state, dtype=float), p)[0])


def section_points(traj: np.ndarray, t_start: float, max_points: int) -> np.ndarray:
    pts: List[Tuple[float, float]] = []
    X = np.asarray(traj, dtype=float)
    for k in range(1, X.shape[0]):
        if X[k, 0] < float(t_start):
            continue
        xp = X[k - 1, 1]
        x = X[k, 1]
        if xp < 0.0 <= x and chua_xdot(X[k, 1:4]) > 0.0:
            lam = (0.0 - xp) / ((x - xp) + 1e-300)
            y = X[k - 1, 2] + lam * (X[k, 2] - X[k - 1, 2])
            z = X[k - 1, 3] + lam * (X[k, 3] - X[k - 1, 3])
            pts.append((float(y), float(z)))
            if len(pts) >= int(max_points):
                break
    return np.asarray(pts, dtype=float)


def median_cloud_distance(a: np.ndarray, b: np.ndarray) -> float:
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    if A.size == 0 or B.size == 0:
        return float("nan")
    # Chunked nearest-neighbor distances to avoid an accidental huge matrix.
    def one_way(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        vals: List[np.ndarray] = []
        for i in range(0, P.shape[0], 128):
            block = P[i : i + 128]
            d = np.linalg.norm(block[:, None, :] - Q[None, :, :], axis=2)
            vals.append(np.min(d, axis=1))
        return np.concatenate(vals) if vals else np.empty(0)

    d1 = one_way(A, B)
    d2 = one_way(B, A)
    return float(np.median(np.concatenate([d1, d2])))


def compute_metrics(
    cand: Dict[str, Any],
    case: Dict[str, Any],
    traj: np.ndarray,
    stored_rows: int,
    cfg: Dict[str, Any],
    ref: Dict[str, Any] | None,
    traj_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    analysis_start = max(float(case["t_burn"]), float(cfg["analysis"]["tail_fraction_start"]) * float(case["t_final"]))
    tail = tail_slice(traj, case, cfg)
    states = traj[:, 1:4]
    tail_states = tail[:, 1:4] if tail.shape[0] else np.empty((0, 3))
    finite = bool(states.size > 0 and np.all(np.isfinite(states)))
    norms = np.linalg.norm(states, axis=1) if states.size else np.array([float("inf")])
    max_norm = float(np.max(norms))
    final = states[-1] if states.size else np.array([float("nan"), float("nan"), float("nan")])
    ranges = trajectory_ranges(tail if tail.shape[0] else traj)
    var = np.var(tail_states, axis=0) if tail_states.size else np.array([float("nan"), float("nan"), float("nan")])
    fft = fft_peak_and_entropy(tail if tail.shape[0] else traj, float(case["h"]), component=0)
    section = section_points(traj, analysis_start, int(cfg["analysis"]["max_section_points"]))
    eq_like = bool(finite and min_final_eq_dist(final) <= float(cfg["analysis"]["equilibrium_tol"]))
    diverged = bool((not finite) or max_norm > float(cfg["analysis"]["divergence_norm"]))
    noncollapsed = bool(np.nanmax(var) > 1.0e-6 and max(ranges["range_x"], ranges["range_y"], ranges["range_z"]) > 1.0e-2)
    payload = {
        "tail_sample": sample_rows(tail_states, int(cfg["analysis"]["max_metric_points"])),
        "section": section,
        "range_vec": np.array([ranges["range_x"], ranges["range_y"], ranges["range_z"]], dtype=float),
        "fft_peak": float(fft["fft_peak"]),
    }
    range_rel = fft_rel = cloud = cloud_norm = secd = secd_norm = float("nan")
    if ref is not None:
        ref_range = np.asarray(ref["range_vec"], dtype=float)
        denom = max(float(np.linalg.norm(ref_range)), 1.0e-12)
        range_rel = float(np.linalg.norm(payload["range_vec"] - ref_range) / denom)
        ref_fft = float(ref.get("fft_peak", float("nan")))
        fft_rel = float(abs(payload["fft_peak"] - ref_fft) / max(abs(ref_fft), 1.0e-12)) if math.isfinite(ref_fft) else float("nan")
        cloud = median_cloud_distance(payload["tail_sample"], np.asarray(ref["tail_sample"], dtype=float))
        cloud_norm = cloud / denom if math.isfinite(cloud) else float("nan")
        secd = median_cloud_distance(payload["section"], np.asarray(ref["section"], dtype=float))
        secd_norm = secd / denom if math.isfinite(secd) else float("nan")
    metric = {
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
        "stored_rows": int(stored_rows),
        "analysis_start": analysis_start,
        "bounded": bool(finite and not diverged),
        "diverged": bool(diverged),
        "equilibrium_like": bool(eq_like),
        "noncollapsed_variance": bool(noncollapsed),
        "final_norm": float(np.linalg.norm(final)),
        "max_norm": max_norm,
        "range_x": ranges["range_x"],
        "range_y": ranges["range_y"],
        "range_z": ranges["range_z"],
        "var_x_tail": float(var[0]),
        "var_y_tail": float(var[1]),
        "var_z_tail": float(var[2]),
        "fft_peak": float(fft["fft_peak"]),
        "psd_entropy": float(fft["psd_entropy"]),
        "section_points": int(section.shape[0]) if section.ndim == 2 else 0,
        "range_relative_distance": range_rel,
        "fft_relative_delta": fft_rel,
        "cloud_median_distance": cloud,
        "cloud_median_distance_norm": cloud_norm,
        "section_median_distance": secd,
        "section_median_distance_norm": secd_norm,
        "trajectory_csv": str(traj_path),
    }
    return metric, payload


def run_candidate(outdir: Path, candidate_index: int) -> Path:
    force_single_openmp_thread_current_process()
    cfg = read_json(outdir / "robustness_overlay_config.json")
    cand = cfg["candidates"][int(candidate_index)]
    lib = load_frac_lib()
    cid_safe = safe_name(cand["candidate_id"])
    traj_dir = outdir / "trajectories" / cid_safe
    traj_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    ref_payload: Dict[str, Any] | None = None
    for case in cfg["cases"]:
        started = time.time()
        traj = integrate_c(lib, cand["robust_start"], case)
        traj_path = traj_dir / f"{case['case_id']}.csv"
        stored = save_sampled_trajectory(traj_path, traj, cfg)
        metric, payload = compute_metrics(cand, case, traj, stored, cfg, ref_payload, traj_path)
        metric["elapsed_sec"] = time.time() - started
        rows.append(metric)
        if case["case_id"] == "R0_base":
            ref_payload = payload
        print(f"{cand['candidate_id']} {case['case_id']} rows={traj.shape[0]} elapsed={metric['elapsed_sec']:.2f}", flush=True)
    path = outdir / f"metrics_{cid_safe}.csv"
    write_csv(path, rows, METRIC_FIELDS + ["elapsed_sec"])
    (outdir / f"metrics_{cid_safe}.done").write_text(
        json.dumps({"candidate_id": cand["candidate_id"], "rows": len(rows), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}),
        encoding="utf-8",
    )
    return path


def load_traj_csv(path: Path) -> np.ndarray:
    rows = read_csv_rows(path)
    data = [[float(r["t"]), float(r["x"]), float(r["y"]), float(r["z"])] for r in rows]
    return np.asarray(data, dtype=float)


def plot_candidate(outdir: Path, cand: Dict[str, Any], metric_rows: Sequence[Dict[str, str]]) -> str:
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(metric_rows), 1)))
    fig = plt.figure(figsize=(12.5, 10.0))
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    axxy = fig.add_subplot(2, 2, 2)
    axxz = fig.add_subplot(2, 2, 3)
    axyz = fig.add_subplot(2, 2, 4)
    for idx, row in enumerate(metric_rows):
        path = Path(row["trajectory_csv"])
        X = load_traj_csv(path)
        t_final = float(row["t_final"])
        analysis_start = max(float(row["t_burn"]), 0.5 * t_final)
        X = X[X[:, 0] >= analysis_start]
        X = sample_rows(X, 1500)
        label = f"{row['case_id']} h={float(row['h']):g} Lm={float(row['Lm']):g} T={float(row['t_final']):g}"
        c = colors[idx]
        ax3.plot(X[:, 1], X[:, 2], X[:, 3], lw=0.65, alpha=0.82, color=c, label=label)
        axxy.plot(X[:, 1], X[:, 2], lw=0.65, alpha=0.82, color=c)
        axxz.plot(X[:, 1], X[:, 3], lw=0.65, alpha=0.82, color=c)
        axyz.plot(X[:, 2], X[:, 3], lw=0.65, alpha=0.82, color=c)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    axxy.set_xlabel("x")
    axxy.set_ylabel("y")
    axxz.set_xlabel("x")
    axxz.set_ylabel("z")
    axyz.set_xlabel("y")
    axyz.set_ylabel("z")
    ax3.set_title("3D overlay")
    axxy.set_title("xy")
    axxz.set_title("xz")
    axyz.set_title("yz")
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=True)
    fig.suptitle(cand["candidate_id"], fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"overlay_{safe_name(cand['candidate_id'])}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def aggregate(outdir: Path, wait: bool) -> Path:
    cfg = read_json(outdir / "robustness_overlay_config.json")
    while wait:
        done = [
            (outdir / f"metrics_{safe_name(c['candidate_id'])}.done").exists()
            for c in cfg["candidates"]
        ]
        if all(done):
            break
        time.sleep(30.0)
    all_rows: List[Dict[str, str]] = []
    plots: List[str] = []
    for cand in cfg["candidates"]:
        path = outdir / f"metrics_{safe_name(cand['candidate_id'])}.csv"
        rows = read_csv_rows(path)
        all_rows.extend(rows)
        if rows:
            plots.append(plot_candidate(outdir, cand, rows))
    write_csv(outdir / "robustness_overlay_metrics.csv", all_rows, METRIC_FIELDS + ["elapsed_sec"])
    summary = {
        "status": "ok" if len(all_rows) == len(cfg["candidates"]) * len(cfg["cases"]) else "partial",
        "metric_rows": len(all_rows),
        "metrics_csv": str(outdir / "robustness_overlay_metrics.csv"),
        "plots": plots,
        "notes": [
            "No hiddenness or target classification is made here.",
            "Baseline is R0_base per candidate; relative distances compare each case against that baseline.",
            "Overlay plots use the post-transient analysis tail only.",
        ],
    }
    write_json(outdir / "robustness_overlay_summary.json", summary)
    return outdir / "robustness_overlay_summary.json"


def launch(outdir: Path, args: argparse.Namespace) -> None:
    cfg = make_config(outdir, args)
    logs = outdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = force_single_openmp_thread_env(os.environ.copy())
    env["PYTHONUNBUFFERED"] = "1"
    launched: List[Dict[str, Any]] = []
    for idx, cand in enumerate(cfg["candidates"]):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--job", "candidate", "--output-dir", str(outdir), "--candidate-index", str(idx)]
        stdout = (logs / f"candidate_{idx}_{safe_name(cand['candidate_id'])}.out").open("ab")
        stderr = (logs / f"candidate_{idx}_{safe_name(cand['candidate_id'])}.err").open("ab")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        launched.append({"job": "candidate", "candidate_id": cand["candidate_id"], "pid": proc.pid, "cmd": cmd})
    cmd = [sys.executable, str(Path(__file__).resolve()), "--job", "aggregate", "--output-dir", str(outdir), "--wait"]
    stdout = (logs / "aggregate.out").open("ab")
    stderr = (logs / "aggregate.err").open("ab")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    launched.append({"job": "aggregate", "pid": proc.pid, "cmd": cmd})
    manifest = {"output_dir": str(outdir), "launched": launched}
    write_json(outdir / "launch_manifest.json", manifest)
    print(json.dumps(json_safe(manifest), indent=2), flush=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlay C/EFORK trajectories under h/Lm/t_final changes.")
    parser.add_argument("--job", choices=["launch", "candidate", "aggregate"], default="launch")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--source-dir", default=str(SOURCE_DIR))
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--q", type=float, default=0.9998)
    parser.add_argument("--divergence-norm", type=float, default=120.0)
    parser.add_argument("--equilibrium-tol", type=float, default=1e-3)
    parser.add_argument("--max-store-points", type=int, default=6000)
    parser.add_argument("--max-metric-points", type=int, default=1000)
    parser.add_argument("--max-section-points", type=int, default=300)
    parser.add_argument("--tail-fraction-start", type=float, default=0.5)
    parser.add_argument("--wait", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    outdir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUTPUT_ROOT / f"robustness_overlay_c_trajectories_{timestamp()}"
    if args.job == "launch":
        launch(outdir, args)
    elif args.job == "candidate":
        run_candidate(outdir, int(args.candidate_index))
    elif args.job == "aggregate":
        aggregate(outdir, bool(args.wait))


if __name__ == "__main__":
    main()
