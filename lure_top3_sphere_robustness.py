#!/usr/bin/env python3
"""Sphere-neighborhood tests and robustness checks for the top Lur'e candidates.

Mathematical purpose:
    Probe whether trajectories initialized on spheres centered at the Chua
    equilibria enter the observed target-attractor class.  A TARGET hit from an
    equilibrium neighborhood is evidence against hiddenness support.

Numerical model:
    Caputo EFORK with finite memory, evaluated through the repository C basin
    classifier.  This keeps the hardware-realistic memory contract explicit.

Validity warning:
    The target class used here is the same operational EFORK basin class used in
    the project basin plots: positive/negative nontrivial bounded target class
    from the Chua piecewise system.  It is a numerical classification, not a
    proof of hiddenness or non-hiddenness.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from chua_basin_comparison_h001 import CLASS_LABELS, load_project_basin_library
from extended_search_utils import json_safe
from parallel_policy import force_single_openmp_thread_current_process, force_single_openmp_thread_env


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = ROOT / "outputs" / "lure_biased_multiparam_q09998_20260515_195444"
DEFAULT_TRAJ_DIR = ROOT / "outputs" / "lure_biased_multiparam_q09998" / "trajectories"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs"

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


def sphere_fields() -> List[str]:
    return RAW_FIELDS + ["route", "mu", "theta", "target_class_id", "target_class_label"]


def robust_raw_fields() -> List[str]:
    return ROBUST_RAW_FIELDS + ["route", "mu", "theta", "target_class_id", "target_class_label"]


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


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fields = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv(path: Path, row: Dict[str, Any], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_lure_survivor(source_dir: Path, candidate_id: str) -> Dict[str, Any]:
    candidates = read_csv_rows(source_dir / "biased_lure_candidates.csv")
    survivors = read_csv_rows(source_dir / "continuation_survivors.csv")
    survivor_by_id = {row["candidate_id"]: row for row in survivors}
    for row in candidates:
        cid = row.get("candidate_id", "")
        if cid != candidate_id or cid not in survivor_by_id:
            continue
        rank = int(cid.rsplit("_", 1)[-1])
        return {
            "candidate_id": cid,
            "candidate_rank": rank,
            "route": "Lure_rank_0001",
            "q": as_float(row.get("q")),
            "mu": "",
            "theta": "",
            "A": as_float(row.get("A")),
            "sigma0": as_float(row.get("sigma0")),
            "omega": as_float(row.get("omega")),
            "rho_H": as_float(row.get("rho_H")),
            "residual_abs": as_float(row.get("residual_abs")),
            "seed": [
                as_float(row.get("seed_x")),
                as_float(row.get("seed_y")),
                as_float(row.get("seed_z")),
            ],
            "robust_start": [
                as_float(survivor_by_id[cid].get("final_x")),
                as_float(survivor_by_id[cid].get("final_y")),
                as_float(survivor_by_id[cid].get("final_z")),
            ],
            "survivor_final_class": survivor_by_id[cid].get("final_class", ""),
            "survivor_source": str(source_dir / "continuation_survivors.csv"),
        }
    raise FileNotFoundError(f"No encontre {candidate_id} en {source_dir}")


def load_machado_candidate(candidate_id: str) -> Dict[str, Any]:
    summary_path = ROOT / "outputs" / "extended_search" / "machado_targeted_verification_lm10_20260515_182252" / "machado_targeted_summary.json"
    data = read_json(summary_path)
    by_id = {row["candidate_id"]: row for row in data.get("reference_attractor", [])}
    corrida1_path = ROOT / "outputs" / "extended_search" / "corrida1" / "corrida1_summary.json"
    corrida1 = read_json(corrida1_path) if corrida1_path.exists() else {}
    cand_rows = {row["candidate_id"]: row for row in corrida1.get("candidates", [])}
    row = by_id.get(candidate_id)
    if row is None:
        raise FileNotFoundError(f"No encontre {candidate_id} en {summary_path}")
    crow = cand_rows.get(candidate_id, {})
    return {
        "candidate_id": candidate_id,
        "candidate_rank": 0 if "mu_4p" in candidate_id else 1,
        "route": "Machado_FDF",
        "q": as_float(row.get("q"), 0.9998),
        "mu": as_float(row.get("mu")),
        "theta": as_float(row.get("theta")),
        "A": as_float(crow.get("A")),
        "sigma0": "",
        "omega": as_float(crow.get("omega")),
        "rho_H": "",
        "residual_abs": "",
        "seed": [
            as_float(crow.get("seed", [float("nan"), float("nan"), float("nan")])[0]) if isinstance(crow.get("seed"), list) else float("nan"),
            as_float(crow.get("seed", [float("nan"), float("nan"), float("nan")])[1]) if isinstance(crow.get("seed"), list) else float("nan"),
            as_float(crow.get("seed", [float("nan"), float("nan"), float("nan")])[2]) if isinstance(crow.get("seed"), list) else float("nan"),
        ],
        "robust_start": [
            as_float(row.get("final_x")),
            as_float(row.get("final_y")),
            as_float(row.get("final_z")),
        ],
        "survivor_final_class": row.get("final_class", ""),
        "survivor_source": str(summary_path),
    }


def load_requested_candidates(source_dir: Path) -> List[Dict[str, Any]]:
    """Load exactly the three candidates requested for this control run."""

    return [
        load_machado_candidate("branch_0_mu_4p00000_theta_0p00000"),
        load_machado_candidate("branch_0_mu_2p00000_theta_3p92699"),
        load_lure_survivor(source_dir, "lure_biased_q_0p99980_rank_0001"),
    ]


def load_equilibria_from_c() -> Dict[str, np.ndarray]:
    lib = load_project_basin_library()
    lib.set_chua_model(0)
    lib.set_chua_params(8.4562, 12.0732, 0.0052, -0.1768, -1.1468)
    lib.get_equilibria.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")]
    lib.get_equilibria.restype = None
    out = np.zeros(9, dtype=np.float64)
    lib.get_equilibria(out)
    return {
        "E0": out[0:3].copy(),
        "E+": out[3:6].copy(),
        "E-": out[6:9].copy(),
    }


def unit_sphere_direction(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / norm


def make_plan(outdir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_dir).resolve()
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
    candidates = load_requested_candidates(source_dir)
    for cand in candidates:
        cand["target_class_id"] = -1
        cand["target_class_label"] = "target_positive_or_negative"
    rng = np.random.default_rng(int(args.seed))
    rows: List[Dict[str, Any]] = []
    index = 0
    for cand in candidates:
        for eq_id, center in eqs.items():
            for radius in radii:
                for sample_id in range(int(args.samples_per_radius)):
                    direction = unit_sphere_direction(rng)
                    x0 = center + radius * direction
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
        "method": "sphere_neighborhood_controls_plus_c_efork_robustness",
        "classification": "target_hit means class_id in {1, 2}, the EFORK nontrivial bounded target class used by the basin backend",
        "candidates": candidates,
        "equilibria": {k: v.tolist() for k, v in eqs.items()},
        "radii": radii,
        "samples_per_radius": int(args.samples_per_radius),
        "cumulative_batches": [100],
        "sphere_contract": sphere_contract,
        "robustness_cases": robustness_cases(args),
        "chunks": int(args.chunks),
        "random_seed": int(args.seed),
    }
    write_csv(outdir / "sphere_plan.csv", rows)
    write_json(outdir / "top3_sphere_robustness_config.json", cfg)
    return cfg


def configure_classifier(lib: Any) -> None:
    lib.set_chua_model(0)
    lib.set_chua_params(8.4562, 12.0732, 0.0052, -0.1768, -1.1468)
    lib.classify_basin_point.argtypes = [
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
    ]
    lib.classify_basin_point.restype = ctypes.c_int


def classify_point(lib: Any, x0: float, y0: float, z0: float, contract: Dict[str, Any]) -> int:
    return int(
        lib.classify_basin_point(
            float(x0),
            float(y0),
            float(z0),
            float(contract["q"]),
            float(contract["h"]),
            float(contract["memory_length"]),
            float(contract["t_final"]),
            float(contract["t_burn"]),
            float(contract["divergence_norm"]),
            float(contract["r_bound"]),
            float(contract["equilibrium_tol"]),
            int(contract["cap_win"]),
            float(contract["mean_x_gap"]),
        )
    )


def run_sphere_chunk(outdir: Path, chunk_id: int, chunks: int) -> Path:
    force_single_openmp_thread_current_process()
    cfg = read_json(outdir / "top3_sphere_robustness_config.json")
    plan = read_csv_rows(outdir / "sphere_plan.csv")
    lib = load_project_basin_library()
    configure_classifier(lib)
    path = outdir / f"sphere_raw_chunk_{chunk_id:03d}.csv"
    if path.exists():
        path.unlink()
    done_path = outdir / f"sphere_raw_chunk_{chunk_id:03d}.done"
    if done_path.exists():
        done_path.unlink()
    rows_done = 0
    for item in plan:
        idx = int(item["case_index"])
        if idx % int(chunks) != int(chunk_id):
            continue
        cid = classify_point(lib, float(item["x0"]), float(item["y0"]), float(item["z0"]), cfg["sphere_contract"])
        target_class_id = int(float(item.get("target_class_id", -999)))
        row = {
            **{k: item.get(k, "") for k in sphere_fields()},
            "h": cfg["sphere_contract"]["h"],
            "memory_length": cfg["sphere_contract"]["memory_length"],
            "t_final": cfg["sphere_contract"]["t_final"],
            "t_burn": cfg["sphere_contract"]["t_burn"],
            "class_id": cid,
            "class_label": CLASS_LABELS.get(cid, f"class_{cid}"),
            "target_hit": bool(cid == target_class_id) if target_class_id in (1, 2) else bool(cid in (1, 2)),
        }
        append_csv(path, row, sphere_fields())
        rows_done += 1
        if rows_done % 50 == 0:
            print(f"sphere chunk {chunk_id}: {rows_done} rows", flush=True)
    done_path.write_text(
        json.dumps({"chunk_id": int(chunk_id), "rows": rows_done, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}),
        encoding="utf-8",
    )
    return path


def robustness_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
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


def run_robustness(outdir: Path) -> Path:
    force_single_openmp_thread_current_process()
    cfg = read_json(outdir / "top3_sphere_robustness_config.json")
    lib = load_project_basin_library()
    configure_classifier(lib)
    raw_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for cand in cfg["candidates"]:
        start = cand["robust_start"]
        for case in cfg["robustness_cases"]:
            cid = classify_point(lib, float(start[0]), float(start[1]), float(start[2]), case)
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
                    "class_label": CLASS_LABELS.get(cid, f"class_{cid}"),
                    "target_class_id": cand.get("target_class_id", ""),
                    "target_class_label": cand.get("target_class_label", ""),
                    "route": cand.get("route", ""),
                    "mu": cand.get("mu", ""),
                    "theta": cand.get("theta", ""),
                    "target_hit": bool(cid == int(cand.get("target_class_id", -999))) if int(cand.get("target_class_id", -999)) in (1, 2) else bool(cid in (1, 2)),
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
    write_csv(outdir / "attractor_robustness_raw.csv", raw_rows, robust_raw_fields())
    write_csv(outdir / "attractor_robustness_summary.csv", summary_rows, ROBUST_SUMMARY_FIELDS)
    (outdir / "attractor_robustness.done").write_text(
        json.dumps({"rows": len(raw_rows), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}),
        encoding="utf-8",
    )
    return outdir / "attractor_robustness_summary.csv"


def aggregate(outdir: Path, wait: bool = False, poll_sec: float = 30.0) -> Path:
    cfg = read_json(outdir / "top3_sphere_robustness_config.json")
    chunks = int(cfg["chunks"])
    while wait:
        if all((outdir / f"sphere_raw_chunk_{idx:03d}.done").exists() for idx in range(chunks)) and (outdir / "attractor_robustness.done").exists():
            break
        time.sleep(float(poll_sec))
    raw_rows: List[Dict[str, str]] = []
    for idx in range(chunks):
        raw_rows.extend(read_csv_rows(outdir / f"sphere_raw_chunk_{idx:03d}.csv"))
    write_csv(outdir / "sphere_raw.csv", raw_rows, sphere_fields())
    summary_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str, float], List[Dict[str, str]]] = defaultdict(list)
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
    write_csv(outdir / "sphere_cumulative_summary.csv", summary_rows, SUMMARY_FIELDS)
    decisions = []
    for cand in cfg["candidates"]:
        rows = [r for r in summary_rows if r["candidate_id"] == cand["candidate_id"] and int(r["cumulative_samples"]) == int(cfg["samples_per_radius"])]
        target_total = sum(int(r["n_target_hits"]) for r in rows)
        smallest = min([float(r["radius"]) for r in rows if int(r["n_target_hits"]) > 0], default=float("nan"))
        decisions.append(
            {
                "candidate_id": cand["candidate_id"],
                "total_sphere_trajectories": sum(int(r["n_executed"]) for r in rows),
                "total_target_hits": target_total,
                "smallest_radius_with_target_hit": "" if math.isnan(smallest) else smallest,
                "hiddenness_status": "not_supported_by_sphere_equilibrium_test" if target_total > 0 else "compatible_with_hiddenness_under_tested_spheres",
                "notes": "Sphere test uses EFORK target class; not a proof either way.",
            }
        )
    write_csv(outdir / "sphere_decision.csv", decisions)
    summary = {
        "status": "ok" if len(raw_rows) == len(read_csv_rows(outdir / "sphere_plan.csv")) else "partial",
        "raw_rows": len(raw_rows),
        "planned_rows": len(read_csv_rows(outdir / "sphere_plan.csv")),
        "sphere_summary_csv": str(outdir / "sphere_cumulative_summary.csv"),
        "sphere_decision_csv": str(outdir / "sphere_decision.csv"),
        "robustness_summary_csv": str(outdir / "attractor_robustness_summary.csv"),
        "decisions": decisions,
    }
    write_json(outdir / "top3_sphere_robustness_summary.json", summary)
    return outdir / "top3_sphere_robustness_summary.json"


def launch(outdir: Path, args: argparse.Namespace) -> Path:
    cfg = make_plan(outdir, args)
    logs = outdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = force_single_openmp_thread_env(os.environ.copy())
    env["PYTHONUNBUFFERED"] = "1"
    launched: List[Dict[str, Any]] = []
    for idx in range(int(args.chunks)):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--job",
            "sphere-chunk",
            "--output-dir",
            str(outdir),
            "--chunk-id",
            str(idx),
            "--chunks",
            str(args.chunks),
        ]
        stdout = (logs / f"sphere_chunk_{idx:03d}.out").open("ab")
        stderr = (logs / f"sphere_chunk_{idx:03d}.err").open("ab")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        launched.append({"job": f"sphere_chunk_{idx:03d}", "pid": proc.pid, "cmd": cmd})
    cmd = [sys.executable, str(Path(__file__).resolve()), "--job", "robustness", "--output-dir", str(outdir)]
    stdout = (logs / "robustness.out").open("ab")
    stderr = (logs / "robustness.err").open("ab")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    launched.append({"job": "robustness", "pid": proc.pid, "cmd": cmd})
    cmd = [sys.executable, str(Path(__file__).resolve()), "--job", "aggregate", "--output-dir", str(outdir), "--wait"]
    stdout = (logs / "aggregate.out").open("ab")
    stderr = (logs / "aggregate.err").open("ab")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    launched.append({"job": "aggregate", "pid": proc.pid, "cmd": cmd})
    manifest = {"output_dir": str(outdir), "chunks": int(cfg["chunks"]), "launched": launched}
    write_json(outdir / "launch_manifest.json", manifest)
    print(json.dumps(json_safe(manifest), indent=2), flush=True)
    return outdir / "launch_manifest.json"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Top-3 Lure sphere controls and robustness.")
    p.add_argument("--job", choices=["launch", "sphere-chunk", "robustness", "aggregate"], default="launch")
    p.add_argument("--output-dir", default="")
    p.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--radii", default="1e-5,3e-5,1e-4,3e-4,1e-3")
    p.add_argument("--samples-per-radius", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260517)
    p.add_argument("--chunks", type=int, default=3)
    p.add_argument("--chunk-id", type=int, default=0)
    p.add_argument("--q", type=float, default=0.9998)
    p.add_argument("--h", type=float, default=0.01)
    p.add_argument("--memory-length", type=float, default=10.0)
    p.add_argument("--t-final", type=float, default=1500.0)
    p.add_argument("--t-burn", type=float, default=100.0)
    p.add_argument("--robust-long-t-final", type=float, default=3000.0)
    p.add_argument("--robust-long-t-burn", type=float, default=200.0)
    p.add_argument("--robust-short-memory", type=float, default=5.0)
    p.add_argument("--robust-refined-h", type=float, default=0.005)
    p.add_argument("--robust-refined-t-final", type=float, default=1000.0)
    p.add_argument("--robust-refined-t-burn", type=float, default=100.0)
    p.add_argument("--divergence-norm", type=float, default=120.0)
    p.add_argument("--r-bound", type=float, default=60.0)
    p.add_argument("--equilibrium-tol", type=float, default=1e-3)
    p.add_argument("--mean-x-gap", type=float, default=0.75)
    p.add_argument("--cap-win", type=int, default=150)
    p.add_argument("--wait", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    outdir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUTPUT_ROOT / f"lure_top3_sphere_robustness_{timestamp()}"
    if args.job == "launch":
        launch(outdir, args)
    elif args.job == "sphere-chunk":
        run_sphere_chunk(outdir, int(args.chunk_id), int(args.chunks))
    elif args.job == "robustness":
        run_robustness(outdir)
    elif args.job == "aggregate":
        aggregate(outdir, wait=bool(args.wait))


if __name__ == "__main__":
    main()
