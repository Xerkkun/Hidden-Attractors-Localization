#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

import chua_initial_cond as chua
from equilibria_analysis import local_jacobian, region_for_sigma, solve_equilibria
from extended_search_utils import fft_peak_and_entropy, min_distance_to_points, trajectory_ranges
from lure_candidate_manifest import (
    DEFAULT_CONFIG,
    ROOT,
    as_bool,
    as_float,
    as_int,
    csv_value,
    json_safe,
    load_config,
    official_chua_params,
    read_csv_rows,
    read_json,
    resolve_path,
)


RAW_FIELDS = [
    "candidate_id",
    "df_family",
    "q",
    "branch_index",
    "A",
    "omega",
    "phase",
    "seed_x",
    "seed_y",
    "seed_z",
    "final_x",
    "final_y",
    "final_z",
    "equilibrium_id",
    "radius",
    "sampling_mode",
    "sample_id",
    "perturbation_dx",
    "perturbation_dy",
    "perturbation_dz",
    "x0",
    "y0",
    "z0",
    "h",
    "memory_length",
    "memory_points",
    "t_final",
    "final_class",
    "target_hit",
    "target_label",
    "final_norm",
    "min_dist_to_equilibria",
    "final_dist_to_Eminus",
    "final_dist_to_E0",
    "final_dist_to_Eplus",
    "range_x",
    "range_y",
    "range_z",
    "mean_x_tail",
    "mean_y_tail",
    "mean_z_tail",
    "var_x_tail",
    "var_y_tail",
    "var_z_tail",
    "fft_peak",
    "psd_entropy",
    "numerical_status",
    "notes",
]

SUMMARY_FIELDS = [
    "candidate_id",
    "equilibrium_id",
    "radius",
    "sampling_mode",
    "h",
    "memory_length",
    "t_final",
    "n_samples",
    "n_equilibrium_convergence",
    "n_target_attractor",
    "n_other_bounded_nontrivial",
    "n_divergent",
    "n_numerical_failure",
    "n_ambiguous_long_transient",
    "target_hit_fraction",
    "robust_target_hit",
    "most_common_class",
]

DECISION_FIELDS = [
    "candidate_id",
    "hiddenness_status",
    "blocking_equilibrium",
    "blocking_radius",
    "blocking_sampling_mode",
    "blocking_h",
    "blocking_memory_length",
    "blocking_t_final",
    "max_target_hit_fraction",
    "total_target_hits",
    "robust_target_hit_found",
    "smallest_radius_with_target_hit",
    "largest_radius_without_target_hit",
    "basin_intersection_open_like",
    "decision_notes",
]

EQ_FIELDS = [
    "eq_id",
    "x",
    "y",
    "z",
    "region",
    "eig_1",
    "eig_2",
    "eig_3",
    "min_arg_margin",
    "matignon_stable",
    "nonsmooth_boundary",
]

LOCAL_BASIN_FIELDS = [
    "candidate_id",
    "equilibrium_id",
    "plane",
    "rho",
    "u",
    "v",
    "x0",
    "y0",
    "z0",
    "final_class",
    "target_hit",
    "final_x",
    "final_y",
    "final_z",
    "notes",
]

CLASS_ORDER = [
    "equilibrium_convergence",
    "target_attractor",
    "other_bounded_nontrivial",
    "divergent",
    "numerical_failure",
    "ambiguous_long_transient",
]

CLASS_COLORS = {
    "equilibrium_convergence": "#111827",
    "target_attractor": "#dc2626",
    "other_bounded_nontrivial": "#2563eb",
    "divergent": "#f59e0b",
    "numerical_failure": "#6b7280",
    "ambiguous_long_transient": "#7c3aed",
}


def manifest_file(cfg: Dict[str, Any]) -> Path:
    return ROOT / str(cfg.get("manifest", {}).get("output_dir", "outputs/lure_route")) / "lure_candidates_manifest.csv"


def rho_file(cfg: Dict[str, Any]) -> Path:
    return ROOT / str(cfg.get("manifest", {}).get("output_dir", "outputs/lure_route")) / "lure_rhoH_diagnostics.csv"


def refined_dir(cfg: Dict[str, Any]) -> Path:
    return ROOT / str(cfg.get("refined", {}).get("output_dir", "outputs/lure_route/refined"))


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k, "")) for k in fields})


def append_csv(path: str | Path, row: Dict[str, Any], fields: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: csv_value(row.get(k, "")) for k in fields})


def memory_points(memory_length: float, h: float) -> int:
    return max(1, int(math.ceil(float(memory_length) / float(h)))) + 1


def bool_from_row(row: Dict[str, Any], key: str) -> bool:
    return bool(as_bool(row.get(key)))


def load_candidates(cfg: Dict[str, Any], candidate_id: str | None, include_contact_strong: bool) -> List[Dict[str, Any]]:
    rows = read_csv_rows(manifest_file(cfg))
    priority = set(str(x) for x in cfg.get("refined", {}).get("priority_classes", []))
    out: List[Dict[str, Any]] = []
    for row in rows:
        cid = str(row.get("candidate_id", ""))
        if candidate_id and cid != candidate_id:
            continue
        should = bool_from_row(row, "should_refine")
        if include_contact_strong and str(row.get("priority_class", "")) == "contact_strong":
            should = True
        if candidate_id:
            should = True
        if not should and str(row.get("priority_class", "")) not in priority:
            continue
        cand = {
            "candidate_id": cid,
            "df_family": row.get("df_family", "lure_classic"),
            "q": as_float(row.get("q")),
            "branch_index": as_int(row.get("branch_index")),
            "A": as_float(row.get("A")),
            "omega": as_float(row.get("omega")),
            "phase": as_float(row.get("phase"), 0.0),
            "seed": np.asarray([as_float(row.get("seed_x")), as_float(row.get("seed_y")), as_float(row.get("seed_z"))], dtype=float),
            "target_seed": np.asarray([as_float(row.get("final_x")), as_float(row.get("final_y")), as_float(row.get("final_z"))], dtype=float),
            "source_hidden_summary_json": row.get("source_hidden_summary_json", ""),
            "source_hidden_check_csv": row.get("source_hidden_check_csv", ""),
            "source_final_attractor_csv": row.get("source_final_attractor_csv", ""),
            "blocking_equilibrium": row.get("blocking_equilibrium", ""),
            "priority_class": row.get("priority_class", ""),
            "manifest_row": row,
        }
        out.append(cand)
    order = {"contact_weak": 0, "no_target_under_sample": 1, "insufficient_data": 2, "contact_strong": 3}
    out.sort(key=lambda c: (order.get(str(c.get("priority_class", "")), 9), str(c["candidate_id"])))
    return out


def equilibrium_rows(cfg: Dict[str, Any], p: Dict[str, Any], outdir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    q_values = []
    try:
        for row in read_csv_rows(manifest_file(cfg)):
            q = as_float(row.get("q"))
            if math.isfinite(q):
                q_values.append(q)
    except Exception:
        pass
    q = q_values[0] if q_values else float(cfg.get("q", 0.9998))
    theta = float(q) * math.pi / 2.0
    eqs = solve_equilibria(p)
    rows: List[Dict[str, Any]] = []
    eigvecs: Dict[str, np.ndarray] = {}
    order = {"E-": 0, "E0": 1, "E+": 2}
    for eq_id, eq in sorted(eqs.items(), key=lambda kv: order.get(kv[0], 9)):
        eq = np.asarray(eq, dtype=float)
        if not np.all(np.isfinite(eq)):
            continue
        J = local_jacobian(p, eq)
        vals, vecs = np.linalg.eig(J)
        eigvecs[eq_id] = vecs
        margins = [abs(np.angle(v)) - theta for v in vals]
        rows.append({
            "eq_id": eq_id,
            "x": float(eq[0]),
            "y": float(eq[1]),
            "z": float(eq[2]),
            "region": region_for_sigma(float(eq[0])),
            "eig_1": complex(vals[0]),
            "eig_2": complex(vals[1]),
            "eig_3": complex(vals[2]),
            "min_arg_margin": float(min(margins)),
            "matignon_stable": bool(all(m > 0.0 for m in margins)),
            "nonsmooth_boundary": bool(region_for_sigma(float(eq[0])) == "switching_boundary"),
        })
    write_csv(outdir / "equilibria_lure_summary.csv", rows, EQ_FIELDS)
    return rows, {k: np.asarray(v, dtype=float) for k, v in eqs.items()}, eigvecs


def normalize_vec(v: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or n < 1e-14:
        return None
    return arr / n


def deterministic_directions(eigvecs: np.ndarray | None = None) -> List[Tuple[str, np.ndarray]]:
    dirs: List[Tuple[str, np.ndarray]] = []
    basis = np.eye(3, dtype=float)
    for label, vec in zip(["x", "y", "z"], basis):
        dirs.append((f"axis_p_{label}", vec.copy()))
        dirs.append((f"axis_m_{label}", -vec.copy()))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                dirs.append((f"diag_{int(sx):+d}_{int(sy):+d}_{int(sz):+d}", np.array([sx, sy, sz], dtype=float) / math.sqrt(3.0)))
    if eigvecs is not None:
        for idx in range(eigvecs.shape[1]):
            raw = eigvecs[:, idx]
            for part_name, part in [("re", np.real(raw)), ("im", np.imag(raw))]:
                unit = normalize_vec(part)
                if unit is None:
                    continue
                dirs.append((f"eig_{idx}_{part_name}_p", unit))
                dirs.append((f"eig_{idx}_{part_name}_m", -unit))
    return dirs


def random_unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    n = float(np.linalg.norm(v))
    if n < 1e-14:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n


def offsets_for_mode(
    mode: str,
    count: int,
    rng: np.random.Generator,
    eigvecs: np.ndarray | None,
    previous_dirs: Sequence[np.ndarray],
) -> List[Tuple[str, np.ndarray]]:
    mode = str(mode)
    offsets: List[Tuple[str, np.ndarray]] = []
    if mode in {"ball", "sphere_shell", "eigen_directions"}:
        offsets.extend(deterministic_directions(eigvecs))
    elif mode == "cone_around_previous_target_hits":
        for idx, base in enumerate(previous_dirs):
            unit = normalize_vec(np.asarray(base, dtype=float))
            if unit is not None:
                offsets.append((f"cone_seed_{idx:04d}", unit))
    while len(offsets) < int(count):
        if mode == "cone_around_previous_target_hits" and previous_dirs:
            base = normalize_vec(previous_dirs[int(rng.integers(0, len(previous_dirs)))])
            if base is None:
                unit = random_unit(rng)
            else:
                unit = normalize_vec(base + 0.18 * random_unit(rng))
                unit = random_unit(rng) if unit is None else unit
        else:
            unit = random_unit(rng)
        if mode == "ball":
            unit = unit * float(rng.random() ** (1.0 / 3.0))
        offsets.append((f"rand_{len(offsets):06d}", unit))
    return offsets[: int(count)]


def previous_target_dirs(raw_rows: Sequence[Dict[str, Any]], candidate_id: str, eq_id: str) -> List[np.ndarray]:
    dirs: List[np.ndarray] = []
    for row in raw_rows:
        if row.get("candidate_id") != candidate_id or row.get("equilibrium_id") != eq_id:
            continue
        if not bool_from_row(row, "target_hit"):
            continue
        v = np.asarray([as_float(row.get("perturbation_dx")), as_float(row.get("perturbation_dy")), as_float(row.get("perturbation_dz"))], dtype=float)
        unit = normalize_vec(v)
        if unit is not None:
            dirs.append(unit)
    return dirs


def build_plan(
    cfg: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    eqs: Dict[str, np.ndarray],
    eigvecs: Dict[str, np.ndarray],
    raw_rows: Sequence[Dict[str, Any]],
    stage_name: str,
) -> List[Dict[str, Any]]:
    refined = cfg["refined"]
    stage = refined["stages"][stage_name]
    radii = [float(r) for r in refined.get("radii", [])]
    base_modes = list(refined.get("sampling_modes", ["ball", "sphere_shell", "eigen_directions"]))
    count = int(refined.get("samples_per_radius", 500))
    seed = int(refined.get("random_seed", 20260513))
    plan: List[Dict[str, Any]] = []
    eq_order_default = {"E-": 0, "E0": 1, "E+": 2}
    for cand_idx, cand in enumerate(candidates):
        eq_ids = list(eqs.keys())
        blocking = str(cand.get("blocking_equilibrium", ""))
        eq_ids.sort(key=lambda eq: (0 if eq == blocking else 1, eq_order_default.get(eq, 9)))
        for eq_idx, eq_id in enumerate(eq_ids):
            prev_dirs = previous_target_dirs(raw_rows, cand["candidate_id"], eq_id)
            modes = list(base_modes)
            if prev_dirs and "cone_around_previous_target_hits" not in modes and stage_name in {"B", "C"}:
                modes.append("cone_around_previous_target_hits")
            for radius_idx, radius in enumerate(radii):
                for mode_idx, mode in enumerate(modes):
                    if mode == "cone_around_previous_target_hits" and not prev_dirs:
                        continue
                    rng = np.random.default_rng(seed + 1000003 * cand_idx + 9176 * eq_idx + 313 * radius_idx + 17 * mode_idx + {"A": 0, "B": 20000000, "C": 30000000}[stage_name])
                    offsets = offsets_for_mode(mode, count, rng, eigvecs.get(eq_id), prev_dirs)
                    center = np.asarray(eqs[eq_id], dtype=float)
                    for sample_id, unit in offsets:
                        perturb = float(radius) * np.asarray(unit, dtype=float)
                        plan.append({
                            "candidate": cand,
                            "candidate_id": cand["candidate_id"],
                            "equilibrium_id": eq_id,
                            "equilibrium": center,
                            "radius": float(radius),
                            "sampling_mode": mode,
                            "sample_id": sample_id,
                            "perturbation": perturb,
                            "x0": center + perturb,
                            "h": float(stage["h"]),
                            "memory_length": float(stage["memory_length"]),
                            "t_final": float(stage["t_final"]),
                            "stage": stage_name,
                        })
    if stage_name in {"B", "C"}:
        summary = aggregate_raw(raw_rows, cfg)
        trigger = {
            (r["candidate_id"], r["equilibrium_id"], float(r["radius"]), r["sampling_mode"])
            for r in summary
            if int(r.get("n_target_attractor", 0)) > 0 and (stage_name == "B" or bool_from_row(r, "robust_target_hit"))
        }
        plan = [p for p in plan if (p["candidate_id"], p["equilibrium_id"], float(p["radius"]), p["sampling_mode"]) in trigger or p["sampling_mode"] == "cone_around_previous_target_hits"]
    return plan


def processed_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str, str, str, str]:
    return (
        str(row.get("candidate_id", "")),
        str(row.get("equilibrium_id", "")),
        f"{as_float(row.get('radius')):.17g}",
        str(row.get("sampling_mode", "")),
        str(row.get("sample_id", "")),
        f"{as_float(row.get('h')):.17g}",
        f"{as_float(row.get('memory_length')):.17g}",
        f"{as_float(row.get('t_final')):.17g}",
    )


def processed_key_from_plan(item: Dict[str, Any]) -> Tuple[str, str, str, str, str, str, str, str]:
    return (
        str(item["candidate_id"]),
        str(item["equilibrium_id"]),
        f"{float(item['radius']):.17g}",
        str(item["sampling_mode"]),
        str(item["sample_id"]),
        f"{float(item['h']):.17g}",
        f"{float(item['memory_length']):.17g}",
        f"{float(item['t_final']):.17g}",
    )


def chua_xdot(x: np.ndarray, p: Dict[str, Any]) -> float:
    try:
        return float(chua.rhs_original(np.asarray(x, dtype=float), p)[0])
    except Exception:
        return float("nan")


def section_points(traj: np.ndarray, p: Dict[str, Any], t_burn: float, max_points: int) -> np.ndarray:
    X = np.asarray(traj, dtype=float)
    if X.ndim != 2 or X.shape[1] < 4 or X.shape[0] < 2:
        return np.empty((0, 2), dtype=float)
    pts: List[Tuple[float, float]] = []
    for k in range(1, X.shape[0]):
        if X[k, 0] < float(t_burn):
            continue
        xp = X[k - 1, 1]
        x = X[k, 1]
        if xp < 0.0 <= x and chua_xdot(X[k, 1:4], p) > 0.0:
            lam = (0.0 - xp) / ((x - xp) + 1e-300)
            y = X[k - 1, 2] + lam * (X[k, 2] - X[k - 1, 2])
            z = X[k - 1, 3] + lam * (X[k, 3] - X[k - 1, 3])
            pts.append((float(y), float(z)))
            if len(pts) >= int(max_points):
                break
    return np.asarray(pts, dtype=float)


def load_reference_csv(path_value: Any) -> np.ndarray:
    path = resolve_path(path_value, ROOT)
    if not path.exists():
        return np.empty((0, 2), dtype=float)
    rows: List[Tuple[float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "y" in row and "z" in row:
                rows.append((as_float(row["y"]), as_float(row["z"])))
    arr = np.asarray(rows, dtype=float)
    if arr.ndim == 2:
        arr = arr[np.all(np.isfinite(arr), axis=1)]
    return arr


def reference_section(cand: Dict[str, Any], p: Dict[str, Any], h: float, memory_length: float, t_final: float, cfg: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    summary_path = cand.get("source_hidden_summary_json", "")
    if summary_path:
        try:
            hs = read_json(resolve_path(summary_path, ROOT))
            ref_csv = hs.get("files", {}).get("ref_csv_out", "")
            ref = load_reference_csv(ref_csv)
            if ref.shape[0] >= int(cfg.get("classification", {}).get("min_section_matches", 20)):
                return ref, "source_hidden_reference_csv"
        except Exception:
            pass
    x0 = np.asarray(cand["target_seed"], dtype=float)
    if not np.all(np.isfinite(x0)):
        return np.empty((0, 2), dtype=float), "missing_target_seed"
    rhs = lambda x: chua.rhs_original(x, p)
    traj = chua.efork3_integrate(rhs, x0, qord=float(cand["q"]), h=float(h), Lm=float(memory_length), t_f=float(t_final))
    ref = section_points(traj, p, 0.5 * float(t_final), 240)
    return ref, "integrated_target_seed"


def section_hit_fraction(section: np.ndarray, ref: np.ndarray, tol: float) -> Tuple[int, int, float]:
    if section.size == 0 or ref.size == 0:
        total = int(section.shape[0]) if section.ndim == 2 else 0
        return total, 0, 0.0
    hits = 0
    for point in section:
        d = np.linalg.norm(ref - point.reshape(1, 2), axis=1)
        if float(np.min(d)) <= float(tol):
            hits += 1
    total = int(section.shape[0])
    return total, hits, float(hits / max(total, 1))


def tail_stats(traj: np.ndarray, h: float) -> Dict[str, float]:
    X = np.asarray(traj, dtype=float)
    states = X[:, 1:4] if X.ndim == 2 and X.shape[1] >= 4 else np.empty((0, 3), dtype=float)
    if states.shape[0] == 0:
        return {k: float("nan") for k in ["mean_x_tail", "mean_y_tail", "mean_z_tail", "var_x_tail", "var_y_tail", "var_z_tail", "fft_peak", "psd_entropy"]}
    start = max(0, int(0.8 * states.shape[0]))
    tail = states[start:, :]
    mean = np.mean(tail, axis=0)
    var = np.var(tail, axis=0)
    out = {
        "mean_x_tail": float(mean[0]),
        "mean_y_tail": float(mean[1]),
        "mean_z_tail": float(mean[2]),
        "var_x_tail": float(var[0]),
        "var_y_tail": float(var[1]),
        "var_z_tail": float(var[2]),
    }
    out.update(fft_peak_and_entropy(np.column_stack((X[start:, 0], tail)), h, component=0))
    return out


def classify_trajectory(item: Dict[str, Any], p: Dict[str, Any], eqs: Dict[str, np.ndarray], cfg: Dict[str, Any], ref_cache: Dict[Tuple[str, float, float, float], Tuple[np.ndarray, str]]) -> Dict[str, Any]:
    cand = item["candidate"]
    h = float(item["h"])
    memory_length = float(item["memory_length"])
    t_final = float(item["t_final"])
    q = float(cand["q"])
    x0 = np.asarray(item["x0"], dtype=float)
    notes = [f"stage={item.get('stage', '')}"]
    base = {
        "candidate_id": cand["candidate_id"],
        "df_family": cand["df_family"],
        "q": q,
        "branch_index": cand["branch_index"],
        "A": cand["A"],
        "omega": cand["omega"],
        "phase": cand["phase"],
        "seed_x": float(cand["seed"][0]),
        "seed_y": float(cand["seed"][1]),
        "seed_z": float(cand["seed"][2]),
        "final_x": float(cand["target_seed"][0]),
        "final_y": float(cand["target_seed"][1]),
        "final_z": float(cand["target_seed"][2]),
        "equilibrium_id": item["equilibrium_id"],
        "radius": float(item["radius"]),
        "sampling_mode": item["sampling_mode"],
        "sample_id": item["sample_id"],
        "perturbation_dx": float(item["perturbation"][0]),
        "perturbation_dy": float(item["perturbation"][1]),
        "perturbation_dz": float(item["perturbation"][2]),
        "x0": float(x0[0]),
        "y0": float(x0[1]),
        "z0": float(x0[2]),
        "h": h,
        "memory_length": memory_length,
        "memory_points": memory_points(memory_length, h),
        "t_final": t_final,
        "target_label": "TARGET",
    }
    try:
        rhs = lambda x: chua.rhs_original(x, p)
        traj = chua.efork3_integrate(rhs, x0, qord=q, h=h, Lm=memory_length, t_f=t_final)
    except Exception as exc:
        return {**base, "final_class": "numerical_failure", "target_hit": False, "numerical_status": "integration_exception", "notes": str(exc)}
    states = traj[:, 1:4] if traj.ndim == 2 and traj.shape[1] >= 4 else np.empty((0, 3))
    if states.size == 0 or not np.all(np.isfinite(states)):
        return {**base, "final_class": "numerical_failure", "target_hit": False, "numerical_status": "nonfinite_trajectory", "notes": "nonfinite trajectory"}
    final = states[-1]
    final_norm = float(np.linalg.norm(final))
    max_norm = float(np.max(np.linalg.norm(states, axis=1)))
    eq_arrays = {k: np.asarray(v, dtype=float) for k, v in eqs.items() if np.all(np.isfinite(v))}
    final_dist = {k: float(np.linalg.norm(final - v)) for k, v in eq_arrays.items()}
    ranges = trajectory_ranges(traj)
    tail = tail_stats(traj, h)
    final_class = "ambiguous_long_transient"
    target_hit = False
    numerical_status = "ok"
    divergence = float(cfg.get("classification", {}).get("divergence_norm", 1.0e5))
    if final_norm > divergence or max_norm > divergence:
        final_class = "divergent"
    else:
        eq_radius = float(cfg.get("classification", {}).get("equilibrium_radius", 1e-3))
        nearest = min(final_dist.items(), key=lambda kv: kv[1]) if final_dist else ("", float("inf"))
        tail_mean = np.asarray([tail["mean_x_tail"], tail["mean_y_tail"], tail["mean_z_tail"]], dtype=float)
        tail_dist = float(np.linalg.norm(tail_mean - eq_arrays[nearest[0]])) if nearest[0] in eq_arrays else float("inf")
        if nearest[1] <= eq_radius and tail_dist <= 2.0 * eq_radius:
            final_class = "equilibrium_convergence"
            notes.append(f"converged_to={nearest[0]}")
        else:
            key = (cand["candidate_id"], h, memory_length, t_final)
            if key not in ref_cache:
                try:
                    ref_cache[key] = reference_section(cand, p, h, memory_length, t_final, cfg)
                except Exception as exc:
                    ref_cache[key] = (np.empty((0, 2), dtype=float), f"reference_failed={exc}")
            ref, ref_source = ref_cache[key]
            notes.append(f"reference={ref_source};reference_points={ref.shape[0]}")
            sec = section_points(traj, p, 0.5 * t_final, 100)
            total, hits, frac = section_hit_fraction(sec, ref, 0.12)
            notes.append(f"sec_total={total};sec_hits={hits};hit_frac={frac:.6g}")
            if total >= int(cfg.get("classification", {}).get("min_section_matches", 20)) and frac >= float(cfg.get("classification", {}).get("target_hit_fraction_required", 0.70)):
                final_class = "target_attractor"
                target_hit = True
            elif total < int(cfg.get("classification", {}).get("min_section_matches", 20)):
                final_class = "ambiguous_long_transient"
            else:
                variance = max(float(tail["var_x_tail"]), float(tail["var_y_tail"]), float(tail["var_z_tail"]))
                range_max = max(float(ranges["range_x"]), float(ranges["range_y"]), float(ranges["range_z"]))
                final_class = "other_bounded_nontrivial" if variance > 1e-6 or range_max > 1e-2 else "ambiguous_long_transient"
    return {
        **base,
        "final_x": float(final[0]),
        "final_y": float(final[1]),
        "final_z": float(final[2]),
        "final_norm": final_norm,
        "min_dist_to_equilibria": min_distance_to_points(states, eq_arrays.values()),
        "final_dist_to_Eminus": final_dist.get("E-", float("nan")),
        "final_dist_to_E0": final_dist.get("E0", float("nan")),
        "final_dist_to_Eplus": final_dist.get("E+", float("nan")),
        **ranges,
        **tail,
        "final_class": final_class,
        "target_hit": bool(target_hit),
        "numerical_status": numerical_status,
        "notes": ";".join(notes),
    }


def group_key(row: Dict[str, Any]) -> Tuple[str, str, float, str, float, float, float]:
    return (
        str(row.get("candidate_id", "")),
        str(row.get("equilibrium_id", "")),
        as_float(row.get("radius")),
        str(row.get("sampling_mode", "")),
        as_float(row.get("h")),
        as_float(row.get("memory_length")),
        as_float(row.get("t_final")),
    )


def aggregate_raw(raw_rows: Sequence[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, float, str, float, float, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        groups[group_key(row)].append(row)
    summary: List[Dict[str, Any]] = []
    for key, rows in groups.items():
        classes = Counter(str(r.get("final_class", "")) for r in rows)
        n = len(rows)
        n_target = classes.get("target_attractor", 0)
        summary.append({
            "candidate_id": key[0],
            "equilibrium_id": key[1],
            "radius": key[2],
            "sampling_mode": key[3],
            "h": key[4],
            "memory_length": key[5],
            "t_final": key[6],
            "n_samples": n,
            "n_equilibrium_convergence": classes.get("equilibrium_convergence", 0),
            "n_target_attractor": n_target,
            "n_other_bounded_nontrivial": classes.get("other_bounded_nontrivial", 0),
            "n_divergent": classes.get("divergent", 0),
            "n_numerical_failure": classes.get("numerical_failure", 0),
            "n_ambiguous_long_transient": classes.get("ambiguous_long_transient", 0),
            "target_hit_fraction": float(n_target / max(n, 1)),
            "robust_target_hit": False,
            "most_common_class": classes.most_common(1)[0][0] if classes else "",
        })
    mark_robust(summary, cfg)
    summary.sort(key=lambda r: (r["candidate_id"], r["equilibrium_id"], float(r["radius"]), r["sampling_mode"], float(r["h"]), float(r["memory_length"])))
    return summary


def neighboring_radii(r1: float, r2: float, radii: Sequence[float]) -> bool:
    ordered = sorted(float(r) for r in radii)
    try:
        i = ordered.index(float(r1))
        j = ordered.index(float(r2))
        return abs(i - j) <= 1
    except ValueError:
        return math.isclose(float(r1), float(r2), rel_tol=0.0, abs_tol=1e-18)


def mark_robust(summary: List[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    min_count = int(cfg.get("classification", {}).get("robust_hit_min_count", 3))
    radii = [float(r) for r in cfg.get("refined", {}).get("radii", [])]
    for row in summary:
        if int(row.get("n_target_attractor", 0)) >= min_count:
            row["robust_target_hit"] = True
        if int(row.get("n_target_attractor", 0)) > 0 and str(row.get("sampling_mode")) == "cone_around_previous_target_hits":
            row["robust_target_hit"] = True

    hit_rows = [r for r in summary if int(r.get("n_target_attractor", 0)) > 0]
    for row in hit_rows:
        for other in hit_rows:
            if row is other:
                continue
            if row["candidate_id"] != other["candidate_id"] or row["equilibrium_id"] != other["equilibrium_id"]:
                continue
            if row["sampling_mode"] == other["sampling_mode"] and row["radius"] == other["radius"]:
                if (float(row["h"]), float(other["h"])) in {(0.01, 0.005), (0.005, 0.01)}:
                    row["robust_target_hit"] = True
                    other["robust_target_hit"] = True
                if float(row["memory_length"]) != float(other["memory_length"]):
                    row["robust_target_hit"] = True
                    other["robust_target_hit"] = True
            modes = {str(row["sampling_mode"]), str(other["sampling_mode"])}
            if {"ball", "sphere_shell"}.issubset(modes) and neighboring_radii(float(row["radius"]), float(other["radius"]), radii):
                row["robust_target_hit"] = True
                other["robust_target_hit"] = True


def load_local_basin_open_flags(outdir: Path) -> Dict[str, bool]:
    path = outdir / "local_basin_summary.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    flags: Dict[str, bool] = {}
    for row in data.get("rows", []):
        cid = str(row.get("candidate_id", ""))
        flags[cid] = flags.get(cid, False) or bool(row.get("basin_intersection_open_like", False))
    return flags


def decide(candidates: Sequence[Dict[str, Any]], summary: Sequence[Dict[str, Any]], raw_rows: Sequence[Dict[str, Any]], outdir: Path) -> List[Dict[str, Any]]:
    by_cid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in summary:
        by_cid[str(row["candidate_id"])].append(row)
    raw_by_cid = Counter(str(row.get("candidate_id", "")) for row in raw_rows)
    local_flags = load_local_basin_open_flags(outdir)
    decisions: List[Dict[str, Any]] = []
    for cand in candidates:
        cid = cand["candidate_id"]
        rows = by_cid.get(cid, [])
        hit_rows = [r for r in rows if int(r.get("n_target_attractor", 0)) > 0]
        robust = [r for r in rows if bool_from_row(r, "robust_target_hit")]
        no_hit_rows = [r for r in rows if int(r.get("n_target_attractor", 0)) == 0 and int(r.get("n_samples", 0)) > 0]
        total_hits = sum(int(r.get("n_target_attractor", 0)) for r in rows)
        max_frac = max([as_float(r.get("target_hit_fraction"), 0.0) for r in rows], default=0.0)
        open_like = bool(local_flags.get(cid, False))
        if str(cand.get("priority_class", "")) == "invalid_or_divergent":
            status = "invalid_or_divergent"
            notes = "Candidate was already marked invalid or divergent in the manifest."
            block = {}
        elif open_like:
            status = "not_supported_by_refined_neighborhood_test"
            notes = "A local basin map found an open-like target intersection."
            block = {}
        elif robust:
            status = "not_supported_by_refined_neighborhood_test"
            notes = "A robust target hit was found from an equilibrium neighborhood."
            block = sorted(robust, key=lambda r: (float(r["radius"]), str(r["equilibrium_id"])))[0]
        elif total_hits > 0:
            status = "inconclusive_isolated_hit"
            notes = "Only isolated target hits were found in completed groups."
            block = sorted(hit_rows, key=lambda r: (float(r["radius"]), str(r["equilibrium_id"])))[0]
        elif raw_by_cid.get(cid, 0) > 0:
            status = "compatible_with_hiddenness_under_tested_radii"
            notes = "No target hits were found. This is not hidden_verified."
            block = {}
        else:
            status = "not_evaluated"
            notes = "No refined trajectories have been executed for this candidate."
            block = {}
        smallest = min([float(r["radius"]) for r in hit_rows], default=float("nan"))
        largest_without = max([float(r["radius"]) for r in no_hit_rows], default=float("nan"))
        decisions.append({
            "candidate_id": cid,
            "hiddenness_status": status,
            "blocking_equilibrium": block.get("equilibrium_id", cand.get("blocking_equilibrium", "")),
            "blocking_radius": block.get("radius", ""),
            "blocking_sampling_mode": block.get("sampling_mode", ""),
            "blocking_h": block.get("h", ""),
            "blocking_memory_length": block.get("memory_length", ""),
            "blocking_t_final": block.get("t_final", ""),
            "max_target_hit_fraction": max_frac,
            "total_target_hits": total_hits,
            "robust_target_hit_found": bool(robust) or open_like,
            "smallest_radius_with_target_hit": "" if math.isnan(smallest) else smallest,
            "largest_radius_without_target_hit": "" if math.isnan(largest_without) else largest_without,
            "basin_intersection_open_like": open_like,
            "decision_notes": notes,
        })
    return decisions


def plot_outputs(outdir: Path, candidates: Sequence[Dict[str, Any]], raw_rows: Sequence[Dict[str, Any]], summary: Sequence[Dict[str, Any]]) -> List[str]:
    plotdir = outdir / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)
    files: List[str] = []
    for cand in candidates:
        cid = cand["candidate_id"]
        for eq in sorted({r.get("equilibrium_id") for r in summary if r.get("candidate_id") == cid}):
            rows = sorted([r for r in summary if r.get("candidate_id") == cid and r.get("equilibrium_id") == eq], key=lambda r: (str(r["sampling_mode"]), float(r["radius"])))
            if not rows:
                continue
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            for mode in sorted({str(r["sampling_mode"]) for r in rows}):
                sub = [r for r in rows if str(r["sampling_mode"]) == mode]
                ax.plot([float(r["radius"]) for r in sub], [int(r["n_target_attractor"]) for r in sub], marker="o", label=mode)
            ax.set_xscale("log")
            ax.set_xlabel("radius")
            ax.set_ylabel("target hits")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7)
            fig.tight_layout()
            safe = cid.replace("/", "_").replace("\\", "_")
            eqsafe = str(eq).replace("+", "plus").replace("-", "minus")
            path = plotdir / f"lure_target_hits_vs_radius_{safe}_{eqsafe}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            files.append(str(path))

        rows = [r for r in summary if r.get("candidate_id") == cid]
        if rows:
            by_radius: Dict[float, Counter[str]] = defaultdict(Counter)
            totals: Counter[float] = Counter()
            for r in rows:
                radius = float(r["radius"])
                totals[radius] += int(r["n_samples"])
                for cls in CLASS_ORDER:
                    by_radius[radius][cls] += int(r.get("n_" + cls, 0))
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            xs = sorted(by_radius)
            for cls in CLASS_ORDER:
                ax.plot(xs, [by_radius[x][cls] / max(totals[x], 1) for x in xs], marker="o", label=cls, color=CLASS_COLORS.get(cls))
            ax.set_xscale("log")
            ax.set_xlabel("radius")
            ax.set_ylabel("class fraction")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=6)
            fig.tight_layout()
            safe = cid.replace("/", "_").replace("\\", "_")
            path = plotdir / f"lure_class_fraction_vs_radius_{safe}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            files.append(str(path))

        raw = [r for r in raw_rows if r.get("candidate_id") == cid]
        if raw:
            fig, ax = plt.subplots(figsize=(5.0, 5.0))
            for cls in CLASS_ORDER:
                sub = [r for r in raw if r.get("final_class") == cls]
                if sub:
                    ax.scatter([as_float(r["x0"]) for r in sub], [as_float(r["y0"]) for r in sub], s=8, alpha=0.7, label=cls, color=CLASS_COLORS.get(cls))
            ax.set_xlabel("x0")
            ax.set_ylabel("y0")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=6)
            fig.tight_layout()
            safe = cid.replace("/", "_").replace("\\", "_")
            eq = str(cand.get("blocking_equilibrium") or "all").replace("+", "plus").replace("-", "minus")
            path = plotdir / f"lure_local_basin_{safe}_{eq}_xy.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            files.append(str(path))

        attr_path = resolve_path(cand.get("source_final_attractor_csv", ""), ROOT)
        if attr_path.exists():
            try:
                arr = np.loadtxt(attr_path, delimiter=",", skiprows=1)
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.shape[1] >= 4:
                    step = max(1, arr.shape[0] // 5000)
                    A = arr[::step, :]
                    fig = plt.figure(figsize=(6.0, 5.0))
                    ax = fig.add_subplot(111, projection="3d")
                    ax.plot(A[:, 1], A[:, 2], A[:, 3], lw=0.6)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("z")
                    fig.tight_layout()
                    safe = cid.replace("/", "_").replace("\\", "_")
                    path = plotdir / f"lure_candidate_attractor_{safe}.png"
                    fig.savefig(path, dpi=180)
                    plt.close(fig)
                    files.append(str(path))
            except Exception:
                pass
    return files


def plane_basis(eq_id: str, plane: str, eigvecs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if plane == "xy":
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    if plane == "xz":
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    if plane == "switching_tangent":
        return np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    vecs = eigvecs.get(eq_id)
    if vecs is not None and vecs.shape[1] >= 2:
        e1 = normalize_vec(np.real(vecs[:, 0]))
        e2 = normalize_vec(np.real(vecs[:, 1]))
        if e1 is not None and e2 is not None:
            return e1, e2
    return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])


def open_like_target(rows: Sequence[Dict[str, Any]], grid: int) -> bool:
    hits = {(as_int(r.get("_iu")), as_int(r.get("_iv"))) for r in rows if bool(r.get("target_hit"))}
    if len(hits) >= max(4, int(0.0025 * grid * grid)):
        return True
    for i, j in hits:
        if (i + 1, j) in hits and (i, j + 1) in hits:
            return True
    return False


def run_local_basin(
    cfg: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
    eigvecs: Dict[str, np.ndarray],
    outdir: Path,
    grid_mode: str,
    max_points: int,
) -> List[Dict[str, Any]]:
    lb = cfg.get("local_basin", {})
    grid = int(lb.get("grid_final" if grid_mode == "final" else "grid_fast", 201))
    planes = list(lb.get("planes", ["xy", "xz", "eigenplane"]))
    rhos = [float(r) for r in lb.get("rhos", [1e-4, 1e-3, 1e-2])]
    stage = cfg["refined"]["stages"]["A"]
    rows_summary: List[Dict[str, Any]] = []
    ref_cache: Dict[Tuple[str, float, float, float], Tuple[np.ndarray, str]] = {}
    for cand in candidates:
        if cand.get("priority_class") != "contact_weak":
            continue
        eq_id = str(cand.get("blocking_equilibrium") or "E-")
        if eq_id not in eqs:
            continue
        center = np.asarray(eqs[eq_id], dtype=float)
        for plane in planes:
            e1, e2 = plane_basis(eq_id, plane, eigvecs)
            for rho in rhos:
                csv_path = outdir / f"local_basin_{cand['candidate_id']}_{eq_id}_{plane}_rho_{rho:.0e}.csv"
                png_path = outdir / f"local_basin_{cand['candidate_id']}_{eq_id}_{plane}_rho_{rho:.0e}.png"
                vals = np.linspace(-rho, rho, grid)
                local_rows: List[Dict[str, Any]] = []
                executed = 0
                for iu, u in enumerate(vals):
                    for iv, v in enumerate(vals):
                        if max_points > 0 and executed >= max_points:
                            break
                        x0 = center + float(u) * e1 + float(v) * e2
                        item = {
                            "candidate": cand,
                            "candidate_id": cand["candidate_id"],
                            "equilibrium_id": eq_id,
                            "equilibrium": center,
                            "radius": float(math.sqrt(u * u + v * v)),
                            "sampling_mode": f"local_basin_{plane}",
                            "sample_id": f"grid_{iu}_{iv}",
                            "perturbation": x0 - center,
                            "x0": x0,
                            "h": float(stage["h"]),
                            "memory_length": float(stage["memory_length"]),
                            "t_final": float(stage["t_final"]),
                            "stage": "local_basin",
                        }
                        raw = classify_trajectory(item, p, eqs, cfg, ref_cache)
                        row = {
                            "candidate_id": cand["candidate_id"],
                            "equilibrium_id": eq_id,
                            "plane": plane,
                            "rho": rho,
                            "u": float(u),
                            "v": float(v),
                            "x0": float(x0[0]),
                            "y0": float(x0[1]),
                            "z0": float(x0[2]),
                            "final_class": raw.get("final_class", ""),
                            "target_hit": bool_from_row(raw, "target_hit"),
                            "final_x": raw.get("final_x", ""),
                            "final_y": raw.get("final_y", ""),
                            "final_z": raw.get("final_z", ""),
                            "notes": raw.get("notes", ""),
                            "_iu": iu,
                            "_iv": iv,
                        }
                        local_rows.append(row)
                        executed += 1
                    if max_points > 0 and executed >= max_points:
                        break
                write_csv(csv_path, local_rows, LOCAL_BASIN_FIELDS)
                img = np.zeros((grid, grid), dtype=int)
                cmap_order = {cls: i for i, cls in enumerate(CLASS_ORDER)}
                for row in local_rows:
                    img[as_int(row["_iv"]), as_int(row["_iu"])] = cmap_order.get(str(row["final_class"]), 0)
                fig, ax = plt.subplots(figsize=(5.2, 4.8))
                ax.imshow(img, extent=[-rho, rho, -rho, rho], origin="lower", aspect="equal")
                ax.set_xlabel("u")
                ax.set_ylabel("v")
                fig.tight_layout()
                fig.savefig(png_path, dpi=180)
                plt.close(fig)
                rows_summary.append({
                    "candidate_id": cand["candidate_id"],
                    "equilibrium_id": eq_id,
                    "plane": plane,
                    "rho": rho,
                    "grid": grid,
                    "points_evaluated": len(local_rows),
                    "target_hits": sum(1 for r in local_rows if bool(r["target_hit"])),
                    "basin_intersection_open_like": open_like_target(local_rows, grid),
                    "csv": str(csv_path),
                    "png": str(png_path),
                })
    (outdir / "local_basin_summary.json").write_text(json.dumps({"rows": json_safe(rows_summary)}, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows_summary


def write_report(outdir: Path, cfg: Dict[str, Any], candidates: Sequence[Dict[str, Any]], eq_rows: Sequence[Dict[str, Any]], summary: Sequence[Dict[str, Any]], decisions: Sequence[Dict[str, Any]], files: Sequence[str]) -> None:
    lines = [
        "# Lure Refined Route",
        "",
        "This report is conservative. It does not declare hidden_verified from a finite absence of contacts.",
        "",
        "## Equilibria",
        "",
    ]
    for row in eq_rows:
        lines.append(f"- `{row['eq_id']}` = ({as_float(row['x']):.10g}, {as_float(row['y']):.10g}, {as_float(row['z']):.10g}), region={row['region']}, Matignon stable={row['matignon_stable']}")
    lines.extend(["", "## Candidates", ""])
    by_summary: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in summary:
        by_summary[str(row["candidate_id"])].append(row)
    dec_by_id = {str(r["candidate_id"]): r for r in decisions}
    for cand in candidates:
        cid = cand["candidate_id"]
        total_samples = sum(int(r.get("n_samples", 0)) for r in by_summary.get(cid, []))
        total_hits = sum(int(r.get("n_target_attractor", 0)) for r in by_summary.get(cid, []))
        dec = dec_by_id.get(cid, {})
        lines.extend([
            f"### {cid}",
            "",
            f"- priority_class: `{cand.get('priority_class', '')}`",
            f"- completed samples: `{total_samples}`",
            f"- target hits: `{total_hits}`",
            f"- hiddenness_status: `{dec.get('hiddenness_status', 'not_evaluated')}`",
            f"- notes: {dec.get('decision_notes', '')}",
            "",
        ])
    lines.extend([
        "## Files",
        "",
        *[f"- `{p}`" for p in files],
        "",
        "## Sign Convention",
        "",
        "The route uses the repository convention `W_code(lambda) = r^T (P - lambda I)^(-1) b`, with `lambda=(j omega)^q`.",
    ])
    (outdir / "lure_refined_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refined verification route for classical Lure candidates.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stage", choices=["A", "B", "C"], default="A")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidate-id")
    parser.add_argument("--include-contact-strong", action="store_true")
    parser.add_argument("--local-basin", action="store_true")
    parser.add_argument("--local-basin-grid", choices=["fast", "final"], default="fast")
    parser.add_argument("--max-trajectories", type=int, default=0, help="Optional cap for this invocation; 0 means no cap.")
    parser.add_argument("--max-local-basin-points", type=int, default=0, help="Optional cap for local basin points; 0 means no cap.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = refined_dir(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    p = official_chua_params()
    chua.PARAMS = p
    candidates = load_candidates(cfg, args.candidate_id, args.include_contact_strong)
    eq_rows, eqs, eigvecs = equilibrium_rows(cfg, p, outdir)

    raw_path = outdir / "lure_refined_raw.csv"
    raw_rows = read_csv_rows(raw_path) if raw_path.exists() else []
    if args.local_basin:
        run_local_basin(cfg, candidates, p, eqs, eigvecs, outdir, args.local_basin_grid, int(args.max_local_basin_points))
        raw_rows = read_csv_rows(raw_path) if raw_path.exists() else []

    plan = build_plan(cfg, candidates, eqs, eigvecs, raw_rows, args.stage)
    processed = {processed_key(r) for r in raw_rows} if args.resume else set()
    executable = [item for item in plan if processed_key_from_plan(item) not in processed]
    if int(args.max_trajectories) > 0:
        executable = executable[: int(args.max_trajectories)]

    ref_cache: Dict[Tuple[str, float, float, float], Tuple[np.ndarray, str]] = {}
    for idx, item in enumerate(executable, start=1):
        row = classify_trajectory(item, p, eqs, cfg, ref_cache)
        append_csv(raw_path, row, RAW_FIELDS)
        if idx % 10 == 0:
            print(f"lure_refined_route progress: {idx} new trajectories", flush=True)

    raw_rows = read_csv_rows(raw_path) if raw_path.exists() else []
    summary = aggregate_raw(raw_rows, cfg)
    decisions = decide(candidates, summary, raw_rows, outdir)
    summary_path = outdir / "lure_refined_summary.csv"
    decision_path = outdir / "lure_refined_decision.csv"
    summary_json_path = outdir / "lure_refined_summary.json"
    write_csv(summary_path, summary, SUMMARY_FIELDS)
    write_csv(decision_path, decisions, DECISION_FIELDS)
    plot_files = plot_outputs(outdir, candidates, raw_rows, summary)
    files_written = [
        str(raw_path),
        str(summary_path),
        str(decision_path),
        str(outdir / "equilibria_lure_summary.csv"),
        str(outdir / "lure_refined_report.md"),
        str(summary_json_path),
        *plot_files,
    ]
    write_report(outdir, cfg, candidates, eq_rows, summary, decisions, files_written)
    rho_rows = {r.get("candidate_id", ""): r for r in read_csv_rows(rho_file(cfg))} if rho_file(cfg).exists() else {}
    summary_json = {
        "stage": args.stage,
        "planned_trajectories": len(plan),
        "executed_new_trajectories": len(executable),
        "resume": bool(args.resume),
        "candidates": [c["candidate_id"] for c in candidates],
        "decisions": decisions,
        "files_written": files_written,
    }
    summary_json_path.write_text(json.dumps(json_safe(summary_json), indent=2, ensure_ascii=False), encoding="utf-8")

    print("candidate_id,q,branch_index,priority_class,target_total,target_from_Eminus,target_from_Eplus,rho_H,rhoH_class,hiddenness_status,should_refine,files_written", flush=True)
    manifest_by_id = {r.get("candidate_id", ""): r for r in read_csv_rows(manifest_file(cfg))}
    dec_by_id = {r.get("candidate_id", ""): r for r in decisions}
    files = ";".join(files_written)
    for cand in candidates:
        cid = cand["candidate_id"]
        m = manifest_by_id.get(cid, {})
        rho = rho_rows.get(cid, {})
        dec = dec_by_id.get(cid, {})
        print(",".join([
            cid,
            str(cand.get("q", "")),
            str(cand.get("branch_index", "")),
            str(cand.get("priority_class", "")),
            str(m.get("target_total", "")),
            str(m.get("target_from_Eminus", "")),
            str(m.get("target_from_Eplus", "")),
            str(rho.get("rho_H", "")),
            str(rho.get("rhoH_class", "")),
            str(dec.get("hiddenness_status", "not_evaluated")),
            str(m.get("should_refine", "")),
            files,
        ]), flush=True)


if __name__ == "__main__":
    main()
