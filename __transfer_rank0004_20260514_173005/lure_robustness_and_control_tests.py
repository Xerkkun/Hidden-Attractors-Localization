#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

import chua_initial_cond as chua
import unified_nyquist_hidden_pipeline as pipe
from equilibria_analysis import local_jacobian, region_for_sigma, solve_equilibria
from lure_candidate_manifest import (
    DEFAULT_CONFIG,
    ROOT,
    as_float,
    as_int,
    csv_value,
    json_safe,
    load_config,
    read_csv_rows,
    resolve_path,
)


DEFAULT_CANDIDATE_ID = "lure_q_0p99000_branch_0_rep01"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "lure_route" / "robustness_and_controls"
ADAPTIVE_EMINUS_DIR = ROOT / "outputs" / "lure_route" / "adaptive_Eminus"

ROBUSTNESS_CASES = {
    "R0": {"h": 0.01, "memory_length": 40.0, "t_final": 3000.0},
    "R1": {"h": 0.005, "memory_length": 40.0, "t_final": 3000.0},
    "R2": {"h": 0.01, "memory_length": 80.0, "t_final": 3000.0},
    "R3": {"h": 0.005, "memory_length": 80.0, "t_final": 6000.0},
}

CONTROL_STAGE_A = {"h": 0.01, "memory_length": 40.0, "t_final": 1500.0}
CONTROL_STAGE_B = {"h": 0.005, "memory_length": 40.0, "t_final": 3000.0}
CONTROL_STAGE_C = {"h": 0.005, "memory_length": 80.0, "t_final": 3000.0}
CONTROL_RADII = [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2]

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

ROBUSTNESS_RAW_FIELDS = [
    "candidate_id",
    "run_id",
    "q",
    "h",
    "memory_length",
    "memory_points",
    "t_final",
    "start_source",
    "start_x",
    "start_y",
    "start_z",
    "n_points",
    "integration_sec",
    "final_x",
    "final_y",
    "final_z",
    "final_norm",
    "max_norm",
    "bounded",
    "divergent",
    "equilibrium_like",
    "range_x",
    "range_y",
    "range_z",
    "mean_x_tail",
    "mean_y_tail",
    "mean_z_tail",
    "var_x_tail",
    "var_y_tail",
    "var_z_tail",
    "fft_peak_x",
    "fft_peak_y",
    "fft_peak_z",
    "psd_entropy_x",
    "psd_entropy_y",
    "psd_entropy_z",
    "number_of_section_crossings",
    "lyap_max_if_available",
    "numerical_status",
    "notes",
]

ROBUSTNESS_SUMMARY_FIELDS = [
    "candidate_id",
    "q",
    "rho_H",
    "rhoH_class",
    "executed_runs",
    "skipped_runs",
    "divergence_norm",
    "all_executed_bounded",
    "tail_variances_noncollapsed",
    "no_equilibrium_like_run",
    "R0_R1_d_mean",
    "R0_R1_d_range",
    "R0_R1_d_range_rel",
    "R0_R1_d_var",
    "R0_R1_d_fft",
    "R0_R1_d_fft_rel",
    "R0_R2_d_mean",
    "R0_R2_d_range",
    "R0_R2_d_range_rel",
    "R0_R2_d_var",
    "R0_R2_d_fft",
    "R0_R2_d_fft_rel",
    "R1_R3_d_mean",
    "R1_R3_d_range",
    "R1_R3_d_range_rel",
    "R1_R3_d_var",
    "R1_R3_d_fft",
    "R1_R3_d_fft_rel",
    "robust_attractor",
    "robustness_status",
    "cost_guard_notes",
]

CONTROL_RAW_FIELDS = [
    "candidate_id",
    "q",
    "rho_H",
    "rhoH_class",
    "equilibrium_id",
    "radius",
    "direction_label",
    "direction_x",
    "direction_y",
    "direction_z",
    "random_direction",
    "stage",
    "run_id",
    "x0",
    "y0",
    "z0",
    "h",
    "memory_length",
    "memory_points",
    "t_final",
    "final_x",
    "final_y",
    "final_z",
    "final_norm",
    "max_norm",
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
    "fft_peak_x",
    "fft_peak_y",
    "fft_peak_z",
    "psd_entropy_x",
    "psd_entropy_y",
    "psd_entropy_z",
    "number_of_section_crossings",
    "section_hits",
    "section_hit_fraction",
    "final_class",
    "target_hit",
    "numerical_status",
    "notes",
]

CONTROL_SUMMARY_FIELDS = [
    "candidate_id",
    "equilibrium_id",
    "radius",
    "stage",
    "n_samples",
    "n_equilibrium_convergence",
    "n_target_attractor",
    "n_other_bounded_nontrivial",
    "n_divergent",
    "n_numerical_failure",
    "n_ambiguous_long_transient",
    "target_hit_fraction",
    "most_common_class",
]

CONTROL_DECISION_FIELDS = [
    "candidate_id",
    "Eplus_target_hits",
    "E0_target_hits",
    "target_hits",
    "same_equilibrium_radius_target_repeats",
    "reproduced_h_target_hits",
    "reproduced_memory_target_hits",
    "robust_control_target_hit",
    "hiddenness_status",
    "decision_notes",
]

STATUS_FIELDS = [
    "candidate_id",
    "q",
    "rho_H",
    "rhoH_class",
    "Eminus_status_previous",
    "Eminus_adaptive_target_hits",
    "robustness_status",
    "robust_attractor",
    "Eplus_target_hits",
    "E0_target_hits",
    "robust_control_target_hit",
    "final_recommended_status",
    "hidden_verified",
    "next_action",
]


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


def read_rows(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "si", "on"}


def memory_points(memory_length: float, h: float) -> int:
    return max(1, int(math.ceil(float(memory_length) / float(h)))) + 1


def manifest_file(cfg: Dict[str, Any]) -> Path:
    return ROOT / str(cfg.get("manifest", {}).get("output_dir", "outputs/lure_route")) / "lure_candidates_manifest.csv"


def rho_file(cfg: Dict[str, Any]) -> Path:
    return ROOT / str(cfg.get("manifest", {}).get("output_dir", "outputs/lure_route")) / "lure_rhoH_diagnostics.csv"


def output_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def vec3_from_row(row: Dict[str, Any], prefix: str) -> np.ndarray:
    return np.asarray(
        [as_float(row.get(f"{prefix}_x")), as_float(row.get(f"{prefix}_y")), as_float(row.get(f"{prefix}_z"))],
        dtype=float,
    )


def finite_vec(v: np.ndarray) -> bool:
    arr = np.asarray(v, dtype=float)
    return arr.shape == (3,) and np.all(np.isfinite(arr))


def candidate_base(candidate_id: str) -> str:
    if "_rep" in candidate_id:
        return candidate_id.rsplit("_rep", 1)[0]
    return candidate_id


def load_candidate(cfg: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    manifest = {row.get("candidate_id", ""): row for row in read_csv_rows(manifest_file(cfg))}
    rho_rows = {row.get("candidate_id", ""): row for row in read_csv_rows(rho_file(cfg))}
    if candidate_id not in manifest:
        raise ValueError(f"No existe {candidate_id} en {manifest_file(cfg)}")
    row = dict(manifest[candidate_id])
    rho = rho_rows.get(candidate_id, {})
    seed = vec3_from_row(row, "seed")
    target = vec3_from_row(row, "final")
    if not finite_vec(target):
        target = last_state_from_csv(row.get("source_final_attractor_csv", ""))
    candidate = {
        **row,
        "candidate_id": candidate_id,
        "duplicate_group": candidate_base(candidate_id),
        "representative": candidate_id,
        "q_float": as_float(row.get("q")),
        "q": as_float(row.get("q")),
        "branch_index": as_int(row.get("branch_index")),
        "seed_vec": seed,
        "seed": seed,
        "target_seed": target if finite_vec(target) else seed,
        "rho_H": rho.get("rho_H", row.get("rho_H", "")),
        "rhoH_class": rho.get("rhoH_class", row.get("rhoH_class", "")),
        "rho_row": rho,
        "manifest_row": row,
    }
    if not finite_vec(candidate["seed_vec"]):
        raise ValueError(f"{candidate_id} no tiene semilla Lure finita en el manifest.")
    if not finite_vec(candidate["target_seed"]):
        raise ValueError(f"{candidate_id} no tiene estado final ni semilla finita para integrar.")
    return candidate


def last_state_from_csv(path_value: Any) -> np.ndarray:
    path = resolve_path(path_value, ROOT)
    if not path or not path.exists():
        return np.asarray([float("nan"), float("nan"), float("nan")], dtype=float)
    last: np.ndarray | None = None
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            state = np.asarray([as_float(row.get("x")), as_float(row.get("y")), as_float(row.get("z"))], dtype=float)
            if finite_vec(state):
                last = state
    return last if last is not None else np.asarray([float("nan"), float("nan"), float("nan")], dtype=float)


def chua_params() -> Dict[str, Any]:
    p = pipe.chua_ic_params_from_config(pipe.CONFIG)
    chua.PARAMS = p
    return p


def eig_rows_for_equilibria(p: Dict[str, Any], q: float, eqs: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    theta = float(q) * math.pi / 2.0
    _, _, r_vec = chua.chua_matrices(p)
    for eq_id in ["E-", "E0", "E+"]:
        if eq_id not in eqs:
            continue
        eq = np.asarray(eqs[eq_id], dtype=float)
        J = local_jacobian(p, eq)
        vals = np.linalg.eigvals(J)
        margins = [abs(np.angle(v)) - theta for v in vals]
        sigma = float(np.asarray(eq, dtype=float) @ np.asarray(r_vec, dtype=float))
        rows.append(
            {
                "eq_id": eq_id,
                "x": float(eq[0]),
                "y": float(eq[1]),
                "z": float(eq[2]),
                "region": region_for_sigma(sigma),
                "eig_1": complex(vals[0]),
                "eig_2": complex(vals[1]),
                "eig_3": complex(vals[2]),
                "matignon_margin": float(min(margins)),
                "matignon_stable": bool(all(m > 0.0 for m in margins)),
            }
        )
    return rows


def load_or_recompute_equilibria(p: Dict[str, Any], q: float, outdir: Path) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    source = ADAPTIVE_EMINUS_DIR / "equilibria_used.csv"
    rows = read_rows(source)
    eqs: Dict[str, np.ndarray] = {}
    for row in rows:
        eq_id = str(row.get("eq_id", ""))
        state = np.asarray([as_float(row.get("x")), as_float(row.get("y")), as_float(row.get("z"))], dtype=float)
        if eq_id and finite_vec(state):
            eqs[eq_id] = state
    required = {"E-", "E0", "E+"}
    if not required.issubset(eqs):
        eqs = {k: np.asarray(v, dtype=float) for k, v in solve_equilibria(p).items() if finite_vec(np.asarray(v, dtype=float))}
        rows_out = eig_rows_for_equilibria(p, q, eqs)
    else:
        rows_out = eig_rows_for_equilibria(p, q, eqs)
    missing = sorted(required.difference(eqs))
    if missing:
        raise RuntimeError(f"Faltan equilibrios requeridos: {', '.join(missing)}")
    write_csv(outdir / "equilibria_used.csv", rows_out, ["eq_id", "x", "y", "z", "region", "eig_1", "eig_2", "eig_3", "matignon_margin", "matignon_stable"])
    return eqs, rows_out


def section_points(traj: np.ndarray, p: Dict[str, Any], t_burn: float, max_points: int = 100) -> np.ndarray:
    X = np.asarray(traj, dtype=float)
    if X.ndim != 2 or X.shape[1] < 4 or X.shape[0] < 2:
        return np.empty((0, 2), dtype=float)
    pts: List[Tuple[float, float]] = []
    for k in range(1, X.shape[0]):
        if X[k, 0] < float(t_burn):
            continue
        xp, x = X[k - 1, 1], X[k, 1]
        if xp < 0.0 <= x:
            try:
                xdot = float(chua.rhs_original(X[k, 1:4], p)[0])
            except Exception:
                xdot = float("nan")
            if xdot > 0.0:
                lam = (0.0 - xp) / ((x - xp) + 1e-300)
                y = X[k - 1, 2] + lam * (X[k, 2] - X[k - 1, 2])
                z = X[k - 1, 3] + lam * (X[k, 3] - X[k - 1, 3])
                pts.append((float(y), float(z)))
                if len(pts) >= int(max_points):
                    break
    return np.asarray(pts, dtype=float)


def hit_fraction(section: np.ndarray, ref: np.ndarray, tol: float = 0.12) -> Tuple[int, int, float]:
    if section.size == 0 or ref.size == 0:
        total = int(section.shape[0]) if section.ndim == 2 else 0
        return total, 0, 0.0
    hits = 0
    for pt in section:
        if float(np.min(np.linalg.norm(ref - pt.reshape(1, 2), axis=1))) <= float(tol):
            hits += 1
    total = int(section.shape[0])
    return total, hits, float(hits / max(total, 1))


def component_fft(values: np.ndarray, h: float) -> Tuple[float, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size <= 8:
        return float("nan"), float("nan")
    data = data - float(np.mean(data))
    if not np.any(np.abs(data) > 0.0):
        return 0.0, 0.0
    spec = np.abs(np.fft.rfft(data * np.hanning(data.size))) ** 2
    freq = np.fft.rfftfreq(data.size, d=float(h))
    if spec.size <= 1 or not np.any(spec[1:] > 0.0):
        return 0.0, 0.0
    idx = 1 + int(np.argmax(spec[1:]))
    prob = spec[1:] / max(float(np.sum(spec[1:])), 1e-300)
    entropy = -float(np.sum(prob * np.log(prob + 1e-300))) / max(math.log(prob.size), 1e-300)
    return float(freq[idx]), entropy


def trajectory_stats(traj: np.ndarray, p: Dict[str, Any], h: float, t_final: float) -> Dict[str, Any]:
    X = np.asarray(traj, dtype=float)
    states = X[:, 1:4] if X.ndim == 2 and X.shape[1] >= 4 else np.empty((0, 3), dtype=float)
    if states.size == 0 or not np.all(np.isfinite(states)):
        return {k: float("nan") for k in [
            "final_x", "final_y", "final_z", "final_norm", "max_norm", "range_x", "range_y", "range_z",
            "mean_x_tail", "mean_y_tail", "mean_z_tail", "var_x_tail", "var_y_tail", "var_z_tail",
            "fft_peak_x", "fft_peak_y", "fft_peak_z", "psd_entropy_x", "psd_entropy_y", "psd_entropy_z",
            "number_of_section_crossings",
        ]}
    start = max(0, int(0.8 * states.shape[0]))
    tail = states[start:, :]
    mean = np.mean(tail, axis=0)
    var = np.var(tail, axis=0)
    ranges = np.ptp(states, axis=0)
    peaks: List[float] = []
    entropies: List[float] = []
    for comp in range(3):
        peak, ent = component_fft(tail[:, comp], h)
        peaks.append(peak)
        entropies.append(ent)
    section = section_points(X, p, 0.5 * float(t_final), max_points=10000)
    final = states[-1]
    return {
        "final_x": float(final[0]),
        "final_y": float(final[1]),
        "final_z": float(final[2]),
        "final_norm": float(np.linalg.norm(final)),
        "max_norm": float(np.max(np.linalg.norm(states, axis=1))),
        "range_x": float(ranges[0]),
        "range_y": float(ranges[1]),
        "range_z": float(ranges[2]),
        "mean_x_tail": float(mean[0]),
        "mean_y_tail": float(mean[1]),
        "mean_z_tail": float(mean[2]),
        "var_x_tail": float(var[0]),
        "var_y_tail": float(var[1]),
        "var_z_tail": float(var[2]),
        "fft_peak_x": peaks[0],
        "fft_peak_y": peaks[1],
        "fft_peak_z": peaks[2],
        "psd_entropy_x": entropies[0],
        "psd_entropy_y": entropies[1],
        "psd_entropy_z": entropies[2],
        "number_of_section_crossings": int(section.shape[0]),
    }


def min_distance_to_equilibria(states: np.ndarray, eqs: Dict[str, np.ndarray]) -> float:
    if states.size == 0:
        return float("nan")
    vals = []
    for eq in eqs.values():
        vals.append(float(np.min(np.linalg.norm(states - np.asarray(eq, dtype=float).reshape(1, 3), axis=1))))
    return min(vals) if vals else float("nan")


def integrate_trajectory(x0: np.ndarray, p: Dict[str, Any], q: float, params: Dict[str, float]) -> np.ndarray:
    return pipe.integrate_efork3_c(
        np.asarray(x0, dtype=float),
        p,
        qord=float(q),
        h=float(params["h"]),
        Lm=float(params["memory_length"]),
        t_total=float(params["t_final"]),
    )


def reference_for(
    candidate: Dict[str, Any],
    p: Dict[str, Any],
    params: Dict[str, float],
    cache: Dict[Tuple[str, float, float, float], np.ndarray],
) -> np.ndarray:
    key = (
        str(candidate["candidate_id"]),
        float(params["h"]),
        float(params["memory_length"]),
        float(params["t_final"]),
    )
    if key in cache:
        return cache[key]
    x0 = np.asarray(candidate["target_seed"], dtype=float)
    traj = integrate_trajectory(x0, p, float(candidate["q_float"]), params)
    ref = section_points(traj, p, 0.5 * float(params["t_final"]), max_points=240)
    cache[key] = ref
    return ref


def classify_trajectory(
    *,
    candidate: Dict[str, Any],
    x0: np.ndarray,
    params: Dict[str, float],
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    ref_cache: Dict[Tuple[str, float, float, float], np.ndarray],
    notes: str = "",
) -> Tuple[Dict[str, Any], np.ndarray | None]:
    t0 = time.time()
    try:
        traj = integrate_trajectory(x0, p, float(candidate["q_float"]), params)
        states = traj[:, 1:4] if traj.ndim == 2 and traj.shape[1] >= 4 else np.empty((0, 3), dtype=float)
        if states.size == 0 or not np.all(np.isfinite(states)):
            raise RuntimeError("nonfinite_or_empty_trajectory")
        stats = trajectory_stats(traj, p, float(params["h"]), float(params["t_final"]))
        final = states[-1]
        final_dist = {k: float(np.linalg.norm(final - v)) for k, v in eqs.items()}
        min_dist = min_distance_to_equilibria(states, eqs)
        divergence = float(cfg.get("classification", {}).get("divergence_norm", 1.0e5))
        eq_radius = float(cfg.get("classification", {}).get("equilibrium_radius", 1.0e-3))
        min_matches = int(cfg.get("classification", {}).get("min_section_matches", 20))
        hit_req = float(cfg.get("classification", {}).get("target_hit_fraction_required", 0.70))
        sec_total = sec_hits = 0
        sec_frac = 0.0
        if float(stats["final_norm"]) > divergence or float(stats["max_norm"]) > divergence:
            final_class = "divergent"
            target_hit = False
        else:
            nearest = min(final_dist.items(), key=lambda kv: kv[1])
            tail_mean = np.asarray([stats["mean_x_tail"], stats["mean_y_tail"], stats["mean_z_tail"]], dtype=float)
            tail_dist = float(np.linalg.norm(tail_mean - eqs[nearest[0]]))
            if nearest[1] <= eq_radius and tail_dist <= 2.0 * eq_radius:
                final_class = "equilibrium_convergence"
                target_hit = False
            else:
                ref = reference_for(candidate, p, params, ref_cache)
                sec = section_points(traj, p, 0.5 * float(params["t_final"]), max_points=100)
                sec_total, sec_hits, sec_frac = hit_fraction(sec, ref, tol=0.12)
                if sec_total >= min_matches and sec_frac >= hit_req:
                    final_class = "target_attractor"
                    target_hit = True
                elif sec_total < min_matches:
                    final_class = "ambiguous_long_transient"
                    target_hit = False
                else:
                    max_var = max(float(stats["var_x_tail"]), float(stats["var_y_tail"]), float(stats["var_z_tail"]))
                    max_range = max(float(stats["range_x"]), float(stats["range_y"]), float(stats["range_z"]))
                    final_class = "other_bounded_nontrivial" if max_var > 1.0e-6 or max_range > 1.0e-2 else "ambiguous_long_transient"
                    target_hit = False
        row = {
            **stats,
            "min_dist_to_equilibria": min_dist,
            "final_dist_to_Eminus": final_dist.get("E-", float("nan")),
            "final_dist_to_E0": final_dist.get("E0", float("nan")),
            "final_dist_to_Eplus": final_dist.get("E+", float("nan")),
            "section_hits": sec_hits,
            "section_hit_fraction": sec_frac,
            "final_class": final_class,
            "target_hit": bool(target_hit),
            "numerical_status": "ok",
            "notes": f"{notes}; sec_total={sec_total}; sec_hits={sec_hits}; hit_frac={sec_frac:.6g}; elapsed_sec={time.time() - t0:.3f}".strip("; "),
        }
        return row, traj
    except Exception as exc:
        empty = {
            "final_x": "",
            "final_y": "",
            "final_z": "",
            "final_norm": "",
            "max_norm": "",
            "min_dist_to_equilibria": "",
            "final_dist_to_Eminus": "",
            "final_dist_to_E0": "",
            "final_dist_to_Eplus": "",
            "range_x": "",
            "range_y": "",
            "range_z": "",
            "mean_x_tail": "",
            "mean_y_tail": "",
            "mean_z_tail": "",
            "var_x_tail": "",
            "var_y_tail": "",
            "var_z_tail": "",
            "fft_peak_x": "",
            "fft_peak_y": "",
            "fft_peak_z": "",
            "psd_entropy_x": "",
            "psd_entropy_y": "",
            "psd_entropy_z": "",
            "number_of_section_crossings": "",
            "section_hits": "",
            "section_hit_fraction": "",
            "final_class": "numerical_failure",
            "target_hit": False,
            "numerical_status": "exception",
            "notes": f"{notes}; {exc}".strip("; "),
        }
        return empty, None


def save_reduced_trajectory(path: str | Path, traj: np.ndarray, max_points: int = 12000, burn_fraction: float = 0.5) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    X = np.asarray(traj, dtype=float)
    if X.ndim != 2 or X.shape[1] < 4:
        write_csv(path, [], ["t", "x", "y", "z"])
        return str(path)
    start = min(X.shape[0] - 1, max(0, int(float(burn_fraction) * X.shape[0])))
    Y = X[start:, :4]
    if Y.shape[0] > int(max_points):
        idx = np.linspace(0, Y.shape[0] - 1, int(max_points)).astype(int)
        Y = Y[idx]
    rows = [{"t": float(r[0]), "x": float(r[1]), "y": float(r[2]), "z": float(r[3])} for r in Y]
    write_csv(path, rows, ["t", "x", "y", "z"])
    return str(path)


def load_trajectory_sample(path: str | Path, max_points: int = 20000) -> np.ndarray:
    rows = read_rows(path)
    vals = []
    for row in rows:
        vals.append([as_float(row.get("t")), as_float(row.get("x")), as_float(row.get("y")), as_float(row.get("z"))])
    arr = np.asarray(vals, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return np.empty((0, 4), dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if arr.shape[0] > max_points:
        idx = np.linspace(0, arr.shape[0] - 1, max_points).astype(int)
        arr = arr[idx]
    return arr


def source_or_reconstructed_attractor(candidate: Dict[str, Any], p: Dict[str, Any]) -> np.ndarray:
    source = resolve_path(candidate.get("source_final_attractor_csv", ""), ROOT)
    if source.exists():
        arr = load_trajectory_sample(source, max_points=18000)
        if arr.shape[0] > 0:
            return arr
    params = ROBUSTNESS_CASES["R0"]
    traj = integrate_trajectory(np.asarray(candidate["seed_vec"], dtype=float), p, float(candidate["q_float"]), params)
    return traj


def robustness_start(candidate: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    target = np.asarray(candidate["target_seed"], dtype=float)
    if finite_vec(target):
        return target, "manifest_final_or_source_final_attractor"
    return np.asarray(candidate["seed_vec"], dtype=float), "lure_seed"


def row_float(row: Dict[str, Any], key: str) -> float:
    return as_float(row.get(key))


def run_metrics_vectors(row: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {
        "mean": np.asarray([row_float(row, "mean_x_tail"), row_float(row, "mean_y_tail"), row_float(row, "mean_z_tail")], dtype=float),
        "range": np.asarray([row_float(row, "range_x"), row_float(row, "range_y"), row_float(row, "range_z")], dtype=float),
        "var": np.asarray([row_float(row, "var_x_tail"), row_float(row, "var_y_tail"), row_float(row, "var_z_tail")], dtype=float),
        "fft": np.asarray([row_float(row, "fft_peak_x"), row_float(row, "fft_peak_y"), row_float(row, "fft_peak_z")], dtype=float),
    }


def pair_diff(row_a: Dict[str, Any] | None, row_b: Dict[str, Any] | None) -> Dict[str, float]:
    if row_a is None or row_b is None:
        return {"d_mean": float("nan"), "d_range": float("nan"), "d_range_rel": float("nan"), "d_var": float("nan"), "d_fft": float("nan"), "d_fft_rel": float("nan")}
    va = run_metrics_vectors(row_a)
    vb = run_metrics_vectors(row_b)
    d_mean = float(np.linalg.norm(va["mean"] - vb["mean"]))
    d_range = float(np.linalg.norm(va["range"] - vb["range"]))
    d_var = float(np.linalg.norm(va["var"] - vb["var"]))
    finite_fft = np.isfinite(va["fft"][0]) and np.isfinite(vb["fft"][0])
    d_fft = abs(float(va["fft"][0] - vb["fft"][0])) if finite_fft else float("nan")
    range_scale = max(float(np.linalg.norm(va["range"])), float(np.linalg.norm(vb["range"])), 1.0e-12)
    fft_scale = max(abs(float(va["fft"][0])) if np.isfinite(va["fft"][0]) else 0.0, abs(float(vb["fft"][0])) if np.isfinite(vb["fft"][0]) else 0.0, 1.0e-12)
    return {
        "d_mean": d_mean,
        "d_range": d_range,
        "d_range_rel": d_range / range_scale,
        "d_var": d_var,
        "d_fft": d_fft,
        "d_fft_rel": d_fft / fft_scale if math.isfinite(d_fft) else float("nan"),
    }


def is_equilibrium_like(row: Dict[str, Any], eqs: Dict[str, np.ndarray], eq_radius: float) -> bool:
    mean = np.asarray([row_float(row, "mean_x_tail"), row_float(row, "mean_y_tail"), row_float(row, "mean_z_tail")], dtype=float)
    var = np.asarray([row_float(row, "var_x_tail"), row_float(row, "var_y_tail"), row_float(row, "var_z_tail")], dtype=float)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)):
        return True
    near_eq = any(float(np.linalg.norm(mean - eq)) <= 2.0 * float(eq_radius) for eq in eqs.values())
    return bool(near_eq and float(np.max(var)) <= 1.0e-8)


def summarize_robustness(
    candidate: Dict[str, Any],
    raw_rows: Sequence[Dict[str, Any]],
    skipped: Sequence[str],
    cfg: Dict[str, Any],
    cost_notes: Sequence[str],
) -> Dict[str, Any]:
    by_run = {str(r.get("run_id", "")): r for r in raw_rows if str(r.get("candidate_id", "")) == candidate["candidate_id"]}
    executed = [rid for rid in ["R0", "R1", "R2", "R3"] if rid in by_run]
    divergence = float(cfg.get("classification", {}).get("divergence_norm", 1.0e5))
    eq_radius = float(cfg.get("classification", {}).get("equilibrium_radius", 1.0e-3))
    p01 = pair_diff(by_run.get("R0"), by_run.get("R1"))
    p02 = pair_diff(by_run.get("R0"), by_run.get("R2"))
    p13 = pair_diff(by_run.get("R1"), by_run.get("R3"))
    all_bounded = all(truthy(by_run[rid].get("bounded")) and row_float(by_run[rid], "max_norm") < divergence for rid in executed) if executed else False
    noncollapsed = all(
        max(row_float(by_run[rid], "var_x_tail"), row_float(by_run[rid], "var_y_tail"), row_float(by_run[rid], "var_z_tail")) > 1.0e-8
        for rid in executed
    ) if executed else False
    no_eq = all(not truthy(by_run[rid].get("equilibrium_like")) for rid in executed) if executed else False
    r0r1_ok = (
        "R0" in by_run
        and "R1" in by_run
        and p01["d_range_rel"] < 0.25
        and (not math.isfinite(p01["d_fft_rel"]) or p01["d_fft_rel"] < 0.25)
    )
    any_divergent = any(truthy(by_run[rid].get("divergent")) for rid in executed)
    any_eq_like = any(truthy(by_run[rid].get("equilibrium_like")) for rid in executed)
    robust = bool(executed and all_bounded and r0r1_ok and noncollapsed and no_eq)
    if any_divergent or any_eq_like:
        status = "not_robust"
    elif robust:
        status = "robust_attractor"
    elif executed:
        status = "numerically_plausible_but_needs_longer_run"
    else:
        status = "not_evaluated"
    return {
        "candidate_id": candidate["candidate_id"],
        "q": candidate["q_float"],
        "rho_H": candidate.get("rho_H", ""),
        "rhoH_class": candidate.get("rhoH_class", ""),
        "executed_runs": ";".join(executed),
        "skipped_runs": ";".join(skipped),
        "divergence_norm": divergence,
        "all_executed_bounded": bool(all_bounded),
        "tail_variances_noncollapsed": bool(noncollapsed),
        "no_equilibrium_like_run": bool(no_eq),
        "R0_R1_d_mean": p01["d_mean"],
        "R0_R1_d_range": p01["d_range"],
        "R0_R1_d_range_rel": p01["d_range_rel"],
        "R0_R1_d_var": p01["d_var"],
        "R0_R1_d_fft": p01["d_fft"],
        "R0_R1_d_fft_rel": p01["d_fft_rel"],
        "R0_R2_d_mean": p02["d_mean"],
        "R0_R2_d_range": p02["d_range"],
        "R0_R2_d_range_rel": p02["d_range_rel"],
        "R0_R2_d_var": p02["d_var"],
        "R0_R2_d_fft": p02["d_fft"],
        "R0_R2_d_fft_rel": p02["d_fft_rel"],
        "R1_R3_d_mean": p13["d_mean"],
        "R1_R3_d_range": p13["d_range"],
        "R1_R3_d_range_rel": p13["d_range_rel"],
        "R1_R3_d_var": p13["d_var"],
        "R1_R3_d_fft": p13["d_fft"],
        "R1_R3_d_fft_rel": p13["d_fft_rel"],
        "robust_attractor": bool(robust),
        "robustness_status": status,
        "cost_guard_notes": " | ".join(cost_notes),
        "_eq_radius": eq_radius,
    }


def run_robustness(
    args: argparse.Namespace,
    candidate: Dict[str, Any],
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    outdir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str], Dict[str, Any]]:
    raw_path = outdir / "attractor_robustness_raw.csv"
    traj_dir = outdir / "trajectories"
    raw_rows: List[Dict[str, Any]] = [dict(r) for r in read_rows(raw_path)] if args.resume else []
    if not args.resume and raw_path.exists():
        raw_path.unlink()
    by_run = {str(r.get("run_id", "")): r for r in raw_rows}
    skipped: List[str] = []
    files: List[str] = [str(raw_path)]
    cost_notes: List[str] = []
    measured_first: float | None = None
    x0, start_source = robustness_start(candidate)
    divergence = float(cfg.get("classification", {}).get("divergence_norm", 1.0e5))
    eq_radius = float(cfg.get("classification", {}).get("equilibrium_radius", 1.0e-3))

    def execute(run_id: str) -> None:
        nonlocal measured_first
        if run_id in by_run:
            return
        params = ROBUSTNESS_CASES[run_id]
        t0 = time.time()
        traj: np.ndarray | None
        try:
            traj = integrate_trajectory(x0, p, float(candidate["q_float"]), params)
            if traj.ndim != 2 or traj.shape[1] < 4 or not np.all(np.isfinite(traj[:, 1:4])):
                raise RuntimeError("nonfinite_or_empty_trajectory")
            stats = trajectory_stats(traj, p, float(params["h"]), float(params["t_final"]))
            max_norm = as_float(stats.get("max_norm"))
            if math.isfinite(max_norm) and max_norm >= divergence:
                final_class = "divergent"
            elif is_equilibrium_like(stats, eqs, eq_radius):
                final_class = "equilibrium_like"
            else:
                final_class = "bounded_nontrivial"
            stats.update(
                {
                    "final_class": final_class,
                    "numerical_status": "ok",
                    "notes": f"robustness_{run_id}; elapsed_sec={time.time() - t0:.3f}",
                }
            )
        except Exception as exc:
            traj = None
            stats = {
                "final_x": "",
                "final_y": "",
                "final_z": "",
                "final_norm": "",
                "max_norm": "",
                "range_x": "",
                "range_y": "",
                "range_z": "",
                "mean_x_tail": "",
                "mean_y_tail": "",
                "mean_z_tail": "",
                "var_x_tail": "",
                "var_y_tail": "",
                "var_z_tail": "",
                "fft_peak_x": "",
                "fft_peak_y": "",
                "fft_peak_z": "",
                "psd_entropy_x": "",
                "psd_entropy_y": "",
                "psd_entropy_z": "",
                "number_of_section_crossings": "",
                "final_class": "numerical_failure",
                "numerical_status": "exception",
                "notes": f"robustness_{run_id}; {exc}",
            }
        elapsed = time.time() - t0
        if measured_first is None:
            measured_first = elapsed
        max_norm = as_float(stats.get("max_norm"))
        bounded = bool(stats.get("numerical_status") == "ok" and math.isfinite(max_norm) and max_norm < divergence)
        row = {
            "candidate_id": candidate["candidate_id"],
            "run_id": run_id,
            "q": candidate["q_float"],
            "h": params["h"],
            "memory_length": params["memory_length"],
            "memory_points": memory_points(params["memory_length"], params["h"]),
            "t_final": params["t_final"],
            "start_source": start_source,
            "start_x": float(x0[0]),
            "start_y": float(x0[1]),
            "start_z": float(x0[2]),
            "n_points": int(traj.shape[0]) if traj is not None else "",
            "integration_sec": elapsed,
            **{k: stats.get(k, "") for k in ROBUSTNESS_RAW_FIELDS if k in stats},
            "bounded": bounded,
            "divergent": bool(stats.get("final_class") == "divergent" or (math.isfinite(max_norm) and max_norm >= divergence)),
            "equilibrium_like": is_equilibrium_like(stats, eqs, eq_radius) if stats.get("numerical_status") == "ok" else False,
            "lyap_max_if_available": "",
            "notes": stats.get("notes", ""),
        }
        by_run[run_id] = row
        raw_rows.append(row)
        append_csv(raw_path, row, ROBUSTNESS_RAW_FIELDS)
        if traj is not None:
            files.append(save_reduced_trajectory(traj_dir / f"attractor_{run_id}.csv", traj, max_points=16000, burn_fraction=0.5))
        print(f"{candidate['candidate_id']} robustness {run_id} class={stats.get('final_class')} bounded={bounded} elapsed_sec={elapsed:.3f}", flush=True)

    execute("R0")
    if measured_first is None and "R0" in by_run:
        measured_first = as_float(by_run["R0"].get("integration_sec"))
    execute("R1")
    p01 = pair_diff(by_run.get("R0"), by_run.get("R1"))
    r0_r1_similar = (
        "R0" in by_run
        and "R1" in by_run
        and truthy(by_run["R0"].get("bounded"))
        and truthy(by_run["R1"].get("bounded"))
        and p01["d_range_rel"] < 0.25
        and (not math.isfinite(p01["d_fft_rel"]) or p01["d_fft_rel"] < 0.25)
    )
    if r0_r1_similar:
        execute("R2")
    else:
        skipped.append("R2")
        cost_notes.append("R2 skipped because R0/R1 were not both bounded and similar.")
    if "R3" not in by_run:
        r0_rows = int(float(ROBUSTNESS_CASES["R0"]["t_final"]) / float(ROBUSTNESS_CASES["R0"]["h"])) + 1
        r3_rows = int(float(ROBUSTNESS_CASES["R3"]["t_final"]) / float(ROBUSTNESS_CASES["R3"]["h"])) + 1
        est_r3 = (measured_first or 0.0) * (r3_rows / max(r0_rows, 1))
        max_sec = float(args.max_estimated_hours) * 3600.0
        if args.force_long or args.force or (r0_r1_similar and est_r3 <= max_sec):
            execute("R3")
        else:
            skipped.append("R3")
            cost_notes.append(f"R3 skipped by cost guard; estimated_sec={est_r3:.3f}; use --force-long or --force.")
    summary = summarize_robustness(candidate, raw_rows, skipped, cfg, cost_notes)
    write_csv(outdir / "attractor_robustness_summary.csv", [summary], ROBUSTNESS_SUMMARY_FIELDS)
    files.append(str(outdir / "attractor_robustness_summary.csv"))
    data = {
        "candidate_id": candidate["candidate_id"],
        "runs": raw_rows,
        "summary": summary,
        "pair_comparisons": {
            "R0_vs_R1": pair_diff(by_run.get("R0"), by_run.get("R1")),
            "R0_vs_R2": pair_diff(by_run.get("R0"), by_run.get("R2")),
            "R1_vs_R3": pair_diff(by_run.get("R1"), by_run.get("R3")),
        },
        "cost_guard": {
            "planned_robustness_trajectories": 4,
            "executed_robustness_trajectories": len([r for r in raw_rows if r.get("candidate_id") == candidate["candidate_id"]]),
            "measured_first_trajectory_sec": measured_first,
            "max_estimated_hours": args.max_estimated_hours,
            "notes": cost_notes,
        },
        "scientific_note": "Describing functions only generated the seed; validation is by causal Caputo memory integration. hidden_verified is never declared here.",
    }
    json_path = outdir / "attractor_robustness_summary.json"
    json_path.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")
    files.append(str(json_path))
    return raw_rows, summary, files, data["cost_guard"]


def normalize_vec(v: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if not math.isfinite(n) or n < 1.0e-14:
        return None
    return arr / n


def add_direction(directions: List[Tuple[str, np.ndarray, bool]], label: str, vec: np.ndarray, random_flag: bool = False) -> None:
    unit = normalize_vec(vec)
    if unit is None:
        return
    for _, existing, _ in directions:
        if float(np.dot(unit, existing)) > 0.999999:
            return
    directions.append((label, unit, random_flag))


def dominant_eigen_directions(p: Dict[str, Any], eq: np.ndarray) -> List[Tuple[str, np.ndarray, bool]]:
    vals, vecs = np.linalg.eig(local_jacobian(p, eq))
    order = sorted(range(len(vals)), key=lambda i: float(np.real(vals[i])), reverse=True)
    dirs: List[Tuple[str, np.ndarray, bool]] = []
    for idx in order:
        raw = vecs[:, idx]
        candidate = np.real(raw)
        if normalize_vec(candidate) is None:
            candidate = np.imag(raw)
        unit = normalize_vec(candidate)
        if unit is None:
            continue
        add_direction(dirs, f"eig_dominant_{idx}_p", unit, False)
        add_direction(dirs, f"eig_dominant_{idx}_m", -unit, False)
        break
    return dirs


def switching_surface_directions(p: Dict[str, Any], eq: np.ndarray) -> List[Tuple[str, np.ndarray, bool]]:
    _P, _b, r_vec = chua.chua_matrices(p)
    r = np.asarray(r_vec, dtype=float)
    sigma = float(np.asarray(eq, dtype=float) @ r)
    if abs(abs(sigma) - 1.0) > 5.0e-2:
        return []
    normal = normalize_vec(r)
    if normal is None:
        return []
    dirs: List[Tuple[str, np.ndarray, bool]] = []
    add_direction(dirs, "switch_normal_p", normal, False)
    add_direction(dirs, "switch_normal_m", -normal, False)
    aux = np.array([0.0, 1.0, 0.0], dtype=float) if abs(float(np.dot(normal, [1.0, 0.0, 0.0]))) > 0.9 else np.array([1.0, 0.0, 0.0], dtype=float)
    tangent = normalize_vec(aux - float(np.dot(aux, normal)) * normal)
    if tangent is not None:
        add_direction(dirs, "switch_tangent_1_p", tangent, False)
        add_direction(dirs, "switch_tangent_1_m", -tangent, False)
        tangent2 = normalize_vec(np.cross(normal, tangent))
        if tangent2 is not None:
            add_direction(dirs, "switch_tangent_2_p", tangent2, False)
            add_direction(dirs, "switch_tangent_2_m", -tangent2, False)
    return dirs


def deterministic_control_directions(p: Dict[str, Any], eq: np.ndarray) -> List[Tuple[str, np.ndarray, bool]]:
    dirs: List[Tuple[str, np.ndarray, bool]] = []
    axes = np.eye(3, dtype=float)
    for label, vec in zip(["x", "y", "z"], axes):
        add_direction(dirs, f"axis_p_{label}", vec, False)
        add_direction(dirs, f"axis_m_{label}", -vec, False)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                add_direction(dirs, f"diag_{int(sx):+d}_{int(sy):+d}_{int(sz):+d}", np.asarray([sx, sy, sz], dtype=float), False)
    for label, vec, rnd in dominant_eigen_directions(p, eq):
        add_direction(dirs, label, vec, rnd)
    for label, vec, rnd in switching_surface_directions(p, eq):
        add_direction(dirs, label, vec, rnd)
    return dirs


def random_unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1.0e-14:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)
    return v / n


def build_control_plan(
    candidate: Dict[str, Any],
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
    random_per_radius: int,
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for eq_idx, eq_id in enumerate(["E+", "E0"]):
        eq = np.asarray(eqs[eq_id], dtype=float)
        deterministic = deterministic_control_directions(p, eq)
        for radius_idx, radius in enumerate(CONTROL_RADII):
            dirs = list(deterministic)
            rng = np.random.default_rng(20260513 + 7919 * eq_idx + 313 * radius_idx)
            for j in range(int(random_per_radius)):
                dirs.append((f"rand_{j:02d}", random_unit(rng), True))
            # Keep the low-cost contract: about 20-24 trajectories per equilibrium/radius.
            dirs = dirs[:24]
            for label, direction, random_flag in dirs:
                x0 = eq + float(radius) * direction
                run_id = f"{eq_id}_{radius:.0e}_{label}_A".replace("+", "plus").replace("-", "minus")
                plan.append(
                    {
                        "candidate": candidate,
                        "candidate_id": candidate["candidate_id"],
                        "equilibrium_id": eq_id,
                        "radius": float(radius),
                        "direction_label": label,
                        "direction": direction,
                        "random_direction": bool(random_flag),
                        "stage": "control_A",
                        "run_id": run_id,
                        "x0": x0,
                        **CONTROL_STAGE_A,
                    }
                )
    return plan


def processed_control_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        str(row.get("candidate_id", "")),
        str(row.get("equilibrium_id", "")),
        f"{as_float(row.get('radius')):.17g}",
        str(row.get("direction_label", "")),
        str(row.get("stage", "")),
    )


def control_item_key(item: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        str(item.get("candidate_id", "")),
        str(item.get("equilibrium_id", "")),
        f"{float(item.get('radius')):.17g}",
        str(item.get("direction_label", "")),
        str(item.get("stage", "")),
    )


def run_control_item(
    item: Dict[str, Any],
    candidate: Dict[str, Any],
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    ref_cache: Dict[Tuple[str, float, float, float], np.ndarray],
    traj_dir: Path,
) -> Tuple[Dict[str, Any], str | None]:
    params = {"h": float(item["h"]), "memory_length": float(item["memory_length"]), "t_final": float(item["t_final"])}
    row_cls, traj = classify_trajectory(
        candidate=candidate,
        x0=np.asarray(item["x0"], dtype=float),
        params=params,
        p=p,
        eqs=eqs,
        cfg=cfg,
        ref_cache=ref_cache,
        notes=f"{item['stage']}; eq={item['equilibrium_id']}; radius={item['radius']}; direction={item['direction_label']}",
    )
    direction = np.asarray(item["direction"], dtype=float)
    x0 = np.asarray(item["x0"], dtype=float)
    row = {
        "candidate_id": candidate["candidate_id"],
        "q": candidate["q_float"],
        "rho_H": candidate.get("rho_H", ""),
        "rhoH_class": candidate.get("rhoH_class", ""),
        "equilibrium_id": item["equilibrium_id"],
        "radius": float(item["radius"]),
        "direction_label": item["direction_label"],
        "direction_x": float(direction[0]),
        "direction_y": float(direction[1]),
        "direction_z": float(direction[2]),
        "random_direction": bool(item["random_direction"]),
        "stage": item["stage"],
        "run_id": item["run_id"],
        "x0": float(x0[0]),
        "y0": float(x0[1]),
        "z0": float(x0[2]),
        "h": params["h"],
        "memory_length": params["memory_length"],
        "memory_points": memory_points(params["memory_length"], params["h"]),
        "t_final": params["t_final"],
        **row_cls,
    }
    traj_path: str | None = None
    if traj is not None:
        safe = str(item["run_id"]).replace("/", "_").replace("\\", "_").replace(".", "p")
        traj_path = save_reduced_trajectory(traj_dir / f"{safe}.csv", traj, max_points=420, burn_fraction=0.0)
    return row, traj_path


def aggregate_control_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, float, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("candidate_id", "")), str(row.get("equilibrium_id", "")), as_float(row.get("radius")), str(row.get("stage", "")))].append(row)
    out: List[Dict[str, Any]] = []
    for key, group in grouped.items():
        classes = Counter(str(r.get("final_class", "")) for r in group)
        n = len(group)
        out.append(
            {
                "candidate_id": key[0],
                "equilibrium_id": key[1],
                "radius": key[2],
                "stage": key[3],
                "n_samples": n,
                "n_equilibrium_convergence": classes.get("equilibrium_convergence", 0),
                "n_target_attractor": classes.get("target_attractor", 0),
                "n_other_bounded_nontrivial": classes.get("other_bounded_nontrivial", 0),
                "n_divergent": classes.get("divergent", 0),
                "n_numerical_failure": classes.get("numerical_failure", 0),
                "n_ambiguous_long_transient": classes.get("ambiguous_long_transient", 0),
                "target_hit_fraction": float(classes.get("target_attractor", 0) / max(n, 1)),
                "most_common_class": classes.most_common(1)[0][0] if classes else "",
            }
        )
    out.sort(key=lambda r: (r["candidate_id"], r["equilibrium_id"], float(r["radius"]), r["stage"]))
    return out


def control_decision(candidate: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    targets = [r for r in rows if truthy(r.get("target_hit"))]
    eplus_hits = sum(1 for r in targets if r.get("equilibrium_id") == "E+")
    e0_hits = sum(1 for r in targets if r.get("equilibrium_id") == "E0")
    same_group_repeats = 0
    group_counts: Dict[Tuple[str, float], int] = defaultdict(int)
    for r in targets:
        if r.get("stage") == "control_A":
            group_counts[(str(r.get("equilibrium_id")), as_float(r.get("radius")))] += 1
    same_group_repeats = sum(1 for count in group_counts.values() if count >= 2)
    reproduced_h = sum(1 for r in targets if r.get("stage") == "refine_B")
    reproduced_mem = sum(1 for r in targets if r.get("stage") == "refine_C")
    robust = bool(same_group_repeats > 0 or reproduced_h > 0 or reproduced_mem > 0)
    if robust:
        status = "not_supported_by_Eplus_E0_control_test"
        notes = "A robust or reproduced TARGET contact appeared from E+ or E0."
    elif targets:
        status = "inconclusive_isolated_hit_Eplus_E0"
        notes = "Only isolated unreproduced TARGET contacts appeared from E+ or E0."
    else:
        status = "compatible_with_hiddenness_under_Eplus_E0_control_test"
        notes = "No TARGET contacts appeared from E+ or E0. This is not hidden_verified."
    return {
        "candidate_id": candidate["candidate_id"],
        "Eplus_target_hits": eplus_hits,
        "E0_target_hits": e0_hits,
        "target_hits": len(targets),
        "same_equilibrium_radius_target_repeats": same_group_repeats,
        "reproduced_h_target_hits": reproduced_h,
        "reproduced_memory_target_hits": reproduced_mem,
        "robust_control_target_hit": bool(robust),
        "hiddenness_status": status,
        "decision_notes": notes,
    }


def run_controls(
    args: argparse.Namespace,
    candidate: Dict[str, Any],
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    outdir: Path,
    measured_first_sec: float | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], List[str], Dict[str, Any]]:
    raw_path = outdir / "eplus_e0_control_raw.csv"
    traj_dir = outdir / "trajectories" / "control_samples"
    raw_rows: List[Dict[str, Any]] = [dict(r) for r in read_rows(raw_path)] if args.resume else []
    if not args.resume and raw_path.exists():
        raw_path.unlink()
    done = {processed_control_key(r) for r in raw_rows}
    plan = build_control_plan(candidate, p, eqs, int(args.random_directions_per_radius))
    planned_initial = len(plan)
    if planned_initial > int(args.max_trajectories) and not args.force:
        raise RuntimeError(f"Cost guard: {planned_initial} control trajectories exceed --max-trajectories={args.max_trajectories}. Use --force or lower directions.")
    control_est = float("nan")
    if measured_first_sec and math.isfinite(measured_first_sec) and measured_first_sec > 0.0:
        r0 = ROBUSTNESS_CASES["R0"]
        scale = (CONTROL_STAGE_A["t_final"] / CONTROL_STAGE_A["h"]) / max((r0["t_final"] / r0["h"]), 1.0)
        control_est = float(measured_first_sec * scale * planned_initial)
        if control_est > float(args.max_estimated_hours) * 3600.0 and not args.force:
            raise RuntimeError(
                f"Cost guard: estimated control runtime {control_est / 3600.0:.3f} h exceeds --max-estimated-hours={args.max_estimated_hours}. Use --force."
            )
    ref_cache: Dict[Tuple[str, float, float, float], np.ndarray] = {}
    files = [str(raw_path)]
    measured_control_first: float | None = None
    for item in plan:
        if control_item_key(item) in done:
            continue
        t0_item = time.time()
        row, traj_path = run_control_item(item, candidate, p, eqs, cfg, ref_cache, traj_dir)
        item_elapsed = time.time() - t0_item
        raw_rows.append(row)
        append_csv(raw_path, row, CONTROL_RAW_FIELDS)
        if traj_path:
            files.append(traj_path)
        print(f"{candidate['candidate_id']} {item['equilibrium_id']} radius={item['radius']:.1e} dir={item['direction_label']} class={row['final_class']} target={row['target_hit']}", flush=True)
        if measured_control_first is None:
            measured_control_first = item_elapsed
            measured_total = measured_control_first * planned_initial
            if not math.isfinite(control_est):
                control_est = measured_total
            if measured_total > float(args.max_estimated_hours) * 3600.0 and not args.force:
                raise RuntimeError(
                    f"Cost guard after first control trajectory: estimated total {measured_total / 3600.0:.3f} h exceeds "
                    f"--max-estimated-hours={args.max_estimated_hours}. Re-run with --resume --force to continue."
                )

    # Target reproductions only for observed TARGET points.
    rows_by_key = {processed_control_key(r): r for r in raw_rows}
    target_a = [r for r in raw_rows if r.get("stage") == "control_A" and truthy(r.get("target_hit"))]
    for target in target_a:
        direction = np.asarray([as_float(target.get("direction_x")), as_float(target.get("direction_y")), as_float(target.get("direction_z"))], dtype=float)
        x0 = np.asarray([as_float(target.get("x0")), as_float(target.get("y0")), as_float(target.get("z0"))], dtype=float)
        base = {
            "candidate": candidate,
            "candidate_id": candidate["candidate_id"],
            "equilibrium_id": target["equilibrium_id"],
            "radius": as_float(target.get("radius")),
            "direction_label": str(target.get("direction_label", "")),
            "direction": direction,
            "random_direction": truthy(target.get("random_direction")),
            "x0": x0,
        }
        item_b = {**base, "stage": "refine_B", "run_id": f"{target.get('run_id')}_B", **CONTROL_STAGE_B}
        if control_item_key(item_b) not in rows_by_key:
            row_b, traj_path = run_control_item(item_b, candidate, p, eqs, cfg, ref_cache, traj_dir)
            raw_rows.append(row_b)
            append_csv(raw_path, row_b, CONTROL_RAW_FIELDS)
            rows_by_key[control_item_key(item_b)] = row_b
            if traj_path:
                files.append(traj_path)
            print(f"{candidate['candidate_id']} reproduce_B {target['equilibrium_id']} radius={as_float(target.get('radius')):.1e} class={row_b['final_class']} target={row_b['target_hit']}", flush=True)
        else:
            row_b = rows_by_key[control_item_key(item_b)]
        if truthy(row_b.get("target_hit")):
            item_c = {**base, "stage": "refine_C", "run_id": f"{target.get('run_id')}_C", **CONTROL_STAGE_C}
            if control_item_key(item_c) not in rows_by_key:
                row_c, traj_path = run_control_item(item_c, candidate, p, eqs, cfg, ref_cache, traj_dir)
                raw_rows.append(row_c)
                append_csv(raw_path, row_c, CONTROL_RAW_FIELDS)
                rows_by_key[control_item_key(item_c)] = row_c
                if traj_path:
                    files.append(traj_path)
                print(f"{candidate['candidate_id']} reproduce_C {target['equilibrium_id']} radius={as_float(target.get('radius')):.1e} class={row_c['final_class']} target={row_c['target_hit']}", flush=True)

    summary = aggregate_control_rows(raw_rows)
    decision = control_decision(candidate, raw_rows)
    write_csv(outdir / "eplus_e0_control_summary.csv", summary, CONTROL_SUMMARY_FIELDS)
    write_csv(outdir / "eplus_e0_control_decision.csv", [decision], CONTROL_DECISION_FIELDS)
    files.extend([str(outdir / "eplus_e0_control_summary.csv"), str(outdir / "eplus_e0_control_decision.csv")])
    cost = {
        "planned_control_trajectories": planned_initial,
        "max_trajectories": int(args.max_trajectories),
        "random_directions_per_radius": int(args.random_directions_per_radius),
        "estimated_control_sec_from_first_robustness": control_est,
        "measured_first_control_trajectory_sec": measured_control_first,
        "max_estimated_hours": float(args.max_estimated_hours),
        "executed_control_rows": len(raw_rows),
    }
    json_path = outdir / "eplus_e0_control_summary.json"
    json_path.write_text(
        json.dumps(
            json_safe(
                {
                    "candidate_id": candidate["candidate_id"],
                    "summary": summary,
                    "decision": decision,
                    "cost_guard": cost,
                    "scientific_note": "TARGET uses the same sectional fingerprint logic as the Lure targeted route. hidden_verified is never declared here.",
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    files.append(str(json_path))
    return raw_rows, summary, decision, files, cost


def previous_eminus_status(candidate_id: str) -> Dict[str, Any]:
    summary = read_rows(ADAPTIVE_EMINUS_DIR / "adaptive_contact_summary.csv")
    decision = read_rows(ADAPTIVE_EMINUS_DIR / "adaptive_contact_decision.csv")
    srow = next((r for r in summary if r.get("candidate_id") == candidate_id), {})
    drow = next((r for r in decision if r.get("candidate_id") == candidate_id), {})
    return {
        "Eminus_status_previous": drow.get("hiddenness_status", srow.get("hiddenness_status", "")),
        "Eminus_adaptive_target_hits": as_int(srow.get("target_hits", 0)),
        "Eminus_robust_target_hit": truthy(srow.get("robust_target_hit", False)),
    }


def combined_status(
    candidate: Dict[str, Any],
    robustness_summary: Dict[str, Any] | None,
    control_dec: Dict[str, Any] | None,
) -> Dict[str, Any]:
    prev = previous_eminus_status(candidate["candidate_id"])
    robust_attractor = truthy((robustness_summary or {}).get("robust_attractor", False))
    robust_status = str((robustness_summary or {}).get("robustness_status", "not_evaluated"))
    eplus_hits = as_int((control_dec or {}).get("Eplus_target_hits", 0))
    e0_hits = as_int((control_dec or {}).get("E0_target_hits", 0))
    robust_control = truthy((control_dec or {}).get("robust_control_target_hit", False))
    target_hits = eplus_hits + e0_hits
    if robustness_summary is None or control_dec is None:
        final_status = "incomplete_requested_skip"
        next_action = "run_missing_robustness_or_control_stage"
    elif not robust_attractor:
        final_status = "not_ready_due_to_numerical_nonrobustness"
        next_action = "improve_integration_or_memory_before_hiddenness_claim"
    elif robust_control:
        final_status = "not_supported_as_hidden"
        next_action = "document_as_self_excited_or_nonhidden_candidate"
    elif target_hits == 0:
        final_status = "strongest_current_candidate_but_not_verified"
        next_action = "compare_against_Machado_and_run_biased_DF_local_search"
    else:
        final_status = "inconclusive_needs_targeted_reproduction"
        next_action = "reproduce_isolated_hits_only"
    return {
        "candidate_id": candidate["candidate_id"],
        "q": candidate["q_float"],
        "rho_H": candidate.get("rho_H", ""),
        "rhoH_class": candidate.get("rhoH_class", ""),
        "Eminus_status_previous": prev["Eminus_status_previous"],
        "Eminus_adaptive_target_hits": prev["Eminus_adaptive_target_hits"],
        "robustness_status": robust_status,
        "robust_attractor": bool(robust_attractor),
        "Eplus_target_hits": eplus_hits,
        "E0_target_hits": e0_hits,
        "robust_control_target_hit": bool(robust_control),
        "final_recommended_status": final_status,
        "hidden_verified": False,
        "next_action": next_action,
    }


def plot_robustness(outdir: Path, raw_rows: Sequence[Dict[str, Any]]) -> List[str]:
    plotdir = outdir / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)
    traj_dir = outdir / "trajectories"
    files: List[str] = []
    colors = {"R0": "#16a34a", "R1": "#2563eb", "R2": "#9333ea", "R3": "#f59e0b"}
    trajectories: Dict[str, np.ndarray] = {}
    for run_id in ["R0", "R1", "R2", "R3"]:
        path = traj_dir / f"attractor_{run_id}.csv"
        if path.exists():
            arr = load_trajectory_sample(path, max_points=9000)
            if arr.shape[0] > 0:
                trajectories[run_id] = arr
    if trajectories:
        for plane, cols, fname in [
            ("xy", (1, 2), "attractor_robustness_overlay_xy.png"),
            ("xz", (1, 3), "attractor_robustness_overlay_xz.png"),
        ]:
            fig, ax = plt.subplots(figsize=(6.4, 5.0))
            for run_id, arr in trajectories.items():
                ax.plot(arr[:, cols[0]], arr[:, cols[1]], lw=0.55, alpha=0.82, color=colors.get(run_id, None), label=run_id)
            ax.set_xlabel("x")
            ax.set_ylabel("y" if plane == "xy" else "z")
            ax.grid(True, alpha=0.22)
            ax.legend(frameon=True, fontsize=8)
            fig.tight_layout()
            path = plotdir / fname
            fig.savefig(path, dpi=220)
            plt.close(fig)
            files.append(str(path))
        fig = plt.figure(figsize=(6.4, 5.4))
        ax = fig.add_subplot(111, projection="3d")
        for run_id, arr in trajectories.items():
            ax.plot(arr[:, 1], arr[:, 2], arr[:, 3], lw=0.5, alpha=0.78, color=colors.get(run_id, None), label=run_id)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(frameon=True, fontsize=8)
        fig.tight_layout()
        path = plotdir / "attractor_robustness_overlay_3d.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        files.append(str(path))
    if raw_rows:
        labels = [str(r.get("run_id", "")) for r in raw_rows]
        ranges = np.asarray([[as_float(r.get("range_x")), as_float(r.get("range_y")), as_float(r.get("range_z"))] for r in raw_rows], dtype=float)
        vars_ = np.asarray([[as_float(r.get("var_x_tail")), as_float(r.get("var_y_tail")), as_float(r.get("var_z_tail"))] for r in raw_rows], dtype=float)
        x = np.arange(len(labels))
        fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.9))
        width = 0.25
        for j, comp in enumerate(["x", "y", "z"]):
            axs[0].bar(x + (j - 1) * width, ranges[:, j], width=width, label=comp)
            axs[1].bar(x + (j - 1) * width, vars_[:, j], width=width, label=comp)
        axs[0].set_ylabel("range")
        axs[1].set_ylabel("tail variance")
        for ax in axs:
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.grid(True, axis="y", alpha=0.22)
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = plotdir / "robustness_invariants.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        files.append(str(path))
    return files


def plot_controls(outdir: Path, summary_rows: Sequence[Dict[str, Any]], raw_rows: Sequence[Dict[str, Any]], candidate: Dict[str, Any], eqs: Dict[str, np.ndarray], p: Dict[str, Any]) -> List[str]:
    plotdir = outdir / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)
    files: List[str] = []
    if summary_rows:
        fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.1), sharey=True)
        for ax, eq_id in zip(axs, ["E+", "E0"]):
            sub = [r for r in summary_rows if r.get("equilibrium_id") == eq_id and r.get("stage") == "control_A"]
            radii = sorted({as_float(r.get("radius")) for r in sub})
            bottom = np.zeros(len(radii))
            x = np.arange(len(radii))
            for cls in CLASS_ORDER:
                vals = []
                for radius in radii:
                    row = next((r for r in sub if math.isclose(as_float(r.get("radius")), radius, rel_tol=0.0, abs_tol=1e-18)), {})
                    vals.append(as_int(row.get(f"n_{cls}", 0)))
                ax.bar(x, vals, bottom=bottom, color=CLASS_COLORS.get(cls, "#999999"), label=cls if eq_id == "E+" else None)
                bottom += np.asarray(vals, dtype=float)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{r:.0e}" for r in radii])
            ax.set_xlabel("radius")
            ax.set_title(eq_id)
            ax.grid(True, axis="y", alpha=0.22)
        axs[0].set_ylabel("trajectory count")
        fig.legend(loc="upper center", ncol=3, fontsize=7.5, frameon=True)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        path = plotdir / "eplus_e0_control_classes_vs_radius.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        files.append(str(path))

    fig = plt.figure(figsize=(7.0, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    attr = load_trajectory_sample(outdir / "trajectories" / "attractor_R0.csv", max_points=12000)
    if attr.shape[0] == 0:
        try:
            attr = source_or_reconstructed_attractor(candidate, p)
            if attr.shape[0] > 12000:
                idx = np.linspace(0, attr.shape[0] - 1, 12000).astype(int)
                attr = attr[idx]
        except Exception:
            attr = np.empty((0, 4), dtype=float)
    if attr.shape[0] > 0:
        ax.plot(attr[:, 1], attr[:, 2], attr[:, 3], color="#16a34a", lw=0.45, alpha=0.9, label="candidate attractor")
    eq_colors = {"E-": "#111827", "E0": "#7c3aed", "E+": "#2563eb"}
    for eq_id in ["E-", "E0", "E+"]:
        eq = np.asarray(eqs[eq_id], dtype=float)
        ax.scatter([eq[0]], [eq[1]], [eq[2]], s=42, color=eq_colors[eq_id], label=eq_id)
    for eq_id in ["E+", "E0"]:
        eq = np.asarray(eqs[eq_id], dtype=float)
        for radius in CONTROL_RADII:
            draw_wire_sphere(ax, eq, radius, color=eq_colors[eq_id], alpha=0.10)
    sample_dir = outdir / "trajectories" / "control_samples"
    plotted = 0
    for row in raw_rows:
        if plotted >= 80:
            break
        run_id = str(row.get("run_id", "")).replace("/", "_").replace("\\", "_").replace(".", "p")
        path = sample_dir / f"{run_id}.csv"
        arr = load_trajectory_sample(path, max_points=420)
        if arr.shape[0] == 0:
            continue
        color = "#dc2626" if truthy(row.get("target_hit")) else ("#2563eb" if row.get("equilibrium_id") == "E+" else "#7c3aed")
        alpha = 0.82 if truthy(row.get("target_hit")) else 0.22
        lw = 0.9 if truthy(row.get("target_hit")) else 0.42
        label = None
        if truthy(row.get("target_hit")):
            label = "TARGET contact"
        ax.plot(arr[:, 1], arr[:, 2], arr[:, 3], color=color, alpha=alpha, lw=lw, label=label)
        plotted += 1
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    handles, labels = ax.get_legend_handles_labels()
    dedup: Dict[str, Any] = {}
    for h, label in zip(handles, labels):
        if label and label not in dedup:
            dedup[label] = h
    ax.legend(dedup.values(), dedup.keys(), fontsize=7.5, frameon=True, loc="best")
    fig.tight_layout()
    png = plotdir / "lure_candidate_with_Eplus_E0_controls_3d.png"
    pdf = plotdir / "lure_candidate_with_Eplus_E0_controls_3d.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    files.extend([str(png), str(pdf)])
    return files


def draw_wire_sphere(ax: Any, center: np.ndarray, radius: float, color: str, alpha: float) -> None:
    u = np.linspace(0, 2 * math.pi, 18)
    v = np.linspace(0, math.pi, 9)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, rstride=2, cstride=2, color=color, alpha=alpha, linewidth=0.35)


def write_report(
    outdir: Path,
    candidate: Dict[str, Any],
    robustness_rows: Sequence[Dict[str, Any]],
    robustness_summary: Dict[str, Any] | None,
    control_summary: Sequence[Dict[str, Any]],
    control_dec: Dict[str, Any] | None,
    status: Dict[str, Any],
    cost: Dict[str, Any],
    files: Sequence[str],
) -> str:
    prev = previous_eminus_status(candidate["candidate_id"])
    lines: List[str] = [
        "# Lure robustness and E+/E0 controls",
        "",
        "## Candidate",
        "",
        f"- candidate_id: `{candidate['candidate_id']}`",
        f"- df_family: `{candidate.get('df_family', '')}`",
        f"- q: `{candidate.get('q_float', '')}`",
        f"- rho_H: `{candidate.get('rho_H', '')}`",
        f"- rhoH_class: `{candidate.get('rhoH_class', '')}`",
        "",
        "## Previous E- State",
        "",
        f"- previous status: `{prev.get('Eminus_status_previous', '')}`",
        f"- adaptive E- target hits: `{prev.get('Eminus_adaptive_target_hits', '')}`",
        "",
        "## Numerical Robustness Parameters",
        "",
        "| run | h | memory_length | t_final |",
        "|---|---:|---:|---:|",
    ]
    for run_id, params in ROBUSTNESS_CASES.items():
        lines.append(f"| {run_id} | {params['h']} | {params['memory_length']} | {params['t_final']} |")
    lines.extend(["", "## Robustness Invariants", "", "| run | bounded | final_norm | max_norm | range_x | range_y | range_z | fft_peak_x | crossings |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in robustness_rows:
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('bounded', '')} | {row.get('final_norm', '')} | {row.get('max_norm', '')} | "
            f"{row.get('range_x', '')} | {row.get('range_y', '')} | {row.get('range_z', '')} | {row.get('fft_peak_x', '')} | {row.get('number_of_section_crossings', '')} |"
        )
    lines.extend(
        [
            "",
            "## Robustness Decision",
            "",
            f"- robust_attractor: `{(robustness_summary or {}).get('robust_attractor', 'not_evaluated')}`",
            f"- robustness_status: `{(robustness_summary or {}).get('robustness_status', 'not_evaluated')}`",
            "",
            "## E+ and E0 Controls",
            "",
            "| equilibrium | radius | stage | samples | target_hits | most_common_class |",
            "|---|---:|---|---:|---:|---|",
        ]
    )
    for row in control_summary:
        lines.append(
            f"| {row.get('equilibrium_id', '')} | {row.get('radius', '')} | {row.get('stage', '')} | "
            f"{row.get('n_samples', '')} | {row.get('n_target_attractor', '')} | {row.get('most_common_class', '')} |"
        )
    lines.extend(
        [
            "",
            "## Control Decision",
            "",
            f"- Eplus_target_hits: `{(control_dec or {}).get('Eplus_target_hits', 0)}`",
            f"- E0_target_hits: `{(control_dec or {}).get('E0_target_hits', 0)}`",
            f"- robust_control_target_hit: `{(control_dec or {}).get('robust_control_target_hit', False)}`",
            f"- control hiddenness_status: `{(control_dec or {}).get('hiddenness_status', 'not_evaluated')}`",
            "",
            "## Combined Decision",
            "",
            f"- final_recommended_status: `{status.get('final_recommended_status')}`",
            f"- hidden_verified: `{status.get('hidden_verified')}`",
            f"- next_action: `{status.get('next_action')}`",
            "",
            "## Cost Guard",
            "",
            f"- planned robustness trajectories: `{cost.get('planned_robustness_trajectories', '')}`",
            f"- planned control trajectories: `{cost.get('planned_control_trajectories', '')}`",
            f"- max trajectories without force: `{cost.get('max_trajectories', '')}`",
            f"- estimated control seconds: `{cost.get('estimated_control_sec_from_first_robustness', '')}`",
            "",
            "## Scientific Note",
            "",
            "The describing function generated only the seed. Validation here is by causal Caputo memory integration. No `hidden_verified` label is declared.",
            "",
            "## Files",
            "",
        ]
    )
    for f in files:
        lines.append(f"- `{f}`")
    path = outdir / "lure_robustness_and_controls_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-cost robustness and E+/E0 control tests for the best Lure candidate.")
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-trajectories", type=int, default=200)
    parser.add_argument("--max-estimated-hours", type=float, default=8.0)
    parser.add_argument("--random-directions-per-radius", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-long", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = output_path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    (outdir / "trajectories").mkdir(parents=True, exist_ok=True)

    candidate = load_candidate(cfg, args.candidate_id)
    p = chua_params()
    chua.QORD = np.float64(candidate["q_float"])
    eqs, _eq_rows = load_or_recompute_equilibria(p, float(candidate["q_float"]), outdir)

    planned_controls = len(build_control_plan(candidate, p, eqs, int(args.random_directions_per_radius))) if not args.skip_controls else 0
    planned_robustness = 0 if args.skip_robustness else 4
    if planned_controls > int(args.max_trajectories) and not args.force:
        raise RuntimeError(f"Cost guard: {planned_controls} planned control trajectories exceed --max-trajectories={args.max_trajectories}. Use --force.")
    print(f"candidate_id={candidate['candidate_id']}", flush=True)
    print(f"planned_robustness_trajectories={planned_robustness}", flush=True)
    print(f"planned_control_trajectories={planned_controls}", flush=True)

    all_files: List[str] = [str(outdir / "equilibria_used.csv")]
    robustness_rows: List[Dict[str, Any]] = []
    robustness_summary: Dict[str, Any] | None = None
    measured_first: float | None = None
    robust_cost: Dict[str, Any] = {"planned_robustness_trajectories": planned_robustness}
    if not args.skip_robustness:
        robustness_rows, robustness_summary, files, robust_cost = run_robustness(args, candidate, p, eqs, cfg, outdir)
        all_files.extend(files)
        if robustness_rows:
            measured_first = as_float(robustness_rows[0].get("integration_sec"))
    else:
        print("skip_robustness=True", flush=True)

    control_rows: List[Dict[str, Any]] = []
    control_summary: List[Dict[str, Any]] = []
    control_dec: Dict[str, Any] | None = None
    control_cost: Dict[str, Any] = {"planned_control_trajectories": planned_controls, "max_trajectories": int(args.max_trajectories)}
    if not args.skip_controls:
        control_rows, control_summary, control_dec, files, control_cost = run_controls(args, candidate, p, eqs, cfg, outdir, measured_first)
        all_files.extend(files)
    else:
        print("skip_controls=True", flush=True)

    plot_files = []
    if robustness_rows:
        plot_files.extend(plot_robustness(outdir, robustness_rows))
    if control_rows or control_summary:
        plot_files.extend(plot_controls(outdir, control_summary, control_rows, candidate, eqs, p))
    all_files.extend(plot_files)

    status = combined_status(candidate, robustness_summary, control_dec)
    status_path = outdir / "lure_candidate_status_update.csv"
    write_csv(status_path, [status], STATUS_FIELDS)
    all_files.append(str(status_path))

    merged_cost = {**robust_cost, **control_cost, "max_trajectories": int(args.max_trajectories)}
    report_path = write_report(outdir, candidate, robustness_rows, robustness_summary, control_summary, control_dec, status, merged_cost, all_files)
    all_files.append(report_path)

    print("candidate_id,robustness_status,robust_attractor,Eplus_target_hits,E0_target_hits,robust_control_target_hit,final_recommended_status,hidden_verified,files_written", flush=True)
    print(
        ",".join(
            [
                str(status["candidate_id"]),
                str(status["robustness_status"]),
                str(status["robust_attractor"]),
                str(status["Eplus_target_hits"]),
                str(status["E0_target_hits"]),
                str(status["robust_control_target_hit"]),
                str(status["final_recommended_status"]),
                str(status["hidden_verified"]),
                ";".join(all_files),
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
