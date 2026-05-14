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
from corrida1_refined_verification import (
    load_selected_candidates,
    section_points,
    simulate_and_classify,
    tail_stats,
)
from equilibria_analysis import local_jacobian, region_for_sigma, solve_equilibria
from extended_search_utils import chua_ic_params, json_safe, trajectory_ranges, write_csv
from lure_candidate_manifest import load_config


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "machado_targeted_verification.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "extended_search" / "machado_targeted_verification"
PROTECTED_INPUT_DIRS = [
    ROOT / "outputs" / "extended_search" / "corrida1",
    ROOT / "runs_machado_sweep_fast",
]
REQUIRED_CANDIDATES = [
    "branch_0_mu_4p00000_theta_0p00000",
    "branch_0_mu_2p00000_theta_3p92699",
]
EXPECTED_EQUILIBRIA = {
    "E-": np.array([-6.588307887, -0.002836402256, 6.585471484], dtype=float),
    "E0": np.array([0.0, 0.0, 0.0], dtype=float),
    "E+": np.array([6.588307887, 0.002836402256, -6.585471484], dtype=float),
}

EQ_FIELDS = ["eq_id", "x", "y", "z", "region", "eig_1", "eig_2", "eig_3", "matignon_margin", "matignon_stable"]

RAW_FIELDS = [
    "candidate_id",
    "q",
    "mu",
    "theta",
    "branch",
    "equilibrium_id",
    "radius",
    "direction_label",
    "direction_x",
    "direction_y",
    "direction_z",
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
    "final_class",
    "target_hit",
    "numerical_status",
    "notes",
]

SUMMARY_FIELDS = [
    "candidate_id",
    "equilibrium_id",
    "radius",
    "n_tests",
    "n_target",
    "n_equilibrium_convergence",
    "n_other_bounded_nontrivial",
    "n_divergent",
    "n_numerical_failure",
    "n_ambiguous_long_transient",
    "target_fraction",
]

REPRO_RAW_FIELDS = RAW_FIELDS + ["reproduction_stage", "source_run_key", "robust_target_hit"]
REPRO_SUMMARY_FIELDS = [
    "candidate_id",
    "source_run_key",
    "equilibrium_id",
    "radius",
    "direction_label",
    "n_reproductions",
    "n_reproduced_target",
    "robust_target_hit",
    "hiddenness_status",
    "notes",
]

ROBUST_RAW_FIELDS = [
    "candidate_id",
    "case_id",
    "q",
    "h",
    "memory_length",
    "memory_points",
    "t_final",
    "start_source",
    "start_x",
    "start_y",
    "start_z",
    "bounded",
    "final_class",
    "max_norm",
    "final_x",
    "final_y",
    "final_z",
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
    "section_fingerprint_count",
    "lyap_max_if_available",
    "numerical_status",
    "trajectory_csv",
    "notes",
]

REFERENCE_FIELDS = [
    "candidate_id",
    "q",
    "mu",
    "theta",
    "branch",
    "h",
    "memory_length",
    "memory_points",
    "t_final",
    "start_source",
    "start_x",
    "start_y",
    "start_z",
    "bounded",
    "final_class",
    "max_norm",
    "final_x",
    "final_y",
    "final_z",
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
    "section_fingerprint_count",
    "numerical_status",
    "trajectory_csv",
    "reference_status",
    "notes",
]

ROBUST_SUMMARY_FIELDS = [
    "candidate_id",
    "executed_cases",
    "all_bounded_nontrivial",
    "no_equilibrium_convergence",
    "no_divergence",
    "range_relative_spread",
    "fft_relative_spread",
    "var_tail_collapsed",
    "robust_attractor",
    "robustness_status",
    "notes",
]

DECISION_FIELDS = [
    "candidate_id",
    "q",
    "mu",
    "theta",
    "previous_completed_trajectories",
    "previous_target_hits",
    "targeted_equilibrium_tests",
    "targeted_target_hits",
    "reproduced_target_hits",
    "robust_target_hit",
    "Eminus_target_hits",
    "E0_target_hits",
    "Eplus_target_hits",
    "robustness_status",
    "robust_attractor",
    "final_recommended_status",
    "hidden_verified",
    "next_action",
]


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return base / path


def safe_output_dir(path: str | Path) -> Path:
    outdir = resolve_path(path)
    resolved = outdir.resolve()
    for protected in PROTECTED_INPUT_DIRS:
        try:
            resolved.relative_to(protected.resolve())
        except ValueError:
            continue
        raise ValueError(f"Refusing to write inside protected input directory: {rel(protected)}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    (outdir / "trajectories").mkdir(parents=True, exist_ok=True)
    return outdir


def read_rows(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def append_csv(path: str | Path, row: Dict[str, Any], fields: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: csv_value(row.get(key, "")) for key in fields})


def ensure_csv(path: str | Path, fields: Sequence[str]) -> None:
    path = Path(path)
    if not path.exists():
        write_csv(path, [], fields)


def csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return ";".join(str(float(x)) for x in value.ravel())
    if isinstance(value, (list, tuple)):
        return ";".join(str(x) for x in value)
    if isinstance(value, complex):
        return f"{value.real:.16g}{value.imag:+.16g}j"
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "si", "on"}


def memory_points(memory_length: float, h: float) -> int:
    return max(1, int(math.ceil(float(memory_length) / float(h)))) + 1


def enforce_memory_contract(cfg: Dict[str, Any]) -> None:
    """Validate the short-memory contract used for hardware-oriented tests.

    The EFORK implementation uses a finite memory horizon Lm to approximate the
    Caputo history.  For embedded targets such as STM32, Lm is part of the
    experimental hypothesis rather than a tunable accuracy knob; all hiddenness,
    reproduction, and robustness stages must therefore stay inside the same
    configured memory bound.
    """
    contract = cfg.get("memory_contract", {})
    if "max_memory_length" not in contract:
        return
    max_lm = float(contract["max_memory_length"])
    if max_lm <= 0.0:
        raise ValueError("memory_contract.max_memory_length must be positive.")
    checks: List[Tuple[str, float]] = []
    for section in ("reference_attractor", "targeted_equilibrium_controls"):
        params = cfg.get(section, {})
        if "memory_length" in params:
            checks.append((f"{section}.memory_length", float(params["memory_length"])))
    for idx, lm in enumerate(cfg.get("reproduction", {}).get("memory_length_values", []), start=1):
        checks.append((f"reproduction.memory_length_values[{idx}]", float(lm)))
    for case_id, params in cfg.get("attractor_robustness", {}).get("cases", {}).items():
        if "memory_length" in params:
            checks.append((f"attractor_robustness.cases.{case_id}.memory_length", float(params["memory_length"])))
    offenders = [(name, lm) for name, lm in checks if lm <= 0.0 or lm > max_lm]
    if offenders:
        details = "; ".join(f"{name}={lm:g}" for name, lm in offenders)
        raise ValueError(f"Memory contract violation: max Lm is {max_lm:g}, but {details}.")


def class_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    c = dict(cfg.get("classification", {}))
    return {
        "divergence_norm": float(c.get("divergence_norm", 120.0)),
        "equilibrium_radius": float(c.get("equilibrium_radius", 1e-3)),
        "section_tolerance": float(c.get("section_tolerance", 0.12)),
        "min_sec_match": int(c.get("min_section_matches", 20)),
        "min_section_matches": int(c.get("min_section_matches", 20)),
        "hit_fraction_required": float(c.get("hit_fraction_required", 0.70)),
        "tail_fraction": float(c.get("tail_fraction", 0.20)),
        "section_burn_fraction": 0.5,
        "reference_max_section_points": 240,
        "test_max_section_points": 100,
        "nontrivial_variance_tol": 1e-6,
        "nontrivial_range_tol": float(c.get("nontrivial_range_tol", 1e-2)),
    }


def load_source_data(cfg: Dict[str, Any], candidate_ids: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    source = resolve_path(cfg["source_summary"])
    candidates, summary = load_selected_candidates(source, candidate_ids)
    if abs(float(summary.get("frac_order", cfg.get("q", 0.9998))) - float(cfg["q"])) > 5e-10:
        raise ValueError(f"q mismatch between config and source summary: config={cfg['q']} source={summary.get('frac_order')}")
    params = {
        "model": summary.get("model", {}).get("kind", "piecewise") if isinstance(summary.get("model"), dict) else "piecewise",
        "params": summary.get("params", {}),
    }
    return candidates, summary, params


def cfg_with_params(cfg: Dict[str, Any], source_params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    if "params" not in out:
        out["params"] = source_params.get("params", {})
    if "model" not in out:
        out["model"] = source_params.get("model", "piecewise")
    return out


def selected_candidate_ids(cfg: Dict[str, Any], raw_candidate_id: str) -> List[str]:
    configured = [str(v) for v in cfg.get("candidates", REQUIRED_CANDIDATES)]
    missing = [cid for cid in REQUIRED_CANDIDATES if cid not in configured]
    if missing:
        raise ValueError(f"Config must include required candidates: {missing}")
    if raw_candidate_id == "all":
        return configured
    if raw_candidate_id not in configured:
        raise ValueError(f"candidate-id must be one of {configured} or all.")
    return [raw_candidate_id]


def previous_stats(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    prev_dir = resolve_path(cfg.get("previous_corrida1_dir", "outputs/extended_search/corrida1"))
    decisions = {row.get("candidate_id", ""): row for row in read_rows(prev_dir / "refined_hiddenness_decision.csv")}
    raw = read_rows(prev_dir / "refined_candidate_verification_raw.csv")
    grouped: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"completed": 0, "target_hits": 0})
    for row in raw:
        cid = str(row.get("candidate_id", ""))
        grouped[cid]["completed"] += 1
        grouped[cid]["target_hits"] += int(truthy(row.get("target_hit")))
    out: Dict[str, Dict[str, Any]] = {}
    for cid in set(decisions) | set(grouped) | set(REQUIRED_CANDIDATES):
        out[cid] = {
            "previous_hiddenness_status": decisions.get(cid, {}).get("hiddenness_status", "not_evaluated_cost_guard"),
            "previous_completed_trajectories": int(grouped.get(cid, {}).get("completed", 0)),
            "previous_target_hits": int(grouped.get(cid, {}).get("target_hits", 0)),
            "decision_row": decisions.get(cid, {}),
        }
    return out


def write_equilibria(p: Dict[str, Any], q: float, outdir: Path) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    eqs = {k: np.asarray(v, dtype=float) for k, v in solve_equilibria(p).items()}
    theta = float(q) * math.pi / 2.0
    rows: List[Dict[str, Any]] = []
    checks: Dict[str, Any] = {}
    for eq_id in ["E-", "E0", "E+"]:
        eq = eqs.get(eq_id)
        if eq is None:
            continue
        J = local_jacobian(p, eq)
        eig = np.linalg.eigvals(J)
        margin = float(min(abs(np.angle(v)) - theta for v in eig))
        rows.append(
            {
                "eq_id": eq_id,
                "x": float(eq[0]),
                "y": float(eq[1]),
                "z": float(eq[2]),
                "region": region_for_sigma(float(eq[0])),
                "eig_1": complex(eig[0]),
                "eig_2": complex(eig[1]),
                "eig_3": complex(eig[2]),
                "matignon_margin": margin,
                "matignon_stable": bool(margin > 0.0),
            }
        )
        expected = EXPECTED_EQUILIBRIA.get(eq_id)
        checks[eq_id] = {
            "expected": expected.tolist() if expected is not None else None,
            "computed": eq.tolist(),
            "distance_to_expected": float(np.linalg.norm(eq - expected)) if expected is not None else None,
            "matches_expected_1e-6": bool(expected is not None and np.linalg.norm(eq - expected) <= 1e-6),
        }
    write_csv(outdir / "equilibria_targeted_summary.csv", rows, EQ_FIELDS)
    return eqs, rows, checks


def normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= 1e-300:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return arr / n


def format_diag_value(value: float) -> str:
    return "+1" if value > 0 else "-1"


def eigen_directions(p: Dict[str, Any], eq: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    J = local_jacobian(p, eq)
    vals, vecs = np.linalg.eig(J)
    order = sorted(range(len(vals)), key=lambda i: float(np.real(vals[i])), reverse=True)
    names = ["dominant", "second", "third"]
    dirs: List[Tuple[str, np.ndarray]] = []
    for name, idx in zip(names, order):
        vec = vecs[:, idx]
        real_vec = np.real(vec)
        imag_vec = np.imag(vec)
        use = real_vec if np.linalg.norm(real_vec) >= np.linalg.norm(imag_vec) else imag_vec
        use = normalize(use)
        dirs.append((f"eig_{name}_p", use))
        dirs.append((f"eig_{name}_m", -use))
    return dirs


def direction_set(eq_id: str, p: Dict[str, Any], eq: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    dirs: List[Tuple[str, np.ndarray]] = []
    basis = np.eye(3, dtype=float)
    for axis, vec in zip(["x", "y", "z"], basis):
        dirs.append((f"axis_p_{axis}", vec.copy()))
        dirs.append((f"axis_m_{axis}", -vec.copy()))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                label = f"diag_{format_diag_value(sx)}_{format_diag_value(sy)}_{format_diag_value(sz)}"
                dirs.append((label, normalize(np.array([sx, sy, sz], dtype=float))))
    dirs.extend(eigen_directions(p, eq))
    if eq_id == "E+":
        critical = {
            "axis_m_x",
            "axis_m_y",
            "axis_m_z",
            "diag_-1_-1_-1",
            "diag_-1_-1_+1",
            "diag_-1_+1_-1",
            "diag_+1_-1_-1",
            "eig_dominant_m",
        }
        labels = {label for label, _ in dirs}
        missing = sorted(critical.difference(labels))
        if missing:
            raise RuntimeError(f"Missing required E+ critical directions: {missing}")
    unique: List[Tuple[str, np.ndarray]] = []
    seen_labels: set[str] = set()
    for label, vec in dirs:
        vec = normalize(vec)
        if label in seen_labels:
            continue
        if any(np.linalg.norm(vec - old_vec) <= 1e-10 and label == old_label for old_label, old_vec in unique):
            continue
        seen_labels.add(label)
        unique.append((label, vec))
    return unique


def build_targeted_plan(
    candidates: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    controls = cfg["targeted_equilibrium_controls"]
    radii = [float(v) for v in controls.get("radii", [])]
    eq_ids = [str(v) for v in controls.get("equilibria", ["E-", "E0", "E+"])]
    plan: List[Dict[str, Any]] = []
    for cand in candidates:
        for eq_id in eq_ids:
            if eq_id not in eqs:
                continue
            eq = np.asarray(eqs[eq_id], dtype=float)
            for radius in radii:
                for label, direction in direction_set(eq_id, p, eq):
                    x0 = eq + float(radius) * direction
                    plan.append(
                        {
                            "candidate": cand,
                            "candidate_id": cand["candidate_id"],
                            "q": float(cand["q"]),
                            "equilibrium_id": eq_id,
                            "equilibrium": eq,
                            "radius": float(radius),
                            "sampling_mode": "targeted_direction",
                            "sample_id": label,
                            "direction_label": label,
                            "direction": direction,
                            "perturbation": float(radius) * direction,
                            "x0": x0,
                            "h": float(controls.get("h", 0.01)),
                            "memory_length": float(controls.get("memory_length", 40.0)),
                            "t_final": float(controls.get("t_final", 1500.0)),
                            "stage": "targeted_equilibrium",
                        }
                    )
    return plan


def raw_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str, str, str]:
    return (
        str(row.get("candidate_id", "")),
        str(row.get("equilibrium_id", "")),
        f"{as_float(row.get('radius')):.17g}",
        str(row.get("direction_label", row.get("sample_id", ""))),
        f"{as_float(row.get('h')):.17g}",
        f"{as_float(row.get('memory_length')):.17g}",
        f"{as_float(row.get('t_final')):.17g}",
    )


def item_key(item: Dict[str, Any]) -> Tuple[str, str, str, str, str, str, str]:
    return (
        str(item["candidate_id"]),
        str(item["equilibrium_id"]),
        f"{float(item['radius']):.17g}",
        str(item["direction_label"]),
        f"{float(item['h']):.17g}",
        f"{float(item['memory_length']):.17g}",
        f"{float(item['t_final']):.17g}",
    )


def targeted_row_from_corrida(row: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    direction = np.asarray(item["direction"], dtype=float)
    out = {
        "candidate_id": row.get("candidate_id", item["candidate_id"]),
        "q": row.get("q", item["q"]),
        "mu": row.get("candidate_mu", item["candidate"].get("mu", "")),
        "theta": row.get("candidate_theta", item["candidate"].get("theta", "")),
        "branch": row.get("candidate_branch", item["candidate"].get("branch", "")),
        "equilibrium_id": row.get("equilibrium_id", item["equilibrium_id"]),
        "radius": row.get("radius", item["radius"]),
        "direction_label": item["direction_label"],
        "direction_x": float(direction[0]),
        "direction_y": float(direction[1]),
        "direction_z": float(direction[2]),
        "x0": row.get("x0", ""),
        "y0": row.get("y0", ""),
        "z0": row.get("z0", ""),
        "h": row.get("h", item["h"]),
        "memory_length": row.get("memory_length", item["memory_length"]),
        "memory_points": row.get("memory_points", memory_points(item["memory_length"], item["h"])),
        "t_final": row.get("t_final", item["t_final"]),
        "final_x": row.get("final_x", ""),
        "final_y": row.get("final_y", ""),
        "final_z": row.get("final_z", ""),
        "final_norm": row.get("final_norm", ""),
        "range_x": row.get("range_x", ""),
        "range_y": row.get("range_y", ""),
        "range_z": row.get("range_z", ""),
        "mean_x_tail": row.get("mean_x_tail", ""),
        "mean_y_tail": row.get("mean_y_tail", ""),
        "mean_z_tail": row.get("mean_z_tail", ""),
        "var_x_tail": row.get("var_x_tail", ""),
        "var_y_tail": row.get("var_y_tail", ""),
        "var_z_tail": row.get("var_z_tail", ""),
        "fft_peak": row.get("fft_peak", ""),
        "psd_entropy": row.get("psd_entropy", ""),
        "final_class": row.get("final_class", ""),
        "target_hit": row.get("target_hit", False),
        "numerical_status": row.get("numerical_status", ""),
        "notes": row.get("notes", ""),
    }
    return out


def aggregate_targeted(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("candidate_id", "")), str(row.get("equilibrium_id", "")), as_float(row.get("radius")))].append(row)
    out: List[Dict[str, Any]] = []
    for (cid, eq_id, radius), group in grouped.items():
        classes = Counter(str(r.get("final_class", "")) for r in group)
        n = len(group)
        n_target = sum(1 for r in group if truthy(r.get("target_hit")) or str(r.get("final_class")) == "target_attractor")
        out.append(
            {
                "candidate_id": cid,
                "equilibrium_id": eq_id,
                "radius": radius,
                "n_tests": n,
                "n_target": n_target,
                "n_equilibrium_convergence": classes.get("equilibrium_convergence", 0),
                "n_other_bounded_nontrivial": classes.get("other_bounded_nontrivial", 0),
                "n_divergent": classes.get("divergent", 0),
                "n_numerical_failure": classes.get("numerical_failure", 0),
                "n_ambiguous_long_transient": classes.get("ambiguous_long_transient", 0),
                "target_fraction": float(n_target / max(n, 1)),
            }
        )
    out.sort(key=lambda r: (r["candidate_id"], r["equilibrium_id"], float(r["radius"])))
    return out


def run_targeted_controls(
    candidates: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
    outdir: Path,
    *,
    resume: bool,
    max_to_run: int | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    raw_path = outdir / "targeted_equilibrium_raw.csv"
    summary_path = outdir / "targeted_equilibrium_summary.csv"
    ensure_csv(raw_path, RAW_FIELDS)
    raw_rows = [dict(r) for r in read_rows(raw_path)] if resume else []
    if not resume:
        write_csv(raw_path, [], RAW_FIELDS)
        raw_rows = []
    done = {raw_key(r) for r in raw_rows}
    plan = build_targeted_plan(candidates, cfg, p, eqs)
    stop_after_target = bool(cfg.get("cost_guard", {}).get("stop_after_first_target_hit_in_controls", True))
    blocked_candidates = {
        str(r.get("candidate_id", ""))
        for r in raw_rows
        if stop_after_target and (truthy(r.get("target_hit")) or str(r.get("final_class", "")) == "target_attractor")
    }
    ref_cache: Dict[Tuple[str, float, float, float], Tuple[np.ndarray, str]] = {}
    c_cfg = class_config(cfg)
    executed_now = 0
    for item in plan:
        cid = str(item.get("candidate_id", ""))
        if cid in blocked_candidates:
            continue
        if item_key(item) in done:
            continue
        if max_to_run is not None and executed_now >= max_to_run:
            break
        row = simulate_and_classify(item, p, eqs, c_cfg, ref_cache)
        out_row = targeted_row_from_corrida(row, item)
        raw_rows.append(out_row)
        append_csv(raw_path, out_row, RAW_FIELDS)
        executed_now += 1
        print(
            f"{item['candidate_id']} {item['equilibrium_id']} rho={item['radius']:.1e} "
            f"{item['direction_label']} class={out_row['final_class']} target={out_row['target_hit']}",
            flush=True,
        )
        if stop_after_target and (truthy(out_row.get("target_hit")) or str(out_row.get("final_class", "")) == "target_attractor"):
            blocked_candidates.add(cid)
    summary = aggregate_targeted(raw_rows)
    write_csv(summary_path, summary, SUMMARY_FIELDS)
    return raw_rows, summary, executed_now


def source_run_key(row: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("candidate_id", "")),
            str(row.get("equilibrium_id", "")),
            f"{as_float(row.get('radius')):.17g}",
            str(row.get("direction_label", "")),
        ]
    )


def reproduction_item_from_target(row: Dict[str, Any], candidates_by_id: Dict[str, Dict[str, Any]], eqs: Dict[str, np.ndarray], params: Dict[str, float], stage: str) -> Dict[str, Any]:
    cid = str(row["candidate_id"])
    x0 = np.asarray([as_float(row.get("x0")), as_float(row.get("y0")), as_float(row.get("z0"))], dtype=float)
    radius = as_float(row.get("radius"))
    direction = np.asarray([as_float(row.get("direction_x")), as_float(row.get("direction_y")), as_float(row.get("direction_z"))], dtype=float)
    eq_id = str(row.get("equilibrium_id", ""))
    return {
        "candidate": candidates_by_id[cid],
        "candidate_id": cid,
        "q": float(candidates_by_id[cid]["q"]),
        "equilibrium_id": eq_id,
        "equilibrium": eqs[eq_id],
        "radius": radius,
        "sampling_mode": "target_reproduction",
        "sample_id": str(row.get("direction_label", "")),
        "direction_label": str(row.get("direction_label", "")),
        "direction": direction,
        "perturbation": radius * direction,
        "x0": x0,
        "h": float(params["h"]),
        "memory_length": float(params["memory_length"]),
        "t_final": float(params["t_final"]),
        "stage": stage,
    }


def run_reproduction(
    target_rows: Sequence[Dict[str, Any]],
    candidates_by_id: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
    outdir: Path,
    *,
    resume: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_path = outdir / "target_reproduction_raw.csv"
    summary_path = outdir / "target_reproduction_summary.csv"
    if not resume:
        write_csv(raw_path, [], REPRO_RAW_FIELDS)
    else:
        ensure_csv(raw_path, REPRO_RAW_FIELDS)
    raw_rows = [dict(r) for r in read_rows(raw_path)] if resume else []
    done = {(r.get("source_run_key"), r.get("reproduction_stage")) for r in raw_rows}
    repro = cfg.get("reproduction", {})
    if not bool(repro.get("enabled", True)):
        write_csv(summary_path, [], REPRO_SUMMARY_FIELDS)
        return raw_rows, []
    c_cfg = class_config(cfg)
    ref_cache: Dict[Tuple[str, float, float, float], Tuple[np.ndarray, str]] = {}
    stop_after_reproduced = bool(cfg.get("cost_guard", {}).get("stop_after_first_reproduced_target", True))
    blocked_repro_candidates = {
        str(r.get("candidate_id", ""))
        for r in raw_rows
        if stop_after_reproduced and (truthy(r.get("target_hit")) or str(r.get("final_class", "")) == "target_attractor")
    }
    for target in target_rows:
        cid = str(target.get("candidate_id", ""))
        if cid in blocked_repro_candidates:
            continue
        src_key = source_run_key(target)
        for idx, memory_length in enumerate(repro.get("memory_length_values", [40, 80]), start=1):
            stage = "B" if idx == 1 else "C"
            if (src_key, stage) in done:
                continue
            params = {"h": float(repro.get("h", 0.005)), "memory_length": float(memory_length), "t_final": float(repro.get("t_final", 3000.0))}
            item = reproduction_item_from_target(target, candidates_by_id, eqs, params, f"reproduction_{stage}")
            row = simulate_and_classify(item, p, eqs, c_cfg, ref_cache)
            out = targeted_row_from_corrida(row, item)
            out["reproduction_stage"] = stage
            out["source_run_key"] = src_key
            out["robust_target_hit"] = bool(truthy(out.get("target_hit")))
            raw_rows.append(out)
            append_csv(raw_path, out, REPRO_RAW_FIELDS)
            if stop_after_reproduced and out["robust_target_hit"]:
                blocked_repro_candidates.add(cid)
                break
    summary = aggregate_reproduction(raw_rows)
    write_csv(summary_path, summary, REPRO_SUMMARY_FIELDS)
    return raw_rows, summary


def aggregate_reproduction(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source_run_key", ""))].append(row)
    out: List[Dict[str, Any]] = []
    for key, group in grouped.items():
        first = group[0]
        reproduced = sum(1 for r in group if truthy(r.get("target_hit")) or str(r.get("final_class")) == "target_attractor")
        robust = reproduced > 0
        out.append(
            {
                "candidate_id": first.get("candidate_id", ""),
                "source_run_key": key,
                "equilibrium_id": first.get("equilibrium_id", ""),
                "radius": first.get("radius", ""),
                "direction_label": first.get("direction_label", ""),
                "n_reproductions": len(group),
                "n_reproduced_target": reproduced,
                "robust_target_hit": robust,
                "hiddenness_status": "not_supported_by_equilibrium_neighborhood_test" if robust else "inconclusive_isolated_hit",
                "notes": "TARGET reproduced in B/C." if robust else "Isolated TARGET did not reproduce in B/C.",
            }
        )
    out.sort(key=lambda r: (r["candidate_id"], r["source_run_key"]))
    return out


def integrate_original(x0: np.ndarray, p: Dict[str, Any], q: float, h: float, memory_length: float, t_final: float) -> np.ndarray:
    return chua.efork3_integrate(lambda x: chua.rhs_original(x, p), x0, qord=q, h=h, Lm=memory_length, t_f=t_final)


def save_reduced_trajectory(path: str | Path, traj: np.ndarray, max_points: int = 8000) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    X = np.asarray(traj, dtype=float)
    if X.ndim != 2 or X.shape[1] < 4:
        write_csv(path, [], ["t", "x", "y", "z"])
        return str(path)
    if X.shape[0] > max_points:
        idx = np.linspace(0, X.shape[0] - 1, max_points).astype(int)
        X = X[idx]
    write_csv(path, [{"t": r[0], "x": r[1], "y": r[2], "z": r[3]} for r in X[:, :4]], ["t", "x", "y", "z"])
    return str(path)


def classify_attractor_traj(traj: np.ndarray, p: Dict[str, Any], eqs: Dict[str, np.ndarray], q: float, h: float, cfg: Dict[str, Any]) -> Dict[str, Any]:
    X = np.asarray(traj, dtype=float)
    states = X[:, 1:4] if X.ndim == 2 and X.shape[1] >= 4 else np.empty((0, 3), dtype=float)
    if states.shape[0] == 0 or not np.all(np.isfinite(states)):
        return {"bounded": False, "final_class": "numerical_failure", "numerical_status": "nonfinite_or_empty"}
    c_cfg = class_config(cfg)
    norms = np.linalg.norm(states, axis=1)
    max_norm = float(np.max(norms))
    final = states[-1]
    ranges = trajectory_ranges(X)
    tail = tail_stats(X, h, c_cfg["tail_fraction"])
    section = section_points(X, p, 0.5 * float(X[-1, 0]), 10000)
    if max_norm > c_cfg["divergence_norm"]:
        final_class = "divergent"
        bounded = False
    else:
        bounded = True
        tail_mean = np.array([tail["mean_x_tail"], tail["mean_y_tail"], tail["mean_z_tail"]], dtype=float)
        eq_hit = False
        for eq in eqs.values():
            if np.linalg.norm(final - eq) <= c_cfg["equilibrium_radius"] or np.linalg.norm(tail_mean - eq) <= 2.0 * c_cfg["equilibrium_radius"]:
                eq_hit = True
                break
        if eq_hit:
            final_class = "equilibrium_convergence"
        elif max(float(ranges["range_x"]), float(ranges["range_y"]), float(ranges["range_z"])) > c_cfg["nontrivial_range_tol"]:
            final_class = "bounded_nontrivial"
        else:
            final_class = "ambiguous_long_transient"
    return {
        **ranges,
        **tail,
        "bounded": bounded,
        "final_class": final_class,
        "max_norm": max_norm,
        "final_x": float(final[0]),
        "final_y": float(final[1]),
        "final_z": float(final[2]),
        "section_fingerprint_count": int(section.shape[0]),
        "lyap_max_if_available": "",
        "numerical_status": "ok",
    }


def robustness_start(candidate: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    target = np.asarray(candidate.get("target_seed", [float("nan")] * 3), dtype=float)
    if target.shape == (3,) and np.all(np.isfinite(target)):
        return target, "target_seed"
    return np.asarray(candidate["seed"], dtype=float), "candidate_seed"


def run_reference_attractors(
    candidates: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
    outdir: Path,
    *,
    resume: bool,
    skip: bool,
) -> List[Dict[str, Any]]:
    path = outdir / "reference_attractor_summary.csv"
    ensure_csv(path, REFERENCE_FIELDS)
    if skip or not bool(cfg.get("reference_attractor", {}).get("enabled", True)):
        return read_rows(path)
    if not resume:
        write_csv(path, [], REFERENCE_FIELDS)
    rows = [dict(r) for r in read_rows(path)] if resume else []
    done = {str(r.get("candidate_id", "")) for r in rows}
    params = cfg.get("reference_attractor", {})
    h = float(params.get("h", 0.01))
    Lm = float(params.get("memory_length", 40))
    t_final = float(params.get("t_final", 1500))
    for cand in candidates:
        cid = cand["candidate_id"]
        if cid in done:
            continue
        x0, start_source = robustness_start(cand)
        try:
            traj = integrate_original(x0, p, float(cand["q"]), h, Lm, t_final)
            stats = classify_attractor_traj(traj, p, eqs, float(cand["q"]), h, cfg)
            traj_path = save_reduced_trajectory(outdir / "trajectories" / f"{cid}_reference_attractor.csv", traj)
            reference_status = "reference_bounded_nontrivial" if truthy(stats.get("bounded")) and stats.get("final_class") == "bounded_nontrivial" else "reference_failed"
            notes = "Reference attractor reconstruction run; no hidden_verified."
        except Exception as exc:
            stats = {
                "bounded": False,
                "final_class": "numerical_failure",
                "max_norm": "",
                "final_x": "",
                "final_y": "",
                "final_z": "",
                "range_x": "",
                "range_y": "",
                "range_z": "",
                "mean_x_tail": "",
                "mean_y_tail": "",
                "mean_z_tail": "",
                "var_x_tail": "",
                "var_y_tail": "",
                "var_z_tail": "",
                "fft_peak": "",
                "psd_entropy": "",
                "section_fingerprint_count": "",
                "numerical_status": "exception",
            }
            traj_path = ""
            reference_status = "reference_failed"
            notes = str(exc)
        row = {
            "candidate_id": cid,
            "q": float(cand["q"]),
            "mu": float(cand["mu"]),
            "theta": float(cand["theta"]),
            "branch": cand.get("branch", ""),
            "h": h,
            "memory_length": Lm,
            "memory_points": memory_points(Lm, h),
            "t_final": t_final,
            "start_source": start_source,
            "start_x": float(x0[0]),
            "start_y": float(x0[1]),
            "start_z": float(x0[2]),
            **stats,
            "trajectory_csv": traj_path,
            "reference_status": reference_status,
            "notes": notes,
        }
        rows.append(row)
        append_csv(path, row, REFERENCE_FIELDS)
    return read_rows(path)


def run_robustness(
    candidates: Sequence[Dict[str, Any]],
    blocked_candidate_ids: set[str],
    reference_failed_candidate_ids: set[str],
    cfg: Dict[str, Any],
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
    outdir: Path,
    *,
    resume: bool,
    skip: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_path = outdir / "attractor_robustness_raw.csv"
    summary_path = outdir / "attractor_robustness_summary.csv"
    if not resume:
        write_csv(raw_path, [], ROBUST_RAW_FIELDS)
    else:
        ensure_csv(raw_path, ROBUST_RAW_FIELDS)
    raw_rows = [dict(r) for r in read_rows(raw_path)] if resume else []
    done = {(r.get("candidate_id"), r.get("case_id")) for r in raw_rows}
    if skip or not bool(cfg.get("attractor_robustness", {}).get("enabled", True)):
        if resume and summary_path.exists():
            return raw_rows, read_rows(summary_path)
        summary = []
        for cand in candidates:
            if cand["candidate_id"] in blocked_candidate_ids:
                continue
            if cand["candidate_id"] in reference_failed_candidate_ids:
                summary.append(
                    {
                        "candidate_id": cand["candidate_id"],
                        "executed_cases": 0,
                        "all_bounded_nontrivial": False,
                        "no_equilibrium_convergence": False,
                        "no_divergence": False,
                        "range_relative_spread": "",
                        "fft_relative_spread": "",
                        "var_tail_collapsed": "",
                        "robust_attractor": False,
                        "robustness_status": "skipped_due_to_reference_attractor_failure",
                        "notes": "Reference attractor did not classify as bounded_nontrivial.",
                    }
                )
                continue
            summary.append(
                {
                    "candidate_id": cand["candidate_id"],
                    "executed_cases": 0,
                    "all_bounded_nontrivial": False,
                    "no_equilibrium_convergence": False,
                    "no_divergence": False,
                    "range_relative_spread": "",
                    "fft_relative_spread": "",
                    "var_tail_collapsed": "",
                    "robust_attractor": False,
                    "robustness_status": "skipped_by_cli" if skip else "disabled_in_config",
                    "notes": "Robustness not run; hidden_verified remains false.",
                }
            )
        write_csv(summary_path, summary, ROBUST_SUMMARY_FIELDS)
        return raw_rows, summary
    cases = cfg["attractor_robustness"].get("cases", {})
    for cand in candidates:
        cid = cand["candidate_id"]
        if cid in blocked_candidate_ids or cid in reference_failed_candidate_ids:
            continue
        x0, start_source = robustness_start(cand)
        for case_id, params in cases.items():
            if (cid, case_id) in done:
                continue
            h = float(params["h"])
            Lm = float(params["memory_length"])
            t_final = float(params["t_final"])
            try:
                traj = integrate_original(x0, p, float(cand["q"]), h, Lm, t_final)
                stats = classify_attractor_traj(traj, p, eqs, float(cand["q"]), h, cfg)
                traj_path = save_reduced_trajectory(outdir / "trajectories" / f"{cid}_{case_id}.csv", traj)
            except Exception as exc:
                stats = {
                    "bounded": False,
                    "final_class": "numerical_failure",
                    "max_norm": "",
                    "final_x": "",
                    "final_y": "",
                    "final_z": "",
                    "range_x": "",
                    "range_y": "",
                    "range_z": "",
                    "mean_x_tail": "",
                    "mean_y_tail": "",
                    "mean_z_tail": "",
                    "var_x_tail": "",
                    "var_y_tail": "",
                    "var_z_tail": "",
                    "fft_peak": "",
                    "psd_entropy": "",
                    "section_fingerprint_count": "",
                    "lyap_max_if_available": "",
                    "numerical_status": "exception",
                    "notes": str(exc),
                }
                traj_path = ""
            row = {
                "candidate_id": cid,
                "case_id": case_id,
                "q": float(cand["q"]),
                "h": h,
                "memory_length": Lm,
                "memory_points": memory_points(Lm, h),
                "t_final": t_final,
                "start_source": start_source,
                "start_x": float(x0[0]),
                "start_y": float(x0[1]),
                "start_z": float(x0[2]),
                **stats,
                "trajectory_csv": traj_path,
                "notes": stats.get("notes", "Attractor robustness run; not hidden_verified."),
            }
            raw_rows.append(row)
            append_csv(raw_path, row, ROBUST_RAW_FIELDS)
    summary = summarize_robustness(raw_rows, candidates, blocked_candidate_ids, reference_failed_candidate_ids, set(cases.keys()))
    write_csv(summary_path, summary, ROBUST_SUMMARY_FIELDS)
    return raw_rows, summary


def vector_from_row(row: Dict[str, Any], prefix: str) -> np.ndarray:
    return np.array([as_float(row.get(f"{prefix}_x")), as_float(row.get(f"{prefix}_y")), as_float(row.get(f"{prefix}_z"))], dtype=float)


def summarize_robustness(
    raw_rows: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    blocked_candidate_ids: set[str],
    reference_failed_candidate_ids: set[str],
    expected_cases: set[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[str(row.get("candidate_id", ""))].append(row)
    out: List[Dict[str, Any]] = []
    for cand in candidates:
        cid = cand["candidate_id"]
        if cid in blocked_candidate_ids:
            out.append(
                {
                    "candidate_id": cid,
                    "executed_cases": 0,
                    "all_bounded_nontrivial": False,
                    "no_equilibrium_convergence": False,
                    "no_divergence": False,
                    "range_relative_spread": "",
                    "fft_relative_spread": "",
                    "var_tail_collapsed": "",
                    "robust_attractor": False,
                    "robustness_status": "skipped_due_to_reproduced_target",
                    "notes": "Equilibrium TARGET reproduced; long robustness skipped.",
                }
            )
            continue
        if cid in reference_failed_candidate_ids:
            out.append(
                {
                    "candidate_id": cid,
                    "executed_cases": 0,
                    "all_bounded_nontrivial": False,
                    "no_equilibrium_convergence": False,
                    "no_divergence": False,
                    "range_relative_spread": "",
                    "fft_relative_spread": "",
                    "var_tail_collapsed": "",
                    "robust_attractor": False,
                    "robustness_status": "skipped_due_to_reference_attractor_failure",
                    "notes": "Reference attractor did not classify as bounded_nontrivial.",
                }
            )
            continue
        rows = grouped.get(cid, [])
        executed_cases = {str(r.get("case_id", "")) for r in rows}
        bounded_nontrivial = [str(r.get("final_class", "")) == "bounded_nontrivial" and truthy(r.get("bounded")) for r in rows]
        all_bounded = bool(rows) and all(bounded_nontrivial) and expected_cases.issubset(executed_cases)
        no_eq = all(str(r.get("final_class", "")) != "equilibrium_convergence" for r in rows)
        no_div = all(str(r.get("final_class", "")) != "divergent" for r in rows)
        ranges = np.array([[as_float(r.get("range_x")), as_float(r.get("range_y")), as_float(r.get("range_z"))] for r in rows], dtype=float)
        if ranges.size and np.all(np.isfinite(ranges)):
            denom = np.maximum(np.max(np.abs(ranges), axis=0), 1e-12)
            range_spread = float(np.max((np.max(ranges, axis=0) - np.min(ranges, axis=0)) / denom))
        else:
            range_spread = float("nan")
        fft = np.array([as_float(r.get("fft_peak")) for r in rows], dtype=float)
        fft_valid = fft[np.isfinite(fft) & (np.abs(fft) > 1e-12)]
        if fft_valid.size >= 2:
            fft_spread = float((np.max(fft_valid) - np.min(fft_valid)) / max(np.max(np.abs(fft_valid)), 1e-12))
        else:
            fft_spread = 0.0
        vars_arr = np.array([[as_float(r.get("var_x_tail")), as_float(r.get("var_y_tail")), as_float(r.get("var_z_tail"))] for r in rows], dtype=float)
        var_collapsed = bool(vars_arr.size and np.nanmax(vars_arr) <= 1e-10)
        if not expected_cases.issubset(executed_cases) and rows:
            status = "plausible_but_R2_not_run" if "R2" not in executed_cases else "partial_robustness_run"
        elif all_bounded and no_eq and no_div and range_spread <= 0.25 and fft_spread <= 0.25 and not var_collapsed:
            status = "robust_attractor"
        elif not rows:
            status = "not_run"
        else:
            status = "not_robust_under_tested_cases"
        out.append(
            {
                "candidate_id": cid,
                "executed_cases": len(executed_cases),
                "all_bounded_nontrivial": all_bounded,
                "no_equilibrium_convergence": no_eq,
                "no_divergence": no_div,
                "range_relative_spread": "" if math.isnan(range_spread) else range_spread,
                "fft_relative_spread": fft_spread,
                "var_tail_collapsed": var_collapsed,
                "robust_attractor": bool(status == "robust_attractor"),
                "robustness_status": status,
                "notes": "No hidden_verified; robustness is operational under tested h/memory.",
            }
        )
    return out


def count_planned(
    candidates: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
    skip_reference: bool,
    skip_controls: bool,
    skip_robustness: bool,
) -> Dict[str, int]:
    reference = 0 if skip_reference else len(candidates)
    controls = 0 if skip_controls else len(build_targeted_plan(candidates, cfg, p, eqs))
    robustness = 0 if skip_robustness else len(candidates) * len(cfg.get("attractor_robustness", {}).get("cases", {}))
    return {
        "reference_attractor": reference,
        "targeted_equilibrium_controls": controls,
        "attractor_robustness": robustness,
        "total_without_reproduction": reference + controls + robustness,
    }


def enforce_cost_guard(planned: int, cfg: Dict[str, Any], max_trajectories: int | None, force: bool) -> int:
    limit = int(max_trajectories or cfg.get("cost_guard", {}).get("max_trajectories_without_force", 1000))
    if planned > limit and not force:
        raise RuntimeError(f"Cost guard: planned trajectories={planned} exceeds max_trajectories={limit}. Use --force or skip a stage.")
    return limit


def target_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if truthy(r.get("target_hit")) or str(r.get("final_class", "")) == "target_attractor"]


def rows_by_candidate(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("candidate_id", ""))].append(row)
    return out


def build_decisions(
    candidates: Sequence[Dict[str, Any]],
    previous: Dict[str, Dict[str, Any]],
    reference_rows: Sequence[Dict[str, Any]],
    targeted_rows: Sequence[Dict[str, Any]],
    repro_summary: Sequence[Dict[str, Any]],
    robustness_summary: Sequence[Dict[str, Any]],
    skipped_by_cost: set[str],
) -> List[Dict[str, Any]]:
    reference_by_cand = {str(row.get("candidate_id", "")): row for row in reference_rows}
    targeted_by_cand = rows_by_candidate(targeted_rows)
    repro_by_cand = rows_by_candidate(repro_summary)
    robust_by_cand = {row.get("candidate_id", ""): row for row in robustness_summary}
    decisions: List[Dict[str, Any]] = []
    for cand in candidates:
        cid = cand["candidate_id"]
        rows = targeted_by_cand.get(cid, [])
        targeted_hits = target_rows(rows)
        reproduced_hits = sum(int(row.get("n_reproduced_target", 0)) for row in repro_by_cand.get(cid, []))
        robust_target_hit = any(truthy(row.get("robust_target_hit")) for row in repro_by_cand.get(cid, []))
        eq_counts = Counter(str(r.get("equilibrium_id", "")) for r in targeted_hits)
        robust_row = robust_by_cand.get(cid, {})
        robust_attractor = truthy(robust_row.get("robust_attractor"))
        robustness_status = str(robust_row.get("robustness_status", "not_run"))
        reference_row = reference_by_cand.get(cid, {})
        reference_status = str(reference_row.get("reference_status", ""))
        if reference_status and reference_status != "reference_bounded_nontrivial":
            final_status = "not_ready_due_to_reference_attractor_failure"
            next_action = "inspect_reference_seed_or_reintegrate_before_equilibrium_controls"
        elif len(rows) == 0:
            final_status = "not_evaluated"
            next_action = "run_targeted_controls_only_before_any_random_sampling"
        elif robust_target_hit:
            final_status = "not_supported_by_equilibrium_neighborhood_test"
            next_action = "discard_as_hidden_or_document_as_self_excited_candidate"
        elif len(targeted_hits) > 0:
            final_status = "inconclusive_isolated_hit"
            next_action = "inspect_classifier_and_reproduce_isolated_hits"
        elif len(targeted_hits) == 0 and robust_attractor:
            final_status = "strongest_current_candidate_but_not_verified"
            next_action = "prepare_figures_and_optional_local_basin_zoom"
        elif len(targeted_hits) == 0 and not robust_attractor:
            final_status = "not_ready_due_to_numerical_nonrobustness"
            next_action = "improve_integrator_or_memory"
        else:
            final_status = "not_evaluated"
            next_action = "run_targeted_controls_only_before_any_random_sampling"
        decisions.append(
            {
                "candidate_id": cid,
                "q": float(cand["q"]),
                "mu": float(cand["mu"]),
                "theta": float(cand["theta"]),
                "previous_completed_trajectories": previous.get(cid, {}).get("previous_completed_trajectories", 0),
                "previous_target_hits": previous.get(cid, {}).get("previous_target_hits", 0),
                "targeted_equilibrium_tests": len(rows),
                "targeted_target_hits": len(targeted_hits),
                "reproduced_target_hits": reproduced_hits,
                "robust_target_hit": robust_target_hit,
                "Eminus_target_hits": eq_counts.get("E-", 0),
                "E0_target_hits": eq_counts.get("E0", 0),
                "Eplus_target_hits": eq_counts.get("E+", 0),
                "robustness_status": robustness_status,
                "robust_attractor": robust_attractor,
                "final_recommended_status": final_status,
                "hidden_verified": False,
                "next_action": next_action,
            }
        )
    return decisions


def read_reduced_traj(path: str | Path, max_points: int = 12000) -> np.ndarray:
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


def candidate_plot_points(candidate: Dict[str, Any], reference_rows: Sequence[Dict[str, Any]], robust_rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    for row in reference_rows:
        if row.get("candidate_id") == candidate["candidate_id"] and row.get("trajectory_csv"):
            arr = read_reduced_traj(row["trajectory_csv"])
            if arr.shape[0] > 0:
                return arr[:, 1:4]
    for row in robust_rows:
        if row.get("candidate_id") == candidate["candidate_id"] and row.get("trajectory_csv"):
            arr = read_reduced_traj(row["trajectory_csv"])
            if arr.shape[0] > 0:
                return arr[:, 1:4]
    target = np.asarray(candidate.get("target_seed", candidate.get("seed")), dtype=float).reshape(1, 3)
    seed = np.asarray(candidate.get("seed"), dtype=float).reshape(1, 3)
    return np.vstack([seed, target])


def plot_outputs(
    candidates: Sequence[Dict[str, Any]],
    eqs: Dict[str, np.ndarray],
    targeted_summary: Sequence[Dict[str, Any]],
    targeted_rows: Sequence[Dict[str, Any]],
    reference_rows: Sequence[Dict[str, Any]],
    robust_rows: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    p: Dict[str, Any],
    outdir: Path,
) -> List[str]:
    files: List[str] = []
    plots = outdir / "plots"
    summary_by_cand = rows_by_candidate(targeted_summary)
    raw_by_cand = rows_by_candidate(targeted_rows)
    colors = {
        "equilibrium_convergence": "#111827",
        "target_attractor": "#dc2626",
        "other_bounded_nontrivial": "#2563eb",
        "divergent": "#f59e0b",
        "numerical_failure": "#6b7280",
        "ambiguous_long_transient": "#7c3aed",
    }
    for cand in candidates:
        cid = cand["candidate_id"]
        pts = candidate_plot_points(cand, reference_rows, robust_rows)
        fig = plt.figure(figsize=(7.2, 5.4))
        ax = fig.add_subplot(111, projection="3d")
        if pts.shape[0] > 1:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=0.8, color="#2563eb", label="candidate attractor/seed")
        else:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=20, color="#2563eb", label="candidate seed")
        for eq_id, eq in eqs.items():
            ax.scatter(eq[0], eq[1], eq[2], s=45, label=eq_id)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"{cid} targeted view")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = plots / f"{cid}_attractor_equilibria_3d.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(rel(path))

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        rows = summary_by_cand.get(cid, [])
        for eq_id in ["E-", "E0", "E+"]:
            sub = sorted([r for r in rows if r.get("equilibrium_id") == eq_id], key=lambda r: as_float(r.get("radius")))
            if not sub:
                continue
            ax.plot([as_float(r.get("radius")) for r in sub], [as_float(r.get("n_target"), 0.0) for r in sub], "o-", label=eq_id)
        ax.set_xscale("log")
        ax.set_xlabel("rho")
        ax.set_ylabel("TARGET count")
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        fig.tight_layout()
        path = plots / f"{cid}_target_hits_vs_radius.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(rel(path))

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        group = raw_by_cand.get(cid, [])
        eq_ids = ["E-", "E0", "E+"]
        bottoms = np.zeros(len(eq_ids), dtype=float)
        for cls, color in colors.items():
            vals = [sum(1 for r in group if r.get("equilibrium_id") == eq and r.get("final_class") == cls) for eq in eq_ids]
            ax.bar(eq_ids, vals, bottom=bottoms, label=cls, color=color)
            bottoms += np.asarray(vals, dtype=float)
        ax.set_ylabel("trajectory count")
        ax.legend(fontsize=6, ncols=2)
        fig.tight_layout()
        path = plots / f"{cid}_classes_by_equilibrium.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(rel(path))

        for ext in ["png", "pdf"]:
            fig = plt.figure(figsize=(9.0, 4.5))
            ax1 = fig.add_subplot(121, projection="3d")
            if pts.shape[0] > 1:
                ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=0.75, color="#2563eb")
            else:
                ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=18, color="#2563eb")
            for eq_id, eq in eqs.items():
                ax1.scatter(eq[0], eq[1], eq[2], s=38)
                ax1.text(eq[0], eq[1], eq[2], eq_id, fontsize=8)
            ax1.set_title("(a) candidate and equilibria")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_zlabel("z")
            target_group = target_rows(group)
            critical_eq = str(target_group[0].get("equilibrium_id")) if target_group else "E+"
            eq = eqs.get(critical_eq, eqs.get("E+", next(iter(eqs.values()))))
            ax2 = fig.add_subplot(122, projection="3d")
            controls = cfg.get("targeted_equilibrium_controls", {})
            for radius in [float(v) for v in controls.get("radii", [1e-5])]:
                sphere_pts = [eq + radius * d for _label, d in direction_set(critical_eq, p, eq)]
                S = np.asarray(sphere_pts)
                ax2.scatter(S[:, 0], S[:, 1], S[:, 2], s=8, alpha=0.55, label=f"rho={radius:.0e}")
            ax2.scatter(eq[0], eq[1], eq[2], s=42, color="#111827", label=critical_eq)
            ax2.set_title(f"(b) targeted neighborhood {critical_eq}")
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.set_zlabel("z")
            ax2.legend(fontsize=6)
            fig.tight_layout()
            path = plots / f"{cid}_hiddenness_targeted_article_style.{ext}"
            fig.savefig(path, dpi=220 if ext == "png" else None)
            plt.close(fig)
            files.append(rel(path))
    return files


def write_report(
    outdir: Path,
    cfg: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    previous: Dict[str, Dict[str, Any]],
    planned: Dict[str, int],
    eq_checks: Dict[str, Any],
    reference_rows: Sequence[Dict[str, Any]],
    decisions: Sequence[Dict[str, Any]],
    files_written: Sequence[str],
) -> None:
    reference_by_cand = {str(row.get("candidate_id", "")): row for row in reference_rows}
    lines = [
        "# Machado targeted verification",
        "",
        "This route replaces the massive Corrida 1 random grid with directed equilibrium-neighborhood tests.",
        "",
        "No `hidden_verified` status is declared.",
        "",
        "## Scope",
        "",
        f"- q: `{float(cfg['q']):.5f}`",
        f"- source_summary: `{cfg['source_summary']}`",
        f"- previous_corrida1_dir: `{cfg['previous_corrida1_dir']}`",
        f"- output_dir: `{cfg['output_dir']}`",
        f"- selected_planned_reference_attractor: `{planned['selected']['reference_attractor']}`",
        f"- selected_planned_targeted_equilibrium_controls: `{planned['selected']['targeted_equilibrium_controls']}`",
        f"- selected_planned_attractor_robustness: `{planned['selected']['attractor_robustness']}`",
        f"- full_route_reference_attractor: `{planned['full_route']['reference_attractor']}`",
        f"- full_route_targeted_equilibrium_controls: `{planned['full_route']['targeted_equilibrium_controls']}`",
        f"- full_route_attractor_robustness: `{planned['full_route']['attractor_robustness']}`",
        "",
        "## Previous Corrida 1 state",
        "",
    ]
    for cand in candidates:
        prev = previous.get(cand["candidate_id"], {})
        lines.append(
            f"- `{cand['candidate_id']}`: completed={prev.get('previous_completed_trajectories', 0)}, "
            f"target_hits={prev.get('previous_target_hits', 0)}, status=`{prev.get('previous_hiddenness_status', '')}`"
        )
    lines.extend(["", "## Reference Attractor Gate", ""])
    for cand in candidates:
        row = reference_by_cand.get(cand["candidate_id"], {})
        lines.append(
            f"- `{cand['candidate_id']}`: reference_status=`{row.get('reference_status', 'not_run')}`, "
            f"final_class=`{row.get('final_class', 'not_run')}`"
        )
    lines.extend(["", "## Equilibrium check", ""])
    for eq_id, check in eq_checks.items():
        lines.append(f"- `{eq_id}` distance_to_expected={check['distance_to_expected']:.6g}, matches_1e-6={check['matches_expected_1e-6']}")
    lines.extend(["", "## Decisions", ""])
    lines.append("| candidate_id | targeted_tests | targeted_hits | reproduced_hits | robust_attractor | final_recommended_status | next_action |")
    lines.append("|---|---:|---:|---:|---|---|---|")
    for row in decisions:
        lines.append(
            f"| `{row['candidate_id']}` | {row['targeted_equilibrium_tests']} | {row['targeted_target_hits']} | "
            f"{row['reproduced_target_hits']} | {row['robust_attractor']} | {row['final_recommended_status']} | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Scientific notes",
            "",
            "- Machado/FDF describing functions are treated as heuristic seed generators.",
            "- Caputo harmonic solutions are interpreted as approximate Weyl-Caputo seeds, not exact periodic orbits.",
            "- A robust TARGET contact from E-, E0 or E+ rejects hiddenness support for that candidate.",
            "- Zero TARGET contacts under tested radii is operational evidence only, not a proof.",
        ]
    )
    (outdir / "machado_targeted_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "q": float(cfg["q"]),
        "planned": planned,
        "equilibrium_checks": eq_checks,
        "reference_attractor": list(reference_rows),
        "decisions": list(decisions),
        "hidden_verified": False,
        "files_written": list(dict.fromkeys(files_written + [rel(outdir / "machado_targeted_report.md")])),
    }
    (outdir / "machado_targeted_summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Directed targeted Machado verification replacing Corrida 1 massive random grid.")
    parser.add_argument("--candidate-id", default="all", help="'all' or one configured Machado candidate id.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-reference-attractor", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-equilibrium-controls", action="store_true")
    parser.add_argument("--max-trajectories", type=int, default=None)
    return parser.parse_args()


def write_cost_plan(outdir: Path, selected: Dict[str, int], full_route: Dict[str, int], limit: int) -> str:
    path = outdir / "machado_targeted_cost_plan.json"
    data = {
        "selected_run": selected,
        "full_route_without_reproduction": full_route,
        "max_trajectories": int(limit),
        "notes": [
            "The full directed route is designed to stay in the hundreds of trajectories, not the old 49344 random-grid plan.",
            "Each candidate first reconstructs a reference attractor from target_seed or seed before equilibrium controls.",
            "Equilibrium controls short-circuit a candidate after the first TARGET contact unless stop_after_first_target_hit_in_controls is set to false.",
            "Reproduction trajectories are added only if TARGET contacts appear.",
            "Robustness is skipped for candidates with reproduced equilibrium-neighborhood TARGET contacts.",
        ],
    }
    path.write_text(json.dumps(json_safe(data), indent=2), encoding="utf-8")
    return rel(path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    enforce_memory_contract(cfg)
    cfg["q"] = float(cfg.get("q", 0.9998))
    if abs(float(cfg["q"]) - 0.9998) > 5e-10:
        raise ValueError("This targeted Machado route is fixed to q=0.9998.")
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    outdir = safe_output_dir(cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
    cfg["output_dir"] = rel(outdir)
    candidate_ids = selected_candidate_ids(cfg, args.candidate_id)
    candidates, _source_summary, source_params = load_source_data(cfg, candidate_ids)
    run_cfg = cfg_with_params(cfg, source_params)
    p = chua_ic_params(run_cfg)
    previous = previous_stats(cfg)
    eqs, _eq_rows, eq_checks = write_equilibria(p, float(cfg["q"]), outdir)
    selected_plan = count_planned(candidates, cfg, p, eqs, args.skip_reference_attractor, args.skip_equilibrium_controls, args.skip_robustness)
    full_plan = count_planned(candidates, cfg, p, eqs, False, False, False)
    total_limit = int(args.max_trajectories or cfg.get("cost_guard", {}).get("max_trajectories_without_force", 1000))
    cost_plan_file = write_cost_plan(outdir, selected_plan, full_plan, total_limit)
    enforce_cost_guard(selected_plan["total_without_reproduction"], cfg, args.max_trajectories, args.force)
    print(
        f"planned_reference_attractor={selected_plan['reference_attractor']} "
        f"planned_targeted_equilibrium_controls={selected_plan['targeted_equilibrium_controls']} "
        f"planned_attractor_robustness={selected_plan['attractor_robustness']} "
        f"planned_total_without_reproduction={selected_plan['total_without_reproduction']} "
        f"full_route_total_without_reproduction={full_plan['total_without_reproduction']} "
        f"max_trajectories={total_limit}",
        flush=True,
    )
    planned = {"selected": selected_plan, "full_route": full_plan}
    files_written = [rel(outdir / "equilibria_targeted_summary.csv"), cost_plan_file]

    reference_rows = run_reference_attractors(
        candidates,
        cfg,
        p,
        eqs,
        outdir,
        resume=args.resume,
        skip=args.skip_reference_attractor,
    )
    files_written.append(rel(outdir / "reference_attractor_summary.csv"))
    reference_by_cand = {str(row.get("candidate_id", "")): row for row in reference_rows}
    if args.skip_reference_attractor or not reference_rows:
        control_candidates = list(candidates)
        reference_failed: set[str] = set()
    else:
        ready_ids = {
            cid
            for cid, row in reference_by_cand.items()
            if str(row.get("reference_status", "")) == "reference_bounded_nontrivial"
        }
        reference_failed = {cand["candidate_id"] for cand in candidates if cand["candidate_id"] not in ready_ids}
        control_candidates = [cand for cand in candidates if cand["candidate_id"] in ready_ids]

    if args.skip_equilibrium_controls or not bool(cfg.get("targeted_equilibrium_controls", {}).get("enabled", True)):
        ensure_csv(outdir / "targeted_equilibrium_raw.csv", RAW_FIELDS)
        ensure_csv(outdir / "targeted_equilibrium_summary.csv", SUMMARY_FIELDS)
        targeted_raw = read_rows(outdir / "targeted_equilibrium_raw.csv")
        targeted_summary = read_rows(outdir / "targeted_equilibrium_summary.csv")
    else:
        targeted_raw, targeted_summary, _executed = run_targeted_controls(control_candidates, cfg, p, eqs, outdir, resume=args.resume)
    files_written.extend([rel(outdir / "targeted_equilibrium_raw.csv"), rel(outdir / "targeted_equilibrium_summary.csv")])

    candidates_by_id = {cand["candidate_id"]: cand for cand in candidates}
    control_candidate_ids = {cand["candidate_id"] for cand in control_candidates}
    targets = [row for row in target_rows(targeted_raw) if row.get("candidate_id") in control_candidate_ids]
    repro_raw, repro_summary = run_reproduction(targets, candidates_by_id, cfg, p, eqs, outdir, resume=args.resume)
    files_written.extend([rel(outdir / "target_reproduction_raw.csv"), rel(outdir / "target_reproduction_summary.csv")])
    blocked = {row["candidate_id"] for row in repro_summary if truthy(row.get("robust_target_hit"))}
    robust_raw, robust_summary = run_robustness(control_candidates, blocked, reference_failed, cfg, p, eqs, outdir, resume=args.resume, skip=args.skip_robustness)
    files_written.extend([rel(outdir / "attractor_robustness_raw.csv"), rel(outdir / "attractor_robustness_summary.csv")])
    skipped_by_cost: set[str] = set()
    decisions = build_decisions(candidates, previous, reference_rows, targeted_raw, repro_summary, robust_summary, skipped_by_cost)
    write_csv(outdir / "machado_targeted_decision.csv", decisions, DECISION_FIELDS)
    files_written.append(rel(outdir / "machado_targeted_decision.csv"))
    plot_files = plot_outputs(candidates, eqs, targeted_summary, targeted_raw, reference_rows, robust_raw, cfg, p, outdir)
    files_written.extend(plot_files)
    write_report(outdir, cfg, candidates, previous, planned, eq_checks, reference_rows, decisions, files_written)
    files_written.extend([rel(outdir / "machado_targeted_report.md"), rel(outdir / "machado_targeted_summary.json")])
    for row in decisions:
        print(f"candidate_id={row['candidate_id']}")
        print(f"previous_completed_trajectories={row['previous_completed_trajectories']}")
        print(f"previous_target_hits={row['previous_target_hits']}")
        print(f"targeted_equilibrium_tests={row['targeted_equilibrium_tests']}")
        print(f"targeted_target_hits={row['targeted_target_hits']}")
        print(f"reproduced_target_hits={row['reproduced_target_hits']}")
        print(f"robust_target_hit={row['robust_target_hit']}")
        print(f"robust_attractor={row['robust_attractor']}")
        print(f"final_recommended_status={row['final_recommended_status']}")
    print("files_written=" + ";".join(list(dict.fromkeys(files_written))), flush=True)


if __name__ == "__main__":
    main()
