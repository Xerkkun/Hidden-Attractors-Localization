#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
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
from lure_candidate_manifest import DEFAULT_CONFIG, ROOT, as_float, as_int, csv_value, load_config, read_csv_rows


OUTDIR = ROOT / "outputs" / "lure_route" / "adaptive_Eminus"
PLOTS_DIR = OUTDIR / "plots"
MANIFEST = ROOT / "outputs" / "lure_route" / "lure_candidates_manifest.csv"
RHOH = ROOT / "outputs" / "lure_route" / "lure_rhoH_diagnostics.csv"
PREVIOUS_REPRO_RAW = ROOT / "outputs" / "lure_route" / "refined_Eminus_contact_weak" / "previous_target_reproduction_raw.csv"

DEFAULT_REPRESENTATIVE = "lure_q_0p99000_branch_0_rep01"
SELECTABLE_IDS = ["lure_q_0p99000_branch_0_rep01", "lure_q_0p99000_branch_0_rep02"]

LINE_RHOS = [0.006, 0.008, 0.009, 0.010, 0.011, 0.012, 0.015]
EPS_VALUES = [1.0e-4, 1.0e-3, 1.0e-2]
PHI_VALUES = [k * math.pi / 4.0 for k in range(8)]
STAGE_A = {"h": 0.01, "memory_length": 40.0, "t_final": 1500.0}
STAGE_B = {"h": 0.005, "memory_length": 40.0, "t_final": 3000.0}
STAGE_C = {"h": 0.005, "memory_length": 80.0, "t_final": 6000.0}

RAW_FIELDS = [
    "candidate_id",
    "duplicate_group",
    "representative",
    "q",
    "rho_H",
    "rhoH_class",
    "contact_id",
    "contact_source",
    "test_type",
    "rho",
    "rho_original",
    "eps_cone",
    "phi",
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
    "final_class",
    "target_hit",
    "numerical_status",
    "notes",
]

DIRECTION_FIELDS = [
    "contact_id",
    "candidate_id",
    "x_contact",
    "y_contact",
    "z_contact",
    "radius_contact",
    "v_x",
    "v_y",
    "v_z",
]

EQ_FIELDS = ["eq_id", "x", "y", "z", "region", "eig_1", "eig_2", "eig_3", "matignon_margin", "matignon_stable"]

SUMMARY_FIELDS = [
    "candidate_id",
    "duplicate_group",
    "contacts_used",
    "total_tests",
    "line_tests",
    "cone_tests",
    "target_hits",
    "line_target_hits",
    "cone_target_hits",
    "reproduced_B_hits",
    "reproduced_C_hits",
    "robust_target_hit",
    "open_like_cone_hit",
    "hiddenness_status",
]

DECISION_FIELDS = [
    "candidate_id",
    "duplicate_group",
    "total_tests",
    "line_tests",
    "cone_tests",
    "target_hits",
    "line_target_hits",
    "cone_target_hits",
    "reproduced_B_hits",
    "reproduced_C_hits",
    "robust_target_hit",
    "open_like_cone_hit",
    "hiddenness_status",
    "decision_notes",
]

CLASS_COLORS = {
    "equilibrium_convergence": "#111827",
    "target_attractor": "#dc2626",
    "other_bounded_nontrivial": "#2563eb",
    "divergent": "#f59e0b",
    "numerical_failure": "#6b7280",
    "ambiguous_long_transient": "#7c3aed",
}


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k, "")) for k in fields})


def append_csv(path: Path, row: Dict[str, Any], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: csv_value(row.get(k, "")) for k in fields})


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    return read_csv_rows(path)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "si", "on"}


def candidate_base(candidate_id: str) -> str:
    return re.sub(r"_rep\d+$", "", candidate_id)


def duplicate_group_for(candidate_id: str) -> str:
    return candidate_base(candidate_id)


def load_selected_candidates(args: argparse.Namespace) -> List[Dict[str, Any]]:
    manifest = {row["candidate_id"]: row for row in read_rows(MANIFEST)}
    rho_rows = {row["candidate_id"]: row for row in read_rows(RHOH)}
    requested = args.candidate_id or []
    if args.all_selected:
        requested = SELECTABLE_IDS.copy()
    if not requested:
        requested = [DEFAULT_REPRESENTATIVE]

    selected: List[Dict[str, Any]] = []
    for cid in requested:
        if cid not in manifest:
            raise ValueError(f"No existe {cid} en {MANIFEST}")
        row = dict(manifest[cid])
        rr = rho_rows.get(cid, {})
        if not (
            row.get("df_family") == "lure_classic"
            and row.get("priority_class") == "contact_weak"
            and row.get("target_total") == "3"
            and row.get("blocking_equilibrium") == "E-"
            and rr.get("rhoH_class") in {"good", "marginal"}
        ):
            raise ValueError(f"{cid} no cumple el filtro Lure/contact_weak/E-/rhoH.")
        seed = np.asarray([as_float(row.get("seed_x")), as_float(row.get("seed_y")), as_float(row.get("seed_z"))], dtype=float)
        target = np.asarray([as_float(row.get("final_x")), as_float(row.get("final_y")), as_float(row.get("final_z"))], dtype=float)
        row.update({
            "rho_H": rr.get("rho_H", ""),
            "rhoH_class": rr.get("rhoH_class", ""),
            "duplicate_group": duplicate_group_for(cid),
            "representative": DEFAULT_REPRESENTATIVE if cid in SELECTABLE_IDS else cid,
            "seed_vec": seed,
            "target_seed": target,
            "q_float": as_float(row.get("q")),
        })
        selected.append(row)
    return selected


def maybe_duplicate_note(selected: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if len(selected) < 2:
        return {}
    a, b = selected[0], selected[1]
    keys = ["q", "A", "omega", "seed_x", "seed_y", "seed_z"]
    same = all(abs(as_float(a.get(k)) - as_float(b.get(k))) <= 1e-10 for k in keys)
    same = same and abs(as_float(a.get("rho_H")) - as_float(b.get("rho_H"))) <= 1e-12
    return {
        "duplicate_group": duplicate_group_for(str(a["candidate_id"])),
        "representative": DEFAULT_REPRESENTATIVE,
        "same_q_A_omega_seed_rhoH": bool(same),
    }


def equilibria_and_rows(p: Dict[str, Any], q: float) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    eqs = {k: np.asarray(v, dtype=float) for k, v in solve_equilibria(p).items() if np.all(np.isfinite(v))}
    theta = float(q) * math.pi / 2.0
    rows: List[Dict[str, Any]] = []
    for eq_id in ["E-", "E0", "E+"]:
        if eq_id not in eqs:
            continue
        eq = eqs[eq_id]
        eig = np.linalg.eigvals(local_jacobian(p, eq))
        margins = [abs(np.angle(v)) - theta for v in eig]
        rows.append({
            "eq_id": eq_id,
            "x": float(eq[0]),
            "y": float(eq[1]),
            "z": float(eq[2]),
            "region": region_for_sigma(float(eq[0])),
            "eig_1": complex(eig[0]),
            "eig_2": complex(eig[1]),
            "eig_3": complex(eig[2]),
            "matignon_margin": float(min(margins)),
            "matignon_stable": bool(all(m > 0.0 for m in margins)),
        })
    return eqs, rows


def load_target_contacts_from_source(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    contacts: List[Dict[str, Any]] = []
    source = Path(str(candidate.get("source_hidden_check_csv", "")))
    if source.exists():
        for row in read_rows(source):
            if row.get("equilibrium") == "E-" and str(row.get("class", "")).upper() == "TARGET":
                contacts.append({
                    "candidate_id": candidate["candidate_id"],
                    "contact_source": str(source),
                    "source_sample_id": row.get("sample_id", ""),
                    "source_radius": row.get("radius", ""),
                    "point": np.asarray([as_float(row.get("x0")), as_float(row.get("y0")), as_float(row.get("z0"))], dtype=float),
                })
    return contacts


def fallback_contacts_from_previous(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in read_rows(PREVIOUS_REPRO_RAW):
        if row.get("candidate_id") != candidate["candidate_id"] or row.get("equilibrium_id") != "E-":
            continue
        if truthy(row.get("target_hit")) or row.get("final_class") == "target_attractor" or "TARGET" in str(row.get("target_label", "")):
            source = "previous_target_reproduction_raw.csv"
        else:
            source = "reconstructed_from_previous_reproduction_file"
        out.append({
            "candidate_id": candidate["candidate_id"],
            "contact_source": source,
            "source_sample_id": row.get("source_previous_sample_id", row.get("sample_id", "")),
            "source_radius": row.get("source_previous_radius", row.get("radius", "")),
            "point": np.asarray([as_float(row.get("x0")), as_float(row.get("y0")), as_float(row.get("z0"))], dtype=float),
            "reuse_row": row,
        })
    return out


def load_contacts(candidate: Dict[str, Any], eq_minus: np.ndarray) -> List[Dict[str, Any]]:
    contacts = load_target_contacts_from_source(candidate)
    if not contacts:
        contacts = fallback_contacts_from_previous(candidate)
    seen: set[Tuple[float, float, float]] = set()
    unique: List[Dict[str, Any]] = []
    for idx, row in enumerate(contacts):
        point = np.asarray(row["point"], dtype=float)
        if not np.all(np.isfinite(point)):
            continue
        key = tuple(round(float(x), 14) for x in point)
        if key in seen:
            continue
        seen.add(key)
        diff = point - eq_minus
        radius = float(np.linalg.norm(diff))
        if not math.isfinite(radius) or radius <= 1e-14:
            continue
        direction = diff / radius
        contact = dict(row)
        contact.update({
            "contact_id": f"{candidate['candidate_id']}_contact_{len(unique):02d}",
            "radius_original": radius,
            "direction": direction,
        })
        unique.append(contact)
    return unique


def orthonormal_frame(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.asarray(v, dtype=float)
    v = v / max(float(np.linalg.norm(v)), 1e-300)
    aux = np.array([0.0, 1.0, 0.0], dtype=float) if abs(float(np.dot(v, [1.0, 0.0, 0.0]))) > 0.9 else np.array([1.0, 0.0, 0.0], dtype=float)
    u = aux - float(np.dot(aux, v)) * v
    u = u / max(float(np.linalg.norm(u)), 1e-300)
    w = np.cross(v, u)
    w = w / max(float(np.linalg.norm(w)), 1e-300)
    return v, u, w


def contact_direction_rows(candidates: Sequence[Dict[str, Any]], contacts_by_candidate: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        for contact in contacts_by_candidate[cand["candidate_id"]]:
            v = contact["direction"]
            p = contact["point"]
            rows.append({
                "contact_id": contact["contact_id"],
                "candidate_id": cand["candidate_id"],
                "x_contact": float(p[0]),
                "y_contact": float(p[1]),
                "z_contact": float(p[2]),
                "radius_contact": float(contact["radius_original"]),
                "v_x": float(v[0]),
                "v_y": float(v[1]),
                "v_z": float(v[2]),
            })
    return rows


def memory_points(memory_length: float, h: float) -> int:
    return max(1, int(math.ceil(float(memory_length) / float(h)))) + 1


def section_points(traj: np.ndarray, p: Dict[str, Any], t_burn: float, max_points: int = 100) -> np.ndarray:
    pts: List[Tuple[float, float]] = []
    X = np.asarray(traj, dtype=float)
    for k in range(1, X.shape[0]):
        if X[k, 0] < t_burn:
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
                if len(pts) >= max_points:
                    break
    return np.asarray(pts, dtype=float)


def hit_fraction(section: np.ndarray, ref: np.ndarray, tol: float = 0.12) -> Tuple[int, int, float]:
    if section.size == 0 or ref.size == 0:
        total = int(section.shape[0]) if section.ndim == 2 else 0
        return total, 0, 0.0
    hits = 0
    for pt in section:
        if float(np.min(np.linalg.norm(ref - pt.reshape(1, 2), axis=1))) <= tol:
            hits += 1
    total = int(section.shape[0])
    return total, hits, float(hits / max(total, 1))


def tail_stats(traj: np.ndarray, h: float) -> Dict[str, float]:
    X = np.asarray(traj, dtype=float)
    states = X[:, 1:4]
    start = max(0, int(0.8 * states.shape[0]))
    tail = states[start:, :]
    mean = np.mean(tail, axis=0)
    var = np.var(tail, axis=0)
    ranges = np.ptp(states, axis=0)
    data = tail[:, 0] - float(np.mean(tail[:, 0]))
    if data.size > 8 and np.any(data):
        spec = np.abs(np.fft.rfft(data * np.hanning(data.size))) ** 2
        freq = np.fft.rfftfreq(data.size, d=float(h))
        if spec.size > 1 and np.any(spec[1:] > 0.0):
            idx = 1 + int(np.argmax(spec[1:]))
            prob = spec[1:] / max(float(np.sum(spec[1:])), 1e-300)
            fft_peak = float(freq[idx])
            psd_entropy = -float(np.sum(prob * np.log(prob + 1e-300))) / max(math.log(prob.size), 1e-300)
        else:
            fft_peak = 0.0
            psd_entropy = 0.0
    else:
        fft_peak = float("nan")
        psd_entropy = float("nan")
    return {
        "range_x": float(ranges[0]),
        "range_y": float(ranges[1]),
        "range_z": float(ranges[2]),
        "mean_x_tail": float(mean[0]),
        "mean_y_tail": float(mean[1]),
        "mean_z_tail": float(mean[2]),
        "var_x_tail": float(var[0]),
        "var_y_tail": float(var[1]),
        "var_z_tail": float(var[2]),
        "fft_peak": fft_peak,
        "psd_entropy": psd_entropy,
    }


def reference_for(candidate: Dict[str, Any], p: Dict[str, Any], params: Dict[str, float], cache: Dict[Tuple[str, float, float, float], np.ndarray]) -> np.ndarray:
    key = (candidate["candidate_id"], float(params["h"]), float(params["memory_length"]), float(params["t_final"]))
    if key in cache:
        return cache[key]
    target = np.asarray(candidate["target_seed"], dtype=float)
    traj = pipe.integrate_efork3_c(target, p, qord=float(candidate["q_float"]), h=float(params["h"]), Lm=float(params["memory_length"]), t_total=float(params["t_final"]))
    ref = section_points(traj, p, 0.5 * float(params["t_final"]), max_points=240)
    cache[key] = ref
    return ref


def classify_point(
    *,
    candidate: Dict[str, Any],
    contact: Dict[str, Any],
    test_type: str,
    rho: float,
    rho_original: float,
    direction: np.ndarray,
    x0: np.ndarray,
    params: Dict[str, float],
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
    ref_cache: Dict[Tuple[str, float, float, float], np.ndarray],
    eps_cone: float | str = "",
    phi: float | str = "",
    notes: str = "",
) -> Dict[str, Any]:
    t0 = time.time()
    base = {
        "candidate_id": candidate["candidate_id"],
        "duplicate_group": candidate["duplicate_group"],
        "representative": candidate["representative"],
        "q": candidate["q"],
        "rho_H": candidate["rho_H"],
        "rhoH_class": candidate["rhoH_class"],
        "contact_id": contact["contact_id"],
        "contact_source": contact["contact_source"],
        "test_type": test_type,
        "rho": float(rho),
        "rho_original": float(rho_original),
        "eps_cone": eps_cone,
        "phi": phi,
        "direction_x": float(direction[0]),
        "direction_y": float(direction[1]),
        "direction_z": float(direction[2]),
        "x0": float(x0[0]),
        "y0": float(x0[1]),
        "z0": float(x0[2]),
        "h": float(params["h"]),
        "memory_length": float(params["memory_length"]),
        "memory_points": memory_points(float(params["memory_length"]), float(params["h"])),
        "t_final": float(params["t_final"]),
    }
    try:
        ref = reference_for(candidate, p, params, ref_cache)
        traj = pipe.integrate_efork3_c(x0, p, qord=float(candidate["q_float"]), h=float(params["h"]), Lm=float(params["memory_length"]), t_total=float(params["t_final"]))
        states = traj[:, 1:4]
        final = states[-1]
        final_norm = float(np.linalg.norm(final))
        max_norm = float(np.max(np.linalg.norm(states, axis=1)))
        final_dist = {k: float(np.linalg.norm(final - v)) for k, v in eqs.items()}
        min_dist = float(min(np.min(np.linalg.norm(states - v.reshape(1, 3), axis=1)) for v in eqs.values()))
        stats = tail_stats(traj, float(params["h"]))
        if final_norm > 1.0e5 or max_norm > 1.0e5:
            final_class = "divergent"
            target_hit = False
            sec_total = sec_hits = 0
            frac = 0.0
        else:
            nearest = min(final_dist.items(), key=lambda kv: kv[1])
            tail_mean = np.asarray([stats["mean_x_tail"], stats["mean_y_tail"], stats["mean_z_tail"]], dtype=float)
            if nearest[1] <= 1.0e-3 and float(np.linalg.norm(tail_mean - eqs[nearest[0]])) <= 2.0e-3:
                final_class = "equilibrium_convergence"
                target_hit = False
                sec_total = sec_hits = 0
                frac = 0.0
            else:
                sec = section_points(traj, p, 0.5 * float(params["t_final"]), max_points=100)
                sec_total, sec_hits, frac = hit_fraction(sec, ref, tol=0.12)
                if sec_total >= 20 and frac >= 0.70:
                    final_class = "target_attractor"
                    target_hit = True
                elif sec_total < 20:
                    final_class = "ambiguous_long_transient"
                    target_hit = False
                else:
                    final_class = "other_bounded_nontrivial"
                    target_hit = False
        out = {
            **base,
            "final_x": float(final[0]),
            "final_y": float(final[1]),
            "final_z": float(final[2]),
            "final_norm": final_norm,
            "min_dist_to_equilibria": min_dist,
            "final_dist_to_Eminus": final_dist.get("E-", float("nan")),
            "final_dist_to_E0": final_dist.get("E0", float("nan")),
            "final_dist_to_Eplus": final_dist.get("E+", float("nan")),
            **stats,
            "final_class": final_class,
            "target_hit": bool(target_hit),
            "numerical_status": "ok",
            "notes": f"{notes}; sec_total={sec_total}; sec_hits={sec_hits}; hit_frac={frac:.6g}; elapsed_sec={time.time() - t0:.3f}".strip("; "),
        }
    except Exception as exc:
        out = {
            **base,
            "final_x": "",
            "final_y": "",
            "final_z": "",
            "final_norm": "",
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
            "fft_peak": "",
            "psd_entropy": "",
            "final_class": "numerical_failure",
            "target_hit": False,
            "numerical_status": "exception",
            "notes": f"{notes}; {exc}".strip("; "),
        }
    return out


def reused_previous_row(candidate: Dict[str, Any], contact: Dict[str, Any], rho: float, x0: np.ndarray) -> Dict[str, str] | None:
    for row in read_rows(PREVIOUS_REPRO_RAW):
        if row.get("candidate_id") != candidate["candidate_id"] or row.get("equilibrium_id") != "E-":
            continue
        prev = np.asarray([as_float(row.get("x0")), as_float(row.get("y0")), as_float(row.get("z0"))], dtype=float)
        if np.linalg.norm(prev - x0) <= 1e-12 and abs(as_float(row.get("h")) - STAGE_A["h"]) <= 1e-15 and abs(as_float(row.get("memory_length")) - STAGE_A["memory_length"]) <= 1e-12:
            return row
    return None


def row_from_reused(candidate: Dict[str, Any], contact: Dict[str, Any], rho: float, direction: np.ndarray, previous: Dict[str, str]) -> Dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "duplicate_group": candidate["duplicate_group"],
        "representative": candidate["representative"],
        "q": candidate["q"],
        "rho_H": candidate["rho_H"],
        "rhoH_class": candidate["rhoH_class"],
        "contact_id": contact["contact_id"],
        "contact_source": contact["contact_source"],
        "test_type": "line",
        "rho": float(rho),
        "rho_original": float(contact["radius_original"]),
        "eps_cone": "",
        "phi": "",
        "direction_x": float(direction[0]),
        "direction_y": float(direction[1]),
        "direction_z": float(direction[2]),
        "x0": previous.get("x0", ""),
        "y0": previous.get("y0", ""),
        "z0": previous.get("z0", ""),
        "h": previous.get("h", STAGE_A["h"]),
        "memory_length": previous.get("memory_length", STAGE_A["memory_length"]),
        "memory_points": previous.get("memory_points", memory_points(STAGE_A["memory_length"], STAGE_A["h"])),
        "t_final": previous.get("t_final", STAGE_A["t_final"]),
        "final_x": previous.get("final_x", ""),
        "final_y": previous.get("final_y", ""),
        "final_z": previous.get("final_z", ""),
        "final_norm": previous.get("final_norm", ""),
        "min_dist_to_equilibria": previous.get("min_dist_to_equilibria", ""),
        "final_dist_to_Eminus": previous.get("final_dist_to_Eminus", ""),
        "final_dist_to_E0": previous.get("final_dist_to_E0", ""),
        "final_dist_to_Eplus": previous.get("final_dist_to_Eplus", ""),
        "range_x": previous.get("range_x", ""),
        "range_y": previous.get("range_y", ""),
        "range_z": previous.get("range_z", ""),
        "mean_x_tail": previous.get("mean_x_tail", ""),
        "mean_y_tail": previous.get("mean_y_tail", ""),
        "mean_z_tail": previous.get("mean_z_tail", ""),
        "var_x_tail": previous.get("var_x_tail", ""),
        "var_y_tail": previous.get("var_y_tail", ""),
        "var_z_tail": previous.get("var_z_tail", ""),
        "fft_peak": previous.get("fft_peak", ""),
        "psd_entropy": previous.get("psd_entropy", ""),
        "final_class": previous.get("final_class", ""),
        "target_hit": truthy(previous.get("target_hit")),
        "numerical_status": previous.get("numerical_status", "ok"),
        "notes": "reused previous refined reproduction at rho_original",
    }


def line_rhos(rho_original: float) -> List[float]:
    vals = list(LINE_RHOS)
    if all(abs(rho_original - r) > 1e-12 for r in vals):
        vals.append(float(rho_original))
    return sorted(vals)


def cone_rhos(rho_original: float) -> List[float]:
    vals = [float(rho_original), 0.010, 0.012]
    out: List[float] = []
    for val in vals:
        if all(abs(val - old) > 1e-12 for old in out):
            out.append(val)
    return out


def processed_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str, str, str]:
    return (
        str(row.get("candidate_id", "")),
        str(row.get("contact_id", "")),
        str(row.get("test_type", "")),
        f"{as_float(row.get('rho')):.17g}",
        f"{as_float(row.get('eps_cone')):.17g}" if str(row.get("eps_cone", "")) else "",
        f"{as_float(row.get('phi')):.17g}" if str(row.get("phi", "")) else "",
        f"{as_float(row.get('h')):.17g}",
    )


def summarize_decisions(candidates: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]], contacts_by_candidate: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_rows: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    by_cid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cid[str(row.get("candidate_id", ""))].append(row)

    for cand in candidates:
        cid = cand["candidate_id"]
        sub = by_cid.get(cid, [])
        target = [r for r in sub if truthy(r.get("target_hit"))]
        line = [r for r in sub if r.get("test_type") == "line"]
        cone = [r for r in sub if r.get("test_type") == "cone"]
        b_hits = [r for r in sub if r.get("test_type") == "refine_B" and truthy(r.get("target_hit"))]
        c_hits = [r for r in sub if r.get("test_type") == "refine_C" and truthy(r.get("target_hit"))]
        cone_targets = [r for r in cone if truthy(r.get("target_hit"))]

        open_like = False
        grouped = defaultdict(list)
        for r in cone_targets:
            grouped[(r.get("contact_id"), r.get("eps_cone"))].append(r)
        if any(len(g) >= 3 for g in grouped.values()):
            open_like = True
        by_phi_group = defaultdict(set)
        by_rho_group = defaultdict(set)
        for r in cone_targets:
            by_phi_group[(r.get("contact_id"), r.get("rho"), r.get("eps_cone"))].add(r.get("phi"))
            by_rho_group[(r.get("contact_id"), r.get("eps_cone"))].add(r.get("rho"))
        if any(len(v) > 1 for v in by_phi_group.values()) or any(len(v) > 1 for v in by_rho_group.values()):
            open_like = True

        robust = bool(open_like or b_hits or c_hits)
        if not robust:
            for r in line:
                if truthy(r.get("target_hit")) and (b_hits or c_hits):
                    robust = True

        if robust:
            status = "not_supported_by_refined_neighborhood_test"
            notes = "Robust or open-like targeted contact was detected from E-."
        elif target:
            status = "inconclusive_isolated_hit"
            notes = "TARGET appeared only as an unreproduced isolated targeted contact."
        else:
            status = "compatible_with_hiddenness_under_targeted_local_test"
            notes = "No TARGET contact appeared in the completed targeted local test. This is not hidden_verified."

        common = {
            "candidate_id": cid,
            "duplicate_group": cand["duplicate_group"],
            "total_tests": len(sub),
            "line_tests": len(line),
            "cone_tests": len(cone),
            "target_hits": len(target),
            "line_target_hits": sum(1 for r in line if truthy(r.get("target_hit"))),
            "cone_target_hits": len(cone_targets),
            "reproduced_B_hits": len(b_hits),
            "reproduced_C_hits": len(c_hits),
            "robust_target_hit": bool(robust),
            "open_like_cone_hit": bool(open_like),
            "hiddenness_status": status,
        }
        summary_rows.append({**common, "contacts_used": len(contacts_by_candidate.get(cid, []))})
        decisions.append({**common, "decision_notes": notes})
    return summary_rows, decisions


def plot_outputs(candidates: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]], contacts_by_candidate: Dict[str, List[Dict[str, Any]]], eq_minus: np.ndarray) -> List[str]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    files: List[str] = []
    for cand in candidates:
        cid = cand["candidate_id"]
        sub = [r for r in rows if r.get("candidate_id") == cid]
        safe = cid.replace("/", "_").replace("\\", "_")

        line = [r for r in sub if r.get("test_type") == "line"]
        if line:
            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            for contact_id in sorted({r["contact_id"] for r in line}):
                rr = sorted([r for r in line if r["contact_id"] == contact_id], key=lambda r: as_float(r["rho"]))
                ax.scatter([as_float(r["rho"]) for r in rr], [list(CLASS_COLORS).index(r["final_class"]) if r["final_class"] in CLASS_COLORS else -1 for r in rr], s=24, label=contact_id.split("_contact_")[-1])
            ax.set_xlabel("rho")
            ax.set_ylabel("class index")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, title="contact")
            path = PLOTS_DIR / f"radial_line_classes_{safe}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
            files.append(str(path))

        cone = [r for r in sub if r.get("test_type") == "cone"]
        if cone:
            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            sc = ax.scatter([as_float(r["phi"]) for r in cone], [as_float(r["rho"]) for r in cone], c=[1 if truthy(r["target_hit"]) else 0 for r in cone], cmap="coolwarm", s=18)
            ax.set_xlabel("phi")
            ax.set_ylabel("rho")
            ax.grid(True, alpha=0.25)
            fig.colorbar(sc, ax=ax, label="target_hit")
            path = PLOTS_DIR / f"cone_classes_{safe}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
            files.append(str(path))

        contacts = contacts_by_candidate.get(cid, [])
        if contacts:
            fig = plt.figure(figsize=(5.8, 4.8))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter([eq_minus[0]], [eq_minus[1]], [eq_minus[2]], s=50, c="#111827", label="E-")
            for contact in contacts:
                p = contact["point"]
                v = contact["direction"]
                ax.scatter([p[0]], [p[1]], [p[2]], s=24)
                ax.quiver(eq_minus[0], eq_minus[1], eq_minus[2], v[0], v[1], v[2], length=contact["radius_original"], normalize=True)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.legend(fontsize=7)
            path = PLOTS_DIR / f"contact_direction_geometry_{safe}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
            files.append(str(path))
    return files


def build_report(summary_rows: Sequence[Dict[str, Any]], decisions: Sequence[Dict[str, Any]], cost: Dict[str, Any], files: Sequence[str]) -> None:
    lines = [
        "# Adaptive Eminus Contact Test",
        "",
        "This targeted local test probes radial and conic neighborhoods around previous E- contact directions.",
        "It does not declare `hidden_verified`; absence of hits is only finite-test compatibility.",
        "",
        "## Decisions",
        "",
    ]
    for row in decisions:
        lines.extend([
            f"### {row['candidate_id']}",
            "",
            f"- total_tests: `{row['total_tests']}`",
            f"- target_hits: `{row['target_hits']}`",
            f"- robust_target_hit: `{row['robust_target_hit']}`",
            f"- open_like_cone_hit: `{row['open_like_cone_hit']}`",
            f"- hiddenness_status: `{row['hiddenness_status']}`",
            f"- notes: {row['decision_notes']}",
            "",
        ])
    lines.extend([
        "## Cost Guard",
        "",
        f"- planned_initial_trajectories: `{cost.get('planned_initial_trajectories')}`",
        f"- max_trajectories: `{cost.get('max_trajectories')}`",
        f"- measured_first_trajectory_sec: `{cost.get('measured_first_trajectory_sec')}`",
        f"- estimated_total_sec: `{cost.get('estimated_total_sec')}`",
        f"- estimated_full_stage_A_days_avoided: `{cost.get('estimated_full_stage_A_days_avoided')}`",
        "",
        "## Files",
        "",
    ])
    lines.extend([f"- `{path}`" for path in files])
    (OUTDIR / "adaptive_contact_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive targeted local test for Lure E- contacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--all-selected", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-trajectories", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _cfg = load_config(args.config)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    candidates = load_selected_candidates(args)
    duplicate_note = maybe_duplicate_note(candidates)
    p = pipe.chua_ic_params_from_config(pipe.CONFIG)
    chua.PARAMS = p
    chua.QORD = np.float64(candidates[0]["q_float"])
    eqs, eq_rows = equilibria_and_rows(p, candidates[0]["q_float"])
    if "E-" not in eqs:
        raise RuntimeError("No se encontro E- en los equilibrios.")
    eq_minus = eqs["E-"]
    write_csv(OUTDIR / "equilibria_used.csv", eq_rows, EQ_FIELDS)

    contacts_by_candidate = {cand["candidate_id"]: load_contacts(cand, eq_minus) for cand in candidates}
    direction_rows = contact_direction_rows(candidates, contacts_by_candidate)
    write_csv(OUTDIR / "contact_directions.csv", direction_rows, DIRECTION_FIELDS)

    planned_line = sum(len(line_rhos(contact["radius_original"])) for contacts in contacts_by_candidate.values() for contact in contacts)
    planned_cone = sum(len(cone_rhos(contact["radius_original"])) * len(EPS_VALUES) * len(PHI_VALUES) for contacts in contacts_by_candidate.values() for contact in contacts)
    planned_initial = planned_line + planned_cone
    if planned_initial > int(args.max_trajectories) and not args.force:
        raise RuntimeError(f"Cost guard: {planned_initial} trayectorias planeadas exceden --max-trajectories={args.max_trajectories}. Usa --force o sube el limite.")

    raw_path = OUTDIR / "adaptive_contact_raw.csv"
    raw_rows: List[Dict[str, Any]] = read_rows(raw_path) if args.resume else []
    if raw_rows and not args.resume:
        raw_path.unlink()
        raw_rows = []
    done = {processed_key(row) for row in raw_rows}
    ref_cache: Dict[Tuple[str, float, float, float], np.ndarray] = {}
    start = time.time()
    measured_first: float | None = None

    def record(row: Dict[str, Any]) -> None:
        raw_rows.append(row)
        append_csv(raw_path, row, RAW_FIELDS)

    # Radial line tests.
    for cand in candidates:
        for contact in contacts_by_candidate[cand["candidate_id"]]:
            v = contact["direction"]
            for rho in line_rhos(contact["radius_original"]):
                x0 = eq_minus + float(rho) * v
                proto = {
                    "candidate_id": cand["candidate_id"],
                    "contact_id": contact["contact_id"],
                    "test_type": "line",
                    "rho": rho,
                    "eps_cone": "",
                    "phi": "",
                    "h": STAGE_A["h"],
                }
                if processed_key(proto) in done:
                    continue
                previous = reused_previous_row(cand, contact, rho, x0)
                if previous is not None:
                    row = row_from_reused(cand, contact, rho, v, previous)
                else:
                    t0 = time.time()
                    row = classify_point(candidate=cand, contact=contact, test_type="line", rho=rho, rho_original=contact["radius_original"], direction=v, x0=x0, params=STAGE_A, p=p, eqs=eqs, ref_cache=ref_cache, notes="radial line adaptive test")
                    if measured_first is None:
                        measured_first = time.time() - t0
                record(row)
                print(f"{cand['candidate_id']} line {contact['contact_id']} rho={rho:.8g} class={row['final_class']} target={row['target_hit']}", flush=True)

    # Cone tests only for contacts without line TARGET.
    raw_rows = read_rows(raw_path)
    line_targets_by_contact = {
        (row["candidate_id"], row["contact_id"])
        for row in raw_rows
        if row.get("test_type") == "line" and truthy(row.get("target_hit"))
    }
    for cand in candidates:
        for contact in contacts_by_candidate[cand["candidate_id"]]:
            if (cand["candidate_id"], contact["contact_id"]) in line_targets_by_contact:
                continue
            v, u, w = orthonormal_frame(contact["direction"])
            for rho in cone_rhos(contact["radius_original"]):
                for eps in EPS_VALUES:
                    for phi in PHI_VALUES:
                        direction = v + float(eps) * (math.cos(phi) * u + math.sin(phi) * w)
                        direction = direction / max(float(np.linalg.norm(direction)), 1e-300)
                        x0 = eq_minus + float(rho) * direction
                        proto = {
                            "candidate_id": cand["candidate_id"],
                            "contact_id": contact["contact_id"],
                            "test_type": "cone",
                            "rho": rho,
                            "eps_cone": eps,
                            "phi": phi,
                            "h": STAGE_A["h"],
                        }
                        if processed_key(proto) in done:
                            continue
                        t0 = time.time()
                        row = classify_point(candidate=cand, contact=contact, test_type="cone", rho=rho, rho_original=contact["radius_original"], direction=direction, x0=x0, params=STAGE_A, p=p, eqs=eqs, ref_cache=ref_cache, eps_cone=eps, phi=phi, notes="conic adaptive test")
                        if measured_first is None:
                            measured_first = time.time() - t0
                        record(row)
                        print(f"{cand['candidate_id']} cone {contact['contact_id']} rho={rho:.8g} eps={eps:.1e} phi={phi:.3g} class={row['final_class']} target={row['target_hit']}", flush=True)

    # Reproduce any target hits with B and C only.
    raw_rows = read_rows(raw_path)
    target_rows = [row for row in raw_rows if row.get("test_type") in {"line", "cone"} and truthy(row.get("target_hit"))]
    cand_by_id = {cand["candidate_id"]: cand for cand in candidates}
    contacts_by_id = {contact["contact_id"]: contact for contacts in contacts_by_candidate.values() for contact in contacts}
    for target in target_rows:
        cand = cand_by_id[target["candidate_id"]]
        contact = contacts_by_id[target["contact_id"]]
        direction = np.asarray([as_float(target["direction_x"]), as_float(target["direction_y"]), as_float(target["direction_z"])], dtype=float)
        x0 = np.asarray([as_float(target["x0"]), as_float(target["y0"]), as_float(target["z0"])], dtype=float)
        for label, params in [("refine_B", STAGE_B), ("refine_C", STAGE_C)]:
            proto = {
                "candidate_id": cand["candidate_id"],
                "contact_id": contact["contact_id"],
                "test_type": label,
                "rho": as_float(target["rho"]),
                "eps_cone": target.get("eps_cone", ""),
                "phi": target.get("phi", ""),
                "h": params["h"],
            }
            if processed_key(proto) in done:
                continue
            row = classify_point(candidate=cand, contact=contact, test_type=label, rho=as_float(target["rho"]), rho_original=contact["radius_original"], direction=direction, x0=x0, params=params, p=p, eqs=eqs, ref_cache=ref_cache, eps_cone=target.get("eps_cone", ""), phi=target.get("phi", ""), notes=f"{label} reproduction of adaptive TARGET")
            record(row)
            print(f"{cand['candidate_id']} {label} {contact['contact_id']} class={row['final_class']} target={row['target_hit']}", flush=True)

    raw_rows = read_rows(raw_path)
    summary_rows, decision_rows = summarize_decisions(candidates, raw_rows, contacts_by_candidate)
    write_csv(OUTDIR / "adaptive_contact_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(OUTDIR / "adaptive_contact_decision.csv", decision_rows, DECISION_FIELDS)
    plot_files = plot_outputs(candidates, raw_rows, contacts_by_candidate, eq_minus)
    elapsed = time.time() - start
    previous_times = [as_float(r.get("elapsed_sec")) for r in read_rows(PREVIOUS_REPRO_RAW) if as_float(r.get("elapsed_sec")) > 0]
    first_estimate = measured_first if measured_first is not None else (sum(previous_times) / len(previous_times) if previous_times else float("nan"))
    estimated_total = float(first_estimate * planned_initial) if math.isfinite(first_estimate) else float("nan")
    cost = {
        "planned_line_trajectories": planned_line,
        "planned_cone_trajectories_if_needed": planned_cone,
        "planned_initial_trajectories": planned_initial,
        "max_trajectories": int(args.max_trajectories),
        "measured_first_trajectory_sec": first_estimate,
        "estimated_total_sec": estimated_total,
        "estimated_total_hours": estimated_total / 3600.0 if math.isfinite(estimated_total) else None,
        "estimated_full_stage_A_days_avoided": 41.8,
        "duplicate_detection": duplicate_note,
        "elapsed_time_sec": elapsed,
    }
    files = [
        str(raw_path),
        str(OUTDIR / "adaptive_contact_summary.csv"),
        str(OUTDIR / "adaptive_contact_decision.csv"),
        str(OUTDIR / "adaptive_contact_summary.json"),
        str(OUTDIR / "adaptive_contact_report.md"),
        str(OUTDIR / "contact_directions.csv"),
        str(OUTDIR / "equilibria_used.csv"),
        *plot_files,
    ]
    (OUTDIR / "adaptive_contact_summary.json").write_text(json.dumps(json_safe({
        "candidates": [c["candidate_id"] for c in candidates],
        "contacts_used": {cid: len(v) for cid, v in contacts_by_candidate.items()},
        "cost_guard": cost,
        "summary": summary_rows,
        "decisions": decision_rows,
        "files": files,
        "scientific_note": "No hidden_verified label is declared; describing functions generate approximate seeds and validation is by causal Caputo memory integration.",
    }), indent=2, ensure_ascii=False), encoding="utf-8")
    build_report(summary_rows, decision_rows, cost, files)

    print("candidate_id,duplicate_group,contacts_used,total_tests,target_hits,robust_target_hit,open_like_cone_hit,hiddenness_status,elapsed_time,estimated_full_stage_A_days_avoided", flush=True)
    for row in summary_rows:
        print(",".join([
            str(row["candidate_id"]),
            str(row["duplicate_group"]),
            str(row["contacts_used"]),
            str(row["total_tests"]),
            str(row["target_hits"]),
            str(row["robust_target_hit"]),
            str(row["open_like_cone_hit"]),
            str(row["hiddenness_status"]),
            f"{elapsed:.3f}",
            "41.8",
        ]), flush=True)


if __name__ == "__main__":
    main()
