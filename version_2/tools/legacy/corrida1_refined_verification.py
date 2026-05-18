from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

import chua_initial_cond as chua
from equilibria_analysis import local_jacobian, region_for_sigma, solve_equilibria
from extended_search_utils import (
    chua_ic_params,
    fft_peak_and_entropy,
    json_safe,
    load_config,
    min_distance_to_points,
    trajectory_ranges,
    write_csv,
)


RAW_FIELDS = [
    "candidate_id",
    "candidate_mu",
    "candidate_theta",
    "candidate_branch",
    "candidate_seed_x",
    "candidate_seed_y",
    "candidate_seed_z",
    "equilibrium_id",
    "equilibrium_x",
    "equilibrium_y",
    "equilibrium_z",
    "radius",
    "sampling_mode",
    "sample_id",
    "perturbation_dx",
    "perturbation_dy",
    "perturbation_dz",
    "x0",
    "y0",
    "z0",
    "q",
    "h",
    "t_final",
    "memory_length",
    "memory_points",
    "memory_mode",
    "history_initialized_from_equilibrium_perturbation",
    "history_carried",
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
    "lyap_max_if_available",
    "final_class",
    "target_label",
    "target_hit",
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
    "decision_notes",
]


EQ_REFINED_FIELDS = [
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


CLASS_ORDER = [
    "equilibrium_convergence",
    "target_attractor",
    "other_bounded_nontrivial",
    "divergent",
    "numerical_failure",
    "ambiguous_long_transient",
]


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "si"}


def _memory_points(memory_length: float, h: float) -> int:
    return max(1, int(math.ceil(float(memory_length) / float(h)))) + 1


def _slug_path(path_value: str | Path) -> Path:
    p = Path(path_value)
    if p.exists():
        return p
    raw = str(path_value)
    if "\\" in raw:
        p2 = Path(raw.replace("\\", "/"))
        if p2.exists():
            return p2
    return p


def _candidate_id(row: Dict[str, Any]) -> str:
    return str(row.get("slug") or row.get("candidate_id") or "")


def load_selected_candidates(source_summary: str | Path, candidate_ids: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = _slug_path(source_summary)
    data = json.loads(path.read_text(encoding="utf-8"))
    records = list(data.get("records", []))
    by_slug = {_candidate_id(row): row for row in records}
    if data.get("best_candidate"):
        by_slug[_candidate_id(data["best_candidate"])] = data["best_candidate"]

    selected: List[Dict[str, Any]] = []
    missing: List[str] = []
    for cid in candidate_ids:
        row = by_slug.get(cid)
        if row is None:
            missing.append(cid)
            continue
        seed = np.asarray(row.get("seed", [float("nan")] * 3), dtype=float)
        target_seed = np.asarray(row.get("final_state_eps1", row.get("seed", [float("nan")] * 3)), dtype=float)
        selected.append({
            "candidate_id": cid,
            "df_family": row.get("method", "machado"),
            "branch": int(row.get("branch_index", row.get("branch", 0))),
            "mu": _float(row.get("mu")),
            "theta": _float(row.get("theta")),
            "q": _float(data.get("frac_order", row.get("q", 0.9998))),
            "A": _float(row.get("a0", row.get("A"))),
            "omega": _float(row.get("omega0", row.get("omega"))),
            "phase": _float(row.get("phase", row.get("theta", 0.0))),
            "k": _float(row.get("k0", row.get("k"))),
            "seed": seed,
            "target_seed": target_seed,
            "target_label": "TARGET=1",
            "source_hidden_summary": row.get("hidden_summary_json", ""),
            "source_record": row,
        })
    if missing:
        raise ValueError(f"No se encontraron candidatos en {path}: {missing}")
    return selected, data


def refined_equilibria_rows(cfg: Dict[str, Any], p: Dict[str, Any], outdir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    q = float(cfg["q"])
    theta = q * math.pi / 2.0
    eqs = solve_equilibria(p)
    rows: List[Dict[str, Any]] = []
    for eq_id, eq in eqs.items():
        eq = np.asarray(eq, dtype=float)
        J = local_jacobian(p, eq)
        eig = np.linalg.eigvals(J)
        margins = [abs(np.angle(v)) - theta for v in eig]
        region = region_for_sigma(float(eq[0]))
        rows.append({
            "eq_id": eq_id,
            "x": float(eq[0]),
            "y": float(eq[1]),
            "z": float(eq[2]),
            "region": region,
            "eig_1": complex(eig[0]),
            "eig_2": complex(eig[1]),
            "eig_3": complex(eig[2]),
            "min_arg_margin": float(min(margins)),
            "matignon_stable": bool(all(m > 0.0 for m in margins)),
            "nonsmooth_boundary": bool(region == "switching_boundary"),
        })
    order = {"E-": 0, "E0": 1, "E+": 2}
    rows.sort(key=lambda r: order.get(str(r["eq_id"]), 10))
    write_csv(outdir / "equilibria_refined_summary.csv", rows, EQ_REFINED_FIELDS)
    return rows, eqs


def deterministic_directions() -> List[Tuple[str, np.ndarray]]:
    dirs: List[Tuple[str, np.ndarray]] = []
    basis = np.eye(3, dtype=float)
    labels = ["x", "y", "z"]
    for label, vec in zip(labels, basis):
        dirs.append((f"axis_p_{label}", vec.copy()))
        dirs.append((f"axis_m_{label}", -vec.copy()))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                v = np.array([sx, sy, sz], dtype=float) / math.sqrt(3.0)
                dirs.append((f"diag_{int(sx):+d}_{int(sy):+d}_{int(sz):+d}", v))
    return dirs


def random_offsets(count: int, mode: str, rng: np.random.Generator) -> List[Tuple[str, np.ndarray]]:
    offsets: List[Tuple[str, np.ndarray]] = []
    for idx in range(int(count)):
        v = rng.normal(size=3)
        n = float(np.linalg.norm(v))
        if n < 1e-300:
            v = np.array([1.0, 0.0, 0.0], dtype=float)
            n = 1.0
        v = v / n
        if mode == "ball":
            v = v * float(rng.random() ** (1.0 / 3.0))
        offsets.append((f"rand_{idx:06d}", v.astype(float)))
    return offsets


def build_stage_a_plan(
    candidates: Sequence[Dict[str, Any]],
    eqs: Dict[str, np.ndarray],
    c1: Dict[str, Any],
) -> List[Dict[str, Any]]:
    radii = [float(x) for x in c1["radii"]]
    modes = list(c1.get("sampling_modes", ["ball", "sphere_shell"]))
    random_count = int(c1.get("samples_per_radius_random", 500))
    seed_base = int(c1.get("random_seed", 20260513))
    stage = c1["stages"]["A"]
    q = float(c1.get("q", 0.9998))
    plan: List[Dict[str, Any]] = []

    for cand_idx, cand in enumerate(candidates):
        for eq_idx, (eq_id, eq) in enumerate(eqs.items()):
            center = np.asarray(eq, dtype=float)
            for radius_idx, radius in enumerate(radii):
                for mode_idx, mode in enumerate(modes):
                    offsets = list(deterministic_directions())
                    rng_seed = seed_base + 1000003 * cand_idx + 9176 * eq_idx + 313 * radius_idx + 17 * mode_idx
                    rng = np.random.default_rng(rng_seed)
                    offsets.extend(random_offsets(random_count, mode, rng))
                    for sample_id, unit in offsets:
                        perturb = float(radius) * np.asarray(unit, dtype=float)
                        x0 = center + perturb
                        plan.append({
                            "stage": "A",
                            "candidate": cand,
                            "candidate_id": cand["candidate_id"],
                            "equilibrium_id": eq_id,
                            "equilibrium": center,
                            "radius": float(radius),
                            "sampling_mode": mode,
                            "sample_id": sample_id,
                            "perturbation": perturb,
                            "x0": x0,
                            "q": q,
                            "h": float(stage["h"]),
                            "memory_length": float(stage["memory_length"]),
                            "t_final": float(stage["t_final"]),
                        })
    return plan


def _group_key(row: Dict[str, Any]) -> Tuple[str, str, float, str, float, float, float]:
    return (
        str(row["candidate_id"]),
        str(row["equilibrium_id"]),
        float(row["radius"]),
        str(row["sampling_mode"]),
        float(row["h"]),
        float(row["memory_length"]),
        float(row["t_final"]),
    )


def build_followup_plan(
    candidates: Sequence[Dict[str, Any]],
    eqs: Dict[str, np.ndarray],
    c1: Dict[str, Any],
    summary_rows: Sequence[Dict[str, Any]],
    stage_name: str,
) -> List[Dict[str, Any]]:
    if stage_name not in {"B", "C"}:
        return []
    trigger_rows = [r for r in summary_rows if int(_float(r.get("n_target_attractor"), 0.0)) > 0]
    if stage_name == "C":
        trigger_rows = [r for r in trigger_rows if str(r.get("robust_target_hit", "")).lower() == "true"]
    if not trigger_rows:
        return []

    cand_by_id = {str(c["candidate_id"]): c for c in candidates}
    stage = c1["stages"][stage_name]
    random_count = int(c1.get("samples_per_radius_random", 500))
    seed_base = int(c1.get("random_seed", 20260513)) + (20000000 if stage_name == "B" else 30000000)
    q = float(c1.get("q", 0.9998))
    plan: List[Dict[str, Any]] = []
    seen_groups = set()
    for gr in trigger_rows:
        gid = (
            str(gr["candidate_id"]),
            str(gr["equilibrium_id"]),
            float(gr["radius"]),
            str(gr["sampling_mode"]),
        )
        if gid in seen_groups:
            continue
        seen_groups.add(gid)
        cand = cand_by_id.get(gid[0])
        eq = eqs.get(gid[1])
        if cand is None or eq is None:
            continue
        center = np.asarray(eq, dtype=float)
        offsets = list(deterministic_directions())
        rng = np.random.default_rng(seed_base + len(plan))
        offsets.extend(random_offsets(random_count, gid[3], rng))
        for sample_id, unit in offsets:
            perturb = gid[2] * np.asarray(unit, dtype=float)
            plan.append({
                "stage": stage_name,
                "candidate": cand,
                "candidate_id": cand["candidate_id"],
                "equilibrium_id": gid[1],
                "equilibrium": center,
                "radius": gid[2],
                "sampling_mode": gid[3],
                "sample_id": sample_id,
                "perturbation": perturb,
                "x0": center + perturb,
                "q": q,
                "h": float(stage["h"]),
                "memory_length": float(stage["memory_length"]),
                "t_final": float(stage["t_final"]),
            })
    return plan


def estimate_plan_work(plan: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_stage: Dict[str, Dict[str, float]] = defaultdict(lambda: {"simulations": 0, "steps": 0, "memory_points": 0, "work_units": 0})
    total_units = 0
    for item in plan:
        steps = int(math.ceil(float(item["t_final"]) / float(item["h"])))
        mem = _memory_points(float(item["memory_length"]), float(item["h"]))
        units = steps * mem
        st = by_stage[str(item.get("stage", "A"))]
        st["simulations"] += 1
        st["steps"] = max(st["steps"], steps)
        st["memory_points"] = max(st["memory_points"], mem)
        st["work_units"] += units
        total_units += units
    return {
        "total_simulations": len(plan),
        "total_work_units_step_memory": int(total_units),
        "by_stage": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in sorted(by_stage.items())},
    }


def ensure_csv(path: Path, fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(fields)).writeheader()


def append_csv_row(path: Path, row: Dict[str, Any], fields: Sequence[str]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writerow({k: csv_value(row.get(k, "")) for k in fields})


def csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return ";".join(str(float(x)) for x in value.ravel())
    if isinstance(value, (list, tuple)):
        return ";".join(str(x) for x in value)
    return value


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def processed_key_from_plan(item: Dict[str, Any]) -> Tuple[Any, ...]:
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


def processed_key_from_row(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row["candidate_id"]),
        str(row["equilibrium_id"]),
        f"{float(row['radius']):.17g}",
        str(row["sampling_mode"]),
        str(row["sample_id"]),
        f"{float(row['h']):.17g}",
        f"{float(row['memory_length']):.17g}",
        f"{float(row['t_final']):.17g}",
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


def section_hit_fraction(section: np.ndarray, ref: np.ndarray, sec_tol: float) -> Tuple[int, int, float]:
    if section.size == 0 or ref.size == 0:
        return int(section.shape[0] if section.ndim == 2 else 0), 0, 0.0
    hits = 0
    for point in section:
        d = np.linalg.norm(ref - point.reshape(1, 2), axis=1)
        if float(np.min(d)) <= float(sec_tol):
            hits += 1
    total = int(section.shape[0])
    return total, int(hits), float(hits / max(total, 1))


def tail_stats(traj: np.ndarray, h: float, tail_fraction: float) -> Dict[str, float]:
    X = np.asarray(traj, dtype=float)
    states = X[:, 1:4] if X.ndim == 2 and X.shape[1] >= 4 else np.empty((0, 3), dtype=float)
    if states.shape[0] == 0:
        return {
            "mean_x_tail": float("nan"),
            "mean_y_tail": float("nan"),
            "mean_z_tail": float("nan"),
            "var_x_tail": float("nan"),
            "var_y_tail": float("nan"),
            "var_z_tail": float("nan"),
        }
    start = max(0, int((1.0 - float(tail_fraction)) * states.shape[0]))
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


def load_reference_from_csv(path_value: str | Path) -> np.ndarray:
    path = _slug_path(path_value)
    if not path.exists():
        return np.empty((0, 2), dtype=float)
    rows: List[Tuple[float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "y" in row and "z" in row:
                rows.append((float(row["y"]), float(row["z"])))
    return np.asarray(rows, dtype=float)


def reference_section_for_candidate(
    cand: Dict[str, Any],
    p: Dict[str, Any],
    q: float,
    h: float,
    memory_length: float,
    t_final: float,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, str]:
    hidden_summary = cand.get("source_hidden_summary")
    if hidden_summary:
        try:
            hs = json.loads(_slug_path(hidden_summary).read_text(encoding="utf-8"))
            ref_csv = hs.get("files", {}).get("ref_csv_out")
            ref = load_reference_from_csv(ref_csv) if ref_csv else np.empty((0, 2), dtype=float)
            if ref.shape[0] >= int(cfg.get("min_sec_match", 20)):
                return ref, "source_hidden_reference_csv"
        except Exception:
            pass

    rhs = lambda x: chua.rhs_original(x, p)
    traj = chua.efork3_integrate(rhs, cand["target_seed"], qord=q, h=h, Lm=memory_length, t_f=t_final)
    burn = float(cfg.get("section_burn_fraction", 0.5)) * float(t_final)
    ref = section_points(traj, p, burn, int(cfg.get("reference_max_section_points", 240)))
    return ref, "integrated_target_seed"


def simulate_and_classify(
    item: Dict[str, Any],
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    reference_cache: Dict[Tuple[str, float, float, float], Tuple[np.ndarray, str]],
) -> Dict[str, Any]:
    cand = item["candidate"]
    q = float(item["q"])
    h = float(item["h"])
    memory_length = float(item["memory_length"])
    t_final = float(item["t_final"])
    memory_points = _memory_points(memory_length, h)
    rhs = lambda x: chua.rhs_original(x, p)
    x0 = np.asarray(item["x0"], dtype=float)
    eq_arrays = {name: np.asarray(val, dtype=float) for name, val in eqs.items()}
    notes: List[str] = [f"stage={item.get('stage', 'A')}"]

    base_row = {
        "candidate_id": cand["candidate_id"],
        "candidate_mu": cand["mu"],
        "candidate_theta": cand["theta"],
        "candidate_branch": cand["branch"],
        "candidate_seed_x": float(cand["seed"][0]),
        "candidate_seed_y": float(cand["seed"][1]),
        "candidate_seed_z": float(cand["seed"][2]),
        "equilibrium_id": item["equilibrium_id"],
        "equilibrium_x": float(item["equilibrium"][0]),
        "equilibrium_y": float(item["equilibrium"][1]),
        "equilibrium_z": float(item["equilibrium"][2]),
        "radius": float(item["radius"]),
        "sampling_mode": item["sampling_mode"],
        "sample_id": item["sample_id"],
        "perturbation_dx": float(item["perturbation"][0]),
        "perturbation_dy": float(item["perturbation"][1]),
        "perturbation_dz": float(item["perturbation"][2]),
        "x0": float(x0[0]),
        "y0": float(x0[1]),
        "z0": float(x0[2]),
        "q": q,
        "h": h,
        "t_final": t_final,
        "memory_length": memory_length,
        "memory_points": memory_points,
        "memory_mode": "truncated_caputo_window",
        "history_initialized_from_equilibrium_perturbation": True,
        "history_carried": False,
        "target_label": cand.get("target_label", "TARGET=1"),
        "lyap_max_if_available": "",
    }
    try:
        traj = chua.efork3_integrate(rhs, x0, qord=q, h=h, Lm=memory_length, t_f=t_final)
    except Exception as exc:
        return {
            **base_row,
            "final_class": "numerical_failure",
            "target_hit": False,
            "numerical_status": "integration_exception",
            "notes": str(exc),
        }

    states = traj[:, 1:4]
    if states.size == 0 or not np.all(np.isfinite(states)):
        return {
            **base_row,
            "final_class": "numerical_failure",
            "target_hit": False,
            "numerical_status": "nonfinite_trajectory",
            "notes": "trajectory contains NaN or Inf",
        }

    final = states[-1]
    final_norm = float(np.linalg.norm(final))
    divergence_norm = float(cfg.get("divergence_norm", 120.0))
    min_dist = min_distance_to_points(states, eq_arrays.values())
    final_dist = {name: float(np.linalg.norm(final - eq)) for name, eq in eq_arrays.items()}
    tail = tail_stats(traj, h, float(cfg.get("tail_fraction", 0.2)))
    ranges = trajectory_ranges(traj)
    numerical_status = "ok"

    final_class = "ambiguous_long_transient"
    target_hit = False
    if final_norm > divergence_norm or float(np.max(np.linalg.norm(states, axis=1))) > divergence_norm:
        final_class = "divergent"
    else:
        eq_radius = float(cfg.get("equilibrium_radius", 1e-3))
        nearest_eq = min(final_dist.items(), key=lambda kv: kv[1])
        tail_mean = np.array([tail["mean_x_tail"], tail["mean_y_tail"], tail["mean_z_tail"]], dtype=float)
        tail_dist = float(np.linalg.norm(tail_mean - eq_arrays[nearest_eq[0]]))
        if nearest_eq[1] <= eq_radius and tail_dist <= 2.0 * eq_radius:
            final_class = "equilibrium_convergence"
            notes.append(f"converged_to={nearest_eq[0]}")
        else:
            cache_key = (cand["candidate_id"], h, memory_length, t_final)
            if cache_key not in reference_cache:
                try:
                    reference_cache[cache_key] = reference_section_for_candidate(cand, p, q, h, memory_length, t_final, cfg)
                except Exception as exc:
                    reference_cache[cache_key] = (np.empty((0, 2), dtype=float), f"reference_failed:{exc}")
            ref, ref_source = reference_cache[cache_key]
            notes.append(f"reference={ref_source};reference_points={ref.shape[0]}")
            burn = float(cfg.get("section_burn_fraction", 0.5)) * t_final
            sec = section_points(traj, p, burn, int(cfg.get("test_max_section_points", 100)))
            sec_total, sec_hits, hit_frac = section_hit_fraction(sec, ref, float(cfg.get("section_tolerance", 0.12)))
            notes.append(f"sec_total={sec_total};sec_hits={sec_hits};hit_frac={hit_frac:.6g}")
            if sec_total >= int(cfg.get("min_sec_match", 20)) and hit_frac >= float(cfg.get("hit_fraction_required", 0.70)):
                final_class = "target_attractor"
                target_hit = True
            elif sec_total < int(cfg.get("min_sec_match", 20)):
                final_class = "ambiguous_long_transient"
            else:
                variance = max(float(tail["var_x_tail"]), float(tail["var_y_tail"]), float(tail["var_z_tail"]))
                range_max = max(float(ranges["range_x"]), float(ranges["range_y"]), float(ranges["range_z"]))
                if variance > float(cfg.get("nontrivial_variance_tol", 1e-6)) or range_max > float(cfg.get("nontrivial_range_tol", 1e-2)):
                    final_class = "other_bounded_nontrivial"

    return {
        **base_row,
        "final_x": float(final[0]),
        "final_y": float(final[1]),
        "final_z": float(final[2]),
        "final_norm": final_norm,
        "min_dist_to_equilibria": min_dist,
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


def aggregate_raw(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, float, str, float, float, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        try:
            groups[_group_key(row)].append(row)
        except Exception:
            continue
    summary: List[Dict[str, Any]] = []
    for key, rows in groups.items():
        classes = Counter(str(r.get("final_class", "")) for r in rows)
        n = len(rows)
        n_target = classes.get("target_attractor", 0)
        most = classes.most_common(1)[0][0] if classes else ""
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
            "most_common_class": most,
        })
    mark_robust_hits(summary)
    summary.sort(key=lambda r: (str(r["candidate_id"]), str(r["equilibrium_id"]), float(r["radius"]), str(r["sampling_mode"]), float(r["h"]), float(r["memory_length"]), float(r["t_final"])))
    return summary


def mark_robust_hits(summary: List[Dict[str, Any]]) -> None:
    for row in summary:
        if int(row.get("n_target_attractor", 0)) >= 3:
            row["robust_target_hit"] = True

    by_candidate_eq_radius_cfg: Dict[Tuple[str, str, float, float, float, float], List[Dict[str, Any]]] = defaultdict(list)
    by_candidate_eq_radius_mode_lm_tf: Dict[Tuple[str, str, float, str, float, float], List[Dict[str, Any]]] = defaultdict(list)
    by_candidate_eq_radius_mode_h_tf: Dict[Tuple[str, str, float, str, float, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in summary:
        if int(row.get("n_target_attractor", 0)) <= 0:
            continue
        by_candidate_eq_radius_cfg[(row["candidate_id"], row["equilibrium_id"], float(row["radius"]), float(row["h"]), float(row["memory_length"]), float(row["t_final"]))].append(row)
        by_candidate_eq_radius_mode_lm_tf[(row["candidate_id"], row["equilibrium_id"], float(row["radius"]), row["sampling_mode"], float(row["memory_length"]), float(row["t_final"]))].append(row)
        by_candidate_eq_radius_mode_h_tf[(row["candidate_id"], row["equilibrium_id"], float(row["radius"]), row["sampling_mode"], float(row["h"]), float(row["t_final"]))].append(row)

    for rows in by_candidate_eq_radius_cfg.values():
        modes = {str(r["sampling_mode"]) for r in rows}
        if {"ball", "sphere_shell"}.issubset(modes):
            for r in rows:
                r["robust_target_hit"] = True

    for rows in by_candidate_eq_radius_mode_lm_tf.values():
        hs = {float(r["h"]) for r in rows}
        if any(abs(h - 0.01) < 1e-12 for h in hs) and any(abs(h - 0.005) < 1e-12 for h in hs):
            for r in rows:
                r["robust_target_hit"] = True

    for rows in by_candidate_eq_radius_mode_h_tf.values():
        lms = {float(r["memory_length"]) for r in rows}
        if len(lms) >= 2:
            for r in rows:
                r["robust_target_hit"] = True


def decide_hiddenness(candidates: Sequence[Dict[str, Any]], summary_rows: Sequence[Dict[str, Any]], raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_cand: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_cand[str(row["candidate_id"])].append(row)
    raw_by_cand = Counter(str(r.get("candidate_id", "")) for r in raw_rows)
    decisions: List[Dict[str, Any]] = []
    for cand in candidates:
        cid = cand["candidate_id"]
        rows = by_cand.get(cid, [])
        total_hits = int(sum(int(r.get("n_target_attractor", 0)) for r in rows))
        robust_rows = [r for r in rows if str(r.get("robust_target_hit", "")).lower() == "true"]
        hit_rows = [r for r in rows if int(r.get("n_target_attractor", 0)) > 0]
        no_hit_rows = [r for r in rows if int(r.get("n_target_attractor", 0)) == 0 and int(r.get("n_samples", 0)) > 0]
        max_frac = max([float(r.get("target_hit_fraction", 0.0)) for r in rows], default=0.0)
        smallest_hit = min([float(r["radius"]) for r in hit_rows], default=float("nan"))
        largest_without = max([float(r["radius"]) for r in no_hit_rows], default=float("nan"))
        blocking = sorted(robust_rows, key=lambda r: (float(r["radius"]), str(r["equilibrium_id"])))
        if robust_rows:
            status = "not_supported_by_refined_neighborhood_test"
            notes = "A robust target hit was found from an equilibrium neighborhood."
        elif total_hits > 0:
            status = "inconclusive_isolated_hit"
            notes = "Only isolated target hits were found in completed groups."
            blocking = sorted(hit_rows, key=lambda r: (float(r["radius"]), str(r["equilibrium_id"])))
        elif raw_by_cand.get(cid, 0) > 0:
            status = "compatible_with_hiddenness_under_tested_radii"
            notes = "No target hits were found in the completed refined samples. This is not hidden_verified."
            blocking = []
        else:
            status = "not_evaluated_cost_guard"
            notes = "No refined trajectories were executed because execution_mode is estimate_only or max_trajectories_this_run is zero."
            blocking = []
        b = blocking[0] if blocking else {}
        decisions.append({
            "candidate_id": cid,
            "hiddenness_status": status,
            "blocking_equilibrium": b.get("equilibrium_id", ""),
            "blocking_radius": b.get("radius", ""),
            "blocking_sampling_mode": b.get("sampling_mode", ""),
            "blocking_h": b.get("h", ""),
            "blocking_memory_length": b.get("memory_length", ""),
            "blocking_t_final": b.get("t_final", ""),
            "max_target_hit_fraction": max_frac,
            "total_target_hits": total_hits,
            "robust_target_hit_found": bool(robust_rows),
            "smallest_radius_with_target_hit": "" if math.isnan(smallest_hit) else smallest_hit,
            "largest_radius_without_target_hit": "" if math.isnan(largest_without) else largest_without,
            "decision_notes": notes,
        })
    return decisions


def write_plots(outdir: Path, candidates: Sequence[Dict[str, Any]], raw_rows: Sequence[Dict[str, Any]], summary_rows: Sequence[Dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotdir = outdir / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)
    cand_names = [c["candidate_id"] for c in candidates]

    for idx, cid in enumerate(cand_names, start=1):
        rows = [r for r in summary_rows if r.get("candidate_id") == cid and r.get("equilibrium_id") == "E-"]
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for mode in sorted({str(r.get("sampling_mode")) for r in rows} or {"ball", "sphere_shell"}):
            mr = sorted([r for r in rows if str(r.get("sampling_mode")) == mode], key=lambda r: float(r["radius"]))
            if mr:
                ax.plot([float(r["radius"]) for r in mr], [int(r["n_target_attractor"]) for r in mr], marker="o", label=mode)
        ax.set_xscale("log")
        ax.set_xlabel("radius")
        ax.set_ylabel("target hits")
        if rows:
            ax.legend()
        fig.tight_layout()
        fig.savefig(plotdir / f"target_hits_vs_radius_Eminus_candidate{idx}.png", dpi=180)
        plt.close(fig)

        rows = [r for r in summary_rows if r.get("candidate_id") == cid]
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        by_radius: Dict[float, Counter[str]] = defaultdict(Counter)
        totals: Counter[float] = Counter()
        for r in rows:
            radius = float(r["radius"])
            totals[radius] += int(r["n_samples"])
            for cls in CLASS_ORDER:
                key = "n_" + cls
                by_radius[radius][cls] += int(r.get(key, 0))
        for cls in CLASS_ORDER:
            xs = sorted(by_radius.keys())
            if xs:
                ys = [by_radius[x][cls] / max(totals[x], 1) for x in xs]
                ax.plot(xs, ys, marker="o", label=cls)
        ax.set_xscale("log")
        ax.set_xlabel("radius")
        ax.set_ylabel("class fraction")
        if rows:
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plotdir / f"class_fraction_vs_radius_all_equilibria_candidate{idx}.png", dpi=180)
        plt.close(fig)

        rows = [r for r in raw_rows if r.get("candidate_id") == cid]
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        for cls in CLASS_ORDER:
            rr = [r for r in rows if r.get("final_class") == cls]
            if rr:
                ax.scatter([float(r["x0"]) for r in rr], [float(r["y0"]) for r in rr], s=8, alpha=0.7, label=cls)
        ax.set_xlabel("x0")
        ax.set_ylabel("y0")
        if rows:
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plotdir / f"refined_neighborhood_samples_xy_candidate{idx}.png", dpi=180)
        plt.close(fig)


def write_report(
    outdir: Path,
    cfg: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    eq_rows: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    decision_rows: Sequence[Dict[str, Any]],
    estimate: Dict[str, Any],
    executed_new: int,
    skipped_existing: int,
    execution_mode: str,
) -> None:
    c1 = cfg["corrida1_refined_verification"]
    lines = [
        "# Corrida 1: Reverificacion Fina Machado",
        "",
        "Esta corrida no declara `hidden_verified`. Solo clasifica compatibilidad bajo radios probados o interseccion detectada con vecindades de equilibrio.",
        "",
        "## Parametros",
        "",
        f"- q: `{c1.get('q', cfg.get('q'))}`",
        f"- h_values: `{c1.get('h_values')}`",
        f"- memory_length_values: `{c1.get('memory_length_values')}`",
        f"- t_final_values: `{c1.get('t_final_values')}`",
        f"- radios: `{c1.get('radii')}`",
        f"- samples_per_radius_random: `{c1.get('samples_per_radius_random')}`",
        f"- random_seed: `{c1.get('random_seed')}`",
        f"- execution_mode: `{execution_mode}`",
        f"- memory_points convention: `ceil(memory_length / h) + 1`; for Etapa A this is `{_memory_points(c1['stages']['A']['memory_length'], c1['stages']['A']['h'])}` points.",
        "",
        "## Estimacion De Costo",
        "",
        f"- simulaciones planificadas actualmente: `{estimate.get('total_simulations', 0)}`",
        f"- unidades aproximadas paso-memoria: `{estimate.get('total_work_units_step_memory', 0)}`",
        f"- nuevas trayectorias ejecutadas en esta llamada: `{executed_new}`",
        f"- trayectorias ya existentes omitidas por reanudacion: `{skipped_existing}`",
        "",
    ]
    if execution_mode != "run" or executed_new == 0:
        lines.extend([
            "La Etapa A completa con la configuracion solicitada es costosa: 2 candidatos, 3 equilibrios, 8 radios, 2 modos y 514 muestras por grupo.",
            "Por eso el modo seguro no lanza la corrida completa accidentalmente. El archivo raw queda listo para reanudar con `CORRIDA1_EXECUTION_MODE=run` y un limite por llamada.",
            "",
        ])
    lines.extend(["## Equilibrios", ""])
    for row in eq_rows:
        lines.append(
            f"- `{row['eq_id']}` = ({float(row['x']):.10g}, {float(row['y']):.10g}, {float(row['z']):.10g}), "
            f"region={row['region']}, Matignon stable={row['matignon_stable']}, margin={float(row['min_arg_margin']):.6g}"
        )
    lines.extend(["", "## Resumen Por Candidato", ""])
    by_cand = defaultdict(list)
    for row in summary_rows:
        by_cand[str(row["candidate_id"])].append(row)
    decisions = {str(r["candidate_id"]): r for r in decision_rows}
    for cand in candidates:
        cid = cand["candidate_id"]
        rows = by_cand.get(cid, [])
        total_samples = sum(int(r.get("n_samples", 0)) for r in rows)
        total_hits = sum(int(r.get("n_target_attractor", 0)) for r in rows)
        hit_radii = sorted({float(r["radius"]) for r in rows if int(r.get("n_target_attractor", 0)) > 0})
        decision = decisions.get(cid, {})
        lines.extend([
            f"### {cid}",
            "",
            f"- trayectorias completadas: `{total_samples}`",
            f"- impactos TARGET=1: `{total_hits}`",
            f"- radios con impactos: `{hit_radii}`",
            f"- decision: `{decision.get('hiddenness_status', '')}`",
            f"- notas: {decision.get('decision_notes', '')}",
            "",
        ])
    lines.extend([
        "## Conclusion Limitada",
        "",
        "Las salidas usan `compatible_with_hiddenness_under_tested_radii`, `not_supported_by_refined_neighborhood_test` o `inconclusive_isolated_hit` solo cuando hay trayectorias ejecutadas suficientes para esa decision. En modo de estimacion aparece `not_evaluated_cost_guard`.",
    ])
    (outdir / "corrida1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_corrida1_refined_verification(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    c1 = cfg.get("corrida1_refined_verification", {})
    if not c1:
        raise ValueError("La configuracion no contiene corrida1_refined_verification.")
    c1 = dict(c1)
    c1.setdefault("q", cfg.get("q", 0.9998))
    cfg["q"] = float(c1.get("q", cfg.get("q", 0.9998)))
    cfg["corrida1_refined_verification"] = c1
    outdir = Path(c1.get("output_dir", "outputs/extended_search/corrida1"))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    p = chua_ic_params(cfg)
    chua.PARAMS = p
    chua.QORD = np.float64(float(cfg["q"]))

    candidates, source_summary = load_selected_candidates(c1["source_summary"], c1["candidate_ids"])
    eq_rows, eqs = refined_equilibria_rows(cfg, p, outdir)

    raw_path = outdir / "refined_candidate_verification_raw.csv"
    summary_path = outdir / "refined_candidate_verification_summary.csv"
    decision_path = outdir / "refined_hiddenness_decision.csv"
    ensure_csv(raw_path, RAW_FIELDS)

    raw_rows = read_csv_rows(raw_path)
    processed = {processed_key_from_row(r) for r in raw_rows}
    stage_a_plan = build_stage_a_plan(candidates, eqs, c1)
    summary_existing = aggregate_raw(raw_rows)
    stage_b_plan = build_followup_plan(candidates, eqs, c1, summary_existing, "B")
    stage_c_plan = build_followup_plan(candidates, eqs, c1, summary_existing, "C")
    plan = stage_a_plan + stage_b_plan + stage_c_plan
    estimate = estimate_plan_work(plan)

    execution_mode = os.environ.get("CORRIDA1_EXECUTION_MODE", str(c1.get("execution_mode", "estimate_only"))).strip().lower()
    force_full = _bool_env("CORRIDA1_FORCE_FULL", bool(c1.get("force_full", False)))
    max_new = int(os.environ.get("CORRIDA1_MAX_TRAJECTORIES_THIS_RUN", c1.get("max_trajectories_this_run", 0)))
    guard_max = int(c1.get("cost_guard_max_trajectories", 2000))

    print("Corrida 1 refined verification estimate", flush=True)
    print(f"planned_simulations={estimate['total_simulations']}", flush=True)
    print(f"estimated_step_memory_work_units={estimate['total_work_units_step_memory']}", flush=True)
    print(f"execution_mode={execution_mode}", flush=True)
    print(f"max_trajectories_this_run={max_new}", flush=True)

    executable_items = [item for item in plan if processed_key_from_plan(item) not in processed]
    skipped_existing = len(plan) - len(executable_items)
    if execution_mode != "run":
        executable_items = []
    elif not force_full:
        if max_new <= 0:
            executable_items = []
        else:
            executable_items = executable_items[:max_new]
        if len(executable_items) > guard_max:
            executable_items = executable_items[:guard_max]

    reference_cache: Dict[Tuple[str, float, float, float], Tuple[np.ndarray, str]] = {}
    executed_new = 0
    for item in executable_items:
        row = simulate_and_classify(item, p, eqs, c1, reference_cache)
        append_csv_row(raw_path, row, RAW_FIELDS)
        executed_new += 1
        if executed_new % 10 == 0:
            print(f"Corrida 1 progress: {executed_new} new trajectories", flush=True)

    raw_rows = read_csv_rows(raw_path)
    summary_rows = aggregate_raw(raw_rows)
    decision_rows = decide_hiddenness(candidates, summary_rows, raw_rows)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    write_csv(decision_path, decision_rows, DECISION_FIELDS)
    write_plots(outdir, candidates, raw_rows, summary_rows)
    write_report(outdir, cfg, candidates, eq_rows, summary_rows, decision_rows, estimate, executed_new, skipped_existing, execution_mode)

    summary_json = {
        "config": c1,
        "source_summary": {
            "run_mode": source_summary.get("run_mode"),
            "frac_order": source_summary.get("frac_order"),
            "records": len(source_summary.get("records", [])),
        },
        "candidates": [
            {
                "candidate_id": c["candidate_id"],
                "mu": c["mu"],
                "theta": c["theta"],
                "branch": c["branch"],
                "A": c["A"],
                "omega": c["omega"],
                "seed": c["seed"],
                "target_seed": c["target_seed"],
            }
            for c in candidates
        ],
        "estimate": estimate,
        "execution_mode": execution_mode,
        "executed_new_trajectories": executed_new,
        "skipped_existing_trajectories": skipped_existing,
        "raw_rows_total": len(raw_rows),
        "summary_rows_total": len(summary_rows),
        "decisions": decision_rows,
        "outputs": {
            "raw_csv": str(raw_path),
            "summary_csv": str(summary_path),
            "decision_csv": str(decision_path),
            "equilibria_csv": str(outdir / "equilibria_refined_summary.csv"),
            "report_md": str(outdir / "corrida1_report.md"),
        },
    }
    (outdir / "corrida1_summary.json").write_text(json.dumps(json_safe(summary_json), indent=2, ensure_ascii=False), encoding="utf-8")

    print("candidate_id,total_samples,total_target_hits,smallest_radius_with_target_hit,blocking_equilibrium,robust_target_hit_found,hiddenness_status", flush=True)
    samples_by_cand = Counter(str(r.get("candidate_id", "")) for r in raw_rows)
    for row in decision_rows:
        print(
            ",".join([
                str(row.get("candidate_id", "")),
                str(samples_by_cand.get(str(row.get("candidate_id", "")), 0)),
                str(row.get("total_target_hits", "")),
                str(row.get("smallest_radius_with_target_hit", "")),
                str(row.get("blocking_equilibrium", "")),
                str(row.get("robust_target_hit_found", "")),
                str(row.get("hiddenness_status", "")),
            ]),
            flush=True,
        )
    print(f"Corrida 1 outputs: {outdir}", flush=True)
