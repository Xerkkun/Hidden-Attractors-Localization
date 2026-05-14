#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

import chua_initial_cond as chua
from equilibria_analysis import local_jacobian, solve_equilibria
from extended_search_utils import chua_ic_params, fft_peak_and_entropy, json_safe, trajectory_ranges, write_csv
from lure_candidate_manifest import load_config


ROOT = Path(__file__).resolve().parent
TARGET_Q = 0.9998
OUTDIR = ROOT / "outputs" / "lure_biased_multiparam_q09998"
MACHADO_CANDIDATE_ID = "branch_0_mu_4p00000_theta_0p00000"

CONT_FIELDS = [
    "candidate_id",
    "seed_id",
    "route_id",
    "q",
    "eta",
    "sigma0_current",
    "step_index",
    "h",
    "memory_length",
    "memory_points",
    "memory_carried",
    "bounded",
    "diverged",
    "equilibrium_hit",
    "final_class",
    "final_x",
    "final_y",
    "final_z",
    "range_x",
    "range_y",
    "range_z",
    "fft_peak",
    "psd_entropy",
    "notes",
]

SURVIVOR_FIELDS = [
    "candidate_id",
    "seed_id",
    "route_id",
    "q",
    "final_class",
    "memory_carried",
    "continuation_reliability",
    "final_x",
    "final_y",
    "final_z",
    "range_x",
    "range_y",
    "range_z",
    "fft_peak",
    "psd_entropy",
    "hiddenness_status",
    "notes",
]

PATH_FIELDS = ["candidate_id", "seed_id", "route_id", "step_index", "q", "eta", "sigma0_current", "x", "y", "z"]

EARLY_RAW_FIELDS = [
    "candidate_id",
    "seed_id",
    "equilibrium_id",
    "direction_label",
    "rho",
    "q",
    "h",
    "memory_length",
    "memory_points",
    "t_final",
    "x0",
    "y0",
    "z0",
    "final_x",
    "final_y",
    "final_z",
    "range_x",
    "range_y",
    "range_z",
    "fft_peak",
    "psd_entropy",
    "final_class",
    "target_hit",
    "hiddenness_status",
    "notes",
]

EARLY_SUMMARY_FIELDS = [
    "candidate_id",
    "n_Eplus_TARGET",
    "n_E0_TARGET",
    "n_Eminus_TARGET",
    "hiddenness_status",
    "notes",
]

ROBUST_FIELDS = [
    "candidate_id",
    "case_id",
    "q",
    "h",
    "memory_length",
    "memory_points",
    "t_final",
    "bounded",
    "final_class",
    "range_x",
    "range_y",
    "range_z",
    "mean_tail",
    "var_tail",
    "fft_peak",
    "psd_entropy",
    "robust_attractor",
    "notes",
]

COMPARE_FIELDS = [
    "candidate_id",
    "machado_candidate_id",
    "q_lure",
    "q_machado",
    "lure_final_x",
    "lure_final_y",
    "lure_final_z",
    "machado_final_x",
    "machado_final_y",
    "machado_final_z",
    "lure_range_x",
    "lure_range_y",
    "lure_range_z",
    "machado_range_x",
    "machado_range_y",
    "machado_range_z",
    "final_state_distance",
    "range_relative_distance",
    "likely_same_attractor_as_machado",
    "distinct_candidate",
    "notes",
]


@dataclass
class FractionalHistory:
    t_window: np.ndarray
    X_window: np.ndarray
    f_window: np.ndarray | None
    q: float
    h: float
    memory_length: float

    @property
    def memory_points(self) -> int:
        return int(self.X_window.shape[0])

    def as_efork_history(self) -> np.ndarray:
        return np.column_stack([self.t_window, self.X_window])

    @classmethod
    def from_trajectory(
        cls,
        traj: np.ndarray,
        q: float,
        h: float,
        memory_length: float,
        rhs: Any | None = None,
    ) -> "FractionalHistory":
        window = chua.extract_memory_window(traj, Lm=memory_length, h=h)
        f_window = None
        if rhs is not None:
            try:
                f_window = np.vstack([rhs(row[1:4]) for row in window]).astype(float)
            except Exception:
                f_window = None
        return cls(
            t_window=window[:, 0].copy(),
            X_window=window[:, 1:4].copy(),
            f_window=f_window,
            q=float(q),
            h=float(h),
            memory_length=float(memory_length),
        )


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def q_tag(q: float) -> str:
    return f"{float(q):.5f}".replace("-", "m").replace(".", "p")


def require_q09998(cfg: Dict[str, Any], *, context: str) -> float:
    q = float(cfg.get("q", cfg.get("frac_order", float("nan"))))
    frac = float(cfg.get("frac_order", q))
    cont_q = float(cfg.get("continuation", {}).get("q_fixed", q))
    vals = [q, frac, cont_q]
    if any(not math.isfinite(v) for v in vals) or any(abs(v - TARGET_Q) > 5e-10 for v in vals):
        raise ValueError(f"{context}: all q values must be {TARGET_Q}; got q={q}, frac_order={frac}, continuation.q_fixed={cont_q}.")
    if bool(cfg.get("enforce_q_consistency", True)) is not True:
        raise ValueError(f"{context}: enforce_q_consistency must be true.")
    return q


def outdir_from_config(cfg: Dict[str, Any]) -> Path:
    outdir = Path(cfg.get("outputs", {}).get("root", "outputs/lure_biased_multiparam_q09998"))
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    if outdir.resolve() != OUTDIR.resolve():
        raise ValueError(f"Output dir must be {rel(OUTDIR)}; got {rel(outdir)}")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
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
        writer.writerow({k: csv_value(row.get(k, "")) for k in fields})


def csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return ";".join(str(float(x)) for x in value.ravel())
    if isinstance(value, (list, tuple)):
        return ";".join(str(x) for x in value)
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "si", "on", "ok"}


def load_candidates_and_seeds(outdir: Path, q: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates = read_csv_rows(outdir / "biased_lure_candidates.csv")
    seeds = read_csv_rows(outdir / "biased_lure_seed_bank.csv")
    for row in candidates + seeds:
        row_q = finite_float(row.get("q"))
        if not math.isfinite(row_q) or abs(row_q - q) > 5e-10:
            raise ValueError(f"q mismatch in biased Lure input row: {row}")
    return candidates, seeds


def selected_seed_items(
    cfg: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    seeds: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    cont = cfg["continuation"]
    by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for seed in seeds:
        if not truthy(seed.get("valid_seed")):
            continue
        by_candidate.setdefault(str(seed.get("candidate_id")), []).append(seed)
    ranked = list(candidates)[: int(cont.get("max_candidates", 6))]
    items: List[Dict[str, Any]] = []
    for cand in ranked:
        group = by_candidate.get(str(cand.get("candidate_id")), [])
        for seed in group[: int(cont.get("max_seeds_per_candidate", 1))]:
            item = {**cand, **{f"seed_{k}": v for k, v in seed.items()}}
            item["seed_id"] = seed["seed_id"]
            item["seed_vec"] = np.array([finite_float(seed["x0"]), finite_float(seed["y0"]), finite_float(seed["z0"])], dtype=float)
            items.append(item)
    return items


def psi_eta_value(sigma: float, p: Dict[str, Any], eta: float, smooth_width: float) -> float:
    eta = float(np.clip(eta, 0.0, 1.0))
    if chua.chua_model(p) != "piecewise":
        return float(chua.psi_sigma(sigma, p))
    gain = float(chua.chua_gain_A(p))
    smooth = math.tanh(float(sigma) / max(float(smooth_width), 1.0e-9))
    exact = float(np.clip(float(sigma), -1.0, 1.0))
    return gain * ((1.0 - eta) * smooth + eta * exact)


def rhs_eta(x: np.ndarray, p: Dict[str, Any], eta: float, smooth_width: float) -> np.ndarray:
    P, b, r = chua.chua_matrices(p)
    sigma = float(np.asarray(r, dtype=float) @ np.asarray(x, dtype=float))
    return P @ np.asarray(x, dtype=float) + b * psi_eta_value(sigma, p, eta, smooth_width)


def memory_points(memory_length: float, h: float) -> int:
    return max(1, int(math.ceil(float(memory_length) / float(h)))) + 1


def tail_vectors(traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(traj, dtype=float)
    states = X[:, 1:4] if X.ndim == 2 and X.shape[1] >= 4 else np.empty((0, 3), dtype=float)
    if states.shape[0] == 0:
        nan = np.array([float("nan")] * 3, dtype=float)
        return nan, nan
    start = max(0, int(0.8 * states.shape[0]))
    tail = states[start:, :]
    return np.mean(tail, axis=0), np.var(tail, axis=0)


def classify_traj(traj: np.ndarray, p: Dict[str, Any], eqs: Dict[str, np.ndarray], h: float, cfg: Dict[str, Any]) -> Dict[str, Any]:
    X = np.asarray(traj, dtype=float)
    states = X[:, 1:4] if X.ndim == 2 and X.shape[1] >= 4 else np.empty((0, 3), dtype=float)
    if states.shape[0] == 0 or not np.all(np.isfinite(states)):
        return {"bounded": False, "diverged": False, "equilibrium_hit": False, "final_class": "numerical_failure"}
    norms = np.linalg.norm(states, axis=1)
    div_thr = float(cfg.get("continuation", {}).get("divergence_norm", 120.0))
    eq_tol = float(cfg.get("continuation", {}).get("equilibrium_tol", 0.001))
    ranges = trajectory_ranges(X)
    fft = fft_peak_and_entropy(X, h=h)
    diverged = bool(float(np.max(norms)) > div_thr)
    final = states[-1]
    tail_mean, _tail_var = tail_vectors(X)
    equilibrium_hit = False
    if not diverged and eqs:
        for eq in eqs.values():
            eq_arr = np.asarray(eq, dtype=float)
            if np.linalg.norm(final - eq_arr) <= eq_tol or np.linalg.norm(tail_mean - eq_arr) <= 2.0 * eq_tol:
                equilibrium_hit = True
                break
    nontrivial = max(float(ranges["range_x"]), float(ranges["range_y"]), float(ranges["range_z"])) >= float(cfg.get("continuation", {}).get("nontrivial_range", 0.01))
    if diverged:
        final_class = "divergent"
    elif equilibrium_hit:
        final_class = "equilibrium_hit"
    elif nontrivial:
        final_class = "bounded_nontrivial"
    else:
        final_class = "bounded_small"
    return {
        "bounded": bool(not diverged),
        "diverged": diverged,
        "equilibrium_hit": bool(equilibrium_hit),
        "final_class": final_class,
        "final_x": float(final[0]),
        "final_y": float(final[1]),
        "final_z": float(final[2]),
        **ranges,
        "fft_peak": fft["fft_peak"],
        "psd_entropy": fft["psd_entropy"],
    }


def route_sigma0(route_id: str, sigma0_seed: float, eta_index: int, eta_count: int, p: Dict[str, Any], eqs: Dict[str, np.ndarray]) -> float:
    if route_id == "C1":
        return float(sigma0_seed)
    tau = float(eta_index / max(eta_count - 1, 1))
    if route_id == "C2":
        return (1.0 - tau) * float(sigma0_seed)
    if route_id == "C3" and eqs:
        _P, _b, r = chua.chua_matrices(p)
        sigmas = [float(np.asarray(r, dtype=float) @ np.asarray(eq, dtype=float)) for eq in eqs.values()]
        target = min(sigmas, key=lambda s: abs(s - float(sigma0_seed)))
        return (1.0 - tau) * float(sigma0_seed) + tau * float(target)
    return float(sigma0_seed)


def continuation_key(row: Dict[str, Any]) -> Tuple[str, str, str, int]:
    return (str(row.get("candidate_id")), str(row.get("seed_id")), str(row.get("route_id")), int(finite_float(row.get("step_index"), -1)))


def run_one_continuation_item(
    item: Dict[str, Any],
    cfg: Dict[str, Any],
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
    outdir: Path,
    resume_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any] | None, float | None]:
    cont = cfg["continuation"]
    q = float(cfg["q"])
    h = float(cont.get("h", 0.01))
    Lm = float(cont.get("memory_length", 20.0))
    t_block = float(cont.get("t_block", 200.0))
    n_blocks = int(cont.get("n_blocks", 8))
    smooth_width = float(cont.get("smooth_width", 0.2))
    routes = [str(v) for v in cont.get("routes", ["C1"])]
    completed = {continuation_key(r) for r in resume_rows}
    rows: List[Dict[str, Any]] = []
    path_rows: List[Dict[str, Any]] = []
    first_elapsed: float | None = None
    survivor: Dict[str, Any] | None = None

    for route_id in routes:
        eta_values = np.linspace(float(cont.get("eta_start", 0.0)), float(cont.get("eta_target", 1.0)), n_blocks + 1)
        current = np.asarray(item["seed_vec"], dtype=float).copy()
        history: FractionalHistory | None = None
        last_cls: Dict[str, Any] = {}
        route_failed = False
        final_traj = None
        for step_index, eta in enumerate(eta_values):
            key = (str(item["candidate_id"]), str(item["seed_id"]), route_id, step_index)
            if key in completed:
                continue
            sigma0_current = route_sigma0(route_id, finite_float(item.get("sigma0")), step_index, len(eta_values), p, eqs)
            rhs = lambda x, et=float(eta): rhs_eta(x, p, et, smooth_width)
            hist_arr = history.as_efork_history() if history is not None else None
            t0 = time.time()
            try:
                full = chua.efork3_integrate(
                    rhs,
                    current,
                    qord=q,
                    h=h,
                    Lm=Lm,
                    t_f=t_block,
                    history=hist_arr,
                    return_full_history=hist_arr is not None,
                )
                if hist_arr is not None:
                    traj = full[full[:, 0] >= -1.0e-12].copy()
                    traj[:, 0] -= traj[0, 0]
                else:
                    traj = full
                final_traj = traj
                cls = classify_traj(traj, p, eqs, h, cfg)
                current = traj[-1, 1:4].astype(float)
                history = FractionalHistory.from_trajectory(full if hist_arr is not None else traj, q=q, h=h, memory_length=Lm, rhs=rhs)
                memory_carried = bool(step_index > 0 and history.memory_points > 1)
                notes = "FractionalHistory window carried." if memory_carried else "initial step; no previous memory window."
                last_cls = cls
            except Exception as exc:
                cls = {
                    "bounded": False,
                    "diverged": False,
                    "equilibrium_hit": False,
                    "final_class": "numerical_failure",
                    "final_x": "",
                    "final_y": "",
                    "final_z": "",
                    "range_x": "",
                    "range_y": "",
                    "range_z": "",
                    "fft_peak": "",
                    "psd_entropy": "",
                }
                memory_carried = False
                notes = str(exc)
                route_failed = True
            elapsed = time.time() - t0
            if first_elapsed is None:
                first_elapsed = elapsed
            row = {
                "candidate_id": item["candidate_id"],
                "seed_id": item["seed_id"],
                "route_id": route_id,
                "q": q,
                "eta": float(eta),
                "sigma0_current": sigma0_current,
                "step_index": step_index,
                "h": h,
                "memory_length": Lm,
                "memory_points": 0 if history is None else history.memory_points,
                "memory_carried": memory_carried,
                "bounded": cls["bounded"],
                "diverged": cls["diverged"],
                "equilibrium_hit": cls["equilibrium_hit"],
                "final_class": cls["final_class"],
                "final_x": cls.get("final_x", ""),
                "final_y": cls.get("final_y", ""),
                "final_z": cls.get("final_z", ""),
                "range_x": cls.get("range_x", ""),
                "range_y": cls.get("range_y", ""),
                "range_z": cls.get("range_z", ""),
                "fft_peak": cls.get("fft_peak", ""),
                "psd_entropy": cls.get("psd_entropy", ""),
                "notes": f"{notes}; elapsed_sec={elapsed:.3f}",
            }
            rows.append(row)
            append_csv(outdir / "continuation_summary.csv", row, CONT_FIELDS)
            path_row = {
                "candidate_id": item["candidate_id"],
                "seed_id": item["seed_id"],
                "route_id": route_id,
                "step_index": step_index,
                "q": q,
                "eta": float(eta),
                "sigma0_current": sigma0_current,
                "x": row["final_x"],
                "y": row["final_y"],
                "z": row["final_z"],
            }
            path_rows.append(path_row)
            append_csv(outdir / "continuation_paths.csv", path_row, PATH_FIELDS)
            if route_failed or cls["diverged"] or cls["equilibrium_hit"]:
                route_failed = True
                break
        if not route_failed and last_cls.get("final_class") == "bounded_nontrivial" and rows:
            last_row = rows[-1]
            reliability = "high" if truthy(last_row.get("memory_carried")) else "low"
            hiddenness_status = "candidate_hidden_like"
            survivor = {
                "candidate_id": item["candidate_id"],
                "seed_id": item["seed_id"],
                "route_id": route_id,
                "q": q,
                "final_class": last_cls.get("final_class", ""),
                "memory_carried": bool(last_row.get("memory_carried")),
                "continuation_reliability": reliability,
                "final_x": last_cls.get("final_x", ""),
                "final_y": last_cls.get("final_y", ""),
                "final_z": last_cls.get("final_z", ""),
                "range_x": last_cls.get("range_x", ""),
                "range_y": last_cls.get("range_y", ""),
                "range_z": last_cls.get("range_z", ""),
                "fft_peak": last_cls.get("fft_peak", ""),
                "psd_entropy": last_cls.get("psd_entropy", ""),
                "hiddenness_status": hiddenness_status,
                "notes": "Continuation survivor; not hidden_verified. Reliability low if memory was not carried.",
            }
            append_csv(outdir / "continuation_survivors.csv", survivor, SURVIVOR_FIELDS)
            if final_traj is not None:
                save_reduced_trajectory(outdir / "trajectories" / f"{item['candidate_id']}_{item['seed_id']}_{route_id}_continuation.csv", final_traj)
    return rows, path_rows, survivor, first_elapsed


def save_reduced_trajectory(path: str | Path, traj: np.ndarray, max_points: int = 5000) -> str:
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


def count_planned_simulations(cfg: Dict[str, Any], n_seed_items: int, n_survivors_guess: int | None = None) -> int:
    cont = cfg.get("continuation", {})
    early = cfg.get("early_equilibrium_filter", {})
    robust = cfg.get("robustness", {})
    routes = len(cont.get("routes", ["C1"]))
    cont_sims = n_seed_items * routes * (int(cont.get("n_blocks", 8)) + 1)
    survivors = n_seed_items if n_survivors_guess is None else int(n_survivors_guess)
    early_dirs = 0
    if bool(early.get("enabled", False)):
        early_dirs = survivors * (8 + 1 + 3 * 8)
    robust_sims = 0
    if bool(robust.get("enabled", False)):
        robust_sims = survivors * len(robust.get("cases", {"R0": {}, "R1": {}, "R2": {}}))
    return int(cont_sims + early_dirs + robust_sims)


def enforce_cost_guard(cfg: Dict[str, Any], planned: int, force: bool) -> None:
    guard = cfg.get("cost_guard", {})
    limit = int(guard.get("max_simulations_without_force", 2000))
    if planned > limit and not force:
        raise RuntimeError(f"Cost guard: planned simulations={planned} exceeds {limit}. Use --force to allow this run.")


def component_stats(traj: np.ndarray, h: float) -> Dict[str, Any]:
    ranges = trajectory_ranges(traj)
    fft = fft_peak_and_entropy(traj, h=h)
    mean, var = tail_vectors(traj)
    return {
        **ranges,
        "mean_tail": ";".join(f"{v:.16g}" for v in mean),
        "var_tail": ";".join(f"{v:.16g}" for v in var),
        "fft_peak": fft["fft_peak"],
        "psd_entropy": fft["psd_entropy"],
    }


def section_points(traj: np.ndarray, p: Dict[str, Any], t_burn: float, max_points: int = 240) -> np.ndarray:
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
                lam = (0.0 - xp) / ((x - xp) + 1.0e-300)
                y = X[k - 1, 2] + lam * (X[k, 2] - X[k - 1, 2])
                z = X[k - 1, 3] + lam * (X[k, 3] - X[k - 1, 3])
                pts.append((float(y), float(z)))
                if len(pts) >= max_points:
                    break
    return np.asarray(pts, dtype=float)


def section_hit_fraction(section: np.ndarray, reference: np.ndarray, tol: float = 0.12) -> Tuple[int, int, float]:
    if section.size == 0 or reference.size == 0:
        return int(section.shape[0] if section.ndim == 2 else 0), 0, 0.0
    hits = 0
    for point in section:
        if float(np.min(np.linalg.norm(reference - point.reshape(1, 2), axis=1))) <= tol:
            hits += 1
    total = int(section.shape[0])
    return total, hits, float(hits / max(total, 1))


def original_integrate(x0: np.ndarray, q: float, p: Dict[str, Any], h: float, memory_length: float, t_final: float) -> np.ndarray:
    return chua.efork3_integrate(lambda x: chua.rhs_original(x, p), x0, qord=q, h=h, Lm=memory_length, t_f=t_final)


def classify_target(
    traj: np.ndarray,
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
    reference: np.ndarray,
    h: float,
    t_final: float,
    divergence_norm: float,
    eq_tol: float,
) -> Dict[str, Any]:
    X = np.asarray(traj, dtype=float)
    states = X[:, 1:4] if X.ndim == 2 and X.shape[1] >= 4 else np.empty((0, 3), dtype=float)
    if states.shape[0] == 0 or not np.all(np.isfinite(states)):
        return {"final_class": "numerical_failure", "target_hit": False}
    stats = component_stats(X, h)
    final = states[-1]
    max_norm = float(np.max(np.linalg.norm(states, axis=1)))
    if max_norm > divergence_norm:
        final_class = "divergent"
        target_hit = False
    else:
        final_class = "other_bounded_nontrivial"
        target_hit = False
        for eq in eqs.values():
            if np.linalg.norm(final - np.asarray(eq, dtype=float)) <= eq_tol:
                final_class = "equilibrium_convergence"
                break
        if final_class != "equilibrium_convergence":
            sec = section_points(X, p, 0.5 * t_final, max_points=120)
            total, hits, frac = section_hit_fraction(sec, reference)
            if total >= 20 and frac >= 0.70:
                final_class = "target_attractor"
                target_hit = True
            elif total < 20:
                final_class = "ambiguous_long_transient"
    return {**stats, "final_x": float(final[0]), "final_y": float(final[1]), "final_z": float(final[2]), "final_class": final_class, "target_hit": bool(target_hit)}


def normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n < 1.0e-300:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return arr / n


def eig_directions(p: Dict[str, Any], eq: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    try:
        vals, vecs = np.linalg.eig(local_jacobian(p, eq))
        idx = int(np.argmax(np.real(vals)))
        vec = np.real(vecs[:, idx])
        return [("eig_dominant_p", normalize(vec)), ("eig_dominant_m", -normalize(vec))]
    except Exception:
        return [("eig_dominant_p", np.array([1.0, 0.0, 0.0])), ("eig_dominant_m", np.array([-1.0, 0.0, 0.0]))]


def early_directions(eq_id: str, p: Dict[str, Any], eq: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    ex, ey, ez = np.eye(3)
    if eq_id == "E+":
        dirs = [
            ("-e_x", -ex),
            ("-e_y", -ey),
            ("-e_z", -ez),
            ("diag_-1_-1_-1", normalize(np.array([-1.0, -1.0, -1.0]))),
            ("diag_-1_-1_+1", normalize(np.array([-1.0, -1.0, 1.0]))),
            ("diag_-1_+1_-1", normalize(np.array([-1.0, 1.0, -1.0]))),
            ("diag_+1_-1_-1", normalize(np.array([1.0, -1.0, -1.0]))),
        ]
        eig = eig_directions(p, eq)
        dirs.append(("eig_dominant_0_m", eig[-1][1]))
        return dirs
    dirs = [("+e_x", ex), ("-e_x", -ex), ("+e_y", ey), ("-e_y", -ey), ("+e_z", ez), ("-e_z", -ez)]
    dirs.extend(eig_directions(p, eq))
    return dirs


def run_early_filter(
    survivors: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    p: Dict[str, Any],
    eqs: Dict[str, np.ndarray],
    outdir: Path,
    resume: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    early = cfg.get("early_equilibrium_filter", {})
    if not bool(early.get("enabled", False)):
        return [], []
    raw_path = outdir / "early_equilibrium_filter_raw.csv"
    if raw_path.exists() and not resume:
        raw_path.unlink()
    raw_rows = [dict(r) for r in read_csv_rows(raw_path)] if resume else []
    done = {(r.get("candidate_id"), r.get("equilibrium_id"), r.get("direction_label")) for r in raw_rows}
    q = float(cfg["q"])
    h = float(early.get("h", 0.01))
    Lm = float(early.get("memory_length", 40.0))
    t_final = float(early.get("t_final", 1500.0))
    rho = float(early.get("rho", 1.0e-5))
    for surv in survivors:
        start = np.array([finite_float(surv.get("final_x")), finite_float(surv.get("final_y")), finite_float(surv.get("final_z"))], dtype=float)
        if not np.all(np.isfinite(start)):
            continue
        ref_traj = original_integrate(start, q, p, h, Lm, t_final)
        reference = section_points(ref_traj, p, 0.5 * t_final, max_points=240)
        for eq_id in ["E+", "E0", "E-"]:
            if eq_id not in eqs:
                continue
            eq = np.asarray(eqs[eq_id], dtype=float)
            for label, direction in early_directions(eq_id, p, eq):
                key = (surv["candidate_id"], eq_id, label)
                if key in done:
                    continue
                x0 = eq + rho * normalize(direction)
                traj = original_integrate(x0, q, p, h, Lm, t_final)
                cls = classify_target(traj, p, eqs, reference, h, t_final, float(cfg.get("continuation", {}).get("divergence_norm", 120.0)), float(cfg.get("continuation", {}).get("equilibrium_tol", 0.001)))
                status = "passed_early_equilibrium_filter"
                if bool(cls.get("target_hit")) and eq_id == "E+":
                    status = "not_supported_by_Eplus_neighborhood_test"
                elif bool(cls.get("target_hit")):
                    status = "not_supported_by_equilibrium_neighborhood_test"
                row = {
                    "candidate_id": surv["candidate_id"],
                    "seed_id": surv.get("seed_id", ""),
                    "equilibrium_id": eq_id,
                    "direction_label": label,
                    "rho": rho,
                    "q": q,
                    "h": h,
                    "memory_length": Lm,
                    "memory_points": memory_points(Lm, h),
                    "t_final": t_final,
                    "x0": float(x0[0]),
                    "y0": float(x0[1]),
                    "z0": float(x0[2]),
                    "hiddenness_status": status,
                    "notes": "TARGET from E+ blocks hiddenness immediately; no hidden_verified.",
                    **cls,
                }
                raw_rows.append(row)
                append_csv(raw_path, row, EARLY_RAW_FIELDS)
    summary = summarize_early(raw_rows)
    write_csv(outdir / "early_equilibrium_filter_summary.csv", summary, EARLY_SUMMARY_FIELDS)
    return raw_rows, summary


def summarize_early(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("candidate_id")), []).append(row)
    out: List[Dict[str, Any]] = []
    for cid, group in grouped.items():
        eplus = sum(1 for r in group if r.get("equilibrium_id") == "E+" and truthy(r.get("target_hit")))
        e0 = sum(1 for r in group if r.get("equilibrium_id") == "E0" and truthy(r.get("target_hit")))
        eminus = sum(1 for r in group if r.get("equilibrium_id") == "E-" and truthy(r.get("target_hit")))
        if eplus > 0:
            status = "not_supported_by_Eplus_neighborhood_test"
        elif e0 + eminus > 0:
            status = "not_supported_by_equilibrium_neighborhood_test"
        else:
            status = "passed_early_equilibrium_filter"
        out.append({"candidate_id": cid, "n_Eplus_TARGET": eplus, "n_E0_TARGET": e0, "n_Eminus_TARGET": eminus, "hiddenness_status": status, "notes": "No hidden_verified."})
    return out


def run_robustness(
    survivors: Sequence[Dict[str, Any]],
    early_summary: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    p: Dict[str, Any],
    outdir: Path,
    resume: bool,
) -> List[Dict[str, Any]]:
    robust = cfg.get("robustness", {})
    if not bool(robust.get("enabled", False)):
        return []
    allowed = {r["candidate_id"] for r in early_summary if r.get("hiddenness_status") == "passed_early_equilibrium_filter"}
    if not early_summary:
        allowed = {s["candidate_id"] for s in survivors}
    raw_path = outdir / "robustness_survivors.csv"
    if raw_path.exists() and not resume:
        raw_path.unlink()
    rows = [dict(r) for r in read_csv_rows(raw_path)] if resume else []
    done = {(r.get("candidate_id"), r.get("case_id")) for r in rows}
    q = float(cfg["q"])
    for surv in survivors:
        cid = str(surv["candidate_id"])
        if cid not in allowed:
            continue
        start = np.array([finite_float(surv.get("final_x")), finite_float(surv.get("final_y")), finite_float(surv.get("final_z"))], dtype=float)
        case_rows: List[Dict[str, Any]] = []
        for case_id, params in robust.get("cases", {}).items():
            if (cid, case_id) in done:
                continue
            h = float(params["h"])
            Lm = float(params["memory_length"])
            t_final = float(params["t_final"])
            traj = original_integrate(start, q, p, h, Lm, t_final)
            cls = classify_traj(traj, p, {}, h, cfg)
            stats = component_stats(traj, h)
            bounded_nontrivial = bool(cls.get("bounded") and cls.get("final_class") == "bounded_nontrivial")
            row = {
                "candidate_id": cid,
                "case_id": case_id,
                "q": q,
                "h": h,
                "memory_length": Lm,
                "memory_points": memory_points(Lm, h),
                "t_final": t_final,
                "bounded": bool(cls.get("bounded")),
                "final_class": cls.get("final_class", ""),
                **stats,
                "robust_attractor": False,
                "notes": "robust_attractor set after all requested cases are compared; no hidden_verified.",
            }
            case_rows.append(row)
            rows.append(row)
            append_csv(raw_path, row, ROBUST_FIELDS)
        if case_rows:
            update_robust_flags(raw_path)
    return read_csv_rows(raw_path)


def parse_vec(text: Any) -> np.ndarray:
    if isinstance(text, (list, tuple, np.ndarray)):
        return np.asarray(text, dtype=float)
    parts = [p for p in str(text).replace(",", ";").split(";") if p != ""]
    try:
        return np.asarray([float(p) for p in parts], dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def update_robust_flags(path: Path) -> None:
    rows = read_csv_rows(path)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("candidate_id")), []).append(row)
    for cid, group in grouped.items():
        nontrivial = [r for r in group if r.get("final_class") == "bounded_nontrivial" and truthy(r.get("bounded"))]
        robust = False
        if len(nontrivial) >= 3:
            ranges = np.vstack([np.array([finite_float(r.get("range_x")), finite_float(r.get("range_y")), finite_float(r.get("range_z"))]) for r in nontrivial])
            if np.all(np.isfinite(ranges)):
                scale = np.maximum(np.max(np.abs(ranges), axis=0), 1.0e-9)
                rel_spread = np.max((np.max(ranges, axis=0) - np.min(ranges, axis=0)) / scale)
                robust = bool(rel_spread <= 0.40)
        for row in group:
            row["robust_attractor"] = robust
            if robust:
                row["notes"] = "R0/R1/R2 bounded_nontrivial with qualitatively consistent ranges; not hidden_verified."
    write_csv(path, rows, ROBUST_FIELDS)


def load_machado_record() -> Dict[str, Any] | None:
    path = ROOT / "runs_machado_sweep_fast" / "chua_piecewise" / "machado_sweep" / "machado_sweep_summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data.get("records", []):
        if row.get("slug") == MACHADO_CANDIDATE_ID or row.get("candidate_id") == MACHADO_CANDIDATE_ID:
            return {**row, "q": float(data.get("frac_order", TARGET_Q))}
    best = data.get("best_candidate", {})
    if best.get("slug") == MACHADO_CANDIDATE_ID or best.get("candidate_id") == MACHADO_CANDIDATE_ID:
        return {**best, "q": float(data.get("frac_order", TARGET_Q))}
    return None


def compare_with_machado(survivors: Sequence[Dict[str, Any]], outdir: Path) -> List[Dict[str, Any]]:
    mach = load_machado_record()
    rows: List[Dict[str, Any]] = []
    for surv in survivors:
        if mach is None:
            rows.append({"candidate_id": surv.get("candidate_id", ""), "machado_candidate_id": MACHADO_CANDIDATE_ID, "q_lure": surv.get("q", ""), "q_machado": "", "likely_same_attractor_as_machado": False, "distinct_candidate": False, "notes": "Machado summary not found."})
            continue
        lure_final = np.array([finite_float(surv.get("final_x")), finite_float(surv.get("final_y")), finite_float(surv.get("final_z"))], dtype=float)
        mach_final = np.asarray(mach.get("final_state_eps1", [float("nan")] * 3), dtype=float)
        lure_range = np.array([finite_float(surv.get("range_x")), finite_float(surv.get("range_y")), finite_float(surv.get("range_z"))], dtype=float)
        mach_range = np.array([finite_float(mach.get("range_x")), finite_float(mach.get("range_y")), finite_float(mach.get("range_z"))], dtype=float)
        final_dist = float(np.linalg.norm(lure_final - mach_final)) if np.all(np.isfinite(lure_final)) and np.all(np.isfinite(mach_final)) else float("nan")
        if np.all(np.isfinite(lure_range)) and np.all(np.isfinite(mach_range)):
            denom = max(float(np.linalg.norm(mach_range)), 1.0e-9)
            range_rel = float(np.linalg.norm(lure_range - mach_range) / denom)
        else:
            range_rel = float("nan")
        same = bool(math.isfinite(final_dist) and math.isfinite(range_rel) and final_dist < 0.5 and range_rel < 0.15)
        distinct = bool(math.isfinite(range_rel) and range_rel > 0.30)
        rows.append(
            {
                "candidate_id": surv.get("candidate_id", ""),
                "machado_candidate_id": MACHADO_CANDIDATE_ID,
                "q_lure": surv.get("q", ""),
                "q_machado": mach.get("q", ""),
                "lure_final_x": lure_final[0],
                "lure_final_y": lure_final[1],
                "lure_final_z": lure_final[2],
                "machado_final_x": mach_final[0],
                "machado_final_y": mach_final[1],
                "machado_final_z": mach_final[2],
                "lure_range_x": lure_range[0],
                "lure_range_y": lure_range[1],
                "lure_range_z": lure_range[2],
                "machado_range_x": mach_range[0],
                "machado_range_y": mach_range[1],
                "machado_range_z": mach_range[2],
                "final_state_distance": final_dist,
                "range_relative_distance": range_rel,
                "likely_same_attractor_as_machado": same,
                "distinct_candidate": distinct,
                "notes": "Comparison is operational only; Machado line is preserved.",
            }
        )
    write_csv(outdir / "lure_biased_vs_machado_comparison.csv", rows, COMPARE_FIELDS)
    return rows


def update_final_report(outdir: Path, cfg: Dict[str, Any], files_written: Sequence[str]) -> None:
    previous_files: List[str] = []
    summary_path = outdir / "lure_biased_multiparam_summary.json"
    if summary_path.exists():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            previous_files = [str(v) for v in previous.get("files_written", [])]
        except Exception:
            previous_files = []
    combined_files = list(dict.fromkeys(previous_files + [str(v) for v in files_written]))
    candidates = read_csv_rows(outdir / "biased_lure_candidates.csv")
    run_metadata: Dict[str, Any] = {}
    metadata_path = outdir / "search_run_metadata.json"
    if metadata_path.exists():
        try:
            run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            run_metadata = {}
    q_audit: Dict[str, Any] = {}
    q_audit_path = ROOT / "outputs" / "q_audit" / "q_audit_summary.json"
    if q_audit_path.exists():
        try:
            q_audit = json.loads(q_audit_path.read_text(encoding="utf-8"))
        except Exception:
            q_audit = {}
    cont = read_csv_rows(outdir / "continuation_survivors.csv")
    early = read_csv_rows(outdir / "early_equilibrium_filter_summary.csv")
    robust = read_csv_rows(outdir / "robustness_survivors.csv")
    compare = read_csv_rows(outdir / "lure_biased_vs_machado_comparison.csv")
    robust_survivors = sorted({r.get("candidate_id", "") for r in robust if truthy(r.get("robust_attractor")) and r.get("candidate_id")})
    passed_eplus = sorted({r.get("candidate_id", "") for r in early if r.get("hiddenness_status") == "passed_early_equilibrium_filter"})
    lines = [
        "# Biased Lure multiparameter exploration q=0.9998",
        "",
        "No `hidden_verified` status is declared.",
        "",
        "The harmonic calculation is a first-harmonic seed generator, not proof of exact Caputo cycles.",
        "",
        "## q consistency",
        "",
        f"- q_global: `{float(cfg['q']):.5f}`",
        "- q_consistency_status: `ok_q_0p9998`",
        "",
        "## q audit summary",
        "",
    ]
    if q_audit:
        for key, value in q_audit.get("classification_counts", {}).items():
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- q audit has not been run yet.")
    lines.extend(
        [
            "",
            "## Execution scope",
            "",
            f"- configured_S0_n_samples: `{run_metadata.get('configured_n_samples', '')}`",
            f"- executed_S0_lhs_samples: `{run_metadata.get('executed_lhs_samples', '')}`",
            f"- source_hint_samples: `{run_metadata.get('source_hint_samples', '')}`",
            f"- local_refine_top: `{run_metadata.get('local_refine_top', '')}`",
            f"- execution_scope: `{run_metadata.get('execution_scope', '')}`",
        ]
    )
    best_residual = sorted(candidates, key=lambda r: finite_float(r.get("residual_abs")))[:10]
    best_rho = sorted(candidates, key=lambda r: finite_float(r.get("rho_H")))[:10]
    lines.extend(
        [
            "",
            "## Best candidates by residual",
            "",
        ]
    )
    if best_residual:
        lines.append("| candidate_id | A | sigma0 | omega | residual_abs | rho_H | status |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in best_residual:
            lines.append(
                f"| `{row.get('candidate_id')}` | {finite_float(row.get('A')):.6g} | {finite_float(row.get('sigma0')):.6g} | "
                f"{finite_float(row.get('omega')):.6g} | {finite_float(row.get('residual_abs')):.6g} | {finite_float(row.get('rho_H')):.6g} | "
                f"{row.get('candidate_status', 'candidate_hidden_like')} |"
            )
    else:
        lines.append("No candidates passed the residual/rho_H filters.")
    lines.extend(["", "## Best candidates by rho_H", ""])
    if best_rho:
        lines.append("| candidate_id | residual_abs | rho_H | rhoH_class |")
        lines.append("|---|---:|---:|---|")
        for row in best_rho:
            lines.append(f"| `{row.get('candidate_id')}` | {finite_float(row.get('residual_abs')):.6g} | {finite_float(row.get('rho_H')):.6g} | {row.get('rhoH_class', '')} |")
    else:
        lines.append("No candidates passed the residual/rho_H filters.")
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            f"- filtered candidates: {len(candidates)}",
            f"- continuation survivors: {len(cont)}",
            f"- passed early E+ filter: {len(passed_eplus)}",
            f"- robust survivors: {len(robust_survivors)}",
            "",
            "## Continuation survivors",
            "",
        ]
    )
    if cont:
        for row in cont[:20]:
            lines.append(f"- `{row.get('candidate_id')}` seed `{row.get('seed_id')}` route `{row.get('route_id')}` class `{row.get('final_class')}` reliability `{row.get('continuation_reliability')}`")
    else:
        lines.append("- none or not executed")
    lines.extend(["", "## Candidates discarded by E+", ""])
    eplus_discards = [r for r in early if r.get("hiddenness_status") == "not_supported_by_Eplus_neighborhood_test"]
    if eplus_discards:
        for row in eplus_discards:
            lines.append(f"- `{row.get('candidate_id')}`: {row.get('n_Eplus_TARGET')} TARGET contacts from E+")
    else:
        lines.append("- none recorded")
    lines.extend(["", "## Robust survivors", ""])
    if robust_survivors:
        for cid in robust_survivors:
            lines.append(f"- `{cid}`: robust_attractor=true under executed R0/R1/R2 checks, not hidden_verified")
    else:
        lines.append("- none recorded")
    lines.extend(["", "## Machado comparison", ""])
    if compare:
        for row in compare[:20]:
            lines.append(f"- `{row.get('candidate_id')}` vs `{MACHADO_CANDIDATE_ID}`: same={row.get('likely_same_attractor_as_machado')} distinct={row.get('distinct_candidate')}")
    else:
        lines.append("- no survivor comparison recorded")
    lines.extend(["", "## Warnings", "", "- Do not declare `hidden_verified` from these runs.", "- Any TARGET contact from E+ at rho=1e-5 blocks hiddenness for that Lure candidate.", "- The Machado candidate line is not modified by this exploration."])
    (outdir / "lure_biased_multiparam_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "q_global": float(cfg["q"]),
        "q_consistency_status": "ok_q_0p9998",
        "n_candidates_raw": len(read_csv_rows(outdir / "biased_lure_all_evaluations.csv")),
        "n_candidates_filtered": len(candidates),
        "n_seeds": len(read_csv_rows(outdir / "biased_lure_seed_bank.csv")),
        "n_continuation_survivors": len(cont),
        "n_passed_Eplus_filter": len(passed_eplus),
        "n_robust_survivors": len(robust_survivors),
        "best_candidate_id": candidates[0].get("candidate_id", "") if candidates else "",
        "best_candidate_status": candidates[0].get("candidate_status", "") if candidates else "",
        "hidden_verified": False,
        "run_metadata": run_metadata,
        "files_written": combined_files,
    }
    (outdir / "lure_biased_multiparam_summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")


def reset_stage_files(outdir: Path, resume: bool) -> None:
    if resume:
        return
    for name in ["continuation_summary.csv", "continuation_paths.csv", "continuation_survivors.csv"]:
        path = outdir / name
        if path.exists():
            path.unlink()


def run_continuation_pipeline(
    config_path: str | Path,
    *,
    resume: bool = False,
    force: bool = False,
    execute: bool = True,
    execute_early: bool | None = None,
    execute_robustness: bool | None = None,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    q = require_q09998(cfg, context=str(config_path))
    outdir = outdir_from_config(cfg)
    p = chua_ic_params(cfg)
    eqs = solve_equilibria(p)
    candidates, seeds = load_candidates_and_seeds(outdir, q)
    items = selected_seed_items(cfg, candidates, seeds)
    planned = count_planned_simulations(cfg, len(items))
    enforce_cost_guard(cfg, planned, force)
    if not execute:
        files_written = [rel(outdir / "continuation_cost_plan.json")]
        (outdir / "continuation_cost_plan.json").write_text(
            json.dumps(
                {
                    "q_global": q,
                    "q_consistency_status": "ok_q_0p9998",
                    "n_seed_items": len(items),
                    "planned_simulations": planned,
                    "continuation_enabled_in_config": bool(cfg.get("continuation", {}).get("enabled", False)),
                    "notes": "No simulations executed. Use --execute-continuation, or run search with --run-continuation, to start long integrations.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        update_final_report(outdir, cfg, files_written)
        files_written.extend([rel(outdir / "lure_biased_multiparam_report.md"), rel(outdir / "lure_biased_multiparam_summary.json")])
        return {
            "files_written": files_written,
            "n_continuation_survivors": len(read_csv_rows(outdir / "continuation_survivors.csv")),
            "n_passed_Eplus_filter": len([r for r in read_csv_rows(outdir / "early_equilibrium_filter_summary.csv") if r.get("hiddenness_status") == "passed_early_equilibrium_filter"]),
            "n_robust_survivors": len({r.get("candidate_id") for r in read_csv_rows(outdir / "robustness_survivors.csv") if truthy(r.get("robust_attractor"))}),
        }
    reset_stage_files(outdir, resume)
    resume_rows = read_csv_rows(outdir / "continuation_summary.csv") if resume else []
    files_written = [rel(outdir / "continuation_summary.csv"), rel(outdir / "continuation_paths.csv"), rel(outdir / "continuation_survivors.csv")]
    survivors: List[Dict[str, Any]] = []
    first_elapsed: float | None = None
    for idx, item in enumerate(items):
        print(f"continuation item {idx + 1}/{len(items)} {item['candidate_id']} {item['seed_id']}", flush=True)
        rows, _paths, survivor, elapsed = run_one_continuation_item(item, cfg, p, eqs, outdir, resume_rows)
        if first_elapsed is None and elapsed is not None:
            first_elapsed = elapsed
            remaining = max(planned - 1, 0)
            estimated_total_sec = first_elapsed * max(planned, 1)
            max_hours = float(cfg.get("cost_guard", {}).get("max_estimated_hours_without_force", 12.0))
            (outdir / "cost_estimate_continuation.json").write_text(
                json.dumps(
                    {
                        "measured_first_simulation_sec": first_elapsed,
                        "planned_simulations": planned,
                        "remaining_simulations_after_measurement": remaining,
                        "estimated_total_sec": estimated_total_sec,
                        "estimated_total_hours": estimated_total_sec / 3600.0,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            files_written.append(rel(outdir / "cost_estimate_continuation.json"))
            if estimated_total_sec > max_hours * 3600.0 and not force:
                raise RuntimeError(f"Cost guard: measured estimate {estimated_total_sec / 3600.0:.3f} h exceeds {max_hours} h. Use --force.")
        if survivor is not None:
            survivors.append(survivor)
    if not (outdir / "continuation_summary.csv").exists():
        write_csv(outdir / "continuation_summary.csv", [], CONT_FIELDS)
    if not (outdir / "continuation_paths.csv").exists():
        write_csv(outdir / "continuation_paths.csv", [], PATH_FIELDS)
    if not (outdir / "continuation_survivors.csv").exists():
        write_csv(outdir / "continuation_survivors.csv", [], SURVIVOR_FIELDS)
    survivor_rows = read_csv_rows(outdir / "continuation_survivors.csv")
    do_early = bool(cfg.get("early_equilibrium_filter", {}).get("enabled", False)) if execute_early is None else bool(execute_early)
    do_robust = bool(cfg.get("robustness", {}).get("enabled", False)) if execute_robustness is None else bool(execute_robustness)
    early_summary: List[Dict[str, Any]] = []
    if do_early:
        _raw, early_summary = run_early_filter(survivor_rows, cfg, p, eqs, outdir, resume)
        files_written.extend([rel(outdir / "early_equilibrium_filter_raw.csv"), rel(outdir / "early_equilibrium_filter_summary.csv")])
    if do_robust:
        run_robustness(survivor_rows, early_summary, cfg, p, outdir, resume)
        files_written.append(rel(outdir / "robustness_survivors.csv"))
    compare_with_machado(survivor_rows, outdir)
    files_written.append(rel(outdir / "lure_biased_vs_machado_comparison.csv"))
    update_final_report(outdir, cfg, files_written)
    files_written.extend([rel(outdir / "lure_biased_multiparam_report.md"), rel(outdir / "lure_biased_multiparam_summary.json")])
    return {
        "files_written": files_written,
        "n_continuation_survivors": len(read_csv_rows(outdir / "continuation_survivors.csv")),
        "n_passed_Eplus_filter": len([r for r in read_csv_rows(outdir / "early_equilibrium_filter_summary.csv") if r.get("hiddenness_status") == "passed_early_equilibrium_filter"]),
        "n_robust_survivors": len({r.get("candidate_id") for r in read_csv_rows(outdir / "robustness_survivors.csv") if truthy(r.get("robust_attractor"))}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuation/filter/robustness for biased Lure q=0.9998 candidates.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--execute-continuation", action="store_true", help="Actually run continuation integrations. Without this, only a cost plan is written.")
    parser.add_argument("--execute-early-filter", action="store_true")
    parser.add_argument("--execute-robustness", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_continuation_pipeline(
        args.config,
        resume=args.resume,
        force=args.force,
        execute=args.execute_continuation,
        execute_early=args.execute_early_filter,
        execute_robustness=args.execute_robustness,
    )
    print(f"q_global={TARGET_Q:.5f}")
    print("q_consistency_status=ok_q_0p9998")
    print(f"n_continuation_survivors={result['n_continuation_survivors']}")
    print(f"n_passed_Eplus_filter={result['n_passed_Eplus_filter']}")
    print(f"n_robust_survivors={result['n_robust_survivors']}")
    print("files_written=" + ";".join(result["files_written"]))


if __name__ == "__main__":
    main()
