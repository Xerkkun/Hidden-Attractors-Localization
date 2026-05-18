from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

import chua_initial_cond as chua
from extended_search_utils import fft_peak_and_entropy, min_distance_to_points, trajectory_ranges, write_csv


CONT_FIELDS = [
    "candidate_id",
    "df_family",
    "eta",
    "q",
    "chi_name",
    "chi_value",
    "step_index",
    "h",
    "memory_length",
    "memory_points",
    "memory_carried",
    "initial_norm",
    "final_norm",
    "bounded",
    "diverged",
    "equilibrium_hit",
    "attractor_label",
    "max_lyapunov_estimate",
    "fft_dominant_frequency",
    "psd_entropy",
    "notes",
]


@dataclass
class FractionalHistory:
    t_window: np.ndarray
    X_window: np.ndarray
    q: float
    h: float
    memory_length: float

    @property
    def memory_points(self) -> int:
        return int(self.X_window.shape[0])

    def as_efork_history(self) -> np.ndarray:
        return np.column_stack([self.t_window, self.X_window])

    @classmethod
    def from_trajectory(cls, traj: np.ndarray, q: float, h: float, memory_length: float) -> "FractionalHistory":
        window = chua.extract_memory_window(traj, Lm=memory_length, h=h)
        return cls(
            t_window=window[:, 0].copy(),
            X_window=window[:, 1:4].copy(),
            q=float(q),
            h=float(h),
            memory_length=float(memory_length),
        )


def psi_eta_value(sigma: float, p: Dict[str, Any], eta: float, smooth_width: float) -> float:
    eta = float(np.clip(eta, 0.0, 1.0))
    if chua.chua_model(p) != "piecewise":
        return float(chua.psi_sigma(sigma, p))
    A = float(chua.chua_gain_A(p))
    smooth = np.tanh(float(sigma) / max(float(smooth_width), 1e-9))
    exact = float(np.clip(float(sigma), -1.0, 1.0))
    return A * ((1.0 - eta) * smooth + eta * exact)


def rhs_eta(x: np.ndarray, p: Dict[str, Any], eta: float, smooth_width: float) -> np.ndarray:
    P, b, r = chua.chua_matrices(p)
    sigma = float(r @ np.asarray(x, dtype=float))
    return P @ np.asarray(x, dtype=float) + b * psi_eta_value(sigma, p, eta, smooth_width)


def staged_values(start: float, target: float, steps: int) -> List[float]:
    steps = max(1, int(steps))
    return [float(v) for v in np.linspace(float(start), float(target), steps)]


def continuation_schedule(cfg: Dict[str, Any], candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    cont = cfg["continuation"]
    eta_values = staged_values(cont.get("eta_start", 0.0), cont.get("eta_target", 1.0), cont.get("eta_steps", 6))
    q_values = staged_values(cont.get("q_start", cfg["q"]), cont.get("q_target", cfg["q"]), cont.get("q_steps", 1))
    chi_name = "mu" if "machado" in str(candidate.get("df_family", "")) else "sigma0"
    chi_target = float(candidate.get("mu", candidate.get("sigma0", 0.0)) or 0.0)
    chi_values = staged_values(cont.get("chi_start", chi_target), chi_target, cont.get("chi_steps", 1))
    schedule: List[Dict[str, Any]] = []
    for qv in q_values:
        for chiv in chi_values:
            for etav in eta_values:
                schedule.append({"eta": etav, "q": qv, "chi_name": chi_name, "chi_value": chiv})
    return schedule


def classify_continuation_traj(
    traj: np.ndarray,
    equilibria: Sequence[np.ndarray],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    states = traj[:, 1:4]
    norms = np.linalg.norm(states, axis=1)
    final = states[-1]
    div_thr = float(cfg["continuation"].get("divergence_norm", 120.0))
    eps_eq = float(cfg["continuation"].get("equilibrium_tol", 0.05))
    diverged = bool((not np.all(np.isfinite(states))) or np.nanmax(norms) > div_thr)
    min_final_eq = min_distance_to_points(final, equilibria)
    equilibrium_hit = bool(np.isfinite(min_final_eq) and min_final_eq < eps_eq)
    ranges = trajectory_ranges(traj)
    nontrivial = max(ranges["range_x"], ranges["range_y"], ranges["range_z"]) > float(cfg["continuation"].get("nontrivial_range", 0.5))
    if diverged:
        label = "divergent"
    elif equilibrium_hit:
        label = "equilibrium_convergence"
    elif nontrivial:
        label = "candidate_bounded_nontrivial"
    else:
        label = "bounded_small"
    return {"diverged": diverged, "bounded": not diverged, "equilibrium_hit": equilibrium_hit, "attractor_label": label, **ranges}


def continue_candidate(
    candidate: Dict[str, Any],
    cfg: Dict[str, Any],
    p: Dict[str, Any],
    equilibria: Sequence[np.ndarray],
) -> Dict[str, Any]:
    cont = cfg["continuation"]
    h = float(cont.get("h", cfg.get("h", 0.01)))
    Lm = float(cont.get("memory_length", cfg.get("Lm", 8.0)))
    t_step = float(cont.get("t_step", 60.0))
    smooth_width = float(cont.get("smooth_width", 0.2))
    x0 = np.array([float(candidate["seed_x"]), float(candidate["seed_y"]), float(candidate["seed_z"])], dtype=float)
    history: FractionalHistory | None = None
    rows: List[Dict[str, Any]] = []
    path_rows: List[Dict[str, Any]] = []
    current = x0.copy()
    survived = True
    final_traj = None

    for step_index, step in enumerate(continuation_schedule(cfg, candidate)):
        q = float(step["q"])
        eta = float(step["eta"])
        hist_arr = history.as_efork_history() if history is not None else None
        traj_full = chua.efork3_integrate(
            lambda x, et=eta: rhs_eta(x, p, et, smooth_width),
            current,
            qord=q,
            h=h,
            Lm=Lm,
            t_f=t_step,
            history=hist_arr,
            return_full_history=history is not None,
        )
        if history is not None:
            traj = traj_full[traj_full[:, 0] >= -1e-12].copy()
            traj[:, 0] -= traj[0, 0]
        else:
            traj = traj_full
        final_traj = traj
        cls = classify_continuation_traj(traj, equilibria, cfg)
        fft = fft_peak_and_entropy(traj, h=h)
        initial_norm = float(np.linalg.norm(current))
        current = traj[-1, 1:4].astype(float)
        history = FractionalHistory.from_trajectory(traj_full if history is not None else traj, q=q, h=h, memory_length=Lm)
        memory_carried = bool(history.memory_points > 1)
        notes = []
        if step_index > 0 and memory_carried:
            notes.append("FractionalHistory window carried between continuation stages.")
        if cls["diverged"]:
            notes.append("step diverged; later continuation stages skipped.")
            survived = False
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "df_family": candidate["df_family"],
            "eta": eta,
            "q": q,
            "chi_name": step["chi_name"],
            "chi_value": step["chi_value"],
            "step_index": step_index,
            "h": h,
            "memory_length": Lm,
            "memory_points": history.memory_points,
            "memory_carried": memory_carried,
            "initial_norm": initial_norm,
            "final_norm": float(np.linalg.norm(current)),
            "bounded": cls["bounded"],
            "diverged": cls["diverged"],
            "equilibrium_hit": cls["equilibrium_hit"],
            "attractor_label": cls["attractor_label"],
            "max_lyapunov_estimate": "",
            "fft_dominant_frequency": fft["fft_peak"],
            "psd_entropy": fft["psd_entropy"],
            "notes": " ".join(notes),
        })
        path_rows.append({
            "candidate_id": candidate["candidate_id"],
            "step_index": step_index,
            "eta": eta,
            "q": q,
            "chi_name": step["chi_name"],
            "chi_value": step["chi_value"],
            "x": float(current[0]),
            "y": float(current[1]),
            "z": float(current[2]),
            "accepted": bool(not cls["diverged"]),
        })
        if cls["diverged"]:
            break

    return {
        "candidate_id": candidate["candidate_id"],
        "rows": rows,
        "path_rows": path_rows,
        "survived": bool(survived and rows and rows[-1]["bounded"] and not rows[-1]["equilibrium_hit"]),
        "final_class": rows[-1]["attractor_label"] if rows else "numerical_failure",
        "memory_carried": bool(rows and rows[-1]["memory_carried"]),
        "final_state": current,
        "final_traj": final_traj,
    }


def run_multiparameter_continuation(
    candidates: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    p: Dict[str, Any],
    equilibria: Sequence[np.ndarray],
    outdir: Path,
) -> List[Dict[str, Any]]:
    n = int(cfg.get("continuation", {}).get("max_candidates", 10))
    selected = list(candidates)[:n]
    summaries: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    path_rows: List[Dict[str, Any]] = []
    for cand in selected:
        try:
            result = continue_candidate(cand, cfg, p, equilibria)
        except Exception as exc:
            result = {
                "candidate_id": cand.get("candidate_id", ""),
                "rows": [{
                    "candidate_id": cand.get("candidate_id", ""),
                    "df_family": cand.get("df_family", ""),
                    "eta": "",
                    "q": cfg.get("q", ""),
                    "chi_name": "",
                    "chi_value": "",
                    "step_index": 0,
                    "h": cfg.get("continuation", {}).get("h", cfg.get("h", "")),
                    "memory_length": cfg.get("continuation", {}).get("memory_length", cfg.get("Lm", "")),
                    "memory_points": 0,
                    "memory_carried": False,
                    "initial_norm": "",
                    "final_norm": "",
                    "bounded": False,
                    "diverged": False,
                    "equilibrium_hit": False,
                    "attractor_label": "numerical_failure",
                    "max_lyapunov_estimate": "",
                    "fft_dominant_frequency": "",
                    "psd_entropy": "",
                    "notes": str(exc),
                }],
                "path_rows": [],
                "survived": False,
                "final_class": "numerical_failure",
                "memory_carried": False,
            }
        rows.extend(result["rows"])
        path_rows.extend(result["path_rows"])
        summaries.append(result)
    write_csv(Path(outdir) / "continuation_summary.csv", rows, CONT_FIELDS)
    write_csv(Path(outdir) / "continuation_paths.csv", path_rows)
    return summaries
