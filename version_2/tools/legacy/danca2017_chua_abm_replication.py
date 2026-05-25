#!/usr/bin/env python3
"""Replicate Danca 2017 Fig. 3 for the non-smooth fractional Chua system.

This module is intentionally independent from the EFORK route used elsewhere in
the project.  Danca's paper integrates Caputo initial value problems with the
Adams-Bashforth-Moulton predictor-corrector method, so this file implements that
method directly and uses the same Chua parameters and q=0.9998 reported for
Fig. 3.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

_CACHE_ROOT = Path(__file__).resolve().parent / ".runtime_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg_cache"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

import chua_initial_cond as chua
from equilibria_analysis import local_jacobian, solve_equilibria
from extended_search_utils import fft_peak_and_entropy, json_safe, trajectory_ranges, write_csv
from parallel_policy import force_single_openmp_thread_current_process


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTDIR = ROOT / "outputs" / ("danca2017_chua_abm_" + time.strftime("%Y%m%d_%H%M%S"))


@dataclass(frozen=True)
class DancaChuaConfig:
    """Numerical contract for the Danca Chua replication.

    Mathematical purpose:
        Define the Caputo IVP used to reproduce the non-smooth Chua hidden
        attractor from Danca 2017, Fig. 3.
    Equations:
        ^C D_t^q x = alpha (y - x - m1 x - psi(x)),
        ^C D_t^q y = x - y + z,
        ^C D_t^q z = -(beta y + gamma z),
        psi(x) = (m0 - m1) sat(x).
    Parameters:
        q, h, t_final, transient, and hiddenness test sizes.
    Output:
        Immutable configuration consumed by the ABM solver and report writer.
    Validity warning:
        Danca's text specifies ABM, q, parameters, and delta=0.01 for Fig. 3,
        but does not publish the hidden-attractor initial condition or h.  The
        initial condition is therefore localized numerically and h is recorded.
    """

    q: float = 0.9998
    h: float = 0.05
    t_final: float = 500.0
    transient: float = 250.0
    alpha: float = 8.4562
    beta: float = 12.0732
    gamma_chua: float = 0.0052
    m0: float = -0.1768
    m1: float = -1.1468
    delta: float = 0.01
    equilibrium_tol: float = 0.01
    divergence_norm: float = 120.0
    nontrivial_range_tol: float = 0.25
    local_samples_per_unstable_eq: int = 100
    figure_local_trajectories: int = 80
    rng_seed: int = 20260515
    store_stride: int = 1

    def params(self) -> Dict[str, Any]:
        return {
            "model": "piecewise",
            "alpha": np.float64(self.alpha),
            "beta": np.float64(self.beta),
            "gamma": np.float64(self.gamma_chua),
            "m0": np.float64(self.m0),
            "m1": np.float64(self.m1),
            "a1": np.float64(0.4),
            "a2": np.float64(-1.5585),
            "rho": np.float64(1.0),
        }


def validate_config(cfg: DancaChuaConfig) -> None:
    if not (0.0 < cfg.q <= 1.0):
        raise ValueError("q must satisfy 0 < q <= 1.")
    if cfg.h <= 0.0 or cfg.t_final <= 0.0:
        raise ValueError("h and t_final must be positive.")
    if cfg.transient < 0.0 or cfg.transient >= cfg.t_final:
        raise ValueError("transient must satisfy 0 <= transient < t_final.")
    if cfg.delta <= 0.0 or cfg.equilibrium_tol <= 0.0:
        raise ValueError("delta and equilibrium_tol must be positive.")


def chua_rhs_factory(cfg: DancaChuaConfig) -> Callable[[np.ndarray], np.ndarray]:
    p = cfg.params()

    def rhs(x: np.ndarray) -> np.ndarray:
        return np.asarray(chua.rhs_original(np.asarray(x, dtype=float), p), dtype=float)

    return rhs


def caputo_abm_integrate(
    rhs: Callable[[np.ndarray], np.ndarray],
    x0: Sequence[float],
    *,
    q: float,
    h: float,
    t_final: float,
    divergence_norm: float,
    store_stride: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Integrate a commensurate Caputo system with Diethelm ABM.

    Mathematical purpose:
        Solve ^C D_t^q x(t) = f(x(t)), x(0)=x0, 0<q<=1, with the
        Adams-Bashforth-Moulton predictor-corrector discretization used by
        Danca for the examples in the paper.
    Equations:
        Predictor: x^P_{n+1}=x0+h^q/Gamma(q+1) sum b_{j,n+1} f(x_j).
        Corrector: x_{n+1}=x0+h^q/Gamma(q+2)
        (f(x^P_{n+1}) + sum a_{j,n+1} f(x_j)).
    Parameters:
        rhs, x0, q, h, final time, divergence bound, and store stride.
    Output:
        Stored trajectory with columns [t,x,y,z] and metadata.
    Validity warning:
        This is the full-history ABM method and scales quadratically with the
        number of time steps.  It is meant as a paper-method replication, not as
        the fast EFORK production backend.
    """

    q = float(q)
    h = float(h)
    n_steps = int(math.ceil(float(t_final) / h))
    x0_arr = np.asarray(x0, dtype=float).reshape(3)
    if not np.all(np.isfinite(x0_arr)):
        raise ValueError("x0 must be finite and three-dimensional.")
    if n_steps < 1:
        raise ValueError("t_final/h must produce at least one step.")

    x_hist = np.zeros((n_steps + 1, 3), dtype=float)
    f_hist = np.zeros((n_steps + 1, 3), dtype=float)
    x_hist[0] = x0_arr
    f_hist[0] = rhs(x0_arr)

    powers = np.arange(n_steps + 2, dtype=float)
    pow_q = powers**q
    pow_q1 = powers ** (q + 1.0)
    hq = h**q
    pred_scale = hq / float(gamma(q + 1.0))
    corr_scale = hq / float(gamma(q + 2.0))

    diverged_at: int | None = None
    for i in range(n_steps):
        b = pow_q[1 : i + 2][::-1] - pow_q[0 : i + 1][::-1]
        predictor = x0_arr + pred_scale * (b @ f_hist[: i + 1])
        fp = rhs(predictor)

        if i == 0:
            a_weights = np.array([q], dtype=float)
        else:
            r = np.arange(i, 0, -1, dtype=int)
            mid = pow_q1[r + 1] + pow_q1[r - 1] - 2.0 * pow_q1[r]
            a0 = pow_q1[i] - (float(i) - q) * pow_q[i + 1]
            a_weights = np.concatenate(([a0], mid))

        corrected = x0_arr + corr_scale * ((a_weights @ f_hist[: i + 1]) + fp)
        x_hist[i + 1] = corrected

        if not np.all(np.isfinite(corrected)) or float(np.linalg.norm(corrected)) > float(divergence_norm):
            diverged_at = i + 1
            x_hist = x_hist[: i + 2]
            break
        f_hist[i + 1] = rhs(corrected)

    last_idx = x_hist.shape[0] - 1
    stride = max(1, int(store_stride))
    keep = np.arange(0, x_hist.shape[0], stride, dtype=int)
    if keep[-1] != last_idx:
        keep = np.append(keep, last_idx)
    t = keep.astype(float) * h
    traj = np.column_stack((t, x_hist[keep]))
    meta = {
        "history_policy": "full_caputo_history_no_finite_memory_truncation",
        "n_steps_requested": int(n_steps),
        "n_steps_completed": int(last_idx),
        "h": float(h),
        "t_final_requested": float(t_final),
        "t_final_completed": float(last_idx * h),
        "diverged": diverged_at is not None,
        "diverged_at_step": diverged_at,
        "store_stride": int(stride),
    }
    return traj, meta


def tail_states(traj: np.ndarray, transient: float) -> np.ndarray:
    arr = np.asarray(traj, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return np.empty((0, 3), dtype=float)
    mask = arr[:, 0] >= float(transient)
    if not np.any(mask):
        mask = np.arange(arr.shape[0]) >= max(0, int(0.5 * arr.shape[0]))
    return arr[mask, 1:4]


def classify_trajectory(
    traj: np.ndarray,
    cfg: DancaChuaConfig,
    eqs: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Classify a Danca-style trajectory without declaring hiddenness.

    Mathematical purpose:
        Separate equilibrium convergence, numerical divergence, and bounded
        nontrivial motion for local-basin tests around equilibria.
    Equations:
        Uses Euclidean distance to equilibria and post-transient component
        ranges; no Lyapunov theorem is inferred from this classification.
    Parameters:
        A trajectory, the numerical contract, and equilibrium points.
    Output:
        A conservative class row with distances, ranges, and notes.
    Validity warning:
        "bounded_nontrivial" is evidence of an observed attractor-like set, not
        a proof of chaos or hiddenness.
    """

    arr = np.asarray(traj, dtype=float)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 4:
        return {"class": "numerical_failure", "bounded": False, "target_hit": False}
    if not np.all(np.isfinite(arr[:, 1:4])):
        return {"class": "numerical_failure", "bounded": False, "target_hit": False}

    final = arr[-1, 1:4]
    final_norm = float(np.linalg.norm(final))
    final_dists = {key: float(np.linalg.norm(final - val)) for key, val in eqs.items()}
    closest_eq = min(final_dists, key=final_dists.get)
    closest_dist = final_dists[closest_eq]
    states_tail = tail_states(arr, cfg.transient)
    ranges = np.ptp(states_tail, axis=0) if states_tail.size else np.zeros(3)
    range_norm = float(np.linalg.norm(ranges))

    if final_norm > float(cfg.divergence_norm):
        klass = "infinity"
        bounded = False
        target_hit = False
    elif closest_dist <= float(cfg.equilibrium_tol):
        klass = f"equilibrium_{closest_eq}"
        bounded = True
        target_hit = False
    elif range_norm >= float(cfg.nontrivial_range_tol):
        klass = "bounded_nontrivial"
        bounded = True
        target_hit = True
    else:
        klass = "other_bounded"
        bounded = True
        target_hit = False

    out: Dict[str, Any] = {
        "class": klass,
        "bounded": bounded,
        "target_hit": target_hit,
        "final_norm": final_norm,
        "closest_equilibrium": closest_eq,
        "closest_equilibrium_distance": closest_dist,
        "range_norm_tail": range_norm,
        "range_x_tail": float(ranges[0]) if ranges.size else 0.0,
        "range_y_tail": float(ranges[1]) if ranges.size else 0.0,
        "range_z_tail": float(ranges[2]) if ranges.size else 0.0,
    }
    for key, dist in final_dists.items():
        out[f"final_dist_{key}"] = dist
    return out


def danca_equilibria_and_stability(cfg: DancaChuaConfig, outdir: Path) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    p = cfg.params()
    eqs = solve_equilibria(p)
    theta = float(cfg.q) * math.pi / 2.0
    rows: List[Dict[str, Any]] = []
    for eq_id, eq in eqs.items():
        jac = local_jacobian(p, eq)
        eig = np.linalg.eigvals(jac)
        margins = [abs(np.angle(v)) - theta for v in eig]
        rows.append(
            {
                "eq_id": eq_id,
                "x": float(eq[0]),
                "y": float(eq[1]),
                "z": float(eq[2]),
                "eig_1": complex(eig[0]),
                "eig_2": complex(eig[1]),
                "eig_3": complex(eig[2]),
                "min_arg_margin": float(min(margins)),
                "matignon_stable": bool(all(m > 0.0 for m in margins)),
                "paper_role": "stable" if eq_id == "E0" else "unstable",
            }
        )
    write_csv(outdir / "danca_chua_equilibria_abm.csv", rows)
    return eqs, rows


def seed_bank() -> List[Dict[str, Any]]:
    return [
        {
            "seed_id": "machado_mu4_seed_project",
            "source": "project_machado_candidate",
            "x0": [1.5793361282055747, 1.560622700429587, -0.44810555580843686],
        },
        {
            "seed_id": "machado_mu2_seed_project",
            "source": "project_machado_candidate",
            "x0": [3.039383584794975, -0.2416862069577155, -6.873467365218827],
        },
        {
            "seed_id": "lure_rank0001_df_seed_project",
            "source": "project_lure_candidate",
            "x0": [5.851768615656864, 0.3704086528621924, -8.360974120619202],
        },
        {
            "seed_id": "lure_rank0001_continuation_initial_project",
            "source": "project_lure_candidate",
            "x0": [5.345187953814787, 0.026031461919277056, -8.377736086443674],
        },
        {
            "seed_id": "integer_hidden_chua_like",
            "source": "trial_and_error",
            "x0": [1.6, 1.6, -0.45],
        },
        {
            "seed_id": "far_right_trial",
            "source": "trial_and_error",
            "x0": [4.0, 0.1, -7.0],
        },
        {
            "seed_id": "far_left_trial",
            "source": "trial_and_error",
            "x0": [-4.0, -0.1, 7.0],
        },
    ]


def save_trajectory(path: Path, traj: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z"])
        writer.writerows(np.asarray(traj, dtype=float).tolist())


def load_trajectory(path: Path) -> np.ndarray:
    rows: List[List[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([float(row["t"]), float(row["x"]), float(row["y"]), float(row["z"])])
    return np.asarray(rows, dtype=float)


def run_reference_search(cfg: DancaChuaConfig, outdir: Path, *, quick: bool = False) -> Dict[str, Any]:
    validate_config(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "trajectories").mkdir(exist_ok=True)
    eqs, _rows = danca_equilibria_and_stability(cfg, outdir)
    rhs = chua_rhs_factory(cfg)
    rows: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    best_traj: np.ndarray | None = None
    seeds = seed_bank()[:3] if quick else seed_bank()
    for item in seeds:
        traj, meta = caputo_abm_integrate(
            rhs,
            item["x0"],
            q=cfg.q,
            h=cfg.h,
            t_final=min(cfg.t_final, 40.0) if quick else cfg.t_final,
            divergence_norm=cfg.divergence_norm,
            store_stride=cfg.store_stride,
        )
        cls = classify_trajectory(traj, cfg, eqs)
        ranges = trajectory_ranges(traj)
        spec = fft_peak_and_entropy(traj, cfg.h)
        row = {
            "seed_id": item["seed_id"],
            "source": item["source"],
            "x0": item["x0"],
            **meta,
            **cls,
            **ranges,
            **spec,
        }
        rows.append(row)
        save_trajectory(outdir / "trajectories" / f"{item['seed_id']}_abm.csv", traj)
        if cls.get("class") == "bounded_nontrivial" and (
            best is None or float(row.get("range_norm_tail", 0.0)) > float(best.get("range_norm_tail", 0.0))
        ):
            best = row
            best_traj = traj

    write_csv(outdir / "danca_reference_seed_trials.csv", rows)
    if best is None:
        best = {
            "seed_id": "",
            "status": "no_bounded_nontrivial_seed_found",
            "notes": "No seed in the configured bank produced a bounded nontrivial ABM trajectory.",
        }
        best_traj = np.empty((0, 4), dtype=float)
    else:
        best["status"] = "reference_attractor_observed_abm"
    if best_traj is not None and best_traj.size:
        save_trajectory(outdir / "danca_fig3_reference_attractor_abm.csv", best_traj)
    summary = {
        "paper": "Danca 2017 Hidden chaotic attractors in fractional-order systems, Fig. 3",
        "method": "Caputo Adams-Bashforth-Moulton predictor-corrector",
        "history_policy": "Full Caputo history is retained for every ABM step; no finite-memory truncation is used for the Danca replication.",
        "paper_disclosed_initial_condition": False,
        "initial_condition_note": "The paper gives q, parameters, ABM method and delta=0.01, but not the hidden-attractor x0; this run records the located seed.",
        "config": asdict(cfg),
        "best_seed": json_safe(best),
        "n_seed_trials": len(rows),
    }
    (outdir / "danca_reference_summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    return summary


def random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(n, 3))
    norm = np.linalg.norm(raw, axis=1)
    norm[norm == 0.0] = 1.0
    return raw / norm[:, None]


def build_hiddenness_plan(cfg: DancaChuaConfig, eqs: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(cfg.rng_seed)
    base_dirs = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
    n = max(1, int(cfg.local_samples_per_unstable_eq))
    plan: List[Dict[str, Any]] = []
    for eq_id in [key for key in ("E+", "E-") if key in eqs]:
        eq = np.asarray(eqs[eq_id], dtype=float)
        dirs: List[np.ndarray] = []
        while len(dirs) < min(len(base_dirs), n):
            dirs.append(base_dirs[len(dirs)])
        if len(dirs) < n:
            dirs.extend(random_unit_vectors(n - len(dirs), rng))
        for idx, direction in enumerate(dirs[:n]):
            direction = np.asarray(direction, dtype=float)
            direction = direction / max(float(np.linalg.norm(direction)), 1e-300)
            plan.append(
                {
                    "case_id": f"{eq_id}_delta_{cfg.delta:g}_{idx:04d}",
                    "eq_id": eq_id,
                    "delta": float(cfg.delta),
                    "direction": direction.tolist(),
                    "x0": (eq + float(cfg.delta) * direction).tolist(),
                    "save_trajectory": idx < int(cfg.figure_local_trajectories),
                }
            )
    return plan


def _hiddenness_worker(payload: Tuple[Dict[str, Any], Dict[str, Any], str]) -> Dict[str, Any]:
    force_single_openmp_thread_current_process()
    case, cfg_raw, outdir_s = payload
    cfg = DancaChuaConfig(**cfg_raw)
    outdir = Path(outdir_s)
    eqs = solve_equilibria(cfg.params())
    rhs = chua_rhs_factory(cfg)
    start = time.time()
    try:
        traj, meta = caputo_abm_integrate(
            rhs,
            case["x0"],
            q=cfg.q,
            h=cfg.h,
            t_final=cfg.t_final,
            divergence_norm=cfg.divergence_norm,
            store_stride=cfg.store_stride,
        )
        cls = classify_trajectory(traj, cfg, eqs)
        if bool(case.get("save_trajectory", False)):
            save_trajectory(outdir / "trajectories" / f"{case['case_id']}_abm.csv", traj)
        return {
            **case,
            **meta,
            **cls,
            "elapsed_sec": time.time() - start,
            "status": "ok",
        }
    except Exception as exc:
        return {
            **case,
            "class": "numerical_failure",
            "bounded": False,
            "target_hit": False,
            "elapsed_sec": time.time() - start,
            "status": "exception",
            "error": repr(exc),
        }


def run_hiddenness_tests(cfg: DancaChuaConfig, outdir: Path, *, workers: int = 1, quick: bool = False) -> Dict[str, Any]:
    validate_config(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "trajectories").mkdir(exist_ok=True)
    eqs, _rows = danca_equilibria_and_stability(cfg, outdir)
    plan = build_hiddenness_plan(cfg, eqs)
    if quick:
        plan = plan[: min(8, len(plan))]
    (outdir / "danca_hiddenness_plan.json").write_text(json.dumps(json_safe(plan), indent=2), encoding="utf-8")
    rows: List[Dict[str, Any]] = []
    payloads = [(case, asdict(cfg), str(outdir)) for case in plan]
    n_workers = max(1, int(workers))
    if n_workers == 1:
        rows = [_hiddenness_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_hiddenness_worker, payload) for payload in payloads]
            for fut in as_completed(futures):
                rows.append(fut.result())
    rows.sort(key=lambda r: str(r.get("case_id", "")))
    write_csv(outdir / "danca_hiddenness_raw.csv", rows)

    summary_rows: List[Dict[str, Any]] = []
    for eq_id in sorted({str(r.get("eq_id")) for r in rows}):
        sub = [r for r in rows if str(r.get("eq_id")) == eq_id]
        target_hits = sum(1 for r in sub if bool(r.get("target_hit", False)))
        equilibrium_hits = sum(1 for r in sub if str(r.get("class", "")).startswith("equilibrium_"))
        infinity_hits = sum(1 for r in sub if str(r.get("class")) == "infinity")
        failures = sum(1 for r in sub if str(r.get("status")) != "ok")
        if target_hits:
            decision = "not_supported_by_Danca_delta_test"
        elif failures:
            decision = "inconclusive_due_to_numerical_failures"
        else:
            decision = "compatible_with_hiddenness_under_Danca_delta_test"
        summary_rows.append(
            {
                "eq_id": eq_id,
                "delta": cfg.delta,
                "n": len(sub),
                "target_hits": target_hits,
                "equilibrium_hits": equilibrium_hits,
                "infinity_hits": infinity_hits,
                "failures": failures,
                "decision": decision,
            }
        )
    write_csv(outdir / "danca_hiddenness_summary.csv", summary_rows)
    overall = {
        "decision": (
            "not_supported_by_Danca_delta_test"
            if any(int(r["target_hits"]) > 0 for r in summary_rows)
            else (
                "inconclusive_due_to_numerical_failures"
                if any(int(r["failures"]) > 0 for r in summary_rows)
                else "compatible_with_hiddenness_under_Danca_delta_test"
            )
        ),
        "note": "This is a numerical local-basin test at delta=0.01, not a formal proof.",
        "summary": summary_rows,
    }
    (outdir / "danca_hiddenness_decision.json").write_text(json.dumps(json_safe(overall), indent=2), encoding="utf-8")
    return overall


def plot_danca_figure3(cfg: DancaChuaConfig, outdir: Path) -> Dict[str, str]:
    eqs = solve_equilibria(cfg.params())
    ref_path = outdir / "danca_fig3_reference_attractor_abm.csv"
    if not ref_path.exists():
        return {"status": "missing_reference", "path": str(ref_path)}
    ref = load_trajectory(ref_path)
    fig = plt.figure(figsize=(12, 5.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    tail = ref[ref[:, 0] >= cfg.transient]
    if tail.shape[0] < 10:
        tail = ref[max(0, int(0.5 * len(ref))) :]
    ax1.plot(tail[:, 1], tail[:, 2], tail[:, 3], lw=0.45, color="#16a34a", alpha=0.92, label="ABM attractor")
    for eq_id, eq in eqs.items():
        ax1.scatter([eq[0]], [eq[1]], [eq[2]], s=38, label=eq_id)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    raw_path = outdir / "danca_hiddenness_raw.csv"
    plotted = 0
    if raw_path.exists():
        with raw_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                case_id = row.get("case_id", "")
                traj_path = outdir / "trajectories" / f"{case_id}_abm.csv"
                if not traj_path.exists():
                    continue
                arr = load_trajectory(traj_path)
                color = "#dc2626" if str(row.get("class", "")).startswith("equilibrium_") else "#2563eb"
                ax2.plot(arr[:, 1], arr[:, 2], arr[:, 3], lw=0.35, alpha=0.55, color=color)
                plotted += 1
                if plotted >= cfg.figure_local_trajectories:
                    break
    for eq_id, eq in eqs.items():
        ax2.scatter([eq[0]], [eq[1]], [eq[2]], s=38, label=eq_id)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    png = outdir / "fig03_danca2017_chua_abm_replica.png"
    pdf = outdir / "fig03_danca2017_chua_abm_replica.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return {"status": "ok", "png": str(png), "pdf": str(pdf)}


PROJECT_FIGURE_SPECS = [
    {
        "candidate_id": "lure_biased_q_0p99980_rank_0001",
        "method": "biased_Lure_harmonic_seed_plus_EFORK",
        "path": ROOT
        / "outputs"
        / "lure_biased_multiparam_q09998"
        / "trajectories"
        / "lure_biased_q_0p99980_rank_0001_lure_biased_q_0p99980_rank_0001_phi_00_C1_continuation.csv",
    },
    {
        "candidate_id": "branch_0_mu_4p00000_theta_0p00000",
        "method": "Machado_describing_function_plus_EFORK",
        "path": ROOT
        / "outputs"
        / "extended_search"
        / "machado_targeted_verification_lm10"
        / "trajectories"
        / "branch_0_mu_4p00000_theta_0p00000_reference_attractor.csv",
    },
]


def plot_project_candidate_figures(cfg: DancaChuaConfig, outdir: Path) -> Dict[str, Any]:
    eqs = solve_equilibria(cfg.params())
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for spec in PROJECT_FIGURE_SPECS:
        path = Path(spec["path"])
        if not path.exists():
            rows.append({**spec, "status": "missing_trajectory"})
            continue
        traj = load_trajectory(path)
        tail = traj[traj[:, 0] >= min(cfg.transient, float(traj[-1, 0]) * 0.5)]
        if tail.shape[0] < 10:
            tail = traj[max(0, int(0.5 * len(traj))) :]
        fig = plt.figure(figsize=(6.2, 5.3))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.plot(tail[:, 1], tail[:, 2], tail[:, 3], lw=0.45, color="#0f766e", alpha=0.9)
        for eq_id, eq in eqs.items():
            ax.scatter([eq[0]], [eq[1]], [eq[2]], s=36, label=eq_id)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        png = outdir / f"fig_project_{spec['candidate_id']}.png"
        pdf = outdir / f"fig_project_{spec['candidate_id']}.pdf"
        fig.savefig(png, dpi=220)
        fig.savefig(pdf)
        plt.close(fig)
        cls = classify_trajectory(traj, cfg, eqs)
        rows.append(
            {
                "candidate_id": spec["candidate_id"],
                "method": spec["method"],
                "source_path": str(path),
                "status": "ok",
                "figure_png": str(png),
                "figure_pdf": str(pdf),
                **trajectory_ranges(traj),
                **cls,
            }
        )
    write_csv(outdir / "project_best_two_figure_summary.csv", rows)
    return {"rows": rows}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def summarize_existing_project_candidates() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    lure_summary = ROOT / "outputs" / "lure_biased_multiparam_q09998" / "lure_biased_multiparam_summary.json"
    if lure_summary.exists():
        data = json.loads(lure_summary.read_text(encoding="utf-8"))
        out.append(
            {
                "candidate_group": "biased_Lure_q09998",
                "best_candidate": data.get("best_candidate_id", ""),
                "method": "biased Lure / describing function residual search + EFORK continuation",
                "q": data.get("q_global", ""),
                "status": data.get("best_candidate_status", ""),
                "hidden_verified_only_if_full_protocol_passed": data.get("hidden_status", "") == "hidden_verified_only_if_full_protocol_passed",
                "n_survivors": data.get("n_continuation_survivors", ""),
                "n_robust_survivors": data.get("n_robust_survivors", ""),
            }
        )
    refined = ROOT / "outputs" / "extended_search" / "corrida1" / "refined_hiddenness_decision.csv"
    for row in read_csv_rows(refined):
        out.append(
            {
                "candidate_group": "Machado_corrida1_refined",
                "best_candidate": row.get("candidate_id", ""),
                "method": "Machado describing function extension + EFORK hiddenness refinement",
                "q": "0.9998",
                "status": row.get("hiddenness_status", row.get("decision", "")),
                "hidden_verified_only_if_full_protocol_passed": False,
                "n_survivors": "",
                "n_robust_survivors": "",
            }
        )
    targeted = ROOT / "outputs" / "extended_search" / "machado_targeted_verification" / "targeted_equilibrium_summary.csv"
    for row in read_csv_rows(targeted):
        out.append(
            {
                "candidate_group": "Machado_targeted_latest",
                "best_candidate": row.get("candidate_id", ""),
                "method": "targeted equilibrium-neighborhood EFORK controls",
                "q": "0.9998",
                "status": f"{row.get('n_target', '')}/{row.get('n_tests', '')} target hits at {row.get('equilibrium_id', '')}, r={row.get('radius', '')}",
                "hidden_verified_only_if_full_protocol_passed": False,
                "n_survivors": "",
                "n_robust_survivors": "",
            }
        )
    return out


def write_comparison_report(cfg: DancaChuaConfig, outdir: Path) -> Path:
    danca_ref = outdir / "danca_reference_summary.json"
    danca_hidden = outdir / "danca_hiddenness_decision.json"
    ref_data = json.loads(danca_ref.read_text(encoding="utf-8")) if danca_ref.exists() else {}
    hidden_data = json.loads(danca_hidden.read_text(encoding="utf-8")) if danca_hidden.exists() else {}
    project_rows = summarize_existing_project_candidates()
    report = outdir / "danca2017_vs_project_candidates_report.md"
    lines: List[str] = []
    lines.append("# Danca 2017 Fig. 3 vs project candidates")
    lines.append("")
    lines.append("## Paper contract")
    lines.append("")
    lines.append("- System: non-smooth Chua system from Danca 2017, Eq. (8).")
    lines.append("- Parameters: `m0=-0.1768`, `m1=-1.1468`, `alpha=8.4562`, `beta=12.0732`, `gamma=0.0052`.")
    lines.append("- Fractional order: `q=0.9998`.")
    lines.append("- Numerical method: Caputo Adams-Bashforth-Moulton predictor-corrector.")
    lines.append("- Memory/history: full Caputo history at every ABM step; no finite-memory truncation.")
    lines.append("- Hiddenness check reported in the paper: vicinity size `delta=0.01` around unstable equilibria, with local trajectories tending to `X0` or infinity.")
    lines.append("- Missing paper datum: the hidden-attractor initial condition for Fig. 3 is not published in the PDF text used here.")
    lines.append("")
    lines.append("## ABM replication status")
    lines.append("")
    lines.append(f"- Config used here: `h={cfg.h}`, `t_final={cfg.t_final}`, `transient={cfg.transient}`, `delta={cfg.delta}`, `store_stride={cfg.store_stride}`.")
    if ref_data:
        best = ref_data.get("best_seed", {})
        lines.append(f"- Located reference seed: `{best.get('seed_id', '')}` from `{best.get('source', '')}`.")
        lines.append(f"- Reference status: `{best.get('status', '')}`, class `{best.get('class', '')}`.")
    else:
        lines.append("- Reference status: not run yet.")
    if hidden_data:
        lines.append(f"- Hiddenness decision label: `{hidden_data.get('decision', '')}`.")
        lines.append("- This label is numerical compatibility under the tested radius, not a theorem.")
    else:
        lines.append("- Hiddenness tests: not run yet.")
    lines.append("")
    lines.append("## Project candidates")
    lines.append("")
    lines.append("| group | candidate | method | q | current status | hidden_verified_only_if_full_protocol_passed |")
    lines.append("|---|---|---|---:|---|---:|")
    for row in project_rows:
        lines.append(
            "| {candidate_group} | {best_candidate} | {method} | {q} | {status} | {hidden_verified_only_if_full_protocol_passed} |".format(
                **{k: str(v).replace("|", "/") for k, v in row.items()}
            )
        )
    lines.append("")
    lines.append("## Replication commands")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python3 danca2017_chua_abm_replication.py --output-dir {outdir} --job all --workers 4")
    lines.append("python3 launch_danca2017_jobs.py --workers 4")
    lines.append("```")
    lines.append("")
    lines.append("## Conservative reading")
    lines.append("")
    lines.append("Danca Fig. 3 and the project candidates use the same non-smooth Chua parameters and q, but not the same localization route.  Danca reports a trial-and-error hidden attractor observed with ABM; the project candidates were generated by Lure/Machado harmonic seeds and validated with EFORK.  A candidate is not promoted to `hidden_verified_only_if_full_protocol_passed` unless equilibrium-neighborhood controls fail to reproduce the target under the tested contract.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_all(cfg: DancaChuaConfig, outdir: Path, *, workers: int, quick: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "run_config.json").write_text(json.dumps(json_safe(asdict(cfg)), indent=2), encoding="utf-8")
    run_reference_search(cfg, outdir, quick=quick)
    run_hiddenness_tests(cfg, outdir, workers=workers, quick=quick)
    plot_danca_figure3(cfg, outdir)
    plot_project_candidate_figures(cfg, outdir)
    write_comparison_report(cfg, outdir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Danca 2017 Fig. 3 ABM replication for non-smooth Chua.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--job", choices=["all", "reference", "hiddenness", "figures", "report"], default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--q", type=float, default=0.9998)
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--t-final", type=float, default=500.0)
    parser.add_argument("--transient", type=float, default=250.0)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--local-samples-per-unstable-eq", type=int, default=100)
    parser.add_argument("--figure-local-trajectories", type=int, default=80)
    parser.add_argument("--store-stride", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="Short smoke test; not a scientific replication.")
    return parser.parse_args()


def main() -> None:
    force_single_openmp_thread_current_process()
    args = parse_args()
    cfg = DancaChuaConfig(
        q=args.q,
        h=args.h,
        t_final=args.t_final,
        transient=args.transient,
        delta=args.delta,
        local_samples_per_unstable_eq=args.local_samples_per_unstable_eq,
        figure_local_trajectories=args.figure_local_trajectories,
        store_stride=max(1, int(args.store_stride)),
    )
    if args.quick:
        cfg = DancaChuaConfig(
            q=cfg.q,
            h=max(cfg.h, 0.05),
            t_final=min(cfg.t_final, 40.0),
            transient=min(cfg.transient, 20.0),
            delta=cfg.delta,
            local_samples_per_unstable_eq=min(cfg.local_samples_per_unstable_eq, 4),
            figure_local_trajectories=min(cfg.figure_local_trajectories, 8),
            store_stride=cfg.store_stride,
        )
    validate_config(cfg)
    outdir = Path(args.output_dir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    if args.job == "all":
        run_all(cfg, outdir, workers=args.workers, quick=args.quick)
    elif args.job == "reference":
        run_reference_search(cfg, outdir, quick=args.quick)
    elif args.job == "hiddenness":
        run_hiddenness_tests(cfg, outdir, workers=args.workers, quick=args.quick)
    elif args.job == "figures":
        plot_danca_figure3(cfg, outdir)
        plot_project_candidate_figures(cfg, outdir)
    elif args.job == "report":
        write_comparison_report(cfg, outdir)


if __name__ == "__main__":
    main()
