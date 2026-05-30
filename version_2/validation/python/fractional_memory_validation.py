"""Fractional memory validation for Caputo-based Chua systems.

This module validates numerically the effect of using full Caputo memory versus
finite-window (truncated) memory in fractional-order Chua integrations.

Mathematical context
--------------------
For 0 < q < 1, the Caputo derivative is:

    ^C D_t^q x(t) = 1/Gamma(1-q) * integral_{t0}^{t} (t-tau)^(-q) x'(tau) dtau

The effective numerical state at step k is the discrete history:

    H_k = {X(t_{k-M}), ..., X(t_k)}

This phase compares:
  - full Caputo memory (memory_mode="full");
  - finite-window approximations with M = 256, 512, 1024 steps;
  - dynamic classification of each trajectory;
  - sensitivity of statistics (rho_attractor, rho_max, range, center) to window size;
  - automatic warnings when a window is insufficient.

Tail-defect bound (diagnostic only, not a pass/fail criterion)
---------------------------------------------------------------
For window length L = M * h:

    E_L(t) = 1/Gamma(1-q) * integral_{t0}^{t-L} (t-tau)^(-q) x'(tau) dtau

If ||x'|| <= K:

    |E_L(t)| <= K / Gamma(2-q) * [(t-t0)^(1-q) - L^(1-q)],   for t-t0 > L

References
----------
- Caputo (1967): original derivative definition.
- Yoon & You (2017), arXiv:1711.10071: adaptive memory method for Caputo derivative.
- Hai, Ren, Yu, Mo & Xu (2020), arXiv:2007.05755: short-memory fractional DEs.
- Danca & Fečkan (2024), arXiv:2406.04686: memory principle in fractional-order codes.

No claims
---------
This phase does NOT certify hidden attractors, chaos, or Lyapunov exponents.
All outputs carry:
    hiddenness_certified_by_this_pipeline: false
    no_hidden_verified_claim: true
    pointwise_comparison_used: false
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Try importing yaml; graceful error if missing
# ---------------------------------------------------------------------------
try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _yaml = None  # type: ignore[assignment]
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Mandatory no-claim metadata injected into every output
# ---------------------------------------------------------------------------
_NO_CLAIM: Dict[str, Any] = {
    "hiddenness_certified_by_this_pipeline": False,
    "no_hidden_verified_claim": True,
    "pointwise_comparison_used": False,
}

_ALLOWED_DYNAMIC_CLASSES = {
    "nan_detected",
    "diverged",
    "too_short",
    "collapsed_to_equilibrium",
    "bounded_nontrivial",
    "inconclusive",
}

_ALLOWED_WINDOW_STATUSES = {
    "full_memory_reference",
    "window_memory_sufficient",
    "window_memory_sensitive",
    "window_memory_insufficient",
}

_ALLOWED_OVERALL_STATUSES = {
    "memory_validation_passed",
    "memory_validation_passed_with_sensitive_windows",
    "memory_validation_inconclusive",
    "memory_validation_failed",
}


# ===========================================================================
# 1. load_memory_config
# ===========================================================================

def load_memory_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate a fractional memory validation YAML config.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    config : dict
        Validated configuration dictionary.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    ValueError
        If required fields are missing or invalid.
    """
    if not _YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required to load memory configs. "
            "Install with: pip install pyyaml"
        )

    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        config = _yaml.safe_load(fh)

    if config is None:
        raise ValueError(f"Empty YAML file: {path}")

    # Required top-level fields
    required = ["case_id", "system_id", "q", "initial_conditions",
                "integrator", "time", "memory", "comparison_policy"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing required fields in {path}: {missing}")

    q = float(config["q"])
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0, 1), got {q}")

    method = config["integrator"].get("method", "")
    if method != "ABM":
        raise ValueError(
            f"integrator.method must be 'ABM', got '{method}'"
        )

    mem = config["memory"]
    if not mem.get("full_reference", False):
        raise ValueError("memory.full_reference must be true")

    windows = mem.get("windows", [])
    if not windows:
        raise ValueError("memory.windows must contain at least one entry")

    policy = config["comparison_policy"]
    if policy.get("pointwise_comparison_used", True):
        raise ValueError(
            "comparison_policy.pointwise_comparison_used must be false"
        )

    return config


# ===========================================================================
# 2. get_fractional_memory_system
# ===========================================================================

def get_fractional_memory_system(system_id: str):
    """Return a callable rhs(t, x) for the given system_id.

    Supported system_ids:
    - 'chua_fractional_saturation' → chua-nonsmooth
    - 'chua_fractional_arctan'     → chua-arctan

    Parameters
    ----------
    system_id : str
        One of the supported system identifiers.

    Returns
    -------
    rhs : callable
        Function with signature rhs(t, x) -> np.ndarray, suitable for
        hidden_attractors.integrations.abm.caputo_abm_integrate.
    system_obj : ChaoticSystem
        The registered ChaoticSystem object.

    Raises
    ------
    ValueError
        If system_id is not recognised.
    """
    from hidden_attractors.systems import get_system

    _ID_MAP = {
        "chua_fractional_saturation": "chua-nonsmooth",
        "chua_fractional_arctan": "chua-arctan",
    }

    registry_name = _ID_MAP.get(system_id)
    if registry_name is None:
        raise ValueError(
            f"Unknown system_id '{system_id}'. "
            f"Supported: {list(_ID_MAP.keys())}"
        )

    system_obj = get_system(registry_name)

    # Wrap to produce rhs(t, x) → np.ndarray
    # caputo_abm_integrate expects rhs(t, x); ChaoticSystem.evaluate takes (state,)
    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        return system_obj.evaluate(np.asarray(x, dtype=float))

    return rhs, system_obj


# ===========================================================================
# 3. classify_fractional_trajectory
# ===========================================================================

def classify_fractional_trajectory(
    times: np.ndarray,
    states: np.ndarray,
    t_burn: float,
    divergence_norm: float,
    collapse_variance_tolerance: float,
    min_range_tolerance: float,
) -> str:
    """Classify a fractional trajectory by its asymptotic dynamics.

    Parameters
    ----------
    times : np.ndarray, shape (N,)
    states : np.ndarray, shape (N, dim)
    t_burn : float
        Burn-in time; post-transient analysis uses t >= t_burn.
    divergence_norm : float
        Max ||X|| threshold above which trajectory is considered diverged.
    collapse_variance_tolerance : float
        If max component-wise variance < this, trajectory collapsed.
    min_range_tolerance : float
        If max component-wise range < this, trajectory collapsed.

    Returns
    -------
    class_label : str
        One of: 'nan_detected', 'diverged', 'too_short',
        'collapsed_to_equilibrium', 'bounded_nontrivial', 'inconclusive'.
    """
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)

    # 1. NaN / Inf check
    if not np.all(np.isfinite(states)):
        return "nan_detected"

    # 2. Divergence check (over full trajectory)
    norms = np.linalg.norm(states, axis=1)
    if np.max(norms) > divergence_norm:
        return "diverged"

    # 3. Post-transient subset
    mask = times >= t_burn
    post_states = states[mask]

    if post_states.shape[0] < 2:
        return "too_short"

    # 4. Collapse check
    var_per_dim = np.var(post_states, axis=0)
    range_per_dim = np.max(post_states, axis=0) - np.min(post_states, axis=0)

    if np.max(var_per_dim) < collapse_variance_tolerance:
        return "collapsed_to_equilibrium"
    if np.max(range_per_dim) < min_range_tolerance:
        return "collapsed_to_equilibrium"

    # 5. Bounded non-trivial
    return "bounded_nontrivial"


# ===========================================================================
# 4. compute_memory_metrics
# ===========================================================================

def compute_memory_metrics(
    times: np.ndarray,
    states: np.ndarray,
    t_burn: float,
) -> Dict[str, Any]:
    """Compute attractor statistics from a trajectory.

    Parameters
    ----------
    times : np.ndarray, shape (N,)
    states : np.ndarray, shape (N, dim)
    t_burn : float
        Burn-in time; metrics computed on post-transient states.

    Returns
    -------
    metrics : dict with keys:
        final_state, max_norm, mean_vector, std_vector, variance_vector,
        min_vector, max_vector, range_vector, rho_attractor, rho_max, n_post
    """
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)

    # final state (last point)
    final_state = states[-1].tolist()
    max_norm = float(np.max(np.linalg.norm(states, axis=1)))

    # post-transient
    mask = times >= t_burn
    post = states[mask]
    n_post = int(post.shape[0])

    if n_post < 1:
        # Degenerate: no post-transient points
        dim = states.shape[1]
        zero = [0.0] * dim
        return {
            "final_state": final_state,
            "max_norm": max_norm,
            "mean_vector": zero,
            "std_vector": zero,
            "variance_vector": zero,
            "min_vector": zero,
            "max_vector": zero,
            "range_vector": zero,
            "rho_attractor": 0.0,
            "rho_max": 0.0,
            "n_post": 0,
        }

    mean_vec = np.mean(post, axis=0)           # shape (dim,)
    std_vec = np.std(post, axis=0)
    var_vec = np.var(post, axis=0)
    min_vec = np.min(post, axis=0)
    max_vec = np.max(post, axis=0)
    range_vec = max_vec - min_vec

    # rho_attractor: RMS distance from centroid (translation-invariant)
    diffs = post - mean_vec[np.newaxis, :]     # (n_post, dim)
    dist_sq = np.sum(diffs ** 2, axis=1)       # (n_post,)
    rho_attractor = float(np.sqrt(np.mean(dist_sq)))

    # rho_max: maximum distance from centroid
    rho_max = float(np.sqrt(np.max(dist_sq)))

    return {
        "final_state": final_state,
        "max_norm": max_norm,
        "mean_vector": mean_vec.tolist(),
        "std_vector": std_vec.tolist(),
        "variance_vector": var_vec.tolist(),
        "min_vector": min_vec.tolist(),
        "max_vector": max_vec.tolist(),
        "range_vector": range_vec.tolist(),
        "rho_attractor": rho_attractor,
        "rho_max": rho_max,
        "n_post": n_post,
    }


# ===========================================================================
# 5. estimate_caputo_tail_bound
# ===========================================================================

def estimate_caputo_tail_bound(
    q: float,
    t_final: float,
    t_burn: float,
    memory_window_steps: int,
    h: float,
    derivative_bound: float,
) -> float:
    """Estimate the crude tail-defect bound for a Caputo window approximation.

    The bound is:

        |E_L(t)| <= derivative_bound / Gamma(2-q) * [(t_final)^(1-q) - L^(1-q)]

    where L = memory_window_steps * h.

    This is a diagnostic value only; it does NOT determine pass/fail.

    Parameters
    ----------
    q : float
        Caputo order, 0 < q < 1.
    t_final : float
        End time of integration.
    t_burn : float
        Burn-in time (not used in the bound, retained for signature clarity).
    memory_window_steps : int
        Window size in number of steps.
    h : float
        Step size.
    derivative_bound : float
        Upper bound K on ||F(X_n)|| over the trajectory.

    Returns
    -------
    tail_bound : float
        Non-negative diagnostic bound estimate.
    """
    L = float(memory_window_steps) * float(h)
    t = float(t_final)
    q = float(q)

    if t <= L:
        return 0.0

    gamma_2mq = math.gamma(2.0 - q)
    if gamma_2mq < 1e-300:
        return float("inf")

    tail = float(derivative_bound) / gamma_2mq * (t ** (1.0 - q) - L ** (1.0 - q))
    return max(0.0, tail)


# ===========================================================================
# 6. compare_window_to_full
# ===========================================================================

def compare_window_to_full(
    window_metrics: Dict[str, Any],
    full_metrics: Dict[str, Any],
    dynamic_class_window: str,
    dynamic_class_full: str,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare finite-window metrics to full-memory reference.

    No point-wise comparison is performed. Comparison uses aggregate
    statistics: rho_attractor, rho_max, range, and centroid.

    Parameters
    ----------
    window_metrics : dict
        Output of compute_memory_metrics for a window run.
    full_metrics : dict
        Output of compute_memory_metrics for the full-memory run.
    dynamic_class_window : str
        Classification label for the window run.
    dynamic_class_full : str
        Classification label for the full-memory run.
    policy : dict
        comparison_policy from the YAML config.

    Returns
    -------
    result : dict
        Keys: rho_relative_error, rho_max_relative_error,
        range_relative_error, center_relative_error,
        class_changed, warning, status, warnings_detail.
    """
    eps = 1e-15

    rho_full = float(full_metrics["rho_attractor"])
    rho_win = float(window_metrics["rho_attractor"])
    rho_rel = abs(rho_win - rho_full) / (rho_full + eps)

    rho_max_full = float(full_metrics["rho_max"])
    rho_max_win = float(window_metrics["rho_max"])
    rho_max_rel = abs(rho_max_win - rho_max_full) / (rho_max_full + eps)

    range_full = np.asarray(full_metrics["range_vector"], dtype=float)
    range_win = np.asarray(window_metrics["range_vector"], dtype=float)
    range_norm_full = float(np.linalg.norm(range_full))
    range_rel = float(np.linalg.norm(range_win - range_full)) / (range_norm_full + eps)

    mean_win = np.asarray(window_metrics["mean_vector"], dtype=float)
    mean_full = np.asarray(full_metrics["mean_vector"], dtype=float)
    std_full = np.asarray(full_metrics["std_vector"], dtype=float)
    std_norm_full = float(np.linalg.norm(std_full))
    center_rel = float(np.linalg.norm(mean_win - mean_full)) / (std_norm_full + eps)

    class_changed = (dynamic_class_window != dynamic_class_full)

    # Tolerances from policy
    rho_tol = float(policy.get("rho_relative_tolerance", 0.25))
    rho_max_tol = float(policy.get("rho_max_relative_tolerance", 0.35))
    range_tol = float(policy.get("range_relative_tolerance", 0.35))
    center_tol = float(policy.get("center_relative_tolerance", 0.60))

    warnings_detail: List[str] = []

    if class_changed:
        warnings_detail.append(
            f"dynamic_class changed: full='{dynamic_class_full}' "
            f"vs window='{dynamic_class_window}'"
        )
    if rho_rel > rho_tol:
        warnings_detail.append(
            f"rho_relative_error={rho_rel:.4f} exceeds tolerance={rho_tol}"
        )
    if rho_max_rel > rho_max_tol:
        warnings_detail.append(
            f"rho_max_relative_error={rho_max_rel:.4f} exceeds tolerance={rho_max_tol}"
        )
    if range_rel > range_tol:
        warnings_detail.append(
            f"range_relative_error={range_rel:.4f} exceeds tolerance={range_tol}"
        )
    if center_rel > center_tol:
        warnings_detail.append(
            f"center_relative_error={center_rel:.4f} exceeds tolerance={center_tol}"
        )

    warning = len(warnings_detail) > 0

    # Determine status
    if class_changed:
        status = "window_memory_insufficient"
    elif warning:
        status = "window_memory_sensitive"
    else:
        status = "window_memory_sufficient"

    return {
        "rho_relative_error": rho_rel,
        "rho_max_relative_error": rho_max_rel,
        "range_relative_error": range_rel,
        "center_relative_error": center_rel,
        "class_changed": class_changed,
        "warning": warning,
        "warnings_detail": warnings_detail,
        "status": status,
    }


# ===========================================================================
# 7. _resolve_window_steps
# ===========================================================================

def _resolve_window_steps(M_value: float, units: str, h: float) -> int:
    """Convert a window size to integer step count.

    Parameters
    ----------
    M_value : float
        Window size value from YAML.
    units : str
        'steps' or 'time'.
    h : float
        Integration step size.

    Returns
    -------
    M_steps : int
    """
    if units == "time":
        return int(round(float(M_value) / h))
    else:
        # 'steps' (default)
        return int(M_value)


# ===========================================================================
# 8. run_memory_case
# ===========================================================================

def run_memory_case(
    config: Dict[str, Any],
    fast: bool = False,
    output_dir: Optional[str | Path] = None,
    save_trajectories: bool = False,
) -> Dict[str, Any]:
    """Run full-memory and window-memory integrations for all ICs in config.

    Uses the official hidden_attractors.integrations.abm.caputo_abm_integrate.
    Does NOT re-implement ABM.

    Parameters
    ----------
    config : dict
        Validated config from load_memory_config.
    fast : bool
        If True, uses fast_test time parameters.
    output_dir : Path or None
        Where to write output files. Created if absent.
    save_trajectories : bool
        If True, saves trajectory arrays as .npy files.

    Returns
    -------
    results : dict
        All run results and comparison data.
    """
    from hidden_attractors.integrations.abm import caputo_abm_integrate

    case_id = config["case_id"]
    system_id = config["system_id"]
    q = float(config["q"])
    h = float(config["integrator"]["h"])

    # Time parameters
    if fast and "fast_test" in config:
        t_final = float(config["fast_test"]["t_final"])
        t_burn = float(config["fast_test"]["t_burn"])
    else:
        t_final = float(config["time"]["t_final"])
        t_burn = float(config["time"]["t_burn"])

    divergence_norm = float(config.get("divergence", {}).get("max_norm", 120.0))
    clf_cfg = config.get("classification", {})
    collapse_var_tol = float(clf_cfg.get("collapse_variance_tolerance", 1e-8))
    min_range_tol = float(clf_cfg.get("min_range_tolerance", 1e-5))

    memory_cfg = config["memory"]
    units = memory_cfg.get("units", "steps")
    window_entries = memory_cfg.get("windows", [])
    windows_steps = [
        _resolve_window_steps(w["M"], units, h) for w in window_entries
    ]

    policy = config["comparison_policy"]

    # Build initial conditions dict
    ic_dict: Dict[str, np.ndarray] = {}
    for ic_key, ic_val in config["initial_conditions"].items():
        ic_dict[ic_key] = np.asarray(ic_val, dtype=float)

    # Get system RHS
    rhs, _system_obj = get_fractional_memory_system(system_id)

    # Output directory
    if output_dir is not None:
        out_dir = Path(output_dir) / case_id
        out_dir.mkdir(parents=True, exist_ok=True)
        if save_trajectories:
            traj_dir = out_dir / "trajectories"
            traj_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None
        traj_dir = None

    # -------------------------------------------------------------------
    # Run all integrations
    # -------------------------------------------------------------------
    # Structure: results_by_ic[ic_key] = {
    #   "full": {times, states, status, metrics, dyn_class},
    #   "M=256": {...}, ...
    # }
    results_by_ic: Dict[str, Dict[str, Any]] = {}

    all_memory_modes = [("full", None)] + [
        (f"M={M}", M) for M in windows_steps
    ]

    for ic_key, x0 in ic_dict.items():
        results_by_ic[ic_key] = {}

        for mode_label, M_steps in all_memory_modes:
            if M_steps is None:
                mem_mode = "full"
                mem_window_len = None
            else:
                mem_mode = "window"
                mem_window_len = M_steps

            times, states, int_status = caputo_abm_integrate(
                rhs=rhs,
                x0=x0,
                q=q,
                h=h,
                t_final=t_final,
                divergence_norm=divergence_norm,
                memory_mode=mem_mode,
                memory_window_length=mem_window_len,
                use_c_backend=True,
            )

            dyn_class = classify_fractional_trajectory(
                times=times,
                states=states,
                t_burn=t_burn,
                divergence_norm=divergence_norm,
                collapse_variance_tolerance=collapse_var_tol,
                min_range_tolerance=min_range_tol,
            )

            metrics = compute_memory_metrics(times, states, t_burn)

            # Estimate derivative bound from RHS evaluated on trajectory
            try:
                f_norms = [
                    np.linalg.norm(rhs(0.0, states[i]))
                    for i in range(0, min(len(states), 500), max(1, len(states) // 500))
                ]
                derivative_bound = float(np.max(f_norms)) if f_norms else 1.0
            except Exception:
                derivative_bound = 1.0

            results_by_ic[ic_key][mode_label] = {
                "times": times,
                "states": states,
                "int_status": int_status,
                "dyn_class": dyn_class,
                "metrics": metrics,
                "M_steps": M_steps,
                "derivative_bound": derivative_bound,
            }

            if save_trajectories and traj_dir is not None:
                label_safe = mode_label.replace("=", "")
                np.save(traj_dir / f"{ic_key}_{label_safe}_times.npy", times)
                np.save(traj_dir / f"{ic_key}_{label_safe}_states.npy", states)

    # -------------------------------------------------------------------
    # Compute comparisons and tail bounds
    # -------------------------------------------------------------------
    comparison_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for ic_key, mode_results in results_by_ic.items():
        full_data = mode_results.get("full")
        full_metrics = full_data["metrics"] if full_data else None
        full_class = full_data["dyn_class"] if full_data else "inconclusive"

        for mode_label, run_data in mode_results.items():
            metrics = run_data["metrics"]
            dyn_class = run_data["dyn_class"]
            M_steps = run_data["M_steps"]
            derivative_bound = run_data["derivative_bound"]

            if M_steps is None:
                memory_time = t_final
                status = "full_memory_reference"
                cmp_result: Dict[str, Any] = {}
                tail_bound = 0.0
            else:
                memory_time = M_steps * h
                cmp_result = compare_window_to_full(
                    window_metrics=metrics,
                    full_metrics=full_metrics,
                    dynamic_class_window=dyn_class,
                    dynamic_class_full=full_class,
                    policy=policy,
                )
                status = cmp_result["status"]
                tail_bound = estimate_caputo_tail_bound(
                    q=q,
                    t_final=t_final,
                    t_burn=t_burn,
                    memory_window_steps=M_steps,
                    h=h,
                    derivative_bound=derivative_bound,
                )

            fs = metrics["final_state"]

            # Summary row
            summary_rows.append({
                "case_id": case_id,
                "system_id": system_id,
                "q": q,
                "initial_condition_id": ic_key,
                "memory_mode": "full" if M_steps is None else "window",
                "M": "" if M_steps is None else M_steps,
                "memory_time": memory_time if M_steps is not None else t_final,
                "h": h,
                "t_final": t_final,
                "t_burn": t_burn,
                "status": status,
                "dynamic_class": dyn_class,
                "final_state_x": fs[0] if len(fs) > 0 else float("nan"),
                "final_state_y": fs[1] if len(fs) > 1 else float("nan"),
                "final_state_z": fs[2] if len(fs) > 2 else float("nan"),
                "rho_attractor": metrics["rho_attractor"],
                "rho_max": metrics["rho_max"],
                "range_x": metrics["range_vector"][0] if len(metrics["range_vector"]) > 0 else float("nan"),
                "range_y": metrics["range_vector"][1] if len(metrics["range_vector"]) > 1 else float("nan"),
                "range_z": metrics["range_vector"][2] if len(metrics["range_vector"]) > 2 else float("nan"),
                "max_norm": metrics["max_norm"],
            })

            # Comparison row (all modes, for completeness)
            cmp_row: Dict[str, Any] = {
                "case_id": case_id,
                "initial_condition_id": ic_key,
                "M": "" if M_steps is None else M_steps,
                "memory_time": memory_time if M_steps is not None else t_final,
                "dynamic_class_full": full_class,
                "dynamic_class_window": dyn_class,
                "class_changed": cmp_result.get("class_changed", False) if M_steps is not None else False,
                "rho_relative_error": cmp_result.get("rho_relative_error", 0.0) if M_steps is not None else 0.0,
                "rho_max_relative_error": cmp_result.get("rho_max_relative_error", 0.0) if M_steps is not None else 0.0,
                "range_relative_error": cmp_result.get("range_relative_error", 0.0) if M_steps is not None else 0.0,
                "center_relative_error": cmp_result.get("center_relative_error", 0.0) if M_steps is not None else 0.0,
                "tail_bound_estimate": tail_bound,
                "warning": cmp_result.get("warning", False) if M_steps is not None else False,
                "status": status,
            }
            comparison_rows.append(cmp_row)

    return {
        "case_id": case_id,
        "system_id": system_id,
        "q": q,
        "t_final": t_final,
        "t_burn": t_burn,
        "windows_steps": windows_steps,
        "results_by_ic": results_by_ic,
        "summary_rows": summary_rows,
        "comparison_rows": comparison_rows,
    }


# ===========================================================================
# 9. _determine_overall_status
# ===========================================================================

def _determine_overall_status(comparison_rows: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Determine overall validation status from comparison rows.

    Parameters
    ----------
    comparison_rows : list of dict

    Returns
    -------
    overall_status : str
    automatic_warnings : list of str
    """
    automatic_warnings: List[str] = []

    # Separate full reference rows from window rows
    window_rows = [r for r in comparison_rows if r.get("M") != ""]

    if not window_rows:
        return "memory_validation_inconclusive", ["No window runs found."]

    # Check if full memory itself failed
    full_rows = [r for r in comparison_rows if r.get("M") == ""]
    full_classes = [r["dynamic_class_full"] for r in full_rows]
    bad_full_classes = {"nan_detected", "diverged", "too_short"}
    if full_classes and all(c in bad_full_classes for c in full_classes):
        automatic_warnings.append(
            "Full-memory reference trajectories all failed (nan/diverged/too_short). "
            "Cannot validate window approximations."
        )
        return "memory_validation_inconclusive", automatic_warnings

    # Collect per-window results
    any_insufficient = False
    any_sensitive = False
    all_windows_fail = True

    for row in window_rows:
        st = row.get("status", "")
        warn = row.get("warning", False)
        class_changed = row.get("class_changed", False)

        if st == "window_memory_insufficient" or class_changed:
            any_insufficient = True
            automatic_warnings.append(
                f"IC={row['initial_condition_id']}, M={row['M']}: "
                f"window_memory_insufficient — class changed "
                f"(full={row['dynamic_class_full']}, window={row['dynamic_class_window']})"
            )

        if st == "window_memory_sensitive" and not class_changed:
            any_sensitive = True
            automatic_warnings.append(
                f"IC={row['initial_condition_id']}, M={row['M']}: "
                f"window_memory_sensitive — metrics exceed tolerance"
            )

        if st in ("window_memory_sufficient", "window_memory_sensitive"):
            all_windows_fail = False

    # Find the largest window for each IC
    # If at least one large window is sufficient/sensitive and no class changes → passed
    # Determine status
    if all_windows_fail and any_insufficient:
        return "memory_validation_failed", automatic_warnings

    if any_insufficient:
        # Some windows change class → failed
        return "memory_validation_failed", automatic_warnings

    if any_sensitive:
        return "memory_validation_passed_with_sensitive_windows", automatic_warnings

    return "memory_validation_passed", automatic_warnings


# ===========================================================================
# 10. _write_csv
# ===========================================================================

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dicts to CSV."""
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# 11. run_fractional_memory_validation
# ===========================================================================

def run_fractional_memory_validation(
    config_path: str | Path,
    output_dir: str | Path,
    fast: bool = False,
    save_trajectories: bool = False,
) -> Dict[str, Any]:
    """Run memory validation for a single YAML config and write outputs.

    Parameters
    ----------
    config_path : str or Path
        Path to the memory validation YAML file.
    output_dir : str or Path
        Root output directory. Outputs go in <output_dir>/<case_id>/.
    fast : bool
        Use fast_test time parameters if True.
    save_trajectories : bool
        Save trajectory .npy files if True.

    Returns
    -------
    summary : dict
        memory_validation_summary content (also written as JSON).
    """
    config = load_memory_config(config_path)
    case_id = config["case_id"]
    system_id = config["system_id"]
    q = float(config["q"])

    memory_cfg = config["memory"]
    units = memory_cfg.get("units", "steps")
    window_entries = memory_cfg.get("windows", [])
    h = float(config["integrator"]["h"])
    windows_steps = [
        _resolve_window_steps(w["M"], units, h) for w in window_entries
    ]

    # Prepare output directory
    out_root = Path(output_dir)
    case_out = out_root / case_id
    case_out.mkdir(parents=True, exist_ok=True)

    # Run the case
    run_result = run_memory_case(
        config=config,
        fast=fast,
        output_dir=out_root,
        save_trajectories=save_trajectories,
    )

    summary_rows = run_result["summary_rows"]
    comparison_rows = run_result["comparison_rows"]

    # Write CSVs
    _write_csv(case_out / "memory_window_summary.csv", summary_rows)
    _write_csv(case_out / "memory_comparison.csv", comparison_rows)

    # Determine overall status
    overall_status, automatic_warnings = _determine_overall_status(comparison_rows)

    # Build JSON summary
    summary: Dict[str, Any] = {
        "stage": "fractional_memory_validation",
        "case_id": case_id,
        "system_id": system_id,
        "q": q,
        "integrator": "ABM",
        "full_memory_reference_present": bool(memory_cfg.get("full_reference", False)),
        "windows_tested": windows_steps,
        "overall_status": overall_status,
        "automatic_warnings": automatic_warnings,
        "pointwise_comparison_used": False,
        "hiddenness_certified_by_this_pipeline": False,
        "no_hidden_verified_claim": True,
    }

    summary_path = case_out / "memory_validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


# ===========================================================================
# 12. run_all_fractional_memory_validations
# ===========================================================================

def run_all_fractional_memory_validations(
    config_dir: str | Path,
    output_dir: str | Path,
    fast: bool = False,
    save_trajectories: bool = False,
) -> List[Dict[str, Any]]:
    """Run memory validation for all YAML configs in a directory.

    Parameters
    ----------
    config_dir : str or Path
        Directory containing *_memory.yaml files.
    output_dir : str or Path
        Root output directory.
    fast : bool
    save_trajectories : bool

    Returns
    -------
    summaries : list of dict
        One summary dict per config file.
    """
    config_dir = Path(config_dir)
    yaml_files = sorted(config_dir.glob("*_memory.yaml"))

    if not yaml_files:
        raise FileNotFoundError(
            f"No *_memory.yaml files found in: {config_dir}"
        )

    summaries = []
    for yaml_path in yaml_files:
        print(f"[fractional_memory_validation] Running: {yaml_path.name}")
        try:
            summary = run_fractional_memory_validation(
                config_path=yaml_path,
                output_dir=output_dir,
                fast=fast,
                save_trajectories=save_trajectories,
            )
            summaries.append(summary)
            print(
                f"  -> case_id={summary['case_id']}, "
                f"overall_status={summary['overall_status']}"
            )
        except Exception as exc:
            print(f"  [ERROR] {yaml_path.name}: {exc}")
            summaries.append({
                "stage": "fractional_memory_validation",
                "yaml_file": str(yaml_path),
                "overall_status": "memory_validation_inconclusive",
                "error": str(exc),
                "hiddenness_certified_by_this_pipeline": False,
                "no_hidden_verified_claim": True,
            })

    return summaries
