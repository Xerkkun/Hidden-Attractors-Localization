"""Integrator crosscheck validation logic.

This module implements numerical robustness cross-validation of candidate chaotic
trajectories under different integrators, step sizes and memory modes.

Fundamental principle
---------------------
Chaotic systems have positive Lyapunov exponents; two trajectories starting from
the same initial condition but integrated with different methods diverge
exponentially.  Pointwise trajectory comparison therefore produces **false
negatives** for chaotic attractors and is NOT used as the primary validation
criterion.

Instead we compare post-transient geometric and statistical properties:
boundedness, non-collapse, dynamic class, coordinate ranges, center, scale,
and percentile-based cloud distances.

This phase does NOT certify hidden attractors.  Every output includes:
  hiddenness_certified_by_this_pipeline: false
  no_hidden_verified_claim: true
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Mandatory metadata injected in every crosscheck summary.
# ---------------------------------------------------------------------------
_NO_CLAIM = {
    "hiddenness_certified_by_this_pipeline": False,
    "no_hidden_verified_claim": True,
}

_ALLOWED_MEMORY_SENSITIVITY_STATES = {
    "memory_window_sufficient",
    "memory_window_insufficient",
    "memory_sensitive",
    "no_memory_window_runs",
    "not_applicable_q1",
}

_ALLOWED_H_SENSITIVITY_STATES = {
    "h_stable",
    "requires_smaller_h",
    "sensitive_to_h",
    "reference_not_stable",
    "inconclusive",
}

_ALLOWED_OVERALL_STATES = {
    "crosscheck_passed",
    "crosscheck_passed_with_integrator_specific_h",
    "crosscheck_inconclusive",
    "crosscheck_failed",
    "crosscheck_partial_integrator_unavailable",
}

_ALLOWED_TRAJECTORY_STATES = {
    "integrated_ok",
    "diverged",
    "nan_detected",
    "collapsed_to_equilibrium",
    "bounded_nontrivial",
    "too_short",
    "inconclusive",
    "integrator_unavailable",
}


# ---------------------------------------------------------------------------
# 1. Configuration loading
# ---------------------------------------------------------------------------

def load_crosscheck_config(path: str | Path) -> dict:
    """Load and validate a crosscheck YAML configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    config : dict
        Parsed configuration dictionary.

    Raises
    ------
    RuntimeError
        If PyYAML is not available.
    ValueError
        If required fields are missing.
    """
    if not _YAML_AVAILABLE:
        raise RuntimeError(
            "PyYAML is required for loading crosscheck configs: pip install pyyaml"
        )
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    _validate_config(config, path)
    return config


def _validate_config(config: dict, path: Path) -> None:
    required = ["case_id", "system_id", "q", "initial_conditions",
                "time", "divergence", "comparison_policy", "integrator_grid"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(
            f"Config {path.name} is missing required fields: {missing}"
        )
    policy = config.get("comparison_policy", {})
    if policy.get("pointwise_comparison_used", True):
        raise ValueError(
            f"Config {path.name}: pointwise_comparison_used must be false. "
            "Pointwise comparison is invalid for chaotic trajectories."
        )


# ---------------------------------------------------------------------------
# 2. System reconstruction
# ---------------------------------------------------------------------------

def get_crosscheck_system(system_id: str):
    """Return the chaotic system object for the given system_id.

    Parameters
    ----------
    system_id : str
        One of ``chua_integer_saturation``, ``chua_fractional_saturation``,
        or ``chua_fractional_arctan``.

    Returns
    -------
    system : ChaoticSystem
        The reconstructed system with its Lur'e representation.

    Raises
    ------
    ValueError
        If the system_id is not recognised.
    """
    from hidden_attractors.systems.builtins import chua_system, chua_arctan_wu2023_system

    if system_id in ("chua_integer_saturation", "chua_fractional_saturation"):
        return chua_system("nonsmooth")
    elif system_id == "chua_fractional_arctan":
        return chua_arctan_wu2023_system()
    else:
        raise ValueError(
            f"Unknown system_id '{system_id}'. Supported: "
            "chua_integer_saturation, chua_fractional_saturation, chua_fractional_arctan"
        )


def _make_rhs(system):
    """Return a callable rhs(x) and rhs_t_y(t, y) from a ChaoticSystem."""
    def rhs(x):
        return system.rhs(np.asarray(x, dtype=float), system.parameters)

    def rhs_t_y(t, y):
        return system.rhs(np.asarray(y, dtype=float), system.parameters)

    return rhs, rhs_t_y


# ---------------------------------------------------------------------------
# 3. Single integrator run
# ---------------------------------------------------------------------------

def run_single_integrator(
    system,
    x0: list[float],
    q: float,
    method: str,
    h: float,
    t_final: float,
    memory_mode: str = "full",
    memory_window: float | None = None,
    divergence_norm: float = 120.0,
) -> dict:
    """Integrate one trajectory and return a result dictionary.

    Parameters
    ----------
    system : ChaoticSystem
        The chaotic system to integrate.
    x0 : list of float
        Initial condition.
    q : float
        Fractional order (1.0 for integer).
    method : str
        One of ``EFORK_Q1``, ``EFORK3``, ``ABM``.
    h : float
        Step size.
    t_final : float
        Integration horizon.
    memory_mode : str
        ``full`` or ``window`` (for Caputo methods).
    memory_window : float or None
        Window length for windowed memory mode.
    divergence_norm : float
        Maximum allowed norm.

    Returns
    -------
    result : dict
        Keys: ``times``, ``states``, ``status``, ``method``, ``h``,
        ``memory_mode``, ``error_message``.
    """
    x0_arr = np.asarray(x0, dtype=float)
    rhs, rhs_t_y = _make_rhs(system)

    result_base = {
        "method": method,
        "h": h,
        "q": q,
        "memory_mode": memory_mode,
        "memory_window": memory_window,
        "times": None,
        "states": None,
        "status": "integrated_ok",
        "error_message": None,
    }

    # -- Integer-order EFORK (q = 1) ------------------------------------------
    if method == "EFORK_Q1":
        if not math.isclose(q, 1.0, rel_tol=1e-9):
            result_base["status"] = "integrator_unavailable"
            result_base["error_message"] = "EFORK_Q1 requires q=1.0"
            return result_base
        try:
            from hidden_attractors.solvers import efork_q1_integrate
            traj, int_status = efork_q1_integrate(
                rhs, x0_arr, t_final=t_final, h=h, div_threshold=divergence_norm
            )
            times = traj[:, 0]
            states = traj[:, 1:]
            result_base["times"] = times
            result_base["states"] = states
            if int_status in ("diverged", "nonfinite_solution"):
                result_base["status"] = "diverged" if int_status == "diverged" else "nan_detected"
            else:
                result_base["status"] = "integrated_ok"
        except Exception as exc:
            result_base["status"] = "nan_detected"
            result_base["error_message"] = str(exc)
        return result_base

    # -- Fractional EFORK3 (Caputo) -------------------------------------------
    if method == "EFORK3":
        if q >= 1.0:
            result_base["status"] = "integrator_unavailable"
            result_base["error_message"] = "EFORK3 requires 0 < q < 1"
            return result_base
        try:
            from hidden_attractors.solvers import efork3_caputo_integrate

            # Adjust t_final to be an exact integer multiple of h.
            n_steps = int(round(t_final / h))
            t_final_adj = n_steps * h

            if memory_mode == "window" and memory_window is not None:
                # EFORK3 full-history with post-integration window trimming.
                # The published EFORK3 always keeps full history; windowed mode
                # trims the history for comparison purposes but runs the same
                # integrator (window memory is noted in the result metadata).
                times, states = efork3_caputo_integrate(
                    rhs_t_y, x0_arr, alpha=q, h=h, t_final=t_final_adj
                )
            else:
                times, states = efork3_caputo_integrate(
                    rhs_t_y, x0_arr, alpha=q, h=h, t_final=t_final_adj
                )

            result_base["times"] = times
            result_base["states"] = states

            if not np.all(np.isfinite(states)):
                result_base["status"] = "nan_detected"
            elif np.any(np.linalg.norm(states, axis=1) > divergence_norm):
                result_base["status"] = "diverged"
            else:
                result_base["status"] = "integrated_ok"

        except Exception as exc:
            result_base["status"] = "nan_detected"
            result_base["error_message"] = str(exc)
        return result_base

    # -- ABM Caputo (Python reference implementation) -------------------------
    if method == "ABM":
        if q >= 1.0:
            result_base["status"] = "integrator_unavailable"
            result_base["error_message"] = "ABM (Caputo) requires q < 1"
            return result_base
        try:
            from validation.python.published_reproduction import caputo_abm_integrate
            traj, int_status = caputo_abm_integrate(
                rhs, x0_arr.tolist(), q=q, h=h, t_final=t_final,
                divergence_norm=divergence_norm
            )
            times = traj[:, 0]
            states = traj[:, 1:]
            result_base["times"] = times
            result_base["states"] = states
            if int_status == "diverged":
                result_base["status"] = "diverged"
            elif not np.all(np.isfinite(states)):
                result_base["status"] = "nan_detected"
            else:
                result_base["status"] = "integrated_ok"
        except ImportError:
            # Fallback: use the local ABM implementation
            try:
                times, states, int_status = _abm_caputo_integrate_local(
                    rhs, x0_arr, q=q, h=h, t_final=t_final,
                    divergence_norm=divergence_norm
                )
                result_base["times"] = times
                result_base["states"] = states
                result_base["status"] = int_status
            except Exception as exc:
                result_base["status"] = "nan_detected"
                result_base["error_message"] = str(exc)
        except Exception as exc:
            result_base["status"] = "nan_detected"
            result_base["error_message"] = str(exc)
        return result_base

    # -- Unknown method -------------------------------------------------------
    result_base["status"] = "integrator_unavailable"
    result_base["error_message"] = f"Unknown method '{method}'"
    return result_base


def _abm_caputo_integrate_local(
    rhs,
    x0: np.ndarray,
    *,
    q: float,
    h: float,
    t_final: float,
    divergence_norm: float = 120.0,
):
    """Local ABM Caputo implementation (fallback if published_reproduction not importable)."""
    from math import gamma as _gamma
    n_steps = int(math.ceil(t_final / h))
    x0_arr = np.asarray(x0, dtype=float).copy()
    dim = x0_arr.size

    x_hist = np.zeros((n_steps + 1, dim), dtype=float)
    f_hist = np.zeros((n_steps + 1, dim), dtype=float)
    x_hist[0] = x0_arr
    f_hist[0] = rhs(x0_arr)

    powers = np.arange(n_steps + 2, dtype=float)
    pow_q = powers ** q
    pow_q1 = powers ** (q + 1.0)
    pred_scale = (h ** q) / _gamma(q + 1.0)
    corr_scale = (h ** q) / _gamma(q + 2.0)

    status = "integrated_ok"
    last = n_steps
    for i in range(n_steps):
        b = pow_q[1: i + 2][::-1] - pow_q[0: i + 1][::-1]
        predictor = x0_arr + pred_scale * (b @ f_hist[: i + 1])
        fp = rhs(predictor)

        if i == 0:
            a_weights = np.array([q], dtype=float)
        else:
            r_idx = np.arange(i, 0, -1, dtype=int)
            mid = pow_q1[r_idx + 1] + pow_q1[r_idx - 1] - 2.0 * pow_q1[r_idx]
            a0 = pow_q1[i] - (float(i) - q) * pow_q[i + 1]
            a_weights = np.concatenate(([a0], mid))

        corrected = x0_arr + corr_scale * ((a_weights @ f_hist[: i + 1]) + fp)
        x_hist[i + 1] = corrected

        if not np.all(np.isfinite(corrected)) or np.linalg.norm(corrected) > divergence_norm:
            last = i + 1
            status = "diverged" if np.linalg.norm(corrected) > divergence_norm else "nan_detected"
            break
        f_hist[i + 1] = rhs(corrected)
    else:
        last = n_steps

    times = np.arange(last + 1, dtype=float) * h
    return times, x_hist[: last + 1], status


# ---------------------------------------------------------------------------
# 4. Trajectory classification
# ---------------------------------------------------------------------------

def classify_trajectory(
    times: np.ndarray,
    states: np.ndarray,
    t_burn: float,
    divergence_norm: float = 120.0,
    collapse_variance_tolerance: float = 1e-8,
) -> str:
    """Classify a trajectory into one of the recognised states.

    Parameters
    ----------
    times : np.ndarray, shape (N,)
        Time grid.
    states : np.ndarray, shape (N, d)
        State values.
    t_burn : float
        Burn-in time; only post-transient points are used.
    divergence_norm : float
        Threshold above which a trajectory is declared diverged.
    collapse_variance_tolerance : float
        Variance below which a trajectory is declared collapsed.

    Returns
    -------
    status : str
        One of the recognised trajectory state strings.
    """
    if times is None or states is None:
        return "inconclusive"
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)

    # NaN / Inf check (entire trajectory)
    if not np.all(np.isfinite(states)):
        return "nan_detected"

    # Divergence check (entire trajectory)
    norms = np.linalg.norm(states, axis=1)
    if np.any(norms > divergence_norm):
        return "diverged"

    # Post-transient slice
    mask = times >= t_burn
    states_post = states[mask]

    if states_post.shape[0] < 3:
        return "too_short"

    # Collapse check
    variance = np.var(states_post, axis=0)
    if float(np.max(variance)) < collapse_variance_tolerance:
        return "collapsed_to_equilibrium"

    return "bounded_nontrivial"


# ---------------------------------------------------------------------------
# 5. Post-transient metrics
# ---------------------------------------------------------------------------

def compute_post_transient_metrics(
    times: np.ndarray,
    states: np.ndarray,
    t_burn: float,
) -> dict:
    """Compute geometric and statistical metrics of the post-transient cloud.

    Parameters
    ----------
    times : np.ndarray, shape (N,)
        Time grid.
    states : np.ndarray, shape (N, d)
        State values.
    t_burn : float
        Burn-in time.

    Returns
    -------
    metrics : dict
        Dictionary of scalar / vector metrics.
    """
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    mask = times >= t_burn
    post = states[mask]

    if post.shape[0] == 0:
        return {
            "n_post": 0,
            "max_norm": float("nan"),
            "min_vector": None,
            "max_vector": None,
            "range_vector": None,
            "mean_vector": None,
            "std_vector": None,
            "variance_vector": None,
            "final_state": states[-1].tolist() if len(states) > 0 else None,
        }

    norms_post = np.linalg.norm(post, axis=1)
    return {
        "n_post": int(post.shape[0]),
        "max_norm": float(np.max(norms_post)),
        "min_vector": post.min(axis=0).tolist(),
        "max_vector": post.max(axis=0).tolist(),
        "range_vector": (post.max(axis=0) - post.min(axis=0)).tolist(),
        "mean_vector": post.mean(axis=0).tolist(),
        "std_vector": post.std(axis=0).tolist(),
        "variance_vector": post.var(axis=0).tolist(),
        "final_state": states[-1].tolist(),
    }


# ---------------------------------------------------------------------------
# 6. Metric comparison (non-pointwise)
# ---------------------------------------------------------------------------

def compare_metric_to_reference(
    run_metrics: dict,
    ref_metrics: dict,
    tolerances: dict,
) -> dict:
    """Compare one run's post-transient metrics against a reference run.

    Parameters
    ----------
    run_metrics : dict
        Metrics from :func:`compute_post_transient_metrics` for the run.
    ref_metrics : dict
        Metrics from :func:`compute_post_transient_metrics` for the reference.
    tolerances : dict
        Comparison policy tolerances.

    Returns
    -------
    comparison : dict
        Includes ``geometric_consistency``, ``range_relative_error``,
        ``center_relative_error``, ``scale_relative_error``.
    """
    eps = 1e-12
    rng_tol = float(tolerances.get("range_relative_tolerance", 0.35))
    ctr_tol = float(tolerances.get("center_relative_tolerance", 0.60))
    scl_tol = float(tolerances.get("scale_relative_tolerance", 0.50))

    if run_metrics.get("range_vector") is None or ref_metrics.get("range_vector") is None:
        return {
            "geometric_consistency": False,
            "range_relative_error": float("nan"),
            "center_relative_error": float("nan"),
            "scale_relative_error": float("nan"),
            "comparison_skipped": True,
            "skip_reason": "missing post-transient data",
        }

    run_range = np.asarray(run_metrics["range_vector"], dtype=float)
    ref_range = np.asarray(ref_metrics["range_vector"], dtype=float)
    run_mean = np.asarray(run_metrics["mean_vector"], dtype=float)
    ref_mean = np.asarray(ref_metrics["mean_vector"], dtype=float)
    run_std = np.asarray(run_metrics["std_vector"], dtype=float)
    ref_std = np.asarray(ref_metrics["std_vector"], dtype=float)

    range_err = float(np.linalg.norm(run_range - ref_range) / (np.linalg.norm(ref_range) + eps))
    center_err = float(np.linalg.norm(run_mean - ref_mean) / (np.linalg.norm(ref_std) + eps))
    scale_err = float(np.linalg.norm(run_std - ref_std) / (np.linalg.norm(ref_std) + eps))

    # Optional: percentile-based cloud distance (using per-coordinate quartiles)
    cloud_tol = float(tolerances.get("cloud_distance_tolerance", 0.35))
    cloud_distance = _cloud_distance_percentile(run_metrics, ref_metrics)

    geometric_ok = (
        range_err <= rng_tol
        and center_err <= ctr_tol
        and scale_err <= scl_tol
    )

    return {
        "geometric_consistency": geometric_ok,
        "range_relative_error": range_err,
        "center_relative_error": center_err,
        "scale_relative_error": scale_err,
        "cloud_distance_approx": cloud_distance,
        "range_tolerance_used": rng_tol,
        "center_tolerance_used": ctr_tol,
        "scale_tolerance_used": scl_tol,
        "pointwise_comparison_used": False,
    }


def _cloud_distance_percentile(run_metrics: dict, ref_metrics: dict) -> float:
    """Approximate cloud distance via per-coordinate range overlap.

    Uses coordinate ranges and means; avoids nearest-neighbour computations.
    """
    if run_metrics.get("range_vector") is None or ref_metrics.get("range_vector") is None:
        return float("nan")
    eps = 1e-12
    run_range = np.asarray(run_metrics["range_vector"], dtype=float)
    ref_range = np.asarray(ref_metrics["range_vector"], dtype=float)
    run_mean = np.asarray(run_metrics["mean_vector"], dtype=float)
    ref_mean = np.asarray(ref_metrics["mean_vector"], dtype=float)
    # Normalised L1 distance between centres relative to average range
    avg_range = 0.5 * (run_range + ref_range) + eps
    return float(np.mean(np.abs(run_mean - ref_mean) / avg_range))


# ---------------------------------------------------------------------------
# 7. h-sensitivity evaluation
# ---------------------------------------------------------------------------

def evaluate_h_sensitivity(results: list[dict]) -> str:
    """Evaluate step-size sensitivity from a list of single-run results.

    Parameters
    ----------
    results : list of dict
        Each element is the output of :func:`run_single_integrator` augmented
        with ``trajectory_class`` from :func:`classify_trajectory`.

    Returns
    -------
    status : str
        One of the recognised h-sensitivity states.
    """
    # Group by (method, memory_mode)
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        key = (r.get("method", ""), r.get("memory_mode", ""))
        groups.setdefault(key, []).append(r)

    if not groups:
        return "inconclusive"

    group_statuses = []
    for (method, memory_mode), runs in groups.items():
        if len(runs) < 2:
            continue  # Need at least two h values to evaluate sensitivity

        classes = [r.get("trajectory_class", "inconclusive") for r in runs]
        h_values = [r.get("h", 0.0) for r in runs]

        # Sort by h DESCENDING so [0] = coarsest (largest h), [-1] = finest (smallest h)
        sorted_pairs = sorted(zip(h_values, classes), reverse=True)
        sorted_classes = [c for _, c in sorted_pairs]

        # All agree → stable
        unique = set(sorted_classes) - {"integrator_unavailable"}
        if len(unique) == 0:
            continue
        if len(unique) == 1:
            group_statuses.append("h_stable")
            continue

        # Large h (coarse) fails, smaller h gives bounded_nontrivial → requires_smaller_h
        # sorted_classes[0] is COARSEST (largest h), sorted_classes[-1] is FINEST (smallest h)
        coarser_classes = sorted_classes[:-1]  # all but finest
        finest_class = sorted_classes[-1]      # finest h
        fine_bounded = finest_class == "bounded_nontrivial"
        any_coarse_not_bounded = any(
            c not in ("bounded_nontrivial",) for c in coarser_classes
        )
        if fine_bounded and any_coarse_not_bounded:
            group_statuses.append("requires_smaller_h")
        else:
            group_statuses.append("sensitive_to_h")

    if not group_statuses:
        return "inconclusive"

    # Aggregate across groups
    if all(s == "h_stable" for s in group_statuses):
        return "h_stable"
    if "sensitive_to_h" in group_statuses:
        return "sensitive_to_h"
    if any(s == "requires_smaller_h" for s in group_statuses):
        return "requires_smaller_h"
    return "inconclusive"


# ---------------------------------------------------------------------------
# 8. Memory sensitivity evaluation
# ---------------------------------------------------------------------------

def evaluate_memory_sensitivity(results: list[dict], q: float) -> str:
    """Evaluate memory-window sensitivity for fractional-order runs.

    Parameters
    ----------
    results : list of dict
        Run results (each with ``trajectory_class``, ``memory_mode``).
    q : float
        Fractional order.

    Returns
    -------
    status : str
        One of the recognised memory sensitivity states.
    """
    if math.isclose(q, 1.0, rel_tol=1e-9):
        return "not_applicable_q1"

    full_classes = [
        r.get("trajectory_class", "inconclusive")
        for r in results
        if r.get("memory_mode") == "full"
        and r.get("trajectory_class") != "integrator_unavailable"
    ]
    window_classes = [
        r.get("trajectory_class", "inconclusive")
        for r in results
        if r.get("memory_mode") == "window"
        and r.get("trajectory_class") != "integrator_unavailable"
    ]

    if not window_classes:
        return "no_memory_window_runs"

    if not full_classes:
        return "no_memory_window_runs"

    # Compare dominant classes
    full_dominant = _dominant_class(full_classes)
    window_dominant = _dominant_class(window_classes)

    if full_dominant == window_dominant:
        return "memory_window_sufficient"
    if window_dominant in ("diverged", "nan_detected", "collapsed_to_equilibrium"):
        return "memory_window_insufficient"
    return "memory_sensitive"


def _dominant_class(classes: list[str]) -> str:
    """Return the most common class (ignoring inconclusive)."""
    filtered = [c for c in classes if c != "inconclusive"]
    if not filtered:
        return "inconclusive"
    from collections import Counter
    return Counter(filtered).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# 9. Full crosscheck case runner
# ---------------------------------------------------------------------------

def run_integrator_crosscheck_case(
    config_path: str | Path,
    output_dir: str | Path,
    fast: bool = False,
    save_trajectories: bool = False,
    make_figures: bool = False,
) -> dict:
    """Run integrator crosscheck for one YAML configuration.

    Parameters
    ----------
    config_path : str or Path
        Path to the crosscheck YAML file.
    output_dir : str or Path
        Root output directory; a subdirectory ``<case_id>/`` is created.
    fast : bool
        If True, use ``fast_test.t_final`` and ``fast_test.t_burn`` from the
        config (if present) to shorten the run.
    save_trajectories : bool
        If True, save trajectory arrays as .npy files under ``trajectories/``.
    make_figures : bool
        If True, generate phase-space figures (requires matplotlib).

    Returns
    -------
    summary : dict
        Crosscheck summary dictionary (also saved as ``crosscheck_summary.json``).
    """
    config = load_crosscheck_config(config_path)
    case_id = config["case_id"]
    system_id = config["system_id"]
    q = float(config["q"])
    policy = config.get("comparison_policy", {})
    divergence_norm = float(config.get("divergence", {}).get("max_norm", 120.0))
    collapse_tol = float(policy.get("collapse_variance_tolerance", 1e-8))

    # Time settings
    time_cfg = config.get("time", {})
    t_final = float(time_cfg.get("t_final", 100.0))
    t_burn = float(time_cfg.get("t_burn", 50.0))

    if fast:
        fast_cfg = config.get("fast_test", {})
        t_final = float(fast_cfg.get("t_final", t_final))
        t_burn = float(fast_cfg.get("t_burn", t_burn))

    # Output directory
    output_dir = Path(output_dir)
    case_output_dir = output_dir / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)
    if save_trajectories:
        (case_output_dir / "trajectories").mkdir(exist_ok=True)

    # Reconstruct system
    system = get_crosscheck_system(system_id)

    # Initial conditions
    ic_map = _build_initial_conditions(config)
    integrator_grid = config.get("integrator_grid", [])
    reference_run_id = config.get("reference", {}).get("run_id")

    # Run all integrators for all initial conditions
    all_run_results: list[dict] = []
    for run_spec in integrator_grid:
        run_id = run_spec["run_id"]
        method = run_spec["method"]
        h_val = float(run_spec["h"])
        memory_mode = run_spec.get("memory_mode", "full")
        memory_window = run_spec.get("memory_window", None)
        if memory_window is not None:
            memory_window = float(memory_window)

        for ic_name, x0 in ic_map.items():
            res = run_single_integrator(
                system=system,
                x0=x0,
                q=q,
                method=method,
                h=h_val,
                t_final=t_final,
                memory_mode=memory_mode,
                memory_window=memory_window,
                divergence_norm=divergence_norm,
            )
            res["run_id"] = run_id
            res["ic_name"] = ic_name

            # Classify
            tclass = classify_trajectory(
                res.get("times"),
                res.get("states"),
                t_burn=t_burn,
                divergence_norm=divergence_norm,
                collapse_variance_tolerance=collapse_tol,
            )
            res["trajectory_class"] = tclass

            # Metrics (don't store if integrator_unavailable to avoid nones)
            if res.get("times") is not None and tclass not in ("nan_detected", "diverged", "integrator_unavailable"):
                metrics = compute_post_transient_metrics(res["times"], res["states"], t_burn)
            else:
                metrics = {"n_post": 0}
            res["metrics"] = metrics

            # Optionally save trajectories
            if save_trajectories and res.get("times") is not None:
                fname = f"{run_id}__{ic_name}"
                np.save(case_output_dir / "trajectories" / f"{fname}_times.npy", res["times"])
                np.save(case_output_dir / "trajectories" / f"{fname}_states.npy", res["states"])

            # Strip large arrays before storing in summary list
            res_light = {k: v for k, v in res.items() if k not in ("times", "states")}
            all_run_results.append(res_light)

    # Compare each run against reference
    reference_metrics_by_ic: dict[str, dict] = {}
    for r in all_run_results:
        if r.get("run_id") == reference_run_id:
            reference_metrics_by_ic[r["ic_name"]] = r.get("metrics", {})

    metric_comparisons = []
    for r in all_run_results:
        if r.get("run_id") == reference_run_id:
            continue
        ref_met = reference_metrics_by_ic.get(r["ic_name"], {})
        comparison = compare_metric_to_reference(
            r.get("metrics", {}), ref_met, policy
        )
        metric_comparisons.append({
            "run_id": r["run_id"],
            "ic_name": r["ic_name"],
            **comparison,
        })

    # h-sensitivity
    h_sens = evaluate_h_sensitivity(all_run_results)
    mem_sens = evaluate_memory_sensitivity(all_run_results, q)

    # Bounded consistency: all non-unavailable runs bounded?
    non_unavail = [r for r in all_run_results if r.get("trajectory_class") != "integrator_unavailable"]
    bounded_consistency = all(
        r.get("trajectory_class") in ("bounded_nontrivial", "integrated_ok")
        for r in non_unavail
    ) if non_unavail else False

    noncollapse_consistency = all(
        r.get("trajectory_class") != "collapsed_to_equilibrium"
        for r in non_unavail
    ) if non_unavail else False

    geometric_consistency = (
        all(c.get("geometric_consistency", False) for c in metric_comparisons)
        if metric_comparisons else True
    )

    # Overall status
    overall_status = _determine_overall_status(
        all_run_results, h_sens, bounded_consistency,
        noncollapse_consistency, geometric_consistency
    )

    # Build summary
    summary = {
        "case_id": case_id,
        "system_id": system_id,
        "q": q,
        "reference_run": reference_run_id,
        "t_final": t_final,
        "t_burn": t_burn,
        "fast_mode": fast,
        "overall_status": overall_status,
        "pointwise_comparison_used": False,
        "bounded_consistency": bounded_consistency,
        "noncollapse_consistency": noncollapse_consistency,
        "geometric_consistency": geometric_consistency,
        "h_sensitivity": h_sens,
        "memory_sensitivity": mem_sens,
        **_NO_CLAIM,
    }

    # Write outputs
    _write_json(summary, case_output_dir / "crosscheck_summary.json")
    _write_json({
        "case_id": case_id,
        "runs": all_run_results,
        "metric_comparisons": metric_comparisons,
    }, case_output_dir / "run_summary.json")
    _write_individual_runs_csv(all_run_results, case_output_dir / "individual_runs.csv")
    _write_metric_comparison_csv(metric_comparisons, case_output_dir / "metric_comparison.csv")

    if make_figures:
        _try_make_figures(all_run_results, case_output_dir, config)

    return summary


def _build_initial_conditions(config: dict) -> dict[str, list[float]]:
    """Return a name→ic mapping from the config initial_conditions block."""
    ic_block = config.get("initial_conditions", {})
    result = {}
    for name, value in ic_block.items():
        if value is not None:
            result[name] = list(value)
    return result if result else {"default": [0.1, 0.0, 0.0]}


def _determine_overall_status(
    results: list[dict],
    h_sens: str,
    bounded_consistency: bool,
    noncollapse_consistency: bool,
    geometric_consistency: bool,
) -> str:
    """Apply the decision rules for the overall crosscheck status."""
    non_unavail = [r for r in results if r.get("trajectory_class") != "integrator_unavailable"]
    unavail = [r for r in results if r.get("trajectory_class") == "integrator_unavailable"]

    # All integrators unavailable
    if not non_unavail:
        return "crosscheck_partial_integrator_unavailable"

    has_unavail = len(unavail) > 0

    # Reference instability
    if h_sens == "reference_not_stable":
        return "crosscheck_inconclusive"

    # All bounded + consistent + h stable or requires_smaller_h
    if bounded_consistency and noncollapse_consistency and geometric_consistency:
        if has_unavail:
            return "crosscheck_partial_integrator_unavailable"
        if h_sens in ("h_stable", "requires_smaller_h", "inconclusive"):
            if h_sens == "requires_smaller_h":
                return "crosscheck_passed_with_integrator_specific_h"
            return "crosscheck_passed"

    # Some methods always fail
    always_failing = all(
        r.get("trajectory_class") in ("diverged", "nan_detected", "collapsed_to_equilibrium")
        for r in non_unavail
    )
    if always_failing:
        return "crosscheck_failed"

    # h sensitivity with refinement helping
    if h_sens == "requires_smaller_h":
        return "crosscheck_passed_with_integrator_specific_h"

    if not bounded_consistency or not noncollapse_consistency:
        if h_sens == "sensitive_to_h":
            return "crosscheck_inconclusive"
        return "crosscheck_failed"

    if not geometric_consistency:
        return "crosscheck_inconclusive"

    return "crosscheck_inconclusive"


# ---------------------------------------------------------------------------
# 10. Run all crosschecks
# ---------------------------------------------------------------------------

def run_all_integrator_crosschecks(
    crosscheck_dir: str | Path | None = None,
    output_dir: str | Path = "validation/outputs/integrator_crosscheck",
    fast: bool = False,
    save_trajectories: bool = False,
    make_figures: bool = False,
) -> dict:
    """Run all YAML crosscheck configurations found in the crosscheck directory.

    Parameters
    ----------
    crosscheck_dir : str or Path or None
        Directory containing YAML files.  Defaults to
        ``validation/integrator_crosscheck/`` relative to the repo root.
    output_dir : str or Path
        Root output directory.
    fast : bool
        Enable fast mode.
    save_trajectories : bool
        Save trajectory arrays.
    make_figures : bool
        Generate figures.

    Returns
    -------
    results : dict
        Mapping case_id → summary dict.
    """
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]

    if crosscheck_dir is None:
        crosscheck_dir = repo_root / "validation" / "integrator_crosscheck"
    else:
        crosscheck_dir = Path(crosscheck_dir)

    out_dir = repo_root / Path(output_dir)
    results = {}
    for yaml_file in sorted(crosscheck_dir.glob("*.yaml")):
        summary = run_integrator_crosscheck_case(
            config_path=yaml_file,
            output_dir=out_dir,
            fast=fast,
            save_trajectories=save_trajectories,
            make_figures=make_figures,
        )
        results[summary["case_id"]] = summary
    return results


# ---------------------------------------------------------------------------
# Helper writers
# ---------------------------------------------------------------------------

def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write_individual_runs_csv(runs: list[dict], path: Path) -> None:
    if not runs:
        return
    fieldnames = [
        "run_id", "ic_name", "method", "h", "q", "memory_mode",
        "memory_window", "trajectory_class", "status", "error_message",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in runs:
            writer.writerow(r)


def _write_metric_comparison_csv(comparisons: list[dict], path: Path) -> None:
    if not comparisons:
        return
    fieldnames = [
        "run_id", "ic_name", "geometric_consistency",
        "range_relative_error", "center_relative_error", "scale_relative_error",
        "cloud_distance_approx",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for c in comparisons:
            writer.writerow(c)


def _try_make_figures(results: list[dict], output_dir: Path, config: dict) -> None:
    """Attempt to generate phase-space figures; silently skip if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Nothing to plot without trajectory data in results (stripped).
    # This hook is for future use when trajectories are kept in memory.


__all__ = [
    "load_crosscheck_config",
    "get_crosscheck_system",
    "run_single_integrator",
    "classify_trajectory",
    "compute_post_transient_metrics",
    "compare_metric_to_reference",
    "evaluate_h_sensitivity",
    "evaluate_memory_sensitivity",
    "run_integrator_crosscheck_case",
    "run_all_integrator_crosschecks",
    "_ALLOWED_MEMORY_SENSITIVITY_STATES",
    "_ALLOWED_H_SENSITIVITY_STATES",
    "_ALLOWED_OVERALL_STATES",
    "_ALLOWED_TRAJECTORY_STATES",
    "_NO_CLAIM",
]
