"""Continuation memory validation for Caputo fractional-order Chua systems.

Phase D: validates numerical continuation in [eta=0, eta=1] by comparing:
  - last_point_restart: restart from final state only (Caputo history reset)
  - history_window_transport: carry discrete history H_k and recompute RHS
    samples under the new field F_{eta+1} at each eta transition

Mathematical context
--------------------
For 0 < q < 1, the Caputo derivative is:

    ^C D_t^q X(t) = 1/Gamma(1-q) * integral_{t0}^{t} (t-tau)^(-q) X'(tau) dtau

The deformed continuation system at parameter eta is:

    ^C D_t^q X = P X + b [ k*sigma + eta*(psi(sigma) - k*sigma) ]

with sigma = r^T X.

At eta=0: X'=P*X + b*k*sigma  (linearised auxiliary system)
At eta=1: X'=P*X + b*psi(sigma) (original Chua nonlinearity)

History transport
-----------------
When eta_i -> eta_{i+1}, the discrete history:
    H_k = {X(t_{k-M}), ..., X(t_k)}
is kept, but the RHS samples must be recomputed:
    f_j^new = F_{eta_{i+1}}(X_j)
because the vector field changes with eta.

The caputo_abm_integrate function already accepts history_times and
history_states, so no re-implementation of ABM is needed.

No claims
---------
This phase does NOT certify:
- Hidden attractors (hidden_verified is never set)
- Chaos or Lyapunov exponents (chaos_certified_by_this_pipeline: false)
- That history_window_transport reproduces exact full Caputo memory
- That last_point_restart is correct for Caputo

References
----------
- Caputo (1967): original derivative definition.
- Guan & Xie (2025): review on hidden attractor localization; continuation
  methods are important but not automatic or universal.
- Yoon & You (2017), arXiv:1711.10071: Caputo memory reduction.
- Hai et al. (2020), arXiv:2007.05755: short-memory fractional DEs.
- Danca & Feckan (2024), arXiv:2406.04686: memory principle in Caputo codes.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional YAML import
# ---------------------------------------------------------------------------
try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _yaml = None  # type: ignore[assignment]
    _YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Reuse Phase-C functions (no re-implementation)
# ---------------------------------------------------------------------------
from validation.python.fractional_memory_validation import (
    classify_fractional_trajectory,
    compute_memory_metrics,
)

# ---------------------------------------------------------------------------
# Mandatory no-claim metadata
# ---------------------------------------------------------------------------
_NO_CLAIM: Dict[str, Any] = {
    "hiddenness_certified_by_this_pipeline": False,
    "chaos_certified_by_this_pipeline": False,
    "no_hidden_verified_claim": True,
    "pointwise_comparison_used": False,
}

_ALLOWED_DYN_CLASSES = {
    "nan_detected",
    "diverged",
    "too_short",
    "collapsed_to_equilibrium",
    "bounded_nontrivial",
    "periodic_candidate",
    "chaotic_candidate_by_geometry",
    "inconclusive",
}

_ALLOWED_ETA_REFINEMENT_STATUSES = {
    "continuation_stable_under_eta_refinement",
    "continuation_requires_eta_refinement",
    "continuation_unstable",
    "continuation_inconclusive",
}

_ALLOWED_RESTART_VS_HISTORY = {
    "restart_and_history_consistent",
    "restart_differs_from_history",
    "restart_artifact_possible",
    "paper_style_restart_differs_from_caputo_history_transport",
    "comparison_inconclusive",
    "continuation_auxiliary_unavailable",
    "deformed_lure_continuation_available",
    "deformed_lure_continuation_skipped",
    "deformed_lure_continuation_inconclusive",
    "deformed_lure_continuation_passed",
    "deformed_lure_continuation_sensitive_to_history",
    "deformed_lure_continuation_failed",
}

_ALLOWED_OVERALL_STATUSES = {
    "continuation_validation_passed",
    "continuation_validation_passed_with_eta_refinement",
    "continuation_validation_sensitive_to_history",
    "continuation_validation_inconclusive",
    "continuation_validation_failed",
    "continuation_validation_partial_original_only",
}


# ===========================================================================
# 1. load_continuation_config
# ===========================================================================

def load_continuation_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate a continuation memory validation YAML config.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    config : dict

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    ValueError
        If required fields are missing or invalid.
    """
    if not _YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required to load continuation configs. "
            "Install with: pip install pyyaml"
        )

    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        config = _yaml.safe_load(fh)

    if config is None:
        raise ValueError(f"Empty YAML file: {path}")

    required = ["case_id", "system_id", "q", "initial_condition",
                "integrator", "eta_grids", "memory_transport", "comparison_policy", "continuation_modes"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing required fields in {path}: {missing}")

    modes = config["continuation_modes"]
    if "deformed_lure_continuation" not in modes or "original_system_strategy_comparison" not in modes:
        raise ValueError(
            "continuation_modes must contain deformed_lure_continuation and "
            "original_system_strategy_comparison"
        )

    q = float(config["q"])
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0, 1), got {q}")

    method = config["integrator"].get("method", "")
    if method != "ABM":
        raise ValueError(f"integrator.method must be 'ABM', got '{method}'")

    if "x0" not in config.get("initial_condition", {}):
        raise ValueError("initial_condition.x0 must be provided")

    eta_grids = config.get("eta_grids", [])
    if not eta_grids:
        raise ValueError("eta_grids must contain at least one entry")

    strategies = config.get("memory_transport", {}).get("strategies", [])
    if not strategies:
        raise ValueError("memory_transport.strategies must contain at least one entry")

    policy = config["comparison_policy"]
    if policy.get("pointwise_comparison_used", True):
        raise ValueError(
            "comparison_policy.pointwise_comparison_used must be false"
        )

    return config


# ===========================================================================
# 2. get_continuation_system
# ===========================================================================

def get_continuation_system(system_id: str) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, Any]:
    """Return system components needed for eta-continuation.

    Parameters
    ----------
    system_id : str
        One of: 'chua_fractional_saturation', 'chua_fractional_arctan'.

    Returns
    -------
    system_obj : ChaoticSystem
    P : np.ndarray, shape (n, n)
        Lur'e linear matrix (lure.matrix).
    b : np.ndarray, shape (n,)
        Lur'e input vector (lure.input_vector).
    r : np.ndarray, shape (n,)
        Lur'e output vector (lure.output_vector).
    psi : callable
        Scalar nonlinearity psi(sigma) -> float.
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
    lure = system_obj.lure

    P = np.asarray(lure.matrix, dtype=float)
    b = np.asarray(lure.input_vector, dtype=float)
    r = np.asarray(lure.output_vector, dtype=float)
    psi = lure.nonlinearity

    return system_obj, P, b, r, psi


# ===========================================================================
# 3. build_eta_rhs
# ===========================================================================

def build_eta_rhs(
    system_obj: Any,
    eta: float,
    k: Optional[float],
) -> Tuple[Any, str]:
    """Build the deformed RHS F_eta(X) for parameter continuation.

    The deformed field is:

        F_eta(X) = P*X + b*[k*sigma + eta*(psi(sigma) - k*sigma)]

    with sigma = r^T X.

    At eta=0: F_0(X) = P*X + b*k*sigma  (linear auxiliary system)
    At eta=1: F_1(X) = P*X + b*psi(sigma) (original nonlinearity)

    Parameters
    ----------
    system_obj : ChaoticSystem
        System with .lure attribute.
    eta : float
        Continuation parameter in [0, 1].
    k : float or None
        DF-derived gain. If None, no auxiliary field is available.

    Returns
    -------
    rhs_eta : callable or None
        rhs_eta(t, x) -> np.ndarray, or None if unavailable.
    availability : str
        'available' or 'continuation_auxiliary_unavailable'.
    """
    if k is None:
        # No DF seed: cannot build deformed field
        return None, "continuation_auxiliary_unavailable"

    lure = system_obj.lure
    P = np.asarray(lure.matrix, dtype=float)
    b = np.asarray(lure.input_vector, dtype=float)
    r = np.asarray(lure.output_vector, dtype=float)
    psi = lure.nonlinearity
    k_val = float(k)
    eta_val = float(eta)

    def rhs_eta(t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        sigma = float(r @ x)
        return P @ x + b * (k_val * sigma + eta_val * (float(psi(sigma)) - k_val * sigma))

    return rhs_eta, "available"


# ===========================================================================
# 4. extract_history
# ===========================================================================

def extract_history(
    times: np.ndarray,
    states: np.ndarray,
    M: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the last M points from a trajectory as history.

    Parameters
    ----------
    times : np.ndarray, shape (N,)
    states : np.ndarray, shape (N, dim)
    M : int
        Number of history points to extract.

    Returns
    -------
    history_times : np.ndarray, shape (min(M, N),)
    history_states : np.ndarray, shape (min(M, N), dim)
    """
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    N = len(times)
    start = max(0, N - int(M))
    return times[start:].copy(), states[start:].copy()


# ===========================================================================
# 5. restart_from_last_point
# ===========================================================================

def restart_from_last_point(
    rhs_eta: Any,
    x_last: np.ndarray,
    q: float,
    h: float,
    t_final: float,
    divergence_norm: float = 120.0,
) -> Dict[str, Any]:
    """Integrate from a single restart point (Caputo history reset).

    This strategy discards all history. The new segment starts at
    t_0 = 0 with x(0) = x_last and empty history.

    Parameters
    ----------
    rhs_eta : callable
        rhs_eta(t, x) -> np.ndarray for the current eta.
    x_last : np.ndarray
        Last state of the previous segment.
    q, h, t_final, divergence_norm : float

    Returns
    -------
    result : dict
        Keys: times, states, status, caputo_history_reset, history_transported,
        rhs_history_recomputed_after_eta_change, history_length, strategy.
    """
    from hidden_attractors.integrations.abm import caputo_abm_integrate

    x0 = np.asarray(x_last, dtype=float)
    times, states, status = caputo_abm_integrate(
        rhs=rhs_eta,
        x0=x0,
        q=q,
        h=h,
        t_final=t_final,
        divergence_norm=divergence_norm,
        memory_mode="full",
        history_times=None,
        history_states=None,
        use_c_backend=True,
    )

    return {
        "times": times,
        "states": states,
        "status": status,
        "strategy": "last_point_restart",
        "caputo_history_reset": True,
        "history_transported": False,
        "rhs_history_recomputed_after_eta_change": False,
        "history_length": 0,
    }


# ===========================================================================
# 6. continue_with_history_window
# ===========================================================================

def continue_with_history_window(
    rhs_eta_new: Any,
    history_times: np.ndarray,
    history_states: np.ndarray,
    q: float,
    h: float,
    t_final: float,
    M: int,
    divergence_norm: float = 120.0,
) -> Dict[str, Any]:
    """Continue integration with transported history window.

    Strategy: experimental_history_transport_abm

    The history states X_j are transported from the previous segment.
    The RHS samples f_j = F_eta_new(X_j) are recomputed using the new
    field before passing to caputo_abm_integrate.

    The last point of history_states becomes x0 for the new segment.
    The times are shifted so that t = 0 at the last history point.

    Parameters
    ----------
    rhs_eta_new : callable
        rhs(t, x) for the NEW eta value.
    history_times : np.ndarray, shape (K,)
    history_states : np.ndarray, shape (K, dim)
    q, h, t_final : float
    M : int
        Nominal history window size (informational).
    divergence_norm : float

    Returns
    -------
    result : dict
        Keys: times, states, status, caputo_history_reset, history_transported,
        rhs_history_recomputed_after_eta_change, history_length, strategy.

    Notes
    -----
    This function uses caputo_abm_integrate's native history_times /
    history_states support. The RHS recomputation (f_j = F_new(X_j))
    happens inside caputo_abm_integrate when it evaluates:
        f_arr[j] = eval_rhs(rhs, t_arr[j], history_states[j])
    for all prehistory indices j.

    By passing rhs_eta_new as the rhs argument, the integrator automatically
    evaluates the history under the NEW field. This is the desired behavior
    for correct eta-transition history transport.
    """
    from hidden_attractors.integrations.abm import caputo_abm_integrate

    history_times = np.asarray(history_times, dtype=float)
    history_states = np.asarray(history_states, dtype=float)
    K = len(history_times)

    # Shift times so last history point is at t=0
    t_offset = history_times[-1] if K > 0 else 0.0
    shifted_times = history_times - t_offset

    x0 = history_states[-1].copy()

    times, states, status = caputo_abm_integrate(
        rhs=rhs_eta_new,
        x0=x0,
        q=q,
        h=h,
        t_final=t_final,
        divergence_norm=divergence_norm,
        memory_mode="full",
        history_times=shifted_times,
        history_states=history_states,
        use_c_backend=True,
    )

    # Trim the history prefix from the returned trajectory (keep only new steps)
    # The integrator returns K + n_steps points; we keep only from K onward.
    n_keep = max(0, len(times) - K)
    if n_keep > 0:
        out_times = times[K:]
        out_states = states[K:]
    else:
        out_times = times
        out_states = states

    return {
        "times": out_times,
        "states": out_states,
        "status": status,
        "strategy": "history_window_transport",
        "caputo_history_reset": False,
        "history_transported": True,
        "rhs_history_recomputed_after_eta_change": True,
        "history_length": K,
    }


# ===========================================================================
# 7. classify_continuation_segment
# ===========================================================================

def classify_continuation_segment(
    times: np.ndarray,
    states: np.ndarray,
    t_burn: float,
    config: Dict[str, Any],
) -> str:
    """Classify a continuation trajectory segment.

    Delegates to Phase-C classify_fractional_trajectory.
    Adds geometric heuristics for periodic_candidate and
    chaotic_candidate_by_geometry if applicable.

    Parameters
    ----------
    times, states : np.ndarray
    t_burn : float
    config : dict

    Returns
    -------
    class_label : str
    """
    divergence_norm = float(config.get("divergence", {}).get("max_norm", 120.0))
    clf = config.get("classification", {})
    collapse_var = float(clf.get("collapse_variance_tolerance", 1e-8))
    min_range = float(clf.get("min_range_tolerance", 1e-5))

    base_class = classify_fractional_trajectory(
        times=times,
        states=states,
        t_burn=t_burn,
        divergence_norm=divergence_norm,
        collapse_variance_tolerance=collapse_var,
        min_range_tolerance=min_range,
    )

    if base_class != "bounded_nontrivial":
        return base_class

    # Apply geometric heuristics only if not failed
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    mask = times >= t_burn
    post = states[mask]

    if post.shape[0] < 10:
        return "bounded_nontrivial"

    # Periodicity heuristic: compare first and second half mean distance
    half = len(post) // 2
    mean1 = np.mean(post[:half], axis=0)
    mean2 = np.mean(post[half:], axis=0)
    rho_full = float(np.sqrt(np.mean(np.sum((post - np.mean(post, axis=0)) ** 2, axis=1))))

    if rho_full < 1e-12:
        return "bounded_nontrivial"

    # If both halves have similar centroids AND the trajectory occupies
    # a thin manifold in 3D, classify as periodic_candidate
    centroid_drift = float(np.linalg.norm(mean2 - mean1))
    if centroid_drift / (rho_full + 1e-15) < 0.05:
        # Strong centroid stability suggests periodicity; check dimension via variance
        var_per_dim = np.var(post, axis=0)
        # Count dimensions with significant variance
        significant_dims = int(np.sum(var_per_dim > 0.01 * np.max(var_per_dim)))
        if significant_dims <= 1:
            return "periodic_candidate"

    # Chaotic candidate: high spread in multiple dimensions, bounded
    var_per_dim = np.var(post, axis=0)
    significant_dims = int(np.sum(var_per_dim > 0.01 * np.max(var_per_dim)))
    if significant_dims >= 2 and rho_full > 0.5:
        return "chaotic_candidate_by_geometry"

    return "bounded_nontrivial"


# ===========================================================================
# 8. compute_segment_metrics
# ===========================================================================

def compute_segment_metrics(
    times: np.ndarray,
    states: np.ndarray,
    t_burn: float,
) -> Dict[str, Any]:
    """Compute attractor metrics for a continuation segment.

    Delegates to Phase-C compute_memory_metrics.
    """
    return compute_memory_metrics(times=times, states=states, t_burn=t_burn)


# ===========================================================================
# 9. compute_jump
# ===========================================================================

def compute_jump(
    prev_metrics: Dict[str, Any],
    curr_metrics: Dict[str, Any],
) -> Dict[str, float]:
    """Compute jump norms between consecutive eta segments.

    Parameters
    ----------
    prev_metrics, curr_metrics : dict
        Output of compute_segment_metrics.

    Returns
    -------
    jump : dict
        Keys: jump_norm, rho_jump, range_jump.
    """
    eps = 1e-15

    fs_prev = np.asarray(prev_metrics["final_state"], dtype=float)
    fs_curr = np.asarray(curr_metrics["final_state"], dtype=float)
    norm_prev = float(np.linalg.norm(fs_prev))
    jump_norm = float(np.linalg.norm(fs_curr - fs_prev)) / (norm_prev + eps)

    rho_prev = float(prev_metrics["rho_attractor"])
    rho_curr = float(curr_metrics["rho_attractor"])
    rho_jump = abs(rho_curr - rho_prev) / (rho_prev + eps)

    range_prev = np.asarray(prev_metrics["range_vector"], dtype=float)
    range_curr = np.asarray(curr_metrics["range_vector"], dtype=float)
    range_norm_prev = float(np.linalg.norm(range_prev))
    range_jump = float(np.linalg.norm(range_curr - range_prev)) / (range_norm_prev + eps)

    return {
        "jump_norm": float(jump_norm),
        "rho_jump": float(rho_jump),
        "range_jump": float(range_jump),
    }


# ===========================================================================
# 10. run_eta_path
# ===========================================================================

def run_eta_path(
    config: Dict[str, Any],
    system_obj: Any,
    N_eta: int,
    strategy: str,
    continuation_mode: str,
    M: Optional[int] = None,
    fast: bool = False,
) -> List[Dict[str, Any]]:
    """Run one eta path for a given strategy and grid size under a continuation mode.

    Parameters
    ----------
    config : dict
    system_obj : ChaoticSystem
    N_eta : int
        Number of eta steps. eta_i = i / N_eta.
    strategy : str
        'last_point_restart' or 'history_window_transport'.
    continuation_mode : str
        'deformed_lure' or 'original_system'.
    M : int or None
        History window size. Required if strategy == 'history_window_transport'.
    fast : bool
        Use fast_test time parameters.

    Returns
    -------
    path_records : list of dict
        One dict per eta step.
    """
    case_id = config["case_id"]
    system_id = config["system_id"]
    q = float(config["q"])
    h = float(config["integrator"]["h"])
    k_val = config.get("lure_seed", {}).get("k")

    if fast and "fast_test" in config:
        t_final = float(config["fast_test"]["t_final"])
        t_burn = float(config["fast_test"]["t_burn"])
    else:
        t_final = float(config["time_per_eta"]["t_final"])
        t_burn = float(config["time_per_eta"]["t_burn"])

    divergence_norm = float(config.get("divergence", {}).get("max_norm", 120.0))
    policy = config["comparison_policy"]
    jump_norm_tol = float(policy.get("jump_norm_tolerance", 0.50))
    rho_jump_tol = float(policy.get("rho_jump_tolerance", 0.35))

    x0 = np.asarray(config["initial_condition"]["x0"], dtype=float)

    # If deformed lure is requested but k is null, return unavailable placeholders immediately
    if continuation_mode == "deformed_lure" and k_val is None:
        path_records: List[Dict[str, Any]] = []
        eta_values = [float(i) / float(N_eta) for i in range(N_eta + 1)]
        for step_idx, eta in enumerate(eta_values):
            record = _make_unavailable_record(
                case_id=case_id,
                system_id=system_id,
                strategy=strategy,
                N_eta=N_eta,
                eta_i=eta,
                step_idx=step_idx,
                M=M,
                h=h,
                t_final=t_final,
                t_burn=t_burn,
            )
            path_records.append(record)
        return path_records

    # Build eta values: eta_0=0, eta_1=1/N, ..., eta_N=1
    eta_values = [float(i) / float(N_eta) for i in range(N_eta + 1)]

    path_records = []
    prev_times: Optional[np.ndarray] = None
    prev_states: Optional[np.ndarray] = None
    prev_metrics: Optional[Dict[str, Any]] = None
    x_carry = x0.copy()

    for step_idx, eta in enumerate(eta_values):
        if continuation_mode == "deformed_lure":
            rhs_eta, availability = build_eta_rhs(system_obj, eta, k_val)
            deformed_lure_available = True
            original_system_comparison = False
        else:
            rhs_eta = _build_original_rhs(system_obj)
            availability = "original_system_available"
            deformed_lure_available = False
            original_system_comparison = True

        # --- Run this segment ---
        if step_idx == 0 or prev_times is None or strategy == "last_point_restart":
            # First step always restarts; subsequent steps depend on strategy
            seg_result = restart_from_last_point(
                rhs_eta=rhs_eta,
                x_last=x_carry,
                q=q,
                h=h,
                t_final=t_final,
                divergence_norm=divergence_norm,
            )
            if step_idx == 0:
                seg_result["strategy"] = "initial_integration"
                seg_result["caputo_history_reset"] = False  # true initial start
        else:
            if strategy == "history_window_transport" and M is not None:
                hist_t, hist_x = extract_history(prev_times, prev_states, M)
                seg_result = continue_with_history_window(
                    rhs_eta_new=rhs_eta,
                    history_times=hist_t,
                    history_states=hist_x,
                    q=q,
                    h=h,
                    t_final=t_final,
                    M=M,
                    divergence_norm=divergence_norm,
                )
            else:
                seg_result = restart_from_last_point(
                    rhs_eta=rhs_eta,
                    x_last=x_carry,
                    q=q,
                    h=h,
                    t_final=t_final,
                    divergence_norm=divergence_norm,
                )

        times_seg = seg_result["times"]
        states_seg = seg_result["states"]
        int_status = seg_result["status"]

        dyn_class = classify_continuation_segment(
            times=times_seg,
            states=states_seg,
            t_burn=t_burn,
            config=config,
        )

        metrics = compute_segment_metrics(times_seg, states_seg, t_burn)

        # Jump norms
        if prev_metrics is not None and step_idx > 0:
            jump = compute_jump(prev_metrics, metrics)
        else:
            jump = {"jump_norm": 0.0, "rho_jump": 0.0, "range_jump": 0.0}

        # Warning
        warning = (
            jump["jump_norm"] > jump_norm_tol
            or jump["rho_jump"] > rho_jump_tol
            or dyn_class in ("nan_detected", "diverged")
        )

        fs = metrics["final_state"]
        record = {
            "case_id": case_id,
            "system_id": system_id,
            "continuation_mode": continuation_mode,
            "deformed_lure_available": deformed_lure_available,
            "original_system_comparison": original_system_comparison,
            "availability": availability,
            "strategy": seg_result["strategy"],
            "N_eta": N_eta,
            "eta_i": eta,
            "step_idx": step_idx,
            "M": M if M is not None else "",
            "h": h,
            "t_final_per_eta": t_final,
            "t_burn": t_burn,
            "history_length": seg_result["history_length"],
            "caputo_history_reset": seg_result["caputo_history_reset"],
            "history_transported": seg_result["history_transported"],
            "rhs_history_recomputed_after_eta_change": seg_result["rhs_history_recomputed_after_eta_change"],
            "int_status": int_status,
            "dynamic_class": dyn_class,
            "final_state_x": fs[0] if len(fs) > 0 else float("nan"),
            "final_state_y": fs[1] if len(fs) > 1 else float("nan"),
            "final_state_z": fs[2] if len(fs) > 2 else float("nan"),
            "rho_attractor": metrics["rho_attractor"],
            "rho_max": metrics["rho_max"],
            "range_x": metrics["range_vector"][0] if len(metrics["range_vector"]) > 0 else float("nan"),
            "range_y": metrics["range_vector"][1] if len(metrics["range_vector"]) > 1 else float("nan"),
            "range_z": metrics["range_vector"][2] if len(metrics["range_vector"]) > 2 else float("nan"),
            "jump_norm": jump["jump_norm"],
            "rho_jump": jump["rho_jump"],
            "range_jump": jump["range_jump"],
            "warning": warning,
        }
        path_records.append(record)

        # Carry forward
        if len(states_seg) > 0:
            x_carry = states_seg[-1].copy()
        prev_times = times_seg
        prev_states = states_seg
        prev_metrics = metrics

    return path_records


def _build_original_rhs(system_obj: Any):
    """Build rhs(t, x) from the original (undeformed) system."""
    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        return system_obj.evaluate(np.asarray(x, dtype=float))
    return rhs


def _make_unavailable_record(
    case_id, system_id, strategy, N_eta, eta_i, step_idx, M, h, t_final, t_burn
) -> Dict[str, Any]:
    """Make a placeholder record when continuation auxiliary is unavailable."""
    return {
        "case_id": case_id,
        "system_id": system_id,
        "continuation_mode": "deformed_lure",
        "deformed_lure_available": False,
        "original_system_comparison": False,
        "availability": "continuation_auxiliary_unavailable",
        "strategy": strategy,
        "N_eta": N_eta,
        "eta_i": eta_i,
        "step_idx": step_idx,
        "M": M if M is not None else "",
        "h": h,
        "t_final_per_eta": t_final,
        "t_burn": t_burn,
        "history_length": 0,
        "caputo_history_reset": False,
        "history_transported": False,
        "rhs_history_recomputed_after_eta_change": False,
        "int_status": "continuation_auxiliary_unavailable",
        "dynamic_class": "inconclusive",
        "final_state_x": float("nan"),
        "final_state_y": float("nan"),
        "final_state_z": float("nan"),
        "rho_attractor": float("nan"),
        "rho_max": float("nan"),
        "range_x": float("nan"),
        "range_y": float("nan"),
        "range_z": float("nan"),
        "jump_norm": float("nan"),
        "rho_jump": float("nan"),
        "range_jump": float("nan"),
        "warning": True,
    }


# ===========================================================================
# 11. compare_eta_grids
# ===========================================================================

def compare_eta_grids(
    results_by_N: Dict[int, List[Dict[str, Any]]],
) -> Tuple[str, List[str]]:
    """Compare eta grid results across N = 10, 25, 50, 100.

    Compares the final-step dynamic class for each N.

    Parameters
    ----------
    results_by_N : dict
        {N_eta: path_records} for each grid size.

    Returns
    -------
    eta_refinement_status : str
    warnings : list of str
    """
    warnings: List[str] = []

    if not results_by_N:
        return "continuation_inconclusive", ["No results provided."]

    # Get final-step class for each N
    def get_final_class(records: List[Dict[str, Any]]) -> str:
        if not records:
            return "inconclusive"
        last = records[-1]
        return last.get("dynamic_class", "inconclusive")

    final_classes: Dict[int, str] = {}
    for N, recs in sorted(results_by_N.items()):
        fc = get_final_class(recs)
        final_classes[N] = fc

    bad = {"nan_detected", "diverged", "inconclusive"}

    # Gather large and small grids
    sorted_Ns = sorted(final_classes.keys())
    if not sorted_Ns:
        return "continuation_inconclusive", ["Empty result set."]

    large_Ns = [N for N in sorted_Ns if N >= 25]
    small_Ns = [N for N in sorted_Ns if N < 25]

    large_classes = {N: final_classes[N] for N in large_Ns}
    small_classes = {N: final_classes[N] for N in small_Ns}

    # Check if large grids are consistent
    large_vals = set(large_classes.values()) - bad
    large_bad = [N for N in large_Ns if final_classes[N] in bad]
    large_good = [N for N in large_Ns if final_classes[N] not in bad]

    if not large_Ns:
        return "continuation_inconclusive", ["No large grids (N>=25) found."]

    # All large grids bad -> inconclusive/failed
    if len(large_bad) == len(large_Ns):
        warnings.append("All large grids (N>=25) produced failed trajectories.")
        return "continuation_unstable", warnings

    # Large grids agree (same non-bad class)
    if len(large_vals) == 1:
        # Check if small grids fail
        small_fail = any(final_classes[N] in bad for N in small_Ns)
        if small_Ns and small_fail:
            warnings.append(
                f"Small grids (N={small_Ns}) fail but large grids agree: "
                f"class={list(large_vals)[0]}"
            )
            return "continuation_requires_eta_refinement", warnings

        # All agree -> stable
        return "continuation_stable_under_eta_refinement", warnings

    # Large grids disagree -> check if at least two largest agree
    two_largest = sorted(large_Ns)[-2:] if len(large_Ns) >= 2 else large_Ns
    if len({final_classes[N] for N in two_largest} - bad) == 1:
        warnings.append(
            f"Largest two grids N={two_largest} agree but others disagree."
        )
        return "continuation_requires_eta_refinement", warnings

    warnings.append(
        f"Grid results inconsistent: {final_classes}"
    )
    return "continuation_unstable", warnings


# ===========================================================================
# 12. compare_restart_vs_history
# ===========================================================================

def compare_restart_vs_history(
    restart_records: List[Dict[str, Any]],
    history_records: List[Dict[str, Any]],
    policy: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], List[str]]:
    """Compare last_point_restart vs history_window_transport final states.

    Parameters
    ----------
    restart_records : list of dict
        Path records from last_point_restart.
    history_records : list of dict
        Path records from history_window_transport.
    policy : dict
        comparison_policy from YAML.

    Returns
    -------
    status : str
    comparison_dict : dict
    warnings : list of str
    """
    warnings: List[str] = []
    eps = 1e-15

    # Check for unavailable
    if not restart_records or not history_records:
        return "comparison_inconclusive", {}, ["Missing records."]

    restart_last = restart_records[-1]
    history_last = history_records[-1]

    rc = restart_last.get("availability", "available")
    hc = history_last.get("availability", "available")
    if "unavailable" in rc or "unavailable" in hc:
        return "continuation_auxiliary_unavailable", {}, []

    restart_class = restart_last.get("dynamic_class", "inconclusive")
    history_class = history_last.get("dynamic_class", "inconclusive")
    class_changed = (restart_class != history_class)

    # Final state relative distance
    fs_r = np.array([
        restart_last.get("final_state_x", float("nan")),
        restart_last.get("final_state_y", float("nan")),
        restart_last.get("final_state_z", float("nan")),
    ], dtype=float)
    fs_h = np.array([
        history_last.get("final_state_x", float("nan")),
        history_last.get("final_state_y", float("nan")),
        history_last.get("final_state_z", float("nan")),
    ], dtype=float)

    fs_rel_dist = float("nan")
    if np.all(np.isfinite(fs_r)) and np.all(np.isfinite(fs_h)):
        fs_rel_dist = float(np.linalg.norm(fs_h - fs_r)) / (float(np.linalg.norm(fs_r)) + eps)

    rho_r = float(restart_last.get("rho_attractor", float("nan")))
    rho_h = float(history_last.get("rho_attractor", float("nan")))
    rho_rel_diff = float("nan")
    if math.isfinite(rho_r) and math.isfinite(rho_h):
        rho_rel_diff = abs(rho_h - rho_r) / (rho_r + eps)

    range_r = np.array([
        restart_last.get("range_x", float("nan")),
        restart_last.get("range_y", float("nan")),
        restart_last.get("range_z", float("nan")),
    ], dtype=float)
    range_h = np.array([
        history_last.get("range_x", float("nan")),
        history_last.get("range_y", float("nan")),
        history_last.get("range_z", float("nan")),
    ], dtype=float)
    range_rel_diff = float("nan")
    if np.all(np.isfinite(range_r)) and np.all(np.isfinite(range_h)):
        rn_r = float(np.linalg.norm(range_r))
        range_rel_diff = float(np.linalg.norm(range_h - range_r)) / (rn_r + eps)

    # Tolerances
    fs_tol = float(policy.get("final_state_relative_tolerance", 0.50))
    rho_tol = float(policy.get("rho_jump_tolerance", 0.35))
    range_tol = float(policy.get("range_relative_tolerance", 0.35))

    exceeded = []
    if math.isfinite(fs_rel_dist) and fs_rel_dist > fs_tol:
        exceeded.append(f"final_state_relative_distance={fs_rel_dist:.4f} > {fs_tol}")
    if math.isfinite(rho_rel_diff) and rho_rel_diff > rho_tol:
        exceeded.append(f"rho_relative_difference={rho_rel_diff:.4f} > {rho_tol}")
    if math.isfinite(range_rel_diff) and range_rel_diff > range_tol:
        exceeded.append(f"range_relative_difference={range_rel_diff:.4f} > {range_tol}")

    comparison_dict: Dict[str, Any] = {
        "restart_dynamic_class": restart_class,
        "history_dynamic_class": history_class,
        "class_changed": class_changed,
        "final_state_relative_distance": fs_rel_dist,
        "rho_relative_difference": rho_rel_diff,
        "range_relative_difference": range_rel_diff,
    }

    if class_changed:
        warnings.append(
            f"Dynamic class changed: restart='{restart_class}' "
            f"vs history='{history_class}'"
        )

    if exceeded:
        warnings.extend(exceeded)

    # Determine status
    if class_changed and exceeded:
        status = "restart_artifact_possible"
    elif class_changed:
        status = "restart_differs_from_history"
    elif exceeded:
        status = "paper_style_restart_differs_from_caputo_history_transport"
    else:
        status = "restart_and_history_consistent"

    comparison_dict["warning"] = len(warnings) > 0
    comparison_dict["status"] = status

    return status, comparison_dict, warnings


# ===========================================================================
# Helper: write CSV
# ===========================================================================

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# 13. run_continuation_memory_validation
# ===========================================================================

def run_continuation_memory_validation(
    config_path: str | Path,
    output_dir: str | Path,
    fast: bool = False,
    save_histories: bool = False,
) -> Dict[str, Any]:
    """Run full continuation memory validation for one YAML config.

    Parameters
    ----------
    config_path : str or Path
    output_dir : str or Path
    fast : bool
    save_histories : bool

    Returns
    -------
    summary : dict
    """
    config = load_continuation_config(config_path)
    case_id = config["case_id"]
    system_id = config["system_id"]
    q = float(config["q"])

    system_obj, P, b, r, psi = get_continuation_system(system_id)

    k_val = config.get("lure_seed", {}).get("k")
    eta_grid_sizes = [int(n) for n in config.get("eta_grids", [10, 25, 50, 100])]
    strategies = config.get("memory_transport", {}).get("strategies", [])
    mem_transport = config.get("memory_transport", {})
    units = mem_transport.get("units", "steps")
    h = float(config["integrator"]["h"])
    window_entries = mem_transport.get("history_windows", [])
    windows_M = [
        int(w["M"]) if units == "steps" else int(round(float(w["M"]) / h))
        for w in window_entries
    ]
    policy = config["comparison_policy"]

    modes_cfg = config.get("continuation_modes", {})
    run_deformed = bool(modes_cfg.get("deformed_lure_continuation", True))
    run_original = bool(modes_cfg.get("original_system_strategy_comparison", True))

    out_root = Path(output_dir)
    case_out = out_root / case_id
    case_out.mkdir(parents=True, exist_ok=True)

    if save_histories:
        hist_dir = case_out / "histories"
        hist_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Run all strategies × N_eta combinations
    # -----------------------------------------------------------------------
    all_grid_rows: List[Dict[str, Any]] = []
    results_by_mode_N: Dict[str, Dict[int, List[Dict[str, Any]]]] = {
        "deformed_lure": {},
        "original_system": {},
    }
    history_results_by_mode_N_M: Dict[str, Dict[Tuple[int, int], List[Dict[str, Any]]]] = {
        "deformed_lure": {},
        "original_system": {},
    }

    active_modes = []
    if run_deformed:
        active_modes.append("deformed_lure")
    if run_original:
        active_modes.append("original_system")

    for mode in active_modes:
        for N_eta in eta_grid_sizes:
            for strategy in strategies:
                if strategy == "last_point_restart":
                    print(
                        f"  [run] case={case_id} mode={mode} strategy=restart N={N_eta}"
                    )
                    records = run_eta_path(
                        config=config,
                        system_obj=system_obj,
                        N_eta=N_eta,
                        strategy="last_point_restart",
                        continuation_mode=mode,
                        M=None,
                        fast=fast,
                    )
                    all_grid_rows.extend(records)
                    results_by_mode_N[mode][N_eta] = records

                elif strategy == "history_window_transport":
                    for M in windows_M:
                        print(
                            f"  [run] case={case_id} mode={mode} strategy=history M={M} N={N_eta}"
                        )
                        records = run_eta_path(
                            config=config,
                            system_obj=system_obj,
                            N_eta=N_eta,
                            strategy="history_window_transport",
                            continuation_mode=mode,
                            M=M,
                            fast=fast,
                        )
                        all_grid_rows.extend(records)
                        history_results_by_mode_N_M[mode][(N_eta, M)] = records

    # -----------------------------------------------------------------------
    # Compare Lure continuation
    # -----------------------------------------------------------------------
    deformed_lure_continuation_available = False
    deformed_lure_continuation_status = "deformed_lure_continuation_skipped"
    eta_ref_status = "continuation_inconclusive"
    
    restart_vs_history_rows: List[Dict[str, Any]] = []
    all_warnings: List[str] = []

    if run_deformed:
        if k_val is None:
            deformed_lure_continuation_status = "continuation_auxiliary_unavailable"
            deformed_lure_continuation_available = False
            best_N = max(eta_grid_sizes) if eta_grid_sizes else 10
            best_M = max(windows_M) if windows_M else 256
            row = {
                "case_id": case_id,
                "system_id": system_id,
                "continuation_mode": "deformed_lure",
                "N_eta": best_N,
                "M": best_M,
                "restart_dynamic_class": "",
                "history_dynamic_class": "",
                "class_changed": False,
                "final_state_relative_distance": float("nan"),
                "rho_relative_difference": float("nan"),
                "range_relative_difference": float("nan"),
                "status": "continuation_auxiliary_unavailable",
                "warning": True,
            }
            restart_vs_history_rows.append(row)
        else:
            deformed_lure_continuation_available = True
            
            # Grid refinement check (deformed Lure)
            eta_ref_status, eta_ref_warns = compare_eta_grids(results_by_mode_N["deformed_lure"])
            all_warnings.extend(eta_ref_warns)

            # Strategy comparison
            best_N = max(eta_grid_sizes) if eta_grid_sizes else 10
            best_M = max(windows_M) if windows_M else 256

            rr = results_by_mode_N["deformed_lure"].get(best_N, [])
            hr = history_results_by_mode_N_M["deformed_lure"].get((best_N, best_M), [])

            status, cmp_dict, rv_h_warnings = compare_restart_vs_history(rr, hr, policy)
            all_warnings.extend(rv_h_warnings)

            # Map status
            if eta_ref_status == "continuation_unstable":
                deformed_lure_continuation_status = "deformed_lure_continuation_failed"
            elif status in ("restart_differs_from_history", "restart_artifact_possible"):
                deformed_lure_continuation_status = "deformed_lure_continuation_sensitive_to_history"
            elif status in ("restart_and_history_consistent", "paper_style_restart_differs_from_caputo_history_transport"):
                deformed_lure_continuation_status = "deformed_lure_continuation_passed"
            else:
                deformed_lure_continuation_status = "deformed_lure_continuation_inconclusive"

            # Add deformed Lure rows
            for N_eta in eta_grid_sizes:
                for M in (windows_M if windows_M else [best_M]):
                    rr_n = results_by_mode_N["deformed_lure"].get(N_eta, [])
                    hr_nm = history_results_by_mode_N_M["deformed_lure"].get((N_eta, M), [])
                    if not rr_n or not hr_nm:
                        continue
                    s, cd, _ = compare_restart_vs_history(rr_n, hr_nm, policy)
                    row = {
                        "case_id": case_id,
                        "system_id": system_id,
                        "continuation_mode": "deformed_lure",
                        "N_eta": N_eta,
                        "M": M,
                        "restart_dynamic_class": cd.get("restart_dynamic_class", ""),
                        "history_dynamic_class": cd.get("history_dynamic_class", ""),
                        "class_changed": cd.get("class_changed", False),
                        "final_state_relative_distance": cd.get("final_state_relative_distance", float("nan")),
                        "rho_relative_difference": cd.get("rho_relative_difference", float("nan")),
                        "range_relative_difference": cd.get("range_relative_difference", float("nan")),
                        "status": s,
                        "warning": cd.get("warning", False),
                    }
                    restart_vs_history_rows.append(row)

    # -----------------------------------------------------------------------
    # Compare original system
    # -----------------------------------------------------------------------
    original_system_strategy_comparison_performed = False
    original_system_restart_vs_history_status = "original_system_comparison_skipped"

    if run_original:
        original_system_strategy_comparison_performed = True
        best_N = max(eta_grid_sizes) if eta_grid_sizes else 10
        best_M = max(windows_M) if windows_M else 256

        rr = results_by_mode_N["original_system"].get(best_N, [])
        hr = history_results_by_mode_N_M["original_system"].get((best_N, best_M), [])

        status, cmp_dict, original_rv_h_warnings = compare_restart_vs_history(rr, hr, policy)
        all_warnings.extend(original_rv_h_warnings)

        # Map to original_system_restart_vs_history_status
        if status in ("restart_and_history_consistent", "paper_style_restart_differs_from_caputo_history_transport"):
            original_system_restart_vs_history_status = "original_restart_and_history_consistent"
        elif status == "restart_differs_from_history":
            original_system_restart_vs_history_status = "original_restart_differs_from_history"
        elif status == "restart_artifact_possible":
            original_system_restart_vs_history_status = "original_restart_artifact_possible"
        else:
            original_system_restart_vs_history_status = "original_comparison_inconclusive"

        # Add original system rows
        for N_eta in eta_grid_sizes:
            for M in (windows_M if windows_M else [best_M]):
                rr_n = results_by_mode_N["original_system"].get(N_eta, [])
                hr_nm = history_results_by_mode_N_M["original_system"].get((N_eta, M), [])
                if not rr_n or not hr_nm:
                    continue
                s, cd, _ = compare_restart_vs_history(rr_n, hr_nm, policy)
                row = {
                    "case_id": case_id,
                    "system_id": system_id,
                    "continuation_mode": "original_system",
                    "N_eta": N_eta,
                    "M": M,
                    "restart_dynamic_class": cd.get("restart_dynamic_class", ""),
                    "history_dynamic_class": cd.get("history_dynamic_class", ""),
                    "class_changed": cd.get("class_changed", False),
                    "final_state_relative_distance": cd.get("final_state_relative_distance", float("nan")),
                    "rho_relative_difference": cd.get("rho_relative_difference", float("nan")),
                    "range_relative_difference": cd.get("range_relative_difference", float("nan")),
                    "status": s,
                    "warning": cd.get("warning", False),
                }
                restart_vs_history_rows.append(row)

    # -----------------------------------------------------------------------
    # Determine overall status
    # -----------------------------------------------------------------------
    if run_deformed and k_val is not None:
        if deformed_lure_continuation_status == "deformed_lure_continuation_failed":
            overall_status = "continuation_validation_failed"
        elif deformed_lure_continuation_status == "deformed_lure_continuation_sensitive_to_history":
            overall_status = "continuation_validation_sensitive_to_history"
        elif deformed_lure_continuation_status == "deformed_lure_continuation_passed":
            if eta_ref_status == "continuation_requires_eta_refinement":
                overall_status = "continuation_validation_passed_with_eta_refinement"
            else:
                overall_status = "continuation_validation_passed"
        else:
            overall_status = "continuation_validation_inconclusive"
    elif run_deformed and k_val is None:
        if original_system_strategy_comparison_performed:
            overall_status = "continuation_validation_partial_original_only"
        else:
            overall_status = "continuation_validation_inconclusive"
    elif run_original:
        overall_status = "continuation_validation_partial_original_only"
    else:
        overall_status = "continuation_validation_inconclusive"

    # deduplicate warnings
    all_warnings = list(set(all_warnings))

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    _write_csv(case_out / "continuation_grid_summary.csv", all_grid_rows)
    _write_csv(case_out / "restart_vs_history_comparison.csv", restart_vs_history_rows)

    summary: Dict[str, Any] = {
        "stage": "continuation_memory_validation",
        "case_id": case_id,
        "system_id": system_id,
        "q": q,
        "eta_grids": eta_grid_sizes,
        "strategies": strategies,
        "history_windows": windows_M,
        
        "deformed_lure_continuation_requested": run_deformed,
        "deformed_lure_continuation_available": deformed_lure_continuation_available,
        "deformed_lure_continuation_status": deformed_lure_continuation_status,
        
        "original_system_strategy_comparison_requested": run_original,
        "original_system_strategy_comparison_performed": original_system_strategy_comparison_performed,
        "original_system_restart_vs_history_status": original_system_restart_vs_history_status,

        "overall_status": overall_status,
        "eta_refinement_status": eta_ref_status,
        "restart_vs_history_status": deformed_lure_continuation_status if k_val is not None else "continuation_auxiliary_unavailable",
        "transported_history": "history_window_transport" in strategies,
        "rhs_history_recomputed_after_eta_change": "history_window_transport" in strategies,
        "automatic_warnings": all_warnings,
        "pointwise_comparison_used": False,
        "hiddenness_certified_by_this_pipeline": False,
        "chaos_certified_by_this_pipeline": False,
        "no_hidden_verified_claim": True,
    }

    with (case_out / "continuation_validation_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


# ===========================================================================
# 14. run_all_continuation_memory_validations
# ===========================================================================

def run_all_continuation_memory_validations(
    config_dir: str | Path,
    output_dir: str | Path,
    fast: bool = False,
    save_histories: bool = False,
) -> List[Dict[str, Any]]:
    """Run continuation memory validation for all YAML configs in a directory.

    Parameters
    ----------
    config_dir : str or Path
    output_dir : str or Path
    fast : bool
    save_histories : bool

    Returns
    -------
    summaries : list of dict
    """
    config_dir = Path(config_dir)
    yaml_files = sorted(config_dir.glob("*_continuation.yaml"))

    if not yaml_files:
        raise FileNotFoundError(
            f"No *_continuation.yaml files found in: {config_dir}"
        )

    summaries = []
    for yaml_path in yaml_files:
        print(f"[continuation_memory_validation] Running: {yaml_path.name}")
        try:
            summary = run_continuation_memory_validation(
                config_path=yaml_path,
                output_dir=output_dir,
                fast=fast,
                save_histories=save_histories,
            )
            summaries.append(summary)
            print(
                f"  -> case_id={summary['case_id']}, "
                f"overall_status={summary['overall_status']}"
            )
        except Exception as exc:
            print(f"  [ERROR] {yaml_path.name}: {exc}")
            summaries.append({
                "stage": "continuation_memory_validation",
                "yaml_file": str(yaml_path),
                "overall_status": "continuation_validation_inconclusive",
                "error": str(exc),
                "hiddenness_certified_by_this_pipeline": False,
                "chaos_certified_by_this_pipeline": False,
                "no_hidden_verified_claim": True,
            })

    return summaries
