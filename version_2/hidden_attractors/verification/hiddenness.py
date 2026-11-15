import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from ..integrations.general import integrate_general
from .terminal_policy import (
    PROBE_POLICY, completion_tail, full_horizon_early_stop, terminal_equilibrium,
)

def generate_neighborhood_points(
    eq_point: np.ndarray,
    radius: float,
    num_samples: int,
    mode: str = "sphere_random",
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate surface samples or volume-uniform ball samples explicitly.

    ``sphere_random``, ``coordinate_axes`` and ``hybrid`` sample a surface.
    ``ball_random`` uses radius U**(1/d), hence uniform volume in d dimensions.
    """
    eq_point = np.asarray(eq_point, dtype=float)
    if (eq_point.ndim != 1 or not np.isfinite(eq_point).all() or len(eq_point) == 0
            or not np.isfinite(radius) or radius <= 0
            or isinstance(num_samples, bool) or int(num_samples) != num_samples or num_samples <= 0):
        raise ValueError("Finite center, positive radius and positive integer sample count required.")
    dim = len(eq_point)
    rng = np.random.default_rng(seed)
    
    if mode == "coordinate_axes":
        axes_pts = []
        for d in range(dim):
            v1 = np.zeros(dim)
            v1[d] = radius
            v2 = np.zeros(dim)
            v2[d] = -radius
            axes_pts.append(eq_point + v1)
            axes_pts.append(eq_point + v2)
            
        pts = np.array(axes_pts)
        if len(pts) > num_samples:
            return pts[:num_samples]
        elif len(pts) < num_samples:
            return pts
        return pts
        
    elif mode in ("sphere_random", "ball_random"):
        pts = []
        for _ in range(num_samples):
            direction = rng.normal(0.0, 1.0, dim)
            direction = direction / np.linalg.norm(direction)
            radial_fraction = rng.random() ** (1.0/dim) if mode == "ball_random" else 1.0
            pts.append(eq_point + radius * radial_fraction * direction)
        return np.array(pts)
        
    elif mode == "hybrid":
        axes_pts = []
        for d in range(dim):
            v1 = np.zeros(dim)
            v1[d] = radius
            v2 = np.zeros(dim)
            v2[d] = -radius
            axes_pts.append(eq_point + v1)
            axes_pts.append(eq_point + v2)
            
        pts = list(axes_pts)
        if len(pts) >= num_samples:
            return np.array(pts[:num_samples])
        else:
            needed = num_samples - len(pts)
            for _ in range(needed):
                direction = rng.normal(0.0, 1.0, dim)
                direction = direction / np.linalg.norm(direction)
                pts.append(eq_point + radius * direction)
            return np.array(pts)
    else:
        raise ValueError(f"Unknown direction sampling mode: {mode}")

def evaluate_target_match(
    trajectory_tail: np.ndarray,
    ref_tail: np.ndarray,
    metric: str = "centroid_distance",
    tolerance: float = 0.5,
    nn_percentile: float = 90.0,
) -> bool:
    """Evaluate if the trajectory tail coincides with the reference attractor tail.

    Parameters
    ----------
    trajectory_tail : np.ndarray, shape (T, d)
        Burn-in-discarded portion of the probe trajectory.
    ref_tail : np.ndarray, shape (R, d)
        Burn-in-discarded portion of the reference (seed) trajectory.
    metric : str
        One of:
        - ``"centroid_distance"`` : Euclidean distance between cloud centroids.
          Fast but fails for elongated or asymmetric attractors.
        - ``"bbox_overlap"`` : Axis-aligned bounding-box intersection.
          Only checks volumetric overlap, not cloud proximity.
        - ``"nn_percentile"`` : For every point in *trajectory_tail* compute
          the distance to its nearest neighbour in *ref_tail*, then test
          whether the *nn_percentile*-th percentile of that distribution is
          ≤ *tolerance*. Robust to shape, rotation, and density differences.
          Recommended for fractional attractor comparison.
    tolerance : float
        Distance threshold used by ``centroid_distance`` and ``nn_percentile``.
    nn_percentile : float
        Percentile (0–100) used by the ``nn_percentile`` metric. Default 90
        (i.e. 90 % of probe points must have a close neighbour in the ref).

    Returns
    -------
    bool
        True if the trajectory matches the reference attractor under the
        chosen metric and tolerance.
    """
    if len(trajectory_tail) == 0 or len(ref_tail) == 0:
        return False

    tr = np.asarray(trajectory_tail, dtype=float)
    ref = np.asarray(ref_tail, dtype=float)
    if tr.ndim != 2 or ref.ndim != 2 or tr.shape[1] != ref.shape[1]:
        raise ValueError("Trajectory and reference tails must have matching dimensions.")
    if not np.isfinite(tr).all() or not np.isfinite(ref).all():
        raise ValueError("Contact tails must contain only finite states.")

    if metric == "centroid_distance":
        centroid_tr = np.mean(tr, axis=0)
        centroid_ref = np.mean(ref, axis=0)
        dist = np.linalg.norm(centroid_tr - centroid_ref)
        return bool(dist <= tolerance)

    elif metric == "bbox_overlap":
        min_tr = np.min(tr, axis=0)
        max_tr = np.max(tr, axis=0)
        min_ref = np.min(ref, axis=0)
        max_ref = np.max(ref, axis=0)

        for d in range(len(min_tr)):
            if max_tr[d] < min_ref[d] or min_tr[d] > max_ref[d]:
                return False
        return True

    elif metric == "nn_percentile":
        from scipy.spatial import cKDTree
        MAX_PTS = 2000
        tr_use = tr[np.linspace(0, len(tr)-1, min(len(tr), MAX_PTS), dtype=int)]
        ref_use = ref[np.linspace(0, len(ref)-1, min(len(ref), MAX_PTS), dtype=int)]
        nn_dists = cKDTree(ref_use).query(tr_use, workers=1)[0]

        threshold = float(np.percentile(nn_dists, nn_percentile))
        return bool(threshold <= tolerance)

    else:
        raise ValueError(
            f"Unknown target_match metric: {metric!r}. "
            "Choose one of: 'centroid_distance', 'bbox_overlap', 'nn_percentile'."
        )

def _evaluate_target_match_legacy(
    trajectory_tail: np.ndarray,
    ref_tail: np.ndarray,
    metric: str = "centroid_distance",
    tolerance: float = 0.5,
    nn_percentile: float = 90.0,
) -> bool:
    """Evaluate if the trajectory tail coincides with the reference attractor tail.

    Parameters
    ----------
    trajectory_tail : np.ndarray, shape (T, d)
        Burn-in-discarded portion of the probe trajectory.
    ref_tail : np.ndarray, shape (R, d)
        Burn-in-discarded portion of the reference (seed) trajectory.
    metric : str
        One of:
        - ``"centroid_distance"`` : Euclidean distance between cloud centroids.
          Fast but fails for elongated or asymmetric attractors.
        - ``"bbox_overlap"`` : Axis-aligned bounding-box intersection.
          Only checks volumetric overlap, not cloud proximity.
        - ``"nn_percentile"`` : For every point in *trajectory_tail* compute
          the distance to its nearest neighbour in *ref_tail*, then test
          whether the *nn_percentile*-th percentile of that distribution is
          ≤ *tolerance*. Robust to shape, rotation, and density differences.
          Recommended for fractional attractor comparison.
    tolerance : float
        Distance threshold used by ``centroid_distance`` and ``nn_percentile``.
    nn_percentile : float
        Percentile (0–100) used by the ``nn_percentile`` metric. Default 90
        (i.e. 90 % of probe points must have a close neighbour in the ref).

    Returns
    -------
    bool
        True if the trajectory matches the reference attractor under the
        chosen metric and tolerance.
    """
    if len(trajectory_tail) == 0 or len(ref_tail) == 0:
        return False

    tr = np.asarray(trajectory_tail, dtype=float)
    ref = np.asarray(ref_tail, dtype=float)

    if metric == "centroid_distance":
        centroid_tr = np.mean(tr, axis=0)
        centroid_ref = np.mean(ref, axis=0)
        dist = np.linalg.norm(centroid_tr - centroid_ref)
        return bool(dist <= tolerance)

    elif metric == "bbox_overlap":
        min_tr = np.min(tr, axis=0)
        max_tr = np.max(tr, axis=0)
        min_ref = np.min(ref, axis=0)
        max_ref = np.max(ref, axis=0)

        for d in range(len(min_tr)):
            if max_tr[d] < min_ref[d] or min_tr[d] > max_ref[d]:
                return False
        return True

    elif metric == "nn_percentile":
        MAX_PTS = 2000
        rng = np.random.default_rng(0)

        tr_use = tr if len(tr) <= MAX_PTS else tr[rng.choice(len(tr), MAX_PTS, replace=False)]
        ref_use = ref if len(ref) <= MAX_PTS else ref[rng.choice(len(ref), MAX_PTS, replace=False)]

        diff = tr_use[:, np.newaxis, :] - ref_use[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        nn_dists = dists.min(axis=1)

        threshold = float(np.percentile(nn_dists, nn_percentile))
        return bool(threshold <= tolerance)

    else:
        raise ValueError(
            f"Unknown target_match metric: {metric!r}. "
            "Choose one of: 'centroid_distance', 'bbox_overlap', 'nn_percentile'."
        )


def _run_neighborhood_probe_legacy(
    system: Any,
    x0: np.ndarray,
    transfer_mode: str,
    integrator: str,
    t_final: float,
    t_burn: float,
    h: float,
    ref_tail: np.ndarray,
    stable_equilibria: List[np.ndarray],
    equilibrium_tol: float = 0.5,
    divergence_norm: float = 120.0,
    target_match_metric: str = "centroid_distance",
    target_match_tol: float = 0.5,
    dynamics_mode: str = "system",
    memory_mode: str = "full",
    memory_window_length: Optional[int] = None,
    early_stop_config: Optional[dict] = None,
    equilibria_dict: Optional[Dict[str, np.ndarray]] = None,
    q_dynamics_effective: Optional[float] = None
) -> Dict[str, Any]:
    """Integrate a single trajectory from a neighborhood and classify its destination with early stopping."""
    x0_arr = np.asarray(x0, dtype=float)
    
    if q_dynamics_effective is not None:
        active_q = q_dynamics_effective
    else:
        import warnings
        warnings.warn("q_dynamics_effective is omitted, falling back to legacy dynamics_mode logic", UserWarning)
        if dynamics_mode == "integer":
            active_q = 1.0
        elif dynamics_mode == "fractional":
            active_q = system.parameters.get("q", 1.0)
        elif dynamics_mode == "system":
            q_val = system.parameters.get("q", 1.0)
            active_q = 1.0 if q_val == 1.0 else q_val
        else:
            raise ValueError(f"Unknown dynamics_mode: {dynamics_mode}")
        
    all_eqs_list = list(equilibria_dict.values()) if equilibria_dict else stable_equilibria
    
    def rhs_probe(t: float, x: np.ndarray) -> np.ndarray:
        return np.asarray(system.evaluate(x), dtype=float)

    t_arr, x_arr, status = integrate_general(
        rhs=rhs_probe,
        x0=x0_arr,
        q=active_q,
        h=h,
        t_final=t_final,
        integrator=integrator,
        memory_mode=memory_mode,
        memory_window_length=memory_window_length,
        divergence_norm=divergence_norm,
        system=system,
        early_stop_config=early_stop_config,
        equilibria=all_eqs_list,
    )
            
    if status in ("diverged", "diverged_early", "nonfinite_solution"):
        return {"destination": "divergence", "status": status, "trajectory": x_arr}
    elif status.startswith("solver_exception"):
        return {"destination": "numerical_failure", "status": status, "trajectory": x_arr}
        
    if status == "converged_equilibrium_early":
        final_state = x_arr[-1]
        eq_name = "stable_equilibrium"
        if stable_equilibria:
            dists = [np.linalg.norm(final_state - eq) for eq in stable_equilibria]
            closest_idx = int(np.argmin(dists))
            if equilibria_dict:
                for name, eq_pt in equilibria_dict.items():
                    if np.allclose(eq_pt, stable_equilibria[closest_idx], atol=equilibrium_tol * 2):
                        eq_name = name
                        break
        return {"destination": "stable_equilibrium", "status": status, "trajectory": x_arr, "equilibrium_name": eq_name}
        
    n_burn = int(np.ceil(t_burn / h))
    if len(x_arr) <= n_burn:
        n_burn = int(len(x_arr) * 0.5)
    tail = x_arr[n_burn:]
    
    final_state = x_arr[-1]
    for eq in stable_equilibria:
        if np.linalg.norm(final_state - eq) <= equilibrium_tol:
            eq_name = "stable_equilibrium"
            if equilibria_dict:
                for name, eq_pt in equilibria_dict.items():
                    if np.allclose(eq_pt, eq, atol=1e-3):
                        eq_name = name
                        break
            return {"destination": "stable_equilibrium", "status": "ok", "trajectory": x_arr, "equilibrium_name": eq_name}
            
    if _evaluate_target_match_legacy(tail, ref_tail, metric=target_match_metric, tolerance=target_match_tol):
        return {"destination": "target_attractor", "status": "ok", "trajectory": x_arr}
        
    return {"destination": "other_attractor", "status": "ok", "trajectory": x_arr}


def run_neighborhood_probe(
    system: Any, x0: np.ndarray, transfer_mode: str, integrator: str,
    t_final: float, t_burn: float, h: float, ref_tail: np.ndarray,
    stable_equilibria: List[np.ndarray], equilibrium_tol: float = 0.5,
    divergence_norm: float = 120.0, target_match_metric: str = "centroid_distance",
    target_match_tol: float = 0.5, dynamics_mode: str = "system",
    memory_mode: str = "full", memory_window_length: Optional[int] = None,
    early_stop_config: Optional[dict] = None,
    equilibria_dict: Optional[Dict[str, np.ndarray]] = None,
    q_dynamics_effective: Optional[float] = None, *,
    classification_policy: str = PROBE_POLICY,
    terminal_window: Optional[float] = None,
    equilibrium_span_tol: Optional[float] = None,
    target_match_nn_percentile: float = 90.0,
) -> Dict[str, Any]:
    """Classify a recorded finite horizon; an early visit is never a destination.

    ``legacy_v1`` is explicit compatibility for historical replay. Its labels
    retain their old meaning. New labels describe finite observations only.
    """
    if classification_policy == "legacy_v1":
        result = _run_neighborhood_probe_legacy(
            system, x0, transfer_mode, integrator, t_final, t_burn, h, ref_tail,
            stable_equilibria, equilibrium_tol, divergence_norm,
            target_match_metric, target_match_tol, dynamics_mode, memory_mode,
            memory_window_length, early_stop_config, equilibria_dict, q_dynamics_effective)
        result["classification_policy"] = "legacy_v1"
        result["diagnostics"] = {"classification_policy": "legacy_v1",
                                 "meaning": "historical_early_capture_rule"}
        result["times"] = np.arange(len(result["trajectory"]), dtype=float)*h
        return result
    if classification_policy != PROBE_POLICY:
        raise ValueError(f"Unknown neighborhood classification policy: {classification_policy}")
    if (not np.isfinite(t_final) or t_final <= 0 or not np.isfinite(t_burn)
            or t_burn < 0 or t_burn >= t_final or not np.isfinite(h) or h <= 0):
        raise ValueError("Require 0 <= t_burn < t_final and positive finite h.")
    span_tol = 2*equilibrium_tol if equilibrium_span_tol is None else equilibrium_span_tol
    if any(not np.isfinite(v) or v <= 0 for v in
           (equilibrium_tol, span_tol, divergence_norm, target_match_tol)):
        raise ValueError("Distance, span and norm tolerances must be finite and positive.")
    if terminal_window is not None and (not np.isfinite(terminal_window) or terminal_window <= 0):
        raise ValueError("terminal_window must be finite and positive.")
    if not np.isfinite(target_match_nn_percentile) or not 0 <= target_match_nn_percentile <= 100:
        raise ValueError("target_match_nn_percentile must lie in [0, 100].")
    x0 = np.asarray(x0, dtype=float)
    diagnostics = {"classification_policy": PROBE_POLICY, "t_requested": float(t_final),
                   "t_reached": None, "tail_start": None, "tail_end": None,
                   "tail_samples": 0, "equilibrium_distance_tol": float(equilibrium_tol),
                   "equilibrium_span_tol": float(span_tol), "reason": None,
                   "meaning": "finite_horizon_observation_only"}
    times, states, status = np.empty(0), np.empty((0, x0.size)), "not_run"

    def result(destination, reason, **extra):
        diagnostics["reason"] = reason
        return {"destination": destination, "status": status, "trajectory": states,
                "times": times, "classification_policy": PROBE_POLICY,
                "diagnostics": diagnostics, **extra}

    active_q = q_dynamics_effective
    if active_q is None:
        import warnings
        warnings.warn("q_dynamics_effective is omitted, falling back to legacy dynamics_mode logic", UserWarning)
        if dynamics_mode == "integer":
            active_q = 1.0
        elif dynamics_mode in ("fractional", "system"):
            active_q = system.parameters.get("q", 1.0)
        else:
            raise ValueError(f"Unknown dynamics_mode: {dynamics_mode}")
    try:
        times, states, status = integrate_general(
            rhs=lambda t, x: np.asarray(system.evaluate(x), dtype=float), x0=x0,
            q=active_q, h=h, t_final=t_final, integrator=integrator,
            memory_mode=memory_mode, memory_window_length=memory_window_length,
            divergence_norm=divergence_norm, system=system,
            early_stop_config=full_horizon_early_stop(early_stop_config),
            equilibria=list(equilibria_dict.values()) if equilibria_dict else stable_equilibria)
        times, states = np.asarray(times, dtype=float), np.asarray(states, dtype=float)
        if times.size and np.isfinite(times[-1]):
            diagnostics["t_reached"] = float(times[-1])
        if states.ndim != 2 or states.shape[1] != x0.size:
            return result("numerical_failure", "invalid_state_shape")
        if status == "nonfinite_solution" or not np.isfinite(states).all():
            return result("numerical_failure", "nonfinite_solution")
        if str(status).startswith("solver_exception"):
            return result("numerical_failure", str(status))
        norms = np.linalg.norm(states, axis=1)
        diagnostics["max_norm"] = float(norms.max()) if len(norms) else None
        crossings = np.flatnonzero(norms > divergence_norm)
        if crossings.size:
            first = int(crossings[0])
            diagnostics["first_threshold_time"] = float(times[first]) if first < len(times) else None
            return result("threshold_crossed", "observed_finite_norm_threshold")
        if status != "ok":
            return result("inconclusive", "incomplete_or_failed_solver:"+str(status))
        try:
            tail_times, tail = completion_tail(times, states, t_final, t_burn,
                                              terminal_window=terminal_window)
        except ValueError as exc:
            return result("inconclusive", str(exc))
        diagnostics.update(tail_start=float(tail_times[0]), tail_end=float(tail_times[-1]),
                           tail_samples=len(tail))
        index, eq_diag = terminal_equilibrium(tail, stable_equilibria, equilibrium_tol, span_tol)
        diagnostics.update(eq_diag)
        if index is not None:
            eq = stable_equilibria[index]
            name = next((name for name, point in (equilibria_dict or {}).items()
                         if np.array_equal(np.asarray(point), np.asarray(eq))), "equilibrium")
            return result("equilibrium_terminal", "finite_terminal_equilibrium", equilibrium_name=name)
        ref = np.asarray(ref_tail, dtype=float)
        if ref.ndim != 2 or ref.shape[1] != x0.size or len(ref) < 3 or not np.isfinite(ref).all():
            return result("inconclusive", "reference_tail_unavailable_or_invalid")
        contact = evaluate_target_match(tail, ref, metric=target_match_metric,
                                        tolerance=target_match_tol,
                                        nn_percentile=target_match_nn_percentile)
        diagnostics["reference_contact"] = {"metric": target_match_metric,
                                             "tolerance": float(target_match_tol),
                                             "nn_percentile": float(target_match_nn_percentile),
                                             "is_contact": bool(contact),
                                             "meaning": "finite_reference_contact_only"}
        return result("reference_contact" if contact else "other_finite",
                      "finite_reference_contact" if contact else "other_finite")
    except Exception as exc:
        status = f"solver_exception:{type(exc).__name__}:{exc}"
        return result("numerical_failure", status)
