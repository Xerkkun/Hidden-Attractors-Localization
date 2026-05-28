import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from ..integrations.general import integrate_general

def generate_neighborhood_points(
    eq_point: np.ndarray,
    radius: float,
    num_samples: int,
    mode: str = "sphere_random",
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate initial conditions in the neighborhood of an equilibrium point."""
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
        
    elif mode == "sphere_random":
        pts = []
        for _ in range(num_samples):
            direction = rng.normal(0.0, 1.0, dim)
            direction = direction / np.linalg.norm(direction)
            pts.append(eq_point + radius * direction)
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

def run_neighborhood_probe(
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
            active_q = system.q
        elif dynamics_mode == "system":
            active_q = 1.0 if system.q == 1.0 else system.q
        else:
            raise ValueError(f"Unknown dynamics_mode: {dynamics_mode}")
        
    all_eqs_list = list(equilibria_dict.values()) if equilibria_dict else stable_equilibria
    
    def rhs_probe(t: float, x: np.ndarray) -> np.ndarray:
        return np.asarray(system.evaluate_rhs(x), dtype=float)

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
            
    if evaluate_target_match(tail, ref_tail, metric=target_match_metric, tolerance=target_match_tol):
        return {"destination": "target_attractor", "status": "ok", "trajectory": x_arr}
        
    return {"destination": "other_attractor", "status": "ok", "trajectory": x_arr}
