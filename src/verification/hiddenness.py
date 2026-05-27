import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from ..integrators.abm import caputo_abm_integrate
from ..integrators.efork import efork_integrate

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
    tolerance: float = 0.5
) -> bool:
    """Evaluate if the trajectory matches the target attractor tail under the given metric."""
    if len(trajectory_tail) == 0 or len(ref_tail) == 0:
        return False
        
    if metric == "centroid_distance":
        centroid_tr = np.mean(trajectory_tail, axis=0)
        centroid_ref = np.mean(ref_tail, axis=0)
        dist = np.linalg.norm(centroid_tr - centroid_ref)
        return bool(dist <= tolerance)
        
    elif metric == "bbox_overlap":
        min_tr = np.min(trajectory_tail, axis=0)
        max_tr = np.max(trajectory_tail, axis=0)
        min_ref = np.min(ref_tail, axis=0)
        max_ref = np.max(ref_tail, axis=0)
        
        overlap = True
        for d in range(len(min_tr)):
            if max_tr[d] < min_ref[d] or min_tr[d] > max_ref[d]:
                overlap = False
                break
        return overlap
        
    else:
        raise ValueError(f"Unknown target_match metric: {metric}")

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
    equilibria_dict: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Any]:
    """Integrate a single trajectory from a neighborhood and classify its destination with early stopping."""
    x0_arr = np.asarray(x0, dtype=float)
    
    # Resolve active q according to dynamics_mode
    if dynamics_mode == "integer":
        active_q = 1.0
    elif dynamics_mode == "fractional":
        active_q = system.q
    elif dynamics_mode == "system":
        active_q = 1.0 if system.q == 1.0 else system.q
    else:
        raise ValueError(f"Unknown dynamics_mode: {dynamics_mode}")
        
    # Get all equilibria for early stopping detection
    all_eqs_list = list(equilibria_dict.values()) if equilibria_dict else stable_equilibria
    
    if integrator == "abm":
        t_arr, x_arr, status = caputo_abm_integrate(
            system.evaluate_rhs, x0_arr, q=active_q, h=h, t_final=t_final,
            divergence_norm=divergence_norm, system=system,
            memory_mode=memory_mode, memory_window_length=memory_window_length,
            early_stop_config=early_stop_config, equilibria=all_eqs_list
        )
    else: # efork
        t_arr, x_arr, status = efork_integrate(
            system, x0_arr, q=active_q, h=h, t_final=t_final,
            memory_mode=memory_mode, memory_window_length=memory_window_length,
            divergence_norm=divergence_norm,
            early_stop_config=early_stop_config, equilibria=all_eqs_list
        )
            
    # 2. Check solver status for divergence and exceptions
    if status in ("diverged", "diverged_early", "nonfinite_solution"):
        return {"destination": "divergence", "status": status, "trajectory": x_arr}
    elif status.startswith("solver_exception"):
        return {"destination": "numerical_failure", "status": status, "trajectory": x_arr}
        
    # 3. Check early convergence to stable equilibrium
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
        
    # Get tail
    n_burn = int(np.ceil(t_burn / h))
    if len(x_arr) <= n_burn:
        n_burn = int(len(x_arr) * 0.5)
    tail = x_arr[n_burn:]
    
    # 4. Check convergence to stable equilibrium (at the end)
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
            
    # 5. Check convergence to target attractor
    if evaluate_target_match(tail, ref_tail, metric=target_match_metric, tolerance=target_match_tol):
        return {"destination": "target_attractor", "status": "ok", "trajectory": x_arr}
        
    # 6. Otherwise other attractor
    return {"destination": "other_attractor", "status": "ok", "trajectory": x_arr}
