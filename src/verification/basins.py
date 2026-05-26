import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple
from .hiddenness import run_neighborhood_probe

def _classify_point_worker(args: Tuple) -> Tuple[int, int, int]:
    """Helper worker to run a single point simulation in parallel."""
    (i, j, x0, system, transfer_mode, integrator, t_final, t_burn, h, ref_tail, stable_eqs, eq_tol, div_norm, metric, tol) = args
    try:
        res = run_neighborhood_probe(
            system=system,
            x0=x0,
            transfer_mode=transfer_mode,
            integrator=integrator,
            t_final=t_final,
            t_burn=t_burn,
            h=h,
            ref_tail=ref_tail,
            stable_equilibria=stable_eqs,
            equilibrium_tol=eq_tol,
            divergence_norm=div_norm,
            target_match_metric=metric,
            target_match_tol=tol
        )
        dest = res["destination"]
        if dest == "equilibrium_stable":
            code = 0
        elif dest == "target_attractor":
            code = 1
        elif dest == "other_attractor":
            code = 2
        elif dest == "divergence":
            code = 3
        else: # numerical_failure
            code = 4
    except Exception:
        code = 4
    return i, j, code

def generate_basin_slice(
    plane: str,
    system: Any,
    transfer_mode: str,
    integrator: str,
    ref_tail: np.ndarray,
    stable_eqs: List[np.ndarray],
    fixed_values: Dict[str, float],
    extent: float = 8.0,
    grid_n: int = 40,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    t_final: float = 100.0, # shorter times for basins to keep it fast
    t_burn: float = 40.0,
    h: float = 0.02,
    workers: int = 1,
    eq_tol: float = 0.5,
    div_norm: float = 120.0,
    metric: str = "centroid_distance",
    tol: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a 2D basin slice mesh grid and evaluation matrix of classifications.
    
    Returns:
        grid_u: 1D grid for first coordinate
        grid_v: 1D grid for second coordinate
        matrix: 2D array of integer codes (grid_n x grid_n)
    """
    grid_n = int(grid_n)
    cx, cy, cz = center
    
    if plane == "xy":
        u_grid = np.linspace(cx - extent, cx + extent, grid_n)
        v_grid = np.linspace(cy - extent, cy + extent, grid_n)
        z_fixed = fixed_values.get("z", cz)
    elif plane == "xz":
        u_grid = np.linspace(cx - extent, cx + extent, grid_n)
        v_grid = np.linspace(cz - extent, cz + extent, grid_n)
        y_fixed = fixed_values.get("y", cy)
    elif plane == "yz":
        u_grid = np.linspace(cy - extent, cy + extent, grid_n)
        v_grid = np.linspace(cz - extent, cz + extent, grid_n)
        x_fixed = fixed_values.get("x", cx)
    else:
        raise ValueError(f"Unknown plane: {plane}")
        
    matrix = np.zeros((grid_n, grid_n), dtype=int)
    
    # Build payload list
    payloads = []
    for i, u in enumerate(u_grid):
        for j, v in enumerate(v_grid):
            if plane == "xy":
                x0 = np.array([u, v, z_fixed], dtype=float)
            elif plane == "xz":
                x0 = np.array([u, y_fixed, v], dtype=float)
            elif plane == "yz":
                x0 = np.array([x_fixed, u, v], dtype=float)
            
            payloads.append((
                i, j, x0, system, transfer_mode, integrator, t_final, t_burn, h, ref_tail, stable_eqs, eq_tol, div_norm, metric, tol
            ))
            
    # Run in parallel if workers > 1
    if workers > 1:
        # Using ThreadPoolExecutor as standard fallback since it's robust in Windows within vscode envs
        # and has zero startup overhead compared to ProcessPool
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_classify_point_worker, payload) for payload in payloads]
            for fut in as_completed(futures):
                i, j, code = fut.result()
                matrix[i, j] = code
    else:
        for payload in payloads:
            i, j, code = _classify_point_worker(payload)
            matrix[i, j] = code
            
    return u_grid, v_grid, matrix
