import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Optional
from .hiddenness import run_neighborhood_probe
from .terminal_policy import PROBE_POLICY, OPERATIONAL_DESTINATIONS

def _classify_point_worker(args: Tuple) -> Tuple[int, int, int, Dict[str, Any]]:
    """Helper worker to run a single point simulation in parallel."""
    args_list = list(args)
    i = args_list[0]
    j = args_list[1]
    x0 = args_list[2]
    system = args_list[3]
    transfer_mode = args_list[4]
    integrator = args_list[5]
    t_final = args_list[6]
    t_burn = args_list[7]
    h = args_list[8]
    ref_tail = args_list[9]
    stable_eqs = args_list[10]
    eq_tol = args_list[11]
    div_norm = args_list[12]
    metric = args_list[13]
    tol = args_list[14]
    dynamics_mode = args_list[15]
    memory_mode = args_list[16]
    memory_window_length = args_list[17]
    early_stop_config = args_list[18]
    equilibria_dict = args_list[19]
    q_dynamics_effective = args_list[20] if len(args_list) > 20 else None
    policy = args_list[21] if len(args_list) > 21 else {}
    
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
            target_match_tol=tol,
            dynamics_mode=dynamics_mode,
            memory_mode=memory_mode,
            memory_window_length=memory_window_length,
            early_stop_config=early_stop_config,
            equilibria_dict=equilibria_dict,
            q_dynamics_effective=q_dynamics_effective,
            **policy
        )
        destination = res["destination"]
        mapping = {name: code for code, name in OPERATIONAL_DESTINATIONS.items()}
        mapping.update(stable_equilibrium=0, equilibrium_stable=0, target_attractor=1,
                       other_attractor=2, divergence=3)
        code = mapping.get(destination, 5)
        diagnostics = {**res.get("diagnostics", {}), "destination": destination,
                       "status": res["status"], "x0": np.asarray(x0).tolist()}
    except Exception as exc:
        code = 4
        diagnostics = {"classification_policy": policy.get("classification_policy", PROBE_POLICY),
                       "destination": "numerical_failure", "reason": f"{type(exc).__name__}:{exc}",
                       "x0": np.asarray(x0).tolist()}
    return i, j, code, diagnostics

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
    t_final: float = 100.0,
    t_burn: float = 40.0,
    h: float = 0.02,
    workers: int = 1,
    eq_tol: float = 0.5,
    div_norm: float = 120.0,
    metric: str = "centroid_distance",
    tol: float = 0.5,
    dynamics_mode: str = "system",
    memory_mode: str = "full",
    memory_window_length: Optional[int] = None,
    x_interval: Optional[List[float]] = None,
    y_interval: Optional[List[float]] = None,
    z_interval: Optional[List[float]] = None,
    around_equilibria: bool = False,
    local_radius: float = 2.0,
    eq_name: str = "global",
    system_id: str = "chua",
    early_stop_config: Optional[dict] = None,
    equilibria_dict: Optional[Dict[str, np.ndarray]] = None,
    q_dynamics_effective: Optional[float] = None, *,
    classification_policy: str = PROBE_POLICY,
    terminal_window: Optional[float] = None,
    equilibrium_span_tol: Optional[float] = None,
    target_match_nn_percentile: float = 90.0,
    return_diagnostics: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 2D basin slice mesh grid and evaluation matrix of classifications,
    with configurable intervals, centering, and real-time terminal progress printing.
    ``return_diagnostics=True`` appends per-point finite-policy diagnostics.
    """
    if q_dynamics_effective is None:
        import warnings
        warnings.warn("q_dynamics_effective is omitted, falling back to legacy dynamics_mode logic", UserWarning)
    if isinstance(grid_n, bool) or int(grid_n) != grid_n or grid_n < 2:
        raise ValueError("grid_n must be an integer at least 2.")
    grid_n = int(grid_n)
    policy = {"classification_policy": classification_policy,
              "terminal_window": terminal_window,
              "equilibrium_span_tol": equilibrium_span_tol,
              "target_match_nn_percentile": target_match_nn_percentile}
    diagnostic_records = []
    cx, cy, cz = center
    
    if around_equilibria:
        u_lims = [cx - local_radius, cx + local_radius]
        v_lims = [cy - local_radius, cy + local_radius]
        w_lims = [cz - local_radius, cz + local_radius]
    else:
        u_lims = x_interval if x_interval is not None else [cx - extent, cx + extent]
        v_lims = y_interval if y_interval is not None else [cy - extent, cy + extent]
        w_lims = z_interval if z_interval is not None else [cz - extent, cz + extent]
        
    if plane == "xy":
        u_grid = np.linspace(u_lims[0], u_lims[1], grid_n)
        v_grid = np.linspace(v_lims[0], v_lims[1], grid_n)
        z_fixed = fixed_values.get("z", cz)
    elif plane == "xz":
        u_grid = np.linspace(u_lims[0], u_lims[1], grid_n)
        v_grid = np.linspace(w_lims[0], w_lims[1], grid_n)
        y_fixed = fixed_values.get("y", cy)
    elif plane == "yz":
        u_grid = np.linspace(v_lims[0], v_lims[1], grid_n)
        v_grid = np.linspace(w_lims[0], w_lims[1], grid_n)
        x_fixed = fixed_values.get("x", cx)
    else:
        raise ValueError(f"Unknown plane: {plane}")
        
    matrix = np.zeros((grid_n, grid_n), dtype=int)
    
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
                i, j, x0, system, transfer_mode, integrator, t_final, t_burn, h, ref_tail, stable_eqs, eq_tol, div_norm, metric, tol,
                dynamics_mode, memory_mode, memory_window_length, early_stop_config, equilibria_dict, q_dynamics_effective, policy
            ))
            
    total_points = len(payloads)
    completed_points = 0
    
    stats = {code: 0 for code in OPERATIONAL_DESTINATIONS}
    last_printed_pct = -5.0
    
    def log_progress(count, force=False):
        nonlocal last_printed_pct
        pct = (count / total_points) * 100.0
        if force or (pct - last_printed_pct >= 5.0) or count == total_points:
            print(f"[{system_id}] Cuenca {plane} {eq_name}: {count}/{total_points}, {pct:.1f}%, TARGET={stats[1]}, EQ={stats[0]}, DIV={stats[3]}, OTHER={stats[2]}, FAIL={stats[4]}, INCONCLUSIVE={stats[5]}")
            last_printed_pct = pct
        
    log_progress(0, force=True)
    
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_classify_point_worker, payload) for payload in payloads]
            for fut in as_completed(futures):
                i, j, code, diagnostics = fut.result()
                diagnostic_records.append({"i": i, "j": j, **diagnostics})
                matrix[i, j] = code
                stats[code] += 1
                completed_points += 1
                log_progress(completed_points)
    else:
        for payload in payloads:
            i, j, code, diagnostics = _classify_point_worker(payload)
            diagnostic_records.append({"i": i, "j": j, **diagnostics})
            matrix[i, j] = code
            stats[code] += 1
            completed_points += 1
            log_progress(completed_points)
                
    log_progress(total_points, force=True)
            
    if return_diagnostics:
        diagnostic_records.sort(key=lambda record: (record["i"], record["j"]))
        return u_grid, v_grid, matrix, diagnostic_records
    return u_grid, v_grid, matrix
