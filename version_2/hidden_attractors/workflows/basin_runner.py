"""Basin runner workflow.

Stability: experimental

Generates 2D basin-of-attraction slices for xy, xz, and yz planes
using parallel trajectory integrations and a target attractor reference.
"""

from __future__ import annotations

import csv
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from hidden_attractors.systems import get_system
from hidden_attractors.integrations.selector import integrate
from hidden_attractors.plotting.basin import plot_basin_slice_file
from hidden_attractors.plotting.matignon import classify_equilibrium_stability
from hidden_attractors.workflows.config_loader import save_effective_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evaluate_target_match(
    trajectory_tail: np.ndarray,
    ref_tail: np.ndarray,
    metric: str = "centroid_distance",
    tolerance: float = 0.5,
    nn_percentile: float = 90.0,
) -> bool:
    """Evaluate if the trajectory tail matches the reference attractor."""
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

    return False


def _classify_point_worker(args: Tuple) -> Tuple[int, int, int]:
    """Helper worker to run a single point integration and classify its destination."""
    i, j, x0, system, integrator, q, h, t_final, t_burn, ref_tail, stable_eqs, eq_tol, div_norm, metric, tol, memory_mode, memory_window, use_c_backend = args

    try:
        times, states, status = integrate(
            rhs=system.rhs,
            x0=x0,
            q=q,
            h=h,
            t_final=t_final,
            integrator=integrator,
            memory_mode=memory_mode,
            memory_window_length=memory_window,
            divergence_norm=div_norm,
            system=system,
            use_c_backend=use_c_backend,
            allow_python_fallback=True,
        )

        # 1. Divergence or failure
        if status in ("diverged", "diverged_early", "nonfinite_solution"):
            return i, j, 3
        elif status.startswith("solver_exception"):
            return i, j, 4

        # 2. Check early convergence
        if status == "converged_equilibrium_early":
            return i, j, 0

        # 3. Check tail convergence
        n_burn = int(math.ceil(t_burn / h))
        if len(states) <= n_burn:
            n_burn = int(len(states) * 0.5)
        tail = states[n_burn:]

        final_state = states[-1]
        for eq in stable_eqs:
            if np.linalg.norm(final_state - eq) <= eq_tol:
                return i, j, 0

        # 4. Check target match
        if _evaluate_target_match(tail, ref_tail, metric=metric, tolerance=tol):
            return i, j, 1

        return i, j, 2

    except Exception:
        return i, j, 4


# ---------------------------------------------------------------------------
# Core Workflow
# ---------------------------------------------------------------------------

def run_basin_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the basin of attraction slice generation and plotting.

    Parameters
    ----------
    config : Dict[str, Any]
        Normalized workflow configuration.

    Returns
    -------
    summary : Dict[str, Any]
        Results metadata.
    """
    import dataclasses

    basin_enabled = config.get("run_basin_slices", False) or config.get("basin", {}).get("enabled", False)
    if not basin_enabled:
        raise ValueError(
            "Basin workflow execution requested, but basin stages are disabled in config "
            "(stages.basin_slices / run_basin_slices or basin.enabled)."
        )

    system_id = config.get("system_id", "chua_fractional_saturation")
    integrator = config.get("integrator", "efork3")
    q = config.get("q")
    h = float(config.get("h", 0.001))
    output_dir = Path(config.get("output_dir", "outputs"))

    # System set up
    system = get_system(system_id)
    
    # Merge custom parameters from config and validate
    system_params = {}
    allowed_override_params = ["alpha", "beta", "gamma", "m0", "m1", "a1", "a2", "rho"]
    for p_name in allowed_override_params:
        if p_name in config and config[p_name] is not None:
            if p_name not in system.parameters:
                raise ValueError(
                    f"Parameter '{p_name}' is not a valid parameter for system '{system.name}'. "
                    f"Allowed parameters: {list(system.parameters.keys())}"
                )
            system_params[p_name] = config[p_name]
    
    if system_params:
        merged_params = dict(system.parameters)
        merged_params.update(system_params)
        system = dataclasses.replace(system, parameters=merged_params)

    if q is None:
        q = float(system.parameters.get("q", 0.99))
    else:
        q = float(q)

    # Load initial conditions
    ics = config.get("initial_conditions", {})
    if not ics:
        single_ic = config.get("final_simulation", {}).get("initial_condition")
        if single_ic is not None:
            ics = {"x0": single_ic}
        else:
            ics = {"x0_plus": [13.0, 0.7, -19.0]}

    ref_label = list(ics.keys())[0]
    ref_x0 = np.asarray(ics[ref_label], dtype=float)

    # Simulation settings for the reference trajectory
    t_final_ref = float(config.get("final_simulation", {}).get("t_final", 500.0))
    t_burn_ref = float(config.get("final_simulation", {}).get("t_burn", 120.0))
    div_norm = float(config.get("final_simulation", {}).get("divergence_norm", 120.0))
    
    memory_mode = config.get("memory_mode", "full")
    memory_window = config.get("memory_window_steps") or config.get("memory_window_length")
    use_c_backend = config.get("use_c_backend", True)
    allow_python_fallback = config.get("allow_python_fallback", True)
    early_stop_config = config.get("early_stop")

    # Integrate the target reference trajectory (single time)
    print(f"  Generating target reference trajectory from {ref_label}...")
    times_ref, states_ref, status_ref = integrate(
        rhs=system.rhs,
        x0=ref_x0,
        q=q,
        h=h,
        t_final=t_final_ref,
        integrator=integrator,
        system=system,
        memory_mode=memory_mode,
        memory_window_length=memory_window,
        divergence_norm=div_norm,
        use_c_backend=use_c_backend,
        allow_python_fallback=allow_python_fallback,
        early_stop_config=early_stop_config,
        equilibria=list(system.equilibrium_points().values()),
    )
    if status_ref != "ok":
        raise RuntimeError(f"Reference trajectory generation failed with status: {status_ref}")

    n_burn_ref = int(math.ceil(t_burn_ref / h))
    ref_tail = states_ref[n_burn_ref:]

    # Save reference attractor trajectory
    ref_csv_path = output_dir / "reference_attractor.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(ref_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z"])
        for t_val, state in zip(times_ref[n_burn_ref:], ref_tail):
            writer.writerow([t_val, state[0], state[1], state[2]])
    print(f"  Reference attractor saved -> {ref_csv_path}")

    # Resolve stable equilibria
    all_eqs = system.equilibrium_points()
    stable_eqs = []
    for name, eq_pt in all_eqs.items():
        res = classify_equilibrium_stability(system, eq_pt, q)
        if res["stable"]:
            stable_eqs.append(eq_pt)
    print(f"  Found {len(stable_eqs)} stable equilibria out of {len(all_eqs)} total.")

    # Basin config parameters
    b_cfg = config.get("basin", {})
    planes = b_cfg.get("planes", ["xy", "xz", "yz"])
    grid_n = int(b_cfg.get("grid_n", 150))
    t_final = float(b_cfg.get("t_final", 80.0))
    t_burn = float(b_cfg.get("t_burn", 20.0))
    h_basin = float(b_cfg.get("h", 0.01))
    
    x_interval = b_cfg.get("x_interval", [-10.0, 10.0])
    y_interval = b_cfg.get("y_interval", [-10.0, 10.0])
    z_interval = b_cfg.get("z_interval", [-10.0, 10.0])

    around_equilibria = bool(b_cfg.get("around_equilibria", True))
    local_radius = float(b_cfg.get("local_radius", 2.0))
    
    fixed_x = float(b_cfg.get("fixed_x", 0.0))
    fixed_y = float(b_cfg.get("fixed_y", 0.0))
    fixed_z = float(b_cfg.get("fixed_z", 0.0))

    # Center is defined as candidate seed or first equilibrium E0
    cx, cy, cz = 0.0, 0.0, 0.0
    if len(all_eqs) > 0:
        center_eq = all_eqs.get("E0", list(all_eqs.values())[0])
        cx, cy, cz = center_eq

    eq_tol = float(config.get("equilibrium_tol", 0.5))
    metric = config.get("target_match_metric", "centroid_distance")
    tol = float(config.get("target_match_tol", 0.5))
    workers = int(config.get("workers", 1))

    # Determine selection targets
    run_targets = []
    eq_selection = b_cfg.get("equilibrium_selection", "all")
    if around_equilibria:
        if eq_selection == "all":
            for eq_name, eq_pt in all_eqs.items():
                run_targets.append((eq_name, eq_pt))
        elif isinstance(eq_selection, list):
            for eq_name in eq_selection:
                if eq_name in all_eqs:
                    run_targets.append((eq_name, all_eqs[eq_name]))
        elif isinstance(eq_selection, str):
            if eq_selection in all_eqs:
                run_targets.append((eq_selection, all_eqs[eq_selection]))
    else:
        run_targets.append(("global", np.array([0.0, 0.0, 0.0])))

    if not run_targets:
        run_targets.append(("global", np.array([0.0, 0.0, 0.0])))

    print()
    print("=" * 72)
    print(" basin_slices - version_2")
    print("=" * 72)
    print(f"  system           = {system_id}")
    print(f"  planes           = {planes}")
    print(f"  targets          = {[t[0] for t in run_targets]}")
    print(f"  grid_n           = {grid_n} ({grid_n * grid_n} points per slice)")
    print(f"  t_final          = {t_final:.1f}")
    print(f"  integrator       = {integrator}")
    print(f"  workers          = {workers}")
    print(f"  output_dir       = {output_dir}")
    print("=" * 72)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    CLASS_LABELS = {
        0: "equilibrium",
        1: "attractor",
        2: "other",
        3: "divergence",
        4: "failure",
    }
    
    csv_rows = []
    summary_runs = []

    for eq_name, center_pt in run_targets:
        cx, cy, cz = center_pt
        for plane in planes:
            # Determine intervals
            if around_equilibria and eq_name != "global":
                u_lims = [cx - local_radius, cx + local_radius]
                v_lims = [cy - local_radius, cy + local_radius]
                w_lims = [cz - local_radius, cz + local_radius]
            else:
                u_lims = x_interval
                v_lims = y_interval
                w_lims = z_interval

            if plane == "xy":
                u_grid = np.linspace(u_lims[0], u_lims[1], grid_n)
                v_grid = np.linspace(v_lims[0], v_lims[1], grid_n)
                z_fixed = fixed_z
            elif plane == "xz":
                u_grid = np.linspace(u_lims[0], u_lims[1], grid_n)
                v_grid = np.linspace(w_lims[0], w_lims[1], grid_n)
                y_fixed = fixed_y
            elif plane == "yz":
                u_grid = np.linspace(v_lims[0], v_lims[1], grid_n)
                v_grid = np.linspace(w_lims[0], w_lims[1], grid_n)
                x_fixed = fixed_x
            else:
                raise ValueError(f"Unknown plane: {plane}")

            matrix = np.zeros((grid_n, grid_n), dtype=int)
            payloads = []

            for i, u in enumerate(u_grid):
                for j, v in enumerate(v_grid):
                    if plane == "xy":
                        x0_point = np.array([u, v, z_fixed], dtype=float)
                    elif plane == "xz":
                        x0_point = np.array([u, y_fixed, v], dtype=float)
                    elif plane == "yz":
                        x0_point = np.array([x_fixed, u, v], dtype=float)

                    payloads.append((
                        i, j, x0_point, system, integrator, q, h_basin, t_final, t_burn,
                        ref_tail, stable_eqs, eq_tol, div_norm, metric, tol, memory_mode, memory_window, use_c_backend
                    ))

            total_points = len(payloads)
            completed = 0
            stats = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

            def log_progress(count):
                if count % max(1, total_points // 20) == 0 or count == total_points:
                    pct = (count / total_points) * 100.0
                    print(f"  [{eq_name} - {plane}] {count}/{total_points} ({pct:.1f}%) | EQ={stats[0]} TARGET={stats[1]} OTHER={stats[2]} DIV={stats[3]} FAIL={stats[4]}")

            print(f"  Sweeping plane {plane.upper()} for equilibrium {eq_name}...")
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(_classify_point_worker, p) for p in payloads]
                    for fut in as_completed(futures):
                        i, j, code = fut.result()
                        matrix[i, j] = code
                        stats[code] += 1
                        completed += 1
                        log_progress(completed)
            else:
                for p in payloads:
                    i, j, code = _classify_point_worker(p)
                    matrix[i, j] = code
                    stats[code] += 1
                    completed += 1
                    log_progress(completed)

            # Build detailed CSV rows for this plane-equilibrium pair
            for p in payloads:
                i, j = p[0], p[1]
                x0_pt = p[2]
                code = matrix[i, j]
                label = CLASS_LABELS[code]
                u_val = float(u_grid[i])
                v_val = float(v_grid[j])
                csv_rows.append([
                    plane,
                    eq_name,
                    u_val,
                    v_val,
                    float(x0_pt[0]),
                    float(x0_pt[1]),
                    float(x0_pt[2]),
                    code,
                    label
                ])

            # Save numpy data
            npy_path = output_dir / f"basin_grid_{eq_name}_{plane}.npy"
            np.save(npy_path, matrix)

            # Plot
            plot_path = ""
            if config.get("plot_enabled", True) and config.get("plot_basin", True):
                plot_path = plot_basin_slice_file(
                    plane=plane,
                    u=u_grid,
                    v=v_grid,
                    mat=matrix,
                    eq_name=eq_name,
                    system_id=system_id,
                    output_dir=output_dir,
                )
                print(f"  Plot saved -> {plot_path}")

            summary_runs.append({
                "plane": plane,
                "equilibrium": eq_name,
                "grid_npy": str(npy_path),
                "plot_png": plot_path if plot_path else None,
                "counts": stats,
            })

    # Save CSV
    data_csv_path = output_dir / "basin_data.csv"
    with open(data_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "plane", "equilibrium", "u", "v", "x0", "y0", "z0", "class_code", "class_label"
        ])
        writer.writerows(csv_rows)
    print(f"  Detailed CSV saved -> {data_csv_path}")

    summary = {
        "workflow_mode": "basin",
        "system_id": system_id,
        "equilibrium_selection": eq_selection if around_equilibria else "global",
        "grid_n": grid_n,
        "targets_used": [t[0] for t in run_targets],
        "planes": planes,
        "integrator": integrator,
        "q": q,
        "memory_mode": memory_mode,
        "memory_window_length": memory_window,
        "reference_attractor_csv": str(ref_csv_path) if ref_csv_path.exists() else None,
        "data_csv": str(data_csv_path) if data_csv_path else None,
        "plot_paths": [run["plot_png"] for run in summary_runs if run.get("plot_png")],
        "results": summary_runs,
    }

    summary_path = output_dir / "basin_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved -> {summary_path}")

    save_effective_config(config, str(output_dir))
    return summary
