"""Bifurcation diagram and parameter sweep workflow.

Stability: experimental

This workflow sweeps a system parameter and extracts local extrema of the steady-state
trajectories to build a bifurcation diagram.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hidden_attractors.systems import get_system
from hidden_attractors.integrations.selector import integrate
from hidden_attractors.analysis.bifurcation import (
    BifurcationPoint,
    bifurcation_points_from_trajectories,
    bifurcation_summary,
    observable_column,
    trajectory_tail,
    local_extrema,
)
from hidden_attractors.plotting.dynamics import plot_bifurcation_diagram
from hidden_attractors.workflows.config_loader import save_effective_config


def run_bifurcation_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the parameter sweep and bifurcation diagram workflow.

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

    system_id = config.get("system_id", "chua_fractional_saturation")
    integrator = config.get("integrator", "efork3")
    q = config.get("q")
    output_dir = Path(config.get("output_dir", "outputs"))
    
    bif_cfg = config.get("bifurcation", {})
    param_name = bif_cfg.get("parameter", "beta")
    val_cfg = bif_cfg.get("values", {"min": 8.0, "max": 16.0, "n": 100})
    p_min = float(val_cfg.get("min", 8.0))
    p_max = float(val_cfg.get("max", 16.0))
    p_n = int(val_cfg.get("n", 100))
    
    continuation = bool(bif_cfg.get("continuation_between_values", True))
    x0_start = np.asarray(bif_cfg.get("initial_condition", [0.1, 0.0, 0.0]), dtype=float)
    
    discard_time = float(bif_cfg.get("discard_time", 200.0))
    sample_time = float(bif_cfg.get("sample_time", 200.0))
    t_final = discard_time + sample_time
    h = float(bif_cfg.get("h", 0.01))
    
    observable = bif_cfg.get("coordinate", "x")
    sampling_method = bif_cfg.get("sampling", {}).get("method", "local_maxima")
    max_pts = int(bif_cfg.get("sampling", {}).get("max_points_per_parameter", 200))
    
    # Set up system
    system = get_system(system_id)
    if param_name != "q" and param_name not in system.parameters:
        raise ValueError(
            f"Swept parameter '{param_name}' is not a valid parameter for system '{system_id}'. "
            f"Allowed: {list(system.parameters.keys())} or 'q'."
        )

    if q is None:
        q = float(system.parameters.get("q", 0.99))
    else:
        q = float(q)
        
    # Get base parameters
    base_params = {}
    for p_key in system.parameters:
        if p_key in config and config[p_key] is not None:
            base_params[p_key] = config[p_key]
        else:
            base_params[p_key] = system.parameters[p_key]

    print()
    print("=" * 72)
    print(" bifurcation - version_2")
    print("=" * 72)
    print(f"  system           = {system_id}")
    print(f"  parameter        = {param_name}")
    print(f"  range            = [{p_min}, {p_max}] ({p_n} points)")
    print(f"  continuation     = {continuation}")
    print(f"  integrator       = {integrator}")
    print(f"  t_final          = {t_final:.1f} (discard {discard_time:.1f})")
    print(f"  output_dir       = {output_dir}")
    print("=" * 72)

    output_dir.mkdir(parents=True, exist_ok=True)
    p_vals = np.linspace(p_min, p_max, p_n)
    
    x0 = x0_start.copy()
    scans_data = []

    # Get settings to forward to integrate
    memory_mode = config.get("memory_mode", "full")
    memory_window_length = config.get("memory_window_steps") or config.get("memory_window_length", 400)
    divergence_norm = config.get("divergence_norm", 120.0)
    early_stop_config = config.get("early_stop")
    use_c_backend = config.get("use_c_backend", True)
    allow_python_fallback = config.get("allow_python_fallback", True)

    from hidden_attractors.integrations.selector import validate_integrator_compatibility

    n_success = 0
    n_failed = 0
    failed_parameter_values = []

    for i, p_val in enumerate(p_vals):
        # Override the swept parameter
        q_val = q
        run_params = dict(base_params)
        if param_name == "q":
            q_val = float(p_val)
        else:
            run_params[param_name] = p_val
        
        # Merge into the system copy
        system_copy = dataclasses.replace(system, parameters=run_params)
        
        try:
            # Validate integrator/q compatibility
            validate_integrator_compatibility(integrator, q_val)
            
            # Integrate
            times, states, status = integrate(
                rhs=system_copy.rhs,
                x0=x0,
                q=q_val,
                h=h,
                t_final=t_final,
                integrator=integrator,
                system=system_copy,
                memory_mode=memory_mode,
                memory_window_length=memory_window_length,
                divergence_norm=divergence_norm,
                use_c_backend=use_c_backend,
                allow_python_fallback=allow_python_fallback,
                early_stop_config=early_stop_config,
                equilibria=list(system_copy.equilibrium_points().values())
            )
            
            if status == "ok" and len(times) > 0:
                traj = np.column_stack([times, states])
                scans_data.append((p_val, traj, status))
                n_success += 1
                if continuation:
                    x0 = states[-1].copy()
            else:
                n_failed += 1
                failed_parameter_values.append(float(p_val))
                x0 = x0_start.copy()
                if len(times) > 0:
                    traj = np.column_stack([times, states])
                    scans_data.append((p_val, traj, status))
        except Exception as exc:
            n_failed += 1
            failed_parameter_values.append(float(p_val))
            x0 = x0_start.copy()
            print(f"  Point {p_val} failed: {exc}")
            
        if (i + 1) % max(1, p_n // 10) == 0 or i == p_n - 1:
            print(f"  Progress: {i+1}/{p_n} points swept.")

    # Extract bifurcation diagram points
    print("  Extracting local extrema...")
    if sampling_method == "poincare":
        raise NotImplementedError("Poincare section sampling is not implemented.")

    mode_map = {
        "local_maxima": "maxima",
        "local_minima": "minima",
        "both": "both",
        "raw_tail_samples": "sample",
    }
    if sampling_method not in mode_map:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
        
    extrema_mode = mode_map[sampling_method]
    col_idx = observable_column(observable)
    
    pts = []
    csv_rows = []
    
    for p_val, trajectory, run_status in scans_data:
        tail = trajectory_tail(trajectory, t_start=discard_time)
        if tail.size == 0:
            continue
        if col_idx >= tail.shape[1]:
            raise ValueError(f"observable column {col_idx} exceeds trajectory width {tail.shape[1]}")
            
        idx = local_extrema(tail[:, col_idx], mode=extrema_mode)
        if max_pts > 0 and idx.size > max_pts:
            select = np.linspace(0, idx.size - 1, int(max_pts), dtype=int)
            idx = idx[select]
            
        for idx_val in idx:
            t = float(tail[idx_val, 0])
            x = float(tail[idx_val, 1])
            y = float(tail[idx_val, 2])
            z = float(tail[idx_val, 3])
            val = float(tail[idx_val, col_idx])
            
            pts.append(
                BifurcationPoint(
                    parameter=p_val,
                    observable=val,
                    time=t,
                    index=int(idx_val),
                    kind=extrema_mode,
                )
            )
            csv_rows.append([
                param_name,
                p_val,
                t,
                x,
                y,
                z,
                observable,
                val,
                sampling_method,
                run_status
            ])

    save_csv = bif_cfg.get("save_csv", True)
    save_plot = bif_cfg.get("save_plot", True)
    plots_enabled = config.get("plot_enabled", True)
    plot_bifurcation = config.get("plot_bifurcation", True)

    # Save CSV
    data_csv_path = None
    if save_csv:
        data_csv_path = output_dir / "bifurcation_data.csv"
        with open(data_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "parameter_name", "parameter_value", "t", "x", "y", "z",
                "coordinate", "coordinate_value", "sample_type", "status"
            ])
            writer.writerows(csv_rows)
        print(f"  Data saved -> {data_csv_path}")

    # Plot
    plot_path = output_dir / "bifurcation_plot.png"
    should_plot = save_plot and plots_enabled and plot_bifurcation
    if should_plot:
        plot_bifurcation_diagram(
            pts,
            plot_path,
            parameter_label=param_name,
            observable_label=f"{observable} ({sampling_method})",
            title=f"Bifurcation Diagram: {system_id}",
        )
        print(f"  Plot saved -> {plot_path}")

    # Save summary
    stats = bifurcation_summary(pts)
    summary = {
        "workflow_mode": "bifurcation",
        "system_id": system_id,
        "parameter_swept": param_name,
        "n_swept_points": p_n,
        "n_success": n_success,
        "n_failed": n_failed,
        "failed_parameter_values": failed_parameter_values,
        "integrator": integrator,
        "memory_mode": memory_mode,
        "memory_window_length": memory_window_length,
        "q_base": q,
        "stats": stats,
        "data_csv": str(data_csv_path) if data_csv_path else None,
        "plot_path": str(plot_path) if (should_plot and plot_path.exists()) else None,
    }
    
    summary_path = output_dir / "bifurcation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved -> {summary_path}")
    
    save_effective_config(config, str(output_dir))
    return summary
