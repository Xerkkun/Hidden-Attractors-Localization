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
    if q is None:
        q = float(system.parameters.get("q", 0.99))
    else:
        q = float(q)
        
    # Get base parameters
    base_params = {}
    for p_key in ["alpha", "beta", "gamma", "m", "n", "m0", "m1", "a1", "a2", "rho"]:
        if p_key in config and config[p_key] is not None:
            base_params[p_key] = config[p_key]

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

    for i, p_val in enumerate(p_vals):
        # Override the swept parameter
        run_params = dict(base_params)
        run_params[param_name] = p_val
        
        # Merge into the system copy
        system_copy = dataclasses.replace(system, parameters=run_params)
        
        # Integrate
        times, states, status = integrate(
            rhs=system_copy.rhs,
            x0=x0,
            q=q,
            h=h,
            t_final=t_final,
            integrator=integrator,
            system=system_copy,
            use_c_backend=config.get("use_c_backend", True),
            allow_python_fallback=config.get("allow_python_fallback", True),
        )
        
        if status in ("ok", "diverged", "diverged_early", "converged_equilibrium_early") and len(times) > 0:
            # Save the trajectory format (t, x, y, z)
            traj = np.column_stack([times, states])
            scans_data.append((p_val, traj))
            
            # If continuation is enabled and solve was stable, update x0
            if continuation and status == "ok":
                x0 = states[-1].copy()
        else:
            # If divergence or failure, reset to base IC for next iteration
            x0 = x0_start.copy()
            
        if (i + 1) % max(1, p_n // 10) == 0 or i == p_n - 1:
            print(f"  Progress: {i+1}/{p_n} points swept.")

    # Extract bifurcation diagram points
    print("  Extracting local extrema...")
    mode_map = {
        "local_maxima": "maxima",
        "local_minima": "minima",
        "both": "both",
    }
    extrema_mode = mode_map.get(sampling_method, "maxima")
    
    pts = bifurcation_points_from_trajectories(
        scans_data,
        parameter_key="parameter",
        observable=observable,
        t_start=discard_time,
        mode=extrema_mode,
        max_points_per_parameter=max_pts,
    )

    # Save CSV
    data_csv_path = output_dir / "bifurcation_data.csv"
    with open(data_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([param_name, f"{observable}_extremum"])
        for pt in pts:
            writer.writerow([pt.parameter, pt.observable])
    print(f"  Data saved -> {data_csv_path}")

    # Plot
    plot_path = output_dir / "bifurcation_plot.png"
    if config.get("plot_enabled", True) and config.get("plot_bifurcation", True):
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
        "stats": stats,
        "data_csv": str(data_csv_path),
        "plot_path": str(plot_path) if plot_path.exists() else None,
    }
    
    summary_path = output_dir / "bifurcation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved -> {summary_path}")
    
    save_effective_config(config, str(output_dir))
    return summary
