import argparse
import os
import json
import numpy as np
from pathlib import Path
from .renderers import render_attractor, render_basin, render_nyquist, render_matignon

def render_all_plots(trajectory=None, equilibria=None, basin_grid=None, grid_x=None, grid_y=None, 
                     freqs=None, w_evals=None, n_evals=None, candidates=None, eigenvalues=None,
                     config=None, run_id="default_run", report_targets=None):
    """
    Programmatic entry point to render and export all available figures.
    """
    if config is None:
        config = {}
    if report_targets is None:
        report_targets = []
        
    outputs = {}
    
    # 1. Attractor trajectories
    if trajectory is not None:
        print(f"Rendering attractor trajectories for run {run_id}...")
        outputs["attractor"] = render_attractor(
            trajectory=trajectory,
            equilibria=equilibria or {},
            config=config,
            run_id=run_id,
            report_targets=report_targets
        )
        
    # 2. Basin of attraction
    if basin_grid is not None and grid_x is not None and grid_y is not None:
        print(f"Rendering basin of attraction for run {run_id}...")
        pdf, png = render_basin(
            grid_x=grid_x,
            grid_y=grid_y,
            basin_grid=basin_grid,
            config=config,
            run_id=run_id,
            report_targets=report_targets
        )
        outputs["basin"] = {"pdf": pdf, "png": png}
        
    # 3. Nyquist / Describing Function
    if freqs is not None and w_evals is not None and n_evals is not None:
        print(f"Rendering Nyquist / Describing Function for run {run_id}...")
        pdf, png = render_nyquist(
            freqs=freqs,
            w_evals=w_evals,
            n_evals=n_evals,
            candidates=candidates,
            config=config,
            run_id=run_id,
            report_targets=report_targets
        )
        outputs["nyquist"] = {"pdf": pdf, "png": png}
        
    # 4. Matignon Stability Sector
    if eigenvalues is not None:
        print(f"Rendering Matignon Stability sector for run {run_id}...")
        q = float(config.get("q", config.get("system_params", {}).get("q", 1.0)))
        pdf, png = render_matignon(
            eigenvalues=eigenvalues,
            q=q,
            config=config,
            run_id=run_id,
            report_targets=report_targets
        )
        outputs["matignon"] = {"pdf": pdf, "png": png}
        
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate all figures for a specific run.")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID to process")
    parser.add_argument("--data-dir", type=str, default="version_2/outputs", help="Directory where run outputs are stored")
    args = parser.parse_args()
    
    # Try to locate the run files and load them
    run_path = Path(args.data_dir) / args.run_id
    if not run_path.exists():
        print(f"Error: Run directory {run_path} does not exist.")
        exit(1)
        
    # Load configuration
    config_file = run_path / "config.json"
    config = {}
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
            
    # Load trajectory
    traj_file = run_path / "trajectory.csv"
    trajectory = None
    if traj_file.exists():
        try:
            # Load with numpy
            trajectory = np.loadtxt(traj_file, delimiter=",", skiprows=1)
        except Exception as e:
            print(f"Could not load trajectory: {e}")
            
    # Load equilibria
    eq_file = run_path / "equilibria.json"
    equilibria = {}
    if eq_file.exists():
        try:
            with open(eq_file, "r", encoding="utf-8") as f:
                equilibria = json.load(f)
                # Convert list to array
                equilibria = {k: np.array(v) for k, v in equilibria.items()}
        except Exception as e:
            print(f"Could not load equilibria: {e}")
            
    # Render if trajectory found
    if trajectory is not None:
        render_all_plots(
            trajectory=trajectory,
            equilibria=equilibria,
            config=config,
            run_id=args.run_id
        )
        print("Regeneration complete.")
    else:
        print("No trajectory found in run directory. Cannot render.")
