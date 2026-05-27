import os
import csv
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Dict, Any

def plot_attractor_trajectories(
    trajectory: np.ndarray,
    equilibria: Dict[str, np.ndarray],
    config: dict,
    output_dir: str
) -> None:
    """Generate and save 3D attractor phase space and 2D projection plots."""
    # Keep legacy support by calling the flexible implementation
    plot_flexible_attractor_and_projections(
        trajectory=trajectory,
        equilibria=equilibria,
        config=config,
        output_dir=output_dir,
        file_prefix="attractor"
    )

def plot_flexible_attractor_and_projections(
    trajectory: np.ndarray,
    equilibria: Dict[str, np.ndarray],
    config: dict,
    output_dir: str,
    file_prefix: str
) -> None:
    """
    Saves a 3D attractor plot and three individual 2D projections (xy, xz, yz)
    into the designated output directory under the specified prefix.
    """
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    t = trajectory[:, 0]
    x = trajectory[:, 1]
    y = trajectory[:, 2]
    z = trajectory[:, 3]
    
    # Burn-in transient states
    n_burn = int(np.ceil(config.get("final_simulation", {}).get("t_burn", config.get("t_burn", 120.0)) / config["h"]))
    if len(trajectory) > n_burn:
        x_plot = x[n_burn:]
        y_plot = y[n_burn:]
        z_plot = z[n_burn:]
    else:
        x_plot = x
        y_plot = y
        z_plot = z
        
    lw = config.get("attractor_plots", {}).get("line_width", 0.7)
    ps = config.get("attractor_plots", {}).get("point_size", 0.0)
    include_eq = config.get("attractor_plots", {}).get("include_equilibria", False)
    
    # --- 1. RENDER 3D PLOT ---
    fig_3d = plt.figure(figsize=(8, 7), dpi=300)
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.plot(x_plot, y_plot, z_plot, color='#10b981', linewidth=lw, alpha=0.85, label='Attractor')
    
    if ps > 0.0:
        ax_3d.scatter(x_plot, y_plot, z_plot, color='#10b981', s=ps, alpha=0.5)
    
    # Mark equilibria if enabled
    if include_eq:
        for name, eq in equilibria.items():
            color = '#ef4444' if name != 'E0' else '#3b82f6'
            marker = '^' if name == 'E0' else 'o'
            ax_3d.scatter([eq[0]], [eq[1]], [eq[2]], color=color, marker=marker, s=50, edgecolors='black', zorder=5, label=name)
        
    ax_3d.set_title(f"3D Phase Space - {config['system_id']}", fontsize=12, fontweight='bold', pad=15)
    ax_3d.set_xlabel('x', fontsize=10)
    ax_3d.set_ylabel('y', fontsize=10)
    ax_3d.set_zlabel('z', fontsize=10)
    ax_3d.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    filename_3d = f"{file_prefix}_3d.png" if "candidate" not in file_prefix else f"{file_prefix}_attractor_3d.png"
    fig_3d.savefig(os.path.join(fig_dir, filename_3d), dpi=300)
    plt.close(fig_3d)
    
    # --- 2. RENDER INDIVIDUAL 2D PROJECTIONS ---
    projections = [
        ("xy", x_plot, y_plot, 'x', 'y'),
        ("xz", x_plot, z_plot, 'x', 'z'),
        ("yz", y_plot, z_plot, 'y', 'z')
    ]
    
    for proj_name, u, v, xlabel, ylabel in projections:
        fig_2d = plt.figure(figsize=(7, 6), dpi=300)
        ax = fig_2d.add_subplot(111)
        ax.grid(True, linestyle='--', linewidth=0.5, color='#cbd5e1')
        ax.plot(u, v, color='#10b981', linewidth=lw, alpha=0.8)
        
        if ps > 0.0:
            ax.scatter(u, v, color='#10b981', s=ps, alpha=0.5)
        
        # Mark equilibria if enabled
        if include_eq:
            for name, eq in equilibria.items():
                color = '#ef4444' if name != 'E0' else '#3b82f6'
                marker = '^' if name == 'E0' else 'o'
                u_coord = eq[0] if xlabel == 'x' else eq[1]
                v_coord = eq[1] if ylabel == 'y' else eq[2]
                ax.scatter([u_coord], [v_coord], color=color, marker=marker, s=50, edgecolors='black', zorder=5, label=name)
            
        ax.set_title(f"Projection {proj_name.upper()} - {config['system_id']}", fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
        
        plt.tight_layout()
        fig_2d.savefig(os.path.join(fig_dir, f"{file_prefix}_{proj_name}.png"), dpi=300)
        plt.close(fig_2d)

def plot_timeseries_data(
    trajectory: np.ndarray,
    config: dict,
    output_dir: str,
    file_prefix: str
) -> None:
    """Generate time series plots (x, y, z, and combined xyz) and save states as CSV."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    t = trajectory[:, 0]
    x = trajectory[:, 1]
    y = trajectory[:, 2]
    z = trajectory[:, 3]
    
    # Save CSV
    csv_filename = f"{file_prefix}_timeseries.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z"])
        writer.writerows(trajectory.tolist())
        
    # Handle maximum timeseries points
    use_tail = config.get("timeseries_use_tail_after_burn", False)
    max_pts = config.get("timeseries_max_points", 20000)
    
    if use_tail:
        t_burn = config.get("final_simulation", {}).get("t_burn", config.get("t_burn", 120.0))
        n_burn = int(np.ceil(t_burn / config["h"]))
        if len(trajectory) > n_burn:
            t_plot = t[n_burn:]
            x_plot = x[n_burn:]
            y_plot = y[n_burn:]
            z_plot = z[n_burn:]
        else:
            t_plot, x_plot, y_plot, z_plot = t, x, y, z
    else:
        t_plot, x_plot, y_plot, z_plot = t, x, y, z
        
    if len(t_plot) > max_pts:
        indices = np.linspace(0, len(t_plot) - 1, max_pts, dtype=int)
        t_plot = t_plot[indices]
        x_plot = x_plot[indices]
        y_plot = y_plot[indices]
        z_plot = z_plot[indices]
        
    lw = config.get("attractor_plots", {}).get("line_width", 0.7)
    
    # Render x, y, z individual plots
    vars_data = [('x', x_plot, '#3b82f6'), ('y', y_plot, '#ef4444'), ('z', z_plot, '#10b981')]
    for var_name, var_val, color in vars_data:
        fig = plt.figure(figsize=(8, 4), dpi=300)
        ax = fig.add_subplot(111)
        ax.grid(True, linestyle='--', linewidth=0.5, color='#cbd5e1')
        ax.plot(t_plot, var_val, color=color, linewidth=lw, alpha=0.9)
        ax.set_title(f"Time Series {var_name.upper()} - {config['system_id']}", fontsize=11, fontweight='bold')
        ax.set_xlabel('t', fontsize=10)
        ax.set_ylabel(var_name, fontsize=10)
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{file_prefix}_timeseries_{var_name}.png"), dpi=300)
        plt.close(fig)
        
    # Render combined xyz plot
    fig = plt.figure(figsize=(10, 5), dpi=300)
    ax = fig.add_subplot(111)
    ax.grid(True, linestyle='--', linewidth=0.5, color='#cbd5e1')
    ax.plot(t_plot, x_plot, color='#3b82f6', linewidth=lw, alpha=0.85, label='x')
    ax.plot(t_plot, y_plot, color='#ef4444', linewidth=lw, alpha=0.85, label='y')
    ax.plot(t_plot, z_plot, color='#10b981', linewidth=lw, alpha=0.85, label='z')
    ax.set_title(f"Time Series XYZ - {config['system_id']}", fontsize=12, fontweight='bold')
    ax.set_xlabel('t', fontsize=10)
    ax.set_ylabel('State Variables', fontsize=10)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{file_prefix}_timeseries_xyz.png"), dpi=300)
    plt.close(fig)

def plot_neighborhood_control_spheres(
    target_trajectory: np.ndarray,
    probe_results: list,
    equilibria: Dict[str, np.ndarray],
    config: dict,
    output_dir: str,
    max_target_points: int = 5000,
    max_probe_points: int = 500
) -> None:
    """Generate a 3D control spheres plot showing target attractor and probes."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(9, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    n_burn = int(np.ceil(config.get("final_simulation", {}).get("t_burn", config.get("t_burn", 120.0)) / config["h"]))
    if len(target_trajectory) > n_burn:
        target_pts = target_trajectory[n_burn:, 1:]
    else:
        target_pts = target_trajectory[:, 1:]
        
    if len(target_pts) > max_target_points:
        indices = np.linspace(0, len(target_pts) - 1, max_target_points, dtype=int)
        target_pts = target_pts[indices]
        
    ax.plot(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2], color='#10b981', lw=0.8, alpha=0.9, label='Target Attractor')
    
    plotted_hit = False
    plotted_miss = False
    
    for r in probe_results:
        traj = r.get("trajectory")
        if traj is None or len(traj) == 0:
            continue
            
        if len(traj) > max_probe_points:
            indices = np.linspace(0, len(traj) - 1, max_probe_points, dtype=int)
            traj_sampled = traj[indices]
        else:
            traj_sampled = traj
            
        if r["destination"] == "target_attractor":
            color = '#ef4444'
            label = "Probe (Target Hit)" if not plotted_hit else ""
            plotted_hit = True
            alpha = 0.8
            lw = 0.5
        else:
            color = '#3b82f6'
            label = "Probe (Alternative/Stable)" if not plotted_miss else ""
            plotted_miss = True
            alpha = 0.4
            lw = 0.4
            
        # Check if the probe trajectory is 4D (with time) or 3D
        if traj_sampled.shape[1] == 4:
            ax.plot(traj_sampled[:, 1], traj_sampled[:, 2], traj_sampled[:, 3], color=color, lw=lw, alpha=alpha, label=label)
        else:
            ax.plot(traj_sampled[:, 0], traj_sampled[:, 1], traj_sampled[:, 2], color=color, lw=lw, alpha=alpha, label=label)
        
    for name, eq in equilibria.items():
        color = '#f97316' if name != 'E0' else '#3b82f6'
        marker = '^' if name == 'E0' else 'o'
        ax.scatter([eq[0]], [eq[1]], [eq[2]], color=color, marker=marker, s=60, edgecolors='black', zorder=5, label=name)
        
    ax.set_title(f"Hiddenness Control Spheres - {config['system_id']}", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_zlabel('z', fontsize=10)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "hiddenness_control_spheres.png"), dpi=300)
    plt.close(fig)
