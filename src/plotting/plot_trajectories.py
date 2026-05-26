import os
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Any, Dict

def plot_attractor_trajectories(
    trajectory: np.ndarray,
    equilibria: Dict[str, np.ndarray],
    config: dict,
    output_dir: str
) -> None:
    """Generate and save 3D attractor phase space and 2D projection plots."""
    t = trajectory[:, 0]
    x = trajectory[:, 1]
    y = trajectory[:, 2]
    z = trajectory[:, 3]
    
    # We burn-in first part of the plot for clean attractor visualization
    n_burn = int(np.ceil(config["t_burn"] / config["h"]))
    if len(trajectory) > n_burn:
        x_plot = x[n_burn:]
        y_plot = y[n_burn:]
        z_plot = z[n_burn:]
    else:
        x_plot = x
        y_plot = y
        z_plot = z
        
    # --- 1. RENDER 3D PLOT ---
    fig_3d = plt.figure(figsize=(8, 7), dpi=300)
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.plot(x_plot, y_plot, z_plot, color='#10b981', linewidth=0.5, alpha=0.85, label='Candidate Attractor')
    
    # Mark equilibria
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
    fig_3d.savefig(os.path.join(output_dir, "attractor_3d.png"), dpi=300)
    plt.close(fig_3d)
    
    # --- 2. RENDER 2D PROJECTIONS (xy, xz, yz) ---
    fig_2d, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    projections = [
        ("xy", x_plot, y_plot, 'x', 'y'),
        ("xz", x_plot, z_plot, 'x', 'z'),
        ("yz", y_plot, z_plot, 'y', 'z')
    ]
    
    for idx, (proj_name, u, v, xlabel, ylabel) in enumerate(projections):
        ax = axes[idx]
        ax.grid(True, linestyle='--', linewidth=0.5, color='#cbd5e1')
        ax.plot(u, v, color='#10b981', linewidth=0.6, alpha=0.8)
        
        # Mark equilibria
        for name, eq in equilibria.items():
            color = '#ef4444' if name != 'E0' else '#3b82f6'
            marker = '^' if name == 'E0' else 'o'
            # Select coordinate index
            u_coord = eq[0] if xlabel == 'x' else eq[1]
            v_coord = eq[1] if ylabel == 'y' else eq[2]
            ax.scatter([u_coord], [v_coord], color=color, marker=marker, s=40, edgecolors='black', zorder=5, label=name if idx == 0 else "")
            
        ax.set_title(f"Projection {proj_name.upper()}", fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if idx == 0:
            ax.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
            
    plt.tight_layout()
    fig_2d.savefig(os.path.join(output_dir, "attractor_projections.png"), dpi=300)
    plt.close(fig_2d)


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
    fig = plt.figure(figsize=(9, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    # We burn-in target attractor
    n_burn = int(np.ceil(config["t_burn"] / config["h"]))
    if len(target_trajectory) > n_burn:
        target_pts = target_trajectory[n_burn:, 1:]
    else:
        target_pts = target_trajectory[:, 1:]
        
    # Sample target for clean rendering
    if len(target_pts) > max_target_points:
        indices = np.linspace(0, len(target_pts) - 1, max_target_points, dtype=int)
        target_pts = target_pts[indices]
        
    # Plot target attractor in vibrant green
    ax.plot(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2], color='#10b981', lw=0.8, alpha=0.9, label='Target Attractor')
    
    # Plot neighborhood probes
    plotted_hit = False
    plotted_miss = False
    
    for r in probe_results:
        traj = r.get("trajectory")
        if traj is None or len(traj) == 0:
            continue
            
        # Sample probe trajectory
        if len(traj) > max_probe_points:
            indices = np.linspace(0, len(traj) - 1, max_probe_points, dtype=int)
            traj_sampled = traj[indices]
        else:
            traj_sampled = traj
            
        # Red if target hit, blue/purple if missed (other attractor or equilibrium)
        if r["destination"] == "target_attractor":
            color = '#ef4444' # red
            label = "Probe (Target Hit)" if not plotted_hit else ""
            plotted_hit = True
            alpha = 0.8
            lw = 0.5
        else:
            color = '#3b82f6' # blue
            label = "Probe (Alternative/Stable)" if not plotted_miss else ""
            plotted_miss = True
            alpha = 0.4
            lw = 0.4
            
        ax.plot(traj_sampled[:, 0], traj_sampled[:, 1], traj_sampled[:, 2], color=color, lw=lw, alpha=alpha, label=label)
        
    # Mark equilibria
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
    fig.savefig(os.path.join(output_dir, "hiddenness_control_spheres.png"), dpi=300)
    plt.close(fig)

