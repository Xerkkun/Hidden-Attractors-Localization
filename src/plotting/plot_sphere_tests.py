import os
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Any

def plot_sphere_test_results(
    eq_name: str,
    eq_pt: np.ndarray,
    radius: float,
    probe_runs: List[Dict[str, Any]],
    output_dir: str,
    trajectory_plot_fraction: float = 0.25,
    max_trajectories_to_plot: int = 60
) -> str:
    """
    Renders a premium 3D visualization of neighborhood sphere probes.
    Displays the central equilibrium, a wireframe/transparent sphere boundary,
    initial conditions colored by their destination, and short trajectory segments.
    """
    fig = plt.figure(figsize=(9, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    # Classifications and colors consistent with request
    color_map = {
        "target_attractor": "red",
        "stable_equilibrium": "blue",
        "divergence": "black",
        "other_attractor": "orange",
        "numerical_failure": "gray",
        "unclassified": "purple"
    }
    
    labels_map = {
        "target_attractor": "Target Attractor",
        "stable_equilibrium": "Stable Equilibrium",
        "divergence": "Divergence",
        "other_attractor": "Other Attractor",
        "numerical_failure": "Numerical Failure",
        "unclassified": "Unclassified"
    }

    # 1. DRAW CENTRAL EQUILIBRIUM
    ax.scatter([eq_pt[0]], [eq_pt[1]], [eq_pt[2]], color='#ef4444' if eq_name != 'E0' else '#3b82f6',
               marker='*', s=150, edgecolors='black', zorder=10, label=f"Equilibrium {eq_name}")

    # 2. DRAW TRANSPARENT SPHERE BOUNDARY
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + eq_pt[0]
    y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + eq_pt[1]
    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + eq_pt[2]
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='#64748b', alpha=0.1, edgecolor='#94a3b8', linewidth=0.3)

    # Track plotted classes for a clean legend
    plotted_classes = set()
    plotted_trajectories_count = 0

    # 3. RENDER EACH SAMPLE POINT AND TRAJECTORY
    for idx, run in enumerate(probe_runs):
        x0 = np.array(run["x0"])
        dest = run["destination"]
        color = color_map.get(dest, "purple")
        label = labels_map.get(dest, "Unclassified") if dest not in plotted_classes else ""
        plotted_classes.add(dest)

        # Plot initial condition point
        ax.scatter([x0[0]], [x0[1]], [x0[2]], color=color, s=25, edgecolors='black', zorder=8, label=label)

        # Plot trajectory segment if available and within plotting limit
        traj = run.get("trajectory")
        if traj is not None and len(traj) > 0 and plotted_trajectories_count < max_trajectories_to_plot:
            # Safe truncation of segment
            n_plot = int(max(2, np.ceil(len(traj) * trajectory_plot_fraction)))
            # If trajectory contains time column, slice properly
            if traj.shape[1] == 4:
                segment = traj[:n_plot, 1:]
            else:
                segment = traj[:n_plot, :]
            
            # Reduce transparency for divergence or failure to keep it clean
            alpha = 0.55 if dest == "target_attractor" else 0.3
            lw = 0.8 if dest == "target_attractor" else 0.5
            
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, lw=lw, alpha=alpha)
            plotted_trajectories_count += 1

    # Style the 3D axes
    ax.set_title(f"Sphere Test: {eq_name} | Radius: {radius}", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_zlabel('z', fontsize=10)
    
    # Center view around equilibrium and set symmetric spans
    span = radius * 1.5
    ax.set_xlim(eq_pt[0] - span, eq_pt[0] + span)
    ax.set_ylim(eq_pt[1] - span, eq_pt[1] + span)
    ax.set_zlim(eq_pt[2] - span, eq_pt[2] + span)

    ax.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    # Save figure
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Clean filename notation
    fig_name = f"fig05c_hiddenness_zoom_{eq_name}_r{radius:.0e}"
    fig_path_png = os.path.join(fig_dir, f"{fig_name}.png")
    fig_path_pdf = os.path.join(fig_dir, f"{fig_name}.pdf")
    plt.tight_layout()
    fig.savefig(fig_path_png, dpi=300)
    fig.savefig(fig_path_pdf)
    plt.close(fig)
    return fig_path_png
