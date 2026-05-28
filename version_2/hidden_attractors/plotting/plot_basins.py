import os
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from typing import Dict, Tuple, List

def plot_basin_slices(
    basin_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: dict,
    output_dir: str
) -> None:
    """Legacy wrapper for backward compatibility."""
    for plane, (u, v, mat) in basin_data.items():
        plot_basin_slice_file(
            plane=plane,
            u=u,
            v=v,
            mat=mat,
            eq_name="global",
            config=config,
            output_dir=output_dir
        )

def plot_basin_slice_file(
    plane: str,
    u: np.ndarray,
    v: np.ndarray,
    mat: np.ndarray,
    eq_name: str,
    config: dict,
    output_dir: str
) -> str:
    """
    Renders and saves a high-quality 2D basin slice plot.
    Saves as figures/basin_{plane}_{eq_name}.png.
    """
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    colors = ['#3b82f6', '#10b981', '#a855f7', '#1e293b', '#cbd5e1']
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    mesh = ax.pcolormesh(u, v, mat.T, cmap=cmap, vmin=0, vmax=4, shading='auto', alpha=0.92)
    
    labels = [
        "Stable Equilibrium",
        "Target Attractor",
        "Other Attractor",
        "Divergence",
        "Numerical Failure"
    ]
    
    patches = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in range(5)]
    ax.legend(patches, labels, loc='upper right', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    center_label = f"Local Centered ({eq_name})" if eq_name != "global" else "Global Scan"
    ax.set_title(f"Basin of Attraction: {plane.upper()} | {center_label}\n{config['system_id']}", fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel(plane[0], fontsize=10)
    ax.set_ylabel(plane[1], fontsize=10)
    
    fig_name = f"basin_{plane}_{eq_name}.png"
    fig_path = os.path.join(fig_dir, fig_name)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    return fig_path
