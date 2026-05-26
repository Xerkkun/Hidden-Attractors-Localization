import os
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from typing import Any, Dict, Tuple

def plot_basin_slices(
    basin_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: dict,
    output_dir: str
) -> None:
    """Generate and save 2D basin of attraction plots withListedColormap."""
    # Custom vibrant premium listed colormap
    # 0: Stable Equilibrium (soft blue)
    # 1: Target Attractor / Candidate (soft green)
    # 2: Other Attractor (soft purple)
    # 3: Divergence (soft yellow)
    # 4: Numerical Failure (soft gray)
    colors = ['#93c5fd', '#86efac', '#c084fc', '#fde047', '#cbd5e1']
    cmap = ListedColormap(colors)
    
    for plane, (u, v, mat) in basin_data.items():
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        
        # Pcolormesh expects X and Y as 2D grids or 1D coords
        # mat is shape (grid_n, grid_n)
        # We plot mat.T to align axes properly
        mesh = ax.pcolormesh(u, v, mat.T, cmap=cmap, vmin=0, vmax=4, shading='auto', alpha=0.9)
        
        # Add a premium legend manually
        labels = [
            "Stable Equilibrium",
            "Target Attractor (Hit)",
            "Other Attractor",
            "Divergence",
            "Numerical Failure"
        ]
        
        patches = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in range(5)]
        ax.legend(patches, labels, loc='upper right', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
        
        ax.set_title(f"Basin of Attraction Slice ({plane.upper()})", fontsize=11, fontweight='bold', pad=12)
        ax.set_xlabel(plane[0], fontsize=9)
        ax.set_ylabel(plane[1], fontsize=9)
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"basin_{plane}.png"), dpi=300)
        plt.close(fig)
