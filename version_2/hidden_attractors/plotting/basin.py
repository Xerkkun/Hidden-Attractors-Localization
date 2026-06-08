"""High-quality 2D basin slice plotting for the hidden attractors package."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, Union

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def _output_path(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def plot_basin_slices(
    basin_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    system_id: str,
    output_dir: str | Path,
) -> Dict[str, str]:
    """Plot multiple basin slices for all planes in basin_data.

    Parameters
    ----------
    basin_data : Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Maps plane name (e.g. 'xy') to (u, v, mat) mesh grids.
    system_id : str
        Name of the dynamical system.
    output_dir : str or Path
        Directory where figures will be saved.

    Returns
    -------
    paths : Dict[str, str]
        Maps plane names to the saved figure paths.
    """
    paths = {}
    for plane, (u, v, mat) in basin_data.items():
        fig_path = plot_basin_slice_file(
            plane=plane,
            u=u,
            v=v,
            mat=mat,
            eq_name="global",
            system_id=system_id,
            output_dir=output_dir,
        )
        paths[plane] = fig_path
    return paths


def plot_basin_slice_file(
    plane: str,
    u: np.ndarray,
    v: np.ndarray,
    mat: np.ndarray,
    eq_name: str,
    system_id: str,
    output_dir: str | Path,
) -> str:
    """Renders and saves a high-quality 2D basin slice plot.

    Saves as output_dir/figures/basin_{plane}_{eq_name}.png.
    """
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 0: Stable Equilibrium (soft blue)
    # 1: Target Attractor (soft green)
    # 2: Other Attractor / Unclassified (soft purple)
    # 3: Divergence (soft yellow/black depending on style, let's use dark gray/black or soft red)
    # 4: Numerical Failure (soft gray)
    colors = ['#3b82f6', '#10b981', '#a855f7', '#1e293b', '#cbd5e1']
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    # We plot mat.T to align axes properly
    mesh = ax.pcolormesh(u, v, mat.T, cmap=cmap, vmin=0, vmax=4, shading='auto', alpha=0.92)

    # Custom premium legend
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
    ax.set_title(f"Basin of Attraction: {plane.upper()} | {center_label}\n{system_id}", fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel(plane[0], fontsize=10)
    ax.set_ylabel(plane[1], fontsize=10)

    fig_name = f"basin_{plane}_{eq_name}.png"
    fig_path = fig_dir / fig_name
    plt.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, fig_path, "basin")
    plt.close(fig)
    return str(fig_path)
