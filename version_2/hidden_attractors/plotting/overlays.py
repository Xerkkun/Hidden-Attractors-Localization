"""Overlay plots for comparing attractor trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.trajectory import sample_rows
from ..io import safe_name


def plot_trajectory_overlay(
    trajectories: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    title: str,
    output_path: str | Path,
    max_points: int = 1500,
) -> str:
    """Plot superposed trajectories in 3D and coordinate projections."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(trajectories), 1)))
    fig = plt.figure(figsize=(12.5, 10.0))
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    axxy = fig.add_subplot(2, 2, 2)
    axxz = fig.add_subplot(2, 2, 3)
    axyz = fig.add_subplot(2, 2, 4)
    for idx, X in enumerate(trajectories):
        data = sample_rows(np.asarray(X, dtype=float), max_points)
        label = labels[idx] if idx < len(labels) else f"case_{idx}"
        c = colors[idx]
        ax3.plot(data[:, 1], data[:, 2], data[:, 3], lw=0.65, alpha=0.82, color=c, label=label)
        axxy.plot(data[:, 1], data[:, 2], lw=0.65, alpha=0.82, color=c)
        axxz.plot(data[:, 1], data[:, 3], lw=0.65, alpha=0.82, color=c)
        axyz.plot(data[:, 2], data[:, 3], lw=0.65, alpha=0.82, color=c)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    axxy.set_xlabel("x")
    axxy.set_ylabel("y")
    axxz.set_xlabel("x")
    axxz.set_ylabel("z")
    axyz.set_xlabel("y")
    axyz.set_ylabel("z")
    ax3.set_title("3D overlay")
    axxy.set_title("xy")
    axxz.set_title("xz")
    axyz.set_title("yz")
    handles, legend_labels = ax3.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, fontsize=8, frameon=True)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)
