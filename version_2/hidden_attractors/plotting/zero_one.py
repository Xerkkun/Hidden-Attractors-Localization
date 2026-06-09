"""Centralized 0-1 chaos-test plotting module.

Stability: internal
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .style import apply_library_style, apply_axes_style, get_figsize
from .export import intercept_and_export_path


def plot_zero_one_phase_styled(
    signal: np.ndarray,
    c_value: float,
    output_path: str | Path,
    *,
    system_id: str = "chua_fractional",
) -> str:
    """Plot p_c vs q_c trajectory to illustrate regular vs chaotic dynamics."""
    apply_library_style()
    
    index = np.arange(1, signal.size + 1, dtype=float)
    p = np.cumsum(signal * np.cos(index * c_value))
    q = np.cumsum(signal * np.sin(index * c_value))
    
    figsize = get_figsize("2d")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(p, q, lw=0.6, color="#2563eb", alpha=0.8)
    ax.set_xlabel("p")
    ax.set_ylabel("q")
    apply_axes_style(ax, grid=False)
    
    ax.set_title("")
    fig.suptitle("")
    
    metadata = {
        "system_id": system_id,
        "caption_key": "fig_zero_one_phase",
        "source_script": "cli/chaos_test.py",
        "source_function": "plot_zero_one_phase_styled",
        "data_sources": ["zero_one_displacement.csv"],
    }
    
    intercept_and_export_path(fig, output_path, "zero_one", metadata_dict=metadata)
    plt.close(fig)
    return str(output_path)
