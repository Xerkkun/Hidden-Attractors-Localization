"""Centralized Lyapunov exponent plotting module.

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
from ..analysis.lyapunov import LyapunovResult


def plot_lyapunov_convergence_styled(
    result: LyapunovResult,
    output_path: str | Path,
    *,
    system_id: str = "chua_fractional",
) -> str:
    """Plot styled Lyapunov convergence curves and export using centralized API."""
    apply_library_style()
    figsize = get_figsize("2d")
    fig, ax = plt.subplots(figsize=figsize)
    
    if result.convergence.size > 0 and result.times.size > 0:
        for idx in range(result.convergence.shape[1]):
            ax.plot(result.times, result.convergence[:, idx], lw=0.95, label=f"LE{idx}")
    else:
        # Plot point markers if no convergence timeseries
        ax.scatter(np.arange(result.exponents.size), result.exponents, color="#111827", s=28)
        
    ax.axhline(0.0, color="#6b7280", ls="--", lw=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("exponent")
    apply_axes_style(ax, grid=True)
    
    # Enforce no titles to comply with rule "Sin titulos internos"
    ax.set_title("")
    fig.suptitle("")
    
    if result.convergence.size > 0:
        ax.legend(loc="best", frameon=True)
        
    metadata = {
        "system_id": system_id,
        "q": str(result.q),
        "integrator": result.method_id,
        "memory_mode": result.derivative_model,
        "t_final": float(result.times[-1]) if result.times.size > 0 else 0.0,
        "t_burn": 0.0,
        "caption_key": "fig_lyapunov_convergence",
        "source_script": "cli/lyapunov.py",
        "source_function": "plot_lyapunov_convergence_styled",
        "data_sources": ["lyapunov_convergence.csv"],
    }
    
    intercept_and_export_path(fig, output_path, "lyapunov", metadata_dict=metadata)
    plt.close(fig)
    return str(output_path)
