"""Centralized bifurcation diagram plotting module.

Stability: internal
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Sequence

from .style import apply_library_style, apply_axes_style, get_figsize
from .export import intercept_and_export_path
from ..analysis.bifurcation import BifurcationPoint


def plot_bifurcation_diagram_styled(
    points: Sequence[BifurcationPoint] | Sequence[dict] | np.ndarray,
    output_path: str | Path,
    *,
    parameter_label: str = "parameter",
    observable_label: str = "observable",
    title: str = "Bifurcation diagram",
    system_id: str = "chua_fractional",
    q: float = 1.0,
    integrator: str = "unknown",
    memory_mode: str = "unknown",
    t_final: float = 0.0,
    t_burn: float = 0.0,
) -> str:
    """Plot styled bifurcation diagram using centralized library APIs."""
    apply_library_style()
    figsize = get_figsize("2d")
    fig, ax = plt.subplots(figsize=figsize)
    
    if len(points) > 0:
        if isinstance(points[0], BifurcationPoint):
            params = np.array([p.parameter for p in points], dtype=float)
            values = np.array([p.observable for p in points], dtype=float)
        elif isinstance(points[0], dict):
            params = np.array([p["parameter"] for p in points], dtype=float)
            values = np.array([p["observable"] for p in points], dtype=float)
        else:
            # Assume numpy array of shape (N, 2) where col 0 is param and col 1 is observable
            arr = np.asarray(points, dtype=float)
            params = arr[:, 0]
            values = arr[:, 1]
            
        ax.scatter(params, values, s=1.0, color="#111827", alpha=0.6, edgecolors="none")
        
    ax.set_xlabel(parameter_label)
    ax.set_ylabel(observable_label)
    apply_axes_style(ax, grid=True)
    
    # Enforce no titles to comply with rule "Sin titulos internos"
    ax.set_title("")
    fig.suptitle("")
    
    metadata = {
        "system_id": system_id,
        "q": str(q),
        "integrator": integrator,
        "memory_mode": memory_mode,
        "t_final": t_final,
        "t_burn": t_burn,
        "caption_key": "fig_bifurcation",
        "source_script": "cli/bifurcation.py",
        "source_function": "plot_bifurcation_diagram_styled",
        "data_sources": ["bifurcation_data.csv"],
    }
    
    intercept_and_export_path(fig, output_path, "bifurcation", metadata_dict=metadata)
    plt.close(fig)
    return str(output_path)
