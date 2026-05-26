import os
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, List, Tuple

def plot_describing_function(
    system: Any,
    candidates: List[Tuple[float, float, float]],
    config: dict,
    output_dir: str
) -> None:
    """Generate and save describing function plot N(A) vs A."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.grid(True, linestyle='--', linewidth=0.5, color='#cbd5e1')
    
    # Amplitude sweep
    as_ = np.linspace(config["amplitude_min"], config["amplitude_max"], 500)
    n_vals = []
    for a in as_:
        try:
            n_vals.append(system.describing_function(a))
        except Exception:
            n_vals.append(np.nan)
            
    ax.plot(as_, n_vals, color='#0d9488', linewidth=1.8, label=r'$N(A)$ describing function')
    
    # Mark resolved seeds
    for idx, (A0, w0, k) in enumerate(candidates):
        ax.scatter([A0], [k], color='#ef4444', s=60, zorder=5,
                   label=f'seed candidate {idx+1}: $A_0$={A0:.3f}, $k$={k:.3f}' if idx == 0 else f'candidate {idx+1}')
                   
    ax.set_title(f"Describing Function - {config['system_id']}", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel(r'Oscillation Amplitude ($A$)', fontsize=10)
    ax.set_ylabel(r'Equivalent Gain ($N(A)$)', fontsize=10)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "describing_function_plot.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
