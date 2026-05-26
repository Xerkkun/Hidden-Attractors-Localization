import os
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, List, Tuple

def plot_nyquist_transfer(
    omega_grid: np.ndarray,
    w_vals: np.ndarray,
    candidates: List[Tuple[float, float, float]],
    config: dict,
    output_dir: str
) -> None:
    """Generate and save Nyquist plot of the transfer function."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Enable a clean grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#cbd5e1')
    ax.axhline(0, color='#64748b', linewidth=1.0)
    ax.axvline(0, color='#64748b', linewidth=1.0)
    
    # Plot W(i omega)
    real_parts = [val.real for val in w_vals if not np.isnan(val.real)]
    imag_parts = [val.imag for val in w_vals if not np.isnan(val.imag)]
    
    if len(real_parts) > 0:
        ax.plot(real_parts, imag_parts, color='#0284c7', linewidth=1.8, label=r'$W(i\omega)$ trajectory')
        
    # Mark crossings
    for idx, (A, w0, k) in enumerate(candidates):
        # W(iw0) = -1/k
        target_pt = -1.0 / k
        ax.scatter([target_pt], [0.0], color='#ef4444', s=60, zorder=5,
                   label=f'crossing {idx+1}: $\\omega_0$={w0:.3f}, $k$={k:.3f}' if idx == 0 else f'crossing {idx+1}')
                   
    ax.set_title(f"Nyquist Plot - {config['system_id']}", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel(r'$\mathrm{Re}(W)$', fontsize=10)
    ax.set_ylabel(r'$\mathrm{Im}(W)$', fontsize=10)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "nyquist_plot.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
