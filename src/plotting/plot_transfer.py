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
    """Generate and save Nyquist plot and real/imag component plots of the transfer function."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. RENDER NYQUIST PLOT
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#cbd5e1')
    ax.axhline(0, color='#64748b', linewidth=1.0)
    ax.axvline(0, color='#64748b', linewidth=1.0)
    
    real_parts = [val.real for val in w_vals if not np.isnan(val.real)]
    imag_parts = [val.imag for val in w_vals if not np.isnan(val.imag)]
    
    if len(real_parts) > 0:
        ax.plot(real_parts, imag_parts, color='#0284c7', linewidth=1.8, label=r'$W(i\omega)$ trajectory')
        
    for idx, (A, w0, k) in enumerate(candidates):
        target_pt = -1.0 / k
        ax.scatter([target_pt], [0.0], color='#ef4444', s=60, zorder=5,
                   label=f'crossing {idx+1}: $\\omega_0$={w0:.3f}, $k$={k:.3f}' if idx == 0 else f'crossing {idx+1}')
                   
    ax.set_title(f"Nyquist Plot - {config['system_id']}", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel(r'$\mathrm{Re}(W)$', fontsize=10)
    ax.set_ylabel(r'$\mathrm{Im}(W)$', fontsize=10)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "transfer_nyquist.png"), dpi=300)
    plt.close(fig)
    
    # 2. RENDER REAL & IMAG COMPONENTS PLOT
    fig_comp, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, dpi=300)
    
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='#cbd5e1')
    axes[0].plot(omega_grid, np.real(w_vals), color='#2563eb', linewidth=1.5, label=r'$\mathrm{Re}(W_q(i\omega))$')
    if len(candidates) > 0:
        # Mark chosen k crossings on real plot
        chosen_k = candidates[0][2]
        axes[0].axhline(-1.0 / chosen_k, color='#ef4444', linestyle='--', linewidth=1.1, label=r'$-1/k$ crossing')
        axes[0].scatter([candidates[0][1]], [-1.0 / chosen_k], color='#ef4444', s=45, zorder=5)
        
    axes[0].set_ylabel(r'$\mathrm{Re}(W)$', fontsize=10)
    axes[0].set_title("Real Component vs Frequency", fontsize=11, fontweight='bold')
    axes[0].legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='#cbd5e1')
    axes[1].plot(omega_grid, np.imag(w_vals), color='#0891b2', linewidth=1.5, label=r'$\mathrm{Im}(W_q(i\omega))$')
    axes[1].axhline(0.0, color='#64748b', linestyle='--', linewidth=1.0, label='Zero crossing')
    if len(candidates) > 0:
        axes[1].scatter([candidates[0][1]], [0.0], color='#ef4444', s=45, zorder=5)
        
    axes[1].set_xlabel(r'$\omega$ (rad/s)', fontsize=10)
    axes[1].set_ylabel(r'$\mathrm{Im}(W)$', fontsize=10)
    axes[1].set_title("Imaginary Component vs Frequency", fontsize=11, fontweight='bold')
    axes[1].legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    fig_comp.savefig(os.path.join(fig_dir, "transfer_real_imag.png"), dpi=300)
    plt.close(fig_comp)
