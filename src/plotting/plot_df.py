import os
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, List, Tuple
from ..lure.transfer import W_eval

def plot_describing_function(
    system: Any,
    candidates: List[Tuple[float, float, float]],
    config: dict,
    output_dir: str
) -> None:
    """Generate and save describing function plot N(A) vs A."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. PLOT DESCRIBING FUNCTION N(A) vs A
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
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
    
    for idx, (A0, w0, k) in enumerate(candidates):
        ax.scatter([A0], [k], color='#ef4444', s=60, zorder=5,
                   label=f'seed candidate {idx+1}: $A_0$={A0:.3f}, $k$={k:.3f}' if idx == 0 else f'candidate {idx+1}')
                   
    ax.set_title(f"Describing Function - {config['system_id']}", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel(r'Oscillation Amplitude ($A$)', fontsize=10)
    ax.set_ylabel(r'Equivalent Gain ($N(A)$)', fontsize=10)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "describing_function.png"), dpi=300)
    plt.close(fig)
    
    # 2. PLOT 2D HARMONIC RESIDUAL MAP |1 + N(A) W(iw)| = 0
    plot_harmonic_residual_map(system, candidates, config, output_dir)

def plot_harmonic_residual_map(
    system: Any,
    candidates: List[Tuple[float, float, float]],
    config: dict,
    output_dir: str
) -> None:
    """Generate and save a 2D contour map of the harmonic residual |1 + N(A)W(iw)|."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Build 2D evaluation grid (A, omega)
    n_a = 150
    n_w = 150
    
    # Linear ranges in the configured search boundaries
    as_grid = np.linspace(config["amplitude_min"], config["amplitude_max"], n_a)
    ws_grid = np.linspace(config["omega_min"], config["omega_max"], n_w)
    
    A_mesh, W_mesh = np.meshgrid(as_grid, ws_grid)
    residual_mesh = np.zeros((n_w, n_a))
    
    # Pre-evaluate transfer function W_q(i omega) to avoid redundant evaluations
    w_evals = []
    for w in ws_grid:
        try:
            val = W_eval(w, system.q, config["transfer_mode"], system.P, system.b, system.r)
            w_evals.append(val)
        except Exception:
            w_evals.append(complex(np.nan, np.nan))
            
    # Pre-evaluate describing function N(A)
    n_evals = []
    for a in as_grid:
        try:
            n_evals.append(system.describing_function(a))
        except Exception:
            n_evals.append(np.nan)
            
    # Calculate residual magnitude |1 + N(A) W_q(i w)|
    for i in range(n_w):
        for j in range(n_a):
            w_val = w_evals[i]
            n_val = n_evals[j]
            if np.isnan(w_val) or np.isnan(n_val):
                residual_mesh[i, j] = np.nan
            else:
                residual_mesh[i, j] = np.abs(1.0 + n_val * w_val)
                
    fig, ax = plt.subplots(figsize=(8.5, 7.2), dpi=300)
    
    # Plot logarithm of the residual to emphasize valleys/roots
    log_residual = np.log10(np.clip(residual_mesh, 1e-8, 10.0))
    contour = ax.contourf(A_mesh, W_mesh, log_residual, levels=25, cmap='viridis_r', alpha=0.92)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(r'$\log_{10}(|1 + N(A)W(i\omega)|)$ residual valley', fontsize=9)
    
    # Mark crossing points (roots) as red star markers
    for idx, (A0, w0, k) in enumerate(candidates):
        ax.scatter([A0], [w0], color='#ef4444', marker='*', s=120, edgecolors='black', linewidths=0.8, zorder=10,
                   label=f'crossing {idx+1}: $A_0$={A0:.2f}, $\\omega_0$={w0:.2f}' if idx == 0 else f'crossing {idx+1}')
                   
    ax.set_title(f"Harmonic Residual Map - {config['system_id']}", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel(r'Oscillation Amplitude ($A$)', fontsize=10)
    ax.set_ylabel(r'Frequency ($\omega$ rad/s)', fontsize=10)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "harmonic_residual_map.png"), dpi=300)
    plt.close(fig)
