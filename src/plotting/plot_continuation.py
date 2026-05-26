import os
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, List

def plot_continuation_eta(
    cont_steps: List[dict],
    config: dict,
    output_dir: str
) -> None:
    """Generate and save numerical continuation plots: eta vs state norm, and eta vs oscillation amplitude."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    etas = [step["lambda_value"] for step in cont_steps]
    norms = [np.linalg.norm(step["x_out"]) for step in cont_steps]
    statuses = [step["status"] for step in cont_steps]
    
    # Calculate peak-to-peak amplitude of the x-component for each step
    amplitudes = []
    for step in cont_steps:
        traj = step.get("trajectory")
        if traj is not None and len(traj) > 0:
            # traj is np.ndarray of shape (N, dim) or (N, dim+1) if time included
            # Let's extract state columns
            states = traj[:, 1:] if traj.shape[1] in (4, 5) else traj
            x_col = states[:, 0]
            # Standard peak-to-peak amplitude after transient burn-in (last half of points)
            half_idx = len(x_col) // 2
            amp = np.max(x_col[half_idx:]) - np.min(x_col[half_idx:])
            amplitudes.append(amp)
        else:
            amplitudes.append(0.0)
            
    # Separate successful vs failed steps
    success_etas = []
    success_norms = []
    success_amps = []
    failed_etas = []
    failed_norms = []
    failed_amps = []
    
    for i in range(len(cont_steps)):
        if statuses[i] == "ok":
            success_etas.append(etas[i])
            success_norms.append(norms[i])
            success_amps.append(amplitudes[i])
        else:
            failed_etas.append(etas[i])
            failed_norms.append(norms[i])
            failed_amps.append(amplitudes[i])
            
    # 1. PLOT CONTINUATION ETA vs NORM
    fig_norm, ax_norm = plt.subplots(figsize=(8, 7), dpi=300)
    ax_norm.grid(True, linestyle='--', linewidth=0.5, color='#cbd5e1')
    
    if len(success_etas) > 0:
        ax_norm.plot(success_etas, success_norms, color='#7c3aed', marker='o', markersize=5, linewidth=1.8, label='Success')
    if len(failed_etas) > 0:
        ax_norm.scatter(failed_etas, failed_norms, color='#ef4444', marker='x', s=60, zorder=5, label='Failure')
        
    ax_norm.set_title(f"Numerical Continuation: Norm vs $\\eta$\n{config['system_id']}", fontsize=12, fontweight='bold', pad=15)
    ax_norm.set_xlabel(r'Continuation Parameter ($\eta$)', fontsize=10)
    ax_norm.set_ylabel(r'Final State Norm $\|\|X_{\mathrm{out}}\|\|$', fontsize=10)
    ax_norm.set_xlim(-0.05, 1.05)
    ax_norm.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    fig_norm.savefig(os.path.join(fig_dir, "continuation_eta_norm.png"), dpi=300)
    plt.close(fig_norm)
    
    # 2. PLOT CONTINUATION ETA vs AMPLITUDE
    fig_amp, ax_amp = plt.subplots(figsize=(8, 7), dpi=300)
    ax_amp.grid(True, linestyle='--', linewidth=0.5, color='#cbd5e1')
    
    if len(success_etas) > 0:
        ax_amp.plot(success_etas, success_amps, color='#2563eb', marker='s', markersize=5, linewidth=1.8, label='Success')
    if len(failed_etas) > 0:
        ax_amp.scatter(failed_etas, failed_amps, color='#ef4444', marker='x', s=60, zorder=5, label='Failure')
        
    ax_amp.set_title(f"Numerical Continuation: Amplitude vs $\\eta$\n{config['system_id']}", fontsize=12, fontweight='bold', pad=15)
    ax_amp.set_xlabel(r'Continuation Parameter ($\eta$)', fontsize=10)
    ax_amp.set_ylabel(r'Peak-to-Peak Amplitude $A_{pp}(x)$', fontsize=10)
    ax_amp.set_xlim(-0.05, 1.05)
    ax_amp.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    fig_amp.savefig(os.path.join(fig_dir, "continuation_eta_amplitude.png"), dpi=300)
    plt.close(fig_amp)
