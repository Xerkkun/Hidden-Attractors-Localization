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
    """Generate and save numerical continuation plot eta vs state norm."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.grid(True, linestyle='--', linewidth=0.5, color='#cbd5e1')
    
    etas = [step["lambda_value"] for step in cont_steps]
    norms = [np.linalg.norm(step["x_out"]) for step in cont_steps]
    statuses = [step["status"] for step in cont_steps]
    
    # Separate successful vs failed steps
    success_etas = []
    success_norms = []
    failed_etas = []
    failed_norms = []
    
    for i in range(len(cont_steps)):
        if statuses[i] == "ok":
            success_etas.append(etas[i])
            success_norms.append(norms[i])
        else:
            failed_etas.append(etas[i])
            failed_norms.append(norms[i])
            
    if len(success_etas) > 0:
        ax.plot(success_etas, success_norms, color='#7c3aed', marker='o', markersize=5, linewidth=1.8, label='Successful continuation steps')
    if len(failed_etas) > 0:
        ax.scatter(failed_etas, failed_norms, color='#ef4444', marker='x', s=60, zorder=5, label='Failed continuation steps')
        
    ax.set_title(f"Numerical Continuation - {config['system_id']}", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel(r'Continuation Parameter ($\eta$)', fontsize=10)
    ax.set_ylabel(r'Final State Norm $\|\|X_{\mathrm{out}}\|\|$', fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "continuation_plot.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
