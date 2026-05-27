import os
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
from ..verification.stability import classify_equilibrium_stability

def plot_matignon_equilibria(
    system: Any,
    equilibria: Dict[str, np.ndarray],
    config: dict,
    output_dir: str
) -> str:
    """
    Renders the premium Matignon stability plane visualization for all equilibria.
    Saves the plot as 'figures/matignon_equilibria.png'.
    """
    q = system.q
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(8, 7), dpi=300)
    ax = fig.add_subplot(111)
    
    # Pre-calculate eigenvalues to determine limits
    all_eigvals = []
    eq_details = []
    
    for name, eq_pt in equilibria.items():
        res = classify_equilibrium_stability(system, eq_pt)
        all_eigvals.extend(res["eigenvalues"])
        eq_details.append((name, res))
        
    all_eigvals = np.array(all_eigvals)
    max_radius = float(np.max(np.abs(all_eigvals))) if len(all_eigvals) > 0 else 1.0
    if max_radius < 1e-12:
        max_radius = 1.0
        
    limit = max_radius * 1.5
    
    # 1. STYLE BACKGROUND (STABLE BY DEFAULT)
    ax.set_facecolor('#f0fdf4')  # light green for stable background
    
    # 2. FILL UNSTABLE REGION
    # Unstable region is |arg(lambda)| <= q * pi / 2
    t_vals = np.linspace(-q * np.pi / 2.0, q * np.pi / 2.0, 300)
    R = limit * 2.0
    x_fill = [0.0] + list(R * np.cos(t_vals)) + [0.0]
    y_fill = [0.0] + list(R * np.sin(t_vals)) + [0.0]
    ax.fill(x_fill, y_fill, color='#fee2e2', alpha=0.85, label='Unstable Region', edgecolor='#fca5a5', linewidth=1.0)
    
    # 3. DRAW FRONTIER RAYS
    ax.plot([0.0, R * np.cos(q * np.pi / 2.0)], [0.0, R * np.sin(q * np.pi / 2.0)], color='#ef4444', linestyle='--', linewidth=1.2, label=r'Frontera $|\arg(\lambda)| = q\pi/2$')
    ax.plot([0.0, R * np.cos(-q * np.pi / 2.0)], [0.0, R * np.sin(-q * np.pi / 2.0)], color='#ef4444', linestyle='--', linewidth=1.2)
    
    # 4. PLOT REAL AND IMAGINARY AXES
    ax.axhline(0.0, color='#64748b', linewidth=0.8, linestyle=':')
    ax.axvline(0.0, color='#64748b', linewidth=0.8, linestyle=':')
    
    # 5. PLOT EQUILIBRIA EIGENVALUES
    colors = {'E0': '#3b82f6', 'E+': '#ef4444', 'E-': '#f59e0b'}
    markers = {'E0': '^', 'E+': 'o', 'E-': 's'}
    
    for name, res in eq_details:
        color = colors.get(name, '#8b5cf6')
        marker = markers.get(name, 'd')
        eigvals = res["eigenvalues"]
        ax.scatter(np.real(eigvals), np.imag(eigvals), color=color, marker=marker, s=80, edgecolors='black', zorder=10, label=f"{name} eigenvalues")
        
    # Axis styling
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.set_title(f"Matignon Stability Plane (q={q:.4f}) - {config['system_id']}", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel(r'Real ($\mathrm{Re}(\lambda)$)', fontsize=10)
    ax.set_ylabel(r'Imaginary ($\mathrm{Im}(\lambda)$)', fontsize=10)
    ax.grid(True, linestyle=':', linewidth=0.5, color='#cbd5e1')
    
    # 6. BUILD BRIEF ANNOTATION TEXTBOX
    info_lines = ["--- Equilibria Stability Summary ---"]
    for name, res in eq_details:
        alpha_min = res["alpha_min"]
        inst_meas = res["instability_measure"]
        is_stable = res["stable"]
        
        if q == 1.0:
            info_lines.append(f"{name}: stable={is_stable}")
        else:
            info_lines.append(f"{name}: stable={is_stable} | min|arg|={alpha_min:.3f} | inst_meas={inst_meas:.3f}")
            
    textstr = "\n".join(info_lines)
    ax.text(0.03, 0.03, textstr, transform=ax.transAxes, fontsize=8, family='monospace', verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#e2e8f0'))
    
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "matignon_equilibria.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    return fig_path
