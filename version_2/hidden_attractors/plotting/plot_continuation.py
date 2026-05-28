import os
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Any, List, Dict

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
    
    amplitudes = []
    for step in cont_steps:
        traj = step.get("trajectory")
        if traj is not None and len(traj) > 0:
            states = traj[:, 1:] if traj.shape[1] in (4, 5) else traj
            x_col = states[:, 0]
            half_idx = len(x_col) // 2
            amp = np.max(x_col[half_idx:]) - np.min(x_col[half_idx:])
            amplitudes.append(amp)
        else:
            amplitudes.append(0.0)
            
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
    
    if len(cont_steps) >= 2:
        try:
            plot_continuation_first_last_comparison(cont_steps, config, output_dir)
            plot_continuation_timeseries_comparison(cont_steps, config, output_dir)
            plot_continuation_progression(cont_steps, config, output_dir)
        except Exception as e:
            print(f"WARNING: Failed to generate premium continuation plots: {e}")

def plot_continuation_first_last_comparison(
    cont_steps: List[dict],
    config: dict,
    output_dir: str
) -> None:
    """Compare the linearized attractor (first step, lambda=0.0) and the final nonlinear attractor (last step, lambda=1.0) in 3D and 2D overlays."""
    first_step = cont_steps[0]
    last_step = cont_steps[-1]
    
    first_traj = first_step.get("trajectory")
    last_traj = last_step.get("trajectory")
    
    if first_traj is None or last_traj is None or len(first_traj) == 0 or len(last_traj) == 0:
        return
        
    if first_traj.shape[1] == 4:
        fx, fy, fz = first_traj[:, 1], first_traj[:, 2], first_traj[:, 3]
    else:
        fx, fy, fz = first_traj[:, 0], first_traj[:, 1], first_traj[:, 2]
        
    if last_traj.shape[1] == 4:
        lx, ly, lz = last_traj[:, 1], last_traj[:, 2], last_traj[:, 3]
    else:
        lx, ly, lz = last_traj[:, 0], last_traj[:, 1], last_traj[:, 2]
        
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    fig_3d = plt.figure(figsize=(8, 7), dpi=300)
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    ax_3d.plot(fx, fy, fz, color='#3b82f6', linestyle='--', linewidth=1.2, alpha=0.9, label=r'Linearized Attractor ($\eta=0.0$)')
    ax_3d.plot(lx, ly, lz, color='#ef4444', linestyle='-', linewidth=1.2, alpha=0.9, label=r'Nonlinear Attractor ($\eta=1.0$)')
    
    ax_3d.set_title(f"Linearized vs Nonlinear Attractor\n{config['system_id']}", fontsize=12, fontweight='bold', pad=15)
    ax_3d.set_xlabel('x', fontsize=10)
    ax_3d.set_ylabel('y', fontsize=10)
    ax_3d.set_zlabel('z', fontsize=10)
    ax_3d.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    fig_3d.savefig(os.path.join(fig_dir, "continuation_first_last_comparison.png"), dpi=300)
    plt.close(fig_3d)
    
    fig_2d, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    
    projections = [
        (0, fx, fy, lx, ly, 'x', 'y', 'XY Projection'),
        (1, fx, fz, lx, lz, 'x', 'z', 'XZ Projection'),
        (2, fy, fz, ly, lz, 'y', 'z', 'YZ Projection')
    ]
    
    for idx, fu, fv, lu, lv, xlabel, ylabel, title in projections:
        ax = axes[idx]
        ax.grid(True, linestyle='--', linewidth=0.5, color='#cbd5e1')
        ax.plot(fu, fv, color='#3b82f6', linestyle='--', linewidth=1.2, alpha=0.9, label=r'Linearized ($\eta=0.0$)')
        ax.plot(lu, lv, color='#ef4444', linestyle='-', linewidth=1.2, alpha=0.9, label=r'Nonlinear ($\eta=1.0$)')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
        
    plt.tight_layout()
    fig_2d.savefig(os.path.join(fig_dir, "continuation_first_last_projections.png"), dpi=300)
    plt.close(fig_2d)

def plot_continuation_timeseries_comparison(
    cont_steps: List[dict],
    config: dict,
    output_dir: str
) -> None:
    """Compare the time series of the first state variable x(t) between the first cycle (linearized) and last cycle."""
    first_step = cont_steps[0]
    last_step = cont_steps[-1]
    
    first_traj = first_step.get("trajectory")
    last_traj = last_step.get("trajectory")
    
    if first_traj is None or last_traj is None or len(first_traj) == 0 or len(last_traj) == 0:
        return
        
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    t1 = first_traj[:, 0]
    x1 = first_traj[:, 1]
    t1_align = t1 - t1[0]
    
    t2 = last_traj[:, 0]
    x2 = last_traj[:, 1]
    t2_align = t2 - t2[0]
    
    fig = plt.figure(figsize=(10, 5), dpi=300)
    ax = fig.add_subplot(111)
    ax.grid(True, linestyle='--', linewidth=0.5, color='#cbd5e1')
    
    ax.plot(t1_align, x1, color='#3b82f6', linestyle='--', linewidth=1.2, alpha=0.9, label=r'Linearized $x(t)$ ($\eta=0.0$)')
    ax.plot(t2_align, x2, color='#ef4444', linestyle='-', linewidth=1.2, alpha=0.9, label=r'Nonlinear $x(t)$ ($\eta=1.0$)')
    
    ax.set_title(f"Time Series Comparison: $x(t)$\n{config['system_id']}", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Aligned Time $t$', fontsize=10)
    ax.set_ylabel('State Variable $x$', fontsize=10)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "continuation_timeseries_comparison_x.png"), dpi=300)
    plt.close(fig)

def plot_continuation_progression(
    cont_steps: List[dict],
    config: dict,
    output_dir: str
) -> None:
    """Plot progression of trajectories at each step of continuation, and trace the path followed by their initial conditions."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(8, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    lambdas = [step["lambda_value"] for step in cont_steps]
    lam_min = min(lambdas) if len(lambdas) > 0 else 0.0
    lam_max = max(lambdas) if len(lambdas) > 0 else 1.0
    lam_span = max(1e-12, lam_max - lam_min)
    
    cmap = plt.get_cmap("plasma")
    
    for idx, step in enumerate(cont_steps):
        traj = step.get("trajectory")
        if traj is None or len(traj) == 0:
            continue
            
        eta = step["lambda_value"]
        color = cmap((eta - lam_min) / lam_span)
        
        if traj.shape[1] == 4:
            x, y, z = traj[:, 1], traj[:, 2], traj[:, 3]
        else:
            x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
            
        max_pts = 1500
        if len(x) > max_pts:
            indices = np.linspace(0, len(x) - 1, max_pts, dtype=int)
            x_plot = x[indices]
            y_plot = y[indices]
            z_plot = z[indices]
        else:
            x_plot, y_plot, z_plot = x, y, z
            
        ax.plot(x_plot, y_plot, z_plot, color=color, linewidth=0.8, alpha=0.85, 
                label=fr"$\eta = {eta:.2f}$" if len(cont_steps) <= 8 else "")
                
    x_in_pts = []
    for step in cont_steps:
        x_in = step.get("x_in")
        if x_in is not None:
            x_in_pts.append(x_in)
            
    if len(cont_steps) > 0:
        x_out_final = cont_steps[-1].get("x_out")
        if x_out_final is not None:
            x_in_pts.append(x_out_final)
            
    if len(x_in_pts) > 0:
        x_in_arr = np.array(x_in_pts)
        ax.plot(x_in_arr[:, 0], x_in_arr[:, 1], x_in_arr[:, 2], 
                color='#475569', linestyle='--', marker='o', markersize=5, linewidth=1.5,
                zorder=10, label="Initial Conditions Path")
                
    ax.set_title(f"Numerical Continuation: Trajectory Progression\n{config['system_id']}", fontsize=11, fontweight='bold', pad=15)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_zlabel('z', fontsize=10)
    
    if len(cont_steps) <= 8:
        ax.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    else:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if "Initial Conditions Path" in by_label:
            ax.legend([by_label["Initial Conditions Path"]], ["Initial Conditions Path"], 
                      loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
                      
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=lam_min, vmax=lam_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(r'Continuation Parameter $\eta$', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "continuation_progression.png"), dpi=300)
    plt.close(fig)

_STATUS_ORDER = [
    "ok",
    "diverged_early",
    "diverged",
    "converged_equilibrium_early",
    "nonfinite_solution",
    "backend_failure",
]

_STATUS_COLORS = {
    "ok":                          "#22c55e",
    "diverged_early":              "#f97316",
    "diverged":                    "#ef4444",
    "converged_equilibrium_early": "#a855f7",
    "nonfinite_solution":          "#64748b",
    "backend_failure":             "#0ea5e9",
}

def plot_continuation_tracking(
    cont_steps: List[dict],
    config: dict,
    output_dir: str,
) -> None:
    """Generate tracking plots for the continuation run."""
    if len(cont_steps) == 0:
        return

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    etas      = [s["lambda_value"] for s in cont_steps]
    out_norms = [s.get("x_out_norm", float(np.linalg.norm(s["x_out"]))) for s in cont_steps]
    statuses  = [s.get("status", "ok")                                   for s in cont_steps]

    sysid = config.get("system_id", "")
    color_pts = [_STATUS_COLORS.get(st, "#64748b") for st in statuses]

    from matplotlib.patches import Patch
    seen_legend: dict = {}
    for st, col in zip(statuses, color_pts):
        seen_legend[st] = col
    legend_handles = [
        Patch(facecolor=c, label=st, edgecolor="white")
        for st, c in seen_legend.items()
    ]

    eta_max = max(etas) * 1.04 if etas else 1.05

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.set_facecolor("#f8fafc")
    ax.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
    ax.plot(etas, out_norms, color="#7c3aed", linewidth=1.6, zorder=2, alpha=0.7)
    ax.scatter(etas, out_norms, c=color_pts, s=50, zorder=3, edgecolors="white", linewidths=0.5)
    ax.set_title(f"Continuation: $\\|x_{{\\mathrm{{out}}}}\\|$ vs $\\eta$\n{sysid}",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(r"$\eta$", fontsize=11)
    ax.set_ylabel(r"$\|x_{\mathrm{out}}\|$", fontsize=11)
    ax.set_xlim(-0.02, eta_max)
    ax.legend(handles=legend_handles, loc="best", fontsize=8,
              framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")

    all_statuses = list(dict.fromkeys(
        [s for s in _STATUS_ORDER if s in statuses] +
        [s for s in statuses if s not in _STATUS_ORDER]
    ))
    status_to_y = {st: i for i, st in enumerate(all_statuses)}

    fig_h = max(3.0, len(all_statuses) * 0.9 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h), dpi=300)
    ax.set_facecolor("#f8fafc")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, color="#cbd5e1")
    for st_val, eta_val in zip(statuses, etas):
        y = status_to_y[st_val]
        col = _STATUS_COLORS.get(st_val, "#64748b")
        ax.scatter(eta_val, y, color=col, s=80, zorder=3, edgecolors="white", linewidths=0.6)
    ax.set_yticks(range(len(all_statuses)))
    ax.set_yticklabels(all_statuses, fontsize=9)
    ax.set_title(f"Continuation: Status per Step\n{sysid}",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(r"$\eta$", fontsize=11)
    ax.set_xlim(-0.02, eta_max)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "continuation_tracking_status.png"), dpi=300)
    plt.close(fig)
