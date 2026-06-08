import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from .style import apply_library_style, apply_axes_style, get_figsize
from .export import export_figure

def render_attractor(
    trajectory: np.ndarray,
    equilibria: Dict[str, np.ndarray],
    config: dict,
    run_id: str = "default_run",
    report_targets: List[str] = None
) -> Dict[str, str]:
    """
    Renders 3D phase space and 2D projections of the attractor.
    Enforces no titles, white background, named axes, explicit transient burn.
    """
    apply_library_style()
    
    t = trajectory[:, 0]
    x = trajectory[:, 1]
    y = trajectory[:, 2]
    z = trajectory[:, 3]
    
    # Transient burn cut
    h = config.get("h", 0.005)
    t_burn = config.get("final_simulation", {}).get("t_burn", config.get("t_burn", 120.0))
    n_burn = int(np.ceil(t_burn / h))
    
    if len(trajectory) > n_burn:
        x_plot = x[n_burn:]
        y_plot = y[n_burn:]
        z_plot = z[n_burn:]
    else:
        x_plot = x
        y_plot = y
        z_plot = z
        
    lw = config.get("attractor_plots", {}).get("line_width", 0.7)
    
    # 1. 3D Plot
    fig_3d = plt.figure(figsize=get_figsize("attractor_3d"))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    apply_axes_style(ax_3d, is_3d=True)
    
    # Draw trajectory first so it's behind equilibria markers
    ax_3d.plot(x_plot, y_plot, z_plot, color='#10b981', linewidth=lw, alpha=0.85, label='Attractor')
    
    # Discrete, non-overlapping equilibria
    include_eq = config.get("attractor_plots", {}).get("include_equilibria", True)
    if include_eq and equilibria:
        for name, eq in equilibria.items():
            color = '#ef4444' if name != 'E0' else '#3b82f6'
            marker = '^' if name == 'E0' else 'o'
            # zorder high, discrete marker
            ax_3d.scatter([eq[0]], [eq[1]], [eq[2]], color=color, marker=marker, s=40, edgecolors='black', zorder=100, label=name)
            
    ax_3d.set_xlabel('$x$', fontsize=10)
    ax_3d.set_ylabel('$y$', fontsize=10)
    ax_3d.set_zlabel('$z$', fontsize=10)
    
    # Legend only when there are multiple classes/curves
    if include_eq and equilibria:
        ax_3d.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='white', edgecolor='#cbd5e1')
        
    # Metadata construction
    system_id = config.get("system_id", "chua_nonsmooth")
    q = config.get("q", config.get("system_params", {}).get("q", 1.0))
    meta_common = {
        "caption_key": f"fig_attractor_{system_id}_3d",
        "source_script": "renderers.py",
        "source_function": "render_attractor",
        "system_id": system_id,
        "q": str(q),
        "parameters": config.get("system_params", {}),
        "integrator": config.get("integrator", "abm"),
        "memory_mode": config.get("memory_mode", "fractional"),
        "t_final": config.get("t_final", t[-1]),
        "t_burn": t_burn,
        "data_sources": ["simulated_trajectory"]
    }
    
    pdf_3d, png_3d = export_figure(
        fig=fig_3d,
        figure_id=f"{system_id}_attractor_3d",
        kind="attractor",
        metadata_dict=meta_common,
        run_id=run_id,
        report_targets=report_targets
    )
    plt.close(fig_3d)
    
    # 2. 2D Projections (xy, xz, yz)
    projections = [
        ("xy", x_plot, y_plot, '$x$', '$y$'),
        ("xz", x_plot, z_plot, '$x$', '$z$'),
        ("yz", y_plot, z_plot, '$y$', '$z$')
    ]
    
    exported_paths = {
        "3d_pdf": str(pdf_3d),
        "3d_png": str(png_3d)
    }
    
    for proj_name, u_vals, v_vals, xlabel, ylabel in projections:
        fig_2d = plt.figure(figsize=get_figsize("2d"))
        ax_2d = fig_2d.add_subplot(111)
        apply_axes_style(ax_2d, grid=True)
        
        ax_2d.plot(u_vals, v_vals, color='#10b981', linewidth=lw, alpha=0.8, label='Attractor')
        
        if include_eq and equilibria:
            for name, eq in equilibria.items():
                color = '#ef4444' if name != 'E0' else '#3b82f6'
                marker = '^' if name == 'E0' else 'o'
                u_coord = eq[0] if '$x$' in xlabel else eq[1]
                v_coord = eq[1] if '$y$' in ylabel else eq[2]
                ax_2d.scatter([u_coord], [v_coord], color=color, marker=marker, s=40, edgecolors='black', zorder=100, label=name)
                
        ax_2d.set_xlabel(xlabel, fontsize=10)
        ax_2d.set_ylabel(ylabel, fontsize=10)
        
        if include_eq and equilibria:
            ax_2d.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='white', edgecolor='#cbd5e1')
            
        proj_meta = meta_common.copy()
        proj_meta["caption_key"] = f"fig_attractor_{system_id}_{proj_name}"
        
        pdf_p, png_p = export_figure(
            fig=fig_2d,
            figure_id=f"{system_id}_attractor_{proj_name}",
            kind="attractor",
            metadata_dict=proj_meta,
            run_id=run_id,
            report_targets=report_targets
        )
        plt.close(fig_2d)
        
        exported_paths[f"{proj_name}_pdf"] = str(pdf_p)
        exported_paths[f"{proj_name}_png"] = str(png_p)
        
    return exported_paths

def render_basin(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    basin_grid: np.ndarray,
    config: dict,
    run_id: str = "default_run",
    report_targets: List[str] = None
) -> Tuple[str, str]:
    """
    Renders the basin of attraction map.
    Saves JSON containing plane, fixed variables, domain, resolution, classification criteria and final time.
    """
    apply_library_style()
    
    fig = plt.figure(figsize=get_figsize("basin"))
    ax = fig.add_subplot(111)
    apply_axes_style(ax)
    
    # Custom discrete colors for classes:
    # 0: target attractor (red), 1: stable equilibrium (blue), 2: divergence (orange), 3: other attractor (purple)
    cmap = plt.cm.colors.ListedColormap(['#ef4444', '#3b82f6', '#f59e0b', '#8b5cf6'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Draw the basin
    im = ax.imshow(
        basin_grid,
        extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
        origin='lower',
        cmap=cmap,
        norm=norm,
        aspect='auto',
        alpha=0.85
    )
    
    # Add a consistent colorbar or legend
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.8)
    cbar.ax.set_yticklabels(['Target Attractor', 'Stable Eq', 'Divergence', 'Other Attractor'])
    cbar.outline.set_edgecolor('#cbd5e1')
    cbar.outline.set_linewidth(0.5)
    
    ax.set_xlabel('$x(0)$', fontsize=10)
    ax.set_ylabel('$y(0)$', fontsize=10)
    
    system_id = config.get("system_id", "chua_nonsmooth")
    q = config.get("q", config.get("system_params", {}).get("q", 1.0))
    
    metadata = {
        "caption_key": f"fig_basin_{system_id}",
        "source_script": "renderers.py",
        "source_function": "render_basin",
        "system_id": system_id,
        "q": str(q),
        "parameters": config.get("system_params", {}),
        "plane": config.get("basin_plane", "xy"),
        "fixed_variables": config.get("fixed_variables", {"z": 0.0}),
        "domain": {
            "x_range": [float(grid_x.min()), float(grid_x.max())],
            "y_range": [float(grid_y.min()), float(grid_y.max())]
        },
        "resolution": [int(grid_x.shape[0]), int(grid_y.shape[0])],
        "classification_criteria": config.get("classification_criteria", "endpoint_state"),
        "t_final": config.get("t_final", 100.0),
        "integrator": config.get("integrator", "abm"),
        "memory_mode": config.get("memory_mode", "fractional"),
        "data_sources": ["basin_simulation_grid"]
    }
    
    pdf_p, png_p = export_figure(
        fig=fig,
        figure_id=f"{system_id}_basin",
        kind="basin",
        metadata_dict=metadata,
        run_id=run_id,
        report_targets=report_targets
    )
    plt.close(fig)
    return str(pdf_p), str(png_p)

def render_nyquist(
    freqs: np.ndarray,
    w_evals: np.ndarray,
    n_evals: np.ndarray,
    candidates: List[Tuple[float, float, float]] = None,
    config: dict = None,
    run_id: str = "default_run",
    report_targets: List[str] = None
) -> Tuple[str, str]:
    """
    Renders describing function / Nyquist plots distinguishing:
      - W_q(j\\omega) transfer function curve
      - Critical point -1 / N(A) locus
      - Candidate crossings
    Saves JSON containing q, omega, lambda, A, N(A), residual, and transfer mode.
    """
    if config is None:
        config = {}
    if candidates is None:
        candidates = []
        
    apply_library_style()
    
    fig = plt.figure(figsize=get_figsize("2d"))
    ax = fig.add_subplot(111)
    apply_axes_style(ax, grid=True)
    
    # 1. Plot W_q(jw) or \hat{W}_q(\lambda) curve
    # Note: Using mathematical notation for labels as requested
    ax.plot(np.real(w_evals), np.imag(w_evals), color='#3b82f6', linewidth=1.5, label=r'$W_q(j\omega)$')
    
    # 2. Plot -1/N(A) locus
    critical_locus = -1.0 / n_evals
    ax.plot(np.real(critical_locus), np.imag(critical_locus), color='#10b981', linewidth=1.5, linestyle='--', label=r'$-1/N(A)$')
    
    # 3. Plot candidate crossing points
    system_id = config.get("system_id", "chua_nonsmooth")
    q = config.get("q", 1.0)
    
    crossing_lambdas = []
    crossing_omegas = []
    crossing_As = []
    crossing_N_As = []
    crossing_residuals = []
    
    for idx, (A, w, k) in enumerate(candidates):
        # Calculate evaluation details
        lambd = (1j * w) ** q
        n_a = k  # equivalent gain
        
        # Approximate evaluation
        val = np.interp(w, freqs, w_evals) if len(freqs) > 0 else complex(np.nan, np.nan)
        residual = np.abs(1.0 + n_a * val)
        
        crossing_lambdas.append(str(lambd))
        crossing_omegas.append(float(w))
        crossing_As.append(float(A))
        crossing_N_As.append(float(n_a))
        crossing_residuals.append(float(residual))
        
        # Draw on plot
        ax.scatter([np.real(val)], [np.imag(val)], color='#ef4444', marker='*', s=80, zorder=50,
                   label=f'Candidate {idx+1}: $A_0$={A:.2f}, $\\omega_0$={w:.2f}' if idx == 0 else f'Candidate {idx+1}')
                   
    # Plot the critical point -1
    ax.scatter([-1.0], [0.0], color='black', marker='x', s=50, zorder=100, label=r'$-1$')
    
    ax.set_xlabel(r'$\operatorname{Re}$', fontsize=10)
    ax.set_ylabel(r'$\operatorname{Im}$', fontsize=10)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='white', edgecolor='#cbd5e1')
    
    # Build metadata containing physical and mathematical settings
    metadata = {
        "caption_key": f"fig_nyquist_{system_id}",
        "source_script": "renderers.py",
        "source_function": "render_nyquist",
        "system_id": system_id,
        "q": str(q),
        "parameters": config.get("system_params", {}),
        "transfer_mode": config.get("transfer_mode", "linear"),
        "omegas": crossing_omegas,
        "lambdas": crossing_lambdas,
        "amplitudes": crossing_As,
        "N_amplitudes": crossing_N_As,
        "residuals": crossing_residuals,
        "data_sources": ["transfer_function_computation"]
    }
    
    pdf_p, png_p = export_figure(
        fig=fig,
        figure_id=f"{system_id}_nyquist",
        kind="nyquist",
        metadata_dict=metadata,
        run_id=run_id,
        report_targets=report_targets
    )
    plt.close(fig)
    return str(pdf_p), str(png_p)

def render_matignon(
    eigenvalues: np.ndarray,
    q: float,
    config: dict,
    run_id: str = "default_run",
    report_targets: List[str] = None
) -> Tuple[str, str]:
    """
    Renders the Matignon stability complex plane.
    Shows the fractional stability sector (|arg(lambda)| > q*pi/2).
    Saves JSON with eigenvalues and stability measure.
    """
    apply_library_style()
    
    fig = plt.figure(figsize=get_figsize("matignon"))
    ax = fig.add_subplot(111)
    apply_axes_style(ax, grid=True)
    
    max_radius = float(np.max(np.abs(eigenvalues))) if len(eigenvalues) > 0 else 1.0
    if max_radius < 1e-12:
        max_radius = 1.0
    limit = max_radius * 1.5
    
    # 1. Fill unstable region (where |arg(\lambda)| <= q\pi/2)
    t_vals = np.linspace(-q * np.pi / 2.0, q * np.pi / 2.0, 300)
    R = limit * 2.0
    x_fill = [0.0] + list(R * np.cos(t_vals)) + [0.0]
    y_fill = [0.0] + list(R * np.sin(t_vals)) + [0.0]
    
    # Red fill for unstable region
    ax.fill(x_fill, y_fill, color='#fee2e2', alpha=0.75, label='Unstable Region', edgecolor='#fca5a5', linewidth=1.0)
    
    # Border boundaries |arg(lambda)| = q*pi/2
    ax.plot([0.0, R * np.cos(q * np.pi / 2.0)], [0.0, R * np.sin(q * np.pi / 2.0)], color='#ef4444', linestyle='--', linewidth=1.2, label=r'$|\arg(\lambda)| = q\pi/2$')
    ax.plot([0.0, R * np.cos(-q * np.pi / 2.0)], [0.0, R * np.sin(-q * np.pi / 2.0)], color='#ef4444', linestyle='--', linewidth=1.2)
    
    # 2. Plot eigenvalues
    ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), color='#3b82f6', marker='o', s=60, edgecolors='black', zorder=100, label='Eigenvalues')
    
    ax.axhline(0.0, color='#64748b', linewidth=0.8, linestyle=':')
    ax.axvline(0.0, color='#64748b', linewidth=0.8, linestyle=':')
    
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    
    ax.set_xlabel(r'$\operatorname{Re}(\lambda)$', fontsize=10)
    ax.set_ylabel(r'$\operatorname{Im}(\lambda)$', fontsize=10)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, facecolor='white', edgecolor='#cbd5e1')
    
    # Compute instability measure
    stability_measures = []
    is_stable = True
    for val in eigenvalues:
        arg_val = np.abs(np.arctan2(np.imag(val), np.real(val)))
        threshold = q * np.pi / 2.0
        inst_measure = threshold - arg_val
        stability_measures.append(float(inst_measure))
        if arg_val <= threshold:
            is_stable = False
            
    system_id = config.get("system_id", "chua_nonsmooth")
    
    metadata = {
        "caption_key": f"fig_matignon_{system_id}",
        "source_script": "renderers.py",
        "source_function": "render_matignon",
        "system_id": system_id,
        "q": str(q),
        "eigenvalues": [str(x) for x in eigenvalues],
        "is_stable": is_stable,
        "instability_measures": stability_measures,
        "data_sources": ["equilibrium_linearization"]
    }
    
    pdf_p, png_p = export_figure(
        fig=fig,
        figure_id=f"{system_id}_matignon",
        kind="matignon",
        metadata_dict=metadata,
        run_id=run_id,
        report_targets=report_targets
    )
    plt.close(fig)
    return str(pdf_p), str(png_p)
