import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..systems import get_system
from ..lure.transfer import W_eval
from ..verification.stability import classify_equilibrium_stability

def first_harmonic_reconstruction(traj: np.ndarray, tail_fraction: float = 0.85) -> np.ndarray:
    """Helper to perform first harmonic (linearized) reconstruction of the attractor trajectory."""
    n0 = int((1.0 - tail_fraction) * len(traj))
    tail = traj[n0:, 1:4]
    tail = tail[np.all(np.isfinite(tail), axis=1)]
    n = len(tail)
    if n < 32:
        return traj[n0:, :]
    centered = tail - tail.mean(axis=0)
    fft_x = np.fft.rfft(centered[:, 0])
    k = int(np.argmax(np.abs(fft_x[1:])) + 1)
    coeffs = np.fft.rfft(centered, axis=0)
    keep = np.zeros_like(coeffs)
    keep[0, :] = coeffs[0, :]
    keep[k, :] = coeffs[k, :]
    recon = np.fft.irfft(keep, n=n, axis=0) + tail.mean(axis=0)
    t = traj[n0:n0+n, 0:1]
    return np.hstack([t, recon])

def downsample(arr: np.ndarray, max_points: int) -> np.ndarray:
    if len(arr) <= max_points:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_points).astype(int)
    return arr[idx]

def save_and_close(fig, path: Path):
    """Saves the figure in both PNG and PDF formats and closes the plot."""
    path_png = path.with_suffix(".png")
    path_pdf = path.with_suffix(".pdf")
    
    path_png.parent.mkdir(parents=True, exist_ok=True)
    
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, str(path_png), "publication")
    pass
    plt.close(fig)
    print(f"[Publication Figures] Saved: {path.name}.png and {path.name}.pdf")

def generate_all_publication_figures(output_dir: str, config: Dict[str, Any]) -> None:
    """
    Core post-processor that parses raw data and configuration from a workflow run
    and produces vector PDF + high-resolution PNG figures.
    """
    out_dir_path = Path(output_dir)
    fig_dir = out_dir_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = out_dir_path / "summary.json"
    effective_cfg_path = out_dir_path / "effective_config.json"
    
    if not summary_path.exists() or not effective_cfg_path.exists():
        print(f"[Publication Figures] WARNING: Missing summary/config in {output_dir}. Skipping generation.")
        return
        
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(effective_cfg_path, "r", encoding="utf-8") as f:
        eff_config = json.load(f)
        
    system_id = eff_config.get("system_id")
    q = eff_config.get("q", 1.0)
    
    system_params = {
        "alpha": eff_config.get("alpha", 8.4562),
        "beta": eff_config.get("beta", 12.0732),
        "gamma": eff_config.get("gamma", 0.0052),
        "m0": eff_config.get("m0", -0.1768),
        "m1": eff_config.get("m1", -1.1468),
        "q": q
    }
    if "chua_fractional_arctan" in system_id:
        system_params["m"] = eff_config.get("m", 1.5)
        system_params["n"] = eff_config.get("n", 10.0)
    elif "polynomial" in system_id:
        system_params["coeff"] = eff_config.get("coeff", 1.0)
        
    # Translate old system IDs if needed
    name_map = {
        "chua_piecewise": "chua-nonsmooth",
        "chua_integer_saturation": "chua-nonsmooth",
        "chua_fractional_saturation": "chua-nonsmooth",
        "chua_integer_arctan": "chua-arctan",
        "chua_fractional_arctan": "chua-arctan",
        "chua_arctan_wu2023": "fractional-chua-arctan-wu2023",
    }
    normalized_sys_id = name_map.get(system_id, system_id)
    system = get_system(normalized_sys_id)
    
    # Inject overrides as attributes and parameters
    merged_params = dict(system.parameters)
    merged_params.update(system_params)
    object.__setattr__(system, "parameters", merged_params)
    object.__setattr__(system, "q", q)
    for k, v in merged_params.items():
        try:
            object.__setattr__(system, k, v)
        except Exception:
            pass
            
    if system.lure is not None:
        object.__setattr__(system, "P", system.lure.matrix)
        object.__setattr__(system, "b", system.lure.input_vector)
        object.__setattr__(system, "r", system.lure.output_vector)
        object.__setattr__(system, "describing_function", system.lure.describing_function)
        
    omega0 = summary.get("omega0")
    a0 = summary.get("amplitude_a0")
    k = summary.get("k")
    
    # 2. FIG02: Continuation plots
    trace_json_path = out_dir_path / "continuation_trace.json"
    if trace_json_path.exists():
        with open(trace_json_path, "r", encoding="utf-8") as f:
            trace = json.load(f)
            
        etas = [s["lambda_value"] for s in trace]
        x_out_norms = [s["x_out_norm"] for s in trace]
        
        steps_coords = []
        for s in trace:
            step_idx = s["step_idx"]
            step_csv = out_dir_path / "continuation_steps" / f"continuation_eta_{step_idx:03d}.csv"
            if step_csv.exists():
                with open(step_csv, "r", encoding="utf-8") as sf:
                    rows = list(csv.DictReader(sf))
                    if rows:
                        last_row = rows[-1]
                        keys = list(last_row.keys())
                        c_keys = [k for k in ["x0", "x1", "x2"] if k in keys]
                        if not c_keys:
                            c_keys = [k for k in ["x", "y", "z"] if k in keys]
                        if len(c_keys) >= 3:
                            steps_coords.append((s["lambda_value"], float(last_row[c_keys[0]]), float(last_row[c_keys[1]]), float(last_row[c_keys[2]])))
                            
        if steps_coords:
            steps_coords = sorted(steps_coords, key=lambda x: x[0])
            
        first_step_csv = out_dir_path / "continuation_steps" / "continuation_eta_000.csv"
        csv_files = sorted((out_dir_path / "continuation_steps").glob("continuation_eta_*.csv"))
        if first_step_csv.exists() and len(csv_files) >= 2:
            last_step_csv = csv_files[-1]
            
            def load_traj_coords(csv_path: Path) -> np.ndarray:
                with csv_path.open("r", newline="", encoding="utf-8") as sf:
                    rows = list(csv.DictReader(sf))
                    keys = list(rows[0].keys())
                    c_keys = [k for k in ["x0", "x1", "x2"] if k in keys]
                    if not c_keys:
                        c_keys = [k for k in ["x", "y", "z"] if k in keys]
                    return np.array([[float(r[c_keys[0]]), float(r[c_keys[1]]), float(r[c_keys[2]])] for r in rows])
                    
            try:
                first_traj = load_traj_coords(first_step_csv)
                last_traj = load_traj_coords(last_step_csv)
                
                fig2d = plt.figure(figsize=(8.0, 7.0), dpi=300)
                ax2d = fig2d.add_subplot(111, projection="3d")
                
                first_small = downsample(first_traj, 1500)
                last_small = downsample(last_traj, 2000)
                
                ax2d.plot(first_small[:, 0], first_small[:, 1], first_small[:, 2], color="blue", linewidth=1.8, label="Linearized Attractor ($\\eta=0.0$)")
                ax2d.plot(last_small[:, 0], last_small[:, 1], last_small[:, 2], color="red", linewidth=1.8, label="Nonlinear Attractor ($\\eta=1.0$)")
                
                if steps_coords:
                    path_arr = np.array([[x[1], x[2], x[3]] for x in steps_coords])
                    ax2d.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2], "k--", marker="o", markersize=4, linewidth=1.2, label="Continuation path")
                    
                ax2d.set_title("Numerical Continuation: Trajectory Evolution", fontsize=11, fontweight="bold", pad=12)
                ax2d.set_xlabel("x")
                ax2d.set_ylabel("y")
                ax2d.set_zlabel("z")
                ax2d.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
                save_and_close(fig2d, fig_dir / "fig02d_continuation_story")
            except Exception as e:
                print(f"[Publication Figures] Failed to render fig02d: {e}")
                
    # 3. FIG03: Final Attractor Trajectory and projections
    final_attractor_csv = out_dir_path / "final_attractor.csv"
    if final_attractor_csv.exists():
        with final_attractor_csv.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            traj_data = np.array([[float(r["t"]), float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
            
        x_col = traj_data[:, 1]
        y_col = traj_data[:, 2]
        z_col = traj_data[:, 3]
        
        fig3 = plt.figure(figsize=(8.0, 7.0), dpi=300)
        ax3 = fig3.add_subplot(111, projection="3d")
        ax3.plot(x_col, y_col, z_col, color="#ef4444", linewidth=1.0, alpha=0.85, label="Final Nonlinear Attractor")
        ax3.scatter([x_col[-1]], [y_col[-1]], [z_col[-1]], color="black", s=45, zorder=5, label="Endpoint")
        ax3.set_title("Final Attractor Trajectory (3D)", fontsize=11, fontweight="bold", pad=12)
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("z")
        ax3.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
        save_and_close(fig3, fig_dir / "fig03_final_attractor")
        
        fig3a, ax3a = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        ax3a.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
        ax3a.plot(x_col, y_col, color="#ef4444", linewidth=0.9, alpha=0.85)
        ax3a.set_title("Final Attractor XY Projection", fontsize=11, fontweight="bold", pad=12)
        ax3a.set_xlabel("x")
        ax3a.set_ylabel("y")
        save_and_close(fig3a, fig_dir / "fig03a_final_attractor_xy")
        
        fig3b, ax3b = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        ax3b.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
        ax3b.plot(x_col, z_col, color="#ef4444", linewidth=0.9, alpha=0.85)
        ax3b.set_title("Final Attractor XZ Projection", fontsize=11, fontweight="bold", pad=12)
        ax3b.set_xlabel("x")
        ax3b.set_ylabel("z")
        save_and_close(fig3b, fig_dir / "fig03b_final_attractor_xz")
        
        fig3c, ax3c = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        ax3c.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
        ax3c.plot(y_col, z_col, color="#ef4444", linewidth=0.9, alpha=0.85)
        ax3c.set_title("Final Attractor YZ Projection", fontsize=11, fontweight="bold", pad=12)
        ax3c.set_xlabel("y")
        ax3c.set_ylabel("z")
        save_and_close(fig3c, fig_dir / "fig03c_final_attractor_yz")
        
        try:
            recon = first_harmonic_reconstruction(traj_data)
            t_final = traj_data[-1, 0]
            mask_50 = traj_data[:, 0] >= (t_final - 50.0)
            traj_data_50 = traj_data[mask_50]
            recon_50 = recon[recon[:, 0] >= (t_final - 50.0)]
            
            orig_small = downsample(traj_data_50, 4000)
            recon_small = downsample(recon_50, 1500)
            
            fig3d, ax3d = plt.subplots(figsize=(8.0, 4.5), dpi=300)
            ax3d.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
            ax3d.plot(orig_small[:, 0], orig_small[:, 1], color="red", linewidth=1.2, alpha=0.85, label="Original")
            ax3d.plot(recon_small[:, 0], recon_small[:, 1], "--", color="purple", linewidth=1.4, alpha=0.9, label="Linearized")
            ax3d.set_title("Linear vs Original: $x(t)$ Component", fontsize=11, fontweight="bold", pad=12)
            ax3d.set_xlabel("Time $t$")
            ax3d.set_ylabel("$x$")
            ax3d.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
            save_and_close(fig3d, fig_dir / "fig03d_linear_vs_original_x")
            
            fig3e, ax3e = plt.subplots(figsize=(8.0, 4.5), dpi=300)
            ax3e.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
            ax3e.plot(orig_small[:, 0], orig_small[:, 2], color="red", linewidth=1.2, alpha=0.85, label="Original")
            ax3e.plot(recon_small[:, 0], recon_small[:, 2], "--", color="purple", linewidth=1.4, alpha=0.9, label="Linearized")
            ax3e.set_title("Linear vs Original: $y(t)$ Component", fontsize=11, fontweight="bold", pad=12)
            ax3e.set_xlabel("Time $t$")
            ax3e.set_ylabel("$y$")
            ax3e.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
            save_and_close(fig3e, fig_dir / "fig03e_linear_vs_original_y")
            
            fig3f, ax3f = plt.subplots(figsize=(8.0, 4.5), dpi=300)
            ax3f.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
            ax3f.plot(orig_small[:, 0], orig_small[:, 3], color="red", linewidth=1.2, alpha=0.85, label="Original")
            ax3f.plot(recon_small[:, 0], recon_small[:, 3], "--", color="purple", linewidth=1.4, alpha=0.9, label="Linearized")
            ax3f.set_title("Linear vs Original: $z(t)$ Component", fontsize=11, fontweight="bold", pad=12)
            ax3f.set_xlabel("Time $t$")
            ax3f.set_ylabel("$z$")
            ax3f.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
            save_and_close(fig3f, fig_dir / "fig03f_linear_vs_original_z")
            
            fig3g = plt.figure(figsize=(8.0, 7.0), dpi=300)
            ax3g = fig3g.add_subplot(111, projection="3d")
            ax3g.plot(orig_small[:, 1], orig_small[:, 2], orig_small[:, 3], color="red", linewidth=1.2, alpha=0.85, label="Original")
            ax3g.plot(recon_small[:, 1], recon_small[:, 2], recon_small[:, 3], "--", color="purple", linewidth=1.5, alpha=0.9, label="Linearized")
            ax3g.set_title("Linear vs Original Attractor in 3D", fontsize=11, fontweight="bold", pad=12)
            ax3g.set_xlabel("x")
            ax3g.set_ylabel("y")
            ax3g.set_zlabel("z")
            ax3g.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
            save_and_close(fig3g, fig_dir / "fig03g_linear_vs_original_3d")
            
        except Exception as e:
            print(f"[Publication Figures] Failed to render first harmonic comparisons: {e}")
            
        try:
            n_burn = int(len(traj_data) * 0.25)
            dt = traj_data[1, 0] - traj_data[0, 0]
            
            tail_x = x_col[n_burn:]
            tail_y = y_col[n_burn:]
            tail_z = z_col[n_burn:]
            
            for component_name, tail_comp, fig_id in [("x", tail_x, "fig11a_fft_x"), ("y", tail_y, "fig11b_fft_y"), ("z", tail_z, "fig11c_fft_z")]:
                centered_comp = tail_comp - tail_comp.mean()
                fft_vals = np.fft.rfft(centered_comp)
                fft_freqs = np.fft.rfftfreq(len(centered_comp), d=dt)
                fft_mag = np.abs(fft_vals) / len(centered_comp)
                
                omega_rad_s = 2.0 * np.pi * fft_freqs
                
                fig11_fft, ax11_fft = plt.subplots(figsize=(7.5, 4.5), dpi=300)
                ax11_fft.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
                ax11_fft.plot(omega_rad_s, fft_mag, color="#111827", linewidth=1.0)
                
                if omega0 and not np.isnan(omega0):
                    ax11_fft.axvline(omega0, color="#ef4444", linestyle=":", label=r"Predicted frequency $\omega_0$")
                    
                ax11_fft.set_title(f"Spectral Analysis FFT: component {component_name}", fontsize=11, fontweight="bold", pad=12)
                ax11_fft.set_xlabel(r"Frequency $\omega$ (rad/s)")
                ax11_fft.set_ylabel("Amplitude")
                ax11_fft.set_xlim(0, 15.0)
                ax11_fft.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
                save_and_close(fig11_fft, fig_dir / fig_id)
                
            for component_name, tail_comp, fig_id in [("x", tail_x, "fig11d_psd_x"), ("y", tail_y, "fig11e_psd_y"), ("z", tail_z, "fig11f_psd_z")]:
                nperseg = min(256, len(tail_comp))
                freqs_psd, psd_vals = welch(tail_comp - tail_comp.mean(), fs=1.0/dt, nperseg=nperseg)
                omega_rad_s = 2.0 * np.pi * freqs_psd
                
                fig11_psd, ax11_psd = plt.subplots(figsize=(7.5, 4.5), dpi=300)
                ax11_psd.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
                ax11_psd.plot(omega_rad_s, psd_vals, color="#0284c7", linewidth=1.2)
                
                if omega0 and not np.isnan(omega0):
                    ax11_psd.axvline(omega0, color="#ef4444", linestyle=":", label=r"Predicted frequency $\omega_0$")
                    
                ax11_psd.set_title(f"Welch PSD Power Density: component {component_name}", fontsize=11, fontweight="bold", pad=12)
                ax11_psd.set_xlabel(r"Frequency $\omega$ (rad/s)")
                ax11_psd.set_ylabel("Power Density")
                ax11_psd.set_xlim(0, 15.0)
                ax11_psd.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
                save_and_close(fig11_psd, fig_dir / fig_id)
                
        except Exception as e:
            print(f"[Publication Figures] Failed to render spectral figures: {e}")
            
    # 5. FIG04: Matignon stability diagram / Reference section
    try:
        from ..verification.equilibria import solve_equilibria
        eq_pts = solve_equilibria(system)
        
        fig4 = plt.figure(figsize=(7.5, 6.5), dpi=300)
        ax4 = fig4.add_subplot(111)
        
        ax4.set_facecolor("#f0fdf4")
        
        all_eigvals = []
        eq_details = []
        for name, eq_pt in eq_pts.items():
            res = classify_equilibrium_stability(system, eq_pt)
            all_eigvals.extend(res["eigenvalues"])
            eq_details.append((name, res))
            
        all_eigvals = np.array(all_eigvals)
        max_rad = float(np.max(np.abs(all_eigvals))) if len(all_eigvals) > 0 else 1.0
        limit = max_rad * 1.5
        R = limit * 2.0
        
        t_vals = np.linspace(-q * np.pi / 2.0, q * np.pi / 2.0, 400)
        x_fill = [0.0] + list(R * np.cos(t_vals)) + [0.0]
        y_fill = [0.0] + list(R * np.sin(t_vals)) + [0.0]
        ax4.fill(x_fill, y_fill, color="#fee2e2", alpha=0.9, edgecolor="#fca5a5", linewidth=0.8, label="Unstable Region")
        
        ax4.plot([0.0, R * np.cos(q * np.pi / 2.0)], [0.0, R * np.sin(q * np.pi / 2.0)], color="#ef4444", linestyle="--", linewidth=1.1, label=r"Frontier $|\arg(\lambda)| = q\pi/2$")
        ax4.plot([0.0, R * np.cos(-q * np.pi / 2.0)], [0.0, R * np.sin(-q * np.pi / 2.0)], color="#ef4444", linestyle="--", linewidth=1.1)
        
        ax4.axhline(0.0, color="#64748b", linewidth=0.7, linestyle=":")
        ax4.axvline(0.0, color="#64748b", linewidth=0.7, linestyle=":")
        
        colors = {"E0": "#3b82f6", "E+": "#ef4444", "E-": "#f59e0b"}
        markers = {"E0": "^", "E+": "o", "E-": "s"}
        
        for name, res in eq_details:
            color = colors.get(name, "#8b5cf6")
            marker = markers.get(name, "d")
            eigvals = res["eigenvalues"]
            ax4.scatter(np.real(eigvals), np.imag(eigvals), color=color, marker=marker, s=70, edgecolors="black", zorder=10, label=f"{name} eigenvalues")
            
        ax4.set_xlim(-limit, limit)
        ax4.set_ylim(-limit, limit)
        ax4.set_aspect("equal")
        ax4.set_title(f"Matignon Stability Plane (q={q:.4f})", fontsize=11, fontweight="bold", pad=12)
        ax4.set_xlabel(r"$\mathrm{Re}(\lambda)$")
        ax4.set_ylabel(r"$\mathrm{Im}(\lambda)$")
        ax4.legend(loc="upper right", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
        save_and_close(fig4, fig_dir / "fig04_reference_section")
    except Exception as e:
        print(f"[Publication Figures] Failed to render fig04: {e}")
        
    # 7. FIG06 & FIG10: Basin slice overlay plots (if available)
    basin_csv_path = out_dir_path / "basin_results.csv"
    if basin_csv_path.exists():
        try:
            with open(basin_csv_path, "r", newline="", encoding="utf-8") as bf:
                b_rows = list(csv.DictReader(bf))
                
            planes = {}
            for r in b_rows:
                plane = r.get("plane", "global")
                if plane not in planes:
                    planes[plane] = []
                planes[plane].append(r)
                
            for plane_name, p_data in planes.items():
                u_vals = np.array([float(x["u_val"]) for x in p_data])
                v_vals = np.array([float(x["v_val"]) for x in p_data])
                codes = np.array([int(x["classification_code"]) for x in p_data])
                
                u_unique = np.unique(u_vals)
                v_unique = np.unique(v_vals)
                
                if len(u_unique) * len(v_unique) == len(codes):
                    U_mesh, V_mesh = np.meshgrid(u_unique, v_unique)
                    code_mesh = np.zeros((len(v_unique), len(u_unique)))
                    for item in p_data:
                        col_idx = np.where(u_unique == float(item["u_val"]))[0][0]
                        row_idx = np.where(v_unique == float(item["v_val"]))[0][0]
                        code_mesh[row_idx, col_idx] = int(item["classification_code"])
                        
                    fig6, ax6 = plt.subplots(figsize=(7.5, 6.5), dpi=300)
                    
                    from matplotlib.colors import ListedColormap
                    custom_colors = ["#ff66b2", "#8b5cf6", "#94a3b8", "#facc15", "#475569", "#3b82f6"]
                    cmap = ListedColormap(custom_colors[:max(3, len(np.unique(codes)))])
                    
                    mesh = ax6.pcolormesh(U_mesh, V_mesh, code_mesh, cmap=cmap, shading="nearest", alpha=0.92)
                    
                    from matplotlib.patches import Patch
                    labels_mapping = {
                        0: "Target Attractor (Pink)",
                        1: "Stable Equilibrium (Purple)",
                        2: "Divergence (Gray)",
                        3: "Other Attractor (Yellow)",
                        4: "Numerical Failure (Dark)",
                        5: "Unclassified (Blue)"
                    }
                    legend_patches = []
                    for code in np.unique(code_mesh):
                        c_idx = int(code)
                        if c_idx < len(custom_colors):
                            color = custom_colors[c_idx]
                            label = labels_mapping.get(c_idx, f"Code {c_idx}")
                            legend_patches.append(Patch(facecolor=color, label=label, edgecolor="black", linewidth=0.5))
                            
                    eq_selected = eff_config.get("basin", {}).get("equilibrium_selection", "E+")
                    if eq_selected in eq_pts:
                        eq_pt = eq_pts[eq_selected]
                        ax6.scatter([eq_pt[0]], [eq_pt[1]], color="red", marker="*", s=160, edgecolors="black", zorder=12, label=f"Equilibrium {eq_selected}")
                        
                    ax6.legend(handles=legend_patches, loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
                    
                    fixed_z = eff_config.get("basin", {}).get("fixed_z", 0.0)
                    ax6.set_title(f"Basin Attraction Slice (Plane xy, z = {fixed_z:.2f})", fontsize=11, fontweight="bold", pad=12)
                    ax6.set_xlabel("u coordinate")
                    ax6.set_ylabel("v coordinate")
                    
                    if plane_name == "xy" or plane_name == "xy_z0":
                        fig_id = "fig06a_basin_overlay_z0"
                    elif "zfinal" in plane_name or "z_final" in plane_name:
                        fig_id = "fig06b_basin_overlay_zfinal"
                    else:
                        fig_id = f"fig10_{plane_name}_basin_slice"
                        
                    save_and_close(fig6, fig_dir / fig_id)
                else:
                    fig6, ax6 = plt.subplots(figsize=(7.5, 6.5), dpi=300)
                    scatter = ax6.scatter(u_vals, v_vals, c=codes, s=4, cmap="plasma", alpha=0.85)
                    fig6.colorbar(scatter, ax=ax6)
                    ax6.set_title(f"Basin Attraction Slice: Plane {plane_name}", fontsize=11, fontweight="bold", pad=12)
                    ax6.set_xlabel("u coordinate")
                    ax6.set_ylabel("v coordinate")
                    save_and_close(fig6, fig_dir / f"fig10_{plane_name}_basin_slice")
                    
        except Exception as e:
            print(f"[Publication Figures] Failed to render basin slices: {e}")
            
    print("[Publication Figures] Success: Completed all publication-grade vectors! [OK]")
