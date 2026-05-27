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

from ..systems.registry import get_system_by_id
from ..lure.transfer import W_eval
from ..verification.stability import classify_equilibrium_stability

def first_harmonic_reconstruction(traj: np.ndarray, tail_fraction: float = 0.85) -> np.ndarray:
    """Helper to perform first harmonic (linearized) reconstruction of the attractor trajectory."""
    # Slices the last tail_fraction of points
    n0 = int((1.0 - tail_fraction) * len(traj))
    tail = traj[n0:, 1:4] # columns t, x, y, z
    tail = tail[np.all(np.isfinite(tail), axis=1)]
    n = len(tail)
    if n < 32:
        # Trajectory too short, return duplicate as fallback
        return traj[n0:, :]
    centered = tail - tail.mean(axis=0)
    fft_x = np.fft.rfft(centered[:, 0])
    # Dominant frequency index (excluding DC component)
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
    
    # Ensure directory exists
    path_png.parent.mkdir(parents=True, exist_ok=True)
    
    # Save both vector PDF and high-res PNG
    fig.savefig(str(path_png), dpi=300, bbox_inches='tight')
    fig.savefig(str(path_pdf), bbox_inches='tight')
    plt.close(fig)
    print(f"[Publication Figures] Saved: {path.name}.png and {path.name}.pdf")

def generate_all_publication_figures(output_dir: str, config: Dict[str, Any]) -> None:
    """
    Core post-processor that parses raw data and configuration from a workflow run
    and produces vector PDF + high-resolution PNG figures named exactly like those in
    'chua_integer_runs/balanced/final_pdf_figs'.
    """
    out_dir_path = Path(output_dir)
    fig_dir = out_dir_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw inputs
    summary_path = out_dir_path / "summary.json"
    effective_cfg_path = out_dir_path / "effective_config.json"
    
    if not summary_path.exists() or not effective_cfg_path.exists():
        print(f"[Publication Figures] WARNING: Missing summary/config in {output_dir}. Skipping generation.")
        return
        
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(effective_cfg_path, "r", encoding="utf-8") as f:
        eff_config = json.load(f)
        
    # Recreate the exact system
    system_id = eff_config.get("system_id")
    q = eff_config.get("q", 1.0)
    
    # Extract parameter overrides safely
    system_params = {
        "alpha": eff_config.get("alpha", 8.4562),
        "beta": eff_config.get("beta", 12.0732),
        "gamma": eff_config.get("gamma", 0.0052),
        "m0": eff_config.get("m0", -0.1768),
        "m1": eff_config.get("m1", -1.1468),
        "q": q
    }
    # ChuaArctan uses m, n instead of m0, m1
    if "chua_fractional_arctan" in system_id:
        system_params["m"] = eff_config.get("m", 1.5)
        system_params["n"] = eff_config.get("n", 10.0)
    elif "polynomial" in system_id:
        system_params["coeff"] = eff_config.get("coeff", 1.0)
        
    system = get_system_by_id(system_id, **system_params)
        
    # Retrieve seed and candidates
    omega0 = summary.get("omega0")
    a0 = summary.get("amplitude_a0")
    k = summary.get("k")
    
    # -------------------------------------------------------------------------
    # 1. FIG01: Transfer function & Describing Function
    # -------------------------------------------------------------------------
    # Fig01, Fig01b, and Fig01c are now natively and perfectly generated in plot_transfer.py
    # to ensure they are always well-adjusted and to prevent any badly-adjusted duplicates.
    pass
    
    # -------------------------------------------------------------------------
    # 2. FIG02: Continuation plots
    # -------------------------------------------------------------------------
    trace_json_path = out_dir_path / "continuation_trace.json"
    if trace_json_path.exists():
        with open(trace_json_path, "r", encoding="utf-8") as f:
            trace = json.load(f)
            
        etas = [s["lambda_value"] for s in trace]
        x_out_norms = [s["x_out_norm"] for s in trace]
        
        # Load the per-step coordinates to plot coordinate progression
        # Let's plot fig02a_continuation_x, fig02b_continuation_y, fig02c_continuation_z
        steps_coords = []
        for s in trace:
            # We can approximate the coordinates of each step from trace norms or read files
            step_idx = s["step_idx"]
            step_csv = out_dir_path / "continuation_steps" / f"continuation_eta_{step_idx:03d}.csv"
            if step_csv.exists():
                with open(step_csv, "r", encoding="utf-8") as sf:
                    rows = list(csv.DictReader(sf))
                    if rows:
                        last_row = rows[-1]
                        # CSV keys can be x0, x1, x2 or x, y, z
                        keys = list(last_row.keys())
                        c_keys = [k for k in ["x0", "x1", "x2"] if k in keys]
                        if not c_keys:
                            c_keys = [k for k in ["x", "y", "z"] if k in keys]
                        if len(c_keys) >= 3:
                            steps_coords.append((s["lambda_value"], float(last_row[c_keys[0]]), float(last_row[c_keys[1]]), float(last_row[c_keys[2]])))
                            
        if steps_coords:
            steps_coords = sorted(steps_coords, key=lambda x: x[0])
            sc_etas = [x[0] for x in steps_coords]
            sc_x = [x[1] for x in steps_coords]
            sc_y = [x[2] for x in steps_coords]
            sc_z = [x[3] for x in steps_coords]
            
            # Las figuras redundantes fig02a, fig02b y fig02c de continuacion por coordenada han sido desactivadas
            pass
            
        # Plot fig02d_continuation_story (3D story plot)
        # Slices first step and last step trajectories
        first_step_csv = out_dir_path / "continuation_steps" / "continuation_eta_000.csv"
        # Find last step
        csv_files = sorted((out_dir_path / "continuation_steps").glob("continuation_eta_*.csv"))
        if first_step_csv.exists() and len(csv_files) >= 2:
            last_step_csv = csv_files[-1]
            
            # Load trajectories
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
                
                # Downsample for premium smooth rendering
                first_small = downsample(first_traj, 1500)
                last_small = downsample(last_traj, 2000)
                
                # Plot first step and last step
                ax2d.plot(first_small[:, 0], first_small[:, 1], first_small[:, 2], color="blue", linewidth=1.8, label="Linearized Attractor ($\\eta=0.0$)")
                ax2d.plot(last_small[:, 0], last_small[:, 1], last_small[:, 2], color="red", linewidth=1.8, label="Nonlinear Attractor ($\\eta=1.0$)")
                
                # Plot continuation path (initial conditions of all steps)
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
                
    # -------------------------------------------------------------------------
    # 3. FIG03: Final Attractor Trajectory and projections
    # -------------------------------------------------------------------------
    final_attractor_csv = out_dir_path / "final_attractor.csv"
    if final_attractor_csv.exists():
        # Load trajectory
        with final_attractor_csv.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            traj_data = np.array([[float(r["t"]), float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
            
        x_col = traj_data[:, 1]
        y_col = traj_data[:, 2]
        z_col = traj_data[:, 3]
        
        # Plot fig03_final_attractor (3D attractor)
        fig3 = plt.figure(figsize=(8.0, 7.0), dpi=300)
        ax3 = fig3.add_subplot(111, projection="3d")
        ax3.plot(x_col, y_col, z_col, color="#ef4444", linewidth=1.0, alpha=0.85, label="Final Nonlinear Attractor")
        
        # If final_traj has endpoint, mark it
        ax3.scatter([x_col[-1]], [y_col[-1]], [z_col[-1]], color="black", s=45, zorder=5, label="Endpoint")
        ax3.set_title("Final Attractor Trajectory (3D)", fontsize=11, fontweight="bold", pad=12)
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("z")
        ax3.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
        save_and_close(fig3, fig_dir / "fig03_final_attractor")
        
        # 2D projections: fig03a_final_attractor_xy, fig03b_final_attractor_xz, fig03c_final_attractor_yz
        # fig03a_final_attractor_xy
        fig3a, ax3a = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        ax3a.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
        ax3a.plot(x_col, y_col, color="#ef4444", linewidth=0.9, alpha=0.85)
        ax3a.set_title("Final Attractor XY Projection", fontsize=11, fontweight="bold", pad=12)
        ax3a.set_xlabel("x")
        ax3a.set_ylabel("y")
        save_and_close(fig3a, fig_dir / "fig03a_final_attractor_xy")
        
        # fig03b_final_attractor_xz
        fig3b, ax3b = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        ax3b.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
        ax3b.plot(x_col, z_col, color="#ef4444", linewidth=0.9, alpha=0.85)
        ax3b.set_title("Final Attractor XZ Projection", fontsize=11, fontweight="bold", pad=12)
        ax3b.set_xlabel("x")
        ax3b.set_ylabel("z")
        save_and_close(fig3b, fig_dir / "fig03b_final_attractor_xz")
        
        # fig03c_final_attractor_yz
        fig3c, ax3c = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        ax3c.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
        ax3c.plot(y_col, z_col, color="#ef4444", linewidth=0.9, alpha=0.85)
        ax3c.set_title("Final Attractor YZ Projection", fontsize=11, fontweight="bold", pad=12)
        ax3c.set_xlabel("y")
        ax3c.set_ylabel("z")
        save_and_close(fig3c, fig_dir / "fig03c_final_attractor_yz")
        
        # fig03d_linear_vs_original_x, fig03e_linear_vs_original_y, fig03f_linear_vs_original_z, fig03g_linear_vs_original_3d
        try:
            recon = first_harmonic_reconstruction(traj_data)
            
            # Recortar rango de tiempo a los ultimos 50 segundos
            t_final = traj_data[-1, 0]
            mask_50 = traj_data[:, 0] >= (t_final - 50.0)
            traj_data_50 = traj_data[mask_50]
            recon_50 = recon[recon[:, 0] >= (t_final - 50.0)]
            
            # Sub-sample for plotting
            orig_small = downsample(traj_data_50, 4000)
            recon_small = downsample(recon_50, 1500)
            
            # fig03d_linear_vs_original_x
            fig3d, ax3d = plt.subplots(figsize=(8.0, 4.5), dpi=300)
            ax3d.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
            ax3d.plot(orig_small[:, 0], orig_small[:, 1], color="red", linewidth=1.2, alpha=0.85, label="Original")
            ax3d.plot(recon_small[:, 0], recon_small[:, 1], "--", color="purple", linewidth=1.4, alpha=0.9, label="Linearized")
            ax3d.set_title("Linear vs Original: $x(t)$ Component", fontsize=11, fontweight="bold", pad=12)
            ax3d.set_xlabel("Time $t$")
            ax3d.set_ylabel("$x$")
            ax3d.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
            save_and_close(fig3d, fig_dir / "fig03d_linear_vs_original_x")
            
            # fig03e_linear_vs_original_y
            fig3e, ax3e = plt.subplots(figsize=(8.0, 4.5), dpi=300)
            ax3e.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
            ax3e.plot(orig_small[:, 0], orig_small[:, 2], color="red", linewidth=1.2, alpha=0.85, label="Original")
            ax3e.plot(recon_small[:, 0], recon_small[:, 2], "--", color="purple", linewidth=1.4, alpha=0.9, label="Linearized")
            ax3e.set_title("Linear vs Original: $y(t)$ Component", fontsize=11, fontweight="bold", pad=12)
            ax3e.set_xlabel("Time $t$")
            ax3e.set_ylabel("$y$")
            ax3e.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
            save_and_close(fig3e, fig_dir / "fig03e_linear_vs_original_y")
            
            # fig03f_linear_vs_original_z
            fig3f, ax3f = plt.subplots(figsize=(8.0, 4.5), dpi=300)
            ax3f.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
            ax3f.plot(orig_small[:, 0], orig_small[:, 3], color="red", linewidth=1.2, alpha=0.85, label="Original")
            ax3f.plot(recon_small[:, 0], recon_small[:, 3], "--", color="purple", linewidth=1.4, alpha=0.9, label="Linearized")
            ax3f.set_title("Linear vs Original: $z(t)$ Component", fontsize=11, fontweight="bold", pad=12)
            ax3f.set_xlabel("Time $t$")
            ax3f.set_ylabel("$z$")
            ax3f.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
            save_and_close(fig3f, fig_dir / "fig03f_linear_vs_original_z")
            
            # fig03g_linear_vs_original_3d
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
            
        # -------------------------------------------------------------------------
        # 4. FIG11: Spectral FFT / PSD diagnostics
        # -------------------------------------------------------------------------
        try:
            # Slices tail of trajectory (last 75% of points) to avoid initial transients
            n_burn = int(len(traj_data) * 0.25)
            dt = traj_data[1, 0] - traj_data[0, 0]
            target_pt = 1.0 / (k * dt)
            
            tail_x = x_col[n_burn:]
            tail_y = y_col[n_burn:]
            tail_z = z_col[n_burn:]
            
            for component_name, tail_comp, fig_id in [("x", tail_x, "fig11a_fft_x"), ("y", tail_y, "fig11b_fft_y"), ("z", tail_z, "fig11c_fft_z")]:
                # Centered FFT
                centered_comp = tail_comp - tail_comp.mean()
                fft_vals = np.fft.rfft(centered_comp)
                fft_freqs = np.fft.rfftfreq(len(centered_comp), d=dt)
                fft_mag = np.abs(fft_vals) / len(centered_comp)
                
                # Convert frequency to rad/s
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
                
            # Welch PSD plots: fig11d_psd_x, fig11e_psd_y, fig11f_psd_z
            for component_name, tail_comp, fig_id in [("x", tail_x, "fig11d_psd_x"), ("y", tail_y, "fig11e_psd_y"), ("z", tail_z, "fig11f_psd_z")]:
                # Welch power spectral density
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
            
    # -------------------------------------------------------------------------
    # 5. FIG04: Matignon stability diagram / Reference section
    # -------------------------------------------------------------------------
    try:
        # Load equilibria dynamically
        # Since equilibria depend on m0, m1 etc., let's construct E0, E+, E-
        # For Chua saturation, the equilibria are at x = 0, and x = +/- (m1 - m0) / (m1 + 1)
        # E0 = [0, 0, 0]
        # E+ = [x_eq, y_eq, z_eq]
        m0 = system.m0
        m1 = system.m1
        beta = system.beta
        gamma = system.gamma
        
        equilibria = {"E0": np.array([0.0, 0.0, 0.0])}
        if m1 + 1.0 != 0.0:
            x_eq = (m1 - m0) / (m1 + 1.0)
            if x_eq > 1.0:
                y_eq = x_eq + (m0 - m1)
                z_eq = -beta / gamma * y_eq if gamma != 0.0 else 0.0
                # Let's verify and refine E+ and E-
                # Or we can classify actual equilibria
                y_eq_actual = (m1 - m0) / (system.m1 + 1.0) # wait, we can just use classify_equilibrium_stability
                
        # To be safe, let's load equilibria from the summary file or the actual system
        eq_names = ["E0", "E+", "E-"]
        # We can construct them using system properties
        eq_pts = {}
        # Let's find actual equilibria:
        # P * X + b * psi(r^T X) = 0
        # If r^T X = 0 -> E0 is [0, 0, 0]
        eq_pts["E0"] = np.array([0.0, 0.0, 0.0])
        
        # For saturation, outer equilibria: x > 1
        # x_eq = alpha * (y - x - m1 * x - (m0 - m1)) = 0 -> y = (m1 + 1)x + (m0 - m1)
        # y - y + z = 0 -> z = 0? Wait, x - y + z = 0 -> z = y - x
        # beta * y + gamma * z = 0 -> beta * y + gamma * (y - x) = 0 -> (beta + gamma) * y = gamma * x
        # y = gamma * x / (beta + gamma)
        # Substituting: gamma * x / (beta + gamma) = (m1 + 1) * x + (m0 - m1)
        # x * [ gamma / (beta + gamma) - (m1 + 1) ] = m0 - m1
        denom_eq = (gamma / (beta + gamma)) - (m1 + 1.0) if (beta + gamma) != 0.0 else -(m1 + 1.0)
        if denom_eq != 0.0:
            x_eq_outer = (m0 - m1) / denom_eq
            if x_eq_outer > 1.0:
                y_eq_outer = gamma * x_eq_outer / (beta + gamma) if (beta + gamma) != 0.0 else 0.0
                z_eq_outer = y_eq_outer - x_eq_outer
                eq_pts["E+"] = np.array([x_eq_outer, y_eq_outer, z_eq_outer])
                eq_pts["E-"] = np.array([-x_eq_outer, -y_eq_outer, -z_eq_outer])
                
        fig4 = plt.figure(figsize=(7.5, 6.5), dpi=300)
        ax4 = fig4.add_subplot(111)
        
        # Color background green (stable region)
        ax4.set_facecolor("#f0fdf4")
        
        # Limit boundary
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
        
        # Unstable region |arg(lambda)| <= q * pi / 2
        t_vals = np.linspace(-q * np.pi / 2.0, q * np.pi / 2.0, 400)
        x_fill = [0.0] + list(R * np.cos(t_vals)) + [0.0]
        y_fill = [0.0] + list(R * np.sin(t_vals)) + [0.0]
        ax4.fill(x_fill, y_fill, color="#fee2e2", alpha=0.9, edgecolor="#fca5a5", linewidth=0.8, label="Unstable Region")
        
        # Frontier rays
        ax4.plot([0.0, R * np.cos(q * np.pi / 2.0)], [0.0, R * np.sin(q * np.pi / 2.0)], color="#ef4444", linestyle="--", linewidth=1.1, label=r"Frontier $|\arg(\lambda)| = q\pi/2$")
        ax4.plot([0.0, R * np.cos(-q * np.pi / 2.0)], [0.0, R * np.sin(-q * np.pi / 2.0)], color="#ef4444", linestyle="--", linewidth=1.1)
        
        # Axes
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
        
    # -------------------------------------------------------------------------
    # 6. FIG05: Sphere probe tests / Destination checks (if available)
    # -------------------------------------------------------------------------
    # La visualizacion y zoom nativos de las pruebas de esferas (fig05b/fig05c) se realizan directamente bajo plot_trajectories y plot_sphere_tests
    pass
            
    # -------------------------------------------------------------------------
    # 7. FIG06 & FIG10: Basin slice overlay plots (if available)
    # -------------------------------------------------------------------------
    basin_csv_path = out_dir_path / "basin_results.csv"
    if basin_csv_path.exists():
        try:
            # basin columns: plane, equilibrium, u_val, v_val, classification_code
            with open(basin_csv_path, "r", newline="", encoding="utf-8") as bf:
                b_rows = list(csv.DictReader(bf))
                
            # Classify basin points by planes
            planes = {}
            for r in b_rows:
                plane = r.get("plane", "global")
                if plane not in planes:
                    planes[plane] = []
                planes[plane].append(r)
                
            for plane_name, p_data in planes.items():
                # Extract grid points
                u_vals = np.array([float(x["u_val"]) for x in p_data])
                v_vals = np.array([float(x["v_val"]) for x in p_data])
                codes = np.array([int(x["classification_code"]) for x in p_data])
                
                # Reshape into a 2D meshgrid grid for a beautiful filled contour
                # Find unique u and v values
                u_unique = np.unique(u_vals)
                v_unique = np.unique(v_vals)
                
                if len(u_unique) * len(v_unique) == len(codes):
                    # We can reshape into 2D matrices
                    U_mesh, V_mesh = np.meshgrid(u_unique, v_unique)
                    # Reshape code values (need to match grid coordinates)
                    # Match sorting order: u in columns, v in rows
                    code_mesh = np.zeros((len(v_unique), len(u_unique)))
                    for item in p_data:
                        col_idx = np.where(u_unique == float(item["u_val"]))[0][0]
                        row_idx = np.where(v_unique == float(item["v_val"]))[0][0]
                        code_mesh[row_idx, col_idx] = int(item["classification_code"])
                        
                    fig6, ax6 = plt.subplots(figsize=(7.5, 6.5), dpi=300)
                    
                    # Premium highly contrasting color-map requested: Pink, Purple, Gray, Yellow
                    from matplotlib.colors import ListedColormap
                    custom_colors = ["#ff66b2", "#8b5cf6", "#94a3b8", "#facc15", "#475569", "#3b82f6"]
                    cmap = ListedColormap(custom_colors[:max(3, len(np.unique(codes)))])
                    
                    # Use pixel-perfect pcolormesh instead of continuous contour polygons to prevent polygonal artifacts
                    mesh = ax6.pcolormesh(U_mesh, V_mesh, code_mesh, cmap=cmap, shading="nearest", alpha=0.92)
                    
                    # Add exact classification labels in a discrete legend instead of colorbar
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
                            
                    # Mark the equilibrium on the xy slice plane as requested
                    eq_selected = eff_config.get("basin", {}).get("equilibrium_selection", "E+")
                    if eq_selected in eq_pts:
                        eq_pt = eq_pts[eq_selected]
                        ax6.scatter([eq_pt[0]], [eq_pt[1]], color="red", marker="*", s=160, edgecolors="black", zorder=12, label=f"Equilibrium {eq_selected}")
                        
                    ax6.legend(handles=legend_patches, loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
                    
                    # Indicate fixed z level in title
                    fixed_z = eff_config.get("basin", {}).get("fixed_z", 0.0)
                    ax6.set_title(f"Basin Attraction Slice (Plane xy, z = {fixed_z:.2f})", fontsize=11, fontweight="bold", pad=12)
                    ax6.set_xlabel("u coordinate")
                    ax6.set_ylabel("v coordinate")
                    
                    # Naming mapping: fig06a_basin_overlay_z0 or fig06b_basin_overlay_zfinal
                    if plane_name == "xy" or plane_name == "xy_z0":
                        fig_id = "fig06a_basin_overlay_z0"
                    elif "zfinal" in plane_name or "z_final" in plane_name:
                        fig_id = "fig06b_basin_overlay_zfinal"
                    else:
                        fig_id = f"fig10_{plane_name}_basin_slice"
                        
                    save_and_close(fig6, fig_dir / fig_id)
                else:
                    # Fallback to scatter if not a perfect square grid
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
