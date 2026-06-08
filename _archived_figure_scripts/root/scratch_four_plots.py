import sys
from pathlib import Path
import json
import numpy as np
import math
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# Ensure version_2 is in system path
sys.path.insert(0, str(Path(r"c:\Users\moren\Desktop\Codes\Hidden Attractors Fractional Order\version_2")))

from hidden_attractors.integrations.fractional_c import fractional_integrate

# ---------------------------------------------------------------------------
# Generalized ADM (Adomian Decomposition Method) of order 4 for any rho
# ---------------------------------------------------------------------------

def _gamma_factors(q: float) -> dict[str, float]:
    return {
        "gq1":  math.gamma(q + 1.0),
        "g2q1": math.gamma(2.0 * q + 1.0),
        "g3q1": math.gamma(3.0 * q + 1.0),
        "g4q1": math.gamma(4.0 * q + 1.0),
        "g2q2": math.gamma(2.0 * (q + 1.0)),
        "g3q3": math.gamma(3.0 * (q + 1.0)),
    }

def _adm_general_step(
    x0: float, y0: float, z0: float,
    alpha: float, beta: float, gamma: float,
    a1: float, a2: float, rho: float,
    q: float, h: float, gf: dict[str, float],
) -> tuple[float, float, float]:
    s  = (rho * x0) ** 2 + 1.0
    s2 = s * s
    s3 = s2 * s

    # Derivatives of F(x) = arctan(rho * x) at x0
    A0 = math.atan(rho * x0)
    g0 = rho / s
    g1_raw = -2.0 * (rho ** 3) * x0 / s2
    g2_raw = 8.0 * (rho ** 5) * (x0 ** 2) / s3 - 2.0 * (rho ** 3) / s2

    # Adomian component C^1
    C1_1 = -alpha * (1.0 + a1) * x0 + alpha * y0 - alpha * a2 * A0
    C2_1 = x0 - y0 + z0
    C3_1 = -beta * y0 - gamma * z0

    # Adomian component C^2
    A1 = g0 * C1_1
    C1_2 = -alpha * (1.0 + a1) * C1_1 + alpha * C2_1 - alpha * a2 * A1
    C2_2 = C1_1 - C2_1 + C3_1
    C3_2 = -beta * C2_1 - gamma * C3_1

    # Adomian component C^3
    ratio_A2 = gf["g2q1"] / gf["g2q2"]
    A2 = g0 * C1_2 + 0.5 * g1_raw * C1_1 * C1_1 * ratio_A2
    C1_3 = -alpha * (1.0 + a1) * C1_2 + alpha * C2_2 - alpha * a2 * A2
    C2_3 = C1_2 - C2_2 + C3_2
    C3_3 = -beta * C2_2 - gamma * C3_2

    # Adomian component C^4
    ratio_A3a = gf["g3q1"] / (gf["gq1"] * gf["g2q1"])
    ratio_A3b = gf["g3q1"] / gf["g3q3"]
    A3 = (g0 * C1_3
          + g1_raw * C1_1 * C1_2 * ratio_A3a
          + (1.0 / 6.0) * g2_raw * (C1_1 ** 3) * ratio_A3b)
    C1_4 = -alpha * (1.0 + a1) * C1_3 + alpha * C2_3 - alpha * a2 * A3
    C2_4 = C1_3 - C2_3 + C3_3
    C3_4 = -beta * C2_3 - gamma * C3_3

    # Series sum
    hq  = h ** q
    h2q = hq * hq
    h3q = h2q * hq
    h4q = h3q * hq

    x1 = (x0
          + C1_1 * hq  / gf["gq1"]
          + C1_2 * h2q / gf["g2q1"]
          + C1_3 * h3q / gf["g3q1"]
          + C1_4 * h4q / gf["g4q1"])

    y1 = (y0
          + C2_1 * hq  / gf["gq1"]
          + C2_2 * h2q / gf["g2q1"]
          + C2_3 * h3q / gf["g3q1"]
          + C2_4 * h4q / gf["g4q1"])

    z1 = (z0
          + C3_1 * hq  / gf["gq1"]
          + C3_2 * h2q / gf["g2q1"]
          + C3_3 * h3q / gf["g3q1"]
          + C3_4 * h4q / gf["g4q1"])

    return x1, y1, z1

def adm_general_integrate(
    alpha, beta, gamma, a1, a2, rho,
    x0: np.ndarray, q: float, h: float, t_final: float,
    divergence_norm: float = 120.0
) -> tuple[np.ndarray, np.ndarray, str]:
    N = int(np.ceil(t_final / h))
    gf = _gamma_factors(q)
    
    times = np.zeros(N + 1, dtype=float)
    states = np.zeros((N + 1, 3), dtype=float)
    
    times[0] = 0.0
    states[0] = x0
    
    cx, cy, cz = float(x0[0]), float(x0[1]), float(x0[2])
    status = "ok"
    last_n = 0
    
    for idx in range(N):
        nx, ny, nz = _adm_general_step(
            cx, cy, cz,
            alpha, beta, gamma, a1, a2, rho,
            q, h, gf
        )
        if not (math.isfinite(nx) and math.isfinite(ny) and math.isfinite(nz)):
            status = "nonfinite"
            break
            
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm > divergence_norm:
            status = "diverged"
            times[idx + 1] = (idx + 1) * h
            states[idx + 1] = [nx, ny, nz]
            last_n = idx + 1
            break
            
        times[idx + 1] = (idx + 1) * h
        states[idx + 1] = [nx, ny, nz]
        cx, cy, cz = nx, ny, nz
        last_n = idx + 1
        
    return times[:last_n + 1], states[:last_n + 1], status

# ---------------------------------------------------------------------------
# Main process script
# ---------------------------------------------------------------------------

def run_four_plots():
    workspace_dir = Path(r"c:\Users\moren\Desktop\Codes\Hidden Attractors Fractional Order")
    run_dir = workspace_dir / "version_2" / "outputs" / "arctan_full_memory_search" / "run_20260605_194127"
    summary_path = run_dir / "summary.json"
    
    if not summary_path.exists():
        print(f"Summary path {summary_path} does not exist.")
        return
        
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    candidates = data.get("candidates", [])
    
    # Selected cases representing the top confirmed chaotic branches in branch_0
    target_cases = {
        "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m1_rho_1p5_branch_0",
        "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m1_rho_1p25_branch_0",
        "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m1p2_rho_1p5_branch_0",
        "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m1p2_rho_1p25_branch_0",
        "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m1p5585_rho_1_branch_0",
        "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m2_rho_0p75_branch_0",
        "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m2p5_rho_0p5_branch_0",
        "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p1_a2_m3_rho_0p5_branch_0",
        "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p2_a2_m0p8_rho_2_branch_0",
        "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p2_a2_m1_rho_1p5_branch_0",
        "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p2_a2_m1p2_rho_1p25_branch_0",
        "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p2_a2_m1p5585_rho_1_branch_0",
        "alpha_8p4562_beta_12p0732_gamma_0p0052_a1_0p2_a2_m2_rho_0p75_branch_0",
    }
    
    selected_cands = [c for c in candidates if c.get("case_id") in target_cases]
    print(f"Loaded {len(selected_cands)} targeted chaotic candidates for comparison.")
    
    ext_output_dir = run_dir / "extended_analysis"
    ext_output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = ext_output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = ext_output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    
    t_final = 500.0
    t_burn = 100.0
    Lm = 40.0 # memory window length in seconds
    
    for rank, cand in enumerate(selected_cands, 1):
        case_id = cand.get("case_id")
        print(f"\nProcessing Case {rank}: {case_id}")
        q = cand.get("q", 0.99)
        h = cand.get("h", 0.005)
        
        # Load parameters
        alpha = cand.get("alpha")
        beta = cand.get("beta")
        gamma = cand.get("gamma")
        a1 = cand.get("a1")
        a2 = cand.get("a2")
        rho = cand.get("rho")
        
        # Resolve original trajectory file path to get x0
        orig_traj_rel = cand.get("trajectory")
        orig_traj_path = workspace_dir / "version_2" / orig_traj_rel
        
        if not orig_traj_path.exists():
            orig_traj_path = run_dir / "trajectories" / f"{case_id}_final.csv"
            
        if not orig_traj_path.exists():
            print(f"Skipping {case_id} because source trajectory file is missing.")
            continue
            
        try:
            # Read CSV trajectory data to get the starting state x0
            traj_data = np.loadtxt(orig_traj_path, delimiter=",", skiprows=1)
            x0 = traj_data[-1, 1:4].copy()
            n_burn = int(t_burn / h)
            
            # Direct RHS function for ctypes C solver
            def fast_rhs(t, val):
                xv, yv, zv = val[0], val[1], val[2]
                f1 = alpha * (yv - (1.0 + a1) * xv - a2 * math.atan(rho * xv))
                f2 = xv - yv + zv
                f3 = -beta * yv - gamma * zv
                return np.array([f1, f2, f3], dtype=np.float64)
            
            # --- 1. ADM integration (order 4, generalized) ---
            print("  Integrating ADM...")
            t_adm, x_adm, status_adm = adm_general_integrate(
                alpha=alpha, beta=beta, gamma=gamma, a1=a1, a2=a2, rho=rho,
                x0=x0, q=q, h=h, t_final=t_final
            )
            
            # --- 2. ABM full-history Caputo integration in C ---
            print("  Integrating ABM Full History...")
            t_abm, x_abm, status_abm, _ = fractional_integrate(
                rhs=fast_rhs, x0=x0, q=q, h=h, t_final=t_final,
                method="abm", memory_mode="full", use_c_backend=True
            )
            
            # --- 3. EFORK Truncated memory Caputo integration in C ---
            win_steps = int(round(Lm / h))
            print(f"  Integrating EFORK Truncated (Lm={Lm}s, steps={win_steps})...")
            t_eft, x_eft, status_eft, _ = fractional_integrate(
                rhs=fast_rhs, x0=x0, q=q, h=h, t_final=t_final,
                method="efork", memory_mode="window", memory_window_length=win_steps, use_c_backend=True
            )
            
            # Save EFORK Trunc trajectory
            if status_eft == "ok":
                np.savetxt(
                    trajectories_dir / f"{case_id}_efork_trunc_extended.csv",
                    np.column_stack((t_eft, x_eft)),
                    delimiter=",",
                    header="t,x,y,z",
                    comments=""
                )
            
            # --- 4. EFORK Full memory Caputo integration in C ---
            print("  Integrating EFORK Full History...")
            t_eff, x_eff, status_eff, _ = fractional_integrate(
                rhs=fast_rhs, x0=x0, q=q, h=h, t_final=t_final,
                method="efork", memory_mode="full", use_c_backend=True
            )
            
            # Save EFORK Full trajectory
            if status_eff == "ok":
                np.savetxt(
                    trajectories_dir / f"{case_id}_efork_full_extended.csv",
                    np.column_stack((t_eff, x_eff)),
                    delimiter=",",
                    header="t,x,y,z",
                    comments=""
                )
            
            # --- 5. Generate 2x2 plot grid ---
            print("  Generating plot...")
            fig = plt.figure(figsize=(12, 11))
            
            # Helper to set up subplot style
            def setup_ax(ax, title):
                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.set_xlabel("x", fontsize=8)
                ax.set_ylabel("y", fontsize=8)
                ax.set_zlabel("z", fontsize=8)
                ax.tick_params(labelsize=7)
                
            # ADM plot
            ax1 = fig.add_subplot(221, projection="3d")
            if status_adm == "ok" and len(x_adm) > n_burn:
                ax1.plot(x_adm[n_burn:, 0], x_adm[n_burn:, 1], x_adm[n_burn:, 2], linewidth=0.45, color='crimson')
                setup_ax(ax1, f"1. ADM 4th Order (ok, N={len(x_adm)})")
            else:
                ax1.text(0.5, 0.5, 0.5, f"ADM Failed/Diverged\nStatus: {status_adm}", ha='center')
                setup_ax(ax1, "1. ADM 4th Order")
                
            # ABM plot
            ax2 = fig.add_subplot(222, projection="3d")
            if status_abm == "ok" and len(x_abm) > n_burn:
                ax2.plot(x_abm[n_burn:, 0], x_abm[n_burn:, 1], x_abm[n_burn:, 2], linewidth=0.45, color='royalblue')
                setup_ax(ax2, f"2. ABM Full History (ok, N={len(x_abm)})")
            else:
                ax2.text(0.5, 0.5, 0.5, f"ABM Failed/Diverged\nStatus: {status_abm}", ha='center')
                setup_ax(ax2, "2. ABM Full History")
                
            # EFORK Truncated plot
            ax3 = fig.add_subplot(223, projection="3d")
            if status_eft == "ok" and len(x_eft) > n_burn:
                ax3.plot(x_eft[n_burn:, 0], x_eft[n_burn:, 1], x_eft[n_burn:, 2], linewidth=0.45, color='darkgreen')
                setup_ax(ax3, f"3. EFORK Truncated (Lm={Lm}s, N={len(x_eft)})")
            else:
                ax3.text(0.5, 0.5, 0.5, f"EFORK Truncated Failed/Diverged\nStatus: {status_eft}", ha='center')
                setup_ax(ax3, f"3. EFORK Truncated (Lm={Lm}s)")
                
            # EFORK Full plot
            ax4 = fig.add_subplot(224, projection="3d")
            if status_eff == "ok" and len(x_eff) > n_burn:
                ax4.plot(x_eff[n_burn:, 0], x_eff[n_burn:, 1], x_eff[n_burn:, 2], linewidth=0.45, color='darkorange')
                setup_ax(ax4, f"4. EFORK Full History (ok, N={len(x_eff)})")
            else:
                ax4.text(0.5, 0.5, 0.5, f"EFORK Full Failed/Diverged\nStatus: {status_eff}", ha='center')
                setup_ax(ax4, "4. EFORK Full History")
                
            fig.suptitle(f"Chua-Arctan Solver Comparison (q={q}, t_final={t_final})\nCase: {case_id}", fontsize=12, fontweight='bold')
            fig.tight_layout()
            
            output_plot_path = figures_dir / f"{case_id}_four_solvers_phase3d.png"
            fig.savefig(output_plot_path, dpi=200)
            plt.close(fig)
            print(f"  Successfully saved comparison plot to: {output_plot_path.name}")
            
        except Exception as e:
            print(f"  ERROR processing candidate {case_id}: {e}")

if __name__ == "__main__":
    run_four_plots()
