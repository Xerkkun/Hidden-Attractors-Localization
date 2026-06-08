import sys
from pathlib import Path
import json
import numpy as np
import math

# Ensure version_2 is in system path
sys.path.insert(0, str(Path(r"c:\Users\moren\Desktop\Codes\Hidden Attractors Fractional Order\version_2")))

from hidden_attractors.native.backends import GeneralFDEBackend
from hidden_attractors.systems import get_system
from hidden_attractors.diagnostics.periodicity import classify_post_transient_periodicity

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
    """Single step of ADM 4th-order for generic rho scaling factor."""
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
    """Integrate the arctan Chua system using the generalized 4th-order ADM solver."""
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
        try:
            nx, ny, nz = _adm_general_step(
                cx, cy, cz,
                alpha, beta, gamma, a1, a2, rho,
                q, h, gf
            )
        except Exception as e:
            status = f"solver_exception:{e}"
            break
            
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
# Main analysis script
# ---------------------------------------------------------------------------

def run_extensions():
    workspace_dir = Path(r"c:\Users\moren\Desktop\Codes\Hidden Attractors Fractional Order")
    run_dir = workspace_dir / "version_2" / "outputs" / "arctan_full_memory_search" / "run_20260605_194127"
    summary_path = run_dir / "summary.json"
    
    if not summary_path.exists():
        print(f"Summary path {summary_path} does not exist.")
        return
        
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    candidates = data.get("candidates", [])
    print(f"Loaded {len(candidates)} candidates from summary.")
    
    # Sort candidates by score descending
    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Build GeneralFDEBackend
    try:
        fde_backend = GeneralFDEBackend.build()
        print("GeneralFDEBackend C library built and loaded successfully.\n")
    except Exception as e:
        print(f"Failed to build GeneralFDEBackend: {e}")
        return

    # Prepare report header
    report_lines = []
    report_lines.append("=====================================================================================")
    report_lines.append("           EXTENDED RUNS COMPARISON REPORT: ADOMIAN ADM VS FULL-HISTORY ABM           ")
    report_lines.append("=====================================================================================")
    report_lines.append(f"Source Run: {run_dir.name}")
    report_lines.append(f"Extended Time: t_final = 500.0, t_burn = 100.0, h = 0.005")
    report_lines.append("=====================================================================================\n")
    
    header = f"{'Rank':<4} | {'Case ID':<35} | {'OrigSc':<6} | {'ADM Stat':<8} | {'ADM Label':<22} | {'ABM Stat':<8} | {'ABM Label':<22}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    
    print(header)
    print("-" * len(header))
    
    # Prepare folders for extended trajectories & plots
    ext_output_dir = run_dir / "extended_analysis"
    ext_output_dir.mkdir(parents=True, exist_ok=True)
    (ext_output_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    (ext_output_dir / "figures").mkdir(parents=True, exist_ok=True)
    
    for rank, cand in enumerate(candidates, 1):
        case_id = cand.get("case_id")
        orig_score = cand.get("score")
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
            print(f"{rank:<4} | {case_id[:35]:<35} | {orig_score:.4f} | Missing  | N/A                    | Missing  | N/A")
            report_lines.append(f"{rank:<4} | {case_id[:35]:<35} | {orig_score:.4f} | Missing  | N/A                    | Missing  | N/A")
            continue
            
        try:
            # Read CSV trajectory data to get the starting state x0
            traj_data = np.loadtxt(orig_traj_path, delimiter=",", skiprows=1)
            x_orig = traj_data[:, 1:4]
            x0 = x_orig[-1].copy()  # last state from continuation
            
            # --- 1. Adomian ADM integration ---
            t_final = 500.0
            t_burn = 100.0
            
            t_adm, x_adm, status_adm = adm_general_integrate(
                alpha=alpha, beta=beta, gamma=gamma, a1=a1, a2=a2, rho=rho,
                x0=x0, q=q, h=h, t_final=t_final
            )
            
            # Classify ADM dynamics
            label_adm = "too_short"
            n_burn = int(t_burn / h)
            if status_adm == "ok" and len(t_adm) > n_burn + 10:
                shifted_traj_adm = np.column_stack((t_adm[n_burn:] - t_adm[n_burn], x_adm[n_burn:]))
                periodicity_adm = classify_post_transient_periodicity(
                    shifted_traj_adm,
                    h=h,
                    config={
                        "t_transient": 0.0,
                        "require_two_components": True,
                        "entropy_min": 0.25,
                        "dominant_ratio_max": 0.65,
                        "relaxed_dominant_ratio": 0.45,
                        "freq_drift_max": 0.05,
                        "min_range": 0.01,
                        "divergence_norm": 120.0,
                    }
                )
                label_adm = periodicity_adm.get("candidate_label")
                
                # Save ADM trajectory
                np.savetxt(
                    ext_output_dir / "trajectories" / f"{case_id}_adm_extended.csv",
                    np.column_stack((t_adm, x_adm)),
                    delimiter=",",
                    header="t,x,y,z",
                    comments=""
                )
            
            # --- 2. Full-history ABM integration in C ---
            system = get_system("chua-arctan")
            import dataclasses
            parameters = {
                "model": "arctan",
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "m0": 0.0,
                "m1": 0.0,
                "a1": a1,
                "a2": a2,
                "rho": rho
            }
            system = dataclasses.replace(system, parameters=parameters)
            
            t_abm, x_abm, status_abm = fde_backend.integrate(
                rhs=lambda t, val: system.evaluate(val),
                x0=x0,
                q=q,
                h=h,
                t_final=t_final,
                divergence_norm=120.0,
                integrator="abm"
            )
            
            # Classify ABM dynamics
            label_abm = "too_short"
            if status_abm == "ok" and len(t_abm) > n_burn + 10:
                shifted_traj_abm = np.column_stack((t_abm[n_burn:] - t_abm[n_burn], x_abm[n_burn:]))
                periodicity_abm = classify_post_transient_periodicity(
                    shifted_traj_abm,
                    h=h,
                    config={
                        "t_transient": 0.0,
                        "require_two_components": True,
                        "entropy_min": 0.25,
                        "dominant_ratio_max": 0.65,
                        "relaxed_dominant_ratio": 0.45,
                        "freq_drift_max": 0.05,
                        "min_range": 0.01,
                        "divergence_norm": 120.0,
                    }
                )
                label_abm = periodicity_abm.get("candidate_label")
                
                # Save ABM trajectory
                np.savetxt(
                    ext_output_dir / "trajectories" / f"{case_id}_abm_extended.csv",
                    np.column_stack((t_abm, x_abm)),
                    delimiter=",",
                    header="t,x,y,z",
                    comments=""
                )
            
            # Print and record comparison row
            print(f"{rank:<4} | {case_id[:35]:<35} | {orig_score:.4f} | {status_adm:<8} | {label_adm:<22} | {status_abm:<8} | {label_abm:<22}")
            report_lines.append(f"{rank:<4} | {case_id[:35]:<35} | {orig_score:.4f} | {status_adm:<8} | {label_adm:<22} | {status_abm:<8} | {label_abm:<22}")
            
            # --- 3. Plot Phase Space comparison if it looks chaotic ---
            # Save 3D plots for both ADM and ABM to compare visually
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            
            fig = plt.figure(figsize=(12, 5.5))
            
            # ADM plot
            ax1 = fig.add_subplot(121, projection="3d")
            if status_adm == "ok":
                plot_adm = x_adm[n_burn:]
                ax1.plot(plot_adm[:, 0], plot_adm[:, 1], plot_adm[:, 2], linewidth=0.45, color='crimson')
            ax1.set_title(f"ADM Extended: {label_adm}\n(N={len(t_adm)})", fontsize=8)
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_zlabel("z")
            
            # ABM plot
            ax2 = fig.add_subplot(122, projection="3d")
            if status_abm == "ok":
                plot_abm = x_abm[n_burn:]
                ax2.plot(plot_abm[:, 0], plot_abm[:, 1], plot_abm[:, 2], linewidth=0.45, color='royalblue')
            ax2.set_title(f"ABM Extended (Caputo): {label_abm}\n(N={len(t_abm)})", fontsize=8)
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.set_zlabel("z")
            
            fig.suptitle(f"Case {rank}: {case_id[:45]}...", fontsize=9)
            fig.tight_layout()
            fig.savefig(ext_output_dir / "figures" / f"{case_id}_comparison_phase3d.png", dpi=200)
            plt.close(fig)
            
        except Exception as e:
            status_str = "Error"
            label_str = str(e)[:20]
            print(f"{rank:<4} | {case_id[:35]:<35} | {orig_score:.4f} | {status_str:<8} | {label_str:<22} | {status_str:<8} | {label_str:<22}")
            report_lines.append(f"{rank:<4} | {case_id[:35]:<35} | {orig_score:.4f} | {status_str:<8} | {label_str:<22} | {status_str:<8} | {label_str:<22}")

    # Write report file
    report_text = "\n".join(report_lines)
    report_file = workspace_dir / "extended_runs_report.txt"
    report_file.write_text(report_text, encoding="utf-8")
    print(f"\nExtended runs report written to: {report_file}")

if __name__ == "__main__":
    run_extensions()
