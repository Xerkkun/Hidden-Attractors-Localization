import sys
from pathlib import Path
import numpy as np
import math
import csv
import json
import argparse
from types import SimpleNamespace

# Ensure version_2 is in python path
sys.path.insert(0, str(Path(r"c:\Users\moren\Desktop\Codes\Hidden Attractors Fractional Order\version_2")))

from hidden_attractors.models.chua import ChuaParameters
from hidden_attractors.seed_generation.chua_arctan_wu2023 import find_centered_arctan_wu2023_branches
from hidden_attractors.continuation.continuation_fractional import run_fractional_continuation
from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.systems.builtins import _chua_lure_system
from hidden_attractors.diagnostics.periodicity import classify_post_transient_periodicity

def main():
    parser = argparse.ArgumentParser(description="Sweep arctan Chua candidates with fractional DF and continuation.")
    parser.add_argument("--memory-mode", default="full", choices=["full", "window"], help="Caputo memory mode")
    parser.add_argument("--output-dir", type=str, default="outputs/arctan_fractional_search", help="Output folder")
    
    args = parser.parse_args()
    
    memory_mode = args.memory_mode
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    q_dynamics = 0.99
    h = 0.01
    t_final = 300.0
    t_transient = 100.0
    
    if memory_mode == "window":
        memory_window_length = int(10.0 / h)  # Lm = 10.0 seconds
    else:
        memory_window_length = None
        
    # Sweep grid parameters consistent with earlier sweep
    a1_values = [0.1, 0.2]
    a2_values = [-1.0, -1.2, -1.5585, -2.0, -2.5, -3.0]
    rho_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
    
    results = []
    
    print(f"Starting Arctan Sweep (q=0.99, memory_mode={memory_mode})...")
    print(f"Writing results to {output_dir}")
    
    case_idx = 0
    for a1 in a1_values:
        for a2 in a2_values:
            for rho in rho_values:
                params = ChuaParameters(
                    model="arctan",
                    alpha=8.4562,
                    beta=12.0732,
                    gamma=0.0052,
                    a1=a1,
                    a2=a2,
                    rho=rho
                )
                
                # 1. Fractional Describing function seed search
                try:
                    branches = find_centered_arctan_wu2023_branches(
                        q=q_dynamics,
                        params=params,
                        transfer_mode="fractional_spectral",
                        nscan=5000  # fast scan is fine
                    )
                except Exception as e:
                    print(f"Error scanning a1={a1}, a2={a2}, rho={rho}: {e}")
                    continue
                    
                for branch in branches:
                    case_idx += 1
                    case_id = f"a1_{a1:.2f}_a2_{a2:.4f}_rho_{rho:.2f}_branch_{branch.branch_index}"
                    case_id_clean = case_id.replace("-", "m").replace(".", "p")
                    
                    print(f"\n--- [{case_idx}] {case_id_clean}: omega0={branch.omega:.4f}, k={branch.gain:.4f} ---")
                    
                    try:
                        # Construct system namespace
                        system = SimpleNamespace(
                            system_id="chua_fractional_arctan",
                            name="chua_fractional_arctan",
                            parameters={
                                "model": "arctan",
                                "alpha": params.alpha,
                                "beta": params.beta,
                                "gamma": params.gamma,
                                "a1": params.a1,
                                "a2": params.a2,
                                "rho": params.rho
                            },
                            lure=_chua_lure_system({
                                "model": "arctan",
                                "alpha": params.alpha,
                                "beta": params.beta,
                                "gamma": params.gamma,
                                "a1": params.a1,
                                "a2": params.a2,
                                "rho": params.rho
                            })
                        )
                        
                        # 2. Run numerical continuation (ABM Caputo continuation)
                        eta_values = np.linspace(0.0, 1.0, 11)
                        print("  Running ABM continuation...")
                        steps = run_fractional_continuation(
                            system=system,
                            seed_x0=branch.seed,
                            k_gain=branch.gain,
                            lambda_values=eta_values,
                            h=h,
                            memory_mode=memory_mode,
                            memory_window_length=memory_window_length,
                            integrator="abm",
                            use_c_backend=True,
                            require_c_backend=False,
                            allow_python_fallback=True,
                            t_transient=30.0,
                            t_keep=30.0,
                            q=q_dynamics,
                        )
                        
                        final_step = steps[-1]
                        final_status = final_step["status"]
                        print(f"  Continuation status: {final_status}")
                        
                        if final_status == "ok":
                            x_final_seed = final_step["x_out"].copy()
                            
                            # 3. Final simulation with ABM
                            print("  Running final simulation with ABM...")
                            times, states, status, info = fractional_integrate(
                                rhs=lambda t, val: system.lure.matrix @ val + system.lure.input_vector * system.lure.nonlinearity(system.lure.output_vector @ val),
                                x0=x_final_seed,
                                q=q_dynamics,
                                h=h,
                                t_final=t_final,
                                method="abm",
                                memory_mode=memory_mode,
                                memory_window_length=memory_window_length,
                                system=system,
                                use_c_backend=True,
                                allow_python_fallback=True
                            )
                            
                            # 4. Diagnostics
                            trajectory_data = np.column_stack((times, states))
                            diag = classify_post_transient_periodicity(
                                trajectory_data,
                                h=h,
                                config={"t_transient": t_transient}
                            )
                            verdict = diag["candidate_label"]
                            print(f"  Diagnostics verdict: {verdict}")
                            
                            # Save trajectory if nonperiodic / chaotic
                            if verdict in ["chaotic_candidate_pending_robustness", "nonperiodic_candidate"]:
                                traj_file = output_dir / f"{case_id_clean}_trajectory.csv"
                                np.savetxt(traj_file, trajectory_data, delimiter=",", header="t,x,y,z", comments="")
                                print(f"  Saved trajectory to {traj_file.name}")
                                
                                # Save simple 3D phase space plot
                                import matplotlib.pyplot as plt
                                n_burn = int(t_transient / h)
                                tail = states[n_burn:]
                                fig = plt.figure(figsize=(8, 6))
                                ax = fig.add_subplot(111, projection="3d")
                                ax.plot(tail[:, 0], tail[:, 1], tail[:, 2], lw=0.5, color="red")
                                ax.set_title(f"{case_id_clean} | {verdict}")
                                ax.set_xlabel("x")
                                ax.set_ylabel("y")
                                ax.set_zlabel("z")
                                plot_file = output_dir / f"{case_id_clean}_phase3d.png"
                                plt.savefig(plot_file, dpi=150)
                                plt.close()
                                print(f"  Saved figure to {plot_file.name}")
                            
                            results.append({
                                "case_id": case_id_clean,
                                "a1": a1,
                                "a2": a2,
                                "rho": rho,
                                "omega0": branch.omega,
                                "k": branch.gain,
                                "A0": branch.amplitude,
                                "continuation_status": final_status,
                                "simulation_status": status,
                                "verdict": verdict,
                                "final_state": x_final_seed.tolist()
                            })
                        else:
                            results.append({
                                "case_id": case_id_clean,
                                "a1": a1,
                                "a2": a2,
                                "rho": rho,
                                "omega0": branch.omega,
                                "k": branch.gain,
                                "A0": branch.amplitude,
                                "continuation_status": final_status,
                                "simulation_status": "none",
                                "verdict": "continuation_failed",
                                "final_state": []
                            })
                            
                    except Exception as e:
                        print(f"  Error processing candidate: {e}")
                        results.append({
                            "case_id": case_id_clean,
                            "a1": a1,
                            "a2": a2,
                            "rho": rho,
                            "omega0": branch.omega,
                            "k": branch.gain,
                            "A0": branch.amplitude,
                            "continuation_status": "error",
                            "simulation_status": "error",
                            "verdict": f"error: {str(e)}",
                            "final_state": []
                        })
                        
    # Write summary CSV
    csv_file = output_dir / "summary.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "a1", "a2", "rho", "omega0", "k", "A0", "continuation_status", "simulation_status", "verdict", "final_state"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    print(f"\nDone! Summary written to {csv_file}")
    
if __name__ == "__main__":
    main()
