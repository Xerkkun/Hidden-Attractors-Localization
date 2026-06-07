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
from hidden_attractors.seed_generation.chua import find_omega_gain_candidates, solve_amplitude_from_gain, build_fractional_seed
from hidden_attractors.continuation.continuation_fractional import run_fractional_continuation
from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.systems.builtins import _chua_lure_system
from hidden_attractors.diagnostics.periodicity import classify_post_transient_periodicity

def main():
    parser = argparse.ArgumentParser(description="Search non-smooth Chua fractional candidates.")
    parser.add_argument("--seed-q", type=float, default=1.0, help="DF seed q (1.0 or 0.9998)")
    parser.add_argument("--memory-mode", default="full", choices=["full", "window"], help="Caputo memory mode")
    parser.add_argument("--output-dir", type=str, default="outputs/saturation_search_default", help="Output folder")
    parser.add_argument("--sweep", action="store_true", help="Perform the full exploration sweep over m1 and m0")
    
    args = parser.parse_args()
    
    seed_q = args.seed_q
    memory_mode = args.memory_mode
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    q_dynamics = 0.9998
    h = 0.01
    t_final = 300.0
    t_transient = 100.0
    
    if memory_mode == "window":
        memory_window_length = int(10.0 / h)  # Lm = 10.0 seconds
    else:
        memory_window_length = None
        
    # Define sweep grid
    if args.sweep:
        m1_values = [-0.8, -1.0, -1.2, -1.4, -1.6]
        m0_values = [-0.1, -0.2, -0.3, -0.4]
    else:
        # Just standard parameters
        m1_values = [-1.1468]
        m0_values = [-0.1768]
        
    results = []
    
    print(f"Starting sweep (seed_q={seed_q}, memory_mode={memory_mode}, sweep={args.sweep})...")
    print(f"Writing results to {output_dir}")
    
    case_idx = 0
    for m1 in m1_values:
        for m0 in m0_values:
            params = ChuaParameters(
                model="nonsmooth",
                alpha=8.4562,
                beta=12.0732,
                gamma=0.0052,
                m0=m0,
                m1=m1
            )
            
            # 1. Describing function seed search
            try:
                candidates = find_omega_gain_candidates(q=seed_q, params=params)
            except Exception as e:
                print(f"Error scanning m1={m1}, m0={m0}: {e}")
                continue
                
            for branch_idx, (omega0, k) in enumerate(candidates):
                case_idx += 1
                case_id = f"m1_{m1:.4f}_m0_{m0:.4f}_branch_{branch_idx}"
                case_id_clean = case_id.replace("-", "m").replace(".", "p")
                
                print(f"\n--- [{case_idx}] {case_id_clean}: omega0={omega0:.4f}, k={k:.4f} ---")
                
                try:
                    A0 = solve_amplitude_from_gain(k, params)
                    seed_x0, vector, matched_ev = build_fractional_seed(seed_q, params, omega0, k, A0)
                    
                    # Construct system namespace
                    system = SimpleNamespace(
                        system_id="chua_fractional_saturation",
                        name="chua_fractional_saturation",
                        parameters={
                            "model": "nonsmooth",
                            "alpha": params.alpha,
                            "beta": params.beta,
                            "gamma": params.gamma,
                            "m0": params.m0,
                            "m1": params.m1
                        },
                        lure=_chua_lure_system({
                            "model": "nonsmooth",
                            "alpha": params.alpha,
                            "beta": params.beta,
                            "gamma": params.gamma,
                            "m0": params.m0,
                            "m1": params.m1
                        })
                    )
                    
                    # 2. Run numerical continuation (ABM Caputo continuation)
                    eta_values = np.linspace(0.0, 1.0, 11)
                    print("  Running ABM continuation...")
                    steps = run_fractional_continuation(
                        system=system,
                        seed_x0=seed_x0,
                        k_gain=k,
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
                            "m1": m1,
                            "m0": m0,
                            "omega0": omega0,
                            "k": k,
                            "A0": A0,
                            "continuation_status": final_status,
                            "simulation_status": status,
                            "verdict": verdict,
                            "final_state": x_final_seed.tolist()
                        })
                    else:
                        results.append({
                            "case_id": case_id_clean,
                            "m1": m1,
                            "m0": m0,
                            "omega0": omega0,
                            "k": k,
                            "A0": A0,
                            "continuation_status": final_status,
                            "simulation_status": "none",
                            "verdict": "continuation_failed",
                            "final_state": []
                        })
                        
                except Exception as e:
                    print(f"  Error processing candidate: {e}")
                    results.append({
                        "case_id": case_id_clean,
                        "m1": m1,
                        "m0": m0,
                        "omega0": omega0,
                        "k": k,
                        "A0": 0.0,
                        "continuation_status": "error",
                        "simulation_status": "error",
                        "verdict": f"error: {str(e)}",
                        "final_state": []
                    })
                    
    # Write summary CSV
    csv_file = output_dir / "summary.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "m1", "m0", "omega0", "k", "A0", "continuation_status", "simulation_status", "verdict", "final_state"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    print(f"\nDone! Summary written to {csv_file}")
    
if __name__ == "__main__":
    main()
