import sys
from pathlib import Path
import numpy as np
import math

# Ensure version_2 is in python path
sys.path.insert(0, str(Path(r"c:\Users\moren\Desktop\Codes\Hidden Attractors Fractional Order\version_2")))

from hidden_attractors.models.chua import ChuaParameters
from hidden_attractors.seed_generation.chua import find_omega_gain_candidates, solve_amplitude_from_gain, build_fractional_seed
from hidden_attractors.continuation.continuation_integer import run_integer_continuation
from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.systems.builtins import _chua_lure_system
from types import SimpleNamespace
from hidden_attractors.diagnostics.periodicity import classify_post_transient_periodicity

def main():
    params = ChuaParameters(
        model="nonsmooth",
        alpha=8.4562,
        beta=12.0732,
        gamma=0.0052,
        m0=-0.1768,
        m1=-1.1468
    )
    
    q_seed = 1.0  # Seed mode: integer continuation
    q_dynamics = 0.9998
    
    print("Finding candidates...")
    candidates = find_omega_gain_candidates(q=q_seed, params=params)
    print(f"Located candidates for q_seed=1.0: {candidates}")
    
    system = SimpleNamespace(parameters={"model": "nonsmooth", "alpha": 8.4562, "beta": 12.0732, "gamma": 0.0052, "m0": -0.1768, "m1": -1.1468}, lure=_chua_lure_system({"model": "nonsmooth", "alpha": 8.4562, "beta": 12.0732, "gamma": 0.0052, "m0": -0.1768, "m1": -1.1468}))
    
    for idx, (omega0, k) in enumerate(candidates):
        print(f"\n--- Testing Candidate {idx+1}: omega0={omega0:.4f}, k={k:.4f} ---")
        try:
            A0 = solve_amplitude_from_gain(k, params)
            print(f"  Solved amplitude: {A0:.4f}")
            
            # Build seed initial condition
            seed_x0, vector, matched_ev = build_fractional_seed(q_seed, params, omega0, k, A0)
            print(f"  Seed x0: {seed_x0}")
            
            # Run integer continuation
            eta_values = np.linspace(0.0, 1.0, 21)
            print("  Running integer continuation...")
            cont_steps = run_integer_continuation(
                system=system,
                seed_x0=seed_x0,
                k_gain=k,
                lambda_values=eta_values,
                h=0.01,
                t_transient=30.0,
                t_keep=30.0,
                div_threshold=120.0,
                integrator="heun"
            )
            
            final_status = cont_steps[-1]["status"]
            print(f"  Continuation final status: {final_status}")
            
            if final_status == "ok":
                x_final_seed = cont_steps[-1]["x_out"].copy()
                print(f"  Final state after continuation: {x_final_seed}")
                
                # Final integration with ABM (full history)
                print("  Running final simulation with ABM...")
                times, states, status, info = fractional_integrate(
                    rhs=lambda t, val: system.lure.matrix @ val + system.lure.input_vector * system.lure.nonlinearity(system.lure.output_vector @ val),
                    x0=x_final_seed,
                    q=q_dynamics,
                    h=0.01,
                    t_final=300.0,
                    method="abm",
                    memory_mode="full",
                    use_c_backend=True,
                    allow_python_fallback=True
                )
                
                print(f"  Final integration status: {status}, steps: {len(times)}")
                
                # Run diagnostics
                trajectory_data = np.column_stack((times, states))
                diag = classify_post_transient_periodicity(
                    trajectory_data,
                    h=0.01,
                    config={"t_transient": 100.0}
                )
                print(f"  Diagnostics verdict: {diag['candidate_label']}")
                
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
