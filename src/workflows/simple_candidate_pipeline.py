"""Lightweight simple candidate pipeline workflow.

Executes a minimal, fast sequence:
  Seed -> Continuation (optional) -> Simulation -> Light classification
This avoids all heavy basin slices, robust/sphere sweeps, and Matignon stability plots.
"""

import os
import numpy as np
from src.systems.registry import get_system_by_id
from src.lure.nyquist import find_harmonic_candidates
from src.lure.seeds import build_lure_seed
from src.continuation.continuation_fractional import run_fractional_continuation
from src.integrators.general import integrate_general
from src.verification.equilibria import solve_equilibria

def run_simple_candidate_pipeline(system_id: str = "chua_fractional_saturation", run_continuation: bool = True) -> dict:
    print(f"=== Running simple candidate pipeline for system '{system_id}' ===")
    
    # 1. Load system
    system = get_system_by_id(system_id)
    q = system.q
    
    # Solve equilibria
    equilibria = list(solve_equilibria(system).values())
    
    # 2. Find Candidates
    candidates = find_harmonic_candidates(system, transfer_mode="fractional")
    if not candidates:
        print("No candidates found.")
        return {"status": "no_candidates"}
        
    A0, omega0, k = candidates[0]
    print(f"Primary candidate found: A0={A0:.4f}, omega0={omega0:.4f}, k={k:.4f}")
    
    # 3. Build Seed
    seed_pos, _ = build_lure_seed(system, A0, omega0, k, q=q, transfer_mode="fractional")
    
    # 4. Parameter Continuation (optional)
    if run_continuation:
        print("Running parameter continuation stage...")
        # Brief 3-stage continuation
        eta_grid = [0.2, 0.6, 1.0]
        steps = run_fractional_continuation(
            system=system,
            seed_x0=seed_pos,
            k_gain=k,
            lambda_values=eta_grid,
            h=0.01,
            memory_mode="full",
            integrator="abm",
            t_transient=2.0,
            t_keep=2.0,
            use_c_backend=False
        )
        if steps and steps[-1]["status"] == "ok":
            sim_seed = steps[-1]["x_out"]
            print("Continuation successfully completed.")
        else:
            print("Continuation failed or early stopped. Reverting to original seed.")
            sim_seed = seed_pos
    else:
        sim_seed = seed_pos
        
    # 5. Final Dynamics Simulation
    h = 0.01
    t_final = 20.0
    print(f"Integrating final dynamics (t={t_final}s, h={h})...")
    t_arr, x_arr, status = integrate_general(
        rhs=system.evaluate_rhs, x0=sim_seed, q=q, h=h, t_final=t_final, integrator="efork3"
    )
    
    # 6. Lightweight Classification
    final_state = x_arr[-1]
    final_norm = np.linalg.norm(final_state)
    
    # Checks
    if not np.isfinite(final_norm) or status == "nonfinite_solution":
        verdict = "nonfinite_diverged"
    elif final_norm > 100.0 or status == "diverged":
        verdict = "unbounded_divergence"
    else:
        # Check if it converged to any equilibrium
        converged_to_eq = False
        closest_eq_dist = 1e9
        for eq in equilibria:
            dist = np.linalg.norm(final_state - eq)
            if dist < closest_eq_dist:
                closest_eq_dist = dist
            if dist < 0.01:
                converged_to_eq = True
                break
        
        if converged_to_eq:
            verdict = f"converged_to_equilibrium (dist={closest_eq_dist:.6f})"
        else:
            # Bounded oscillation or complex attractor
            verdict = "bounded_attractor_oscillation"
            
    print(f"Lightweight Classification Verdict: {verdict}")
    
    return {
        "status": "success",
        "system_id": system_id,
        "seed": seed_pos.tolist(),
        "final_state": final_state.tolist(),
        "final_norm": float(final_norm),
        "verdict": verdict
    }

if __name__ == "__main__":
    run_simple_candidate_pipeline()
