"""Lightweight workflow to compare parameter continuation modes: integer vs fractional.

Compares how parameter continuation (eta/lambda grid) behaves when the step-by-step
restart utilizes:
  1. Integer continuation (D^1 X = P0 X + eta*b*delta, standard restart)
  2. Fractional continuation (D^q X = P0 X + eta*b*delta, causal history restart)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.systems.registry import get_system_by_id
from src.lure.nyquist import find_harmonic_candidates
from src.lure.seeds import build_lure_seed
from src.continuation.continuation_integer import run_integer_continuation
from src.continuation.continuation_fractional import run_fractional_continuation

def run_compare_continuation_modes(system_id: str = "chua_fractional_saturation", output_dir: str = "outputs/compare_continuation_modes") -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Running compare_continuation_modes for system '{system_id}' ===")
    
    # Load system
    system = get_system_by_id(system_id)
    q = system.q
    
    # Find candidates
    candidates = find_harmonic_candidates(system, transfer_mode="fractional")
    if not candidates:
        print("No candidates found.")
        return
        
    A0, omega0, k = candidates[0]
    seed_pos, _ = build_lure_seed(system, A0, omega0, k, q=q, transfer_mode="fractional")
    
    eta_values = np.linspace(0.1, 1.0, 5)
    h = 0.01
    t_transient = 5.0
    t_keep = 5.0
    
    # 1. Integer Continuation
    print("Running integer-order continuation...")
    steps_int = run_integer_continuation(
        system=system,
        seed_x0=seed_pos,
        k_gain=k,
        lambda_values=eta_values,
        h=h,
        t_transient=t_transient,
        t_keep=t_keep,
        integrator="efork"
    )
    
    # 2. Fractional Continuation
    print("Running fractional-order continuation...")
    steps_frac = run_fractional_continuation(
        system=system,
        seed_x0=seed_pos,
        k_gain=k,
        lambda_values=eta_values,
        h=h,
        memory_mode="full",
        integrator="abm",
        t_transient=t_transient,
        t_keep=t_keep,
        use_c_backend=False
    )
    
    # Plotting comparison of eta profiles
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Extract end norms
    norms_int = [np.linalg.norm(s["x_out"]) for s in steps_int]
    norms_frac = [np.linalg.norm(s["x_out"]) for s in steps_frac]
    
    ax.plot(eta_values, norms_int, "o-", label="Integer Continuation (restart-state)", color="#ef4444", linewidth=2)
    ax.plot(eta_values, norms_frac, "s-", label="Fractional Continuation (abm-monolithic)", color="#3b82f6", linewidth=2)
    ax.set_xlabel("Continuation parameter eta")
    ax.set_ylabel("Final State Norm")
    ax.set_title("Continuation Path Norm Profile Comparison")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    
    plot_path = os.path.join(output_dir, "continuation_mode_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Comparison plot saved to: {plot_path}")

if __name__ == "__main__":
    run_compare_continuation_modes()
