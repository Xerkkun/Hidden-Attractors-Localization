"""Lightweight workflow to compare memory policies: full_caputo vs finite_window.

Compares trajectories and checks that:
  - If the window covers the entire time horizon, they match exactly.
  - If the window is smaller, they differ and accumulate a memory-window error.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.systems.registry import get_system_by_id
from src.lure.nyquist import find_harmonic_candidates
from src.lure.seeds import build_lure_seed
from src.integrators.abm import caputo_abm_integrate

def run_compare_memory_policies(system_id: str = "chua_fractional_saturation", output_dir: str = "outputs/compare_memory_policies") -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Running compare_memory_policies for system '{system_id}' ===")
    
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
    
    h = 0.02
    t_final = 6.0  # 300 steps
    
    # 1. Full Caputo
    print("Simulating under full_caputo...")
    t_full, x_full, _ = caputo_abm_integrate(
        rhs=system.evaluate_rhs, x0=seed_pos, q=q, h=h, t_final=t_final, memory_mode="full", use_c_backend=False
    )
    
    # 2. Infinite-like window (covers entire run, say steps = 500)
    print("Simulating under covering finite_window (length=500)...")
    t_cov, x_cov, _ = caputo_abm_integrate(
        rhs=system.evaluate_rhs, x0=seed_pos, q=q, h=h, t_final=t_final,
        memory_mode="window", memory_window_length=500, use_c_backend=False
    )
    
    # 3. Small window (length=10)
    print("Simulating under short finite_window (length=10)...")
    t_short, x_short, _ = caputo_abm_integrate(
        rhs=system.evaluate_rhs, x0=seed_pos, q=q, h=h, t_final=t_final,
        memory_mode="window", memory_window_length=10, use_c_backend=False
    )
    
    # Check assertions
    cov_matches = np.allclose(x_full, x_cov, atol=1e-10)
    short_matches = np.allclose(x_full, x_short, atol=1e-3)
    
    print(f"Window covers whole horizon match: {cov_matches}")
    print(f"Short window match:              {short_matches} (should be False)")
    
    # Plotting comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_full, x_full[:, 0], label="Full Caputo history", color="#10b981", linewidth=2.5)
    ax.plot(t_cov, x_cov[:, 0], "--", label="Covering Window (L=500)", color="#3b82f6", linewidth=1.5)
    ax.plot(t_short, x_short[:, 0], ":", label="Short Window (L=10)", color="#ef4444", linewidth=2.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("x(t)")
    ax.set_title("Memory Policy Trajectory Comparison")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    
    plot_path = os.path.join(output_dir, "memory_policy_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Comparison plot saved to: {plot_path}")

if __name__ == "__main__":
    run_compare_memory_policies()
