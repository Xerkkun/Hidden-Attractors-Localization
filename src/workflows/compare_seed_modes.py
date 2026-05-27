"""Lightweight workflow to compare seed generation modes: integer vs fractional.

We keep the final dynamics integration identical, but construct the seed using:
  1. Integer assumption (closed-form or modal q=1)
  2. Fractional assumption (modal q=system.q)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.systems.registry import get_system_by_id
from src.lure.nyquist import find_harmonic_candidates
from src.lure.seeds import build_lure_seed
from src.integrators.general import integrate_general

def run_compare_seed_modes(system_id: str = "chua_fractional_saturation", output_dir: str = "outputs/compare_seed_modes") -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Running compare_seed_modes for system '{system_id}' ===")
    
    # Load system
    system = get_system_by_id(system_id)
    q = system.q
    print(f"System loaded: q = {q}")
    
    # Find candidates
    candidates = find_harmonic_candidates(system, transfer_mode="fractional")
    if not candidates:
        print("No harmonic candidates found for seeding comparison.")
        return
        
    A0, omega0, k = candidates[0]
    print(f"Harmonic seed candidate: A0={A0:.4f}, omega0={omega0:.4f}, k={k:.4f}")
    
    # 1. Build Integer Seed (closed_form_integer require q=1.0, so we use modal with q=1.0)
    seed_int, _ = build_lure_seed(
        system, A0, omega0, k,
        q=1.0,
        transfer_mode="integer",
        seed_construction="modal"
    )
    
    # 2. Build Fractional Seed (modal q=system.q)
    seed_frac, _ = build_lure_seed(
        system, A0, omega0, k,
        q=q,
        transfer_mode="fractional",
        seed_construction="modal"
    )
    
    print(f"Integer-assumed Seed:    {seed_int}")
    print(f"Fractional-assumed Seed: {seed_frac}")
    print(f"Seed Euclidean Distance: {np.linalg.norm(seed_int - seed_frac):.6f}")
    
    # Integrate both under identical fractional dynamics
    h = 0.01
    t_final = 50.0
    
    print("Integrating under integer seed...")
    t_int, x_int, status_int = integrate_general(
        rhs=system.evaluate_rhs, x0=seed_int, q=q, h=h, t_final=t_final, integrator="efork3"
    )
    
    print("Integrating under fractional seed...")
    t_frac, x_frac, status_frac = integrate_general(
        rhs=system.evaluate_rhs, x0=seed_frac, q=q, h=h, t_final=t_final, integrator="efork3"
    )
    
    # Plotting comparison
    fig = plt.figure(figsize=(12, 5))
    
    # Time-series comparison
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(t_int, x_int[:, 0], label="Integer-Assumed Seed", color="#ef4444", alpha=0.85)
    ax1.plot(t_frac, x_frac[:, 0], label="Fractional-Assumed Seed", color="#3b82f6", alpha=0.85)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("x(t)")
    ax1.set_title("Trajectory Comparison x(t)")
    ax1.legend(framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Phase portrait comparison
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(x_int[:, 0], x_int[:, 1], x_int[:, 2], label="Integer-Assumed", color="#ef4444", alpha=0.7)
    ax2.plot(x_frac[:, 0], x_frac[:, 1], x_frac[:, 2], label="Fractional-Assumed", color="#3b82f6", alpha=0.7)
    ax2.scatter(seed_int[0], seed_int[1], seed_int[2], color="#ef4444", s=50, label="Seed Int")
    ax2.scatter(seed_frac[0], seed_frac[1], seed_frac[2], color="#3b82f6", s=50, label="Seed Frac")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("3D State space")
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "seed_mode_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Comparison plot saved to: {plot_path}")

if __name__ == "__main__":
    run_compare_seed_modes()
