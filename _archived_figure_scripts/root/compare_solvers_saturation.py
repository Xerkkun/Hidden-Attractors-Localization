import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from types import SimpleNamespace

# Ensure version_2 is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "version_2"))

from hidden_attractors.models.chua import ChuaParameters
from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.systems.builtins import _chua_lure_system

def main():
    # Candidate parameters
    alpha = 8.4562
    beta = 12.0732
    gamma = 0.0052
    m1 = -1.2
    m0 = -0.2
    q = 0.9998
    h = 0.01
    t_final = 300.0
    t_transient = 100.0
    
    # Continuation final state (seed for dynamics)
    x0 = np.array([5.658588628546241, -0.35752352140248367, -7.182793468822785])
    
    # System definition
    system = SimpleNamespace(
        system_id="chua_fractional_saturation",
        name="chua_fractional_saturation",
        parameters={
            "model": "nonsmooth",
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "m0": m0,
            "m1": m1
        },
        lure=_chua_lure_system({
            "model": "nonsmooth",
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "m0": m0,
            "m1": m1
        })
    )
    
    print("Integrating Candidate with ABM (full memory)...")
    t_abm, x_abm, status_abm, _ = fractional_integrate(
        rhs=lambda t, val: system.lure.matrix @ val + system.lure.input_vector * system.lure.nonlinearity(system.lure.output_vector @ val),
        x0=x0,
        q=q,
        h=h,
        t_final=t_final,
        method="abm",
        memory_mode="full",
        system=system,
        use_c_backend=True,
        allow_python_fallback=True
    )
    print(f"ABM status: {status_abm}")
    
    print("Integrating Candidate with EFORK (full memory)...")
    t_efork_full, x_efork_full, status_efork_full, _ = fractional_integrate(
        rhs=lambda t, val: system.lure.matrix @ val + system.lure.input_vector * system.lure.nonlinearity(system.lure.output_vector @ val),
        x0=x0,
        q=q,
        h=h,
        t_final=t_final,
        method="efork3",
        memory_mode="full",
        system=system,
        use_c_backend=True,
        allow_python_fallback=True
    )
    print(f"EFORK full status: {status_efork_full}")
    
    print("Integrating Candidate with EFORK (truncated memory, Lm=10.0)...")
    t_efork_trunc, x_efork_trunc, status_efork_trunc, _ = fractional_integrate(
        rhs=lambda t, val: system.lure.matrix @ val + system.lure.input_vector * system.lure.nonlinearity(system.lure.output_vector @ val),
        x0=x0,
        q=q,
        h=h,
        t_final=t_final,
        method="efork3",
        memory_mode="window",
        memory_window_length=int(10.0 / h),
        system=system,
        use_c_backend=True,
        allow_python_fallback=True
    )
    print(f"EFORK truncated status: {status_efork_trunc}")
    
    # Post-transient slice for plotting
    n_burn = int(t_transient / h)
    
    # Create 2x2 grid plot
    fig = plt.figure(figsize=(12, 10))
    
    # 1. ABM Full Memory
    ax1 = fig.add_subplot(221, projection="3d")
    tail_abm = x_abm[n_burn:]
    ax1.plot(tail_abm[:, 0], tail_abm[:, 1], tail_abm[:, 2], lw=0.45, color="blue")
    ax1.set_title("ABM (Full Memory)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    
    # 2. EFORK Full Memory
    ax2 = fig.add_subplot(222, projection="3d")
    tail_ef_full = x_efork_full[n_burn:]
    ax2.plot(tail_ef_full[:, 0], tail_ef_full[:, 1], tail_ef_full[:, 2], lw=0.45, color="green")
    ax2.set_title("EFORK (Full Memory)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    
    # 3. EFORK Truncated Memory
    ax3 = fig.add_subplot(223, projection="3d")
    tail_ef_trunc = x_efork_trunc[n_burn:]
    ax3.plot(tail_ef_trunc[:, 0], tail_ef_trunc[:, 1], tail_ef_trunc[:, 2], lw=0.45, color="red")
    ax3.set_title("EFORK (Truncated Memory, Lm=10.0s)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    
    # 4. Phase B Comparison: Full vs Truncated Continuation final attractors
    # We will load and plot Phase B full memory vs truncated memory trajectories
    ax4 = fig.add_subplot(224, projection="3d")
    
    # Load Phase B full trajectory
    pb_full_path = Path("outputs/saturation_search_seed0p9998_mem_full_sweep/m1_m1p2000_m0_m0p2000_branch_0_trajectory.csv")
    pb_window_path = Path("outputs/saturation_search_seed0p9998_mem_window_sweep/m1_m1p2000_m0_m0p2000_branch_0_trajectory.csv")
    
    if pb_full_path.exists():
        pb_full_data = np.loadtxt(pb_full_path, delimiter=",", skiprows=1)
        tail_pb_full = pb_full_data[n_burn:, 1:4]
        ax4.plot(tail_pb_full[:, 0], tail_pb_full[:, 1], tail_pb_full[:, 2], lw=0.45, color="magenta", label="Phase B Full")
        
    if pb_window_path.exists():
        pb_wind_data = np.loadtxt(pb_window_path, delimiter=",", skiprows=1)
        tail_pb_wind = pb_wind_data[n_burn:, 1:4]
        ax4.plot(tail_pb_wind[:, 0], tail_pb_wind[:, 1], tail_pb_wind[:, 2], lw=0.45, color="orange", label="Phase B Trunc ($L_m=10$)")
        
    ax4.set_title("Phase B Continuation Comparison", fontsize=10, fontweight="bold")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_zlabel("z")
    ax4.legend(fontsize=8, loc="upper right")
    
    fig.suptitle(f"Non-Smooth Chua Chaotic Attractor Solver Comparison\n(alpha={alpha}, beta={beta}, gamma={gamma}, m1={m1}, m0={m0}, q={q})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    
    out_dir = Path("outputs/saturation_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / "m1_m1p2000_m0_m0p2000_branch_0_four_solvers_phase3d.png"
    plt.savefig(out_img, dpi=200)
    plt.close()
    print(f"Saved solver comparison figure to {out_img}")
    
    # Save raw trajectories
    np.savetxt(out_dir / "trajectory_abm.csv", np.column_stack((t_abm, x_abm)), delimiter=",", header="t,x,y,z", comments="")
    np.savetxt(out_dir / "trajectory_efork_full.csv", np.column_stack((t_efork_full, x_efork_full)), delimiter=",", header="t,x,y,z", comments="")
    np.savetxt(out_dir / "trajectory_efork_trunc.csv", np.column_stack((t_efork_trunc, x_efork_trunc)), delimiter=",", header="t,x,y,z", comments="")
    print("Saved raw trajectories in outputs/saturation_comparison/")

if __name__ == "__main__":
    main()
