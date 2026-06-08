import sys
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt

# Insert version_2 in path
ROOT = Path("c:/Users/moren/Desktop/Codes/Hidden Attractors Fractional Order")
VERSION2 = ROOT / "version_2"
if str(VERSION2) not in sys.path:
    sys.path.insert(0, str(VERSION2))

from hidden_attractors.analysis.lyapunov_api import compute_lyapunov_spectrum
from hidden_attractors.models.chua import ChuaParameters, rhs_nonsmooth, jacobian_nonsmooth

# Parameters for candidate: biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776
ALPHA = 8.4562
BETA = 12.0732
GAMMA = 0.0052
M0 = -0.1768
M1 = -1.1468
Q = 0.9998
H = 0.01

def get_params():
    return ChuaParameters(
        model="nonsmooth",
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        m0=M0,
        m1=M1,
    )

def rhs(state):
    return rhs_nonsmooth(np.asarray(state, dtype=float), get_params())

def jacobian(state):
    return jacobian_nonsmooth(np.asarray(state, dtype=float), get_params())

# Load the trajectory to get the last state
traj_path = VERSION2 / "outputs/biased_saturation_search_q09998_corrected/trajectories/biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776_trajectory.csv"
if not traj_path.exists():
    print(f"Error: trajectory not found at {traj_path}")
    sys.exit(1)

data = np.loadtxt(traj_path, delimiter=",", skiprows=1)
last_state = data[-1, 1:4]
print(f"Loaded last state: {last_state}")

out_dir = VERSION2 / "outputs/biased_saturation_search_q09998_corrected/lyapunov"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Run variational ABM-QR
print("Starting variational ABM-QR Lyapunov calculation...")
summary_var = compute_lyapunov_spectrum(
    rhs=rhs,
    jacobian=jacobian,
    x0=last_state,
    q=Q,
    method="fractional_variational_abm_qr",
    h=H,
    t_final=160.0,
    t_burn=40.0,
    reorthonormalization_time=0.1,
    memory_mode="full",
    div_threshold=120.0,
    history_aware_qr=True,
)
result_var = summary_var.result
conv_var = np.asarray(result_var.convergence, dtype=float)
times_var = np.asarray(result_var.times, dtype=float)
print(f"Variational Exponents: {result_var.exponents}")

# 2. Run cloned dynamics
print("Starting cloned dynamics Lyapunov calculation...")
summary_clone = compute_lyapunov_spectrum(
    rhs=rhs,
    jacobian=None,
    x0=last_state,
    q=Q,
    method="fractional_cloned_dynamics_abm_gs_published",
    h=H,
    t_final=160.0,
    t_burn=40.0,
    reorthonormalization_time=0.1,
    memory_mode="published_block_restart",
    div_threshold=120.0,
    t_clone=2.0,
    k_blocks=100,
    delta=1e-4,
    orders=[Q, Q, Q],
    memory_protocol="published_block_restart",
)
result_clone = summary_clone.result
conv_clone = np.asarray(result_clone.convergence, dtype=float)
times_clone = np.asarray(result_clone.times, dtype=float)
print(f"Cloned Exponents: {result_clone.exponents}")

# Save results to a combined CSV
with open(out_dir / "biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776_lyapunov_two_methods_convergence.csv", "w") as f:
    f.write("time,method,lambda_1,lambda_2,lambda_3\n")
    for t_val, lambdas in zip(times_var, conv_var):
        f.write(f"{t_val},fractional_variational_abm_qr,{lambdas[0]},{lambdas[1]},{lambdas[2]}\n")
    for t_val, lambdas in zip(times_clone, conv_clone):
        f.write(f"{t_val},fractional_cloned_dynamics_abm_gs_published,{lambdas[0]},{lambdas[1]},{lambdas[2]}\n")

# Save summary
summary_json = {
    "candidate_id": "biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776",
    "variational_exponents": [float(e) for e in result_var.exponents],
    "cloned_exponents": [float(e) for e in result_clone.exponents],
}
with open(out_dir / "biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776_lyapunov_two_methods_summary.json", "w") as f:
    json.dump(summary_json, f, indent=2)

# Plot convergence (2 subplots)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True)

# Upper panel: Variational
ax1.plot(times_var, conv_var[:, 0], label=r"$\lambda_1$", color="#0b57ff", lw=0.9)
ax1.plot(times_var, conv_var[:, 1], label=r"$\lambda_2$", color="#10b981", lw=0.9)
ax1.plot(times_var, conv_var[:, 2], label=r"$\lambda_3$", color="#ef4444", lw=0.9)
ax1.axhline(0.0, color="red", lw=0.8, ls=":")
ax1.set_ylabel("fractional_variational_abm_qr")
ax1.grid(True, ls=":", lw=0.55)
ax1.legend(fontsize=8, loc="best")
ax1.set_title("Convergencia de Exponentes de Lyapunov (Candidato Sesgado)")

# Lower panel: Cloned
ax2.plot(times_clone, conv_clone[:, 0], label=r"$\lambda_1$", color="#0b57ff", lw=0.9)
ax2.plot(times_clone, conv_clone[:, 1], label=r"$\lambda_2$", color="#10b981", lw=0.9)
ax2.plot(times_clone, conv_clone[:, 2], label=r"$\lambda_3$", color="#ef4444", lw=0.9)
ax2.axhline(0.0, color="red", lw=0.8, ls=":")
ax2.set_ylabel("fractional_cloned_dynamics_abm_gs_published")
ax2.grid(True, ls=":", lw=0.55)
ax2.legend(fontsize=8, loc="best")
ax2.set_xlabel("t (s)")

plt.tight_layout()

# Save figures in Figs/ and in output directory
figs_dest = ROOT / "DF y NC Chua entero y fraccionario copy" / "Figs"
figs_dest.mkdir(parents=True, exist_ok=True)

for path in [out_dir, figs_dest]:
    plt.savefig(path / "chua_frac_ns_biased_fig12_lyapunov.png", dpi=220)
    plt.savefig(path / "chua_frac_ns_biased_fig12_lyapunov.pdf")

print("Done! Figures saved successfully.")
