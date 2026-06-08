import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("c:/Users/moren/Desktop/Codes/Hidden Attractors Fractional Order")
VERSION2 = ROOT / "version_2"
CSV_PATH = VERSION2 / "outputs/biased_saturation_search_q09998_corrected/lyapunov/biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776_lyapunov_two_methods_convergence.csv"

if not CSV_PATH.exists():
    print(f"Error: Lyapunov CSV not found at {CSV_PATH}")
    sys.exit(1)

# Load data
df = pd.read_csv(CSV_PATH)

# Separate by method
df_var = df[df["method"] == "fractional_variational_abm_qr"]
df_clone = df[df["method"] == "fractional_cloned_dynamics_abm_gs_published"]

times_var = df_var["time"].values
conv_var = df_var[["lambda_1", "lambda_2", "lambda_3"]].values

times_clone = df_clone["time"].values
conv_clone = df_clone[["lambda_1", "lambda_2", "lambda_3"]].values

# Plot convergence (2 subplots)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True)

# Upper panel: Variational
ax1.plot(times_var, conv_var[:, 0], label=r"$\lambda_1$", color="#0b57ff", lw=0.9)
ax1.plot(times_var, conv_var[:, 1], label=r"$\lambda_2$", color="#10b981", lw=0.9)
ax1.plot(times_var, conv_var[:, 2], label=r"$\lambda_3$", color="#ef4444", lw=0.9)
ax1.axhline(0.0, color="black", lw=0.8, ls=":")
ax1.set_ylabel("Método Variacional")
ax1.grid(True, ls=":", lw=0.55)
ax1.legend(fontsize=8, loc="best")
ax1.set_title("Convergencia de Exponentes de Lyapunov (Candidato Sesgado)")

# Lower panel: Cloned
ax2.plot(times_clone, conv_clone[:, 0], label=r"$\lambda_1$", color="#0b57ff", lw=0.9)
ax2.plot(times_clone, conv_clone[:, 1], label=r"$\lambda_2$", color="#10b981", lw=0.9)
ax2.plot(times_clone, conv_clone[:, 2], label=r"$\lambda_3$", color="#ef4444", lw=0.9)
ax2.axhline(0.0, color="black", lw=0.8, ls=":")
ax2.set_ylabel("Dinámica Clonada")
ax2.grid(True, ls=":", lw=0.55)
ax2.legend(fontsize=8, loc="best")
ax2.set_xlabel("t (s)")

plt.tight_layout()

# Destination directories
figs_dest = ROOT / "DF y NC Chua entero y fraccionario copy" / "Figs"
figs_dest.mkdir(parents=True, exist_ok=True)
out_dir = VERSION2 / "outputs/biased_saturation_search_q09998_corrected/lyapunov"

for path in [out_dir, figs_dest]:
    plt.savefig(path / "chua_frac_ns_biased_fig12_lyapunov.png", dpi=220)
    plt.savefig(path / "chua_frac_ns_biased_fig12_lyapunov.pdf")

print("Regenerated Lyapunov convergence plot with clean labels successfully!")
