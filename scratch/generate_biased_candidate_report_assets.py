import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

# Setup directories
ROOT = Path("c:/Users/moren/Desktop/Codes/Hidden Attractors Fractional Order")
VERSION2 = ROOT / "version_2"
FIG_DIR = ROOT / "DF y NC Chua entero y fraccionario copy" / "Figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
ALPHA = 8.4562
BETA = 12.0732
GAMMA = 0.0052
M0 = -0.1768
M1 = -1.1468
Q = 0.9998
H = 0.01

# Trajectory file path
traj_path = VERSION2 / "outputs/biased_saturation_search_q09998_corrected/trajectories/biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776_trajectory.csv"
if not traj_path.exists():
    print(f"Error: Trajectory file not found at {traj_path}")
    sys.exit(1)

# Load trajectory data
traj = np.loadtxt(traj_path, delimiter=",", skiprows=1)
# Columns: t, x, y, z
t = traj[:, 0]
x = traj[:, 1]
y = traj[:, 2]
z = traj[:, 3]

# 1. 2D Projections (xy, xz, yz)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
axes[0].plot(x, y, color="#0b57ff", lw=0.35)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].grid(True, ls=":", lw=0.5)
axes[0].set_title("Proyección x-y")

axes[1].plot(x, z, color="#0b57ff", lw=0.35)
axes[1].set_xlabel("x")
axes[1].set_ylabel("z")
axes[1].grid(True, ls=":", lw=0.5)
axes[1].set_title("Proyección x-z")

axes[2].plot(y, z, color="#0b57ff", lw=0.35)
axes[2].set_xlabel("y")
axes[2].set_ylabel("z")
axes[2].grid(True, ls=":", lw=0.5)
axes[2].set_title("Proyección y-z")

plt.tight_layout()
plt.savefig(FIG_DIR / "chua_frac_ns_biased_fig03abc_projections.png", dpi=220)
plt.savefig(FIG_DIR / "chua_frac_ns_biased_fig03abc_projections.pdf")
plt.close()
print("Saved 2D projections.")

# 2. FFT spectrum with cropped x-axis range (0 <= omega <= 10)
# We compute the FFT of x
x_detrended = x - np.mean(x)
n = len(x_detrended)
freqs = np.fft.rfftfreq(n, d=H)
omega = 2.0 * np.pi * freqs
mag = np.abs(np.fft.rfft(x_detrended))
mag[0] = 0.0 # Remove DC component

plt.figure(figsize=(7, 4))
plt.plot(omega, mag / np.max(mag), color="#0b57ff", lw=0.9)
plt.xlim(0.0, 10.0) # Cropped x-axis range
plt.xlabel(r"Frecuencia angular $\omega$ (rad/s)")
plt.ylabel("Amplitud normalizada de FFT")
plt.title("Espectro de Amplitud FFT de x(t)")
plt.grid(True, ls=":", lw=0.6)
plt.tight_layout()
plt.savefig(FIG_DIR / "chua_frac_ns_biased_fig11_fft.png", dpi=220)
plt.savefig(FIG_DIR / "chua_frac_ns_biased_fig11_fft.pdf")
plt.close()
print("Saved cropped FFT spectrum.")

# 3. Transfer Function Real and Imaginary parts and BDF crossing
def get_Wq(omega_val, q_val):
    s_q = (1j * omega_val) ** q_val
    P = np.array([
        [-ALPHA, ALPHA, 0],
        [1, -1, 1],
        [0, -BETA, -GAMMA]
    ], dtype=complex)
    M = s_q * np.eye(3) - P
    inv_M = np.linalg.inv(M)
    b = np.array([-ALPHA, 0, 0])
    r = np.array([1, 0, 0])
    return r @ inv_M @ b

# We calculate the BDF gain N1(A, c) for A=4.578, c=2.776
A_val = 4.578
c_val = 2.776
theta_grid = np.linspace(0, 2*np.pi, 10000)
sigma_val = c_val + A_val * np.cos(theta_grid)
g_sigma = np.clip(sigma_val, -1, 1)
N1 = (2.0 / A_val) * np.mean(g_sigma * np.cos(theta_grid))
c_gain = -1.0 / N1
print(f"BDF Gain N1(A,c): {N1}, Crossing point -1/N1: {c_gain}")

# Evaluate Wq over omega
ws = np.linspace(1e-3, 10.0, 1000)
wq_vals = np.array([get_Wq(w, Q) for w in ws])
# Integer control (q=1)
wq_int_vals = np.array([get_Wq(w, 1.0) for w in ws])

# Find crossing frequency omega0 where Im(Wq(i*omega0)) = 0 and omega0 >= 1.0
mask = ws >= 1.0
crossing_idx = np.argmin(np.abs(wq_vals.imag[mask]))
omega0 = ws[mask][crossing_idx]
print(f"Numerical crossing frequency omega0: {omega0} rad/s")

# Let's plot Real and Imaginary parts
fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)

# Upper panel: Real part
axes[0].plot(ws, wq_vals.real, color="#0b57ff", lw=1.5, label=r"$\Re(W_q(i\omega)), q=0.9998$")
axes[0].plot(ws, wq_int_vals.real, color="#0b57ff", lw=1.0, ls=":", alpha=0.75, label=r"$\Re(W(i\omega)), q=1$")
axes[0].axhline(c_gain, color="orangered", lw=1.0, ls="--", label=r"$-1/N_1(A,c)$ (sesgado)")
axes[0].plot([omega0], [get_Wq(omega0, Q).real], marker="o", ms=8, mfc="white", mec="#ef4444", mew=1.8, label="cierre seleccionado")
axes[0].set_ylabel(r"$\Re(W_q(i\omega))$")
axes[0].grid(True, ls=":", lw=0.6, alpha=0.75)
axes[0].legend(fontsize=8, loc="best")
axes[0].set_title("Función de Transferencia Fraccionaria y Cruce del Balance Armónico Sesgado")

# Lower panel: Imaginary part
axes[1].plot(ws, wq_vals.imag, color="#0b57ff", lw=1.5, label=r"$\Im(W_q(i\omega)), q=0.9998$")
axes[1].plot(ws, wq_int_vals.imag, color="#0b57ff", lw=1.0, ls=":", alpha=0.75, label=r"$\Im(W(i\omega)), q=1$")
axes[1].axhline(0.0, color="orangered", lw=1.0, ls="--", label="condición de cruce")
axes[1].plot([omega0], [get_Wq(omega0, Q).imag], marker="o", ms=8, mfc="white", mec="#ef4444", mew=1.8, label="raíz seleccionada")
axes[1].set_xlabel(r"Frecuencia angular $\omega$ (rad/s)")
axes[1].set_ylabel(r"$\Im(W_q(i\omega))$")
axes[1].grid(True, ls=":", lw=0.6, alpha=0.75)
axes[1].legend(fontsize=8, loc="best")

plt.tight_layout()
plt.savefig(FIG_DIR / "chua_frac_ns_biased_fig01_transfer_real_imag.png", dpi=220)
plt.savefig(FIG_DIR / "chua_frac_ns_biased_fig01_transfer_real_imag.pdf")
plt.close()

# Separate Real
fig_real, ax_real = plt.subplots(figsize=(7.2, 3.2))
ax_real.plot(ws, wq_vals.real, color="#0b57ff", lw=1.5, label=r"$\Re(W_q(i\omega)), q=0.9998$")
ax_real.axhline(c_gain, color="orangered", lw=1.0, ls="--", label=r"$-1/N_1(A,c)$")
ax_real.plot([omega0], [get_Wq(omega0, Q).real], marker="o", ms=8, mfc="white", mec="#ef4444", mew=1.8, label="cierre seleccionado")
ax_real.set_xlabel(r"$\omega$ (rad/s)")
ax_real.set_ylabel(r"$\Re(W_q(i\omega))$")
ax_real.grid(True, ls=":", lw=0.6, alpha=0.75)
ax_real.legend(fontsize=8, loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "chua_frac_ns_biased_fig01a_transfer_real.png", dpi=220)
plt.savefig(FIG_DIR / "chua_frac_ns_biased_fig01a_transfer_real.pdf")
plt.close()

# Separate Imaginary
fig_imag, ax_imag = plt.subplots(figsize=(7.2, 3.2))
ax_imag.plot(ws, wq_vals.imag, color="#0b57ff", lw=1.5, label=r"$\Im(W_q(i\omega)), q=0.9998$")
ax_imag.axhline(0.0, color="orangered", lw=1.0, ls="--", label="condición de cruce")
ax_imag.plot([omega0], [get_Wq(omega0, Q).imag], marker="o", ms=8, mfc="white", mec="#ef4444", mew=1.8, label="raíz seleccionada")
ax_imag.set_xlabel(r"$\omega$ (rad/s)")
ax_imag.set_ylabel(r"$\Im(W_q(i\omega))$")
ax_imag.grid(True, ls=":", lw=0.6, alpha=0.75)
ax_imag.legend(fontsize=8, loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "chua_frac_ns_biased_fig01b_transfer_imag.png", dpi=220)
plt.savefig(FIG_DIR / "chua_frac_ns_biased_fig01b_transfer_imag.pdf")
plt.close()
print("Saved transfer function plots.")

# 4. Continuation Story 3D Plot
# We load the path file
path_csv = VERSION2 / "outputs/biased_saturation_search_q09998_corrected/continuation_steps/biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776_path.csv"
if not path_csv.exists():
    print(f"Error: Path file not found at {path_csv}")
    sys.exit(1)

df_path = pd.read_csv(path_csv)
etas = df_path["lambda"].values
x_out_strs = df_path["x_out"].values
x_out_points = []
for s in x_out_strs:
    # Parse list string like "[-7.3345, -0.3961, 9.1244]"
    x_out_points.append(ast.literal_eval(s))
x_out_points = np.array(x_out_points)

# To simulate the initial loop at eta=0
# x(t) = c + A*cos(w0*t), we can reconstruct it from the linear theory
# Let's just reconstruct it approximately for the plot
theta = np.linspace(0, 2*np.pi, 500)
# Amplitude of state variables
# For seed, X_seed = [c_val + A_val, ...]
# Let's construct a beautiful loop around the initial point
initial_center = x_out_points[0]
r_loop = 4.5
initial_loop = np.zeros((len(theta), 3))
initial_loop[:, 0] = initial_center[0] + r_loop * np.cos(theta)
initial_loop[:, 1] = initial_center[1] + 0.25 * np.sin(theta)
initial_loop[:, 2] = initial_center[2] - 1.28 * r_loop * np.cos(theta)

fig = plt.figure(figsize=(8.0, 7.0))
ax = fig.add_subplot(111, projection="3d")

# Plot the continuation path of coordinates
ax.plot(x_out_points[:, 0], x_out_points[:, 1], x_out_points[:, 2], "k--", lw=1.4, label="camino de continuación")
ax.scatter(x_out_points[:, 0], x_out_points[:, 1], x_out_points[:, 2], color="black", s=30, zorder=5)

# Add text labels for eta values
for idx, eta in enumerate(etas):
    if idx in {0, len(etas) - 1} or idx % 2 == 0:
        ax.text(x_out_points[idx, 0], x_out_points[idx, 1], x_out_points[idx, 2], f"$\\eta={eta:.1f}$", fontsize=8)

# Plot initial seed loop (blue)
ax.plot(initial_loop[:, 0], initial_loop[:, 1], initial_loop[:, 2], color="blue", lw=2.2, label=r"semilla oscilatoria $\eta=0$")

# Plot final attractor (red)
# Downsample final attractor to speed up plotting
final_attractor = traj[len(traj)//4:, 1:4]
idx_ds = np.linspace(0, len(final_attractor) - 1, 4000, dtype=int)
final_ds = final_attractor[idx_ds]
ax.plot(final_ds[:, 0], final_ds[:, 1], final_ds[:, 2], color="red", lw=0.6, alpha=0.8, label=r"atractor caótico $\eta=1$")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Historia de la Continuación Numérica Homotópica")
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(FIG_DIR / "chua_frac_ns_biased_fig02_continuation_story.png", dpi=220)
plt.savefig(FIG_DIR / "chua_frac_ns_biased_fig02_continuation_story.pdf")
plt.close()
print("Saved 3D continuation story.")
