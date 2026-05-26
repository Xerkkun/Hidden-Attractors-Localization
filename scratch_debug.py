import sys
from pathlib import Path
workspace_root = Path(__file__).resolve().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.systems.registry import get_system_by_id
from src.lure.transfer import W_eval
import numpy as np
from scipy.optimize import root_scalar

sys_int = get_system_by_id("chua_integer_saturation")
q = 1.0
transfer_mode = "integer"
omega_min = 0.5
omega_max = 4.0
grid_size_omega = 200
scan_n = max(grid_size_omega, 20000)

ws = np.linspace(omega_min, omega_max, scan_n)
ims = []
for w in ws:
    val = W_eval(w, q, transfer_mode, sys_int.P, sys_int.b, sys_int.r)
    ims.append(val.imag)

ims = np.array(ims)
sign_changes = np.where(ims[:-1] * ims[1:] < 0.0)[0]
print("Number of sign changes in Im(W):", len(sign_changes))

omega_roots = []
for j in sign_changes:
    def root_f(w):
        return W_eval(w, q, transfer_mode, sys_int.P, sys_int.b, sys_int.r).imag
    sol = root_scalar(root_f, bracket=[ws[j], ws[j+1]], method="bisection")
    if sol.converged:
        omega_roots.append(sol.root)
print("omega_roots:", omega_roots)

for w0 in omega_roots:
    W0 = W_eval(w0, q, transfer_mode, sys_int.P, sys_int.b, sys_int.r)
    k = -1.0 / W0.real
    print(f"w0={w0}, k={k}")
    
    # Solve Phi(A) = 0
    # Let's check what phi_func outputs for some amplitudes
    def phi_func(A):
        from scipy.integrate import quad
        if A <= 0:
            return 0.0
        def integrand(t):
            return (sys_int.psi(A * np.cos(w0 * t)) - k * A * np.cos(w0 * t)) * np.cos(w0 * t)
        val, _ = quad(integrand, 0.0, 2.0 * np.pi / w0, limit=100)
        return val
        
    print("phi_func(1.0):", phi_func(1.0))
    print("phi_func(5.85):", phi_func(5.85))
    print("phi_func(5.86):", phi_func(5.86))
    
    as_ = np.linspace(0.01, 20.0, 200)
    phi_vals = [phi_func(a) for a in as_]
    phi_sign_changes = np.where(np.array(phi_vals[:-1]) * np.array(phi_vals[1:]) < 0.0)[0]
    print("phi sign changes indices:", phi_sign_changes)
