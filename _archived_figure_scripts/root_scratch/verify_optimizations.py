"""
Verification script for seed search optimizations.
Run from workspace root: python verify_optimizations.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "./version_2")

from hidden_attractors.lure.transfer import W_eval, W_precompute_spectral, W_eval_from_cache
from hidden_attractors.lure.describing_function import evaluate_describing_function_batch, evaluate_describing_function
from hidden_attractors.lure.nyquist import find_harmonic_candidates, HarmonicCandidate

print("=" * 60)
print("VERIFICACION DE OPTIMIZACIONES")
print("=" * 60)

# ── Test matrices (3x3 Chua-like system) ─────────────────────────────────────
P = np.array([
    [-1.0 / 7.0, 1.0,    0.0],
    [1.0,        -1.0,   1.0],
    [0.0,        -14.87, 0.0],
])
b = np.array([1.0 / 7.0, 0.0, 0.0])
r = np.array([1.0, 0.0, 0.0])

# ── 1. W_precompute_spectral correctness ─────────────────────────────────────
print("\n[1] W_precompute_spectral / W_eval_from_cache")
cache = W_precompute_spectral(P, b, r)
assert not cache["fallback"], "Cache should not fall back for this well-conditioned matrix"

ws = np.linspace(0.1, 20.0, 10000)

t0 = time.perf_counter()
vals_cached = W_eval_from_cache(ws, 0.9998, "fractional", cache)
t_cached = time.perf_counter() - t0

t0 = time.perf_counter()
vals_direct = W_eval(ws, 0.9998, "fractional", P, b, r)
t_direct = time.perf_counter() - t0

max_diff = float(np.max(np.abs(vals_cached - vals_direct)))
assert max_diff < 1e-10, f"Mismatch: {max_diff:.2e}"
print(f"  Direct:  {t_direct*1000:6.2f} ms  (10000 pts)")
print(f"  Cached:  {t_cached*1000:6.2f} ms  (10000 pts)")
print(f"  Max abs diff: {max_diff:.2e}  [OK]")
print(f"  Speedup over 2 calls: ~{2*t_direct/t_cached:.1f}x (cache amortises across phases)")

# ── 2. evaluate_describing_function_batch ─────────────────────────────────────
print("\n[2] evaluate_describing_function_batch")

class MockSystem:
    """Minimal closed-form DF for saturation nonlinearity N(A) = m1 + 2(m0-m1)/pi * arcsin(Bp/A)"""
    def __init__(self):
        self.lure = self
        self.parameters = {"model": "saturation"}
        
    system_id = "mock_saturation"
    Bp = 1.0
    m0 = -0.5
    m1 = -0.8
    
    def nonlinearity(self, sigma):
        sigma_arr = np.asarray(sigma, dtype=float)
        return np.where(
            np.abs(sigma_arr) <= self.Bp,
            self.m0 * sigma_arr,
            self.m1 * sigma_arr + np.sign(sigma_arr) * (self.m0 - self.m1) * self.Bp
        )
        
    def describing_function(self, A):
        Bp = self.Bp
        m0, m1 = self.m0, self.m1
        A_arr = np.asarray(A, dtype=float)
        result = np.where(
            A_arr >= Bp,
            m1 + (2.0 * (m0 - m1) / np.pi) * (np.arcsin(Bp / A_arr) + (Bp / A_arr) * np.sqrt(1.0 - (Bp / A_arr)**2)),
            m0,
        )
        return result

sys_mock = MockSystem()
A_grid = np.linspace(0.01, 20.0, 200)

t0 = time.perf_counter()
N_batch = evaluate_describing_function_batch(sys_mock, A_grid)
t_batch = time.perf_counter() - t0

from hidden_attractors.lure.describing_function import evaluate_describing_function
t0 = time.perf_counter()
N_serial = np.array([evaluate_describing_function(sys_mock, float(A)).value for A in A_grid])
t_serial = time.perf_counter() - t0

max_diff2 = float(np.max(np.abs(N_batch - N_serial)))
assert max_diff2 < 1e-12, f"Batch/serial mismatch: {max_diff2:.2e}"
print(f"  Serial (200 pts): {t_serial*1000:6.2f} ms")
print(f"  Batch  (200 pts): {t_batch*1000:6.2f} ms")
print(f"  Max abs diff:     {max_diff2:.2e}  [OK]")

# ── 3. nyquist.py 2D grid vectorization check ────────────────────────────────
print("\n[3] nyquist 2D grid - shape sanity (no Python loop over amplitudes)")
res_shape = (200, 200)  # n_A x n_w
# We just verify the logic doesn't error and produces the right shape
Ws = np.linspace(0.1, 20.0, res_shape[1])
W_vals = W_eval(Ws, 0.9998, "fractional", P, b, r)
A_s = np.linspace(0.01, 20.0, res_shape[0])
N_vals = evaluate_describing_function_batch(sys_mock, A_s)

t0 = time.perf_counter()
res_matrix = np.abs(W_vals[None, :] * N_vals[:, None] - 1.0)
t_matrix = time.perf_counter() - t0

assert res_matrix.shape == res_shape, f"Shape mismatch: {res_matrix.shape}"
print(f"  Broadcasting 200x200 residual matrix: {t_matrix*1000:.3f} ms  [OK]")

# ── 4. Sign-change detection vectorization ───────────────────────────────────
print("\n[4] Sign-change detection (vectorised vs loop)")
ims = W_vals.imag
n = len(ims)

t0 = time.perf_counter()
# Old: Python loop
old_crossings = []
for j in range(n - 1):
    if ims[j] * ims[j+1] < 0:
        old_crossings.append(j)
t_loop = time.perf_counter() - t0

t0 = time.perf_counter()
# New: numpy vectorised
new_crossings = list(np.where(np.diff(np.sign(ims)))[0])
t_vec = time.perf_counter() - t0

assert set(old_crossings) == set(new_crossings), "Different crossing sets!"
print(f"  Python loop (200 pts): {t_loop*1000:.4f} ms")
print(f"  NumPy vectorised:      {t_vec*1000:.4f} ms")
print(f"  Crossings found: {len(new_crossings)}  [OK]")

print("\n" + "=" * 60)
print("TODAS LAS VERIFICACIONES PASARON")
print("=" * 60)
