"""
benchmarks/bench_efork_single_trajectory.py
============================================
Performance benchmarks for single-trajectory EFORK integration.

What is measured
----------------
- **Python EFORK (integer order)**: ``efork_q1_integrate`` with the integer
  Chua RHS at two trajectory lengths.  Pure-Python inner loop — this is the
  *slowest expected path* and sets the floor for regression detection.
- **C EFORK (fractional order)**: ``FractionalChuaBackend.integrate_efork3``
  at two trajectory lengths.  This is the production path; regressions here
  are the most critical.
- **Per-step overhead**: one ``efork_q1_step`` call, isolating the Butcher
  tableau arithmetic from the I/O overhead of the full integration.

Why this matters
----------------
EFORK is O(N*Lm/h) in time and O(Lm/h) in memory.  A hidden-attractor search
typically runs hundreds to thousands of trajectories.  A 2× slowdown in the
kernel can turn a 20-minute run into a 40-minute run.  Track this benchmark in
CI to catch accidental O(N²) regressions, memory leaks, and compilation-flag
changes.

Running
-------
# With pytest-benchmark (recommended):
    pip install pytest-benchmark
    python -m pytest benchmarks/bench_efork_single_trajectory.py -v

# Standalone (no pytest-benchmark required):
    python benchmarks/bench_efork_single_trajectory.py
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

# ── Shared constants (duplicated here so the file is self-contained) ──────────
Q = 0.9998
H = 0.005
LM = 10.0
T_FINAL_SHORT = 50.0
T_FINAL_LONG = 200.0
SEED_CANONICAL = np.array([0.1, 0.0, 0.0])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _integer_rhs(x: np.ndarray) -> np.ndarray:
    """Classic Chua ODE (q=1) right-hand side — piecewise model."""
    from hidden_attractors.models.chua import (
        chua_piecewise_parameters,
        nonlinearity_piecewise,
    )
    p = chua_piecewise_parameters()
    nl = nonlinearity_piecewise(x[0], p)
    dx = p.alpha * (x[1] - x[0] - nl)
    dy = x[0] - x[1] + x[2]
    dz = -p.beta * x[1] - p.gamma * x[2]
    return np.array([dx, dy, dz])


def _time_callable(fn: Callable, *, repeats: int = 3) -> tuple[float, float]:
    """Return (min_seconds, mean_seconds) over *repeats* calls."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


# ─────────────────────────────────────────────────────────────────────────────
# pytest-benchmark tests
# ─────────────────────────────────────────────────────────────────────────────

def test_efork_q1_step(benchmark):
    """Single EFORK-3 step — isolates Butcher-tableau overhead."""
    from hidden_attractors.solvers import efork_q1_step

    x0 = SEED_CANONICAL.copy()
    result = benchmark(efork_q1_step, _integer_rhs, x0, H)
    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_efork_integer_short(benchmark):
    """Pure-Python EFORK, integer order, T=50 s  (~10 000 steps)."""
    from hidden_attractors.solvers import efork_q1_integrate

    def _run():
        traj, status = efork_q1_integrate(
            _integer_rhs, SEED_CANONICAL.copy(), t_final=T_FINAL_SHORT, h=H
        )
        return traj, status

    traj, status = benchmark(_run)
    expected_rows = int(math.ceil(T_FINAL_SHORT / H)) + 1
    assert traj.shape[1] == 4  # t, x, y, z
    assert traj.shape[0] <= expected_rows
    assert status == "ok"


def test_efork_integer_long(benchmark):
    """Pure-Python EFORK, integer order, T=200 s  (~40 000 steps)."""
    from hidden_attractors.solvers import efork_q1_integrate

    def _run():
        return efork_q1_integrate(
            _integer_rhs, SEED_CANONICAL.copy(), t_final=T_FINAL_LONG, h=H
        )

    traj, status = benchmark(_run)
    assert traj.shape[1] == 4
    assert status == "ok"


def test_efork_c_short(benchmark, frac_backend):
    """C EFORK-3 backend, fractional q=0.9998, T=50 s."""
    def _run():
        return frac_backend.integrate_efork3(
            SEED_CANONICAL, q=Q, h=H, Lm=LM, t_final=T_FINAL_SHORT
        )

    traj = benchmark(_run)
    assert traj.ndim == 2
    assert traj.shape[1] == 4
    assert np.all(np.isfinite(traj))


def test_efork_c_long(benchmark, frac_backend):
    """C EFORK-3 backend, fractional q=0.9998, T=200 s."""
    def _run():
        return frac_backend.integrate_efork3(
            SEED_CANONICAL, q=Q, h=H, Lm=LM, t_final=T_FINAL_LONG
        )

    traj = benchmark(_run)
    assert traj.ndim == 2
    assert traj.shape[1] == 4
    assert np.all(np.isfinite(traj))


# ─────────────────────────────────────────────────────────────────────────────
# Standalone (no pytest-benchmark)
# ─────────────────────────────────────────────────────────────────────────────

def _standalone():
    """Run all benchmarks without pytest and print a summary table."""
    from hidden_attractors.native.backends import FractionalChuaBackend
    from hidden_attractors.solvers import efork_q1_integrate, efork_q1_step

    backend = FractionalChuaBackend.build()
    results: list[tuple[str, float, float]] = []

    def measure(label: str, fn: Callable, repeats: int = 5) -> None:
        lo, mean = _time_callable(fn, repeats=repeats)
        results.append((label, lo, mean))
        print(f"  {label:<48}  min={lo*1e3:8.1f} ms   mean={mean*1e3:8.1f} ms")

    print("\n=== EFORK single-trajectory benchmarks ===\n")

    measure(
        "efork_q1_step (1 step)",
        lambda: efork_q1_step(_integer_rhs, SEED_CANONICAL.copy(), H),
        repeats=1000,
    )
    measure(
        f"efork_q1_integrate  T={T_FINAL_SHORT}s (Python)",
        lambda: efork_q1_integrate(
            _integer_rhs, SEED_CANONICAL.copy(), t_final=T_FINAL_SHORT, h=H
        ),
    )
    measure(
        f"efork_q1_integrate  T={T_FINAL_LONG}s (Python)",
        lambda: efork_q1_integrate(
            _integer_rhs, SEED_CANONICAL.copy(), t_final=T_FINAL_LONG, h=H
        ),
    )
    measure(
        f"C EFORK-3  q={Q}  T={T_FINAL_SHORT}s",
        lambda: backend.integrate_efork3(
            SEED_CANONICAL, q=Q, h=H, Lm=LM, t_final=T_FINAL_SHORT
        ),
    )
    measure(
        f"C EFORK-3  q={Q}  T={T_FINAL_LONG}s",
        lambda: backend.integrate_efork3(
            SEED_CANONICAL, q=Q, h=H, Lm=LM, t_final=T_FINAL_LONG
        ),
    )

    print("\n--- Performance ratios ---")
    baseline = next(t for l, t, _ in results if "C EFORK-3" in l and str(T_FINAL_SHORT) in l)
    for label, lo, _ in results:
        if "Python" in label and str(T_FINAL_SHORT) in label:
            print(f"  Python / C ratio (T={T_FINAL_SHORT}s): {lo / baseline:.1f}×")


if __name__ == "__main__":
    _standalone()
