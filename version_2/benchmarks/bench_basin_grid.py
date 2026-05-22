"""
benchmarks/bench_basin_grid.py
================================
Performance benchmarks for basin-of-attraction grid sweeps.

What is measured
----------------
- **Single-point classification** via ``BasinBackend.classify_point``:
  measures per-call overhead at varying trajectory lengths (T_final).
- **Grid sweep — pure Python loop** (N×N grid, no multiprocessing):
  total wall time and per-point throughput.  Sizes: 5×5, 10×10, 20×20.
- **Grid sweep — classification only** (no trajectory storage):
  exercises the hot path without numpy memory allocation overhead.

Why this matters
----------------
A realistic basin sweep over a 50×50 grid with T=200 s and H=0.005 requires
250 000 trajectory evaluations.  At 500 ms/point that is 34 hours; at 50 ms/point
it is 3.5 hours.  Even a 20 % regression in per-point cost is consequential.
Track this benchmark to:
- Detect changes in the C compilation flags (OpenMP, -O3, etc.).
- Detect Python-level overhead added to the classification dispatch.
- Provide a scaling profile for estimating full-basin compute budgets.

Running
-------
    python -m pytest benchmarks/bench_basin_grid.py -v
    python benchmarks/bench_basin_grid.py           # standalone
"""

from __future__ import annotations

import itertools
import time
from typing import Callable

import numpy as np

# ── Canonical problem parameters ──────────────────────────────────────────────
Q = 0.9998
H = 0.005
LM = 10.0
T_BURN = 20.0
T_FINAL_CLASSIFY = 60.0   # classification trajectory length (per point)
DIV_NORM = 120.0
R_BOUND = 60.0
EQ_TOL = 1.0e-3
CAP_WIN = 150
MEAN_X_GAP = 0.75


# ─────────────────────────────────────────────────────────────────────────────
# Grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_grid(n: int, x_range=(-1.5, 1.5), z_range=(-0.5, 0.5)):
    """Return a flat list of (x, 0, z) initial conditions on an n×n grid."""
    xs = np.linspace(*x_range, n)
    zs = np.linspace(*z_range, n)
    return [(float(x), 0.0, float(z)) for x, z in itertools.product(xs, zs)]


def _classify_grid(backend, grid: list, **kwargs) -> list[int]:
    """Classify every point in *grid*; return list of class IDs."""
    return [
        backend.classify_point(pt, **kwargs)
        for pt in grid
    ]


# ─────────────────────────────────────────────────────────────────────────────
# pytest-benchmark tests
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFY_KWARGS = dict(
    q=Q, h=H, Lm=LM,
    t_final=T_FINAL_CLASSIFY, t_burn=T_BURN,
    divergence_norm=DIV_NORM, r_bound=R_BOUND,
    equilibrium_tol=EQ_TOL, cap_win=CAP_WIN, mean_x_gap=MEAN_X_GAP,
)


def test_classify_single_point(benchmark, basin_backend):
    """Cost of classifying a single initial condition."""
    pt = [0.1, 0.0, 0.0]

    result = benchmark(basin_backend.classify_point, pt, **CLASSIFY_KWARGS)
    assert 0 <= result <= 5


def test_classify_single_point_t100(benchmark, basin_backend):
    """Single-point classification at T_final=100 s (longer trajectory)."""
    pt = [0.1, 0.0, 0.0]
    kwargs = {**CLASSIFY_KWARGS, "t_final": 100.0}

    result = benchmark(basin_backend.classify_point, pt, **kwargs)
    assert 0 <= result <= 5


def test_grid_5x5(benchmark, basin_backend):
    """5×5 grid sweep — 25 points.  Fastest sanity check."""
    grid = _make_grid(5)

    def _run():
        return _classify_grid(basin_backend, grid, **CLASSIFY_KWARGS)

    labels = benchmark(_run)
    assert len(labels) == 25
    assert all(0 <= c <= 5 for c in labels)


def test_grid_10x10(benchmark, basin_backend):
    """10×10 grid sweep — 100 points.  Primary regression sentinel."""
    grid = _make_grid(10)

    def _run():
        return _classify_grid(basin_backend, grid, **CLASSIFY_KWARGS)

    labels = benchmark(_run)
    assert len(labels) == 100


def test_grid_20x20(benchmark, basin_backend):
    """20×20 grid sweep — 400 points.  Realistic small-basin run."""
    grid = _make_grid(20)

    def _run():
        return _classify_grid(basin_backend, grid, **CLASSIFY_KWARGS)

    labels = benchmark(_run)
    assert len(labels) == 400


# ─────────────────────────────────────────────────────────────────────────────
# Standalone
# ─────────────────────────────────────────────────────────────────────────────

def _time_callable(fn: Callable, repeats: int = 3) -> tuple[float, float]:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


def _standalone():
    from hidden_attractors.native.backends import BasinBackend

    backend = BasinBackend.build()
    pt = [0.1, 0.0, 0.0]

    print("\n=== Basin grid benchmarks ===\n")

    # Single-point cost
    lo, mean = _time_callable(
        lambda: backend.classify_point(pt, **CLASSIFY_KWARGS), repeats=10
    )
    per_point_ms = mean * 1e3
    print(f"  Single-point classify  T={T_FINAL_CLASSIFY}s"
          f"  min={lo*1e3:.1f} ms  mean={per_point_ms:.1f} ms")

    print()
    for n in [5, 10, 20]:
        grid = _make_grid(n)
        total = n * n
        lo, mean = _time_callable(
            lambda g=grid: _classify_grid(backend, g, **CLASSIFY_KWARGS),
            repeats=2,
        )
        throughput = total / mean
        print(
            f"  {n:2d}×{n:2d} grid ({total:4d} pts)"
            f"  min={lo:.2f}s  mean={mean:.2f}s"
            f"  throughput={throughput:.1f} pts/s"
            f"  ≈{mean/total*1e3:.1f} ms/pt"
        )

    # Extrapolate to production scale
    print()
    print("--- Production-scale estimates (extrapolated from 10×10 baseline) ---")
    grid_10 = _make_grid(10)
    _, mean_10 = _time_callable(
        lambda: _classify_grid(backend, grid_10, **CLASSIFY_KWARGS), repeats=2
    )
    per_pt = mean_10 / 100.0
    for size in [50, 100]:
        pts = size * size
        est_h = pts * per_pt / 3600.0
        print(f"  {size}×{size} grid ({pts} pts): ~{est_h:.1f} h  (single-process, no OpenMP)")


if __name__ == "__main__":
    _standalone()
