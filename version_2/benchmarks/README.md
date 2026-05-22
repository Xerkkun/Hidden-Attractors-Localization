# Benchmarks

Performance benchmarks for the three compute-heavy areas of the library.

## Structure

```
benchmarks/
├── conftest.py                        # Shared fixtures (backend, params, systems)
├── bench_efork_single_trajectory.py   # Single trajectory: Python vs C EFORK
├── bench_basin_grid.py                # Basin classification grids (5×5 … 20×20)
└── bench_seed_generation.py           # DF seed pipeline (scan, solve, Fourier)
```

## Quick start

```bash
# Install benchmark tooling
pip install pytest-benchmark

# Run all benchmarks (verbose, sorted by mean time)
python -m pytest benchmarks/ -v --benchmark-sort=mean

# Run one file only
python -m pytest benchmarks/bench_efork_single_trajectory.py -v

# Standalone (no pytest required — prints a summary table)
python benchmarks/bench_efork_single_trajectory.py
python benchmarks/bench_basin_grid.py
python benchmarks/bench_seed_generation.py
```

## Saving and comparing baselines

```bash
# Save current results as baseline
python -m pytest benchmarks/ --benchmark-save=baseline

# Compare against baseline after a code change
python -m pytest benchmarks/ --benchmark-compare=baseline
```

A regression is flagged when mean time increases by more than **15 %**.
Use `--benchmark-compare-fail=mean:15%` to enforce this in CI.

## What each benchmark measures

### `bench_efork_single_trajectory.py`

| Test | What it isolates |
|------|-----------------|
| `test_efork_q1_step` | Single EFORK Butcher step — pure arithmetic |
| `test_efork_integer_short` | Python loop, T=50 s (~10 000 steps) |
| `test_efork_integer_long` | Python loop, T=200 s (~40 000 steps) |
| `test_efork_c_short` | C backend, fractional q=0.9998, T=50 s |
| `test_efork_c_long` | C backend, fractional q=0.9998, T=200 s |

**Key ratio to watch**: C / Python speedup at the same trajectory length.
If it drops below 5×, something changed in the compilation flags.

### `bench_basin_grid.py`

| Test | Grid size | Points |
|------|-----------|--------|
| `test_classify_single_point` | 1×1 | 1 |
| `test_classify_single_point_t100` | 1×1 | 1 (longer traj) |
| `test_grid_5x5` | 5×5 | 25 |
| `test_grid_10x10` | 10×10 | 100 |
| `test_grid_20x20` | 20×20 | 400 |

The standalone script extrapolates timing to 50×50 and 100×100 production
grids so you can estimate realistic compute budgets.

### `bench_seed_generation.py`

| Test | Bottleneck |
|------|-----------|
| `test_frequency_scan_default` | Nyquist scan, nscan=20 000 |
| `test_frequency_scan_fast` | Nyquist scan, nscan=2 000 |
| `test_amplitude_solver_piecewise` | Grid+bisection amplitude solve |
| `test_full_harmonic_seed_classic` | End-to-end seed construction |
| `test_full_harmonic_seed_machado` | Machado-family DF variant |
| `test_fourier_coefficients` | Quadrature, n_quad=4 096 |
| `test_fourier_coefficients_fine` | Quadrature, n_quad=16 384 |
| `test_biased_lure_seed` | Full biased pipeline + least-squares |
| `test_lure_generic_seed` | Generic Lur'e wrappers |
| `test_q_sweep_seed_generation` | 10-value fractional-order sweep |

## Regression policy

> **A change that increases any benchmark mean by more than 15 % relative to the
> saved baseline must include a written justification in the PR description.**

Acceptable causes:
- Deliberately adding work (e.g. new validation steps).
- Improved numerical accuracy that requires more iterations.

Unacceptable causes:
- Accidental O(N²) loops, unneeded copies, import-time overhead.

## Platform notes

- C backend benchmarks require `gcc` in `PATH` (Windows) or `clang` (macOS).
- Set `ALLOW_NO_OPENMP=1` to build without OpenMP when the toolchain is
  unavailable; results will differ from OpenMP-enabled builds.
- Do not compare baselines across platforms — record the platform in the
  baseline name (e.g. `baseline_linux`, `baseline_windows`).
