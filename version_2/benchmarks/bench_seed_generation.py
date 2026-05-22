"""
benchmarks/bench_seed_generation.py
=====================================
Performance benchmarks for the describing-function seed-generation pipeline.

What is measured
----------------
- **Frequency scan** (``find_omega_gain_candidates``): the Nyquist/DF
  frequency sweep.  Cost scales with ``nscan`` and dominates the seed pipeline.
- **Amplitude solver** (``solve_amplitude_from_gain``): bisection over
  ``N(A) = k``.  Cheap in the piecewise case; more expensive for arctan.
- **Full seed construction** (``find_harmonic_seed``): end-to-end from ``q``
  to a ``HarmonicSeed`` object (scan + solve + eigenvector).
- **Fourier coefficients** (``fourier_coefficients_psi``): the quadrature
  inner loop used by the biased DF.  Costs scales with ``n_quad``.
- **Biased seed** (``reconstruct_biased_lure_seed``): full biased pipeline
  including two least-squares solves.
- **Lur'e generic path** (``find_lure_harmonic_seed``): exercises the
  system-independent wrappers.

Why this matters
----------------
Seed generation is called once per candidate ``q`` value.  A parameter sweep
over many fractional orders can easily require 10 000+ seed evaluations.  The
Nyquist scan (``nscan=20_000``) is the dominant cost; ``n_quad=4096`` for the
Fourier coefficients is secondary but non-trivial on slow machines.

Running
-------
    python -m pytest benchmarks/bench_seed_generation.py -v
    python benchmarks/bench_seed_generation.py           # standalone
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

# ── Canonical parameters ──────────────────────────────────────────────────────
Q = 0.9998
NSCAN_DEFAULT = 20_000
NSCAN_FAST = 2_000     # used for "fast" fixture to separate scan cost from rest
N_QUAD = 4096


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _time_callable(fn: Callable, repeats: int = 5) -> tuple[float, float]:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


# ─────────────────────────────────────────────────────────────────────────────
# pytest-benchmark tests
# ─────────────────────────────────────────────────────────────────────────────

def test_frequency_scan_default(benchmark, chua_params):
    """Nyquist/DF frequency scan — nscan=20 000 (production default)."""
    from hidden_attractors.seed_generation.chua import find_omega_gain_candidates

    def _run():
        return find_omega_gain_candidates(Q, chua_params, nscan=NSCAN_DEFAULT)

    pairs = benchmark(_run)
    assert len(pairs) >= 1
    assert all(omega > 0 and gain > 0 for omega, gain in pairs)


def test_frequency_scan_fast(benchmark, chua_params):
    """Nyquist/DF frequency scan — nscan=2 000 (fast path for CI)."""
    from hidden_attractors.seed_generation.chua import find_omega_gain_candidates

    def _run():
        return find_omega_gain_candidates(Q, chua_params, nscan=NSCAN_FAST)

    pairs = benchmark(_run)
    assert len(pairs) >= 1


def test_amplitude_solver_piecewise(benchmark, chua_params):
    """Amplitude bisection for the piecewise Chua DF."""
    from hidden_attractors.seed_generation.chua import (
        find_omega_gain_candidates,
        solve_amplitude_from_gain,
    )

    pairs = find_omega_gain_candidates(Q, chua_params, nscan=NSCAN_FAST)
    _, gain = pairs[0]

    def _run():
        return solve_amplitude_from_gain(gain, chua_params)

    amp = benchmark(_run)
    assert amp > 0.0


def test_full_harmonic_seed_classic(benchmark, chua_params):
    """End-to-end ``find_harmonic_seed`` — classic DF, nscan=20 000."""
    from hidden_attractors.seed_generation.chua import find_harmonic_seed

    def _run():
        return find_harmonic_seed(q=Q, params=chua_params, method="classic")

    seed = benchmark(_run)
    assert seed.amplitude > 0
    assert np.all(np.isfinite(seed.seed))


def test_full_harmonic_seed_machado(benchmark, chua_params):
    """End-to-end ``find_harmonic_seed`` — Machado DF (mu=1.5)."""
    from hidden_attractors.seed_generation.chua import find_harmonic_seed

    def _run():
        return find_harmonic_seed(
            q=Q, params=chua_params, method="machado", mu=1.5
        )

    seed = benchmark(_run)
    assert seed.amplitude > 0
    assert np.all(np.isfinite(seed.seed))


def test_fourier_coefficients(benchmark, chua_params):
    """Fourier coefficient quadrature — n_quad=4096, harmonics=10."""
    from hidden_attractors.seed_generation.chua import (
        find_omega_gain_candidates,
        fourier_coefficients_psi,
        solve_amplitude_from_gain,
    )

    pairs = find_omega_gain_candidates(Q, chua_params, nscan=NSCAN_FAST)
    _, gain = pairs[0]
    amp = solve_amplitude_from_gain(gain, chua_params)

    def _run():
        return fourier_coefficients_psi(
            amp, sigma0=0.0, params=chua_params, harmonics=10, n_quad=N_QUAD
        )

    coeffs = benchmark(_run)
    assert "y_mean" in coeffs
    assert 1 in coeffs["coefficients"]


def test_fourier_coefficients_fine(benchmark, chua_params):
    """Fourier coefficient quadrature — n_quad=16 384, harmonics=20 (fine)."""
    from hidden_attractors.seed_generation.chua import (
        find_omega_gain_candidates,
        fourier_coefficients_psi,
        solve_amplitude_from_gain,
    )

    pairs = find_omega_gain_candidates(Q, chua_params, nscan=NSCAN_FAST)
    _, gain = pairs[0]
    amp = solve_amplitude_from_gain(gain, chua_params)

    def _run():
        return fourier_coefficients_psi(
            amp, sigma0=0.0, params=chua_params, harmonics=20, n_quad=16_384
        )

    coeffs = benchmark(_run)
    assert "y_mean" in coeffs


def test_biased_lure_seed(benchmark, chua_params):
    """Full biased seed pipeline — two least-squares solves + quadrature."""
    from hidden_attractors.seed_generation.chua import (
        find_omega_gain_candidates,
        reconstruct_biased_lure_seed,
        solve_amplitude_from_gain,
    )

    pairs = find_omega_gain_candidates(Q, chua_params, nscan=NSCAN_FAST)
    omega, gain = pairs[0]
    amp = solve_amplitude_from_gain(gain, chua_params)

    def _run():
        return reconstruct_biased_lure_seed(
            q=Q,
            params=chua_params,
            amplitude=amp,
            sigma0=0.0,
            omega=omega,
        )

    seed = benchmark(_run)
    assert np.all(np.isfinite(seed.seed))


def test_lure_generic_seed(benchmark, chua_system):
    """Generic Lur'e ``find_lure_harmonic_seed`` via LureSystem wrapper."""
    from hidden_attractors.seed_generation.lure import find_lure_harmonic_seed

    def _run():
        return find_lure_harmonic_seed(q=Q, system=chua_system, method="classic")

    seed = benchmark(_run)
    assert seed.amplitude > 0
    assert np.all(np.isfinite(seed.seed))


def test_q_sweep_seed_generation(benchmark, chua_params):
    """Seed generation over 10 fractional orders — simulates a parameter scan."""
    from hidden_attractors.seed_generation.chua import find_harmonic_seed

    q_values = np.linspace(0.990, 0.9998, 10)

    def _run():
        seeds = []
        for q in q_values:
            try:
                s = find_harmonic_seed(
                    q=float(q), params=chua_params,
                    method="classic", nscan=NSCAN_FAST,
                )
                seeds.append(s)
            except (RuntimeError, IndexError):
                pass  # no candidate at this q — expected for some values
        return seeds

    seeds = benchmark(_run)
    assert len(seeds) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Standalone
# ─────────────────────────────────────────────────────────────────────────────

def _standalone():
    from hidden_attractors.models.chua import chua_piecewise_parameters
    from hidden_attractors.seed_generation.chua import (
        find_harmonic_seed,
        find_omega_gain_candidates,
        fourier_coefficients_psi,
        reconstruct_biased_lure_seed,
        solve_amplitude_from_gain,
    )

    params = chua_piecewise_parameters()
    print("\n=== Seed-generation benchmarks ===\n")

    # 1. Frequency scan
    lo, mean = _time_callable(
        lambda: find_omega_gain_candidates(Q, params, nscan=NSCAN_DEFAULT), repeats=5
    )
    print(f"  frequency_scan  nscan={NSCAN_DEFAULT:<6d}  min={lo*1e3:.1f} ms  mean={mean*1e3:.1f} ms")

    lo_fast, mean_fast = _time_callable(
        lambda: find_omega_gain_candidates(Q, params, nscan=NSCAN_FAST), repeats=5
    )
    print(f"  frequency_scan  nscan={NSCAN_FAST:<6d}  min={lo_fast*1e3:.1f} ms  mean={mean_fast*1e3:.1f} ms")
    print(f"  scan speedup {NSCAN_DEFAULT}→{NSCAN_FAST}: {mean/mean_fast:.1f}×")

    # 2. Amplitude solver
    pairs = find_omega_gain_candidates(Q, params, nscan=NSCAN_FAST)
    _, gain = pairs[0]
    lo, mean = _time_callable(
        lambda: solve_amplitude_from_gain(gain, params), repeats=10
    )
    print(f"\n  amplitude_solver (piecewise)               min={lo*1e3:.2f} ms  mean={mean*1e3:.2f} ms")

    # 3. Full harmonic seed
    lo, mean = _time_callable(
        lambda: find_harmonic_seed(q=Q, params=params, method="classic"), repeats=5
    )
    print(f"  find_harmonic_seed (classic)               min={lo*1e3:.1f} ms  mean={mean*1e3:.1f} ms")

    # 4. Fourier coefficients
    amp = solve_amplitude_from_gain(gain, params)
    lo, mean = _time_callable(
        lambda: fourier_coefficients_psi(
            amp, sigma0=0.0, params=params, harmonics=10, n_quad=N_QUAD
        ),
        repeats=10,
    )
    print(f"  fourier_coefficients  n_quad={N_QUAD}       min={lo*1e3:.1f} ms  mean={mean*1e3:.1f} ms")

    # 5. Biased seed
    omega, _ = pairs[0]
    lo, mean = _time_callable(
        lambda: reconstruct_biased_lure_seed(
            q=Q, params=params, amplitude=amp, sigma0=0.0, omega=omega
        ),
        repeats=5,
    )
    print(f"  biased_lure_seed                           min={lo*1e3:.1f} ms  mean={mean*1e3:.1f} ms")

    # 6. q-sweep estimate
    print("\n--- q-sweep extrapolation ---")
    lo_seed, mean_seed = _time_callable(
        lambda: find_harmonic_seed(q=Q, params=params, nscan=NSCAN_FAST), repeats=5
    )
    for n_q in [100, 1_000, 10_000]:
        est_s = n_q * mean_seed
        print(f"  {n_q:>6d} q values (fast scan): ~{est_s:.1f} s  ({est_s/60:.1f} min)")


if __name__ == "__main__":
    _standalone()
