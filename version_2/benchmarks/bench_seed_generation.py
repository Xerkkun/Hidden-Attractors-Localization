"""Synthetic timing checks for public describing-function utilities.

Round-number coefficients and compact numerical resolutions keep these
workloads independent of validation cases.  The checks measure implementation
cost only; generated states are not evidence for a scientific claim.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np


PERFORMANCE_ORDER = 1.0
BASE_SCAN_SIZE = 48
EXTENDED_SCAN_SIZE = 96
BASE_QUADRATURE_SIZE = 64
EXTENDED_QUADRATURE_SIZE = 96


def _time_callable(
    fn: Callable[[], object],
    *,
    repeats: int = 3,
) -> tuple[float, float]:
    """Return the minimum and mean elapsed time."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times), sum(times) / len(times)


def _spectral_input(parameters):
    """Build one deterministic input shared by several timing checks."""
    from hidden_attractors.seed_generation.chua import (
        find_omega_gain_candidates,
        solve_amplitude_from_gain,
    )

    pairs = find_omega_gain_candidates(
        PERFORMANCE_ORDER,
        parameters,
        nscan=BASE_SCAN_SIZE,
    )
    omega, gain = pairs[0]
    amplitude = solve_amplitude_from_gain(gain, parameters)
    return omega, gain, amplitude


def test_frequency_scan_base(benchmark, performance_parameters):
    """Measure a compact frequency scan."""
    from hidden_attractors.seed_generation.chua import (
        find_omega_gain_candidates,
    )

    pairs = benchmark(
        find_omega_gain_candidates,
        PERFORMANCE_ORDER,
        performance_parameters,
        nscan=BASE_SCAN_SIZE,
    )
    assert pairs
    assert all(omega > 0 and gain > 0 for omega, gain in pairs)


def test_frequency_scan_extended(benchmark, performance_parameters):
    """Measure a second compact frequency scan."""
    from hidden_attractors.seed_generation.chua import (
        find_omega_gain_candidates,
    )

    pairs = benchmark(
        find_omega_gain_candidates,
        PERFORMANCE_ORDER,
        performance_parameters,
        nscan=EXTENDED_SCAN_SIZE,
    )
    assert pairs


def test_amplitude_solver(benchmark, performance_parameters):
    """Measure the amplitude solver with a synthetic gain."""
    from hidden_attractors.seed_generation.chua import solve_amplitude_from_gain

    _, gain, _ = _spectral_input(performance_parameters)
    amplitude = benchmark(
        solve_amplitude_from_gain,
        gain,
        performance_parameters,
    )
    assert amplitude > 0.0


def test_harmonic_state_construction(benchmark, performance_parameters):
    """Measure the public harmonic-state construction path."""
    from hidden_attractors.seed_generation.chua import find_harmonic_seed

    state = benchmark(
        find_harmonic_seed,
        q=PERFORMANCE_ORDER,
        params=performance_parameters,
        method="classic",
        nscan=BASE_SCAN_SIZE,
    )
    assert state.amplitude > 0.0
    assert np.all(np.isfinite(state.seed))


def test_fourier_coefficients_base(benchmark, performance_parameters):
    """Measure compact Fourier coefficient quadrature."""
    from hidden_attractors.seed_generation.chua import fourier_coefficients_psi

    _, _, amplitude = _spectral_input(performance_parameters)
    coefficients = benchmark(
        fourier_coefficients_psi,
        amplitude,
        sigma0=0.0,
        params=performance_parameters,
        harmonics=4,
        n_quad=BASE_QUADRATURE_SIZE,
    )
    assert "y_mean" in coefficients
    assert 1 in coefficients["coefficients"]


def test_fourier_coefficients_extended(benchmark, performance_parameters):
    """Measure the same quadrature path at a second compact resolution."""
    from hidden_attractors.seed_generation.chua import fourier_coefficients_psi

    _, _, amplitude = _spectral_input(performance_parameters)
    coefficients = benchmark(
        fourier_coefficients_psi,
        amplitude,
        sigma0=0.0,
        params=performance_parameters,
        harmonics=4,
        n_quad=EXTENDED_QUADRATURE_SIZE,
    )
    assert "y_mean" in coefficients


def test_biased_state_construction(benchmark, performance_parameters):
    """Measure the public biased-state reconstruction path."""
    from hidden_attractors.seed_generation.chua import (
        reconstruct_biased_lure_seed,
    )

    omega, _, amplitude = _spectral_input(performance_parameters)
    state = benchmark(
        reconstruct_biased_lure_seed,
        q=PERFORMANCE_ORDER,
        params=performance_parameters,
        amplitude=amplitude,
        sigma0=0.0,
        omega=omega,
        harmonics=4,
        n_quad=BASE_QUADRATURE_SIZE,
    )
    assert np.all(np.isfinite(state.seed))


def _standalone() -> None:
    """Run the synthetic timing checks without pytest-benchmark."""
    from hidden_attractors.models.chua import ChuaParameters
    from hidden_attractors.seed_generation.chua import (
        find_harmonic_seed,
        find_omega_gain_candidates,
        fourier_coefficients_psi,
        reconstruct_biased_lure_seed,
        solve_amplitude_from_gain,
    )

    parameters = ChuaParameters(
        alpha=8.0,
        beta=12.0,
        gamma=0.1,
        m0=-0.2,
        m1=-1.0,
    )
    omega, gain, amplitude = _spectral_input(parameters)

    measurements: tuple[tuple[str, Callable[[], object]], ...] = (
        (
            "Frequency scan",
            lambda: find_omega_gain_candidates(
                PERFORMANCE_ORDER,
                parameters,
                nscan=BASE_SCAN_SIZE,
            ),
        ),
        (
            "Amplitude solver",
            lambda: solve_amplitude_from_gain(gain, parameters),
        ),
        (
            "Harmonic state",
            lambda: find_harmonic_seed(
                q=PERFORMANCE_ORDER,
                params=parameters,
                method="classic",
                nscan=BASE_SCAN_SIZE,
            ),
        ),
        (
            "Fourier coefficients",
            lambda: fourier_coefficients_psi(
                amplitude,
                sigma0=0.0,
                params=parameters,
                harmonics=4,
                n_quad=BASE_QUADRATURE_SIZE,
            ),
        ),
        (
            "Biased state",
            lambda: reconstruct_biased_lure_seed(
                q=PERFORMANCE_ORDER,
                params=parameters,
                amplitude=amplitude,
                sigma0=0.0,
                omega=omega,
                harmonics=4,
                n_quad=BASE_QUADRATURE_SIZE,
            ),
        ),
    )

    print("Synthetic describing-function performance checks")
    for label, operation in measurements:
        minimum, mean = _time_callable(operation)
        print(f"{label:<24} min={minimum * 1e3:8.2f} ms mean={mean * 1e3:8.2f} ms")


if __name__ == "__main__":
    _standalone()
