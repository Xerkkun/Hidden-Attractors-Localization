"""Synthetic timing checks for single-trajectory EFORK integration.

The Python path uses a stable linear right-hand side.  The native path uses
the neutral coefficients from ``conftest.py``.  These compact workloads detect
software-performance regressions only; their outputs are not evidence about
chaos, hiddenness, or a physical circuit.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np


PERFORMANCE_ORDER = 0.9
PERFORMANCE_STEP = 0.02
PERFORMANCE_MEMORY = 0.5
BASE_DURATION = 0.4
EXTENDED_DURATION = 0.8
INITIAL_STATE = np.array([0.2, -0.1, 0.05])
LINEAR_OPERATOR = np.array(
    [
        [-0.4, 0.1, 0.0],
        [0.0, -0.3, 0.1],
        [0.1, 0.0, -0.2],
    ]
)


def _linear_rhs(state: np.ndarray) -> np.ndarray:
    """Return a deterministic stable linear derivative."""
    return LINEAR_OPERATOR @ state


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


def test_efork_q1_step(benchmark):
    """Measure one pure-Python EFORK step."""
    from hidden_attractors.solvers import efork_q1_step

    result = benchmark(
        efork_q1_step,
        _linear_rhs,
        INITIAL_STATE.copy(),
        PERFORMANCE_STEP,
    )
    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_efork_integer_base(benchmark):
    """Measure a compact pure-Python trajectory."""
    from hidden_attractors.solvers import efork_q1_integrate

    def run():
        return efork_q1_integrate(
            _linear_rhs,
            INITIAL_STATE.copy(),
            t_final=BASE_DURATION,
            h=PERFORMANCE_STEP,
        )

    trajectory, status = benchmark(run)
    expected_rows = int(math.ceil(BASE_DURATION / PERFORMANCE_STEP)) + 1
    assert trajectory.shape == (expected_rows, 4)
    assert status == "ok"


def test_efork_integer_extended(benchmark):
    """Measure the same Python path with a second compact duration."""
    from hidden_attractors.solvers import efork_q1_integrate

    def run():
        return efork_q1_integrate(
            _linear_rhs,
            INITIAL_STATE.copy(),
            t_final=EXTENDED_DURATION,
            h=PERFORMANCE_STEP,
        )

    trajectory, status = benchmark(run)
    assert trajectory.shape[1] == 4
    assert status == "ok"


def test_native_fractional_base(benchmark, frac_backend):
    """Measure the compact native fractional path."""

    def run():
        return frac_backend.integrate_efork3(
            INITIAL_STATE,
            q=PERFORMANCE_ORDER,
            h=PERFORMANCE_STEP,
            Lm=PERFORMANCE_MEMORY,
            t_final=BASE_DURATION,
        )

    trajectory = benchmark(run)
    assert trajectory.ndim == 2
    assert trajectory.shape[1] == 4
    assert np.all(np.isfinite(trajectory))


def test_native_fractional_extended(benchmark, frac_backend):
    """Measure the native path with a second compact duration."""

    def run():
        return frac_backend.integrate_efork3(
            INITIAL_STATE,
            q=PERFORMANCE_ORDER,
            h=PERFORMANCE_STEP,
            Lm=PERFORMANCE_MEMORY,
            t_final=EXTENDED_DURATION,
        )

    trajectory = benchmark(run)
    assert trajectory.ndim == 2
    assert trajectory.shape[1] == 4
    assert np.all(np.isfinite(trajectory))


def _standalone() -> None:
    """Run the synthetic timing checks without pytest-benchmark."""
    from hidden_attractors.models.chua import ChuaParameters
    from hidden_attractors.native.backends import FractionalChuaBackend
    from hidden_attractors.solvers import efork_q1_integrate, efork_q1_step

    backend = FractionalChuaBackend.build()
    backend.set_params(
        ChuaParameters(
            alpha=8.0,
            beta=12.0,
            gamma=0.1,
            m0=-0.2,
            m1=-1.0,
        )
    )

    measurements: tuple[tuple[str, Callable[[], object]], ...] = (
        (
            "Python single step",
            lambda: efork_q1_step(
                _linear_rhs,
                INITIAL_STATE.copy(),
                PERFORMANCE_STEP,
            ),
        ),
        (
            "Python base trajectory",
            lambda: efork_q1_integrate(
                _linear_rhs,
                INITIAL_STATE.copy(),
                t_final=BASE_DURATION,
                h=PERFORMANCE_STEP,
            ),
        ),
        (
            "Python extended trajectory",
            lambda: efork_q1_integrate(
                _linear_rhs,
                INITIAL_STATE.copy(),
                t_final=EXTENDED_DURATION,
                h=PERFORMANCE_STEP,
            ),
        ),
        (
            "Native base trajectory",
            lambda: backend.integrate_efork3(
                INITIAL_STATE,
                q=PERFORMANCE_ORDER,
                h=PERFORMANCE_STEP,
                Lm=PERFORMANCE_MEMORY,
                t_final=BASE_DURATION,
            ),
        ),
        (
            "Native extended trajectory",
            lambda: backend.integrate_efork3(
                INITIAL_STATE,
                q=PERFORMANCE_ORDER,
                h=PERFORMANCE_STEP,
                Lm=PERFORMANCE_MEMORY,
                t_final=EXTENDED_DURATION,
            ),
        ),
    )

    print("Synthetic EFORK performance checks")
    for label, operation in measurements:
        minimum, mean = _time_callable(operation)
        print(f"{label:<32} min={minimum * 1e3:8.2f} ms mean={mean * 1e3:8.2f} ms")


if __name__ == "__main__":
    _standalone()
