"""Synthetic timing checks for the native basin-classification interface.

Small deterministic point batches exercise dispatch and classification cost.
They are performance fixtures rather than validation data, and the labels
returned here must not be interpreted as a dynamical-system result.
"""

from __future__ import annotations

import itertools
import time
from typing import Callable, Sequence

import numpy as np


PERFORMANCE_ORDER = 0.9
PERFORMANCE_STEP = 0.02
PERFORMANCE_MEMORY = 0.5
BASE_DURATION = 0.4
EXTENDED_DURATION = 0.8

CLASSIFY_OPTIONS = {
    "q": PERFORMANCE_ORDER,
    "h": PERFORMANCE_STEP,
    "Lm": PERFORMANCE_MEMORY,
    "t_final": BASE_DURATION,
    "t_burn": 0.1,
    "divergence_norm": 50.0,
    "r_bound": 8.0,
    "equilibrium_tol": 0.01,
    "cap_win": 8,
    "mean_x_gap": 0.2,
}


def _make_point_batch(side: int) -> list[tuple[float, float, float]]:
    """Return a compact deterministic batch of neutral initial states."""
    coordinates = np.linspace(-0.2, 0.2, side)
    return [
        (float(first), 0.0, float(third))
        for first, third in itertools.product(coordinates, coordinates)
    ]


def _classify_batch(
    backend,
    points: Sequence[Sequence[float]],
    **options,
) -> list[int]:
    """Classify each point through the public native wrapper."""
    return [backend.classify_point(point, **options) for point in points]


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


def test_classify_single_point(benchmark, basin_backend):
    """Measure one native classification call."""
    result = benchmark(
        basin_backend.classify_point,
        (0.1, 0.0, -0.1),
        **CLASSIFY_OPTIONS,
    )
    assert 0 <= result <= 5


def test_classify_single_point_extended(benchmark, basin_backend):
    """Measure the same call with a second compact duration."""
    options = {**CLASSIFY_OPTIONS, "t_final": EXTENDED_DURATION}
    result = benchmark(
        basin_backend.classify_point,
        (0.1, 0.0, -0.1),
        **options,
    )
    assert 0 <= result <= 5


def test_classify_point_batch_base(benchmark, basin_backend):
    """Measure a small point batch."""
    points = _make_point_batch(2)

    def run():
        return _classify_batch(basin_backend, points, **CLASSIFY_OPTIONS)

    labels = benchmark(run)
    assert len(labels) == len(points)
    assert all(0 <= label <= 5 for label in labels)


def test_classify_point_batch_extended(benchmark, basin_backend):
    """Measure a second small point batch."""
    points = _make_point_batch(3)

    def run():
        return _classify_batch(basin_backend, points, **CLASSIFY_OPTIONS)

    labels = benchmark(run)
    assert len(labels) == len(points)
    assert all(0 <= label <= 5 for label in labels)


def _standalone() -> None:
    """Run the synthetic timing checks without pytest-benchmark."""
    from hidden_attractors.models.chua import ChuaParameters
    from hidden_attractors.native.backends import BasinBackend

    backend = BasinBackend.build()
    backend.set_params(
        ChuaParameters(
            alpha=8.0,
            beta=12.0,
            gamma=0.1,
            m0=-0.2,
            m1=-1.0,
        )
    )
    point = (0.1, 0.0, -0.1)
    points = _make_point_batch(3)

    measurements: tuple[tuple[str, Callable[[], object]], ...] = (
        (
            "Single point",
            lambda: backend.classify_point(point, **CLASSIFY_OPTIONS),
        ),
        (
            "Compact point batch",
            lambda: _classify_batch(backend, points, **CLASSIFY_OPTIONS),
        ),
    )

    print("Synthetic classifier performance checks")
    for label, operation in measurements:
        minimum, mean = _time_callable(operation)
        print(f"{label:<24} min={minimum * 1e3:8.2f} ms mean={mean * 1e3:8.2f} ms")


if __name__ == "__main__":
    _standalone()
