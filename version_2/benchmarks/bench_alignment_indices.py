"""Benchmark NumPy/SVD and Numba/Householder SALI--GALI history analysis.

The same deterministic tangent histories and public API are used for both
backends. Numba compilation is measured separately and excluded from repeated
timings. Equality is checked before speed ratios are reported. Results are
host-specific software-engineering evidence, not evidence of chaos or of
universal backend superiority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np
import scipy

from hidden_attractors import alignment_indices_from_tangent_history


GALI_ORDERS = (2, 3, 4)
BACKENDS = ("numpy", "numba")


@dataclass(frozen=True, slots=True)
class Workload:
    name: str
    samples: int
    vectors: int
    dimension: int
    seed: int


WORKLOADS = (
    Workload("small_s64_k4_d8", 64, 4, 8, 2026080301),
    Workload("medium_s512_k4_d8", 512, 4, 8, 2026080302),
    Workload("large_s4096_k4_d8", 4096, 4, 8, 2026080303),
)


def _history(workload: Workload) -> np.ndarray:
    generator = np.random.default_rng(workload.seed)
    history = generator.standard_normal(
        (workload.samples, workload.vectors, workload.dimension)
    )
    return np.ascontiguousarray(history, dtype=np.float64)


def _run(history: np.ndarray, backend: str):
    return alignment_indices_from_tangent_history(
        history,
        gali_orders=GALI_ORDERS,
        backend=backend,
        q=1.0,
        method_id="alignment_indices_backend_benchmark",
    )


def _timed(history: np.ndarray, backend: str) -> tuple[float, object]:
    started = time.perf_counter_ns()
    result = _run(history, backend)
    return (time.perf_counter_ns() - started) * 1.0e-9, result


def _summary(samples: list[float]) -> dict[str, object]:
    values = np.asarray(samples, dtype=float)
    return {
        "samples_seconds": [float(value) for value in values],
        "minimum_seconds": float(np.min(values)),
        "median_seconds": float(np.median(values)),
        "mean_seconds": float(np.mean(values)),
        "population_stdev_seconds": float(np.std(values, ddof=0)),
        "q25_seconds": float(np.quantile(values, 0.25)),
        "q75_seconds": float(np.quantile(values, 0.75)),
    }


def run_benchmark(*, repeats: int = 7) -> dict[str, object]:
    if repeats < 3:
        raise ValueError("repeats must be at least 3")

    warmup_history = _history(Workload("jit_warmup", 4, 4, 8, 2026080300))
    compile_seconds, warmup = _timed(warmup_history, "numba")
    if warmup.backend != "numba":
        raise RuntimeError("the explicit Numba warm-up did not use Numba")

    records: list[dict[str, object]] = []
    worst_gali_difference = 0.0
    worst_log_gali_difference = 0.0
    for workload in WORKLOADS:
        history = _history(workload)
        reference = _run(history, "numpy")
        accelerated = _run(history, "numba")
        gali_difference = float(np.max(np.abs(reference.gali - accelerated.gali)))
        log_difference = float(
            np.max(np.abs(reference.log_gali - accelerated.log_gali))
        )
        if not np.array_equal(reference.censored, accelerated.censored):
            raise AssertionError(f"censoring mismatch for {workload.name}")
        if gali_difference > 5.0e-12 or log_difference > 5.0e-12:
            raise AssertionError(
                f"backend mismatch for {workload.name}: "
                f"gali={gali_difference}, log_gali={log_difference}"
            )
        worst_gali_difference = max(worst_gali_difference, gali_difference)
        worst_log_gali_difference = max(worst_log_gali_difference, log_difference)

        measurements: dict[str, list[float]] = {name: [] for name in BACKENDS}
        for repeat in range(repeats):
            order = BACKENDS if repeat % 2 == 0 else tuple(reversed(BACKENDS))
            for backend in order:
                elapsed, result = _timed(history, backend)
                if result.status != "ok" or result.backend != backend:
                    raise RuntimeError(
                        f"unexpected {backend} result for {workload.name}: "
                        f"status={result.status}, backend={result.backend}"
                    )
                measurements[backend].append(elapsed)

        timing = {name: _summary(values) for name, values in measurements.items()}
        numpy_median = float(timing["numpy"]["median_seconds"])
        numba_median = float(timing["numba"]["median_seconds"])
        records.append(
            {
                "workload": asdict(workload),
                "history_sha256": hashlib.sha256(history.tobytes()).hexdigest(),
                "gali_orders": list(GALI_ORDERS),
                "timing": timing,
                "numpy_over_numba_median_ratio": numpy_median / numba_median,
                "maximum_absolute_gali_difference": gali_difference,
                "maximum_absolute_log_gali_difference": log_difference,
            }
        )

    return {
        "benchmark_id": "hafo_alignment_indices_numpy_numba_20260803",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "finite deterministic public-API software backend benchmark",
        "evidence_boundary": (
            "host- and workload-specific engineering evidence only; no universal "
            "performance claim and no chaos, attraction, hiddenness, or "
            "fractional-memory validation"
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "numba": numba.__version__,
        },
        "numba_first_call_seconds_including_compilation": compile_seconds,
        "repeats": repeats,
        "backend_parity_tolerance": 5.0e-12,
        "worst_absolute_gali_difference": worst_gali_difference,
        "worst_absolute_log_gali_difference": worst_log_gali_difference,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(repeats=args.repeats)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output is None:
        print(rendered)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

