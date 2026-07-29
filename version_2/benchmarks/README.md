# Benchmarks

These benchmarks measure implementation performance. They are engineering
checks, not scientific-validation records, and their timings do not support
claims about chaos or hiddenness.

## Benchmark groups

```text
benchmarks/
├── conftest.py
├── bench_efork_single_trajectory.py
├── bench_basin_grid.py
└── bench_seed_generation.py
```

- `bench_efork_single_trajectory.py` compares supported trajectory backends.
- `bench_basin_grid.py` measures basin-classification workload scaling.
- `bench_seed_generation.py` measures the public describing-function seed
  components.

The benchmark fixtures define representative workloads only. Consult the
generated benchmark metadata before comparing two runs.

## Run benchmarks

```bash
python -m pip install pytest-benchmark
python -m pytest benchmarks/ -v --benchmark-sort=mean
python -m pytest benchmarks/bench_efork_single_trajectory.py -v
```

The benchmark files may also be executed directly when they provide a
standalone entry point:

```bash
python benchmarks/bench_efork_single_trajectory.py
python benchmarks/bench_basin_grid.py
python benchmarks/bench_seed_generation.py
```

## Save and compare a baseline

```bash
python -m pytest benchmarks/ --benchmark-save=baseline
python -m pytest benchmarks/ --benchmark-compare=baseline
```

Compare only equivalent environments and numerical contracts. Record the
Python version, platform, processor, backend, dependency versions, and
benchmark commit with every retained baseline. A timing change should be
investigated before it is treated as a regression; numerical correctness
remains covered by the test suite rather than inferred from speed.

## Platform notes

- Native-backend benchmarks require a supported compiler.
- Results from different platforms or compiler settings are not directly
  comparable.
- Benchmark baselines are local engineering artifacts unless explicitly
  attached to a reproducible software-performance record.
