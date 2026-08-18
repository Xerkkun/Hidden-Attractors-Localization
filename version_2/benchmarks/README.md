# Benchmarks

These benchmarks measure implementation performance. They are engineering
checks, not scientific-validation records, and their timings do not support
claims about chaos or hiddenness.

## Benchmark groups

```text
benchmarks/
├── conftest.py
├── bench_efork_single_trajectory.py
├── bench_fractional_gl_kernels.py
├── bench_correlation_sum.py
├── bench_permutation_entropy.py
├── bench_multi_term_caputo.py
├── bench_alignment_indices.py
├── bench_covariant_lyapunov.py
├── bench_tempered_convolution_quadrature.py
├── bench_basin_grid.py
└── bench_seed_generation.py
```

- `bench_efork_single_trajectory.py` compares supported trajectory backends.
- `bench_fractional_gl_kernels.py` compares warmed Numba, native-C/OpenMP and
  offline FFT sampled-data GL operators; build and JIT warm-up are reported
  separately.
- `bench_correlation_sum.py` compares the exact same finite q=2 counts through
  Python, warmed Numba and native C/OpenMP, with cyclic backend ordering and
  host/build metadata.
- `bench_permutation_entropy.py` compares identical Bandt--Pompe ordinal
  histograms through Python, warmed Numba and native C/OpenMP and records a
  host-local assessment of the automatic backend threshold.
- `bench_multi_term_caputo.py` separates semantic-facade overhead, exact
  duplicate-order coalescence during combined-L1 construction, and the shared
  `O(N^2*d)` history sweep; every timing ratio is gated by numerical parity.
- `bench_alignment_indices.py` compares identical deterministic tangent
  histories through NumPy/SVD and warmed Numba/Householder, records JIT cost
  separately, and gates every timing row by SALI/GALI/log-volume parity.
- `bench_covariant_lyapunov.py` generates deterministic, dynamically consistent
  positive-diagonal Q/R histories, checks NumPy/Numba parity before timing, and
  measures both public-history reconstruction and the complete integer-map CLV
  pipeline with the first Numba call reported separately.
- `bench_tempered_convolution_quadrature.py` parity-gates Python, warmed Numba,
  and offline FFT for tempered RL/Caputo with BDF1/BDF2; FFT remains a batch
  convolution and is not labelled fast history or an FDE solver.
- `bench_tempered_fast_history.py` parity-gates recurrent FBDF1/GNGF2
  Python/Numba against exact-weight direct and offline FFT baselines, times
  automatic `Q` selection separately, and reports active-history memory.
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
python benchmarks/bench_fractional_gl_kernels.py --repeats 11 --output validation/outputs/benchmarks/fractional_gl_kernels_2026-08-02.json
python benchmarks/bench_correlation_sum.py --repeats 7 --output benchmark_outputs/hafo_correlation_sum_benchmark.json
python benchmarks/bench_permutation_entropy.py --repeats 5 --output benchmark_outputs/hafo_permutation_entropy_benchmark.json
python benchmarks/bench_multi_term_caputo.py --repeats 7 --output benchmark_outputs/hafo_multi_term_caputo_benchmark.json
python benchmarks/bench_alignment_indices.py --repeats 7 --output validation/outputs/benchmarks/alignment_indices_numpy_numba_20260803.json
python benchmarks/bench_covariant_lyapunov.py --repeats 7 --output benchmark_outputs/hafo_covariant_lyapunov_benchmark.json
python benchmarks/bench_tempered_convolution_quadrature.py --repeats 3 --output validation/outputs/benchmarks/tempered_convolution_quadrature_backends_20260803.json
python benchmarks/bench_tempered_fast_history.py --repeats 3 --output benchmark_outputs/hafo_tempered_fast_history_benchmark.json
python benchmarks/bench_basin_grid.py
python benchmarks/bench_seed_generation.py
```

Retain every new JSON with its environment, compiler/OpenMP, kernel hash,
warm-up, repetitions and dispersion fields. This checkout does not retain the
historical correlation-sum or permutation-entropy outputs, so their old timing
figures are not current evidence. Do not generalize any local crossover or
speedup to another machine without a new run.

For the historical CLV run on this host, warmed Numba reduced the largest complete
integer-map timing by about 52.8% relative to NumPy. The entire public Numba
reconstruction phase represented about 1.60% of that warmed end-to-end time;
even a hypothetical zero-cost replacement gives an idealized ceiling of only
about 1.016x. This evidence supports retaining NumPy and Numba rather than
adding a native-C CLV backend now. It does not prove that C can never help:
profile representative application systems first, then require parity-gated,
repeated end-to-end C measurements before changing the backend policy.
The retained JSON embeds the hash of an earlier benchmark-script revision, so
these figures do not characterize the current checkout without a rerun.

For the retained tempered-CQ run, FFT had the smallest median in 12/12 finite
definition/BDF/workload cases, while Numba matched Python exactly and the
largest FFT difference was `1.95e-13`. FFT was only `1.01x`--`1.30x` faster
than warmed Numba across these workloads. No C or Julia candidate was
implemented or timed, so the decision is to retain Python/Numba/FFT and admit a
new language backend only after an independently verified end-to-end candidate
beats run-to-run noise on representative HAFO/Toolbox workloads.

For the historical Fast Method II run, all 12 definition/generator/workload cases
passed parity before timing. Automatic calibration selected `Q=65` for
`N=128,512` and `Q=129` for `N=2048`. At `N=2048`, the evaluator's analytical
active-history model was `14,016 B`, versus `131,072 B` for complete base plus
tempered weight arrays. In these finite batch workloads, direct Numba or offline
FFT had the smallest median in every case; the recurrent call still preserves
the distinct compressed-history contract. Neither C nor Julia was implemented
or measured, so the current decision is not to add either backend without a
parity-gated end-to-end candidate on representative HAFO/Toolbox workloads.
Its retained JSON likewise embeds an earlier script hash and must not be read as
a benchmark of the current checkout.

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
