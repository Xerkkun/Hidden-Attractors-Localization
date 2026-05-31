# Fractional Variational ABM-QR Lyapunov Exponents Benchmarks

This directory contains benchmark specifications (in YAML format) for verifying the correctness of the `fractional_variational_abm_qr` Lyapunov exponent estimation method.

## Directory Structure

- `synthetic_zero_rhs.yaml`: Trivial zero system verifying convergence to exactly zero exponents.
- `synthetic_linear_stable.yaml`: Stable linear diagonal system verifying convergence to non-positive exponents.
- `published_danca_kuznetsov2018_template.yaml`: Template for future validation against Danca & Kuznetsov (2018), marked as pending due to missing reference data.

## Schema Details

Each benchmark YAML must specify the following blocks:
1. `case_id`: Unique identifier.
2. `benchmark_type`: `"synthetic"` or `"published"`.
3. `method_id`: `"fractional_variational_abm_qr"`.
4. `system`: System formulation, dimension, parameters, Caputo order `q`, and initial state `x0`.
5. `integration`: Step size `h`, times, and history-aware parameters.
6. `expected`: Exponents, tolerances, or qualitative sign patterns.
7. `reference`: Authors, DOI, notes, and completeness flags.
