# Fractional Variational ABM-QR Lyapunov Exponents Benchmarks

This directory contains local synthetic checks and explicit published
reference data for fractional variational Lyapunov contracts.

## Directory Structure

- `synthetic_zero_rhs.yaml`: Trivial zero system verifying convergence to exactly zero exponents.
- `synthetic_linear_stable.yaml`: Stable linear diagonal system verifying convergence to non-positive exponents.
- `published_danca_kuznetsov2018_template.yaml`: Template for future validation against Danca & Kuznetsov (2018), marked as pending due to missing reference data.
- `published_dk2018_rabinovich_fabrikant_q0999.yaml`: Quantitative RF reproduction case.
- `published_dk2018_lorenz_q0985.yaml`: Quantitative Lorenz reproduction case.
- `published_dk2018_4d_nonsmooth_q098_qualitative.yaml`: Qualitative-only 4D non-smooth reference.

## Schema Details

Each benchmark YAML must specify the following blocks:
1. `case_id`: Unique identifier.
2. `benchmark_type`: `"synthetic"` or `"published"`.
3. `method_id`: local full-history QR or explicit DK2018 block-restart ABM-GS lane.
4. `system`: System formulation, dimension, parameters, Caputo order `q`, and initial state `x0`.
5. `integration`: Step size `h`, times, and history-aware parameters.
6. `expected`: Exponents, tolerances, or qualitative sign patterns.
7. `reference`: Authors, DOI, notes, and completeness flags.

Published extensive calculations set `execution.native_required: true`.
Python orchestrates configuration and evidence output; the numerical loop,
RHS and Jacobian execute in C.
