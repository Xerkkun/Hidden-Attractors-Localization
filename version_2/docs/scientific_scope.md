# Scientific Scope

`hidden-attractors-fo` has two independent public uses:

1. numerical localization workflows for Lur'e-compatible systems; and
2. direct characterization of supplied systems, trajectories, and scalar time
   series without running an attractor-localization workflow.

## Independent characterization

Implemented tools include:

- equilibrium and Jacobian evaluation;
- integer and commensurate Caputo integration;
- boundedness and trajectory summaries;
- FFT/PSD spectral characteristics;
- Poincare sections;
- 0-1 diagnostics;
- bifurcation helpers;
- Lyapunov diagnostics for modeled trajectories;
- finite-time covariant Lyapunov vectors and angle diagnostics for
  memoryless integer `q=1` flows and maps; and
- Rosenstein/Eckmann Lyapunov estimates plus Kaplan--Yorke dimension for
  uniformly sampled scalar time series.

These functions compute numerical characteristics of data or dynamical
systems. They do not require a claim about an attractor's basin or hiddenness.

### Public characterization entry points

The following callables are exported directly from `hidden_attractors`. They
form the supported high-level characterization surface:

| Entry point | Numerical result |
| --- | --- |
| `compute_boundedness_metrics` | Finite-trajectory boundedness statistics and status |
| `compute_fft_psd` | FFT and power-spectral-density characteristics |
| `detect_poincare_crossings` | Oriented Poincare-section crossings |
| `zero_one_test` | Finite-data 0-1 statistic and diagnostic state |
| `bifurcation_points_from_trajectories` | Parameter-labelled extrema extracted from trajectories |
| `bifurcation_summary` | Structured summary of bifurcation points |
| `compute_trajectory_metrics` | Generic trajectory ranges, variation, distances, and diagnostics |
| `trajectory_metrics` | Array-based trajectory characteristics |
| `trajectory_metrics_for_system` | Trajectory characteristics using a registered system's equilibria |
| `hidden_attractors.analysis.integer_qr_benettin_lyapunov_exponents` | Lower-level integer QR/Benettin Lyapunov spectrum |
| `integer_system_lyapunov_exponents` | Integer-order equation-based Lyapunov spectrum |
| `validate_lyapunov_method_request` | Method, order, Jacobian, and memory-contract validation |
| `compute_lyapunov_spectrum` | Structured integer/fractional Lyapunov dispatch |
| `integer_covariant_vectors_from_qr_history` | Integer `q=1` Ginelli backward reconstruction from a validated QR history |
| `integer_flow_covariant_lyapunov_vectors`, `integer_map_covariant_lyapunov_vectors`, and `integer_system_covariant_lyapunov_vectors` | Finite-time integer `q=1` CLV histories for flows, maps, or compatible HAFO system objects |
| `covariant_lyapunov_angles` | Geometric pair and principal-subspace angles for a supplied vector history; it does not validate CLV provenance |
| `estimate_time_series_lyapunov` | Rosenstein/Eckmann estimates from a uniformly sampled scalar series |
| `kaplan_yorke_dimension` | Kaplan--Yorke dimension from a supplied exponent spectrum |

All return values are finite-data or finite-time numerical estimates. In
particular, no Lyapunov, spectral, Poincare, bifurcation, boundedness, or 0-1
result alone establishes chaos or hiddenness.

The CLV construction facades reject `q != 1`. A nonlocal fractional
derivative requires an operator-specific history-space tangent cocycle, norm,
and renormalization rule, so fractional CLV remains `research_required`.

### Systems and parameters

The stable system surface is also independent of hidden-attractor
localization. `get_system`, `list_systems`, and `register_system` manage
`ChaoticSystem` definitions; `requirements_for` and
`check_system_capability` report whether a registered system supplies the
mathematical inputs required by a selected workflow.

For the maintained integer Chua control, non-smooth Chua lane, and arctan
validation definition, `chua_parameters`, `chua_nonsmooth_parameters`, and
`chua_arctan_wu2023_parameters` return explicit parameter records.
`equilibria_nonsmooth`, `equilibria_arctan`, `jacobian_nonsmooth`,
`jacobian_arctan`, `rhs_nonsmooth`, and `rhs_arctan` evaluate their
corresponding model quantities. These calls can be used without starting a
localization or hiddenness protocol.

## Localization and verification

For Lur'e-compatible localization, describing-function and Nyquist
calculations construct seeds, continuation transports them, and integration
produces finite trajectories. Hiddenness requires a separate declared
sampled-neighborhood or basin contract around all relevant equilibria.

## Supported boundary and out of scope

The maintained fractional route covers commensurate Caputo systems with a
scalar Lur'e decomposition. ABM and EFORK integrations, Matignon equilibrium
checks, and the integer-order controls have distinct recorded contracts.
Other derivative definitions, noncommensurate orders, and systems without an
implemented compatible interface are out of scope for the current public
contract.

DF and Nyquist calculations produce a seed, not a hiddenness proof.
Continuation transports that seed; it does not promote a trajectory to a
hidden-attractor result. Visual phase portraits and other diagnostics remain
supporting evidence only.

## Evidence labels

Current classifications include `hidden_under_tested_neighborhoods` and
`compatible_with_hiddenness`. Both labels remain conditional on the recorded
finite sampling contract.

## Published reference coverage

Case-specific literature comparisons are validation records, not general
PyPI examples. The tagged repository stores their machine-readable coverage
in `validation/published_reference_coverage.json`. A record marked as a
partial reference implementation is not presented as a full reproduction or
as a reason to promote unrelated numerical evidence.

## Evidence boundary

Numerical diagnostics are finite-data or finite-time estimates. A bounded
trajectory, positive estimated exponent, broadband spectrum, dispersed
Poincare section, or 0-1 statistic does not alone establish hiddenness.
Sampled-neighborhood results also do not constitute a global mathematical
proof.

Every reproducible result should record the package version, method, numerical
parameters, sample interval, units, solver/backend, memory policy when
fractional, and output hashes.

The public wheel and sdist contain supported software and examples. Complete
validation records remain in the matching release tag and archived snapshot.
