# `hidden-attractors-fo`

`hidden-attractors-fo` provides two independent capabilities:

- numerical localization workflows for compatible dynamical systems under
  explicit evidence contracts; and
- direct characterization of systems, trajectories, and scalar time series.

The characterization API can be used without searching for a hidden
attractor. It includes trajectory and boundedness metrics, FFT/PSD summaries,
the 0--1 test, Poincare crossings, bifurcation post-processing, Lyapunov
calculations, integer `q=1` SALI/GALI/LDI, and integer covariant Lyapunov
vectors with angle diagnostics.

Version 1.1.0 includes the supported scalar time-series Lyapunov interface:
Rosenstein's largest-exponent estimate, an Eckmann reconstructed spectrum, and
the associated Kaplan--Yorke dimension through the optional `nolds` backend.

All numerical outputs remain conditional on their solver, sampling, memory,
transient, embedding, estimator, and neighborhood-control settings.

## Start here

- [Installation](installation.md)
- [Quick Start](quick_start.md)
- [Getting Started](getting_started.md)
- [Supported Examples](examples_index.md)
- [Systems](systems.md)
- [Public Workflows](workflows.md)
- [Adapting New Systems](adapting_new_systems.md)

## Analysis and evidence

- [Dynamical Analysis](dynamical_analysis.md)
- [Lyapunov Methods](lyapunov_methods.md)
- [Integer Covariant Lyapunov Vectors](covariant_lyapunov_vectors.md)
- [Poincare Diagnostics](poincare_diagnostics.md)
- [Scientific Scope](scientific_scope.md)
- [Validation Boundary](validation_evidence.md)
- [Validation Methodology](validation_methodology.md)
- [Integer Chua `q=1` Reference](integer_chua_reference.md)

## Reference

- [API Reference](api_reference.md)
- [API Stability Tiers](api_stability.md)
- [Public Calculation Reference Map](code_reference_map.md)
- [Optional External Tools](external_tools.md)
- [Testing](testing.md)
- [Citation](citation.md)

## En español

La biblioteca permite caracterizar sistemas, trayectorias y series temporales
sin iniciar una búsqueda de atractores ocultos. Las estimaciones son
numéricas y conservan los parámetros que delimitan su interpretación. Las
páginas enlazadas arriba describen las interfaces soportadas, los límites de
evidencia y las validaciones completadas.
