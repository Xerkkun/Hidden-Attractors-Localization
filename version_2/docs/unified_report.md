# Version 1.1.0 Public Scope

The public documentation describes the supported package and the boundaries
of its completed validation records. Only those supported interfaces and
completed evidence boundaries are distributed as package documentation.

## Two independent entry routes

`hidden-attractors-fo` supports:

1. localization workflows for compatible dynamical systems under explicit
   numerical and neighborhood-control contracts; and
2. direct characterization of systems, trajectories, and scalar time series,
   whether or not localization is requested.

The characterization route includes:

- equilibria and Jacobians supplied by registered systems;
- boundedness and trajectory metrics;
- FFT and power-spectral-density summaries;
- the Gottwald--Melbourne 0--1 test;
- Poincare-section crossings;
- bifurcation post-processing for existing trajectories;
- equation-based Lyapunov calculations under registered method contracts; and
- scalar time-series Lyapunov estimates.

## Time-series Lyapunov integration

`estimate_time_series_lyapunov` is a supported public function. Through the
optional `nolds` backend it returns:

- Rosenstein's largest-exponent estimate;
- an Eckmann reconstructed finite spectrum; and
- the associated Kaplan--Yorke dimension.

The result also records sampling units, estimator parameters, backend
provenance, diagnostics, and warnings. The same operation is available from:

```bash
hidden-attractors lyapunov spectrum --help
```

These values are finite-data estimates. They do not by themselves establish
an asymptotic Lyapunov spectrum, chaos, or hiddenness.

## Evidence boundary

A seed, continuation path, bounded trajectory, positive finite-time exponent,
or negative neighborhood sample is not a global proof. Scientific
interpretation remains conditional on the exact solver, memory policy,
horizon, transient removal, sampling, equilibria, radii, classifier,
estimator, and numerical-failure policy recorded with the result.

Completed validation records are kept separately from the installed API.
Public calculations can be used on user-supplied systems and data without
claiming correspondence to any validation case.

The corrected non-smooth Chua record applies this rule explicitly: its local
claim stops at `r = 0.01` after 7,200 zero-contact, zero-failure ball samples.
The zero-contact samples at `r = 0.03` and `r = 0.1`, followed by 37 contacts
at `r = 0.3`, are retained as a separate macro-basin audit and do not enlarge
the local claim. The recorded trajectory is regular/periodic, so no chaos
claim is attached to this validation.

## Public references

- [Quick Start](quick_start.md)
- [Workflows](workflows.md)
- [Dynamical Analysis](dynamical_analysis.md)
- [Lyapunov Methods](lyapunov_methods.md)
- [Validation Boundary](validation_evidence.md)
- [Code Reference Map](code_reference_map.md)
- [API Reference](api_reference.md)
