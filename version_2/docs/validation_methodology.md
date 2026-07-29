# Validation and Verification Methodology

The library separates numerical operations so that an initialization result,
a trajectory diagnostic, and a hiddenness assessment are not confused.

## Numerical evidence layers

1. **Seed generation** locates an initial condition through an explicit
   approximation such as a describing function or frequency scan.
2. **Target integration** checks whether the supplied initial condition
   produces a finite, bounded, persistent trajectory under the recorded
   numerical settings.
3. **Dynamical characterization** applies diagnostics such as spectra, the
   0--1 test, Poincare crossings, or finite-time Lyapunov estimates.
4. **Equilibrium-neighborhood controls** sample initial conditions around all
   recorded equilibria when a hiddenness assessment is requested.

Each layer reports only its own result. A seed is not an attractor proof, a
positive finite-time exponent is not an asymptotic chaos proof, and sampled
neighborhood controls are not an exhaustive basin proof.

## Operational status labels

Some machine-readable records retain status names such as `seed_found`,
`candidate_attractor`, `chaotic_candidate`, `hidden_compatible`, and
`hidden_verified`. These are operational labels interpreted through the
stored contract. In public summaries, the preferred hiddenness wording is
`hiddenness_supported_under_tested_neighborhoods` or
`compatible_with_hiddenness_under_tested_radii`.

Any such result must preserve the solver, dynamic order, step size, horizon,
memory policy, transient removal, equilibria, radii, samples, sampling mode,
classifier, and failure policy that bound the calculation.

## Local and extended neighborhoods

The operative sampled hiddenness check concerns sufficiently small
neighborhoods of every equilibrium. A contact detected only at a larger
radius describes sampled basin geometry outside the local contract; it does
not by itself establish that the attractor is self-excited. Local and
macro-radius results must therefore be stored and interpreted separately.

The sampling geometry is part of the evidence:

- an **interior ball** samples points throughout a declared neighborhood;
- a **spherical surface** samples only a fixed-distance boundary;
- a **spherical shell** samples a declared radial band.

A surface or shell result must not be reported as filled-ball coverage.
Concrete radii and sample allocations belong in the associated validation
record rather than in the public method description.

### Per-probe time and history

Every probe records its initial time and initial state. For a Caputo problem it
also records the history function and the interval on which that history is
defined. A fresh probe normally starts with its own declared history (for
example, a constant history equal to that probe state); it must not silently
reuse the history of a reference trajectory or another probe. Explicit history
transport is a different contract and must be identified as such.

### Predeclared classification and stopping

The target representation, comparison metric, classification threshold,
divergence threshold, and failure policy are fixed before neighborhood
sampling begins. They are not retuned after inspecting probe outcomes.

An ordered radial protocol may optionally use
`complete_first_contact_radius`: once a contact is observed, it completes every
planned probe at that radius across all equilibria and then omits larger radii.
This gives a complete denominator at the first contact radius while preserving
a causal stopping rule. The omitted radii remain untested.

## Fractional frequency and integration contracts

For `0 < q < 1`, a frequency scan evaluates the configured branch of

```text
(j omega)^q = omega^q exp(j q pi / 2)
```

The harmonic calculation supplies a seed approximation. Validation of the
trajectory requires an explicit fractional integrator and memory policy.
Full-history, finite-window, and local recurrence methods are different
numerical contracts and must not be merged.

## Lur'e compatibility

Describing-function routes require an explicit scalar Lur'e representation.
`LureCompatibilityValidator` reports one of:

- `LURE_DIRECT`
- `LURE_LINEAR_CHANGE`
- `LURE_APPROXIMATE`
- `NOT_COMPATIBLE`

Compatibility means that the configured representation meets its tested
reconstruction tolerance. It does not establish the existence or hiddenness
of an attractor.

## Non-smooth vector fields

For piecewise-continuous systems, switching surfaces must be handled
explicitly:

- Jacobians are interpreted only where the vector field is differentiable.
- A discontinuous right-hand side requires an appropriate regularized or
  nonsmooth integration contract.
- Equilibria on switching surfaces can be reported as
  `nonsmooth_indeterminate` for derivative-based stability checks.
- Crossing and numerical-failure information remains part of the result.

## Symmetry

When a system transformation is used to generate symmetric initial
conditions, its equivariance must be checked numerically under the same model
parameters. Symmetry-derived samples are deduplicated and remain subject to
the same integration and evidence boundaries as the original sample.
