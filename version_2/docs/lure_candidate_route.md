# Lur'e Seed Generation

The supported seed stage is available for systems that provide an explicit
scalar Lur'e representation. It produces initialization records for numerical
continuation; it does not establish the existence, chaoticity, or hiddenness
of an attractor.

## Supported commands

```bash
hidden-attractors seed lure-centered --help
hidden-attractors seed lure-biased --help
```

The centered and biased routes return the same core fields:

```json
{
  "family": "lure_classical_centered | lure_classical_biased",
  "centered_or_biased": "centered | biased",
  "A": 0.0,
  "sigma0": 0.0,
  "omega": 0.0,
  "mu": 1.0,
  "theta": 0.0,
  "q": 1.0,
  "harmonic_residual": 0.0,
  "x0": [0.0, 0.0, 0.0],
  "reconstruction_metadata": {}
}
```

For a fractional-order route, the frequency balance uses the configured
branch of `(j omega)^q`. The describing function is a first-harmonic
approximation used to locate an initial condition. It must be followed by
integration of the target system under an explicit numerical contract.

## Pre-continuation checks

A direct integration before continuation is diagnostic. A periodic-looking or
equilibrium-bound seed may still be passed to
`ContinuationPlan(lambda_values=...)`. Invalid parameters, non-finite values,
numerical failure, or an exact duplicate may stop the route before
continuation.

## Evidence boundary

Continuation reaching `lambda=1` only establishes that the numerical path
reached the target parameterization. Subsequent trajectory diagnostics and
robustness checks are separate. If hiddenness is assessed, equilibrium-
neighborhood controls must record all equilibria, radii, sample counts,
sampling mode, solver, horizon, memory policy, classifier, and failure policy.

No seed-generation or continuation output alone should be reported as a
hidden-attractor result.
