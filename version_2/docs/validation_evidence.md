# Public Validation Boundary

The public documentation is limited to supported interfaces and completed
validation results. It does not publish material outside that boundary.

For user-facing commands and result schemas, see `USER_MANUAL.md`. Scientific
interpretation remains bounded by the method metadata returned by the library.

## Validation Layers

The release keeps four layers separate:

1. **Software validation** checks inputs, outputs, numerical invariants,
   packaging, and reproducibility metadata.
2. **Published-reference validation** compares an implemented method with a
   reference whose numerical data are sufficient for the declared comparison.
3. **Finite-time characterization** reports boundedness, spectral, Poincare,
   0--1, bifurcation, and Lyapunov diagnostics for supplied systems or data.
4. **Hiddenness assessment** samples declared neighborhoods of every relevant
   equilibrium under an explicit numerical contract.

A result from one layer does not automatically establish a result from another.
In particular, a seed, continuation path, phase portrait, positive estimated
exponent, or 0--1 statistic is not by itself evidence of hiddenness.

## Completed Public Reference Controls

The release exposes the completed executable integer Chua reference workflow
for seed generation, continuation, integration, and conservative
equilibrium-neighborhood controls. The integer QR--Benettin Lyapunov method is
the only equation-based registry method marked as validated against published
benchmarks.

The scalar time-series Lyapunov route is fully integrated as a finite-data
diagnostic. It returns Rosenstein and Eckmann estimates, a Kaplan--Yorke
dimension, sampling units, parameters, backend provenance, fit diagnostics,
and warnings. It does not claim an asymptotic spectrum.

Other implemented Lyapunov methods retain their exact registry status:
synthetic validation only, recorded benchmark discrepancy, or numerical
comparison only. They remain callable diagnostics, not published-validated
methods.

## Corrected Non-Smooth Chua Record

The completed `paper07_chua_nonsmooth_corrected` record separates its local
contract from the extended basin audit. The local contract contains 7,200
ball samples through `r = 0.01`, with zero target contacts and zero numerical
failures. The radii `0.03`, `0.1`, and `0.3` belong to a separate
macro-basin exploration; that layer contains 10,200 samples and records 37
contacts at `r = 0.3`.

The defensible result is therefore
`compatible_with_hiddenness_under_sampled_local_balls` for `r <= 0.01`.
It is not a global basin proof. The recorded trajectory is classified as
regular/periodic under its dynamical filter, so this result is not a chaos
claim.

## Neighborhood Interpretation

Local neighborhoods and extended spherical audits are different numerical
questions. A contact detected on a sphere of large radius around an
equilibrium is not, by itself, evidence that the attractor is self-excited.
The operative hiddenness test concerns sufficiently small neighborhoods of all
equilibria. Large-radius spherical probes are reported as extended
basin-geometry audits.

The evidence record distinguishes interior balls, spherical surfaces, and
spherical shells. These geometries answer different finite-sampling questions:
surface or shell coverage is not described as filled-ball coverage. Local and
macro-radius records are stored separately even when they share a target
classifier.

Every sampled-neighborhood result must retain its geometry, radii, sample
counts, integrator, step size, horizon, memory policy, target classifier,
classification threshold, and numerical failure count. For Caputo dynamics it
also retains the initial time and history definition for every probe; history
is not transferred between probes unless the declared contract explicitly does
so.

The target classifier and its threshold are fixed before the sweep. An optional
causal stopping rule may complete all planned probes at the first radius with
contacts, across every equilibrium, and then omit larger radii. That record is
complete at the first-contact radius but does not claim coverage of omitted
radii. All such evidence remains conditional on its finite contract and is not
a global basin proof.

## Distribution Boundary

The wheel and source distribution contain supported software, public examples,
and user documentation. Release-tagged validation records are maintained
separately from the PyPI payload. Public documentation does not promote
case-specific material whose validation status is partial, discrepant, or
reference-only.
