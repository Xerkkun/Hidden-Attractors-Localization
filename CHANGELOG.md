# Changelog

## Unreleased

### Added

- Experimental scalar-time-series Lyapunov API with a Rosenstein largest
  exponent, Eckmann spectrum, Kaplan--Yorke dimension, structured provenance,
  deterministic RANSAC seeding, and JSON-capable CLI output.

### Fixed

- The optional `nolds` complexity adapter now propagates `sample_rate` to
  `lyap_r` through `tau=1/sample_rate`, so Rosenstein estimates have the
  documented inverse-time units.
- The trajectory Lyapunov CLI now validates uniform sampling and reports the
  scalar LLE, spectrum, Kaplan--Yorke dimension, backend, and diagnostic
  evidence boundary.

## 1.0.0

### Added

- Generated API reference covering every function, class, and method under
  `version_2/hidden_attractors`.
- README for the integer Chua Lur'e reference example.
- Release-oriented documentation for the three official examples and new Lur'e
  system adaptation workflow.

### Changed

- Updated root and package READMEs for the `1.0.0` release.
- Rewrote quick start, getting started, examples, systems, user manual, and
  release manifest pages to use the unified `hidden-attractors` CLI.
- Clarified reproduced, partial, contract-classified, radius-limited, and non-promoted article cases.
- Neutralized release/readiness wording so public files avoid venue-specific external targets.

### Scientific-results policy

- No numerical result, parameter, tolerance, seed, classifier, or promoted
  scientific conclusion was changed.
- Existing evidence boundaries remain: integer Chua reproduced; non-smooth
  fractional Danca case partial and methodologically relevant for BDF/saturation;
  nearby non-smooth candidate classified under the recorded local-neighborhood
  contract; arctan c590 lane kept as one radius-limited smooth-nonlinearity
  validation example for local radii `r <= 0.3`; Lyapunov comparison lanes
  remain diagnostic.

### Release readiness

- Final release metadata expects executed sample outputs and a regenerated freeze audit.
- No blocking documentation-balance item remains for the recorded validation contracts.
- Optional future work is limited to stronger basin/global analyses and benchmark expansion.
