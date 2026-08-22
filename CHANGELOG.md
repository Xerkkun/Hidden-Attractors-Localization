# Changelog

## 1.2.0

### Added

- Validated public contracts for the promoted existing fractional-method,
  dynamical-analysis, and geometric/topological execution paths.
- Native input-hardening checks and import-order regression coverage.
- A conservative `hidden-attractors update` command that checks stable PyPI
  releases, invokes pip only after explicit confirmation in supported launch
  contexts, refuses self-replacement from an active Windows `.exe` launcher,
  and handles failures without automatic elevation or downgrade.

### Changed

- Publication now requires an exact semantic-version tag that resolves to the
  workflow source commit.
- Release identity is distinct from the historical `v1.1.0` source snapshot.

### Fixed

- Public callable names remain callable after importing same-named submodules.
- Relocated validation runs and test fixtures no longer write into promoted
  repository evidence directories.

## 1.1.0

### Added

- Fully integrated scalar-time-series Lyapunov diagnostics with Rosenstein and
  Eckmann estimators, Kaplan--Yorke dimension, explicit units, deterministic
  seeding, structured provenance, and JSON-capable CLI output.
- Runtime output/cache overrides for installed-wheel execution.

### Changed

- Public documentation and source-distribution contents are limited to
  implemented software, completed validation, and the validated integer
  end-to-end example.
- The source distribution uses an explicit whitelist instead of recursively
  packaging internal notes, plans, tests, and release records.
- Runtime outputs default to `./outputs`; caches use the user-cache directory
  instead of the installed package tree.

### Fixed

- Rosenstein estimates receive the documented sampling interval and report
  inverse-time units.

## 1.0.0

- First public PyPI release of the unified `hidden-attractors` package and CLI.
- Added integer and commensurate Caputo integration, Lur'e seed construction,
  continuation, diagnostics, verification contracts, metadata, and validation
  tooling.
