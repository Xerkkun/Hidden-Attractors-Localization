# Testing Policy

## Permanent tests
Protect installation, public CLI, scientific contracts, validation contracts,
figure export contracts, bibliographic traceability and repository hygiene.

## Slow tests
Reproduce published cases, long integrations, full hiddenness probes,
full report generation and solver-memory comparisons.

## Migration tests
Temporary tests used during refactors. They must either become general
invariants or be removed after the migration is closed.

## Deprecated aliases
Compatibility tests for old commands. These are removed after the documented
deprecation window.

## Deletion rule
A passing test is not deleted because it passed. It is deleted only if it is
redundant, obsolete, fragile without protecting a contract, or replaced by a
stronger invariant.
