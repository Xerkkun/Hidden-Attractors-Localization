# Testing Policy

## Core tests

Core tests protect installation, the public API and CLI, numerical contracts,
supported configuration schemas, and repository hygiene.

## Extended tests

Computationally expensive checks are marked explicitly. They are retained when
they verify a documented numerical method or reproduce a published reference,
and their evidence boundary remains attached to the corresponding test data.

## Validation boundary

Passing software tests demonstrates conformance to their recorded contracts.
It does not by itself certify chaos, hiddenness, physical performance, or a
scientific result.

## Maintenance

- Refactor-specific checks are retained only when they protect a current,
  general invariant.
- Compatibility tests cover only documented public aliases.
- A test is removed only when it is redundant, obsolete, or replaced by a
  stronger invariant.
