# Arctan c590 promotion boundary

The Chua fractional arctan c590 candidate is promoted in version `1.0.0` as
finite, radius-limited hiddenness evidence under the recorded Caputo ABM
full-memory contract.

## Promoted local claim

- Label: `hiddenness_supported_under_tested_local_radii`.
- Candidate: `chua_arctan_c590_q09999_seed9`.
- Promoted validation package: `version_2/validation/chua_fractional_arctan_c590/`.
- Tested local radii with zero contacts: `1e-5`, `1e-4`, `1e-3`, `1e-2`,
  `0.03`, `0.1`, and `0.3`.
- Local finite probes: `8400`.
- Local target contacts: `0`.
- Equilibria tested: `E0`, `E+`, and `E-`.

## Extended-radius audit

The extended radii `1.0` and `2.0` were deliberately added after the local
hiddenness checks. They record `610` contacts in `5100` macro-radius probes.
Those contacts are documented as an extended-radius basin audit and do not erase
the local-radius promotion, just as other methodology lanes keep their claim
bounded by the tested radius contract.

## Scientific boundary

This is finite numerical evidence under a declared radius, sampling, integrator,
memory, and classifier contract. It is not a global mathematical basin proof.
The Wu2023 bibliographic ADM reproduction remains separate from the promoted
c590 Caputo lane.
