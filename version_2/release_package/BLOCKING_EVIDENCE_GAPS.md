# Blocking evidence gaps

This release package must not promote the fractional Chua arctan lane as a
validated hidden attractor until the gaps below are closed in promoted evidence.

## Decision

Release version `v1.0.0` is blocked. The package remains a release-candidate
state with conservative claims because the current arctan evidence is not a
complete promoted hiddenness validation package.

## Evidence audited

- `version_2/validation/reference_cases/fractional_chua_arctan_wu2023/validation_summary.json`
  keeps `hidden_verified=false`, records the Wu2023 local ADM reproduction as
  `memory_policy=none_local_adm`, and does not provide full Caputo-memory
  hiddenness certification.
- `version_2/validation/chaos_validation/` contains diagnostics for
  `wu2023_chua_fractional_arctan_q099`, but those diagnostics retain
  `hidden_verified=false` and include regular-like/inconclusive classifications.
- `version_2/outputs/arctan_hidden_candidate_search/c590_q09999_seed9_candidate_20260623/candidate_summary_latest.json`
  records a proposed full-history ABM c590 candidate with 13,500 finite
  deterministic neighborhood tests, all three equilibria listed, and 610
  target contacts. Its own verdict is
  `requires_review_due_to_macro_radius_contacts`.
- `version_2/outputs/` is ordinary ignored output, not promoted validation
  evidence. Those files cannot be cited as release-certified evidence until a
  curated validation package and manifest are promoted under `version_2/validation/`.

## Blocking gaps

1. No canonical promoted directory exists for the proposed c590 fractional
   arctan hiddenness package, for example
   `version_2/validation/chua_fractional_arctan/`.
2. The c590 contacts at large radii require scientific review before any
   hiddenness label can be promoted.
3. The promoted Wu2023 reference lane is bibliographic/algebraic/local ADM
   reproduction, not a full-history Caputo hiddenness reproduction.
4. A release-grade arctan package must include JSON/CSV decisions, run
   configuration, random seeds, all-equilibrium neighborhoods, Matignon checks,
   integrator, memory policy, `h`, `t_final`, burn-in, radii, sample counts,
   thresholds, and figure provenance without local absolute paths.
5. The final scientific freeze audit must be regenerated only after the
   promoted evidence set is fixed.

## Promotion condition

Arctan may be promoted only when a tracked validation package records a stable
verdict under the official contract and `hidden-attractors validate
release-readiness --submission-strict` passes without final pending items.