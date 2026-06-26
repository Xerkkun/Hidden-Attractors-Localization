# Blocking release items

This file records items that block declaring `hidden-attractors-fo` as
`v1.0.0`. They are intentionally public release-package metadata, not hidden
local notes.

## Blocking items

1. Fractional Chua arctan hiddenness is not promoted. The current Wu2023 lane
   has `hidden_verified=false`, and the proposed c590 lane remains under review
   because its latest finite sampling summary reports macro-radius target
   contacts.
2. The final scientific freeze audit is pending for the final promoted evidence
   set.
3. Sample outputs remain templates until executed sample outputs are required
   and recorded.
4. The archive manifest commit is intentionally marked pending until the final
   cleanup commit is selected.

## Non-blocking retained scope

- Integer Chua `q=1` remains the reproduced software reference lane.
- The non-smooth fractional Chua biased-DF lane remains a methodology example
  with contract-limited evidence, not a global mathematical proof.
- Machado/FDF remains documented as theory and internal/planned support. It is
  not exposed as a public release CLI command.

## Required closure before `v1.0.0`

- Promote only evidence that lives under `version_2/validation/` with complete
  manifests and no local absolute paths.
- Re-run the release readiness validator in strict and submission-strict modes.
- Regenerate the final freeze audit after scientific evidence is frozen.
- Update `archive_manifest.json`, release notes, and sample-output metadata to
  the final commit and executed status.