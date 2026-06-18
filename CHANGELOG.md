# Changelog

## 0.1.0-cpc-preparation

### Added

- CPC metadata files: `CITATION.cff`, `.zenodo.json`, `codemeta.json`, `AUTHORS.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`, and `REPRODUCIBILITY.md`.
- CPC editorial skeleton under `paper/`.
- CPC submission skeleton under `version_2/cpc_submission/`.
- `hidden-attractors validate cpc-readiness`.
- CPC readiness and root hygiene tests.
- Lightweight CPC sample input and expected-output templates.

### Fixed

- Closed tracked-file leakage from `version_2/validation_outputs/`; promoted evidence now lives under `version_2/validation/`, while regenerable outputs remain ignored.
- Root hygiene policy for local reports and generated outputs.
- Figure-script policy centralization.
- CPC metadata records the OSF DOI and contributor/provenance notes.

### Pending before CPC submission

- Regenerate freeze audit on the final CPC-preparation commit.
- Replace template sample outputs with executed outputs.
- Complete manuscript narrative and bibliographic metadata.
- Update `archive_manifest.json` to `commit_status: current` after the final audit commit is frozen.
