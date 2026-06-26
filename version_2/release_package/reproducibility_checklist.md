# Reproducibility checklist

- Record repository commit and package version.
- Install from `version_2/` with development, analysis, and legacy extras.
- Run `hidden-attractors --help`.
- Run `hidden-attractors validate contract --allow-pending`.
- Run `hidden-attractors validate release-readiness`.
- Run `hidden-attractors validate release-readiness --strict` for repository/software readiness.
- Run `hidden-attractors validate release-readiness --submission-strict` only for the final archive package.
- Run the intended pytest lane and record exact command, platform, Python, and dependency versions.
- Treat `validation/freeze_audit/` as the frozen source for published scientific test counts.
- Keep local outputs under ignored `outputs/`, `validation_outputs/`, `runs*/`, or `figures/`.
- Prepare local writing and manuscript drafts locally under ignored `paper/`, ensuring they remain untracked by Git.
- Interpret arctan c590 only under the promoted local-radius contract (`r <= 0.3`, 8400 finite probes, zero contacts); do not treat it as a global basin proof.
- Review `release_package/ARCTAN_C590_PROMOTION_BOUNDARY.md` before interpreting the arctan c590 claim.
- Confirm `hidden-attractors seed --help` exposes only `lure-centered` and `lure-biased`; Machado/FDF remains theory/internal planned support.

## CI and freeze audit boundary

The GitHub Actions CI matrix for the release cleanup has passed. This confirms package hygiene and cross-platform test execution for the recorded repository state. The scientific freeze audit is tracked separately under `validation/freeze_audit/`.

The project keeps a small hygiene/readiness test suite because numerical tests do not protect repository publication boundaries. These tests guard against retracking local outputs, local manuscripts, absolute paths, legacy CLI entry points, unpromoted validation outputs, and overclaimed release metadata.

To run these tests specifically:
```bash
python -m pytest -q -m "hygiene"
python -m pytest -q -m "release_readiness"
python -m pytest -q -m "not hygiene and not release_readiness"
```

CI status: passed for release cleanup. Freeze audit status is recorded in `archive_manifest.json` and `validation/freeze_audit/final_freeze_pytest_summary.json`.

Current release cleanup state is machine-readable in `release_package/archive_manifest.json`; v1.0.0 requires `freeze_audit_status: passed` and `sample_status: executed`.
