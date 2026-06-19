# Reproducibility checklist

- Record repository commit and package version.
- Install from `version_2/` with development, analysis, and legacy extras.
- Run `hidden-attractors --help`.
- Run `hidden-attractors validate contract --allow-pending`.
- Run `hidden-attractors validate release-readiness`.
- Run `hidden-attractors validate release-readiness --strict` for repository/software readiness.
- Run `hidden-attractors validate release-readiness --submission-strict` only for the final submission package.
- Run the intended pytest lane and record exact command, platform, Python, and dependency versions.
- Treat `validation/freeze_audit/` as the frozen source for published scientific test counts.
- Keep local outputs under ignored `outputs/`, `validation_outputs/`, `runs*/`, or `figures/`.
- Prepare local writing and manuscript drafts locally under ignored `paper/`, ensuring they remain untracked by Git.
- Do not promote arctan as a validated hidden attractor until the validation contract supports it.

## CI and freeze audit boundary

The GitHub Actions CI matrix for the release cleanup has passed. This confirms package hygiene and cross-platform test execution for the current repository state. It does not replace the full scientific freeze audit, which remains a separate artifact to regenerate once final promoted validation cases are fixed.

The project keeps a small hygiene/readiness test suite because numerical tests do not protect repository publication boundaries. These tests guard against retracking local outputs, local manuscripts, absolute paths, legacy CLI entry points, unpromoted validation outputs, and overclaimed release metadata.

To run these tests specifically:
```bash
python -m pytest -q -m "hygiene"
python -m pytest -q -m "release_readiness"
python -m pytest -q -m "not hygiene and not release_readiness"
```

CI status: passed for current release cleanup. Freeze audit: last full scientific freeze audit corresponds to commit `2bcea3430c50d3fb4e5eb70c8621cb3550dcc59a` and must be regenerated only when the final scientific evidence set is frozen.

Current release cleanup state: `ci_status: passed`; `freeze_audit_status: pending_final_scientific_freeze`; `sample_status: template_only_pending_execution`.
