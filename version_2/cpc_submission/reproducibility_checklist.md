# Reproducibility checklist

- Record repository commit and package version.
- Install from `version_2/` with development, analysis, and legacy extras.
- Run `hidden-attractors --help`.
- Run `hidden-attractors validate contract --allow-pending`.
- Run `hidden-attractors validate cpc-readiness`.
- Run `hidden-attractors validate cpc-readiness --strict` for repository/software readiness.
- Run `hidden-attractors validate cpc-readiness --submission-strict` only for the final submission package.
- Run the intended pytest lane and record exact command, platform, Python, and dependency versions.
- Treat `validation/freeze_audit/` as the frozen source for published scientific test counts.
- Keep local outputs under ignored `outputs/`, `validation_outputs/`, `runs*/`, or `figures/`.
- Do not promote arctan as a validated hidden attractor until the validation contract supports it.

## CI and freeze audit boundary

The GitHub Actions CI matrix for the CPC cleanup has passed. This confirms package hygiene and cross-platform test execution for the current repository state. It does not replace the full scientific freeze audit, which remains a separate artifact to regenerate once final promoted validation cases are fixed for submission.

CI status: passed for current CPC cleanup. Freeze audit: last full scientific freeze audit corresponds to commit `2bcea3430c50d3fb4e5eb70c8621cb3550dcc59a` and must be regenerated only when the final scientific evidence set is frozen.

Current CPC cleanup state: `ci_status: passed`; `freeze_audit_status: pending_final_scientific_freeze`; `sample_status: template_only_pending_execution`.
