# Reproducibility checklist

- Record repository commit and package version.
- Install from `version_2/` with development, analysis, and legacy extras.
- Run `hidden-attractors --help`.
- Run `hidden-attractors validate contract --allow-pending`.
- Run `hidden-attractors validate cpc-readiness`.
- Run `hidden-attractors validate cpc-readiness --strict` only after the final freeze audit and sample executions are recorded.
- Run the intended pytest lane and record exact command, platform, Python, and dependency versions.
- Treat `validation/freeze_audit/` as the frozen source for published test counts.
- Keep local outputs under ignored `outputs/`, `validation_outputs/`, `runs*/`, or `figures/`.
- Do not promote arctan as a validated hidden attractor until the validation contract supports it.

Current CPC cleanup state: `freeze_audit_status: pending_after_cpc_cleanup`; `sample_status: template_only_pending_execution`.
