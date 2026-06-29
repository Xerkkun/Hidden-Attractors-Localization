# Freeze Audit Artifacts

This directory contains the promoted final freeze audit snapshot for the release evidence bundle.

The committed `final_freeze_pytest_summary.json` is a historical promoted artifact. It records the exact audit metadata from the original freeze run, including `working_tree_dirty: true` and its diff hash. That dirty-tree state is retained for traceability; it is not a current clean-tree regeneration result and must not be edited away.

For a current local regeneration, write to an ignored scratch path first:

```bash
python validation/python/run_final_freeze_audit.py --require-clean --output-dir validation_outputs/freeze_audit_current
```

Only replace promoted files in this directory from a clean tree, after reviewing the generated summary and stdout. Current summaries that report a dirty working tree must either fail readiness or explicitly declare `historical_artifact: true` with a note explaining why the dirty state is retained.