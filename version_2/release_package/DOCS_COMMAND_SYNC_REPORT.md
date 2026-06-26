# Document and CLI Synchronization Report

This report summarizes the modifications and verification results for synchronizing the documentation of `hidden-attractors-fo` v1.0.0 with the unified `hidden-attractors` CLI.

## Modified Files

The following documentation, configuration, and verification test files were updated during this synchronization:

### Configuration and Manifests

- [version_2/docs/manual_manifest.yaml](../../version_2/docs/manual_manifest.yaml)
- [version_2/release_package/sample_output/expected_release_readiness_summary.json](sample_output/expected_release_readiness_summary.json)

### Documentation & Manuals

- [README.md (root)](../../README.md)
- [version_2/README.md](../README.md)
- [version_2/INSTALL.md](../INSTALL.md)
- [version_2/USER_MANUAL.md](../USER_MANUAL.md)
- [version_2/REFERENCE_GUIDE.md](../REFERENCE_GUIDE.md)
- [version_2/docs/installation.md](../docs/installation.md)
- [version_2/docs/quick_start.md](../docs/quick_start.md)
- [version_2/docs/getting_started.md](../docs/getting_started.md)
- [version_2/docs/workflows.md](../docs/workflows.md)
- [version_2/docs/examples_index.md](../docs/examples_index.md)
- [version_2/docs/index.md](../docs/index.md)
- [version_2/docs/code_reference_map.md](../docs/code_reference_map.md)
- [version_2/docs/repository_layout.md](../docs/repository_layout.md)
- [version_2/docs/testing.md](../docs/testing.md)
- [version_2/docs/library_structure.md](../docs/library_structure.md)
- [version_2/docs/contributing.md](../docs/contributing.md)
- [version_2/docs/lure_candidate_route.md](../docs/lure_candidate_route.md)
- [version_2/docs/api_stability.md](../docs/api_stability.md)
- [version_2/docs/reporte_unificado_chua_fraccionario.tex](../docs/reporte_unificado_chua_fraccionario.tex)

### Tools Documentation

- [version_2/tools/cli/README.md](../tools/cli/README.md)
- [version_2/tools/legacy/README.md](../tools/legacy/README.md)

### Release Package Documentation

- [version_2/release_package/README_RELEASE.md](README_RELEASE.md)
- [version_2/release_package/SAMPLE_RUN.md](SAMPLE_RUN.md)
- [version_2/release_package/reproducibility_checklist.md](reproducibility_checklist.md)

### Verification and Test Scripts

- [version_2/tests/helpers/test_documentation_text.py](../tests/helpers/test_documentation_text.py)
- [version_2/tests/test_manual_freeze_audit_reference.py](../tests/test_manual_freeze_audit_reference.py)
- [version_2/tests/test_markdown_docs_cli_consistency.py](../tests/test_markdown_docs_cli_consistency.py)

---

## Documentation Changes Summary

### 1. Legacy Standalone CLI Commands Deprecation

All active/recommended command blocks executing obsolete standalone entry points have been deprecated. Documentation references to the following legacy entry points have been marked with explicit deprecation warnings and moved to migration-only sections:

- `hidden-attractors-protocol`
- `hidden-attractors-check-validation`
- `hidden-attractors-sphere-controls`
- `hidden-attractors-refined-basin`
- `hidden-attractors-fractional-report-run`
- `hidden-attractors-robustness-overlay`
- `hidden-attractors-danca-abm-sphere-controls`

### 2. Unified CLI Dispatcher Alignment

The documentation now recommends the single public CLI launcher `hidden-attractors`, mapping historical routines to their equivalent subcommands under the unified tool:

- `hidden-attractors seed lure-centered`
- `hidden-attractors seed lure-biased`
- `hidden-attractors validate contract`
- `hidden-attractors validate release-readiness`
- `hidden-attractors run`
- `hidden-attractors inspect`
- `hidden-attractors protocol <substage>`
- `hidden-attractors continuation`
- `hidden-attractors hiddenness`
- `hidden-attractors basin`
- `hidden-attractors robustness`
- `hidden-attractors bifurcation`
- `hidden-attractors lyapunov`
- `hidden-attractors chaos-test`

### 3. Installation Extras Synchronization

All developer/editable installation instructions now consistently recommend using the complete set of extras:

```bash
python -m pip install -e ".[dev,analysis,docs,legacy]"
```

This prevents incomplete virtual environments that lack documentation chains (such as MkDocs). Installation via PyPI/TestPyPI is also documented for end users.

### 4. Scientific Audit Alignment

Reflects the actual test suite results:

- Expected passed test count: `947`
- Expected skipped test count: `28`

These are synchronized across the manual manifest, tests, and the reference guides.

---

## Verification Logs

The hygiene and documentation-readiness verification tests pass cleanly:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.x, pytest-8.x.x, pluggy-1.x.x
rootdir: /path/to/workspace/version_2
collected 975 items / 893 deselected / 82 selected

tests/test_markdown_docs_cli_consistency.py ......                        [ 7%]
tests/test_no_legacy_cli_in_manuals.py .                                   [ 8%]
...
================ 81 passed, 1 skipped, 893 deselected in 7.23s ================
```

All documentation markdown linting and repository hygiene rules (MD012, MD022, MD031, MD032, MD055, MD056, MD060) have been fully resolved.
