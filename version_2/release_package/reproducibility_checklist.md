# Reproducibility checklist

- Record repository commit and package version.
- Install from PyPI with `python -m pip install hidden-attractors-fo` for user smoke tests.
- Install from `version_2/` with development, analysis, docs, and legacy extras for repository validation.
- Confirm the Python import is `import hidden_attractors`.
- Run `hidden-attractors --help`.
- Run `hidden-attractors seed --help` and confirm only `lure-centered` and `lure-biased` are exposed.
- Run `hidden-attractors validate contract --allow-pending`.
- Run `hidden-attractors validate release-readiness`.
- Run `hidden-attractors validate release-readiness --strict` for repository/software readiness.
- Run `hidden-attractors validate release-readiness --submission-strict` only for the final archive package.
- Build the PyPI artifacts with `python -m build`.
- Check distribution metadata with `python -m twine check dist/*`.
- Smoke-test wheel installation with `python tools/release/validate_wheel_install.py`.
- Run the intended pytest lane and record exact command, platform, Python, and dependency versions.
- Treat `validation/freeze_audit/` as the frozen source for published scientific test counts.
- Keep local outputs under ignored `outputs/`, `validation_outputs/`, `runs*/`, or `figures/`.
- Prepare local writing and manuscript drafts locally under ignored `paper/`, ensuring they remain untracked by Git.
- Interpret arctan c590 only under the recorded local-radius contract (`r <= 0.3`, finite-time evidence, zero local contacts); do not treat it as a global basin proof or as the central project result.
- Review `release_package/ARCTAN_C590_PROMOTION_BOUNDARY.md` before interpreting the arctan c590 claim.
- Review `release_package/PYPI_RELEASE_CHECKLIST.md` and `release_package/PUBLISHING_POLICY.md` before TestPyPI or PyPI upload.

## CI and freeze audit boundary

The GitHub Actions CI matrix for the release cleanup has passed. This confirms package hygiene and cross-platform test execution for the recorded repository state. The scientific freeze audit is tracked separately under `validation/freeze_audit/`.

The project keeps a small hygiene/readiness test suite because numerical tests do not protect repository publication boundaries. These tests guard against retracking local outputs, local manuscripts, absolute paths, legacy CLI entry points, unpromoted validation outputs, and overclaimed release metadata.

To run these tests specifically:

```bash
python -m pytest -q -m "hygiene"
python -m pytest -q -m "release_readiness"
python -m pytest -q -m "not hygiene and not release_readiness"
```

Packaging-only tests can be run with:

```bash
python -m pytest -q -m "packaging"
```

The build-running packaging test is opt-in:

```bash
RUN_PYPI_BUILD_TEST=1 python -m pytest -q tests/test_pypi_packaging.py -m "packaging"
```

CI status: passed for release cleanup. Freeze audit status is recorded in `archive_manifest.json` and `validation/freeze_audit/final_freeze_pytest_summary.json`.

Current release cleanup state is machine-readable in `release_package/archive_manifest.json`; v1.0.0 requires `freeze_audit_status: passed`, `sample_status: executed`, and `pypi_readiness` build/check fields recorded as passed after local wheel validation.
