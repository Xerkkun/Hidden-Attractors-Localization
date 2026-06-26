# Final PyPI Release Traceability and Verification Report

This report documents the final traceability and package verification checks performed on version `1.0.0` of `hidden-attractors-fo` prior to release publication.

## 1. Release Commit and Target Metadata

- **Package Name**: `hidden-attractors-fo`
- **Import Module**: `hidden_attractors`
- **Release Version**: `1.0.0`
- **Target HEAD Commit**: `700529c09d2744c2c021891900a48b03b846ee4b`
- **Target Release Branch**: `main`
- **OSF/Zenodo DOI**: `10.17605/OSF.IO/ZGK74`
- **Source Repository**: `Xerkkun/Hidden-Attractors-Localization`

---

## 2. Package Build and Twine Verification

The source distribution and pure Python wheel packages were built from the clean repository matching the target commit and verified using `twine check`:

1. **Source Distribution**: `hidden_attractors_fo-1.0.0.tar.gz` (twine verified: **PASSED**)
2. **Binary Wheel**: `hidden_attractors_fo-1.0.0-py3-none-any.whl` (twine verified: **PASSED**)

Command run:
```bash
python -m build
python -m twine check dist/*
```

Twine output:
```text
Checking dist\hidden_attractors_fo-1.0.0-py3-none-any.whl: PASSED
Checking dist\hidden_attractors_fo-1.0.0.tar.gz: PASSED
```

---

## 3. Scientific Freeze Audit and Readiness

The scientific test inventory and package release checks were run on the target code layout and verified fully ready:

- **Total Unit & Integration Tests**: `947 passed, 28 skipped`
- **Unified Release Readiness**: `validate release-readiness --submission-strict` (**PASSED**)
- **Contract Verification**: `validate contract --allow-pending` (**PASSED**)
- **Working Tree Cleanliness Check**: Verified clean release baseline matching target commit.

Audit summary path:
- [final_freeze_pytest_summary.json](../validation/freeze_audit/final_freeze_pytest_summary.json)
- [final_freeze_pytest_stdout.txt](../validation/freeze_audit/final_freeze_pytest_stdout.txt)

---

## 4. Clean-Room Wheel Smoke Installation

A clean-room installation smoke test was executed inside a temporary virtual environment (simulating an end-user install of the pure Python wheel) on Python 3.14. The validation suite confirms:
- Dependency installation completes with zero warning flags.
- Command-line entry point `hidden-attractors` loads properly.
- All CLI subcommands and help texts are accessible.
- Imports execute cleanly without exporting theoretical/internal Machado/FDF routes.

Smoke verification status: **PASSED**

---

## 5. Manual Release Publishing Instructions

According to the repository policies, actual tagging and uploading are manual steps performed by the maintainer. To finalize the release:

1. **Verify via TestPyPI**:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

2. **Create Git Tag for the Release**:
   ```bash
   git tag -a v1.0.0 -m "Release hidden-attractors-fo v1.0.0"
   git push origin v1.0.0
   ```

3. **Publish to PyPI**:
   Trusted Publishing is configured on PyPI. Once the tag `v1.0.0` is pushed, the `.github/workflows/publish-pypi.yml` workflow will automatically execute to build and securely publish the verified artifacts.
