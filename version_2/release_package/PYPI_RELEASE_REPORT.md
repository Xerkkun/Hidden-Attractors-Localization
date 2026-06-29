# PyPI Release Validation Report

This report summarizes the packaging, validation, PyPI publication, and release readiness checks for version `1.0.0` of `hidden-attractors-fo`.

## Release Metadata

- **Package Name**: `hidden-attractors-fo`
- **Import Module**: `hidden_attractors`
- **Version**: `1.0.0`
- **Target HEAD Commit**: `700529c09d2744c2c021891900a48b03b846ee4b`
- **Target Branch**: `main`
- **Zenodo/OSF DOI**: `10.17605/OSF.IO/ZGK74`
- **License**: `MIT`
- **CI Environments**: Python 3.11, 3.12, 3.13 on macOS, Ubuntu, and Windows

---

## Verification Summary

| Check | Tool / Command | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Freeze Audit** | `run_final_freeze_audit.py` | **PASSED** | Historical promoted snapshot records `947 passed, 28 skipped`; dirty-tree metadata is retained and documented for traceability. |
| **Release Readiness** | `validate release-readiness --submission-strict` | **PASSED** | Unified CLI, archive manifest commit matched, metadata verified |
| **Contract Status** | `validate contract --allow-pending` | **PASSED** | No structural or numeric contract discrepancy warnings |
| **Twine Check** | `twine check dist/*` | **PASSED** | Verified metadata and structures of source distribution and wheel |
| **Clean Venv Smoke Test** | `validate_wheel_install.py` | **PASSED** | Installs correctly; imports run clean; CLI loads without errors |
| **TestPyPI Verification** | Manual upload & test | **PASSED** | Status, install, CLI smoke, and import tests passed before PyPI publication |

---

## PyPI Publication Results

The package is published on PyPI as `hidden-attractors-fo` version `1.0.0`:

- **PyPI project**: <https://pypi.org/project/hidden-attractors-fo/>
- **PyPI status**: published
- **TestPyPI preflight**: passed
- **Clean wheel install smoke**: passed

Install the published package with:

```bash
python -m pip install hidden-attractors-fo
```

## Build Output Files

The package distribution build generated the following artifacts under `version_2/dist/`:

1. **Source Distribution**: `hidden_attractors_fo-1.0.0.tar.gz` (Metadata and structure validated)
2. **Binary Wheel**: `hidden_attractors_fo-1.0.0-py3-none-any.whl` (Pure Python wheel, verified on Python 3.14)

---

## Clean-Room Installation Log Excerpt

```text
Successfully built hidden_attractors_fo-1.0.0.tar.gz and hidden_attractors_fo-1.0.0-py3-none-any.whl
$ twine check dist/*
Checking dist\hidden_attractors_fo-1.0.0-py3-none-any.whl: PASSED
Checking dist\hidden_attractors_fo-1.0.0.tar.gz: PASSED
$ pip install dist\hidden_attractors_fo-1.0.0-py3-none-any.whl
Successfully installed PyYAML-6.0.3 contourpy-1.3.3 cycler-0.12.1 fonttools-4.63.0 hidden-attractors-fo-1.0.0 kiwisolver-1.5.0 llvmlite-0.47.0 matplotlib-3.11.0 numba-0.65.1 numpy-2.4.6 packaging-26.2 pillow-12.2.0 pyparsing-3.3.2 python-dateutil-2.9.0.post0 scipy-1.18.0 six-1.17.0
$ hidden-attractors --help
usage: hidden-attractors [-h]
                         {run,init,inspect-config,inspect,validate,protocol,hiddenness,basin,robustness,bifurcation,lyapunov,chaos-test,published,report,seed,continuation} ...
$ hidden-attractors seed --help
positional arguments:
  {lure-centered,lure-biased}
$ python -c "import hidden_attractors; print('import ok')"
import ok
validate_wheel_install passed
```

---

## Publishing Instructions (Manual Release)

For future patch releases, complete the following manual steps:

1. **Verify via TestPyPI**:

   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

2. **Tag the scientific release**:

   ```bash
   git tag -a v1.0.0 -m "Release hidden-attractors-fo v1.0.0"
   git push origin v1.0.0
   ```

3. **Trigger GitHub Action for trusted publishing**:
   The workflow `.github/workflows/publish-pypi.yml` is configured to run automatically upon pushing tag `v*`. Make sure Trusted Publishing is configured on PyPI for `Xerkkun/Hidden-Attractors-Localization` pointing to this workflow.
