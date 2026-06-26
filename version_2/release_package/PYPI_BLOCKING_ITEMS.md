# PyPI blocking items

All previously identified blocking items are now resolved.

## Resolved items

### 1. Lazy legacy-import fix in `strict_target_refinement.py`

**Status**: RESOLVED

The `_load_danca_legacy_helpers()` function was introduced in `hidden_attractors/workflows/strict_target_refinement.py`
to make the `danca2017_chua_abm_replication` import lazy. This prevented `ModuleNotFoundError`
when importing the module from a clean PyPI wheel environment.

Additionally, a duplicate-argument bug was found in `make_parser()` (the same parser arguments
were added twice), causing `argparse.ArgumentError` at runtime. The duplicate block was removed.

### 2. Eager legacy import in `danca_abm_sphere_controls.py` and `cli/published.py`

**Status**: RESOLVED

`hidden_attractors/workflows/danca_abm_sphere_controls.py` had an eager module-level import of
`danca2017_chua_abm_replication` and `equilibria_analysis` (from `tools/legacy`), which caused
`ModuleNotFoundError: No module named 'danca2017_chua_abm_replication'` when `hidden-attractors --help`
was run in a clean wheel environment.

Two changes were made:

1. `hidden_attractors/cli/published.py`: Made the import of `danca_abm_sphere_controls` lazy
   (inside the function body) so that importing `cli.main` does not trigger the legacy import.
2. `hidden_attractors/workflows/danca_abm_sphere_controls.py`: Wrapped the module-level legacy
   imports in a `try/except ModuleNotFoundError` with placeholder stubs that raise clear
   `RuntimeError` messages if the command is invoked without `tools/legacy`.

## Final validation result

After both fixes, `tools/release/validate_wheel_install.py` was run and passed:

```text
$ python -m build          → Successfully built hidden_attractors_fo-1.0.0.tar.gz and hidden_attractors_fo-1.0.0-py3-none-any.whl
$ twine check dist/*       → PASSED (wheel and sdist)
$ pip install dist/*.whl   → Successfully installed (all deps: numpy, scipy, matplotlib, numba, PyYAML)
$ hidden-attractors --help → OK (shows all command groups)
$ hidden-attractors seed --help → OK (shows lure-centered, lure-biased; no Machado/FDF)
$ python -c "import hidden_attractors; print('import ok')" → import ok
validate_wheel_install passed
```

## Next steps (manual)

The library is now ready for TestPyPI/PyPI upload. Run these commands from `version_2/`:

```bash
# Upload to TestPyPI first
python -m twine upload --repository testpypi dist/*

# Verify TestPyPI install
python -m venv .venv_testpypi
.venv_testpypi\Scripts\activate
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps hidden-attractors-fo==1.0.0
python -m pip install numpy matplotlib scipy numba PyYAML
hidden-attractors --help
python -c "import hidden_attractors; print('testpypi import ok')"

# Only after TestPyPI passes, upload to real PyPI
python -m twine upload dist/*

# Tag the release
git tag -a v1.0.0 -m "Release hidden-attractors-fo v1.0.0"
git push origin v1.0.0
```
