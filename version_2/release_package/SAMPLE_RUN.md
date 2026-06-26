# Sample run

These commands exercise release metadata, repository hygiene, the public CLI, and the PyPI package build without promoting new scientific evidence.

```bash
cd version_2
python -m pip install -e ".[dev,analysis,legacy]"
hidden-attractors --help
hidden-attractors seed --help
hidden-attractors validate release-readiness --submission-strict --json
hidden-attractors validate contract --allow-pending
python -m pytest -q tests/test_release_readiness_metadata.py
python -m pytest -q tests/test_root_repository_hygiene.py
python -m pytest -q tests/test_release_package_samples.py
python -m pytest -q tests/test_release_docs_no_mojibake.py
python -m pytest -q tests/test_pypi_packaging.py
```

Package build sample:

```bash
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
python tools/release/validate_wheel_install.py
```

The wheel smoke test verifies:

```bash
hidden-attractors --help
hidden-attractors seed --help
python -c "import hidden_attractors; print('import ok')"
```

The sample YAML files are templates for interface checks and write to ignored local folders under `outputs/release_samples/`. They are not promoted evidence. The integer sample can be tried with:

```bash
hidden-attractors run -c release_package/sample_input/chua_integer_reference_minimal.yaml
```

Do not replace `sample_output/` templates with executed outputs unless the command was actually run on the final release commit. Exact test counts can change; the frozen source for published counts is `validation/freeze_audit/`.
