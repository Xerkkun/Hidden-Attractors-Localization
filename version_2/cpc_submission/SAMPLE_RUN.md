# Sample run

These commands exercise CPC metadata, repository hygiene, and the public CLI without promoting new scientific evidence.

```bash
cd version_2
python -m pip install -e ".[dev,analysis,legacy]"
hidden-attractors --help
hidden-attractors validate cpc-readiness
hidden-attractors validate contract --allow-pending
python -m pytest -q tests/test_cpc_readiness_metadata.py
python -m pytest -q tests/test_root_repository_hygiene.py
python -m pytest -q tests/test_cpc_submission_samples.py
python -m pytest -q tests/test_cpc_no_mojibake.py
```

The sample YAML files are templates for interface checks and write to ignored local folders under `outputs/cpc_samples/`. They are not promoted evidence. The integer sample can be tried with:

```bash
hidden-attractors run -c cpc_submission/sample_input/chua_integer_reference_minimal.yaml
```

Do not replace `sample_output/` templates with executed outputs unless the command was actually run on the final CPC-preparation commit. Exact test counts can change; the frozen source for published counts is `validation/freeze_audit/`.
