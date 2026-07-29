# Comprehensive release sample

From `version_2/`, execute the recorded software-validation control in a new
output directory:

```bash
python examples/chua_integer_lure_reference/run_example.py \
  --config release_package/sample_input/chua_integer_comprehensive.yaml \
  --quick \
  --steps search continuation verification \
  --output-dir <empty-output-directory>
```

Quick mode retains search, continuation, final integration, equilibrium-
neighborhood sampling, and structured output while reducing numerical sizes.
The tracked input writes only beneath `outputs/release_samples/` unless the
command-line output directory overrides it.

The recorded execution produced:

- five continuation steps;
- a finite final trajectory;
- six sampled-neighborhood probes and two target contacts;
- `hidden_candidate_allowed: false`;
- 2.455 seconds elapsed on the recorded Windows environment;
- identical deterministic files in two independent runs.

The elapsed time is environment provenance, not a benchmark. The numerical
decision is a finite sampled software control, not promoted evidence and not a
global proof.

The complete compact record, including SHA-256 hashes, is:

`release_package/sample_output/comprehensive_sample_summary.json`

Release checks:

```bash
python tools/release/validate_release_readiness.py --submission-strict --json
python -m pytest -q tests/test_release_package_samples.py
python -m pytest -q tests/test_public_distribution_contract.py
python -m pytest -q tests/test_pypi_packaging.py
python -m build
python -m twine check dist/*
python tools/release/validate_wheel_install.py
```

These commands validate software and packaging. They do not promote or rewrite
scientific validation records.
