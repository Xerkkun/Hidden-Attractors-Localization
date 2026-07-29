# Comprehensive sample input

`chua_integer_comprehensive.yaml` is the input for the comprehensive recorded
software-validation sample. The verification command is:

```bash
python examples/chua_integer_lure_reference/run_example.py \
  --config release_package/sample_input/chua_integer_comprehensive.yaml \
  --quick \
  --steps search continuation verification \
  --output-dir <empty-output-directory>
```

Quick mode preserves every workflow stage while reducing the continuation,
trajectory, and sampled-neighborhood sizes so the package can be checked in a
clean environment. It is a software-validation control, not promoted
scientific evidence.
