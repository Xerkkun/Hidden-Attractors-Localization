# Examples

This directory contains small public API examples and one explicitly labelled
validation reference. The API examples demonstrate reusable library features;
the validation reference retains its own numerical and evidence boundary.

## Validation reference

The integer-order reference reproduces a registered software-validation record.
It does not broaden the claims stored with that record.

Run from `version_2`:

```bash
python examples/chua_integer_lure_reference/run_example.py --quick
```

| Validation record | Role | Evidence boundary |
| --- | --- | --- |
| `chua_integer_lure_reference/` | Integer Chua `q=1` Lur'e reference: seed, continuation, trajectory, neighborhood controls, figures | Reproduced integer Chua control only |

## Small API examples

```bash
python examples/quickstart_equilibria.py
python examples/minimal_chua_protocol.py
python examples/custom_system_definition.py
python examples/new_system_workflow_spec.py
python examples/integer_lure_chua_protocol.py
python examples/dynamical_analysis_gallery.py
```

`minimal_chua_protocol.py` writes the explicit command and JSON contract by
default. Add `--run` only when launching the numerical protocol intentionally.

`dynamical_analysis_gallery.py` accepts `--trajectory-csv path/to/trajectory.csv`
for plotting an existing trajectory.

## Rules

When adding an example:

1. import from `hidden_attractors` whenever possible;
2. register new systems through `hidden_attractors.systems`;
3. record a `WorkflowInputSpec` before presenting reusable workflows;
4. write outputs under `outputs/` or require `--output-dir`;
5. document whether the run is a smoke check, a long job, a diagnostic, or a validation helper;
6. update `docs/api_reference.md` when new functions, classes, or methods become part of the release surface.
