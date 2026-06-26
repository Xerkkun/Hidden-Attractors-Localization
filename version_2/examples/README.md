# Examples

These are the runnable examples shipped with `hidden-attractors-fo`. They are
user-facing entry points and should import from `hidden_attractors` rather than
copying workflow internals.

## Official report examples

Run from `version_2`:

```bash
python examples/chua_integer_lure_reference/run_example.py --quick
python examples/chua_nonsmooth_biased_hidden_attractor/run_example.py --quick
python examples/chua_arctan_wu2023/run_example.py --quick
```

| Example | Role | Evidence boundary |
| --- | --- | --- |
| `chua_integer_lure_reference/` | Integer `q=1` Lur'e reference: seed, continuation, trajectory, neighborhood controls, figures | Reproduced integer control only |
| `chua_nonsmooth_biased_hidden_attractor/` | Proposed biased-DF route for non-smooth fractional Chua | Candidate/compatible only under declared local radii; not full Danca reproduction |
| `chua_arctan_wu2023/` | Wu2023 bibliographic lane plus promoted Caputo c590 lane | c590 promoted for local radii `r <= 0.3`; Wu2023 ADM lane remains bibliographic |

## Small API examples

```bash
python examples/quickstart_equilibria.py
python examples/list_final_candidates.py
python examples/minimal_chua_protocol.py
python examples/custom_system_definition.py
python examples/new_system_workflow_spec.py
python examples/integer_lure_chua_protocol.py
python examples/dynamical_analysis_gallery.py
python examples/create_robustness_overlay_config.py
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
