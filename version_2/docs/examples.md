# Examples

Examples are small, runnable entry points that import from `hidden_attractors`
when possible. They write ordinary outputs under `outputs/`; promoted evidence
must be moved through the validation and figure-manifest workflow.

## Official report examples

```bash
cd version_2
python examples/chua_integer_lure_reference/run_example.py --quick
python examples/chua_nonsmooth_biased_hidden_attractor/run_example.py --quick
python examples/chua_arctan_wu2023/run_example.py --quick
```

| Directory | What it demonstrates | Evidence boundary |
| --- | --- | --- |
| `examples/chua_integer_lure_reference/` | Integer `q=1` Lur'e seed, continuation, final trajectory, hiddenness controls, figures, Lyapunov diagnostic | Reproduced reference for the integer route only |
| `examples/chua_nonsmooth_biased_hidden_attractor/` | Biased describing-function methodology for a non-smooth fractional Chua candidate | Candidate evidence under tested local radii; not full Danca reproduction |
| `examples/chua_arctan_wu2023/` | Wu2023 arctan bibliographic lane plus proposed Caputo full-history c590 lane | Non-promoted/requires review; not verified hiddenness |

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

`minimal_chua_protocol.py` writes a contract and command by default. Add `--run`
only when you intend to launch the numerical workflow.

`dynamical_analysis_gallery.py` can also plot an existing trajectory:

```bash
python examples/dynamical_analysis_gallery.py --trajectory-csv path/to/trajectory.csv
```

## Rules for new examples

1. Import from `hidden_attractors` rather than duplicating workflow logic.
2. Register new models through `hidden_attractors.systems`.
3. Provide a `WorkflowInputSpec` before claiming a reusable hiddenness workflow.
4. Write outputs to a unique folder under `outputs/` or require `--output-dir`.
5. State whether the script is a smoke example, a long run, a diagnostic, or a
   validation helper.
6. Link to [API Reference](api_reference.md) when adding new public functions or
   methods used by the example.

