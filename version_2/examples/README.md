# Examples

These examples are the user-facing entry points for the library. They should be
small, runnable from the repository root, and should not launch long numerical
jobs unless the filename and help text make that clear.

## Comandos

```bash
python examples/quickstart_equilibria.py
python examples/list_final_candidates.py
python examples/minimal_chua_protocol.py
python examples/custom_system_definition.py
python examples/new_system_workflow_spec.py
python examples/integer_lure_chua_protocol.py
python examples/dynamical_analysis_gallery.py
python examples/create_robustness_overlay_config.py
python examples/aggregate_existing_robustness_overlay.py outputs/robustness_overlay_c_trajectories_20260517
```

`dynamical_analysis_gallery.py` can also plot a real trajectory from `outputs/`
when passed `--trajectory-csv path/to/trajectory.csv`.

`minimal_chua_protocol.py` writes the explicit C-backed unified-workflow command
and JSON contract by default. Add `--run` only when you want to launch the
numerical protocol.

`custom_system_definition.py` shows how a user can register another chaotic
system through the public `hidden_attractors.systems` API.

`new_system_workflow_spec.py` shows the next step after registration: writing a
`WorkflowInputSpec` that records the solver, classifier, target-reference,
basin, and strict-refinement inputs required before reusable workflows can be
audited.

`integer_lure_chua_protocol.py` is the small order-one example. It exercises
the unified seed and validation stages: a `lure_classical_centered` DF seed,
`ContinuationPlan(lambda_values=...)`, final trajectory, equilibrium-ball
controls, Lyapunov estimate, and reusable plots. The
same functions are intended for other integer-order systems that provide a
manual Lur'e form and the required describing functions.

## Rules

When adding a new example:

1. import from `hidden_attractors` whenever possible;
2. register new systems through `hidden_attractors.systems`;
3. avoid duplicating long workflow logic;
4. write outputs to a new folder under `outputs/` or require
   `--output-dir`;
5. update this README if the example becomes part of the normal workflow.
