# Legacy Scripts

This folder contains research scripts migrated from the old project root. They
are preserved for reproducibility and traceability, but they are not the clean
library interface.

## Current Status

This folder is still required. Do not delete it yet.

The actively supported public entry point `hidden-attractors-unified-chua`
still delegates to `unified_nyquist_hidden_pipeline.py`, and several reports
reference historical runs produced by scripts here.

Installable command shape:

```bash
hidden-attractors-legacy --list
hidden-attractors-legacy extended-search --help
hidden-attractors-extended-search --help
hidden-attractors-danca2017 --help
hidden-attractors-nyquist-pipeline --help
```

Already adapted:

- `unified_nyquist_hidden_pipeline.py` accepts explicit CLI options instead of
  requiring manual `HIDDEN_ATTRACTORS_*` environment variables.
- Heavy basin and EFORK stages are routed through the C-backed path when the
  unified workflow is used with the current wrapper.
- Seed-generation pieces that used to live only here now have public helpers in
  `hidden_attractors.seed_generation`.
- Shared C backend wrappers now live under `hidden_attractors.native`.

Still legacy / not fully migrated:

- `run_extended_search.py`, `danca2017_chua_abm_replication.py`, and related
  exploratory scripts remain compatibility workflows, not the final stable API.
- Several scripts still import SciPy-dependent helpers from
  `chua_initial_cond.py`.
- Some old workflows still contain Python-side exploratory logic that should be
  migrated into small package modules before being treated as stable.

Install their optional dependencies with:

```bash
python -m pip install -e ".[legacy]"
```

The legacy extra requires a Python environment where SciPy is available. If the
base package is running on a newer Python before SciPy wheels exist for that
version, use a separate supported Python environment for these scripts.

When a script here needs new functionality, migrate the reusable part into
`hidden_attractors/` and add an example or workflow wrapper.

## Migration Contract

Legacy scripts may keep fixed Danca/Chua defaults when that is required to
reproduce old artifacts.  New behavior should still follow the package
contract:

1. load or build `hidden_attractors.workflows.WorkflowInputSpec`;
2. write the effective spec next to outputs;
3. document fixed-system assumptions in `--help` and run metadata;
4. delegate reusable calculations to `hidden_attractors/`;
5. keep historical numerical contracts unchanged unless the user requests a
   new run profile explicitly.

Use:

```bash
hidden-attractors-workflow-requirements --help
```

to inspect which inputs a workflow needs before a legacy script is migrated
into a generic command.
