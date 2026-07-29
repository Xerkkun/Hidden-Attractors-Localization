# Public Workflows

`hidden-attractors-fo` supports two independent uses:

1. localization workflows for compatible dynamical systems; and
2. direct characterization of systems, trajectories, or scalar time series.

## Inspect Available Systems

```bash
hidden-attractors inspect systems
hidden-attractors inspect workflow-requirements
```

Programmatically:

```python
from hidden_attractors import get_system, list_systems, requirements_for

print(list_systems())
system = get_system("chua-nonsmooth")
print(requirements_for(system, workflow="characterization"))
```

## Independent Characterization

These commands do not start a hidden-attractor search:

```bash
hidden-attractors lyapunov spectrum --help
hidden-attractors chaos-test zero-one --help
hidden-attractors bifurcation run --help
```

The corresponding public Python functions include
`compute_trajectory_metrics`, `compute_boundedness_metrics`,
`compute_fft_psd`, `detect_poincare_crossings`, `zero_one_test`,
`bifurcation_points_from_trajectories`, `compute_lyapunov_spectrum`, and
`estimate_time_series_lyapunov`.

Every result is a finite-data or finite-time numerical characteristic. These
functions do not require, imply, or certify a hidden-attractor result.

## Localization Route

For a compatible scalar Lur'e system, the public route is:

1. validate model capabilities;
2. construct a describing-function seed;
3. continue the seed to the target system;
4. integrate under explicit numerical settings;
5. compute independent dynamical diagnostics; and
6. run finite equilibrium-neighborhood controls when a hiddenness assessment
   is requested.

The main CLI groups are:

```bash
hidden-attractors seed lure-centered --help
hidden-attractors seed lure-biased --help
hidden-attractors continuation run --help
hidden-attractors validate contract --help
```

A seed or continuation path is an initialization result, not evidence of
hiddenness. Neighborhood controls remain conditional on their recorded radii,
sampling, solver, horizon, memory policy, and classifier.

## Configuration

Copy the non-runnable structural contract, then create an explicit
configuration:

```bash
hidden-attractors init --example workflow_contract
hidden-attractors inspect-config --config my_workflow.yaml
hidden-attractors run --config my_workflow.yaml
```

With no `--example`, `hidden-attractors init` copies the packaged abstract
contract into `configs/examples/` under the current directory. The
effective configuration used for a numerical result should be archived
alongside the software version and output hashes.

## Python Workflow Contracts

```python
from hidden_attractors import (
    WorkflowInputSpec,
    check_system_capability,
    validate_full_workflow_system,
)

check_system_capability(system, "characterization")
validate_full_workflow_system(system)
```

Use the narrowest workflow that answers the numerical question. Characterizing
a supplied trajectory does not require seed generation, continuation, basin
sampling, or any hiddenness classification.
