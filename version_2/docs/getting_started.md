# Getting Started

`hidden-attractors-fo` has two supported interfaces:

1. the unified CLI, `hidden-attractors`, for reproducible runs; and
2. the Python package, `hidden_attractors`, for registering systems, composing
   workflow specs, and building audited analysis scripts.

For the shortest command list, see [Quick Start](quick_start.md). For the full
symbol inventory, see [API Reference](api_reference.md).

## Installation

Install from PyPI for normal use:

```bash
python -m pip install hidden-attractors-fo
```

Use the package from Python as:

```python
import hidden_attractors
```

Install from a checkout for development (with all extras):

```bash
python -m pip install -e ".[dev,analysis,docs]"
```

## High-level CLI

Copy the abstract contract, fill every required value, then run or inspect that
explicit file:

```bash
hidden-attractors init -e workflow_contract
hidden-attractors inspect-config -c my_workflow.yaml
hidden-attractors run -c my_workflow.yaml
```

Useful inspection and validation checks:

```bash
hidden-attractors inspect systems
hidden-attractors inspect candidates --source path/to/candidates.json
hidden-attractors inspect workflow-requirements
hidden-attractors seed --help
hidden-attractors validate contract
```

The public release surface is the single `hidden-attractors` command.
Historical standalone commands are legacy/deprecated and appear only in
migration notes. The documented seed commands are limited to the implemented
release surface.

## Python API basics

Load a configuration:

```python
from hidden_attractors.workflows.config_loader import load_config

config = load_config("my_workflow.yaml")
```

Retrieve a built-in system:

```python
from hidden_attractors.systems import get_system

system = get_system("chua-nonsmooth")
```

Integrate through the selector:

```python
from hidden_attractors.integrations.selector import integrate

times, states, status = integrate(
    rhs=system.rhs,
    x0=[0.1, 0.0, 0.0],
    q=0.99,
    h=0.01,
    t_final=50.0,
    integrator="efork3",
    system=system,
)
```

Register a new system:

```python
import numpy as np
from hidden_attractors.systems import ChaoticSystem, register_system


def rhs(state, parameters):
    x, y, z = state
    sigma = float(parameters["sigma"])
    rho = float(parameters["rho"])
    beta = float(parameters["beta"])
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


register_system(
    ChaoticSystem(
        name="lorenz63-demo",
        dimension=3,
        rhs=rhs,
        parameters={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        description="Minimal external system registration example.",
    ),
    replace=True,
)
```

Registration alone does not make a full hidden-attractor workflow. To use the
methodology for a new Lur'e-type system, also provide equilibria, a Jacobian,
the scalar Lur'e split `(P, b, r, psi)`, the describing function convention, and
a `WorkflowInputSpec` that records solver, memory, target, classifier, basin,
robustness, and hiddenness inputs.

## Configuration skeleton

Workflows use hierarchical YAML. A minimal fractional configuration records the
system, integrator, memory contract, stages, and output location:

```yaml
experiment:
  name: "fractional-demo"
  output_dir: "outputs/fractional-demo"

system:
  system_id: "chua_fractional_saturation"
  q: 0.9

integrator:
  name: "efork3"
  h: 0.01
  memory_mode: "window"
  memory_policy: "finite_window"
  memory_window_steps: 100

stages:
  seed_search: true
  continuation: true
  final_simulation: true
  hiddenness_tests: false
  basin_slices: false
```

For a reproducible calculation, record the solver, memory policy, horizon,
transient, thresholds, and random seed with the output.

## Official methodology for new Lur'e systems

The release methodology is intentionally conservative:

```text
register system
-> declare numerical contract
-> validate algebra/equilibria/Jacobian/Matignon
-> build DF/Nyquist seed
-> soft precheck
-> continuation
-> post-continuation filtering
-> dynamic reference
-> robustness checks
-> hiddenness tests around all equilibria
-> diagnostics and figures
-> manifest/report
```

DF/Nyquist and continuation only generate and transport candidates. Hiddenness
requires finite neighborhood or basin evidence around all equilibria under the
recorded numerical contract. Diagnostics such as FFT/PSD, 0-1, Poincare, and
Lyapunov estimates are useful but do not certify hiddenness.

## Public API Examples And Completed Validation

| Kind | Command | Use |
| --- | --- | --- |
| Public API example | `python examples/custom_system_definition.py` | Minimal system registry example |
| Public API example | `python examples/new_system_workflow_spec.py` | Records the prerequisites for an auditable reusable workflow |
| Repository validation | `python examples/chua_integer_lure_reference/run_example.py --quick` | Reproduces the complete `q=1` validation route |

Case-specific research records remain outside the installed library and its
public user documentation.

Ordinary runs write to `outputs/`. Promoted evidence belongs under
`validation/`, and promoted figures belong under `library_figures/` through the
central export policy.
