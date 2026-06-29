# Getting Started

`hidden-attractors-fo` has two supported interfaces:

1. the unified CLI, `hidden-attractors`, for reproducible runs; and
2. the Python package, `hidden_attractors`, for registering systems, composing
   workflow specs, and building audited research scripts.

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

For release testing, install from TestPyPI:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps hidden-attractors-fo
```

Install from a checkout for development (with all extras):

```bash
python -m pip install -e ".[dev,analysis,docs,legacy]"
```

## High-level CLI

Run or inspect built-in presets:

```bash
hidden-attractors run -p chua_integer
hidden-attractors run -p chua_fractional
hidden-attractors run -p chua_arctan
hidden-attractors inspect-config -p chua_fractional
```

Initialize an editable example configuration:

```bash
hidden-attractors init -e chua_fractional
hidden-attractors run -c configs/examples/chua_fractional_centered_lure_df.yaml
```

Useful inspection and validation checks:

```bash
hidden-attractors inspect systems
hidden-attractors inspect candidates
hidden-attractors inspect workflow-requirements
hidden-attractors seed --help
hidden-attractors validate contract --allow-pending
hidden-attractors validate bibliography
hidden-attractors validate release-readiness --submission-strict
```

The public release surface is the single `hidden-attractors` command.
Historical standalone commands are legacy/deprecated and should appear only in
migration notes. Machado/FDF seed routes remain theory/internal planned support
and are not exposed as public CLI commands in this release.

## Python API basics

Load a configuration:

```python
from hidden_attractors.workflows.config_loader import load_config

config = load_config("configs/examples/chua_fractional_centered_lure_df.yaml")
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
  q: 0.9998

integrator:
  name: "efork3"
  h: 0.01
  memory_mode: "window"
  memory_policy: "finite_window"
  memory_window_steps: 4000

stages:
  seed_search: true
  continuation: true
  final_simulation: true
  hiddenness_tests: false
  basin_slices: false
```

For full Caputo validation, use a full-history policy where required and record
all horizons, burn-in windows, radii, sample counts, thresholds, and random
seeds.

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

## Example lanes

| Lane | Command | Use |
| --- | --- | --- |
| Integer Chua reference | `python examples/chua_integer_lure_reference/run_example.py --quick` | First example for the complete seed-continuation-hiddenness workflow at `q=1` |
| Non-smooth fractional BDF | `python examples/chua_nonsmooth_biased_hidden_attractor/run_example.py --quick` | Proposed biased-DF route for fractional Chua; evidence remains contract-limited |
| Arctan Wu2023/c590 | `python examples/chua_arctan_wu2023/run_example.py --quick` | Separates Wu2023 bibliographic reproduction from one smooth c590 Caputo lane reported only as local/radius-limited finite-time evidence |
| Custom registration | `python examples/custom_system_definition.py` | Minimal system registry example |
| Workflow spec | `python examples/new_system_workflow_spec.py` | Shows the next layer needed before reusable workflows are auditable |

Ordinary runs write to `outputs/`. Promoted evidence belongs under
`validation/`, and promoted figures belong under `library_figures/` through the
central export policy.
