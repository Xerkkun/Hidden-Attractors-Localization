# Getting Started with Hidden Attractors Localization

`hidden-attractors-fo` provides two main interfaces: a high-level command-line interface (CLI) for running standard pre-packaged workflows, and a low-level Python API for custom scripting and custom integrations.

---

## 1. High-Level Interface (Nivel Sencillo)

The CLI offers a simple entry point to run presets, initialize custom folders, and inspect configuration trees.

### Running Presets

To run a built-in workflow preset:
```bash
hidden-attractors run --preset chua_integer
hidden-attractors run --preset chua_fractional
hidden-attractors run --preset chua_arctan_only_fractional
```

### Initializing Examples

To copy a pre-packaged template config to your local working directory:
```bash
hidden-attractors init --example chua_fractional
```

### Inspecting Configurations

To preview the fully normalized effective configuration mapping (including defaults) before execution:
```bash
hidden-attractors inspect-config --preset chua_fractional
```

---

## 2. Low-Level Python API (Nivel Bajo)

For custom scripting, you can load configurations, fetch systems, and run integrations programmatically.

### Loading Configuration
```python
from hidden_attractors.workflows.config_loader import load_config

# Load, normalize, and validate a configuration YAML
config = load_config("configs/examples/chua_fractional_centered_lure_df.yaml")
```

### Retrieving Systems
```python
from hidden_attractors.systems import get_system

# Get a system definition from the registry (e.g., Chua's system with arctan nonlinearity)
system = get_system("chua-arctan")
```

### Performing Integrations
```python
from hidden_attractors.integrations.selector import integrate

# Run a numerical integration using the unified selector
times, states, status = integrate(
    rhs=system.rhs,
    x0=[0.1, 0.0, 0.0],
    q=0.99,
    h=0.01,
    t_final=50.0,
    integrator="efork3",
    system=system
)
```

---

## 3. Configuration Guide

### Activating/Deactivating Stages
You can control which parts of the workflow run using flags in the `stages` section of your configuration YAML:
```yaml
stages:
  seed_search: true
  continuation: true
  final_simulation: true
  hiddenness_tests: false
  basin_slices: false
  bifurcation: false
  attractor_only: false
```

### Changing the Integrator
To change the numerical integrator, edit the `integrator.name` parameter. The selector validates compatibility with the fractional order `q` automatically:
```yaml
integrator:
  name: rk4  # allowed for q = 1.0; use 'abm' or 'efork3' for q < 1.0
  h: 0.01
```

### Activating Memory Window
Windowed memory is required for long fractional-order simulations where full Caputo history is computationally prohibitive. Configure it under the `integrator` section:
```yaml
integrator:
  memory_mode: window        # Options: full, window, none
  memory_policy: finite_window
  memory_window_steps: 4000  # Number of history steps to keep
```
You can also specify `memory_window_time` (e.g. `2.0` seconds), which will be converted to steps using `memory_window_steps = round(memory_window_time / h)`.

### Running specific workflows

#### Running Attractor Only
Set `run_attractor_only` (or `stages.attractor_only`) to `true` to integrate the system from an initial condition without running seed search or continuation.

#### Running Bifurcation Sweeps
Configure the `bifurcation` section in the YAML, then run the bifurcation workflow. You can sweep parameters like `beta` or the fractional order `q`:
```yaml
stages:
  bifurcation: true
bifurcation:
  parameter: q
  values:
    min: 0.95
    max: 1.0
    n: 50
```

#### Running Basin Slices
Generate 2D slices of the basins of attraction by enabling `basin_slices` (or `stages.basin_slices`):
```yaml
stages:
  basin_slices: true
basin:
  planes: ["xy", "xz"]
  grid_n: 100
  around_equilibria: true
```

---

## 4. Methodological Scope & Directories

> [!IMPORTANT]
> **Heuristics vs Proof**: Describing Function (DF), Lur'e decomposition, and Machado-family analysis only construct **candidate seeds** for starting continuation. They are heuristics and **do not prove** that an attractor is mathematically hidden. Proof of localization requires exhaustive neighborhood probing and basin slice tests.

> [!NOTE]
> **Outputs Directory**: Ordinary execution results, plots, summaries, and data files are saved under `outputs/`.
> The `validation/` directory is reserved exclusively for promoted validation evidence and final benchmark records, never for routine output files.
