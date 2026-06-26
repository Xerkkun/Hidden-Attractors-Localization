# Unified Reference Guide: Hidden Attractors Fractional Order Library

This document is the master reference for the unified `hidden_attractors` library (located under `version_2/`). It details all system registries, numerical integrators, analysis modules, plotting capabilities, configuration parameters, and execution workflows.

---

## 1. Library Architecture & Stability Tiers

The library separates user-facing interfaces from experimental features and internal engines:
* **Stable API** (`stable`): Core system definitions, registry utilities, and I/O managers. (e.g. `hidden_attractors.systems`, `hidden_attractors.models`).
* **Experimental API** (`experimental`): High-level workflows, numerical integration wrappers, and diagnostic analyzers (e.g. `hidden_attractors.workflows`, `hidden_attractors.analysis`, `hidden_attractors.plotting`).
* **Internal API** (`internal`): Compiled C-backend bindings, parallel worker pools, and path resolution caches (e.g. `hidden_attractors.native`, `hidden_attractors.cli`).

The primary stable user-facing command-line interface is the installed
`hidden-attractors` command. The Python implementation module
`hidden_attractors.cli` remains internal. Specialized workflow commands are
reproducible analysis interfaces with narrower support guarantees than the primary CLI, and auxiliary
commands are documented for traceability rather than as stable APIs.

| Command / Subcommand | Group | Real options or usage | Documentary status |
|---|---|---|---|
| `hidden-attractors run` | Running Workflows | `-c/--config`, `-p/--preset` | Primary stable entry point |
| `hidden-attractors init` | Setup | `-e/--example` | Copy configuration templates |
| `hidden-attractors inspect-config` | Configuration | `-c/--config`, `-p/--preset` | Previews normalized config |
| `hidden-attractors inspect` | Registry & Candidates | `candidates`, `systems`, `workflow-requirements` | Inspection utility |
| `hidden-attractors validate` | Validation Contracts | `contract`, `bibliography` | Numerical/bibliographical validation |
| `hidden-attractors protocol` | Caputo Protocol | `generate-seeds`, `soft-precheck`, `continue`, `filter-survivors`, `build-reference`, `robustness`, `hiddenness`, `diagnostics` | Stage-by-stage protocol engine |
| `hidden-attractors robustness` | Robustness | `overlay` | Robustness overlay sweep |
| `hidden-attractors hiddenness` | Neighborhood Probing | `sphere-controls`, `strict-target-refinement` | Neighborhood analysis |
| `hidden-attractors basin` | Basins | `refined`, `strict-target-refinement` | Basin-of-attraction analysis |
| `hidden-attractors published` | Replication | `danca-abm-sphere-controls` | Reproduces published Danca papers |
| `hidden-attractors report` | Reporting | `fractional-run` | Automated scientific report generation |

For older standalone commands (e.g. `hidden-attractors-protocol`, `hidden-attractors-sphere-controls`), see the [CLI Migration Guide](docs/cli_migration_legacy_entrypoints.md) as they are no longer installed as public entry points.

See [Thesis Claims Matrix](THESIS_CLAIMS.md) for the current claim matrix and the distinction between reproduced, rejected, candidate, non-certified and pending results.

The synchronized manual metadata are defined in [docs/manual_manifest.yaml](docs/manual_manifest.yaml); scientific claims remain governed by [THESIS_CLAIMS.md](THESIS_CLAIMS.md).

For a complete user-facing description of installation, CLI usage, examples, outputs, evidence labels and limitations, see [USER_MANUAL.md](USER_MANUAL.md).



---

## 2. Configuration Reference (Hierarchical YAML Schema)

All workflows are controlled by a uniform hierarchical YAML schema. Below is a comprehensive list of all sections and keys:

```yaml
# ── EXPERIMENT METADATA ───────────────────────────────────────────────────
experiment:
  name: "Chua Experiment"                 # Name of the run
  description: "Description of run"       # Long description
  output_dir: "outputs/chua"              # Destination directory for CSVs/plots
  run_id: "auto"                          # Unique run identifier (or "auto" for timestamp)
  random_seed: 42                         # Reproducibility seed

# ── SYSTEM DEFINITION ─────────────────────────────────────────────────────
system:
  system_id: "chua_fractional_saturation" # Registered system key
  q: 0.998                                # Fractional order (defaults to system preset if null)
  parameters:                             # Parameter overrides passed to vector fields
    alpha: 8.4562
    beta: 12.0732
    gamma: 0.0052
    m0: -0.1768                           # Outer slope (saturation)
    m1: -1.1468                           # Inner slope (saturation)
    m: 0.4                                # parameter m (arctan)
    n: 1.2                                # parameter n (arctan)

# ── API STABILITY MODES ────────────────────────────────────────────────────
modes:
  transfer_mode: "fractional"             # "fractional" or "integer" transfer function
  seed_mode: "fractional"                 # Order for seed generation
  continuation_mode: "fractional"         # Order for parameter continuation
  dynamics_mode: "system"                 # Order for final simulation ("system", "integer", "fractional")

# ── NUMERICAL INTEGRATOR ──────────────────────────────────────────────────
integrator:
  name: "efork3"                          # Solver name: "abm", "efork3", "heun", "rk4", "adm_wu2023"
  h: 0.001                                # Step size (delta t)
  memory_mode: "window"                   # Caputo memory type: "full" or "window"
  memory_policy: "finite_window"          # Alias: "full_caputo" or "finite_window"
  memory_window_steps: 400                # Size of window length in steps
  use_c_backend: true                     # Toggle compiled C execution backends
  allow_python_fallback: true             # Fallback to Python solvers if C shared library is missing

# ── WORKFLOW STAGES (DISPATCHER) ─────────────────────────────────────────
stages:
  seed_search: true                       # Run describing function balance-equation scans
  continuation: true                      # Sweep eta (homotopy) from seed to true system parameters
  final_simulation: true                  # Integrate the verified attractor trajectory
  hiddenness_tests: false                 # Test attractor contacts around equilibrium points
  sphere_tests: false                     # Probe neighborhood spheres of size delta
  basin_slices: false                     # Generate 2D basin-of-attraction slices
  bifurcation: false                      # Generate parameter sweep bifurcation diagrams
  attractor_only: false                   # Bypasses Lure/Continuation to simulate directly from ICs

# ── DESCRIBING FUNCTION SEED SEARCH ───────────────────────────────────────
seed_search:
  strategy: "k_phi"                       # Frequency strategy: "k_phi", "imw_gain", "nyquist_df"
  construction: "modal"                   # IC construction: "modal" or "closed_form_integer"
  branch_index: 0                         # Index of candidate seed to promote (0 = best)
  omega_min: 0.01                         # Minimum search frequency
  omega_max: 20.0                         # Maximum search frequency
  amplitude_min: 0.01                     # Minimum search amplitude
  amplitude_max: 20.0                     # Maximum search amplitude
  grid_size_omega: 200                    # Frequency grid resolution
  grid_size_amplitude: 200                # Amplitude grid resolution
  root_refinement: true                   # Refine grid roots using numerical solvers
  df_residual_tol: 1e-8                   # Residual tolerance for root acceptance
  describing_function_mode: "auto"        # "wu2023" (analytical arctan) or "auto" (numerical quadrature)
  seed_sign_convention: "kuznetsov"       # Direction convention: "kuznetsov"
  seed_theta: 0.0                         # Custom phase shift

# ── STEADY-STATE SIMULATION ───────────────────────────────────────────────
simulation:
  t_final: 300.0                          # Integration duration
  t_burn: 120.0                           # Transients time discarded before calculations
  initial_condition: [0.1, 0.1, 0.1]      # Set when attractor_only is active

# ── EARLY STOPPING ENGINE ─────────────────────────────────────────────────
early_stop:
  enabled: true
  divergence_enabled: true
  divergence_norm: 80.0                   # Norm threshold indicating divergence
  divergence_consecutive_steps: 5
  equilibrium_enabled: true
  equilibrium_tol: 1e-3                   # Distance indicating collapse to equilibrium
  equilibrium_derivative_tol: 1e-4        # Derivative tolerance for equilibrium check
  equilibrium_consecutive_steps: 200

# ── PLOT VISUALIZATION ────────────────────────────────────────────────────
plots:
  enabled: true                           # Master toggle
  save_figures: true                      # Save PNG files to disk
  attractor: true                         # 3D Phase spaces + 2D projections
  timeseries: true                        # Time-series evolution (t vs coordinates)
  transfer: true                          # Nyquist transfer functions
  describing_function: true               # Describing function overlay curves
  continuation: true                      # Parameter continuation tracking
  sphere_tests: true                      # Neighborhood spheres
  basin: true                             # 2D Basin slice grids
  bifurcation: true                       # Bifurcation diagrams
  matignon: true                          # Matignon stability plane
```

---

## 3. Registered Chaotic Systems

Defined in [`hidden_attractors/systems/builtins.py`](hidden_attractors/systems/builtins.py), systems are retrieved via `get_system(name)`. Dotted / legacy names are supported via aliases:

| Registry Key | System Variant | Default Parameters | Default $q$ | Lur'e Split Support |
| :--- | :--- | :--- | :--- | :--- |
| **`chua-nonsmooth`** | Non-smooth piecewise saturation Chua | $\alpha=8.4562, \beta=12.0732, \gamma=0.0052, m_0=-0.1768, m_1=-1.1468$ | `0.9998` | Yes |
| **`chua-arctan`** | Chua system with arctan nonlinearity | $\alpha=8.4562, \beta=12.0732, \gamma=0.0052, a_1=0.4, a_2=1.2, \rho=1.0$ | `0.95` | Yes |
| **`fractional_chua_arctan_wu2023`** | Exact Wu et al. parameters | $\alpha=8.4562, \beta=12.0732, \gamma=0.0052, m=0.4, n=1.2$ | `0.95` | Yes |

*Aliases mapped: `chua_integer_saturation` $\rightarrow$ `chua-nonsmooth`, `chua_fractional_saturation` $\rightarrow$ `chua-nonsmooth`, `chua_fractional_arctan` $\rightarrow$ `chua-arctan`.*

*(Note: Chua arctan c590 is promoted as finite local-radius hiddenness evidence for `r <= 0.3`; this is not a global basin proof.)*

---

## 4. Integrators Compatibility Matrix

Numerical integration goes through the selector [`hidden_attractors/integrations/selector.py`](hidden_attractors/integrations/selector.py#L119-L195) to validate compatibility with fractional order $q$:

* **Fractional-Order ($0 < q < 1$)**:
  * `efork3` (or `efork`): Predictor-corrector Caputo method with fractional order (uses Numba/C-backend).
  * `abm`: Classical Adams-Bashforth-Moulton method.
  * `adm_wu2023`: Localized Adams-Moulton method for Caputo systems.
* **Integer-Order ($q = 1.0$)**:
  * `heun`: Second-order Runge-Kutta.
  * `rk4`: Fourth-order Runge-Kutta.
  * `efork_q1`: Special integer limit of the EFORK solver.
  * *Note: Requesting `efork3` at $q=1.0$ will trigger a warning and automatically redirect to the `efork_q1` path.*

---

## 5. Diagnostic & Stability Analysis Primitives

Located under `hidden_attractors/analysis/`, these functions operate on simulated trajectories:

* **Lyapunov ExponentQR Reorthonormalisation**:
  * Wrapper [`integer_system_lyapunov_exponents(system, x0, h, t_final)`](hidden_attractors/analysis/lyapunov.py#L209-L274) uses Benettin QR factorization to compute Lyapunov spectrum.
* **0-1 Test for Chaos**:
  * Gottwald & Melbourne method. Approximates asymptotic growth rates. Returns $\approx 1.0$ for chaos and $\approx 0.0$ for periodic behavior.
* **Spectral Entropy**:
  * Computes the normalized Shannon entropy of the FFT power spectral density.
* **Zero-Crossing & Peak Ratio**:
  * Extracts dominant frequency and calculates amplitude drift to classify trajectories into `chaotic_like`, `periodic_like`, or `equilibrium_collapsed`.
* **Matignon Stability**:
  * Classifies stability of equilibrium points in fractional order by evaluating eigenvalues $\lambda_i$ and verifying if $|\arg(\lambda_i)| > q\pi/2$.

---

## 6. Visualization & Plotting Suite

All visualization functions are exported in `hidden_attractors.plotting`:

* **`plot_phase_space(trajectory, path, ...)`**: Renders interactive 3D trajectories.
* **`plot_phase_projections(trajectory, path, ...)`**: Renders 2D projections (xy, xz, yz side-by-side).
* **`plot_time_series(trajectory, path, ...)`**: Shows coordinates evolution over time.
* **`plot_lure_nyquist_describing_function(...)`**: Nyquist curves overlaid with $-1/N(A)$ describing function loci.
* **`plot_basin_slice_file(plane, u, v, mat, ...)`**: Color-coded attraction basin maps (stable equilibria vs. chaotic attractors vs. divergence).
* **`plot_bifurcation_diagram(points, path, ...)`**: Extrema sweeps showing parameter bifurcations.
* **`plot_matignon_plane(...)`**: Stability regions plane layout.
* **`export_figure(fig, figure_id, kind, metadata_dict, ...)`**: Central export function saving PNG/PDF to `library_figures/` and updating manifests.

All figure exports are subject to the [Figure Export Policy](docs/figure_export_policy.md) to guarantee reproducibility.

---

## 7. Testing Suite

The library is backed by a robust test suite (at the current thesis-freeze audit, the suite reports 797 passed tests and 34 skipped tests; future runs should be checked against `validation/freeze_audit/`):


* **Verification Execution**:
  ```bash
  pytest
  ```
* **Coverage**: Runs unit verification on coordinate models, Lure decompositions, C integrators accuracy, YAML schema parsing, CLI argument overrides, and diagnostics computations.
