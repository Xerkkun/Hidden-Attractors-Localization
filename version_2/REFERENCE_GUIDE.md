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
reproducible analysis interfaces while the project is in alpha, and auxiliary
commands are documented for traceability rather than as stable APIs.

| Command | Group | Real options or usage | Documentary status |
|---|---|---|---|
| `hidden-attractors` | Main user command | `run -c/--config`, `run -p/--preset`, `init -e/--example`, `inspect-config -c/--config`, `inspect-config -p/--preset`, `validate-bibliography -m/--manifest --strict --json -o/--markdown-output` | Primary stable user-facing CLI; Python module is internal |
| `hidden-attractors-protocol` | Specialized workflow | Official protocol stage interface | Reproducible protocol interface; alpha |
| `hidden-attractors-robustness-overlay` | Specialized workflow | Robustness overlay workflow | Reproducible analysis workflow; alpha |
| `hidden-attractors-sphere-controls` | Specialized workflow | Equilibrium-ball controls | Reproducible analysis workflow; alpha |
| `hidden-attractors-refined-basin` | Specialized workflow | Basin refinement workflow | Reproducible analysis workflow; alpha |
| `hidden-attractors-strict-target-refinement` | Specialized workflow | Strict target refinement workflow | Reproducible analysis workflow; alpha |
| `hidden-attractors-danca-abm-sphere-controls` | Specialized workflow | Danca ABM sphere controls | Reproducible analysis workflow; alpha |
| `hidden-attractors-fractional-report-run` | Specialized workflow | Fractional report generation workflow | Reproducible report workflow; alpha |
| `hidden-attractors-list-candidates` | Auxiliary/internal | Candidate listing helper | Traceability helper; not a stable API |
| `hidden-attractors-systems` | Auxiliary/internal | System registry inspection helper | Registry helper; not a stable API |
| `hidden-attractors-workflow-requirements` | Auxiliary/internal | Workflow capability checks | Diagnostic helper; not a stable API |
| `hidden-attractors-check-validation` | Auxiliary/internal | Validation-contract checks | Validation diagnostic helper; not a stable API |

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

---

## 7. Testing Suite

The library is backed by 156 tests in `version_2/tests/`:

* **Verification Execution**:
  ```bash
  pytest
  ```
* **Coverage**: Runs unit verification on coordinate models, Lure decompositions, C integrators accuracy, YAML schema parsing, CLI argument overrides, and diagnostics computations.
