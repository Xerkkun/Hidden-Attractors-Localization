# User Manual

This is the primary user manual for the `hidden-attractors-fo` research library. It provides installation instructions, CLI workflows, reproducible examples, and verification rules for identifying and auditing hidden attractor candidates in integer- and fractional-order systems.

For related repository documents and evidence logs, see:
- **Official Claims Matrix**: [THESIS_CLAIMS.md](THESIS_CLAIMS.md)
- **Quick Start Guide**: [docs/quick_start.md](docs/quick_start.md)
- **Figure Export Policy**: [docs/figure_export_policy.md](docs/figure_export_policy.md)
- **Dependency Policy**: [docs/dependency_policy.md](docs/dependency_policy.md)
- **Official Test Freeze Audit**: [validation/freeze_audit/](validation/freeze_audit/)

---

## 1. Purpose and scientific scope

The library is designed to generate, transport, simulate, audit, and classify candidates for hidden attractors in commensurate fractional-order systems compatible with the scalar Lur'e representation.

### Mathematical Formulation

The base model of a commensurate fractional-order Lur'e system is defined as:

$$\sideset{^C}{_t^q}{\operatorname{D}} X = P X + b \psi(r^T X), \quad \sigma = r^T X$$

Where:
- $X \in \mathbb{R}^n$ is the state vector.
- $P \in \mathbb{R}^{n \times n}$ is the linear system matrix.
- $b, r \in \mathbb{R}^n$ are the input and output vectors.
- $\psi: \mathbb{R} \to \mathbb{R}$ is the scalar nonlinearity.
- $\sideset{^C}{_t^q}{\operatorname{D}}$ represents the Caputo fractional derivative operator of order $q$.
- For $q = 1.0$, the system is an ordinary differential equation (ODE).
- For $0 < q < 1.0$, the system behaves as a fractional-order dynamical system with power-law memory.

### Linear Transfer Function Conventions

The linear transfer function $W_q(s)$ in the Laplace domain is defined as:

$$W_q(s) = r^T (s^q I - P)^{-1} b$$

For steady-state harmonic balance and describing function (DF) sweeps, the complex frequency evaluation uses the spectral parameter $\lambda$:

$$\hat{W}_q(\lambda) = r^T (\lambda I - P)^{-1} b, \quad \lambda = (j \omega)^q = \omega^q e^{j q \pi / 2}$$

> [!IMPORTANT]
> When $q < 1.0$, it is strictly prohibited to evaluate the frequency response using the integer-order convention $W(j\omega)$ as a substitute for $(j\omega)^q$.

### Scope of numerical evidence

The library records finite-time numerical evidence under explicit solver, tolerance, memory, and neighborhood contracts. Describing functions, continuation, plots, Lyapunov estimates, and basin slices are diagnostic tools; final labels are assigned only through the documented validation workflow.

---

## 2. Installation

To install the library in editable mode, run the following command from the repository root:

```bash
python -m pip install -e version_2
```

For full development, document compiling, and validation testing, it is recommended to install all development, analysis, and legacy dependencies:

```bash
python -m pip install -e "version_2[dev,analysis,legacy]"
```

### Extras Packages

You can install specific groups of dependencies separately:

```bash
python -m pip install -e "version_2[dev]"      # Pytest, coverage, benchmarks
python -m pip install -e "version_2[analysis]" # Chaos diagnostics (antropy, nolds)
python -m pip install -e "version_2[docs]"     # MkDocs and documentation compilers
python -m pip install -e "version_2[legacy]"   # Legacy dependency support (PyYAML)
```

### Requirements

- **Python**: Version `>= 3.11`. The package is automatically tested on Python versions `3.11`, `3.12`, and `3.13` in the CI pipeline (see [Dependency Policy](docs/dependency_policy.md)).

### Installation Verification

Run the following commands to verify that the environment compiles and runs successfully:

```bash
hidden-attractors --help
cd version_2
python -m compileall hidden_attractors examples tests tools/cli
python -m pytest -q
```

> [!NOTE]
> The official results of the latest release audit (freeze audit) are stored under [validation/freeze_audit/](validation/freeze_audit/), reporting **797 passed** and **34 skipped** tests. Local test counts may vary if custom test files are added.

---

## 3. Repository structure

| Path | Purpose |
| :--- | :--- |
| `hidden_attractors/` | Core library package containing models, solvers, and utilities. |
| `hidden_attractors/cli/` | Unified command-line interface dispatcher. |
| `hidden_attractors/models/` | Core vector fields, Jacobians, and nonlinearities. |
| `hidden_attractors/systems/` | Extensible chaotic-system registry and Lur'e splits. |
| `hidden_attractors/seed_generation/` | Describing function (DF) and Nyquist seed generators. |
| `hidden_attractors/integrations/` | Numerical solvers (RK4, Heun, ABM, EFORK-3). |
| `hidden_attractors/workflows/` | Core pipelines (continuation, robustness, basins). |
| `hidden_attractors/plotting/` | Canonical plotting and figure export tools. |
| `configs/examples/` | Configuration YAML templates for presets and systems. |
| `examples/` | Runnable python examples for testing API usage. |
| `validation/` | Promoted validation evidence, logs, and LaTeX reports. |
| `library_figures/` | Promoted figures and plots metadata (manifest). |
| `docs/` | User manuals, technical documents, and reports. |
| `THESIS_CLAIMS.md` | Official claims matrix documenting what is scientifically proven. |

> [!WARNING]
> Historical scripts under `tools/legacy/` or temporary folders are kept only for compatibility or archive purposes. They do not constitute public API routes.

---

## 4. Public CLI

The only public CLI command installed in the user environment is:

```bash
hidden-attractors
```

Grouped subcommands are structured as follows:

| Task | Command |
| :--- | :--- |
| Show help | `hidden-attractors --help` |
| Run preset | `hidden-attractors run -p chua_integer` |
| Run YAML | `hidden-attractors run -c path/to/config.yaml` |
| Initialize config | `hidden-attractors init -e chua_fractional` |
| Inspect systems | `hidden-attractors inspect systems` |
| Inspect candidates | `hidden-attractors inspect candidates` |
| Inspect workflow requirements | `hidden-attractors inspect workflow-requirements` |
| Validate contract | `hidden-attractors validate contract --allow-pending` |
| Seed centered Lur’e DF | `hidden-attractors seed lure-centered --help` |
| Seed biased Lur’e DF | `hidden-attractors seed lure-biased --help` |
| Continuation | `hidden-attractors continuation run --help` |
| Multiparameter continuation | `hidden-attractors continuation multiparameter --help` |
| Hiddenness sphere controls | `hidden-attractors hiddenness sphere-controls --help` |
| Basin refinement | `hidden-attractors basin refined --help` |
| Robustness overlay | `hidden-attractors robustness overlay --help` |
| Bifurcation | `hidden-attractors bifurcation run --help` |
| Lyapunov | `hidden-attractors lyapunov compute --help` |
| Chaos tests | `hidden-attractors chaos-test --help` |

> [!NOTE]
> Legacy entry points were removed from the public installed command surface. Use the grouped CLI instead.

---

## 5. Minimal examples

Below are quick commands to execute and inspect calculations:

### A. Registry Inspection
```bash
hidden-attractors --help
hidden-attractors inspect systems
hidden-attractors inspect candidates
hidden-attractors validate contract --allow-pending
```

### B. Execution using a Preset
```bash
hidden-attractors run -p chua_integer
```

### C. Execution using a Custom YAML
```bash
hidden-attractors init -e chua_fractional
hidden-attractors run -c configs/examples/chua_fractional_centered_lure_df.yaml
```

### Expected Output
- **Terminal Summary**: Displays real-time details of the seed, parameters, continuation steps, solver diagnostics, and validation decisions.
- **JSON Summary**: A structured file containing run parameters, execution timestamps, and final verdict.
- **CSV Data**: State trajectories (time, $x$, $y$, $z$) stored in the configured output directory.
- **Figures**: Automatically exported PNG/PDF plots under the `library_figures/` directory if they are promoted.

---

## 6. Reproducible example 1: Chua integer

The integer-order Chua system acts as a baseline case to validate the Lur'e split, frequency analysis, parameter continuation, and diagnostic modules under a memoryless ODE framework.

- **System**: Chua system with non-smooth piecewise-linear saturation nonlinearity.
- **Order**: $q = 1.0$ (ordinary differential equation).
- **Status**: reproduced integer reference.
- **Command**:
  ```bash
  hidden-attractors run -p chua_integer
  ```
- **Expected Output**: A stable chaotic attractor trajectory generated from a continuation seed starting from describing function analysis, with 0 contacts detected in the neighborhoods of the stable equilibria (hence classified as a hidden attractor).

> [!WARNING]
> The integer Chua case validates the integer-order route and Lur’e/DF pipeline, but it does not validate fractional hiddenness.

---

## 7. Reproducible example 2: Chua nonsmooth BDF

The Danca 2017 fractional Chua case is used as a partial reference implementation. The published article does not report all numerical values required for full independent trajectory reproduction, including the exact seed and hidden-attractor initial condition. The library therefore records equations, equilibria, local stability, configured Caputo controls, and validation outcomes under its own numerical contract.

- **System**: Chua system with piecewise-linear saturation nonlinearity.
- **Order**: $q = 0.9998$ (Caputo fractional order).
- **Status**: partial reference implementation (not fully reproduced).
- **Example Run (Nearby Candidate)**:
  ```bash
  hidden-attractors run -c configs/examples/chua_fractional_biased_lure_df.yaml
  ```
- **Interpretation**: 
  - The BDF route is a heuristic for generating periodic seeds. A successful continuation to the target system does not mathematically prove hiddenness.
  - One nearby candidate evaluated under the library's own contract, `danca2017_nearby_saturation_candidate_q09998` (branch_0), registered **1305 autoexcited contacts** and was officially classified as `self_excited_contact_detected` (rejected).
- **Validation artifacts**: Refer to [validation/00_manifest/validation_manifest.json](validation/00_manifest/validation_manifest.json) and [validation/09_hiddenness_tests/hiddenness_tests_validation_summary.json](validation/09_hiddenness_tests/hiddenness_tests_validation_summary.json) if available.

---

## 8. Pending/non-certified example: Chua arctan

The Wu 2023 arctan Chua case is implemented at the model/algebra level and includes documented reference inputs. Full independent reproduction of the published attractor workflow is not claimed because some seed-generation and sweep data are not reported and the library uses a separate Caputo-compatible validation route.

- **System**: Chua system with $\psi(x) = \rho (\frac{2}{\pi} \arctan(m x) - \frac{2}{\pi} \arctan(n x))$.
- **Order**: $q < 1.0$ (typically $q=0.99$).
- **Status**: partial algebraic implementation; validation complete pending.
- **Pending verification steps**:
  1. Define Lur'e split vectors.
  2. Evaluate fractional transfer function $W_q(s)$.
  3. Formulate the arctan describing function.
  4. Perform Nyquist condition loops.
  5. Run homotopic parameter continuation.
  6. Simulate the system with Caputo ABM/EFORK solvers.
  7. Run sphere probing around all equilibria.
  8. Perform conservative classification.

---

## 9. YAML configuration format

Workflows are parameterized using a hierarchical, structured YAML schema. 

> [!WARNING]
> Do not use the old flat schema.

### Core YAML Sections

| Key | Meaning |
| :--- | :--- |
| `experiment` | Run identifier, description, output directory, and random seed. |
| `system` | Registered system ID, order $q$, and parameter overrides. |
| `modes` | Execution modes for seed search, transfer functions, and dynamics. |
| `integrator` | Solver name (`efork3`, `abm`), step size $h$, and memory window policy. |
| `stages` | Toggles for running seed search, continuation, and diagnostics. |
| `seed_search` | Describing function amplitude, frequency, and grid bounds. |
| `simulation` | Integration time $t_{final}$ and transients burn time $t_{burn}$. |
| `sphere_tests` | Sampling size and radii around equilibria. |
| `basin` | 2D basin slice grid resolutions and planes. |
| `bifurcation` | Parameter sweep bifurcation boundaries. |
| `plots` | Master toggles for phase space, Nyquist, and basin plots. |
| `attractor_plots` | Attractor visual projection settings. |

### Minimal Schematic Configuration

```yaml
experiment:
  name: "Chua Saturation Run"
  output_dir: "outputs/chua"
  random_seed: 42

system:
  system_id: "chua_fractional_saturation"
  q: 0.9998

integrator:
  name: "efork3"
  h: 0.001
  memory_mode: "window"
  memory_window_steps: 400

stages:
  seed_search: true
  continuation: true
  final_simulation: true
  hiddenness_tests: true
```

---

## 10. Output files and expected results

Execution runs produce structured outputs separated by tier:

| Output | Type | Description |
| :--- | :--- | :--- |
| **JSON Summary** | Metadata | Machine-readable description containing step, solver, parameters, and verdict. |
| **CSV Trajectories** | Numerical | Trajectory coordinates ($t, x, y, z$) of simulation runs. |
| **Figures** | Visual | Automatically formatted plots (phase-spaces, basins, Nyquist). |
| **Validation Manifest** | Status | JSON files verifying contract compliance (e.g. `validation_manifest.json`). |

- **Execution output**: Stored locally in `outputs/` or configured folders for inspection. Unpromoted systems, like the arctan system (`outputs/chua_fractional_arctan/`), write their outputs to the `outputs/` directory.
- **Promoted evidence**: Promoted validation evidence, such as the saturation systems (`validation/chua_integer_saturation/` and `validation/chua_fractional_saturation/`), are stored under `validation/`. Figures are saved in `library_figures/` following the guidelines in [docs/figure_export_policy.md](docs/figure_export_policy.md).
- **Legacy artifacts**: Frozen files kept under legacy namespaces.


---

## 11. Evidence states and hiddenness labels

The library enforces a separation between numerical evidence labels and official thesis claim statuses.

### Evidence Labels
- `hiddenness_supported_under_tested_neighborhoods`: Attraction basin does not overlap with any tested neighborhood of any equilibrium.
- `compatible_with_hiddenness_under_tested_radii`: No contacts detected, but tested parameters/radii are incomplete.
- `self_excited_contact_detected`: An equilibrium trajectory reached the attractor; the attractor is self-excited.
- `hiddenness_inconclusive`: Divergence or integration failure.
- `candidate_rejected`: Attractor collapsed to equilibrium or diverged.

### Claim Status (THESIS_CLAIMS.md)
- `probado`: Verified by extensive tests and formal contract.
- `reproducido`: Exists in literature and is reproduced under a verified config.
- `rechazado`: Candidate rejected by neighborhood test.
- `candidato`: Promised seed/attractor, but contract is incomplete.
- `no_certificado`: Partial evidence only.
- `pendiente`: Methodology not yet applied.

---

## 12. Hiddenness verification protocol

### Operational Definition
An attractor is hidden if its basin of attraction does not intersect any open neighborhood of any equilibrium point.

> [!WARNING]
> This is evaluated numerically under finite time, finite step $h$, and finite probing radii. It is not an absolute mathematical proof of global hiddenness.

### Official Audit Steps
1. **Calculate all equilibrium points** of the vector field.
2. **Classify stability** locally (using Matignon's criteria $|\arg(\lambda_i)| > q\pi/2$ if $q < 1.0$).
3. **Simulate trajectories** starting from small spheres around all equilibrium points.
4. **Simulate the candidate attractor** starting from the continuation seed.
5. **Compare final states** of equilibrium runs vs the candidate run.
6. **Detect contacts** between equilibrium trajectories and the candidate attractor cloud.
7. **Report contract parameters**: Probing radii, samples per radius, solver, step size $h$, time, and memory length.
8. **Classify conservatively** into the corresponding evidence label.

---

## 13. Fractional-order solvers and memory policy

Fractional derivatives are non-local operators, meaning Caputo integration requires evaluating history since $t_0$.

- **Full Memory**: Evaluates full history. Computationally expensive ($O(N^2)$ complexity).
- **Window Memory**: Truncates history to a fixed step window (e.g., $N_w = 400$). Greatly accelerates simulation, but changes the mathematical contract.
- **Continuation warning**:
  > For Caputo systems, a continuation step should carry a history window or explicit memory policy, not only the last state.
- **Solvers**:
  - `efork3`: Predictor-corrector Caputo solver.
  - `abm`: Full-history Adams-Bashforth-Moulton Caputo solver.
  - `rk4` / `heun`: Used only for integer order ($q=1.0$).

---

## 14. Figure export policy

All figures promoted to the publication tree must comply with the rules in [docs/figure_export_policy.md](docs/figure_export_policy.md).

- **Canonical exports**: Must be generated using the central `export_figure` API function and stored in `library_figures/` with a corresponding JSON manifest entry.
- **Legacy figures**: Saved in legacy subdirectories and not promoted as official evidence.
- **Direct Savefig**: The use of raw `plt.savefig()` is prohibited in workflow codes except within `plotting/export.py`.
- **Proof boundary**: A phase-space plot is a diagnostic helper; it does not constitute proof of chaos or hiddenness.

---

## 15. Troubleshooting

| Problem | Likely cause | Fix |
| :--- | :--- | :--- |
| `hidden-attractors` not found | Package not installed in editable mode | Install using `python -m pip install -e version_2`. |
| Validation command not found | Calling deprecated legacy wrappers | Use `hidden-attractors validate contract` instead. |
| Missing optional analysis modules | Optional dependencies not installed | Install the extras: `pip install -e "version_2[analysis]"`. |
| Native backend compiler failure | GCC compiler or OpenMP support missing | Set `ALLOW_NO_OPENMP=1` or rely on the Python solver fallback. |
| Fractional simulation too slow | Full-history memory cost is high | Enable finite memory window (e.g. `memory_window_steps: 400`). |
| Candidate not hidden | Trajectory intersects equilibrium basin | Classify as `self_excited_contact_detected` or `candidate_rejected`. |
| Arctan validation missing | Arctan workflow pending | Do not claim verified hiddenness; verify claims in `THESIS_CLAIMS.md`. |

---

## 16. Limitations

- **No global proof**: Numerical integration and neighborhood sweeps do not guarantee global mathematical hiddenness.
- **System boundary**: Restricted to scalar Lur'e systems; does not generaliza to arbitrary fractional-order dynamics.
- **Commensurate order only**: Commensurate fractional orders are supported; incommensurate systems are out of scope.
- **Caputo only**: Formulations assume Caputo fractional derivatives; other definitions (e.g., Riemann-Liouville) are unsupported.
- **Heuristic seeding**: Describing functions and continuation are seeds generators, not proof of existence.
- **Finite-time simulation**: Results depend on the selected step size $h$, integration horizon $t_{final}$, and memory window.

---

## 17. Citation and reproducibility

- **Citation metadata**: Consult [docs/citation.md](docs/citation.md) for citation keys and formats.
- **Release citation**: A release citation requires the official `CITATION.cff` and Zenodo DOI if available.
- **Reproducibility matrix**: Publications and reports must document the exact software environment, commit hash, and validation outcomes from [validation/freeze_audit/](validation/freeze_audit/).

### Result Submission Template
```text
Software version: 0.1.0
Git commit: [commit_hash]
Python version: [python_version]
Integrator: efork3
q: 0.9998
h: 0.001
t_final: 300.0
memory policy: finite_window (400 steps)
validation contract: configs/validation_contract.json
claim status: [reproducido/candidato/pendiente]
```

## Citation and reproducibility

Citation metadata is available in the repository root.

For archived validation records and environment information, see:

- `validation/freeze_audit/`
- `REPRODUCIBILITY.md`
- `CITATION.cff`
