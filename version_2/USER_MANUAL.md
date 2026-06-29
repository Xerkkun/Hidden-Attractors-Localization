# User Manual

This is the primary user manual for the `hidden-attractors-fo` research
library. It covers installation, the unified CLI, reproducible examples,
article-reproduction status, evidence labels, API inventory, and the boundary
between numerical diagnostics and hiddenness claims.

Canonical references:

- **Official Claims Matrix**: [THESIS_CLAIMS.md](THESIS_CLAIMS.md)
- **Quick Start Guide**: [docs/quick_start.md](docs/quick_start.md)
- **API Reference**: [docs/api_reference.md](docs/api_reference.md)
- **Figure Export Policy**: [docs/figure_export_policy.md](docs/figure_export_policy.md)
- **Dependency Policy**: [docs/dependency_policy.md](docs/dependency_policy.md)
- **Official Test Freeze Audit**: [validation/freeze_audit/](validation/freeze_audit/)

## 1. Purpose and scientific scope

`hidden-attractors-fo` supports reproducible numerical workflows for
integer-order and commensurate Caputo fractional-order systems that admit a
scalar Lur'e representation

```text
^C D_t^q X = P X + b psi(r^T X),   0 < q <= 1.
```

For frequency-domain seed generation, the fractional transfer convention is

```text
W_q(s) = r^T (s^q I - P)^(-1) b
lambda = (j omega)^q
W_hat_q(lambda) = r^T (lambda I - P)^(-1) b
```

DF/Nyquist, continuation, FFT/PSD, Poincare sections, 0-1 tests, Lyapunov
estimates, and phase portraits are diagnostics or seed-generation tools. They
do not establish hiddenness by themselves. Hiddenness labels require sampled
local-neighborhood or basin evidence around all equilibria under a recorded
numerical contract. Large-radius spherical probes are extended basin-geometry
audits, not automatic evidence of a self-excited attractor.

The current release scope is scalar Lur'e-compatible Chua-type systems and
extension templates for new Lur'e systems. Arbitrary nonlinear systems,
incommensurate fractional orders, and non-Caputo derivatives require separate
contracts before they can enter the official workflow.

## 2. Installation

For normal use from PyPI:

```bash
python -m pip install hidden-attractors-fo
```

The PyPI project name is `hidden-attractors-fo`; the Python import name is:

```python
import hidden_attractors
```

For development from a repository checkout:

```bash
python -m pip install -e "version_2[dev,analysis,docs,legacy]"
```

From inside `version_2`:

```bash
python -m pip install -e ".[dev,analysis,docs,legacy]"
```

On this Windows workspace, the safer local interpreter is usually:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\version_2[dev,analysis,docs,legacy]"
```

Verify the install:

```bash
hidden-attractors --help
hidden-attractors inspect systems
hidden-attractors seed --help
cd version_2
python -m compileall hidden_attractors examples tests tools/cli
python -m pytest -q
```

For PyPI packaging readiness:

```bash
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
python tools/release/validate_wheel_install.py
```

The official release/freeze counts are stored under
[validation/freeze_audit/](validation/freeze_audit/). Local counts can differ
when optional tools or local test selections change.

## 3. Repository structure

| Path | Purpose |
| :--- | :--- |
| `hidden_attractors/` | Importable package: models, systems, solvers, analysis, verification, plotting, workflows, CLI dispatch. |
| `hidden_attractors/systems/` | `ChaoticSystem`, `LureSystem`, registry, workflow-capability requirements. |
| `hidden_attractors/seed_generation/` and `hidden_attractors/lure/` | DF/Nyquist/Lur'e seed helpers; seed output is heuristic. |
| `hidden_attractors/integrations/`, `solvers/`, `native/` | Integer and Caputo integration contracts, Python solvers, C backends. |
| `hidden_attractors/workflows/` | Protocol, continuation, robustness, hiddenness, basins, report, and reusable specs. |
| `hidden_attractors/verification/` | Equilibria, stability, neighborhood probes, candidate gates, status labels. |
| `configs/examples/` | YAML templates and presets. |
| `examples/` | Runnable examples and official report cases. |
| `validation/` | Promoted evidence and manifests. |
| `library_figures/` | Promoted figures and figure manifests. |
| `docs/` | User docs, API reference, report source, validation notes. |
| `tools/legacy/` | Historical compatibility material, not public release API. |

## 4. Public CLI

The primary public console command is `hidden-attractors`. Below is the complete table of public command groups and subcommands:

| Command Group | Subcommand | Description | Command Example |
| :--- | :--- | :--- | :--- |
| **run** | (direct) | Run an experiment configuration (preset or YAML) | `hidden-attractors run -p chua_integer`, `hidden-attractors run -p chua_fractional`, `hidden-attractors run -p chua_arctan`, `hidden-attractors run -c path/to/config.yaml` |
| **init** | (direct) | Copy template configs to the current directory | `hidden-attractors init -e chua_fractional` |
| **inspect-config** | (direct) | Preview the normalized configuration | `hidden-attractors inspect-config -p chua_fractional` |
| **inspect** | `systems` | Inspect registered chaotic systems | `hidden-attractors inspect systems` |
| | `candidates` | List final candidate records | `hidden-attractors inspect candidates` |
| | `workflow-requirements` | Inspect reusable workflow requirements | `hidden-attractors inspect workflow-requirements` |
| **validate** | `contract` | Validate numerical validation evidence contract | `hidden-attractors validate contract --allow-pending` |
| | `bibliography` | Validate claims bibliography manifest | `hidden-attractors validate bibliography` |
| | `release-readiness` | Validate release packaging/readiness metadata | `hidden-attractors validate release-readiness --submission-strict` |
| **protocol** | `generate-seeds` | Run seed generation protocol stage | `hidden-attractors protocol generate-seeds` |
| | `soft-precheck` | Run soft precheck protocol stage | `hidden-attractors protocol soft-precheck` |
| | `continue` | Run continuation protocol stage | `hidden-attractors protocol continue` |
| | `filter-survivors` | Run survivor filtering protocol stage | `hidden-attractors protocol filter-survivors` |
| | `build-reference` | Run reference building protocol stage | `hidden-attractors protocol build-reference` |
| | `robustness` | Run robustness protocol stage | `hidden-attractors protocol robustness` |
| | `hiddenness` | Run hiddenness protocol stage | `hidden-attractors protocol hiddenness` |
| | `diagnostics` | Run diagnostics protocol stage | `hidden-attractors protocol diagnostics` |
| **seed** | `lure-centered` | Centered Lur'e seed generation | `hidden-attractors seed lure-centered --help` |
| | `lure-biased` | Biased Lur'e seed generation | `hidden-attractors seed lure-biased --help` |
| **continuation** | `run` | Run scalar continuation | `hidden-attractors continuation run --help` |
| | `multiparameter` | Run multiparameter continuation | `hidden-attractors continuation multiparameter --help` |
| **hiddenness** | `sphere-controls` | Run sphere controls validation workflow | `hidden-attractors hiddenness sphere-controls --help` |
| | `strict-target-refinement` | Run strict target refinement workflow | `hidden-attractors hiddenness strict-target-refinement --help` |
| **basin** | `refined` | Run refined basin workflow | `hidden-attractors basin refined --help` |
| | `strict-target-refinement` | Run strict target refinement workflow for basins | `hidden-attractors basin strict-target-refinement --help` |
| **robustness** | `overlay` | Run robustness overlay workflow | `hidden-attractors robustness overlay --help` |
| **bifurcation** | `run` | Run parameter sweep bifurcation workflow | `hidden-attractors bifurcation run --help` |
| | `plot` | Plot bifurcation diagram from CSV data | `hidden-attractors bifurcation plot --help` |
| | `inspect` | Inspect bifurcation summary JSON | `hidden-attractors bifurcation inspect --help` |
| **lyapunov** | `compute` | Compute Lyapunov exponents workflow | `hidden-attractors lyapunov compute --help` |
| | `spectrum` | Estimate trajectory-based Lyapunov exponent | `hidden-attractors lyapunov spectrum --help` |
| | `validate` | Validate Lyapunov summary JSON | `hidden-attractors lyapunov validate --help` |
| **chaos-test** | `zero-one` | Run 0-1 chaos-test diagnostic | `hidden-attractors chaos-test zero-one --help` |
| | `inspect` | Inspect 0-1 chaos-test summary JSON | `hidden-attractors chaos-test inspect --help` |
| **published** | `danca-abm-sphere-controls` | Run published Danca ABM sphere controls | `hidden-attractors published danca-abm-sphere-controls --help` |
| **report** | `fractional-run` | Run fractional report run workflow | `hidden-attractors report fractional-run --help` |

Legacy standalone commands were removed from the public installed surface. They may appear in migration documentation only as legacy/deprecated names.

## 5. Minimal examples

Quick inspection:

```bash
hidden-attractors inspect systems
hidden-attractors inspect candidates
hidden-attractors validate contract --allow-pending
```

Run presets:

```bash
hidden-attractors run -p chua_integer
hidden-attractors run -p chua_fractional
```

Run the three official example folders from `version_2`:

```bash
python examples/chua_integer_lure_reference/run_example.py --quick
python examples/chua_nonsmooth_biased_hidden_attractor/run_example.py --quick
python examples/chua_arctan_wu2023/run_example.py --quick
```

Python API starting points:

```python
from hidden_attractors import get_system, register_system
from hidden_attractors.systems import ChaoticSystem
from hidden_attractors.workflows.specs import WorkflowInputSpec
from hidden_attractors.workflows.config_loader import load_config
from hidden_attractors.integrations.selector import integrate
```

Every defined function, class, and method in the active package is documented in
[docs/api_reference.md](docs/api_reference.md). That inventory includes private
helpers for auditability; private names are not stable API unless explicitly
exported or documented as public workflow inputs.

## 6. Reproducible example 1: Chua integer

The integer-order Chua case is the release reference for the Lur'e software
route. It exercises seed generation, continuation, final trajectory, sampled
neighborhood controls, figures, and a finite-time Lyapunov diagnostic without
Caputo memory.

- **System**: non-smooth Chua saturation.
- **Order**: `q = 1.0`.
- **Status**: reproduced integer reference/control.
- **Command**:

```bash
cd version_2
python examples/chua_integer_lure_reference/run_example.py --quick
```

Promoted validation artifacts live under
`validation/reference_cases/chua_integer_q1/`. The hiddenness summary reports no
target-basin hits from sampled equilibrium neighborhoods in the corrected
integer rerun. This validates the integer route only; it does not validate any
fractional hidden-attractor claim.

## 7. Reproducible example 2: Chua nonsmooth BDF

The non-smooth fractional Chua example demonstrates the proposed biased
describing-function route for a Caputo candidate.

- **System**: non-smooth Chua saturation.
- **Order**: typically `q = 0.9998`.
- **Command**:

```bash
cd version_2
python examples/chua_nonsmooth_biased_hidden_attractor/run_example.py --quick
```

This is not a full reproduction of the Danca 2017 hidden-attractor trajectory.
The paper does not report enough numerical data for an independent full
trajectory reproduction: exact DF frequency/gain/amplitude, seed coordinates,
exact hidden-attractor initial condition, and complete continuation settings are
missing.

The library therefore separates two claims:

| Lane | Status | Reason |
| --- | --- | --- |
| Example 1 biased-DF methodology | Candidate/compatible under declared local radii | It proposes a reproducible seed-continuation-verification workflow, but the local tests are not a global proof. |
| Official nearby fractional candidate `danca2017_nearby_saturation_candidate_q09998` | Classified under the recorded local contract | `validation/09_hiddenness_tests/hiddenness_tests_validation_summary.json` records 1305 target contacts from local neighborhoods of `E+` and `E-`. |

## 8. Radius-limited promoted example: Chua arctan c590

The arctan Chua example is a smooth-nonlinearity validation example that
separates the Wu 2023 bibliographic lane from the Caputo full-history c590
validation lane.

- **System**: Chua with smooth arctan nonlinearity.
- **Reported order**: `q = 0.99` for the Wu 2023 lane.
- **Command**:

```bash
cd version_2
python examples/chua_arctan_wu2023/run_example.py --quick
```

The Wu2023 bibliographic ADM path remains separate: it uses a local recurrence
and does not accumulate full Caputo history. Under that local ADM contract, the
reported initial conditions classify as periodic/nonchaotic after transient
filtering.

The c590 lane uses a Caputo full-history dynamic search and neighborhood
sampling. It is reported as `hiddenness_supported_under_tested_local_radii` for
local radii `r <= 0.3` with 8400 finite probes and zero contacts around all
equilibria. Macro-radius contacts at `r=1.0` and `r=2.0` are retained as
extended audit evidence. This is a finite local-radius hiddenness claim, not a
global basin proof.

## 9. YAML configuration format

Workflows use hierarchical YAML. A minimal structure is:

```yaml
experiment:
  name: "demo"
  output_dir: "outputs/demo"

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

For new Lur'e systems, registration is not enough. The workflow must also
record equilibria, Jacobian, Lur'e split `(P, b, r, psi)`, describing-function
branch convention, solver/memory policy, target reference, classifier thresholds,
radii, sample counts, and random seed policy.

## 10. Output files and expected results

| Output | Typical location | Meaning |
| :--- | :--- | :--- |
| JSON summaries | `outputs/.../*.json` or `validation/.../*.json` | Machine-readable parameters, statuses, verdicts, and provenance. |
| CSV trajectories/tables | `outputs/.../*.csv` or `validation/.../*.csv` | Numerical trajectory rows, decisions, basin labels, or diagnostics. |
| Figures | `library_figures/` for promoted figures; `outputs/` for ordinary runs | PNG/PDF visual diagnostics generated through the plotting/export policy. |
| Manifests | `validation/00_manifest/`, `library_figures/manifests/`, release metadata | Traceability from results to files, parameters, and software provenance. |

Ordinary exploratory outputs stay under `outputs/` or other ignored local output
folders. Promoted validation evidence belongs under `validation/` and must use
relative paths.

## 11. Evidence states and hiddenness labels

Canonical attractor statuses:

| Status | Meaning |
| :--- | :--- |
| `candidate` | A localized branch or seed under investigation. |
| `hidden_under_tested_neighborhoods` | No equilibrium-neighborhood target contact under the completed recorded contract. |
| `compatible_with_hiddenness` | No contact detected under limited or incomplete tested radii/metadata. |
| `self_excited` | Contact detected from at least one equilibrium local neighborhood under the recorded contract. |
| `nonchaotic` | Periodic, quasiperiodic, equilibrium-like, or otherwise not chaotic under diagnostics. |
| `diverged` | Numerical divergence/unbounded behavior. |
| `inconclusive` | Conflicting diagnostics or numerical failure. |
| `rejected` | Invalid, collapsed, self-excited under the local contract, or otherwise not promotable. |
| `not_tested` | Hiddenness checks have not been run. |

The claims matrix uses broader evidence statuses such as `validated`,
`reproduced`, `rejected`, `candidate`, `partial`, and `pending`.

## 12. Hiddenness verification protocol

Operational definition: an attractor is hidden if its basin of attraction does
not intersect any open neighborhood of any equilibrium point.

The finite numerical protocol approximates this by:

1. computing all equilibria;
2. classifying local stability, with Matignon's criterion for `0 < q < 1`;
3. integrating sampled points in neighborhoods or basin slices around every equilibrium;
4. integrating the candidate reference trajectory;
5. comparing endpoint/trajectory clouds with target criteria;
6. reporting contacts, divergences, equilibrium collapses, and unknown outcomes;
7. preserving the complete numerical contract.

A finite sample is evidence under the declared contract, not a global
mathematical proof.

### Hiddenness contract by radial scale

A contact detected on a sphere of large radius around an equilibrium is not, by itself, evidence that the attractor is self-excited. The operative hiddenness test concerns sufficiently small neighborhoods of all equilibria. Large-radius spherical probes are reported as extended basin-geometry audits.

Use the following interpretation uniformly for integer Chua, non-smooth fractional Chua, smooth arctan fractional Chua, and future Lur'e systems:

| Label family | Interpretation |
| :--- | :--- |
| `local_neighborhood_contact_detected` / `self_excited_contact_detected` | Evidence against hiddenness under the tested local contract. |
| `extended_radius_contact_detected` / `macro_radius_contact_detected` | Extended basin-geometry audit contact; not an automatic rejection of local hiddenness. |
| `hiddenness_supported_under_tested_local_neighborhoods` | No local-neighborhood contact was detected under the tested radii and sampling contract. |
| `compatible_with_hiddenness_under_tested_radii` | Compatible with hiddenness under limited or incomplete tested radii. |
| `candidate_rejected_under_local_contract` | Candidate rejected because the local contract, not merely an extended-radius audit, recorded disqualifying contact or another blocking condition. |

## 13. Fractional-order solvers and memory policy

Caputo dynamics depend on history. The memory policy is part of the numerical
model, not an implementation detail.

| Policy | Meaning |
| :--- | :--- |
| `full` / `full_caputo` | Full accumulated Caputo history. Expensive but closest to the declared fractional IVP. |
| `window` / `finite_window` | Finite memory window for scalability or robustness comparisons. Changes the contract. |
| `none_local_adm` | Local recurrence used for the Wu2023 ADM reproduction lane; not a full-history Caputo ABM/EFORK route. |

Use `rk4`, `heun`, or `efork_q1` only for integer order. Use `abm` or `efork3`
for fractional Caputo workflows when their contract applies.

## 14. Figure export policy

Promoted figures must follow [docs/figure_export_policy.md](docs/figure_export_policy.md):

- export through the central plotting/export API;
- store PNG/PDF plus metadata;
- update the figure manifest;
- avoid local absolute paths;
- avoid treating visual plots as hiddenness proof.

Ordinary example figures can remain under `outputs/` until promoted.

## 15. Troubleshooting

| Problem | Likely cause | Fix |
| :--- | :--- | :--- |
| `hidden-attractors` not found | Editable install missing or wrong environment | Run `python -m pip install -e version_2` in the intended environment. |
| Slow fractional run | Full history or large sampling plan | Use `--quick` for smoke checks, or reduce horizons/samples for exploratory runs. |
| Native backend compile failure | Missing C compiler/OpenMP | Install a compiler or set documented fallback/OpenMP options where supported. |
| Candidate becomes self-excited | Local-neighborhood trajectory reached target under the recorded contract | Keep `self_excited`/`rejected`; do not promote hiddenness. |
| Arctan result looks periodic | The local ADM lane produced regular post-transient behavior | Treat as nonchaotic/non-promoted under that contract. |
| Docs mention a missing function | API reference is stale | Regenerate/update [docs/api_reference.md](docs/api_reference.md) and link affected guides. |

## 16. Limitations

- Numerical evidence is finite-time and contract-dependent.
- Hiddenness is not inferred from a single trajectory, plot, DF seed, or Lyapunov estimate.
- The maintained full methodology is restricted to scalar Lur'e-compatible systems.
- Fractional workflows assume commensurate Caputo order unless a separate contract is added.
- Current native C backends are Chua-oriented; new systems may need new backends or adapters for heavy workflows.
- Machado/FDF is documented as a theory/planned seed family, not a promoted public workflow in this release.

## 17. Citation and reproducibility

Citation metadata is stored at the repository root in `CITATION.cff`,
`.zenodo.json`, and `codemeta.json`. The archived DOI currently recorded in the
repository metadata is:

```text
10.17605/OSF.IO/ZGK74
```

Reproducible reports should state:

```text
Software version: 1.0.0
Git commit: <commit>
Python version: <version>
System and parameters: <system_id, q, parameters>
Integrator and memory policy: <name, h, memory>
Time contract: <t_final, t_burn>
Validation contract: configs/validation_contract.json
Claim status: <reproduced/validated/candidate/rejected/partial/pending>
```

For release packaging, also consult `REPRODUCIBILITY.md`, `RELEASE_NOTES.md`,
`CHANGELOG.md`, `version_2/MANIFEST.md`, and `version_2/release_package/`.
