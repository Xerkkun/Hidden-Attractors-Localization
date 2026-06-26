# Quick Start Guide / Guia rapida

This is the shortest supported path for installing, checking, and running
`hidden-attractors-fo`. The scientific claim source remains
`THESIS_CLAIMS.md`; this guide only explains how to run the software.

## 1. PyPI installation

For normal use:

```bash
python -m pip install hidden-attractors-fo
```

The import name is:

```python
import hidden_attractors
```

## 2. Development installation

From a repository checkout:

```bash
python -m pip install -e "version_2[dev,analysis,docs,legacy]"
```

From inside `version_2`:

```bash
python -m pip install -e ".[dev,analysis,docs,legacy]"
```

On this Windows workspace, prefer the repository virtual environment when it is
available:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\version_2[dev,analysis,docs,legacy]"
```

## 3. Check the CLI

The package exposes one public console command:

```bash
hidden-attractors --help
hidden-attractors inspect systems
hidden-attractors inspect candidates
hidden-attractors seed --help
hidden-attractors validate contract --allow-pending
```

All maintained workflows are subcommands of `hidden-attractors`. Historical
standalone commands are legacy/deprecated and are not the public release API.
Machado/FDF seed routes are documented as theory/internal planned support only;
they are not exposed as public release CLI commands.

## 4. Run a preset

```bash
cd version_2
hidden-attractors init -e chua_fractional
hidden-attractors inspect-config -p chua_fractional
hidden-attractors run -p chua_fractional
```

To use your own configuration:

```bash
hidden-attractors run -c path/to/config.yaml
```

## 5. Official examples

These are the release examples that explain the methodology. Use `--quick` for
smoke checks; full runs may take minutes or hours depending on hiddenness
sampling and fractional memory settings.

| Example | Command from `version_2` | Role | Current evidence status |
| --- | --- | --- | --- |
| Integer Chua Lur'e reference | `python examples/chua_integer_lure_reference/run_example.py --quick` | Control case for seed, continuation, trajectory, and neighborhood checks at `q=1` | Reproduced integer reference; does not validate fractional hiddenness |
| Non-smooth fractional Chua BDF | `python examples/chua_nonsmooth_biased_hidden_attractor/run_example.py --quick` | Proposed biased-DF route for the non-smooth Caputo Chua case | Candidate/compatible only under declared local radii; a separate official candidate was rejected as self-excited |
| Arctan fractional Chua Wu2023/c590 | `python examples/chua_arctan_wu2023/run_example.py --quick` | Separates Wu2023 bibliographic reproduction from the promoted c590 Caputo full-history lane | c590 is finite-time local/radius-limited evidence with macro radii retained as extended audit |

## 6. Reproduced vs non-reproduced article cases

| Source case | Library status | Reason |
| --- | --- | --- |
| Kuznetsov-style integer Chua reference | Reproduced as the integer `q=1` software reference | Published/reference data are sufficient for the maintained Lur'e seed and trajectory route |
| Danca 2017 non-smooth fractional Chua | Partial reference implementation, not full trajectory reproduction | The article does not report the exact DF frequency/gain/amplitude, seed coordinates, hidden-attractor initial condition, or complete continuation settings |
| Official nearby non-smooth fractional candidate | Rejected under current contract | Neighborhood tests recorded target contacts from `E+` and `E-`, so the candidate is self-excited under the recorded protocol |
| Wu2023 arctan fractional Chua and c590 | Wu2023 is partial algebraic/ADM reproduction; c590 is a separate local-radius lane | The ADM local recurrence is not full-memory Caputo ABM/EFORK; c590 is local/radius-limited and not a global proof |
| DK2018 and Fischer 2020 Lyapunov lanes | Diagnostic comparison with documented discrepancies | These validate or audit method behavior; they do not certify chaos or hiddenness |

## 7. API map

The full release API inventory is [API Reference](api_reference.md). It is
generated from `version_2/hidden_attractors` and documents every defined
function, class, and method, including internal helpers. New users usually need
only these entry points:

```python
from hidden_attractors import get_system, register_system
from hidden_attractors.systems import ChaoticSystem
from hidden_attractors.workflows.specs import WorkflowInputSpec
from hidden_attractors.workflows.config_loader import load_config
from hidden_attractors.integrations.selector import integrate
```

For a new Lur'e-compatible system, register a `ChaoticSystem`, provide
equilibria/Jacobian, add an explicit Lur'e split `(P, b, r, psi)`, then record a
`WorkflowInputSpec` before running seed, continuation, robustness, and
hiddenness workflows.

## 8. Scientific boundary

DF/Nyquist, continuation, FFT/PSD, 0-1 tests, Lyapunov estimates, Poincare
sections, and phase portraits are diagnostics or seed-generation tools. A
hiddenness label requires neighborhood or basin checks around all equilibria
under the recorded numerical contract: `q`, `h`, integrator, memory policy,
time horizon, burn-in, radii, samples, and classifier thresholds.

These checks produce finite-time evidence under a contract, not a global
mathematical proof.
