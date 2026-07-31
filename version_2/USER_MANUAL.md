# User Manual

This manual covers the installed `hidden-attractors-fo` 1.1.0 library, its
validated example, runtime paths, and numerical evidence boundaries. It does
not extend beyond the current validation-backed public capabilities.

## 1. Purpose and scientific scope

The library implements reusable numerical components for integer-order and
commensurate Caputo fractional-order systems with a scalar Lur'e form

```text
^C D_t^q X = P X + b psi(r^T X),  0 < q <= 1.
```

The fractional transfer convention is

```text
W_q(s) = r^T (s^q I - P)^(-1) b
lambda = (j omega)^q
```

Seed construction, continuation, finite-time diagnostics, and sampled
neighborhood tests are separate stages. No single stage is presented as a
global mathematical proof.

## 2. Installation

```bash
python -m pip install hidden-attractors-fo
```

```python
import hidden_attractors
```

Verify the command-line entry point:

```bash
hidden-attractors --help
hidden-attractors inspect systems
```

For a repository checkout:

```bash
python -m pip install -e ".[dev,analysis,docs]"
```

## 3. Public CLI

The installed console command is `hidden-attractors`. The principal command
groups expose system inspection, configuration execution, seed construction,
continuation, diagnostics, and validation helpers:

```bash
hidden-attractors inspect systems
hidden-attractors run --help
hidden-attractors seed --help
hidden-attractors continuation --help
hidden-attractors lyapunov --help
hidden-attractors validate --help
hidden-attractors protocol --help
```

Repository-only release checks may require the tagged validation tree and are
not expected to run from a wheel alone.

From that matching tagged checkout, validate its recorded contract with:

```bash
hidden-attractors validate contract
```

## 4. Validated end-to-end example

The distributed comprehensive example is the integer-order Chua Lur'e
reference/control:

```bash
python examples/chua_integer_lure_reference/run_example.py --quick
```

It executes:

1. Lur'e describing-function seed construction.
2. Deterministic continuation to the target system.
3. Integer-order integration.
4. Sampled controls around every equilibrium in the declared contract.
5. JSON, CSV, and figure generation.

The example is reproducible software validation. Its sampled finite-time
decision does not establish global hiddenness.

## 5. Python API

Minimal system inspection:

```python
from hidden_attractors import get_system

system = get_system("chua-nonsmooth")
print(system.name)
```

Configuration loading and integration:

```python
from hidden_attractors.integrations.selector import integrate
import numpy as np

system = get_system("chua-nonsmooth")
times, states, status = integrate(
    lambda _t, state: system.evaluate(state),
    np.array([0.1, 0.0, 0.0]),
    q=1.0,
    h=0.01,
    t_final=1.0,
    integrator="rk4",
)
```

Stable and experimental exports are distinguished by the API-tier metadata.
Experimental diagnostics return structured provenance and must retain their
documented evidence boundary.

## 6. Independent dynamical characterization

Analysis functions can be used directly on a supplied system, trajectory, or
uniformly sampled scalar signal. They do not require a localization workflow.
The public surface includes:

- generic trajectory and boundedness metrics;
- FFT/PSD, Poincare, 0-1, and bifurcation post-processing;
- equation-based integer and fractional Lyapunov estimators; and
- scalar-time-series Lyapunov reconstruction.

Version 1.1.0 fully integrates the scalar-series route through
`estimate_time_series_lyapunov`. Install the optional backend and call:

```bash
python -m pip install "hidden-attractors-fo[analysis]"
```

```python
from hidden_attractors import estimate_time_series_lyapunov

result = estimate_time_series_lyapunov(
    signal,
    sample_interval=0.01,
    time_unit="s",
    observable="x",
)

print(result.largest_exponent)       # Rosenstein
print(result.spectrum)               # Eckmann, descending
print(result.kaplan_yorke_dimension)
```

The result records sampling units, embedding and neighborhood parameters,
backend version, fitting diagnostics, a pairwise-memory estimate, evidence
status, and warnings. It is a finite-data reconstruction, not an asymptotic
proof or a hiddenness classifier.

The equations, exact discretization boundaries, validation states, and
ready-to-use calls for trajectory metrics, boundedness, FFT, the library's
Welch scaling, Poincare sections, the 0--1 test, bifurcation
post-processing, equation-based and scalar-series Lyapunov methods,
complexity adapters, RK4, Heun, ABM, EFORK-3, and ADM are documented in:

- [`docs/mathematical_diagnostics.md`](docs/mathematical_diagnostics.md)

In particular, the integer QR--Benettin route advances the state with
`efork_q1_step` but advances its tangent basis with explicit Euler. The
fractional history-aware QR transformation is a project-specific extension.
The one-sided `psd_welch` density scaling is tested against SciPy under the
same explicit window, overlap, and detrending contract.

## 7. YAML configuration

A runnable configuration identifies the system, numerical order, integrator,
step size, time interval, initial condition, and output directory. Fractional
configurations also declare their memory policy.

```yaml
system_id: chua
q: 1.0
integrator:
  method: rk4
  h: 0.01
  t_final: 10.0
initial_state: [0.1, 0.0, 0.0]
output_dir: outputs/example
```

Use the loader rather than reading YAML directly so aliases, defaults, and
validation errors are handled consistently.

## 8. Runtime outputs and caches

Generated files are written under `./outputs` by default. Set an explicit
location when required:

```powershell
$env:HIDDEN_ATTRACTORS_OUTPUT_DIR = "C:\tmp\hidden-attractors-output"
$env:HIDDEN_ATTRACTORS_CACHE_DIR = "C:\tmp\hidden-attractors-cache"
```

On POSIX shells:

```bash
export HIDDEN_ATTRACTORS_OUTPUT_DIR=/tmp/hidden-attractors-output
export HIDDEN_ATTRACTORS_CACHE_DIR=/tmp/hidden-attractors-cache
```

The library never uses its installed `site-packages` directory for outputs,
native compilation products, Matplotlib state, or runtime caches.

## 9. Evidence states and hiddenness verification

The public contract distinguishes:

- a generated seed;
- a continued trajectory;
- a bounded finite-time trajectory;
- diagnostic evidence;
- a sampled-neighborhood decision.

Describing-function/Nyquist output is a seed construction. FFT/PSD, Poincare,
0-1 statistics, and Lyapunov estimates are diagnostics. A sampled-neighborhood
decision additionally records every equilibrium, radii, sampling rule, solver,
memory policy, time horizon, target-matching rule, tolerances, and numerical
failures.

The complete promoted records are stored under `validation/` in the tagged
repository and archived snapshot. They are not duplicated inside the wheel or
sdist.

## 10. Fractional solvers and memory policy

Caputo integration depends on trajectory history. A reproducible fractional
run therefore records the method, step size, order `q`, complete or truncated
memory policy, memory length when applicable, and initialization history.

Results from distinct memory policies are not treated as interchangeable.
Integer-order `q = 1` controls use the declared integer solver contract.

## 11. Figure export policy

Figures generated by library workflows use the canonical plotting/export
helpers. Outputs are written beneath the selected runtime output directory and
carry enough metadata to identify the producing command and numerical input.

The public plotting surface contains 33 `plot_*` and `render_*` functions.
Each has a direct call, minimum-input description, and a reproducible numerical
example produced by the library in:

- [`docs/plot_function_catalog.md`](docs/plot_function_catalog.md)

The examples are not hand-built pedagogical curves. They are calculated from
the registered `chua-nonsmooth` system with its declared parameters and
numerical contracts. The shared provenance includes an `efork_q1` trajectory,
integer continuation in `lambda`, an RK4 sweep of `beta`, QR--Benettin
finite-time Lyapunov exponents, integrated equilibrium-neighborhood controls,
and a finite-time basin grid.

Regenerate the complete real-system numerical catalog from a source checkout
with:

```bash
python figure_scripts/generate_plot_catalog_examples.py
```

Regenerate one function with:

```bash
python figure_scripts/generate_plot_catalog_examples.py --only plot_phase_space
```

The generator verifies its inventory against
`hidden_attractors.plotting.__all__`. The resulting
`docs/assets/generated_plot_catalog/catalog_results.json` records the exact
parameters, condition, step sizes, time horizons, thresholds, input bundle,
output hashes, and all 62 PNG files: 33 representative images plus 29
additional outputs from multi-output functions. The catalog page displays and
links every one of those outputs.

These are genuine numerical outputs of a named mathematical system, not
experimental measurements or new validation evidence. A catalog plot alone
does not establish chaos, hiddenness, asymptotic stability, integrator
convergence, or scientific performance. The public name
`plot_continuation_eta` and `_eta` filenames remain for compatibility, while
their axes use the actual continuation variable `lambda`.

## 12. Troubleshooting

If an optional diagnostic is unavailable, install the analysis extra:

```bash
python -m pip install "hidden-attractors-fo[analysis]"
```

If native compilation is unavailable, select a Python backend. If a
repository-only validation command reports missing manifests, run it from the
matching release tag rather than from an installed wheel. If a configuration
is rejected, inspect the normalized input and the reported missing contract
fields before executing a long run.

## 13. Limitations

- Numerical trajectories and classifications are finite-time results.
- Sampled neighborhoods do not cover a continuous basin globally.
- Floating-point, compiler, and platform differences can affect tolerances.
- Commensurate Caputo and integer-order contracts are supported; other
  derivative definitions require a separate implemented contract.
- The PyPI artifacts contain the installed package, public examples, and
  concise user documentation. The complete validation data tree is maintained
  in the source repository.

## 14. Citation and reproducibility

Citation metadata is supplied in `CITATION.cff`, `.zenodo.json`, and
`codemeta.json` at the repository root. The archived DOI is
`10.17605/OSF.IO/ZGK74`.

For an auditable run, record the package version, release tag, Python version,
platform, dependencies, configuration file, random seed, and output hashes.
The repository freeze record is `validation/freeze_audit/`.

Public documentation included with the source distribution:

- [`docs/quick_start.md`](docs/quick_start.md)
- [`docs/scientific_scope.md`](docs/scientific_scope.md)
- [`docs/api_stability.md`](docs/api_stability.md)
- [`docs/mathematical_diagnostics.md`](docs/mathematical_diagnostics.md)
- [`docs/plot_function_catalog.md`](docs/plot_function_catalog.md)
- [`docs/citation.md`](docs/citation.md)
