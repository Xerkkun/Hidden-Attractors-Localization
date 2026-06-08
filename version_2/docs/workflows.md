# Official Workflow

All maintained Caputo hidden-attractor work follows one sequence:

```text
numerical_contract
-> algebraic_validation
-> seed_generation
-> soft_precheck
-> continuation
-> post_continuation_filter
-> dynamic_reference
-> robustness
-> hiddenness_tests
-> diagnostics
```

The public contracts are in `hidden_attractors.workflows.protocol`. Machine
summaries use the same envelope at every stage:

```text
schema_version, protocol_version, stage, status, candidate_id, system,
numerical_contract, inputs, outputs, metrics, verdict, files, provenance
```

## CLI

### Simple-level CLI Presets

The high-level unified CLI `hidden-attractors` allows executing preset workflows directly:
* `hidden-attractors init -e <name>` or `hidden-attractors init --example <name>`: Copies configuration templates into the working directory.
* `hidden-attractors inspect-config -p <name>` or `hidden-attractors inspect-config --preset <name>`: Previews the fully resolved configuration mapping.
* `hidden-attractors run -p <name>` or `hidden-attractors run --preset <name>`: Executes the workflow preset (e.g., `chua_integer`, `chua_fractional`, `chua_arctan`, `chua_bifurcation`, `chua_basin`).
* `hidden-attractors run -c path/to/config.yaml` or `hidden-attractors run --config path/to/config.yaml`: Executes a workflow from a YAML file.

You can override configuration parameters directly on the CLI command line using nested dot notation overrides (e.g., `--final_simulation.t_final 50.0 --h 0.01`).

The primary stable user-facing CLI is `hidden-attractors`. Specialized
workflows are reproducible analysis interfaces, but may change while the
project remains in alpha. Auxiliary or internal commands are documented for
traceability, not as stable public interfaces.

| Command | Group | Usage | Documentary status |
|---|---|---|---|
| `hidden-attractors` | Main user command | `run -c/--config`, `run -p/--preset`, `init -e/--example`, `inspect-config -c/--config`, `inspect-config -p/--preset`, `validate-bibliography -m/--manifest --strict --json -o/--markdown-output` | Primary stable user-facing CLI; Python module is internal |
| `hidden-attractors-protocol` | Specialized workflow | Protocol stage summaries and validation envelopes | Reproducible protocol interface; alpha |
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

### Low-level Programmatic Usage

For advanced custom workflows, the library exposes clean API entry points:
* `hidden_attractors.workflows.config_loader.load_config(path)`: Loads, normalizes, and validates hierarchical configuration dictionaries.
* `hidden_attractors.systems.get_system(name)`: Looks up a registered dynamical system definition (such as `chua-arctan` or `chua-nonsmooth`).
* `hidden_attractors.integrations.selector.integrate(rhs, x0, q, h, t_final, ...)`: Resolves, checks compatibility, and triggers the appropriate integer or fractional solver.

---

The official promotion interface is:

```bash
hidden-attractors-protocol generate-seeds --contract configs/unified_caputo_protocol.json --output outputs/run/03_seed_generation/summary.json
hidden-attractors-protocol soft-precheck --contract configs/unified_caputo_protocol.json --payload precheck.json --output outputs/run/04_soft_precheck/summary.json
hidden-attractors-protocol continue --contract configs/unified_caputo_protocol.json --candidate-id candidate_001 --lambda-values 0,0.25,0.5,0.75,1 --output outputs/run/05_continuation/summary.json
hidden-attractors-protocol filter-survivors --contract configs/unified_caputo_protocol.json --payload decisions.json --output outputs/run/06_post_continuation_filter/summary.json
hidden-attractors-protocol build-reference --contract configs/unified_caputo_protocol.json --payload reference.json --output outputs/run/07_dynamic_reference/summary.json
hidden-attractors-protocol robustness --contract configs/unified_caputo_protocol.json --payload robustness.json --output outputs/run/08_robustness/summary.json
hidden-attractors-protocol hiddenness --contract configs/unified_caputo_protocol.json --payload hiddenness.json --output outputs/run/09_hiddenness_tests/summary.json
hidden-attractors-protocol diagnostics --contract configs/unified_caputo_protocol.json --payload diagnostics.json --output outputs/run/10_diagnostics/summary.json
```

Native numerical adapters may produce the payloads, but only official
envelopes are eligible for promoted validation evidence.

## Stage Rules

`numerical_contract` fixes `q`, time discretization, transient duration,
backend, memory policy, numerical thresholds, similarity thresholds,
hiddenness radii and reproducible random sampling. EFORK/C is preferred for
validated sweeps. ABM full-history remains the benchmark for final candidates
and Danca-style replication. Finite-memory runs measure scalability and
robustness; they are not the sole truth source.

`algebraic_validation` records the Caputo Lur'e representation, equilibria,
piecewise Jacobians when needed, Matignon classification, fractional transfer
function and describing-function families.

`seed_generation` has one schema and four families:

| Family | Definition |
|---|---|
| `lure_classical_centered` | `sigma0=0`, `mu=1` |
| `lure_classical_biased` | free `sigma0`, `mu=1` |
| `machado_centered` | `sigma0=0`, variable `mu`, variable `theta` |
| `machado_biased` | free `sigma0`, variable `mu`, variable `theta` |

Classical centered describing function and centered Lur'e are one family.
Machado/FDF enlarges the seed search; it does not establish hiddenness.

`soft_precheck` rejects only invalid configuration, impossible
amplitude/frequency, NaN/Inf or catastrophic numerical failure, and exact
duplicates. A periodic-looking direct seed receives
`pre_continuation_periodic` and remains admissible for continuation.

`continuation` uses `ContinuationPlan(lambda_values=...)`: `lambda=0` is the
auxiliary start system and `lambda=1` is the target system. Historical
implementations may internally map `lambda` to epsilon, eta or smoothing, but
official output never exposes those as competing protocols.

`post_continuation_filter` is where hard periodicity and long target-system
dynamics are evaluated. Survivors advance as `continuation_survivor`.

`dynamic_reference` constructs the target object for comparison: trajectory,
geometry, equilibrium distances, spectral/recurrence information, optional
Lyapunov estimate and similarity signature.

`robustness` requires controlled changes in step size, horizon, memory policy,
backend when feasible, initial point and reconstruction phase. Its allowed
verdicts are `robust_target_hit`, `weak_target_hit`, `not_reproduced` and
`numerical_failure`.

`hiddenness_tests` runs only on robust survivors. It samples within balls
around every equilibrium, with declared radii and increasing sample counts,
and generates close and large `xy`, `xz` and `yz` basin slices. A strong final
label is invalid unless every declared test passes.

`diagnostics` includes FFT, PSD, Lyapunov and bifurcation views. These
quantities complement, but never replace, neighborhood and basin tests.

## Compatibility

`tools/legacy/` remains temporarily because maintained wrappers still depend
on the native fractional Chua and Danca adapters there. It is not an
alternative methodology. See [Migration To The Unified Methodology](migration_unified_methodology.md).

---

## Biased Chua Fractional Hidden Attractor Workflow (BDF)

The BDF workflow represents the unified pipeline for discovering and verifying hidden attractors in the non-smooth fractional Chua system with biased describing functions ($c \neq 0$).

The pipeline is implemented programmatically under:
- [biased_chua.py](file:///c:/Users/moren/Desktop/Codes/Hidden%20Attractors%20Fractional%20Order/version_2/hidden_attractors/workflows/biased_chua.py)

The official executable entry point is:
- [run_example.py](file:///c:/Users/moren/Desktop/Codes/Hidden%20Attractors%20Fractional%20Order/version_2/examples/chua_nonsmooth_biased_hidden_attractor/run_example.py)

### Pipeline Steps:
1. **Paso 1: Búsqueda Centrada (c=0)** - Sirve como línea base de referencia (DF centrada).
2. **Paso 2: Búsqueda de raíces BDF (c≠0)**, continuación homotópica afín y simulación larga con clasificación de periodicidad.
3. **Paso 3: Verificación de ocultedad estándar** mediante barrido de esferas y validación de contrato.
4. **Paso 4: Verificación de ocultedad extendida** mediante muestreo volumétrico (ball sampling) en paralelo hasta radio $r=2.0$.
5. **Paso 5: Resumen y figuras** con exportación a la galería centralizada en `version_2/library_figures/` mediante la API de ploteo centralizada.

