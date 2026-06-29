# Official Examples and Workflow Index

This page classifies the maintained execution paths for the release. Examples
are not scientific claims by themselves; claim status is governed by
`THESIS_CLAIMS.md` and promoted validation artifacts.

## A. Official report examples

| Example | Path | Command from `version_2` | Methodological role | Evidence status |
| --- | --- | --- | --- | --- |
| Example 0: integer Chua Lur'e reference | `examples/chua_integer_lure_reference/` | `python examples/chua_integer_lure_reference/run_example.py --quick` | Reproduces the integer `q=1` route: Lur'e seed, continuation, final trajectory, neighborhood controls, figures, and Lyapunov diagnostic | Reproduced integer reference; not fractional validation |
| Example 1: non-smooth fractional Chua BDF | `examples/chua_nonsmooth_biased_hidden_attractor/` | `python examples/chua_nonsmooth_biased_hidden_attractor/run_example.py --quick` | Proposed biased describing-function route for non-smooth Caputo Chua | Candidate/compatible under tested local radii; separate official nearby candidate classified under its recorded local contract |
| Example 2: arctan Chua Wu2023/c590 | `examples/chua_arctan_wu2023/` | `python examples/chua_arctan_wu2023/run_example.py --quick` | Separates Wu2023 bibliographic reproduction from a smooth-nonlinearity Caputo full-history c590 lane | c590 reported for local radii `r <= 0.3`; macro-radius contacts documented as extended audit |

### Full example commands

```bash
cd version_2
python examples/chua_integer_lure_reference/run_example.py --quick
python examples/chua_nonsmooth_biased_hidden_attractor/run_example.py --quick
python examples/chua_arctan_wu2023/run_example.py --quick
```

Longer runs remove `--quick`; some lanes also support `--all` or `--steps`.
Check each example README before launching long hiddenness sampling.

## B. Article reproduction status

| Article/source | Current library coverage | Not reproduced or not promoted because |
| --- | --- | --- |
| Kuznetsov-style integer Chua reference | Executable integer reference case | N/A for the maintained integer route; it is a software reference, not a fractional claim |
| Danca 2017 non-smooth fractional Chua | Partial reference implementation and proposed BDF methodology lane | Published data omit exact seed/IC, DF frequency/gain/amplitude, and continuation details; official nearby candidate is classified under the recorded local-neighborhood tests |
| Wu2023 arctan Chua | Algebra, equilibria, Lur'e split, reported initial conditions, ADM local reproduction, and separate c590 Caputo lane | The published ADM/initial-condition lane is not full-memory Caputo validation; c590 is one local-radius smooth-nonlinearity lane for `r <= 0.3` with macro-radius contacts retained as extended audit |
| DK2018 Lyapunov benchmarks | Opt-in diagnostic comparison lane | RF `lambda_3` discrepancy remains recorded; the lane does not certify chaos/hiddenness |
| Fischer 2020 cloned dynamics | Diagnostic comparison lane with documented discrepancies | Quantitative/sign-pattern discrepancies remain; not a full validation of the local QR method |

## C. CLI presets

CLI presets are configuration profiles for `hidden-attractors`; they are not
standalone examples.

| Preset | Command | Status | Recommended first use |
| --- | --- | --- | --- |
| `chua_integer` | `hidden-attractors run -p chua_integer` | Stable user route | Yes |
| `chua_fractional` | `hidden-attractors run -p chua_fractional` | Stable user route with fractional evidence boundary | Yes |
| `chua_arctan` | `hidden-attractors run -p chua_arctan` | Radius-limited smooth arctan c590 lane plus bibliographic Wu2023 lane | Inspect with validation boundary |
| `chua_bifurcation` | `hidden-attractors bifurcation run -p chua_bifurcation` | Advanced diagnostic | No |
| `chua_basin` | `hidden-attractors run -p chua_basin` | Heavy basin workflow | No |

## D. Specialized subcommands

These commands are maintained analysis interfaces used by the official pipeline, validation tests, or advanced audits:

- `hidden-attractors protocol <substage>` (e.g. `generate-seeds`, `soft-precheck`, `continue`, `filter-survivors`, `build-reference`, `robustness`, `hiddenness`, `diagnostics`)
- `hidden-attractors seed lure-centered`
- `hidden-attractors seed lure-biased`
- `hidden-attractors continuation run`
- `hidden-attractors continuation multiparameter`
- `hidden-attractors hiddenness sphere-controls`
- `hidden-attractors hiddenness strict-target-refinement`
- `hidden-attractors basin refined`
- `hidden-attractors basin strict-target-refinement`
- `hidden-attractors robustness overlay`
- `hidden-attractors bifurcation run`
- `hidden-attractors bifurcation plot`
- `hidden-attractors bifurcation inspect`
- `hidden-attractors lyapunov compute`
- `hidden-attractors lyapunov spectrum`
- `hidden-attractors lyapunov validate`
- `hidden-attractors chaos-test zero-one`
- `hidden-attractors chaos-test inspect`
- `hidden-attractors published danca-abm-sphere-controls`
- `hidden-attractors validate contract`
- `hidden-attractors validate bibliography`
- `hidden-attractors validate release-readiness`
- `hidden-attractors report fractional-run`

They are not alternative methodologies. They implement stages of the same contracted workflow.

## E. Small API examples

| Script | Purpose |
| --- | --- |
| `examples/quickstart_equilibria.py` | Check Chua equilibria against the vector field |
| `examples/list_final_candidates.py` | Load candidate records through the public candidate API |
| `examples/custom_system_definition.py` | Register a new `ChaoticSystem` |
| `examples/new_system_workflow_spec.py` | Write a `WorkflowInputSpec` for auditable reusable workflows |
| `examples/integer_lure_chua_protocol.py` | Small order-one Lur'e protocol demo |
| `examples/dynamical_analysis_gallery.py` | Generate public plotting/diagnostic outputs from trajectory data |

## F. Legacy boundary

`tools/legacy/` preserves historical scripts for traceability. It is not a
release API and must not be used as a competing solver path for new promoted
results.
