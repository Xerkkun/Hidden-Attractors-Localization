# Validation Evidence

Promoted validation evidence follows the official Caputo protocol defined by
`configs/validation_contract.json` and
`configs/unified_caputo_protocol.json`. JSON stores traceability and decisions;
CSV stores numerical tables; figures support inspection; Markdown and LaTeX
state the interpretation.

## Canonical Stage Layout

```text
validation/
  00_manifest/
  01_numerical_contract/
  02_algebraic_validation/
  03_seed_generation/
  04_soft_precheck/
  05_continuation/
  06_post_continuation_filter/
  07_dynamic_reference/
  08_robustness/
  09_hiddenness_tests/
  10_diagnostics/
  final_validation_report.tex
  final_validation_report.pdf
```

The stage order is binding:

```text
numerical_contract -> algebraic_validation -> seed_generation -> soft_precheck
-> continuation -> post_continuation_filter -> dynamic_reference -> robustness
-> hiddenness_tests -> diagnostics
```

Integrator checks, including EFORK versus ABM full-history and memory
sensitivity, are evidence for `numerical_contract`. Danca-style ABM
replication is a comparison under `robustness`, not a separate hiddenness
methodology.

## Uniform Summary JSON

Every stage summary must contain:

```json
{
  "schema_version": "1.0",
  "protocol_version": "caputo_hidden_attractors_v1",
  "stage": "soft_precheck",
  "status": "passed",
  "candidate_id": "candidate_001",
  "system": "fractional_nonsmooth_chua",
  "numerical_contract": {},
  "inputs": {},
  "outputs": {},
  "metrics": {},
  "verdict": "pre_continuation_periodic",
  "files": {},
  "provenance": {}
}
```

Use `hidden-attractors-protocol <stage-command>` to write or validate an
official envelope. A `pre_continuation_periodic` verdict is diagnostic and
must remain eligible for continuation.

## Evidence Per Stage

| Stage | Required interpretation |
|---|---|
| `numerical_contract` | Effective solver/backend/memory contract and integrator benchmarks. |
| `algebraic_validation` | Equilibria, Jacobians, Matignon margins, transfer function and scalar nonlinearity. |
| `seed_generation` | Uniform seed records from the four declared families; no hiddenness claim. |
| `soft_precheck` | Admissibility and diagnostic labels only; no periodicity rejection. |
| `continuation` | `ContinuationPlan(lambda_values=...)`, intermediate states and failures. |
| `post_continuation_filter` | Target-system hard decisions, including periodicity and duplicate removal. |
| `dynamic_reference` | Trajectory geometry, spectra/recurrence, signature and optional Lyapunov estimate. |
| `robustness` | Reproduction verdicts under numerical/backend/initial-state perturbations. |
| `hiddenness_tests` | Ball samples around all equilibria plus close/large `xy`, `xz`, `yz` basin slices. |
| `diagnostics` | Complementary FFT, PSD, Lyapunov and bifurcation evidence. |

## Reference Evidence

Standalone stage trees with the former naming were removed. The directory
`validation/reference_cases/` remains only for motivated benchmarks such as
the verified integer baseline and the published EFORK manufactured-solution
tables; those records are not a competing protocol run.

## Contract Check

```bash
hidden-attractors-check-validation --allow-pending
```

`--allow-pending` permits official stages not yet generated in a current run.
Completed new runs should be checked without that option.
