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

## Fractional Lyapunov Evidence

Phase-F Lyapunov evidence lives under `validation/chaos_validation/`. The
local `fractional_variational_abm_qr` fixed-lower-limit full-history contract
and the DK2018 block-restart ABM-GS reproduction contract are promoted
independently. Neither contract alone certifies chaos or hiddenness.

The fast suite runs native smoke checks only. DK2018 quantitative reproduction
is an explicit opt-in run with `RUN_PUBLISHED_LYAPUNOV=1`. The long native run
recorded on 2026-05-31 passed Lorenz and left RF pending because only
`lambda_3` exceeds absolute tolerance `0.05`. This does not promote the local
full-history QR method.

## F4 Internal Lyapunov Validation

F4 assembles controlled method-level checks under
`validation/chaos_validation/lyapunov_methods/F4_internal_validation/`.
The fast runner uses short internal controls and reuses existing published
artifacts:

```powershell
python .\validation\python\run_f4_internal_lyapunov_validation.py --all --fast --use-existing
```

The closure state `f4_complete_with_documented_discrepancies` records that
every implemented method has a control, sensitivity reference, and
bibliographic or internal reference. It does not promote fractional method
validation. DK2018 RF `lambda_3` and Fischer 2020 discrepancies remain
explicit. See [F4 Internal Lyapunov Validation](f4_internal_lyapunov_validation.md).

## F5.4 Poincare Diagnostics

F5.4 records standardized Poincare crossing outputs under
`validation/chaos_validation/dynamics_diagnostics/poincare/`. Integer ODE
cases may use `x=0, xdot>0`; Caputo cases use geometric sampled crossings with
`exact_poincare_map=false`. These outputs do not certify chaos, hiddenness, or
exact periodic orbits in Caputo systems. See
[Poincare Diagnostics](poincare_diagnostics.md).

## F5 Complementary Dynamics Diagnostics

F5.1 boundedness, F5.2 zero-one, F5.3 FFT/PSD, and F5.4 Poincare write
standardized outputs under
`validation/chaos_validation/dynamics_diagnostics/`. They reuse compressed
post-transient trajectory caches and can be regenerated with:

```powershell
python .\validation\python\run_f5_dynamics_diagnostics.py --all --use-existing-poincare --fast
```

The summary state `f5_diagnostics_structured_outputs_ready` records output
readiness only. It does not certify chaos, hiddenness, or exact periodic
orbits in Caputo systems, and it does not automatically promote the separate
official protocol diagnostics stage. See
[F5 Dynamics Diagnostics](f5_dynamics_diagnostics.md).
