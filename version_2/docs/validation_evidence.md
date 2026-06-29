# Validation Evidence

The synchronized manual metadata are defined in [docs/manual_manifest.yaml](manual_manifest.yaml); scientific claims remain governed by `THESIS_CLAIMS.md`.

For a complete user-facing description of installation, CLI usage, examples, outputs, evidence labels and limitations, see `USER_MANUAL.md`.

See `THESIS_CLAIMS.md` for the current claims classification (reproduced, validated, rejected, candidate, partial, pending).

> [!WARNING]
> **Chua Arctan Validation Status**: The c590 Caputo arctan candidate is one smooth-nonlinearity validation lane with finite radius-limited hiddenness evidence for local radii `r <= 0.3`; Wu2023 remains a separate bibliographic ADM lane.
>
> **Machado/FDF Validation Status**: The Machado/FDF system is documented as theory and a planned seed family. It is not a promoted public workflow in this release.

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

## Interpretation of radius-dependent contacts

A contact detected on a sphere of large radius around an equilibrium is not, by itself, evidence that the attractor is self-excited. The operative hiddenness test concerns sufficiently small neighborhoods of all equilibria. Large-radius spherical probes are reported as extended basin-geometry audits. Thus `local_neighborhood_contact_detected` or `self_excited_contact_detected` blocks hiddenness under the tested local contract, while `extended_radius_contact_detected` or `macro_radius_contact_detected` records basin geometry outside that local claim boundary. `hiddenness_supported_under_tested_local_neighborhoods`, `compatible_with_hiddenness_under_tested_radii`, and `candidate_rejected_under_local_contract` must be read with the stored radial contract.

All figures supporting validation evidence are promoted to the canonical `library_figures/` directory under strict reproducibility guidelines. Direct modifications are prohibited. See the [Figure Export Policy](figure_export_policy.md) for details.

### Path & Figures Portability Rules

* **Canonical Pathing**: All promoted validation evidence must use relative paths under the repository.
* **Prohibition of Local Paths**: Personal absolute paths (e.g., `/[UserDir]/`, `C:\[UserDir]\`, `[Desktop]\`) are strictly prohibited in code, tests, and promoted validation.
* **No references to local LaTeX directories**: Local report directories at the project root are ignored local drafting areas. Do not reference files inside those directories from code, tests, manifests, official reports, or validation summaries.\n
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
  "provenance": {},
  "run_metadata": {},
  "metadata_validation_errors": []
}
```

Use `hidden-attractors protocol <stage>` to write or validate an
official envelope. A `pre_continuation_periodic` verdict is diagnostic and
must remain eligible for continuation.

Each maintained run also writes `run_metadata.json`. It records the effective
`q`, `h`, `t_final`, `t_burn`, memory policy and window, integrator backend,
software provenance, Lur'e split, selected seed, parameters, continuation
path, decision tolerances, SciPy version, and random-seed policy. A strong
hiddenness promotion requires a valid metadata envelope,
full-history Caputo integration, robust reference reproduction, all
equilibrium-ball controls, zero target contacts, zero numerical failures, and
all six close/large basin slices. Missing any one condition yields
`compatible_with_hiddenness_under_tested_radii`.

## Evidence Per Stage

| Stage | Required interpretation |
| --- | --- |
| `numerical_contract` | Effective solver/backend/memory contract and integrator benchmarks. |
| `algebraic_validation` | Equilibria, Jacobians, Matignon margins, transfer function and scalar nonlinearity. |
| `seed_generation` | Seed records from implemented families; Machado/FDF remains documented as planned/theoretical unless explicitly promoted by a future release. |
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
hidden-attractors validate contract --allow-pending
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
readiness under the finite-time evidence scope; it does not automatically
promote the separate official protocol diagnostics stage. See
[F5 Dynamics Diagnostics](f5_dynamics_diagnostics.md).

## F6 And F7 Integration Layers

F6 combines F5, the Lyapunov registry, available case-specific spectra, and
optional F4 metadata into conservative per-case candidate labels:

```powershell
python .\validation\python\run_integrated_chaos_validator.py
```

F7 compares method applicability, validation state, missing results, and
diagnostic conflicts:

```powershell
python .\validation\python\run_method_comparison.py
```

These integration layers preserve the separation between full-history Caputo
QR, DK2018 block-restart, Fischer published GS, and the experimental QR lane.
They report finite-time evidence levels with method status attached. See
[F6 Integrated Chaos Validator](f6_integrated_chaos_validator.md) and
[F7 Method Comparison](f7_method_comparison.md).

## Phase F Closure Status

Phase F is frozen as a structured finite-time chaos-evidence layer:

```text
phase_F_frozen
```

Route A full-history QR is `assessed_with_documented_validation_gap`: its
internal controls and sensitivity evidence remain recorded without promotion.
Route B Fischer published GS is
`assessed_with_documented_discrepancies`: the long reproduction and bounded
sensitivity sweeps are retained with their unresolved discrepancies. Route C
closes the structured diagnostic scope. Evidence levels are numerical and tied
to the recorded solver, memory and time horizon. See
[Phase F Closure Status](phase_f_closure.md).

## Release evidence boundary

Release preparation separates four layers:

* **Promoted evidence**: lives under `validation/` and is controlled by the validation contract.
* **Promoted scientific figures**: live under `library_figures/` and must be generated through `hidden_attractors.plotting.export.export_figure`.
* **Local and exploratory outputs**: live under `outputs/`, `validation_outputs/`, `runs*/`, or `figures/` and remain outside Git.
* **Local writing drafts and templates**: remain outside the tracked software repository and do not create new scientific claims.

The canonical arctan Chua package `validation/chua_fractional_arctan/` reports the c590 route as `hiddenness_supported_under_tested_neighborhoods` for local radii `r <= 0.3`, with 8400 finite probes and zero target contacts around all equilibria. Macro radii `1.0` and `2.0` remain extended audit evidence; the claim is finite and radius-limited, not a global mathematical proof of hiddenness.
