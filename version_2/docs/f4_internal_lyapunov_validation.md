# F4 Internal Lyapunov Validation

F4 is an internal consistency layer for the implemented Lyapunov families. It
does not promote fractional method validation and does not certify chaos or
hiddenness.

## Reproducible Fast Assembly

```powershell
python .\validation\python\run_f4_internal_lyapunov_validation.py --all --fast --use-existing
```

The fast command runs controlled short checks and reuses the existing long
DK2018 and Fischer 2020 artifacts. Published quantitative reruns remain
explicit opt-ins:

```text
RUN_F4_LONG=1
RUN_PUBLISHED_LYAPUNOV=1
RUN_PUBLISHED_CLONED=1
```

## Stages

| Stage | Evidence | Current interpretation |
|---|---|---|
| `F4_1_integer_linear` | Exact 3D and 4D diagonal controls, q=1 cloned GS/QR checks, sensitivity to `h` and `delta` | Controlled linear checks passed |
| `F4_2_integer_chua_q1` | Promoted q=1 Chua seed, integer QR and q=1 cloned comparisons | Finite and bounded finite-time indicators; no invented published spectrum |
| `F4_3_fractional_published_dk2018` | Existing DK2018 Lorenz/RF artifact plus separate local full-history QR synthetic controls | Lorenz passes; RF `lambda_3` discrepancy remains explicit |
| `F4_4_cloned_dynamics_fischer2020` | Existing 24-row Fischer artifact and 164 bounded sensitivity runs | Controlled benchmark with documented discrepancies |

## Method Boundaries

The DK2018 block-restart ABM-GS reproduction lane and the local
fixed-lower-limit history-aware QR lane remain separate. Evidence from one
cannot validate the other.

The Fischer 2020 GS lane remains `validated=false`; the cloned-dynamics QR
variant remains internal and experimental. Its comparisons are diagnostics,
not published validation.

## Closure State

The generated summary is:

```text
validation/chaos_validation/lyapunov_methods/F4_internal_validation/
  f4_internal_validation_summary.json
```

The realistic closure state is:

```text
f4_complete_with_documented_discrepancies
```

This means that every implemented method has a controlled benchmark, a
sensitivity reference, and a bibliographic or internal reference. It does not
mean that fractional Lyapunov methods have been validated against published
benchmarks.

```text
chaos_certified_by_this_pipeline: false
hiddenness_certified_by_this_pipeline: false
fractional_lyapunov_validated_by_f4: false
caputo_lyapunov_validated_by_f4: false
```

## Phase F Closure Status

F4 evidence contributes to the Phase F diagnostic closure without promoting
fractional methods. Route A is labeled
`assessed_with_documented_validation_gap`; Route B is labeled
`assessed_with_documented_discrepancies`. This preserves rigorous controls,
published reproduction attempts, and sensitivity sweeps while keeping the
strict validation boundary explicit. See [Phase F Closure Status](phase_f_closure.md).

