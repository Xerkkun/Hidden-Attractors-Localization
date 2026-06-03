# F7 Method Discrepancy Report

## Executive Summary

F7 compares finite-time numerical evidence. It does not certify chaos or hiddenness.
The current Chua cases retain conservative comparison labels when diagnostics conflict
or case-specific Lyapunov spectra are unavailable.

## Cases

| Case | Lyapunov comparison | F5 comparison | F6 chaos level |
|---|---|---|---|
| `chua_integer_q1_reference` | `insufficient_comparable_methods` | `f5_diagnostics_conflict` | `chaos_evidence_inconclusive` |
| `danca2017_chua_fractional_saturation_q09998` | `method_validation_pending` | `f5_diagnostics_conflict` | `chaos_evidence_inconclusive` |
| `wu2023_chua_fractional_arctan_q099` | `method_validation_pending` | `f5_diagnostics_conflict` | `chaos_evidence_inconclusive` |

## Lyapunov Methods

| Method | Applicable rows | Validated | Benchmark status |
|---|---:|---|---|
| `integer_qr_benettin` | `1` | `True` | `validated_against_published_benchmarks` |
| `fractional_variational_abm_qr` | `2` | `False` | `published_benchmarks_pending` |
| `fractional_variational_dk2018_block_restart_abm_gs` | `2` | `False` | `published_benchmarks_pending_reproduced_discrepancy` |
| `fractional_cloned_dynamics_abm_gs_published` | `3` | `False` | `published_benchmarks_pending_discrepancy` |
| `fractional_cloned_dynamics_abm_qr` | `3` | `False` | `internal_variant_pending` |

## F5 Diagnostics

| Case | Boundedness | Zero-one | PSD/FFT | Poincare | Comparison |
|---|---|---|---|---|---|
| `chua_integer_q1_reference` | `bounded_candidate` | `zero_one_regular_candidate` | `spectral_inconclusive` | `cloud_like` | `f5_diagnostics_conflict` |
| `danca2017_chua_fractional_saturation_q09998` | `bounded_candidate` | `zero_one_regular_candidate` | `spectral_inconclusive` | `cloud_like` | `f5_diagnostics_conflict` |
| `wu2023_chua_fractional_arctan_q099` | `bounded_candidate` | `zero_one_regular_candidate` | `spectral_inconclusive` | `cloud_like` | `f5_diagnostics_conflict` |

## Conflicts Detected

- `chua_integer_q1_reference`: one or more available spectra belong to methods with validation pending; zero-one is regular-like while Poincare geometry is cloud-like
- `danca2017_chua_fractional_saturation_q09998`: one or more available spectra belong to methods with validation pending; zero-one is regular-like while Poincare geometry is cloud-like
- `wu2023_chua_fractional_arctan_q099`: one or more available spectra belong to methods with validation pending; zero-one is regular-like while Poincare geometry is cloud-like

## Limitations

- F2 full-history QR is not validated against published benchmarks.
- F3 published GS retains documented Fischer 2020 discrepancies.
- F3 QR is an experimental internal variant.
- Zero-one and PSD/FFT are diagnostics, not proofs.
- Poincare sections for Caputo trajectories are geometric crossings, not exact return maps.
- DK2018 block-restart evidence does not validate full-history Caputo QR.

## Conservative Conclusion

F7 reports support, disagreement, and missing evidence. It certifies neither chaos nor hiddenness.
