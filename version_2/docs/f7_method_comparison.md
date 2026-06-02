# F7 Method Comparison

F7 compares Lyapunov lanes and F5 diagnostics by configured case. It records
applicability, missing spectra, validation status, consensus, discrepancies,
and methodological limitations. It does not certify chaos or hiddenness.

Run:

```powershell
python .\validation\python\run_method_comparison.py
```

Outputs live under:

```text
validation/chaos_validation/method_comparison/
```

## Lyapunov Contracts

F7 keeps these contracts separate:

| Method | Scope | Boundary |
|---|---|---|
| `integer_qr_benettin` | `q=1` only | Not applicable to fractional systems. |
| `fractional_variational_abm_qr` | Caputo `0<q<1` | Full-history QR; published validation pending. |
| `fractional_variational_dk2018_block_restart_abm_gs` | Caputo `0<q<1` | DK2018 reproduction lane; does not validate full-history QR. |
| `fractional_cloned_dynamics_abm_gs_published` | `0<q<=1` | Fischer published GS; documented discrepancies remain. |
| `fractional_cloned_dynamics_abm_qr` | `0<q<=1` | Experimental internal QR variant. |

Near-zero largest exponents use:

```text
abs(lambda_max) < 0.02
```

Published benchmark rows from other systems are not transplanted into a Chua
case. A missing case-specific spectrum is reported as unavailable.

## F5 Comparison

Boundedness is a finite-time boundedness diagnostic, not a chaos indicator.
Zero-one, PSD/FFT, and Poincare provide complementary support. For Caputo
trajectories, Poincare crossings are geometric sampled crossings, not exact
classical return maps.

The generated discrepancy report lists current conflicts and retained
limitations:

```text
validation/chaos_validation/method_comparison/method_discrepancy_report.md
```

```text
chaos_verified: false
hidden_verified: false
```
