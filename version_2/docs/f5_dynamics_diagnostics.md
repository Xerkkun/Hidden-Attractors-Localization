# F5 Dynamics Diagnostics

F5 assembles four complementary finite-time numerical diagnostics:
boundedness, the 0-1 test, FFT/PSD, and Poincare sections. The combined output
supports candidate characterization under the recorded finite-time contract.

Run the shared-cache pipeline with:

```powershell
python .\validation\python\run_f5_dynamics_diagnostics.py --all --use-existing-poincare --fast
```

Trajectories are integrated once with the Poincare numerical contracts and
stored as compressed post-transient caches under
`validation/chaos_validation/dynamics_diagnostics/trajectories/`.

## F5.1 Boundedness

The principal finite-time metric is:

```text
R_observed = sup_{t in [T_b,T]} ||X(t)||
```

The output also records coordinate spans, final norm, finite fraction, and a
late-versus-early norm growth ratio. `bounded_candidate` means that the sampled
post-transient trajectory remained numerically bounded.

## F5.2 Zero-One Test

For each coordinate and reproducible values of `c`, F5 computes:

```text
p_c(n) = sum_j phi_j cos(j c)
q_c(n) = sum_j phi_j sin(j c)
K = median(K_c)
```

The interpretation thresholds are:

| Statistic | Label |
|---|---|
| `K > 0.8` | `zero_one_chaotic_candidate` |
| `K < 0.2` | `zero_one_regular_candidate` |
| otherwise | `zero_one_inconclusive` |

Fischer et al. use the 0-1 statistic together with Lyapunov evidence. F5 keeps
that boundary: a high 0-1 statistic is interpreted with complementary
diagnostics. White noise is included as an explicit limitation because it may
also produce a high statistic.

## F5.3 FFT And PSD

For each post-transient coordinate, F5 computes a one-sided normalized power
spectrum and:

```text
peak_dominance = P_max / sum_k P_k
```

The conservative labels are:

```text
broadband_spectrum
dominant_periodic_peak
quasiperiodic_candidate
spectral_inconclusive
```

FFT/PSD describes spectral geometry and contributes complementary evidence.

## F5.4 Poincare

Poincare sections were implemented before F5.1-F5.3 and remain unchanged in
scope. Integer Chua uses:

```text
x = 0, xdot > 0
```

Caputo cases use geometric sampled crossings rather than exact classical
return maps.
See [Poincare Diagnostics](poincare_diagnostics.md).

## Published Cases

The four diagnostics use the same configured cases:

```text
chua_integer_q1_reference
danca2017_chua_fractional_saturation_q09998
wu2023_chua_fractional_arctan_q099
```

The Danca 2017 initial condition is marked
`diagnostic_only_not_reported_by_article`; it is not an exact article
reproduction claim.

## Closure

When all four diagnostics have standardized outputs, the F5 summary may report:

```text
f5_diagnostics_structured_outputs_ready
```

This means that the complementary output bundle is ready for inspection.

F6 consumes this structured bundle. Conflicting F5 indicators remain
inconclusive. See
[F6 Integrated Chaos Validator](f6_integrated_chaos_validator.md).

## Phase F Closure Status

F5 supplies standardized diagnostics to the frozen Phase F evidence layer.
Current cases remain `chaos_evidence_inconclusive` where diagnostics conflict.
See [Phase F Closure Status](phase_f_closure.md).

