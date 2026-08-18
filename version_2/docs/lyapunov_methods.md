# Lyapunov Methods

The library provides two independent Lyapunov-analysis routes:

1. finite-time spectra computed from a modeled dynamical system; and
2. finite-data estimates reconstructed from a uniformly sampled scalar time
   series.

Neither route requires a hidden-attractor search.

## Equation-Based Dispatcher

`compute_lyapunov_spectrum` validates the requested order, Jacobian
requirements, memory contract, integration settings, and
reorthonormalization settings before dispatch. The returned summary includes
the method identifier, spectrum, compatibility state, warnings, and validation
metadata.

```python
import numpy as np
from hidden_attractors import compute_lyapunov_spectrum, get_system

system = get_system("chua-nonsmooth")
summary = compute_lyapunov_spectrum(
    system=system,
    x0=np.array([0.1, 0.2, 0.3]),
    q=1.0,
    h=0.01,
    t_final=100.0,
    method="integer_qr_benettin",
)

print(summary.result.exponents)
print(summary.method_info.method_id)
print(summary.compatibility_status)
```

The public registry contains four implemented methods:

| Method | Scope | Recorded validation state |
| --- | --- | --- |
| `integer_qr_benettin` | Integer ODE, `q=1` | Exact linear controls and internal cross-checks; no quantitative published-spectrum reproduction |
| `fractional_variational_abm_qr` | Commensurate Caputo, `0<q<1` | Synthetic numerical validation only |
| `fractional_cloned_dynamics_abm_gs_published` | Fractional or integer cloned dynamics | Quarantined: recorded published-benchmark discrepancy; explicit reproduction opt-in required |
| `fractional_cloned_dynamics_abm_qr` | Fractional or integer cloned dynamics | Numerical comparison only |

No current registry entry claims a complete quantitative published-benchmark
validation. All methods are callable finite-time diagnostics whose returned
warnings and status must remain attached to reported values.

## Integer QR--Benettin

The controlled `q=1` route propagates the state with the three-stage
`efork_q1_step` and propagates the variational basis with an explicit Euler
update. It applies QR reorthonormalization at the configured interval,
accumulates `log(abs(diag(R)))`, and divides by elapsed physical time.
Consequently, step-size refinement is required even though the state update
itself is higher order.

```python
from hidden_attractors import integer_system_lyapunov_exponents

result = integer_system_lyapunov_exponents(
    system,
    np.array([0.1, 0.2, 0.3]),
    h=0.01,
    t_final=100.0,
    t_burn=20.0,
)
print(result.exponents)
```

The integer method rejects fractional orders and must not be interpreted as a
Caputo-memory algorithm.

## Scalar Time-Series Route

The current 1.2.0 surface includes Lyapunov analysis for a uniformly sampled scalar
observable through `estimate_time_series_lyapunov`.

```python
from hidden_attractors import estimate_time_series_lyapunov

result = estimate_time_series_lyapunov(
    signal,
    sample_interval=0.01,
    time_unit="s",
    observable="x",
)

print(result.largest_exponent)
print(result.spectrum)
print(result.kaplan_yorke_dimension)
```

The result records:

- the Rosenstein largest-exponent estimate;
- the Eckmann reconstructed spectrum;
- the Kaplan--Yorke dimension derived from that finite spectrum;
- sampling, embedding, neighborhood, and fit parameters;
- physical inverse-time units;
- backend name and version; and
- diagnostics and warnings.

This route uses the optional `nolds` backend and does not require a right-hand
side or Jacobian. It validates finite input, sample timing, parameter
compatibility, and estimated memory use before calling the backend.

The equivalent CLI route reads a trajectory CSV:

```bash
hidden-attractors lyapunov spectrum \
  --trajectory outputs/case/trajectory.csv \
  --observable x \
  --window-length 4096 \
  --json-output outputs/case/time_series_lyapunov.json
```

## Interpretation

All returned exponents are finite-time or finite-data estimates. Convergence
depends on the method, horizon, transient handling, step size, sampling,
embedding, and numerical conditioning. A positive estimated exponent does not
by itself certify chaos, an asymptotic spectrum, or hiddenness.
No Lyapunov estimate alone can certify chaos or hiddenness.

See also [Dynamical Analysis](dynamical_analysis.md), [API Reference](api_reference.md),
and the [Public Calculation Reference Map](code_reference_map.md).
