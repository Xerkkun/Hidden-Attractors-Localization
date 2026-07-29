# Optional External Tools

`hidden-attractors-fo` keeps optional integrations behind small adapters.
External packages perform their own calculations and retain their own citation
and licensing requirements.

## Registered tools

```python
from hidden_attractors.integrations import external_tool_report

for row in external_tool_report():
    print(row["name"], row["available"], row["recommended_use"])
```

The registry describes three optional tools:

| Tool | Package role | Local boundary |
| --- | --- | --- |
| PyDSTool | Modeling, continuation, and branch tracking | Listed as a companion tool; local bifurcation helpers only post-process existing trajectories. |
| `nolds` | Nonlinear scalar time-series measures | Used through the complexity adapter and the integrated scalar Lyapunov interface. |
| `antropy` | Entropy and fractal measures | Used through the complexity adapter. |

## Complexity measures

Install one or both optional backends:

```bash
python -m pip install nolds
python -m pip install antropy
```

Then request explicit measures:

```python
from hidden_attractors.integrations import compute_complexity_measures

metrics = compute_complexity_measures(
    signal,
    backend="auto",
    measures=["permutation_entropy", "sample_entropy"],
    sample_rate=100.0,
)
```

`backend="auto"` selects an installed backend that supports each requested
measure. Unknown measures, incompatible explicit backends, and unavailable
dependencies raise an error.

When `lyapunov_rosenstein` is requested, `sample_rate` is converted to
`tau=1/sample_rate`, so the returned estimate has inverse-time units.

## Integrated time-series Lyapunov interface

The supported high-level interface delegates delay reconstruction to `nolds`:

```python
from hidden_attractors import estimate_time_series_lyapunov

result = estimate_time_series_lyapunov(
    signal,
    sample_interval=0.01,
    observable="x",
)

print(result.largest_exponent)
print(result.spectrum)
print(result.kaplan_yorke_dimension)
```

The result records the Rosenstein largest-exponent estimate, an Eckmann
reconstructed spectrum, the corresponding Kaplan--Yorke dimension, estimator
parameters, units, backend version, diagnostics, and warnings.

These are finite-data estimates whose interpretation depends on sampling,
embedding, retained length, and estimator settings. They do not by themselves
certify chaos or hiddenness.
