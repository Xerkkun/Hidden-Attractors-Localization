# Quick Start

## Inspect a system

```python
from hidden_attractors import get_system

system = get_system("chua-nonsmooth")
print(system.name)
print(system.parameters)
```

The same installed registry is available from the unified CLI:

```bash
hidden-attractors inspect systems
```

## Characterize a scalar time series

Version 1.1.0 integrates Lyapunov diagnostics for uniformly sampled scalar
time series. Install the analysis extra and call:

```python
import numpy as np
from hidden_attractors.analysis import estimate_time_series_lyapunov

t = np.arange(0.0, 20.0, 0.01)
x = np.sin(t)
result = estimate_time_series_lyapunov(
    x,
    sample_interval=0.01,
    time_unit="s",
    observable="x",
)
print(result.largest_exponent)
print(result.spectrum)
print(result.kaplan_yorke_dimension)
```

This API characterizes supplied data independently of any hidden-attractor
search. Its estimates remain finite-data diagnostics and include method,
sampling, units, parameters, and backend provenance.

## Other independent diagnostics

The analysis package also exposes trajectory metrics, boundedness, spectral
summaries, Poincare sections, bifurcation helpers, and 0-1 diagnostics. These
functions accept trajectories or scalar observables directly; they do not
require a candidate-search workflow.

## Validated comprehensive example

From a source distribution or repository checkout:

```bash
python examples/chua_integer_lure_reference/run_example.py --quick
```

The example validates the registered equations, Jacobian, equilibria and
Lur'e split; recomputes the seed through the direct rational-transfer route;
then runs continuation, integration, sampled equilibrium-neighborhood
controls, Nyquist/Fourier figures and structured outputs. It contains no
frequency-scan, biased-transfer or multiparameter-search configuration. Its
finite numerical decision is not a global proof.

## Output locations

Outputs default to `./outputs`. Set `HIDDEN_ATTRACTORS_OUTPUT_DIR` and
`HIDDEN_ATTRACTORS_CACHE_DIR` to choose explicit locations.

The complete validation records are preserved in the matching repository tag
and archived DOI snapshot rather than copied into the installed package.
