# External Tools

The project should reuse and cite established tools instead of copying mature
algorithms into this repository. Local code focuses on adapting outputs from
the hidden-attractor workflows to those tools and documenting how they fit.

## PyDSTool

- Project: [PyDSTool](https://pydstool.github.io/PyDSTool/FrontPage.html)
- Role: dynamical-systems modeling, simulation, phase-plane analysis,
  continuation, and bifurcation analysis.
- Use here: reference or optional companion tool for continuation and branch
  tracking when the environment is compatible.

PyDSTool is broader than the local post-processing helpers. The functions in
`hidden_attractors.analysis.bifurcation` only extract maxima, minima, or samples
from already computed trajectories. They are not a continuation engine.

## pyComplexity Notebook

- Reference: [pyComplexity.ipynb](https://github.com/relopezbriega/relopezbriega.github.io/blob/master/downloads/pyComplexity.ipynb)
- Role: notebook-style exposition of complexity analysis.
- Use here: documentation and notebook style reference for presenting scalar
  complexity diagnostics clearly.

Do not copy notebook code into this repository unless the license and citation
requirements have been reviewed. The local adapter delegates scalar measures to
installable libraries such as `nolds` and `antropy`.

## nolds

- Project: [nolds](https://pypi.org/project/nolds/)
- Role: nonlinear time-series measures such as sample entropy, correlation
  dimension, Lyapunov estimates, Hurst exponent, and DFA.
- Install:

```bash
python -m pip install nolds
```

Usage:

```python
from hidden_attractors.integrations import compute_complexity_measures

metrics = compute_complexity_measures(trajectory[:, 1], backend="nolds")
```

## antropy

- Project: [antropy](https://pypi.org/project/antropy/)
- Role: entropy and fractal diagnostics such as permutation entropy, spectral
  entropy, sample entropy, Higuchi fractal dimension, and DFA.
- Install:

```bash
python -m pip install antropy
```

Usage:

```python
from hidden_attractors.integrations import compute_complexity_measures

metrics = compute_complexity_measures(trajectory[:, 1], backend="antropy")
```

## Tool Registry

The current registry is available from Python:

```python
from hidden_attractors.integrations import external_tool_report

for row in external_tool_report():
    print(row["name"], row["available"], row["recommended_use"])
```

This registry is intentionally small. Add new external tools only when they
solve a real analysis need and the repository can document how they apply to the
fractional Chua/Lur'e study systems.
