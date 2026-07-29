# Dynamical Analysis

This page documents reusable analysis functions that can be used independently
of hidden-attractor localization. A CSV may use the common convention:

```text
t,x,y,z
```

That convention matches trajectory artifacts written by the documented CLI
workflows.

## Generic Trajectory Metrics

For new code, pass time and states separately. This avoids guessing whether the
first column of a matrix is time and works for any state dimension:

```python
from hidden_attractors import compute_trajectory_metrics

metrics = compute_trajectory_metrics(
    times,
    states,              # shape: (n_samples, state_dimension)
    t_start=10.0,
    divergence_norm=120.0,
)

print(metrics["dimension"])
print(metrics["range_0"])
print(metrics["psd_entropy_component_0"])
```

Equilibria are optional. If supplied, every equilibrium must have the same
dimension as the state vectors. The result reports finite-time boundedness,
ranges, variances, a component-0 FFT summary, and final-state proximity; it does
not certify chaos or hiddenness.

## Load a Trajectory

```python
from hidden_attractors.io import load_trajectory_csv

trajectory = load_trajectory_csv("outputs/my_case/trajectories/reference_attractor.csv")
```

Headerless numeric CSV files are also accepted when they already follow the
same `t,x,y,z` order.

## Phase Space

```python
from hidden_attractors.plotting import plot_phase_space, plot_phase_projections

plot_phase_space(trajectory, "outputs/my_case/phase_space_3d.png")
plot_phase_projections(trajectory, "outputs/my_case/phase_projections.png")
```

`plot_phase_space` accepts two or three observables:

```python
plot_phase_space(trajectory, "outputs/my_case/xy.png", dims=("x", "y"))
plot_phase_space(trajectory, "outputs/my_case/xyz.png", dims=("x", "y", "z"))
```

## Time Series

```python
from hidden_attractors.plotting import plot_time_series

plot_time_series(trajectory, "outputs/my_case/time_series.png", columns=("x", "y", "z"))
```

## Bifurcation Diagrams From Existing Trajectories

The local bifurcation helper is a post-processing tool. It does not perform
continuation and does not replace specialized continuation software. It
extracts observable values from trajectories supplied by the user.

```python
from hidden_attractors.analysis import bifurcation_points_from_trajectories
from hidden_attractors.plotting import plot_bifurcation_diagram

scans = [
    (0.90, trajectory_q090),
    (0.95, trajectory_q095),
    (0.99, trajectory_q099),
]

points = bifurcation_points_from_trajectories(
    scans,
    observable="x",
    t_start=100.0,
    mode="maxima",
)

plot_bifurcation_diagram(
    points,
    "outputs/my_case/bifurcation_xmax.png",
    parameter_label="q",
    observable_label="local maxima of x",
)
```

The scan can also be a list of dictionaries:

```python
scans = [
    {"q": 0.90, "trajectory": trajectory_q090},
    {"q": 0.95, "trajectory": trajectory_q095},
]

points = bifurcation_points_from_trajectories(scans, parameter_key="q")
```

Supported observable names are `t`, `x`, `y`, and `z`; integer column indices
are also accepted.

## Lyapunov Diagnostics From a Scalar Time Series

Use the supported time-series API when only a uniformly sampled observable
is available and no right-hand side or Jacobian can be evaluated:

```python
from hidden_attractors.analysis import estimate_time_series_lyapunov

result = estimate_time_series_lyapunov(
    trajectory[:, 1],
    sample_interval=0.01,
    observable="x",
    eckmann_emb_dim=9,
    eckmann_matrix_dim=3,
    random_seed=0,
)

print(result.largest_exponent)          # Rosenstein LLE
print(result.spectrum)                  # Eckmann spectrum, descending
print(result.kaplan_yorke_dimension)
```

The sample interval is passed to both `nolds.lyap_r` and `nolds.lyap_e` as
`tau`, so exponents use inverse units of the trajectory time coordinate.
The result records the backend version, estimator parameters, captured
warnings, fit diagnostics, units, and evidence status. The polynomial fit is
the dependency-light default. When RANSAC is requested, a lock and fixed
local seed preserve reproducibility without leaking changes to NumPy's global
random state.

The corresponding CLI accepts a CSV with `t` and the selected observable:

```bash
hidden-attractors lyapunov spectrum \
  --trajectory outputs/case/trajectory.csv \
  --observable x \
  --window-length 4096 \
  --json-output outputs/case/time_series_lyapunov.json
```

Rosenstein estimates only the largest exponent. Eckmann reconstructs a
finite-dimensional spectrum from the same scalar signal; the default
`eckmann_matrix_dim=3` is appropriate only when a three-dimensional spectrum
is intended. The Kaplan--Yorke dimension is computed from that ordered
spectrum. These are finite-time, sampling- and embedding-dependent
diagnostics, not proofs of chaos, asymptotic spectra, or hiddenness.
Because the Rosenstein backend forms a quadratic pairwise-distance matrix,
the API estimates its memory requirement and rejects requests above the
configured limit instead of risking an out-of-memory failure.

## Complexity Measures Through External Libraries

Complexity metrics are delegated to optional libraries. A thin adapter passes
the same `x(t)` signal to the selected installed implementation.

```python
from hidden_attractors.integrations import compute_complexity_measures

metrics = compute_complexity_measures(
    trajectory[:, 1],
    backend="auto",
    measures=["permutation_entropy", "sample_entropy"],
    sample_rate=100.0,
)
```

`backend="auto"` resolves each requested measure through the backend that
implements it. Unknown measures and unavailable requested measures raise an
explicit error instead of returning an empty result.

When `lyapunov_rosenstein` is selected, `sample_rate` is converted to
`tau=1/sample_rate`; the returned value therefore has inverse-time units.

Install one backend when those measures are needed:

```bash
python -m pip install nolds
python -m pip install antropy
```

## Example Script

```bash
python examples/dynamical_analysis_gallery.py
```

With a trajectory CSV:

```bash
python examples/dynamical_analysis_gallery.py --trajectory-csv <trajectory_csv>
```

The script writes figures, `bifurcation_points.csv`, and `summary.json` under
`outputs/examples/dynamical_analysis_gallery/` by default.

## When To Use PyDSTool Instead

Use these functions when an existing trajectory needs plots or scalar
diagnostics. Use PyDSTool when the task requires continuation, branch tracking,
or a broader dynamical-systems modeling environment and its Python
compatibility requirements are satisfied.
