# Poincare Diagnostics

`detect_poincare_crossings` is the public, system-independent entry point for
extracting direction-aware crossings from a sampled trajectory.

```python
from hidden_attractors import detect_poincare_crossings

result = detect_poincare_crossings(
    times,
    states,
    component=0,
    level=0.0,
    direction="positive",
)

print(result.points)
print(result.crossing_times)
```

The detector validates finite inputs, linearly interpolates each crossing, and
records the selected component, level, direction, and numerical metadata.

## Integer And Fractional Interpretation

For an integer-order ODE, an oriented section can additionally be interpreted
with the vector field, for example `x=0` with `dx/dt>0`.

For sampled Caputo trajectories, the result is a finite geometric crossing of
the stored sequence. A finite difference can describe crossing orientation,
but it is not treated as a classical instantaneous derivative or an exact
Poincare return map.

## Evidence Boundary

Crossing counts, point clouds, and return-like plots are finite geometric
characteristics. They may support trajectory comparison, but they do not
independently establish periodicity, chaos, or hiddenness.
