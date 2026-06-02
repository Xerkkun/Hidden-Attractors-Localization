# Poincare Diagnostics

## Scope

Phase F5.4 implements Poincare sections as standardized geometric diagnostics.
The detector validates sampled crossing detection, linear interpolation,
direction filtering, integer RHS direction checks, and Caputo geometric
metadata. Poincare alone does not certify chaos or hiddenness.

F5.4 is also consumed by the shared
[F5 Dynamics Diagnostics](f5_dynamics_diagnostics.md) runner together with
boundedness, zero-one, and FFT/PSD outputs.

## Integer ODE section

For an integer ODE, the natural Chua section is:

```text
x = 0, xdot > 0
```

The detector linearly interpolates the crossing and evaluates the ordinary
RHS at that point. Positive crossings require a positive RHS component.

## Caputo section

For a Caputo trajectory, F5.4 records a numerical geometric crossing:

```text
x[n] < 0 <= x[n+1] and x[n+1] - x[n] > 0
```

This is not an exact classical Poincare map. A finite difference may describe
the crossing orientation but is not treated as a classical instantaneous
derivative. Danca 2021 notes that autonomous Caputo fractional systems do not
have exact non-constant periodic solutions. Accordingly, F5.4 does not claim
exact periodic orbits in Caputo systems.

## Published Cases

| Case | Contract | Initial condition |
|---|---|---|
| Chua integer reference | `q=1`, `x=0, xdot>0` | `[5.856145086257356, 0.369331578246782, -8.366536168331880]` |
| Danca 2017 saturation | `q=0.9998`, `h=0.01`, `t_final=500`, geometric crossing | Reproducible diagnostic seed transferred from the integer case because article coordinates are not reported |
| Wu 2023 arctan | `q=0.99`, `h=0.01`, `t_final=100`, finite memory `40.0`, geometric crossing | `[13.8, 0.7093, -19.8768]` and its reported symmetric counterpart |

The Danca saturation parameters are `alpha=8.4562`, `beta=12.0732`,
`gamma=0.0052`, `m0=-0.1768`, and `m1=-1.1468`. The Wu arctan parameters are
`alpha=8.4562`, `beta=12.0732`, `gamma=0.0052`, `a1=0.4`, `a2=-1.5585`, and
`rho=1.0`.

## Outputs

Each case writes:

```text
poincare_points.csv
poincare_section.csv
poincare_summary.json
poincare_metadata.json
README.md
```

The summary reports geometric labels such as `point_like_or_fixed_return`,
`curve_like`, and `cloud_like`. These labels are finite-time numerical
descriptions, not isolated proofs of chaos.
