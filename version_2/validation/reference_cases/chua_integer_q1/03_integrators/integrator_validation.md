# Corrected Integer EFORK-3 Integration

Inspection against the published three-stage EFORK formula and the supplied
`ejemplo1.py` implementation found that the prior Chua adaptation evaluated
the third stage with the two increments reversed. For `q=1`, the corrected
stage is

```text
K3 = h f(x_n + a31 K1 + a32 K2).
```

All integration-dependent integer Chua artifacts from the earlier
implementation were removed and the `balanced` run was regenerated with the
corrected Python and C backends. The corrected continuation ends at
`(1.9929740016, 1.2639718498, -2.5510633539)` for `epsilon=1`; after the
final burn-in the target seed is
`(4.0918726535, -0.0838709978, -7.5090758547)`.

The independent method validation package at
`../efork3_ghoreishi_ghaffari/` reproduces Tables 3, 4, 9, and 10 of
Ghoreishi, Ghaffari, and Saad (2023), with a maximum displayed-value
difference of `5.5489224e-9`.

The MATLAB source confirms the harmonic/canonical quantities archived in
`08_literature_comparison`; it does not export a same-grid EFORK trajectory.
No same-grid EFORK-versus-`ode45` trajectory difference is claimed here.
