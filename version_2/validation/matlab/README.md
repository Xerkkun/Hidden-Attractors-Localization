# MATLAB FDE12 oracle

`run_dk2018_fde12_oracle.m` is an independent MATLAB comparison runner for
the DK2018 block-restart ABM-GS contract. It mirrors the numerical loop in
the supplied `FO_Lyapunov.m`, omits plotting, and writes convergence CSV data
only to a caller-selected runtime path.

The Garrappa `fde12.m` solver is an external dependency and is not vendored in
this repository. The official 2012 and 2025 revisions declare the function as
lowercase `fde12`. The supplied `FO_Lyapunov.m` invokes uppercase `FDE12`,
which MATLAB R2025b does not execute on this Windows installation. The oracle
runner calls the official lowercase name directly without modifying the
external solver.

For the RF case from Danca and Kuznetsov (2018), both Garrappa revisions
tested on 2026-05-30 produced:

```text
[0.0611165016657, 0.00389813962371, -1.83246468204]
```

The native C lane produced:

```text
[0.06111650301597507, 0.0038981398659820223, -1.8324646836344436]
```

The C result therefore matches the independent MATLAB oracle. Against the
published RF vector `[0.0749, 0.0018, -2.0850]`, `lambda_1` and `lambda_2`
are within absolute tolerance `0.05`; only `lambda_3` fails, with absolute
difference `0.2525353`. This is evidence for a reproduced discrepancy, not
grounds to promote the published benchmark or relax its tolerance.

This runner does not certify chaos or hiddenness.
