# Two-phase lead-lag PLL integer Lur'e reference

This example reproduces the two-dimensional phase-space model of Bianchi et
al., *Limitations of PLL simulation: hidden oscillations in MatLab and SPICE*
(2015), DOI [10.1109/ICUMT.2015.7382409](https://doi.org/10.1109/ICUMT.2015.7382409).
It is a non-Chua integer-order system on the cylinder
\(\mathbb R\times\mathbb S^1\).

The registered coordinates are

\[
u=x-x_e,\qquad v=\theta_\Delta-\theta_s,
\]

and give the exact T1 scalar Lur'e form

\[
\dot z=A_0z+b\left[\sin(\theta_s+c^Tz)-\sin\theta_s\right].
\]

The reproducible route keeps the two localization decisions separate:

1. The grid-free integer transfer equation is executed first.  Its imaginary
   part has a strict sign for every positive frequency, so the standard real
   describing function cannot close.  No frequency scan is invoked.
2. The A1 route starts from the exact running solution at zero loop gain and
   continues fixed points of the Andronov one-turn map to \(L=500\).  The same
   map independently finds the unstable running cycle that separates the
   running and locked basins.

The three initial conditions printed in the paper are excluded from seed
construction and continuation.  They are evaluated only after both return-map
cycles have been generated.

Hiddenness controls use the explicit cylinder distance

\[
d^2=\left(\frac{\Delta u}{\tau_1/2}\right)^2+
\left(\frac{\operatorname{wrap}(\Delta v)}{\pi}\right)^2.
\]

The maintained contract tests four scaled radii and 12 samples per radius
around both principal equilibrium classes, for 96 trajectories.  A zero-hit
result supports hiddenness only under those tested neighborhoods; it is not a
global basin proof.

Run from `version_2`:

```powershell
& '..\.venv\Scripts\python.exe' examples\pll_lead_lag_integer_lure_reference\run_example.py
```

The default command is the `full` profile and uses 101 continuation values.
To preserve the promoted quick bundle while generating an isolated full
record, use:

```powershell
& '..\.venv\Scripts\python.exe' examples\pll_lead_lag_integer_lure_reference\run_example.py `
  --output-dir validation\reference_cases\pll_lead_lag_integer_q1_full
```

The stored full run completed the five phases, all 96 probes and the three
post-localization regressions.  Its post-hoc Python--Julia full comparison is
`output/comparison/pll_lead_lag_integer_comparison_full.json` in the sibling
`julia-dynamical-systems-demo` project.

Quick mode preserves all 96 cylinder probes but shortens the continuation
schedule:

```powershell
& '..\.venv\Scripts\python.exe' examples\pll_lead_lag_integer_lure_reference\run_example.py --quick
```
