# Kalman--Fitts integer non-Chua Lur'e reference

This example tests a fourth-order system that is not a Chua model:

\[
\dot x=Ax+b\tanh(c^Tx/\varepsilon),
\quad b=(0,0,0,1)^T,\quad c=(0,0,-1,0)^T,
\]

with \(m_1=0.9\), \(m_2=1.1\), \(\beta=0.03\), and
\(\varepsilon=0.01\).  The matrix and parameterization are those of
Kuznetsov et al., *Coexistence of hidden attractors and multistability in
counterexamples to the Kalman conjecture*, IFAC-PapersOnLine 52(16), 7--12
(2019), DOI [10.1016/j.ifacol.2019.11.747](https://doi.org/10.1016/j.ifacol.2019.11.747).

The example keeps two routes separate:

1. The direct integer transfer equation is solved algebraically from the
   declared \((A,b,c)\).  It returns a negative harmonic-linearization gain,
   while the `tanh` describing function is strictly positive.  Therefore the
   direct route stops with an explicit incompatibility; no frequency grid is
   used and no fallback is hidden inside it.
2. The published alternative is then executed.  A stable cycle of the
   `sign(c^T x)` precursor is recomputed from a generic switching-section
   point by an Andronov point map.  That generated state is continued through
   \((1-\lambda)\operatorname{sign}(\sigma)+
   \lambda\tanh(\sigma/\varepsilon)\), \(0\leq\lambda\leq1\).

The point from Table 2 is never used to start the calculation.  It is compared
only after the final target trajectory has been obtained.  Hiddenness controls
sample balls around the unique stable equilibrium and report a finite tested
statement, not a global proof.

Run everything from `version_2`:

```powershell
& '..\.venv\Scripts\python.exe' examples\kalman_fitts_integer_lure_reference\run_example.py
```

Short smoke run:

```powershell
& '..\.venv\Scripts\python.exe' examples\kalman_fitts_integer_lure_reference\run_example.py --quick
```

The maintained candidate catalog is
[`docs/integer_lure_test_catalog.md`](../../docs/integer_lure_test_catalog.md).
