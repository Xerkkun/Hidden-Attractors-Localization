# Literature And External-Tool Comparison

## Registered Sources

The supplied report `170526.pdf` contains the original theoretical and
numerical procedure for the integer-order Chua reference: Lur'e decomposition,
describing-function/Nyquist closure, harmonic seed, epsilon continuation,
Lyapunov calculation, and the 504 equilibrium-neighborhood probes. Its
algebraic and seed material remains a registered source. Its
integration-dependent numbers are superseded by the regenerated corrected
EFORK-3 run in stages `03`, `05`, and `06`.

Guan and Xie (2025) identifies Chua's circuit as a foundational
hidden-attractor example and discusses numerical continuation among
localization methods. In Example 6 on PDF page 14 it also publishes the same
parameter set I and displays `omega0=2.0392`, `k=0.2098`, `a0=5.8576`, and
the initial point `(5.8576, 0.3694, -8.3686)`. The registered
`paper_numeric_comparison.csv` compares those printed values with the
higher-precision Python results; its relative differences include the
article's four-decimal rounding.

The local MATLAB script `verifica_chua_entero.m` is copied into `sources/`
and was executed with MATLAB R2025b. Its stored output reproduces
`omega0=2.0391869399590008`, `k=0.2098673545150839`,
`a0=5.8561450862573574`, `Phi(a0)=-1.275e-16`, and a transfer error of
`4.721e-15`.
The paired table `matlab_numeric_comparison.csv` records the Python-versus-
MATLAB differences used to declare this numerical reproduction complete for
the harmonic seed and canonical-transfer checks.

The supplied Wolfram Language file `chua_entero_algebraico_sin_numericos.wl`
is also copied into `sources/` and was executed through `wolframscript`. It
prints symbolic Lur'e, transfer-function, canonical-transform, and
describing-function identities. The source explicitly contains no numerical
parameter evaluation, so it is evidence for the algebraic derivation rather
than an independent numerical reproduction of the stored trajectory results.

## Interpretation

This stage separates three claims:

1. The library artifact set records the implemented `q=1` numerical result.
2. The MATLAB run makes a numerical external recomputation auditable.
3. The Wolfram execution makes the symbolic algebra traceable without being
   misrepresented as a second numerical computation.
4. The literature review supplies a same-parameter, displayed-value comparison
   for the seed quantities, while Python and MATLAB retain higher precision.
5. The three-stage EFORK numerical method is separately reproduced against
   Ghoreishi, Ghaffari, and Saad (2023); integration-dependent Chua outputs
   were regenerated after aligning `K3 = a31*K1 + a32*K2`.
