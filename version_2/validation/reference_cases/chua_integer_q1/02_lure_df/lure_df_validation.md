# Lur'e And Describing-Function Validation

The promoted seed summary records the manual Lur'e split, two positive Nyquist
frequency roots, the selected low-frequency branch, and the harmonic closure
residual. The selected seed is the input to epsilon continuation.

The MATLAB source registered in stage `08_literature_comparison` implements
the same matrix, Nyquist, amplitude, and canonical-transform checks. Its local
R2025b execution agrees with Python to `2.0e-16`, `8.0e-17`, and `1.4e-15`
absolute difference for `omega0`, `k`, and `a0`, respectively.

Guan and Xie (2025), Example 6 on PDF page 14, displays
`omega0=2.0392`, `k=0.2098`, and `a0=5.8576` for the same parameter set.
Against those rounded published values, Python differs by `0.000640%`,
`0.032104%`, and `0.024838%`, respectively.

The Python workflow now also emits `fig01c_transfer_real_imag.png`, a
two-panel version of the real and imaginary transfer-function conditions used
for the selected closure. This is the publication figure linked in the
documentation.
