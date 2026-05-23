# Dynamic Analysis

The `q=1` trajectory was regenerated after correcting the EFORK-3 third-stage
ordering. The corrected native Benettin computation records Lyapunov exponents
`(0.2082461785, 0.0137326983, -1.3669300927)`, with a positive leading
exponent.

For the primary `x(t)` component, the Nyquist/DF seed predicts
`omega0=2.039186939959001 rad/s`; the stored FFT detects
`2.303578659448180 rad/s` (`12.9655%` difference), while Welch PSD detects
`2.300971181828511 rad/s` (`12.8377%` difference). These frequency
comparisons diagnose the final nonlinear trajectory and are not a hiddenness
test. FFT and PSD peak locations remain unchanged after regeneration at the
reported resolution.
