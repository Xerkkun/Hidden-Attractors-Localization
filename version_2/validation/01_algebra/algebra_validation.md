# Algebra Validation

The non-smooth fractional Chua model at `q=0.9998` reproduces Danca's parameter set and the MATLAB validation values. Python returns the same three equilibria, zero vector-field residuals to floating-point precision, central-difference agreement with the analytic regional Jacobians, and the same inner/outer spectra exported by MATLAB and Wolfram. Matignon classification is stable at `E0` and unstable at `E+` and `E-`.

The supplied Wolfram source verifies the symbolic identities after renaming its protected local symbol `Tr` to `Treal`.
