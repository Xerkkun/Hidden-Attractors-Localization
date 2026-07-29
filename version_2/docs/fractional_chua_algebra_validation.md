# Fractional Chua Algebra Validation

This page summarizes the completed algebraic validation interface without
publishing case-specific search inputs or candidate-construction data. Exact
model parameters, tolerances, and machine-readable results are kept with the
corresponding records under `validation/`.

## Validated algebraic scope

The Wolfram validation treats the nonlinear scale under the explicit
assumption `rho > 0` and checks the following identities independently of
trajectory integration:

- reconstruction of the original vector field from its scalar Lur'e form;
- the transfer-function identity under the documented sign convention;
- equilibrium equations and numerical substitution residuals;
- analytic Jacobians and their numerical cross-tool comparisons;
- the closed-form describing function against its defining integral.

The Python comparison reads the exported Wolfram quantities and checks the
same registered model representation. These checks establish algebraic and
cross-tool consistency. They do not establish bounded dynamics, chaos,
attraction, or hiddenness.

## Transfer-function convention

The validation stores which resolvent convention is used. If one tool writes

```text
W_report(z) = r^T (z I - P)^(-1) q_v
```

and another evaluates

```text
W_code(z) = r^T (P - z I)^(-1) q_v,
```

then `W_code = -W_report`. Closure equations are compared only after this sign
normalization. The fractional frequency substitution is likewise recorded as
part of the algebraic contract.

## Describing-function boundary

For the registered arctangent nonlinearity, Wolfram verifies symbolically that
the defining harmonic integral equals the stored closed-form describing
function under `rho > 0` and positive amplitude. Numerical quadrature provides
an independent comparison at validation points stored outside this public
page.

The describing function is an initialization and consistency tool. Its
algebraic agreement is not evidence that a resulting trajectory exists, is
chaotic, or has a hidden basin.

## Evidence location

The reproducible inputs and outputs remain under `validation/wolfram/` and the
associated algebraic-validation records. Those records contain the parameter
sets, equilibrium tables, Jacobian comparisons, tolerances, and exported
summaries required to audit a particular case.

Wolfram Engine is optional for ordinary library use. Pre-generated validation
outputs allow the Python consistency checks to run without making Wolfram a
runtime dependency.
