# Numerical Contract: Integrator Validation

The Caputo fractional-order EFORK-3 numerical integration backend is validated under several complementary contracts.

## 1. Reproduction of Published Reference Errors
We reproduce the terminal errors of three-stage Caputo fractional EFORK-3 published in Ghoreishi, Ghaffari, and Saad (2023), Tables 3 and 4, for two exact analytical manufactured solution test problems:
- **Example 1** (Fractional decay with Mittag-Leffler analytical trajectory):
  - $\alpha = 0.25$, step sizes $h = 1/40, \dots, 1/640$.
  - $\alpha = 0.50$, step sizes $h = 1/40, \dots, 1/640$.
- **Example 2** (Fractional polynomial problem):
  - $\alpha = 0.25$, step sizes $h = 1/40, \dots, 1/640$.
  - $\alpha = 0.50$, step sizes $h = 1/40, \dots, 1/640$.

All 20 run combinations reproduce the exact analytical errors within a tolerance of $6\times 10^{-9}$, confirming the algebraic correctness of our core EFORK solver.

## 2. Integer-Order Limit ($q=1$)
We check the $q=1$ limit of the EFORK coefficient formulas. At $q=1$, EFORK matches the explicit three-stage Runge-Kutta stages:
$$k_1 = h f(y_n)$$
$$k_2 = h f(y_n + \frac{1}{2} k_1)$$
$$k_3 = h f(y_n + \frac{1}{2} k_1 - \frac{1}{4} k_2)$$
$$y_{n+1} = y_n + \frac{2}{3} k_1 + \frac{5}{3} k_2 - \frac{4}{3} k_3$$

Our step advances match this stage ordering exactly to floating-point precision ($< 10^{-15}$). Additionally, integration over $[0, 1.0]$ for scalar exponential decay matches the exact $e^{-1.0}$ and `scipy.integrate.solve_ivp` solutions to within $10^{-5}$ for $h=0.01$.

## 3. Comparison with Predictor-Corrector ABM
We compared EFORK-3 against a self-contained full-history Adams-Bashforth-Moulton (Diethelm ABM) solver. Under short integration windows:
- A fractional scalar decay trajectory matches ABM terminal value within $1.5\times 10^{-2}$.
- A non-smooth Chua fractional system trajectory matches ABM terminal state within $2.0\times 10^{-3}$.

This verifies that the two independent fractional integrators are consistent, and differences are well within the expected discrepancy for different numerical methods on the same fractional grid.

## 4. Finite Memory sensitivity ($L_m$)
We benchmarked the finite memory Caputo history window length $L_m$ using the native C compiled backend `FractionalChuaBackend` under `Lm = 0.10`, `0.20`, and `0.40`. All runs produced finite-valued, well-behaved bounded numerical states.
If C compiled backend was unavailable on the validation runner, the sensitivity tests were recorded as skipped without failing the contract.

## 5. Scope of this stage / Alcance de esta etapa
This stage formally validates EFORK-3 as an accurate numerical Caputo fractional integrator.
**This stage does not prove or validate:**
- The existence or chaotic nature of any attractors.
- Attractor hiddenness or localized basin decisions.
- Global parameter robustness or physical circuit implementation.

These scientific and structural properties are evaluated in later stages (`dynamic_reference`, `robustness`, `hiddenness_tests`, `diagnostics`).
