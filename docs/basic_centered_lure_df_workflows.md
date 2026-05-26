# Centered Lur'e Describing Function Workflows

This document outlines the design, mathematical foundations, and execution of the three default, homogeneous, and label-differentiable centered Lur'e workflows added to the library:

1. `chua_integer_centered_lure_df` (System ID: `chua_integer_saturation`)
2. `chua_fractional_centered_lure_df` (System ID: `chua_fractional_saturation`)
3. `chua_arctan_fractional_centered_lure_df` (System ID: `chua_fractional_arctan`)

---

## 1. System Equations and Common Lur'e Form

All three systems share the common Lur'e framework:

$$\begin{aligned}
D_t^q X &= P X + b \, \psi(r^T X), \\
\sigma &= r^T X,
\end{aligned}$$

where the state vector is $X = [x, y, z]^T$, and the standard feedback directions are:

$$r = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}, \quad b = \begin{pmatrix} -\alpha \\ 0 \\ 0 \end{pmatrix}.$$

### A. Saturated Chua System (Integer & Fractional)
The system equations are:

$$\begin{aligned}
D_t^q x &= \alpha(y - x(m_1 + 1)) - \alpha \, \psi(x) \\
D_t^q y &= x - y + z \\
D_t^q z &= -(\beta y + \gamma z)
\end{aligned}$$

where the nonlinearity is:

$$\psi(\sigma) = (m_0 - m_1) \operatorname{sat}(\sigma), \quad \operatorname{sat}(\sigma) = \begin{cases} \sigma & \text{if } |\sigma| \le 1 \\ \operatorname{sign}(\sigma) & \text{if } |\sigma| > 1 \end{cases}$$

This is represented in Lur'e form with:

$$P = \begin{pmatrix} -\alpha(m_1 + 1) & \alpha & 0 \\ 1 & -1 & 1 \\ 0 & -\beta & -\gamma \end{pmatrix}.$$

### B. Arctan Chua System (Fractional)
The system equations are:

$$\begin{aligned}
C D_t^q x &= -\alpha(1 + m)x + \alpha y - \alpha(n - m)\operatorname{arctan}(x) \\
C D_t^q y &= x - y + z \\
C D_t^q z &= -\beta y - \gamma z
\end{aligned}$$

This is represented in Lur'e form with:

$$P = \begin{pmatrix} -\alpha(1 + m) & \alpha & 0 \\ 1 & -1 & 1 \\ 0 & -\beta & -\gamma \end{pmatrix}, \quad \psi(\sigma) = (n - m)\operatorname{arctan}(\sigma).$$

---

## 2. Integer vs. Fractional Transfer Functions

The transfer function relates the output feedback coordinate $\sigma$ to the input vector $b$:

$$W(s) = r^T (s I - P)^{-1} b$$

- **Integer Mode (`transfer_mode = "integer"`)**: Evaluated along the imaginary axis $s = i \omega$:
  $$W(i \omega) = r^T (i \omega I - P)^{-1} b.$$
- **Fractional Mode (`transfer_mode = "fractional"`)**: Evaluated along the fractional principal branch $s^q = (i \omega)^q$:
  $$W_q(i \omega) = r^T (\lambda I - P)^{-1} b, \quad \lambda = (i \omega)^q = \omega^q \exp\left(i q \frac{\pi}{2}\right).$$

*Note: The transfer function is evaluated directly using the spectral parameter $\lambda$ without treating the fractional order as an integer system.*

---

## 3. Numerical Continuation Modes

To transport the heuristic describing-function seed to the original system, we deform the linearised system using a parameter $\eta \in [0, 1]$:

$$D_t^q X = P_0 X + b \, \eta \, \phi(r^T X),$$

where:
- $P_0 = P + k b r^T$ (the linearised matrix)
- $\phi(\sigma) = \psi(\sigma) - k \sigma$ (the nonlinear residual)

- **`continuation_mode = "integer"`**: Uses integer-order ODE integration to step $\eta$ from 0 to 1, regardless of the target fractional order.
- **`continuation_mode = "fractional"`**: Integrates the Caputo fractional system causal history at each stage, ensuring continuous memory propagation.

---

## 4. Full vs. Truncated Sliding Window Memory

Because fractional derivatives have non-local memory, simulating them requires keeping a history of states.
- **Full Memory (`memory_mode = "full"`)**: Retains the entire historical trajectory from the initial step. Accuracy is exact but computational complexity scales quadratically.
- **Windowed Memory (`memory_mode = "window"`)**: Approximates the fractional integral by truncating the memory summation to only include the last $M = \text{memory\_window\_length}$ steps, significantly accelerating calculations.

---

## 5. Important Scientific Warning: The Weyl-Caputo Bridge

> [!IMPORTANT]
> The describing function method, Fourier coefficients, Nyquist criteria, and harmonic balance are heuristic tools formulated for an idealised, steady-state periodic regime (Liouville-Weyl/Weyl fractional calculus).
> 
> Real physical and numerical simulations of fractional-order initial value problems are computed under causal operators (Caputo fractional derivative). The describing function seeds are approximations and do not mathematically prove the existence of an exact period cycle in Caputo systems. They serve as numerical starting coordinates for parameter continuation and trajectory exploration.
> 
> Therefore, describing function crossings and seed generation **never** prove the hiddenness or existence of an attractor on their own. Strong, conservative verification requires rigorous neighborhood sampling around all equilibria and complete basin classifications.

---

## 6. Output States & Verdicts

The final execution of the workflow returns one of the following conservative classification labels:
- `hidden_verified_under_tested_radii`: All equilibria are unstable, and no neighborhood trajectories excited the candidate attractor (0 contact hits).
- `compatible_with_hiddenness_under_tested_radii`: No contact was detected, but at least one stable equilibrium is present.
- `self_excited_contact_detected`: At least one trajectory from the neighborhood of an equilibrium point converged to the candidate attractor, proving it is self-excited.
- `not_supported`: The describing function seed itself failed to converge to the bounded candidate attractor.
- `numerical_failure`: Numerical overflow, NaN, or solver exception occurred.
