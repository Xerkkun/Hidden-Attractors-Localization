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

## 2. Core Lur'e Parameters & Differences

The workflow utilizes distinct modes for each simulation layer:

| Parameter | Options | Description |
| :--- | :--- | :--- |
| **`transfer_mode`** | `integer` or `fractional` | Decides if the linear feedback loop is modeled as an integer or fractional order system. Influences $W(s)$ evaluation along the imaginary axis $s = i\omega$ vs the fractional principal branch $s^q = (i\omega)^q$. |
| **`seed_construction`** | `modal` or `closed_form_integer` | Configures seed reconstruction. `modal` projects the crossing eigenvectors, while `closed_form_integer` emulates standard sinusoidal approximations. |
| **`continuation_mode`** | `integer` or `fractional` | Defines how eta continuation steps step $\eta$ from 0 to 1. `integer` uses standard ODE steps; `fractional` retains Caputo causality history. |
| **`dynamics_mode`** | `integer`, `fractional`, or `system` | Configures the final integration and attractor evaluation. `system` respects the system's actual order $q$; `integer` forces $q=1.0$; `fractional` forces $q < 1.0$. |

---

## 3. Describing Function Evaluation & Resolution

To handle non-smooth systems elegantly without numerical instabilities, the library utilizes a unified Describing Function Manager with 5 evaluation modes:

1. **`closed_form`**: Uses the analytical formula directly.
2. **`piecewise_closed_form`**: Evaluates different closed-form sub-equations depending on regime boundaries.
3. **`quadrature`**: Resolves the describing function via standard scipy global quadrature `quad(...)`.
4. **`segmented_quadrature`**: Partitions the integration interval $[0, \pi]$ exactly at the non-smooth regime breakpoints to prevent roundoff error warnings.
5. **`auto`** (Default): Selects the best method automatically:
   - If there is a closed-form formula, use it.
   - If the system is non-smooth (e.g. saturation) and has a piecewise form, use `piecewise_closed_form`.
   - If the system is non-smooth but has no closed form, use `segmented_quadrature` at the breakpoints.
   - If the system is smooth, use `closed_form` if registered, else fall back to standard `quadrature`.

---

## 4. Why Piecewise Closed-Form for Saturation

For Saturated Chua, the nonlinearity $\psi(\sigma) = (m_0 - m_1)\operatorname{sat}(\sigma)$ is non-smooth because its derivative is discontinuous at $|\sigma| = 1$.

Evaluating its first-harmonic describing function using global numerical quadrature $N(A) = \frac{2}{\pi A} \int_{0}^{\pi} \psi(A \cos \theta) \cos \theta \, d\theta$ causes the integrator to hit the regime change boundary ($A \cos \theta = \pm 1$), throwing the notorious `IntegrationWarning: The occurrence of roundoff error is detected...`.

To avoid this, we implement the analytical **piecewise closed-form** describing function:

- **For $0 < A \le 1$:** The input lies entirely inside the linear regime:
  $$N_{\text{sat}}(A) = m_0 - m_1$$
- **For $A > 1$:** The input crosses into the saturated regime:
  $$N_{\text{sat}}(A) = \frac{2(m_0 - m_1)}{\pi} \left[ \arcsin\left(\frac{1}{A}\right) + \frac{1}{A}\sqrt{1 - \frac{1}{A^2}} \right]$$

Using this piecewise closed-form representation completely avoids integration roundoff errors, providing an exact, stable, and instantaneous solution.

---

## 5. Important Scientific Warning: The Weyl-Caputo Bridge

> [!IMPORTANT]
> The describing function method, Fourier coefficients, Nyquist criteria, and harmonic balance are heuristic tools formulated for an idealised, steady-state periodic regime (Liouville-Weyl/Weyl fractional calculus).
> 
> Real physical and numerical simulations of fractional-order initial value problems are computed under causal operators (Caputo fractional derivative). The describing function seeds are approximations and do not mathematically prove the existence of an exact period cycle in Caputo systems. They serve as numerical starting coordinates for parameter continuation and trajectory exploration.
> 
> Therefore, describing function crossings and seed generation **never** prove the hiddenness or existence of an attractor on their own. Strong, conservative verification requires rigorous neighborhood sampling around all equilibria and complete basin classifications.
