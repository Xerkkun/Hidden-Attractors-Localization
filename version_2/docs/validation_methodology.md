# Validation & Verification Methodology

This document outlines the rigorous mathematical and computational validation hierarchy implemented in the `hidden_attractors` library to distinguish between seed-generation heuristic stages and formal dynamical proofs of hiddenness.

---

## 1. Validation State Separation

To prevent false claims of hiddenness, the library strictly divides execution into five promotion states:

1. **`seed_found`**: Initial candidate generated through frequency balance (e.g., Describing Functions, Nyquist scanning).
2. **`candidate_attractor`**: Verified to be bounded and converge to non-trivial persistent dynamics under standard integration parameters.
3. **`chaotic_candidate`**: Confirmed chaotic via Gottwald-Melbourne 0-1 test or positive maximum Lyapunov exponents.
4. **`hidden_compatible`**: Free of intersections with equilibrium neighborhoods across all tested radii.
5. **`hidden_verified`**: Meets all requirements of `hidden_compatible` plus passes the complete set of boundary basin-of-attraction slices (`xy_close`, `xy_large`, `xz_close`, `xz_large`, `yz_close`, `yz_large`).

---

## 2. Weyl–Caputo Operator Evaluation

For systems with fractional order $q < 1.0$, frequency scans and transfer functions are evaluated formally on the principal branch:

$$\lambda = (j\omega)^q = \omega^q e^{j q \pi / 2}$$

Prohibiting integer-order shortcuts ensures the predicted harmonic seeds correctly correspond to the Caputo fractional derivative memory structure. When $q < 1.0$, a mandated "Weyl–Caputo Note" is automatically appended to summaries.

---

## 3. Lur'e Compatibility

Describing function approximations are valid only if the system fits the Lur'e feedback representation. The `LureCompatibilityValidator` evaluates compatibility on a random point cloud and classifies systems into:
- **`LURE_DIRECT`**: Directly equivalent.
- **`LURE_LINEAR_CHANGE`**: Equivalent after a linear change of coordinates $X = S Z$.
- **`LURE_APPROXIMATE`**: Matches with a small reconstruction residual.
- **`NOT_COMPATIBLE`**: Fails to match, blocking Describing Function scans unless forced by configurations (restricting promotion to `seed_found` only).

---

## 4. Non-Smooth Vector Fields

 piece-wise continuous systems (e.g., containing $\text{sat}(x)$, $|x|$, or $\text{sign}(x)$) violate global differentiability:
- **Lipschitz Continuity**: $\text{sat}(x)$ and $|x|$ are continuous but non-differentiable at switching surfaces (e.g., $x = \pm 1$).
- **Discontinuities**: $\text{sign}(x)$ is discontinuous, blocking standard ODE solvers by default and requiring regularized or Filippov-based solvers.
- **Switching Crossings**: Trajectories crossing these surfaces trigger alerts as global symbolic Jacobians are invalid at these boundaries.
- **Matignon stability**: Equilibria lying exactly on switching boundaries are classified as `nonsmooth_indeterminate`.

---

## 5. Symmetry Exploitation

System symmetries (such as inversion $T(X) = -X$ or rotation $T(x,y,z) = (-x,-y,z)$) are verified numerically. If a symmetry is confirmed:
- Symmetric seeds $T(X_0)$ are automatically generated and deduplicated.
- Continuation sweeps and attraction basin tests are queued for both symmetric branches to map all coexisting attractors.
