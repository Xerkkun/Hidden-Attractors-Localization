# Thesis Claims Matrix

## Purpose

This document serves as the official Claims Matrix for the thesis. It enumerates the scientific claims that can be rigorously supported by the evidence currently available in the repository.

By maintaining this matrix, the project establishes a strict defensive barrier against scientific overclaims in academic text, presentations, publications, and repository documentation. Specifically:
- **Describing function (DF) analysis, Nyquist conditions, and numerical continuation** are heuristic tools for **generating seeds and candidates** only. They do not constitute mathematical proofs of existence or hiddenness.
- **Hiddenness verification** requires checking the transient behavior starting from tested neighborhoods of **all equilibrium points** of the system.
- Claims are categorized by their evidence status to maintain a conservative, auditable record of what is proven, candidate-only, or rejected.

---

## Evidence Status Vocabulary

- **`probado`**: The claim is supported by automated regression tests, numerical validation summaries, or a complete contract execution within the declared scope.
- **`reproducido`**: A known benchmark result or published case that the repository reproduces under a specific, documented configuration.
- **`rechazado`**: A candidate that failed the validation contract (e.g., due to self-excited contact detected, periodicity, divergence, or lack of chaos).
- **`candidato`**: A promising numerical attractor or seed found, but lacking a complete mathematical or global hiddenness contract validation.
- **`no_certificado`**: Partial evidence is available; it must not be promoted as a strong thesis result.
- **`pendiente`**: The methodology has not yet been applied or numerical validation is incomplete.

---

## Claim Table

| claim_id | afirmación | sistema | orden | estado | evidencia_json | evidencia_csv | figuras | comentario_metodologico |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`CLAIM-CHUA-INTEGER-001`** | El flujo de trabajo reproduce correctamente un atractor caótico de Chua entero usado como caso de referencia para validar la ruta entera de semillas, simulación y diagnóstico. | Chua entero / Chua integer reference | entero, q = 1 | `reproducido` | `validation/references/kuznetsov2017_expected.json`, `validation/reference_cases/chua_integer_q1/06_hiddenness/hiddenness_validation_summary.json` | `validation/reference_cases/chua_integer_q1/06_hiddenness/summary_by_radius_integer.csv` | `library_figures/by_run/integer_lure_seed/png/nyquist.png`, `library_figures/by_run/integer_lure_seed/png/continuation.png`, `library_figures/by_run/integer_lure_hiddenness/png/hiddenness.png` | Valida la ruta entera y sirve como control de software, pero no prueba por sí mismo resultados fraccionarios. |
| **`CLAIM-CHUA-NONSMOOTH-EX1-001`** | El Ejemplo 1 genera candidatos numéricos consistentes mediante función descriptiva sesgada y continuación para el sistema de Chua no suave. | Chua no suave / saturation nonlinearity / Example 1 | fraccionario, q = 0.9998 | `candidato` | `examples/chua_nonsmooth_biased_hidden_attractor/run_example.py` | `validation/09_hiddenness_tests/hiddenness_decisions.csv` (para m1=-1.1468, m0=-0.1768, c=+2.776) | `library_figures/by_run/step2_biased_df/png/biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776_report.png` | La BDF se usa como generador heurístico de semillas. El resultado no es una prueba matemática global. Es compatible bajo pruebas locales (0 hits en Step 3), pero existe contradicción con el candidato danca2017_nearby_saturation_candidate_q09998 en el manifiesto oficial, el cual fue rechazado por contacto autoexcitado. |
| **`CLAIM-CHUA-FRAC-REJECTED-001`** | El candidato oficial evaluado bajo el contrato de vecindades no se promueve como atractor oculto porque se detectó contacto autoexcitado. | Chua no suave / saturation nonlinearity / Candidate branch_0 | fraccionario, q = 0.9998 | `rechazado` | `validation/09_hiddenness_tests/hiddenness_tests_validation_summary.json` | `validation/09_hiddenness_tests/hiddenness_decisions.csv` | ninguna | Muestra que el filtro de ocultedad funciona de forma conservadora. El candidato danca2017_nearby_saturation_candidate_q09998 registró 1305 contactos autoexcitados desde vecindades de E+ y E-. |
| **`CLAIM-CHUA-ARCTAN-FRAC-001`** | La metodología para el sistema de Chua fraccionario con no linealidad arctan está pendiente de aplicación o cierre bajo el contrato completo de semillas, continuación, simulación y ocultedad. | Chua fraccionario arctan | fraccionario, q < 1 (por ejemplo, q = 0.9998) | `pendiente` | ninguna / pendiente | ninguna / pendiente | ninguna / pendiente | No existe todavía promoción de atractor verificado como oculto para este sistema. Debe aplicarse la metodología completa: Lur'e form, fractional transfer, describing function, Nyquist condition, continuation, Caputo ABM/EFORK integration, neighborhood tests of all equilibria, conservative classification. |
| **`CLAIM-METHOD-LURE-FRAC-001`** | La tesis desarrolla una metodología reproducible para generar, transportar, simular y auditar candidatos a atractores ocultos en sistemas compatibles con forma Lur’e de orden entero y fraccionario. | metodología general | entero/fraccionario | `probado` | `docs/tests_inventory.md` | ninguna | ninguna | En contextos de tesis, el propósito de CLAIM-METHOD-LURE-FRAC-001 es defensivo: si no se logra demostrar matemáticamente la existencia global de un atractor oculto en el caso fraccionario (que es un problema abierto extremadamente difícil), el aporte principal de la tesis pasa a ser metodológico y computacional (el framework de software, el protocolo de auditoría de vecindades, y la reproducibilidad). Se restringe a sistemas compatibles con forma Lur’e o transformables a ella con no linealidad escalar compatible. |

---

## Claims Explicitly Not Made

The following claims are strictly **prohibited** as they are not supported by evidence:
- It is **not** claimed that the describing function (DF) method mathematically proves the exact existence of autonomous limit cycles in Caputo fractional systems.
- It is **not** claimed that satisfying the Nyquist stability criterion mathematically proves hiddenness.
- It is **not** claimed that a bounded, finite-time simulation run mathematically proves the hiddenness or boundedness of an attractor.
- It is **not** claimed that a successful numerical continuation mathematically proves hiddenness.
- It is **not** claimed that the fractional Chua arctan system contains a verified, certified hidden attractor.
- It is **not** claimed that the methodology automatically generalizes to any arbitrary fractional-order chaotic system (it is strictly restricted to scalar Lur'e systems).
- It is **not** claimed that global hiddenness is verified if only finite radii, times, and tested neighborhood samples have been evaluated.
- It is **not** claimed that a visual plot of an attractor is sufficient evidence of hiddenness.

---

## Hiddenness Verification Contract

An attractor candidate is classified as **hidden** under the tested numerical contract only if its numerical basin of attraction does not intersect any open neighborhood of any equilibrium point, within the limits of the declared simulation step, time, integration scheme, memory policy, and probing radii.

### Verification Steps:
1. **Calculate all equilibrium points** of the system.
2. **Classify equilibria stability** using Matignon's fractional criterion when $q < 1$.
3. **Simulate trajectories** starting from small spherical shells around each stable/unstable equilibrium.
4. **Simulate the candidate attractor** starting from the seed generated by the BDF continuation.
5. **Compare final states** and trajectories of the equilibrium runs vs. the candidate run.
6. **Detect contacts** (basin overlap/hits) between equilibrium trajectories and the candidate attractor cloud.
7. **Report parameters**: Probing radii, number of samples per radius, integrator step $h$, final time $t_{final}$, and memory policy (e.g., full history vs. finite memory).
8. **Classify conservatively** into one of the following decisions:
   - `hiddenness_supported_under_tested_neighborhoods` (0 hits across all tested neighborhoods)
   - `self_excited_contact_detected` (non-zero hits detected, candidate self-excited)
   - `hiddenness_inconclusive` (divergence or numerical issues)
   - `candidate_rejected` (rejected due to contact or non-chaotic dynamics)

*Note: This classification constitutes numerical evidence under a specific tested contract, not an absolute global mathematical proof.*

---

## Weyl–Caputo Interpretation

- **Caputo Fractional Derivative**: Describes a causal physical system with memory starting from $t_0$.
- **Weyl / Liouville-Weyl Derivative**: Allows the formulation of ideal, steady-state harmonic responses.
- **Describing Function / Harmonic Balance**: Utilizes this steady-state approximation.
- **BDF Seed Synthesis**: The steady-state harmonic solution obtained is used purely as a numerical initial condition (seed) for starting Caputo simulations or homotopic continuation. It does not mathematically prove the existence of an autonomous limit cycle in the Caputo sense.

### Mathematical Convention

The linear transfer function of the scalar Lur'e fractional-order system is defined as:

$$W_q(s) = r^T (s^q I - P)^{-1} b$$

For steady-state harmonic balance, the complex frequency evaluation uses:

$$\lambda = (j\omega)^q = \omega^q e^{j q \pi / 2}$$

It is strictly forbidden to use $W(j\omega)$ (the integer-order frequency response) when $q < 1$.
