# Scientific Scope

## Scope

The primary scope of `hidden-attractors-fo` is the numerical localization and
verification of hidden-attractor candidates in commensurate-order Caputo
fractional systems that admit, or are manually transformed into, a scalar
Lur'e representation

\[
{}^C D_t^q X = P X + b\psi(r^T X),
\qquad 0<q\le 1.
\]

The scalar feedback coordinate is

\[
\sigma=r^T X.
\]

Describing-function and Nyquist computations are seed-generation tools, not
hiddenness proofs.

The maintained protocol records equilibria, local Matignon classifications,
harmonic or Nyquist seeds, candidate transport, Caputo integration contracts,
and finite neighborhood or basin evidence. A numerically auditable candidate
is not automatically a mathematical proof of hiddenness.

## Out of scope

- Arbitrary nonlinear systems without a supplied Lur'e split.
- Multi-nonlinearity systems unless reduced to an admissible scalar feedback channel.
- Incommensurate fractional systems unless a separate solver and stability contract is implemented.
- Non-Caputo fractional derivatives unless explicitly documented.
- Purely visual attractor claims.
- Hiddenness claims based only on DF/Nyquist, FFT/PSD, Lyapunov estimates, 0-1 test, phase portraits or bifurcation plots.
- Exact periodic-orbit claims for autonomous Caputo systems without theoretical qualification.

## Admissible systems

An admissible maintained workflow supplies a vector field, parameters,
equilibria, a Jacobian when local stability is used, the commensurate Caputo
order `q`, and an explicit solver and memory policy. DF/Nyquist routes also
require a manually reviewed scalar Lur'e split `(P, b, r, psi)`, the branch
convention for `(j omega)^q`, and a seed interpretation.

The system registry can store more general vector fields for diagnostics.
Registration alone does not make a system admissible for the full
hidden-attractor protocol.

## Required evidence before hiddenness claims

- DF/Nyquist computations produce seeds.
- Continuation transports seeds or candidates and records their destination scope.
- ABM/EFORK simulate a Caputo system under an explicit numerical contract.
- Matignon classifies equilibria locally.
- Lyapunov, FFT, PSD, Poincare, 0-1 and bifurcation outputs are diagnostics.
- Hiddenness can be promoted only through neighborhood or basin tests around all equilibria under the recorded numerical contract.

## Canonical Attractor Status Taxonomy

All attractor verification workflows, output reports, and manifests use a unified taxonomy of status labels. Any older or legacy labels are silently normalized to these canonical values upon import.

| Status | Description |
| :--- | :--- |
| `candidate` | A localized trajectory branch under investigation, showing chaotic dynamics but pending complete hiddenness checks. |
| `hidden_under_tested_neighborhoods` | Strict verification contract passed: no equilibrium basin intersections detected within all required neighborhood radii under robust controls and metadata. |
| `compatible_with_hiddenness` | Neighborhood tests showed no basin intersections, but the full auditable verification protocol or metadata is incomplete. |
| `self_excited` | Candidate is confirmed self-excited under the recorded local-neighborhood contract. |
| `nonchaotic` | Dynamics are regular, periodic, or quasiperiodic rather than chaotic. |
| `diverged` | Trajectory is unbounded or diverged to infinity. |
| `inconclusive` | Diagnostics are conflicting or numerical integration failures occurred during neighborhood tests. |
| `rejected` | Candidate is rejected due to divergence, invalid configuration, local-contract contact, or failure to localize. |
| `not_tested` | Candidate has not been subjected to verification tests. |

> [!NOTE]
> Legacy `hidden_verified` and `hiddenness_supported_under_tested_neighborhoods` map to canonical `hidden_under_tested_neighborhoods`, and legacy `compatible_with_hiddenness_under_tested_radii` maps to canonical `compatible_with_hiddenness`.

## Published reference coverage

The library separates complete reproductions from partial implementations, reference-data extraction, and diagnostic comparison lanes.

- **Kuznetsov et al. 2017, integer Chua reference**: reproduced as an executable regression case. The available published data are sufficient to verify the scalar Lur'e split, frequency, gain, amplitude, seed construction, and maintained integer-order reference trajectory.

- **Danca 2017, fractional non-smooth Chua case**: partially implemented and used as a reference case. The paper does not report enough numerical information to claim full trajectory reproduction of the published hidden attractor. Missing data include the exact describing-function frequency, gain, amplitude, seed coordinates, exact hidden-attractor initial condition, and quantitative Lyapunov data. The library therefore validates the model equations, equilibria, local Matignon classification, configured Caputo/ABM-style controls, and neighborhood-test infrastructure, but it does not claim full reproduction of the published attractor.

- **Danca 2017, generalized Lorenz and Rabinovich-Fabrikant cases**: reference-data only. They are not promoted as built-in quantitative reproduction claims in the current release.

- **Wu et al. 2023, fractional arctan Chua case**: partially implemented as a reference case. The library includes the model algebra, equilibria, Jacobian, reported initial conditions, scalar Lur'e representation, and isolated ADM comparison path where documented. The paper does not report all numerical values required for a complete independent reproduction of the describing-function seed and published attractor workflow, including `omega0`, `k`, `a0`, and exact seed coordinates. The current library workflow also uses a Caputo-compatible validation route that is not identical to the published ADM/DSP-oriented workflow. Therefore this case must not be described as fully reproduced.

- **Diethelm-Ford-Freed ABM and Ghoreishi-Ghaffari-Saad EFORK methods**: used as numerical-method validation references. These lanes validate solver contracts and implementation behavior; they should not be described as reproduction of a hidden-attractor result.

- **Published Lyapunov comparison lanes**: opt-in diagnostic comparisons. When discrepancies are recorded, they must remain visible and the result must be described as a comparison lane with documented discrepancies, not as full quantitative reproduction.

## What the library only treats as diagnostics

Lyapunov estimates, cloned-dynamics spectra, FFT/PSD panels, Poincare sections,
0-1 statistics, phase portraits, bifurcation plots, finite-memory sensitivity
checks, robustness overlays and Machado/FDF variants are diagnostics or seed
extensions. Each is useful for auditing a candidate. None is a hiddenness
proof by itself. In particular, Machado/FDF is auxiliary and not a hiddenness
proof. The `machado_centered` and `machado_biased` families are registered in the library schema as planned/unsupported, so they cannot be executed in this release.\n
## Reference metadata audit

This table is backed by `docs/references.bib` and repository validation
metadata. During the scope review, four conflicting records were reconciled:

- Kuznetsov et al. 2017 had two internal DOI values that resolved to unrelated
  articles. The maintained Chua reference is the IFAC-PapersOnLine article at
  [10.1016/j.ifacol.2017.08.470](https://doi.org/10.1016/j.ifacol.2017.08.470).
- Danca 2017 used `10.1007/s11071-017-3462-1` in older documentation while the
  published-case YAML used `10.1007/s11071-017-3472-7`. The latter matches the
  author publication list and is now used consistently.
- Machado 2015 previously pointed to unrelated Nonlinear Dynamics metadata.
  The project bibliography and publisher page identify the Signal Processing
  article [10.1016/j.sigpro.2014.05.012](https://doi.org/10.1016/j.sigpro.2014.05.012).
- Wu et al. 2023 previously had stale registry metadata. The maintained
  published-case contract uses the Results in Physics article
  [10.1016/j.rinp.2023.106866](https://doi.org/10.1016/j.rinp.2023.106866).

## Literature comparison table

| Article | System / object | Order | Method in article | Library coverage | Library extension | Library modules / evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Chua / Madan reference on Chua circuit family | Base Chua circuit | Integer | Circuit model and chaotic dynamics | Theoretical support only: base model family | Fractional and auditable workflow variants | `hidden_attractors/models/chua.py` |
| Caputo 1967 | Caputo derivative | Fractional | Dissipation model with fractional derivative | Theoretical support only | Explicit solver and memory contracts | `hidden_attractors/integrations/abm.py`, `hidden_attractors/solvers/history.py` |
| Matignon 1996 | Fractional local stability | Commensurate fractional | Eigenvalue-angle stability criterion | Theoretical support only | Automated equilibrium classification | `hidden_attractors/verification/stability.py` |
| Leonov & Kuznetsov 2013 | Hidden/self-excited distinction | Integer dynamical systems | Basin-neighborhood definition | Theoretical support only | Finite sampled promotion contract | `hidden_attractors/verification/hiddenness.py`, `hidden_attractors/workflows/sphere_controls.py` |
| Kuznetsov 2016 | Hidden-attractor review | General | Review and classification | Theoretical support only | Auditable labels and workflow boundaries | `hidden_attractors/verification/classifiers.py` |
| Genesio, Tesi & Villoresi 1993 | Frequency-domain nonlinear-circuit analysis | Integer | Harmonic balance / describing function | Theoretical support only | Seed-generation API | `hidden_attractors/seed_generation/chua.py` |
| Kuznetsov et al. 2017 | Saturation non-smooth Chua | q = 1 | Describing function and trajectory localization from a computed seed | Executable regression: Lur'e split, omega, gain, amplitude, seed branch, and maintained integer-order reference trajectory | Fractional frequency evaluation, Caputo candidate transport, algebraic validation, and neighborhood-test protocol | `hidden_attractors/seed_generation/lure.py`, `hidden_attractors/workflows/integer_lure.py`, `validation/published_cases/kuznetsov2017_chua_integer.yaml` |
| Diethelm, Ford & Freed 2002 | Caputo FDE integration | Fractional | ABM predictor-corrector | Implements and validates the maintained ABM numerical contract | Neighborhood controls and solver comparisons | `hidden_attractors/integrations/abm.py`, `hidden_attractors/workflows/danca_abm_sphere_controls.py` |
| Danca 2017 | Fractional non-smooth Chua; generalized Lorenz and Rabinovich-Fabrikant context | Chua case q = 0.9998 | Caputo ABM integration and equilibrium-neighborhood analysis | Partial reference implementation: equations, parameters, equilibria, local Matignon classification, configured ABM-style controls, and neighborhood-test infrastructure. Full published hidden-attractor trajectory reproduction is not claimed because key numerical data are not reported. | Automated neighborhood tests, robustness checks, ABM/EFORK comparison, and validation envelopes | `hidden_attractors/integrations/abm.py`, `hidden_attractors/verification/stability.py`, `hidden_attractors/verification/hiddenness.py`, `hidden_attractors/workflows/danca_abm_sphere_controls.py`, `validation/published_cases/danca2017_chua_fractional_saturation.yaml` |
| Wu et al. 2023 | Fractional Chua with arctan nonlinearity | Reported case q = 0.99 | Initial-value localization algorithm, numerical simulation, and DSP implementation | Partial reference implementation: arctan model algebra, equilibria, Jacobian, reported initial conditions, scalar Lur'e representation, and isolated ADM comparison path. Full independent reproduction of the published attractor workflow is not claimed because required seed-generation and sweep data are incomplete. | Lur'e/Nyquist checks, fractional-frequency convention, Caputo-compatible validation route, and software tests | `hidden_attractors/models/chua.py`, `hidden_attractors/systems/builtins.py`, `hidden_attractors/seed_generation/chua_arctan_wu2023.py`, `hidden_attractors/integrations/adm_wu2023.py`, `validation/published_cases/wu2023_chua_fractional_arctan.yaml` |
| Machado 2015 | Backlash / fractional describing function | Fractional describing-function parameter, not Caputo state order | Fractional describing function | Theoretical support only; Machado/FDF is an auxiliary seed-generation extension and is currently planned/unsupported. | Auxiliary seed-generation extension; currently planned/unsupported | `hidden_attractors/seed_generation/lure.py`, `hidden_attractors/seed_generation/chua.py`, `tests/test_classical_route_scope.py` |
| Ghoreishi, Ghaffari & Saad 2023 | Fractional Runge-Kutta schemes | Fractional | EFORK schemes | Numerical-method validation reference using manufactured-solution checks | Chua solver comparison lane | `hidden_attractors/solvers/efork_published.py`, `docs/efork3_validation.md` |
| Guan & Xie 2025 | Hidden-attractor localization review | General | Review of localization methods | Theoretical support only | Method inventory and reference comparisons | `hidden_attractors/workflows/integer_lure.py`, `docs/integer_chua_reference.md` |
| Petras 2008 | Fractional Chua family | Fractional | Fractional Chua model note | Theoretical support only | Family context for non-smooth Chua algebra | `hidden_attractors/models/chua.py` |
| Sene 2021 | Caputo-Liouville Chua family | Fractional | Fractional Chua analysis | Theoretical support only | Documented model-family boundary | `hidden_attractors/models/chua.py` |
| Tavazoei & Haeri 2009 | Periodicity caveat | Autonomous fractional systems | Non-existence qualification for periodic solutions | Theoretical support only | Explicit caveat against unqualified exact periodic-orbit claims | `docs/scientific_scope.md` |
| Kaslik & Sivasundaram 2012 | Periodicity caveat | Fractional systems | Periodic-solution non-existence analysis | Theoretical support only | Explicit caveat against unqualified exact periodic-orbit claims | `docs/scientific_scope.md` |
| Area, Losada & Nieto 2014 | Periodic fractional derivatives | Fractional | Periodic derivative and primitive analysis | Theoretical support only | Explicit caveat against unqualified exact periodic-orbit claims | `docs/scientific_scope.md` |
| Vigue et al. 2019 | Fractional continuation | Fractional | Continuation of periodic solutions with qualifications | Theoretical support only | Candidate transport boundary; no exact-orbit promotion | `validation/python/published_continuation_comparison.py` |
| Deng 2007 | Short-memory ABM | Fractional | Short-memory predictor-corrector | Theoretical support only | Explicit finite-memory policies | `hidden_attractors/solvers/history.py`, `validation/fractional_memory_validation/README.md` |
| Mendes, Salgado & Aguirre 2019 | Initial-condition memory effect | Caputo fractional | Numerical treatment of infinity-memory effect | Theoretical support only | Memory-policy audit | `validation/fractional_memory_validation/README.md` |
| Hai, Yu, Xu & Ren 2022 | Short-term memory stability | Fractional | Stability analysis for short-term memory | Theoretical support only | Window-sensitivity audit | `validation/fractional_memory_validation/README.md` |
| Benettin et al. 1980 | Lyapunov characteristic exponents | Integer | Variational propagation and orthonormalization | Diagnostic only: integer-reference implementation | Frozen `q=1` QR lane | `hidden_attractors/analysis/lyapunov.py`, `docs/lyapunov_methods.md` |
| Skokos 2010 | Lyapunov exponent computation | Integer | Computational review | Theoretical support only | Diagnostic documentation | `hidden_attractors/analysis/lyapunov.py` |
| Christiansen & Rugh 1997 | Lyapunov spectra | Integer | Continuous Gram-Schmidt orthonormalization | Theoretical support only | Diagnostic comparison context | `hidden_attractors/analysis/lyapunov.py` |
| Danca & Kuznetsov 2018 | Fractional Lyapunov exponents | Caputo fractional | Extended original-variational system with memory and reorthonormalization | Diagnostic comparison lane with recorded RF discrepancy | Keeps the local full-history QR contract separate | `hidden_attractors/analysis/lyapunov_fractional.py`, `validation/chaos_validation/lyapunov_methods/fractional_variational_dk2018_block_restart_abm_gs_published/` |
| Fischer, Zourmba & Mohamadou 2020 | Cloned-dynamics Lyapunov spectra | Fractional | Cloned dynamics and Gram-Schmidt | Diagnostic comparison lane; published comparison rows retain documented discrepancies | Separate GS and experimental QR comparison lanes | `hidden_attractors/analysis/lyapunov_cloned.py`, `validation/chaos_validation/lyapunov_methods/fractional_cloned_dynamics_abm_gs_published/` |
