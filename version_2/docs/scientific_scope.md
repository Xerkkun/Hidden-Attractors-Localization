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

Use `hiddenness_supported_under_tested_neighborhoods` only when neighborhood
and basin tests around all equilibria, robust reference checks, and the
reproducible `run_metadata.json` envelope satisfy the recorded promotion
contract. If one condition is missing, use
`compatible_with_hiddenness_under_tested_radii`. Legacy `hidden_verified`
inputs are normalized to the sampled-neighborhood label.

## What the library can reproduce

- The Kuznetsov et al. 2017 integer Chua reference branch: the scalar Lur'e
  split, `omega`, `k`, amplitude, seed and the maintained reference trajectory.
- The Danca 2017 fractional non-smooth Chua inputs: equilibria, local Matignon
  classification, ABM-style controls where configured, and solver-comparison
  envelopes. The documented Lorenz and Rabinovich-Fabrikant material is not
  promoted here as a new quantitative reproduction claim.
- The Wu et al. 2023 arctan Chua case: model algebra, equilibria, Jacobian,
  reported initial conditions, scalar Lur'e representation, and isolated ADM
  comparison path.
- The Diethelm-Ford-Freed ABM and Ghoreishi-Ghaffari-Saad EFORK numerical
  contracts exercised by their corresponding validation lanes.
- Published Lyapunov comparison lanes when explicitly run. Smoke tests remain
  separate from published quantitative validation, and recorded
  discrepancies remain visible.

## What the library only treats as diagnostics

Lyapunov estimates, cloned-dynamics spectra, FFT/PSD panels, Poincare sections,
0-1 statistics, phase portraits, bifurcation plots, finite-memory sensitivity
checks, robustness overlays and Machado/FDF variants are diagnostics or seed
extensions. Each is useful for auditing a candidate. None is a hiddenness
proof by itself. In particular, Machado/FDF is auxiliary and not a hiddenness
proof. Las familias `machado_centered` y `machado_biased` están registradas en el esquema de la biblioteca como planificadas (`planned`/`unsupported`), por lo que no se pueden ejecutar en esta entrega.

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

| Article | System / object | Order | Method in article | What the library reproduces | What the library extends | Library modules / evidence |
|---|---|---|---|---|---|---|
| Madan & Chua 1986 | Base Chua circuit | Integer | Circuit model and chaotic dynamics | Theoretical support only: base model family | Fractional and auditable workflow variants | `models/chua.py` |
| Caputo 1967 | Caputo derivative | Fractional | Dissipation model with fractional derivative | Theoretical support only | Explicit solver and memory contracts | `integrations/abm.py`, `solvers/history.py` |
| Matignon 1996 | Fractional local stability | Commensurate fractional | Eigenvalue-angle stability criterion | Theoretical support only | Automated equilibrium classification | `verification/stability.py` |
| Leonov & Kuznetsov 2013 | Hidden/self-excited distinction | Integer dynamical systems | Basin-neighborhood definition | Theoretical support only | Finite sampled promotion contract | `verification/hiddenness.py`, `workflows/sphere_controls.py` |
| Kuznetsov 2016 | Hidden-attractor review | General | Review and classification | Theoretical support only | Auditable labels and workflow boundaries | `verification/classifiers.py` |
| Genesio, Tesi & Villoresi 1993 | Frequency-domain nonlinear-circuit analysis | Integer | Harmonic balance / describing function | Theoretical support only | Seed-generation API | `seed_generation/chua.py` |
| Kuznetsov et al. 2017 | Saturation non-smooth Chua | `q=1` | Describing function and direct integration from a localized seed | Reproduces Lur'e split, `omega`, `k`, amplitude, seed branch and integer reference trajectory | Fractional frequency evaluation, Caputo candidate transport, algebraic validation and hiddenness protocol | `seed_generation/lure.py`, `workflows/integer_lure.py`, `validation/published_cases/kuznetsov2017_chua_integer.yaml` |
| Diethelm, Ford & Freed 2002 | Caputo FDE integration | Fractional | ABM predictor-corrector | Reproduces maintained ABM numerical contract | Neighborhood controls and solver comparisons | `integrations/abm.py`, `workflows/danca_abm_sphere_controls.py` |
| Danca 2017 | Fractional non-smooth Chua; documented generalized Lorenz and Rabinovich-Fabrikant context | Chua case `q=0.9998` | ABM Caputo and equilibrium-neighborhood analysis | Reproduces equilibrium data, Matignon classification and configured ABM-style controls | Automated neighborhood tests, robustness checks, ABM/EFORK comparison and validation envelopes | `integrations/abm.py`, `verification/stability.py`, `verification/hiddenness.py`, `workflows/danca_abm_sphere_controls.py`, `validation/published_cases/danca2017_chua_fractional_saturation.yaml` |
| Wu et al. 2023 | Fractional Chua with arctan nonlinearity | Official case `q=0.99` | Initial-value localization algorithm, numerical simulation and DSP implementation | Reproduces arctan model, equilibria, Jacobian, reported initial conditions, Lur'e representation and isolated ADM comparison | Lur'e/Nyquist checks, fractional-frequency convention and software tests | `models/chua.py`, `systems/builtins.py`, `seed_generation/chua_arctan_wu2023.py`, `integrations/adm_wu2023.py`, `validation/published_cases/wu2023_chua_fractional_arctan.yaml` |
| Machado 2015 | Backlash / fractional describing function | Fractional describing-function parameter, not Caputo state order | Fractional describing function | Theoretical support only: no direct hidden-attractor claim | Auxiliary seed families only; not a hiddenness proof | `seed_generation/lure.py`, `seed_generation/chua.py`, `tests/test_classical_route_scope.py` |
| Ghoreishi, Ghaffari & Saad 2023 | Fractional Runge-Kutta schemes | Fractional | EFORK schemes | Reproduces manufactured-solution EFORK checks | Chua solver comparison lane | `solvers/efork_published.py`, `docs/efork3_validation.md` |
| Guan & Xie 2025 | Hidden-attractor localization review | General | Review of localization methods | Theoretical support only | Method inventory and reference comparisons | `workflows/integer_lure.py`, `docs/integer_chua_reference.md` |
| Petras 2008 | Fractional Chua family | Fractional | Fractional Chua model note | Theoretical support only | Family context for non-smooth Chua algebra | `models/chua.py` |
| Sene 2021 | Caputo-Liouville Chua family | Fractional | Fractional Chua analysis | Theoretical support only | Documented model-family boundary | `models/chua.py` |
| Tavazoei & Haeri 2009 | Periodicity caveat | Autonomous fractional systems | Non-existence qualification for periodic solutions | Theoretical support only | Explicit caveat against unqualified exact periodic-orbit claims | `docs/scientific_scope.md` |
| Kaslik & Sivasundaram 2012 | Periodicity caveat | Fractional systems | Periodic-solution non-existence analysis | Theoretical support only | Explicit caveat against unqualified exact periodic-orbit claims | `docs/scientific_scope.md` |
| Area, Losada & Nieto 2014 | Periodic fractional derivatives | Fractional | Periodic derivative and primitive analysis | Theoretical support only | Explicit caveat against unqualified exact periodic-orbit claims | `docs/scientific_scope.md` |
| Vigue et al. 2019 | Fractional continuation | Fractional | Continuation of periodic solutions with qualifications | Theoretical support only | Candidate transport boundary; no exact-orbit promotion | `validation/python/published_continuation_comparison.py` |
| Deng 2007 | Short-memory ABM | Fractional | Short-memory predictor-corrector | Theoretical support only | Explicit finite-memory policies | `solvers/history.py`, `validation/fractional_memory_validation/README.md` |
| Mendes, Salgado & Aguirre 2019 | Initial-condition memory effect | Caputo fractional | Numerical treatment of infinity-memory effect | Theoretical support only | Memory-policy audit | `validation/fractional_memory_validation/README.md` |
| Hai, Yu, Xu & Ren 2022 | Short-term memory stability | Fractional | Stability analysis for short-term memory | Theoretical support only | Window-sensitivity audit | `validation/fractional_memory_validation/README.md` |
| Benettin et al. 1980 | Lyapunov characteristic exponents | Integer | Variational propagation and orthonormalization | Diagnostic only: integer-reference implementation | Frozen `q=1` QR lane | `analysis/lyapunov.py`, `docs/lyapunov_methods.md` |
| Skokos 2010 | Lyapunov exponent computation | Integer | Computational review | Theoretical support only | Diagnostic documentation | `analysis/lyapunov.py` |
| Christiansen & Rugh 1997 | Lyapunov spectra | Integer | Continuous Gram-Schmidt orthonormalization | Theoretical support only | Diagnostic comparison context | `analysis/lyapunov.py` |
| Danca & Kuznetsov 2018 | Fractional Lyapunov exponents | Caputo fractional | Extended original-variational system with memory and reorthonormalization | Reproduces an opt-in published comparison lane with a recorded RF discrepancy | Keeps the local full-history QR contract separate | `analysis/lyapunov_fractional.py`, `validation/chaos_validation/lyapunov_methods/fractional_variational_dk2018_block_restart_abm_gs_published/` |
| Fischer, Zourmba & Mohamadou 2020 | Cloned-dynamics Lyapunov spectra | Fractional | Cloned dynamics and Gram-Schmidt | Diagnostic only: published comparison rows retain documented discrepancies | Separate GS and experimental QR comparison lanes | `analysis/lyapunov_cloned.py`, `validation/chaos_validation/lyapunov_methods/fractional_cloned_dynamics_abm_gs_published/` |
