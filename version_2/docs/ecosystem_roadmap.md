# Hidden Attractors FO as a research dynamics engine

## Product thesis

Hidden Attractors FO will be a general, evidence-aware computation library for continuous, discrete, and fractional-order dynamical systems, with a distinctive high-level route for locating and qualifying hidden-attractor candidates.

The project is not intended to reproduce the function lists of pynamicalsys or DynamicalSystems.jl. Its value should come from a coherent research contract:

- one structured system definition across flows, maps, and fractional models;
- numerical results that retain method, parameters, status, warnings, and provenance;
- explicit separation between trajectory generation, diagnostic evidence, attractor classification, and hiddenness qualification;
- reproducible workflows rather than disconnected plotting functions;
- optimized implementations selected per operation without exposing backend details to users;
- a stable API consumable by scripts, notebooks, command-line workflows, and Toolbox Chaos.

## Architectural layers

1. **System model** — variables, parameters, equations, Jacobian, equilibria, references, capabilities, and system kind.
2. **Numerical execution** — integer, discrete, and fractional solvers behind one result contract.
3. **Analysis** — spectra, Lyapunov methods, recurrence, dimensions, entropies, Poincare sections, escape, bifurcation, and basin metrics.
4. **Attractor workflows** — detection, clustering, continuation, coexistence, basin boundaries, and uncertainty.
5. **Hidden-attractor localization** — Lur'e seed generation, continuation, equilibrium-neighbourhood controls, robustness, and evidence-ranked conclusions.
6. **Validation and provenance** — reference solutions, convergence, cross-backend agreement, manifests, frozen configurations, and machine-readable reports.

Each public algorithm must return a structured result. A result must never reduce a finite numerical protocol to an unqualified boolean such as `is_chaotic=True` or `is_hidden=True`.

## Distinctive research contribution

The library should compete on scientific reliability and composability, not on raw feature count. Its differentiators are:

- evidence-ranked outcomes such as `candidate`, `supported_under_protocol`, `unresolved`, and `numerically_unstable`;
- equilibrium-aware classification, including no equilibrium and non-isolated equilibrium sets;
- first-class fractional-order contracts and validation;
- localization workflows that preserve their seed, continuation path, controls, and limitations;
- reusable experiment manifests that can regenerate figures and tables;
- a safe expression system that allows a graphical client to define models without executing user code.

## Capability roadmap and release gates

### Gate A — common engine contract

- [x] structured `ChaoticSystem` metadata for flows and maps;
- [x] safe expression-defined systems;
- [x] structured trajectory result for flows and maps;
- [x] amplitude spectrum and Welch PSD;
- [x] Toolbox Chaos bridge prototype;
- [ ] versioned JSON schema for system definitions;
- [ ] persistent result provenance with package version, commit, platform, and configuration hash;
- [ ] fresh-environment installation test for the GUI/engine pair.

### Gate B — dependable general dynamics core

- equilibria and numerical root search;
- analytic, automatic, and finite-difference Jacobians with declared provenance;
- local stability for flows and maps;
- event detection and Poincare sections;
- integer Lyapunov spectra with convergence history;
- parameter sweeps, bidirectional continuation, and bifurcation summaries;
- escape times, attractor clustering, basin fractions, Jaccard agreement, and unresolved classification;
- reference validation for representative flows and maps.

### Gate C — nonlinear time-series analysis

- delay reconstruction, mutual information, and false nearest neighbours;
- recurrence plots and recurrence quantification;
- correlation and information dimensions;
- entropy and complexity families with assumption checks;
- surrogate-data workflows;
- transition and regime-change diagnostics.

### Gate D — global attractor analysis

- attractor detection independent of manually named classes;
- periodic-orbit detection and stability;
- basin boundaries, basin entropy, edge states, and tipping diagnostics;
- continuation of attractors and state transitions;
- parallel ensembles on CPU, followed by GPU work only where benchmarks justify it.

### Gate E — fractional-order research depth

- multiple Caputo formulations and cross-method validation;
- commensurate and incommensurate order support where mathematically justified;
- memory-acceleration strategies with controlled error;
- fractional stability tools;
- Lyapunov and bifurcation protocols whose limitations are explicit;
- published-case reproduction separated from manufactured and internal validation.

### Gate F — hidden-attractor localization platform

- multiple seed-generation strategies;
- continuation from auxiliary systems;
- adaptive search outside equilibrium-connected basins;
- complete equilibrium-neighbourhood controls;
- multi-resolution hiddenness evidence;
- comparison of localization routes under a shared protocol;
- benchmark corpus with independently reviewable outputs.

## Optimization policy

There will be one canonical public API and one canonical implementation path per operation. NumPy/SciPy are preferred for mature numerical primitives; Numba or native code is used for profiled kernels; parallel and GPU implementations are added only with controlled equivalence tests. Julia may be an optional interoperability backend for algorithms that are scientifically and operationally justified, but it is not a mandatory runtime for Toolbox Chaos.

Performance claims require identical numerical contracts, same-machine comparison, warm-up policy, repetitions, robust summary statistics, and saved raw timings. Portability demonstrations across different machines are not speed comparisons.

## Adoption by other research groups

External adoption depends on lowering scientific and operational risk:

- stable minimal API and deprecation policy;
- model and result schemas that other groups can extend;
- focused tutorials based on published systems;
- reference datasets and expected tolerances;
- contributor guide and review checklist for new algorithms;
- public issue templates for model requests, numerical discrepancies, and validation evidence;
- citation metadata, archived releases, and reproducible examples;
- at least one independently installed and reproduced workflow before a JOSS submission.

## Definition of done for a new analysis method

A method is not complete when it only produces a plot. It is complete when it has:

1. a documented mathematical definition and scope;
2. a structured input and result contract;
3. unit tests and at least one independent reference comparison;
4. convergence or sensitivity checks where applicable;
5. failure and unresolved states;
6. provenance and serializable outputs;
7. a runnable example;
8. a documented evidence boundary;
9. a benchmark only if performance is part of the claim.
