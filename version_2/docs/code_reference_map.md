# Code Reference Map

Every calculation-facing function should be traceable to one of three sources:
a published reference, a local numerical contract documented in the report, or
a clearly marked utility role. This page is the first audit table for that
policy.

## Core Model

| Code | Purpose | Reference source |
|---|---|---|
| `hidden_attractors.models.chua.ChuaParameters` | Parameter container for the non-smooth fractional Chua case | M. F. Danca, "Hidden Chaotic Attractors in Fractional-Order Systems"; R. N. Madan and L. O. Chua, "Chaos in Chua's Circuit" |
| `hidden_attractors.models.chua.nonlinearity_nonsmooth` | Non-smooth Chua characteristic, linear by pieces | R. N. Madan and L. O. Chua, "Chaos in Chua's Circuit"; Danca's non-smooth fractional Chua example |
| `hidden_attractors.models.chua.rhs_nonsmooth` | Vector field used inside Caputo/EFORK integrations | M. Caputo, "Linear Models of Dissipation whose Q is almost Frequency Independent-II"; M. F. Danca, "Hidden Chaotic Attractors in Fractional-Order Systems" |
| `hidden_attractors.models.chua.equilibria_nonsmooth` | Equilibria of the non-smooth Chua model | Chua circuit model plus the fractional stability interpretation of D. Matignon, "Stability Results for Fractional Differential Equations with Applications to Control Processing" |
| `hidden_attractors.models.chua.jacobian_nonsmooth` | Regional Jacobian and switching-surface guard for the non-smooth model | Danca's non-smooth fractional Chua equilibria/stability table; Matignon local stability criterion |

## Seed Generation

| Code | Purpose | Reference source |
|---|---|---|
| `hidden_attractors.seed_generation.find_harmonic_seed` | Classical or Machado describing-function seed construction | Genesio--Tesi frequency-domain harmonic-balance approach; Tenreiro Machado fractional describing-function family; local Weyl-to-Caputo warning in the report |
| `hidden_attractors.seed_generation.reconstruct_biased_lure_seed` | Biased Lur'e seed reconstruction from DC and first harmonic equations | Local biased describing-function contract documented in the Chua report |
| `hidden_attractors.solvers.FractionalHistory` | EFORK-compatible finite memory window container | Local finite-memory EFORK contract; heavy integration is delegated to C backends |
| `hidden_attractors.solvers.efork3_caputo_integrate` | Three-stage Caputo EFORK reference integrator used for manufactured-solution reproduction | F. Ghoreishi, R. Ghaffari, and N. Saad, "Fractional Order Runge-Kutta Methods," Tables 3, 4, 9, and 10; provided `ejemplo1.py` implementation archived in validation evidence |
| `tools/validation/validate_chua_fractional_nonsmooth_algebra.py` | Generate algebra and Lur'e/DF evidence for `q=0.9998`, with MATLAB/report sign normalization | Danca (2017) for the exact case; local MATLAB/Wolfram reproductions; Petras (2008) as model-family support only |

## Trajectory Diagnostics

| Code | Purpose | Reference source |
|---|---|---|
| `hidden_attractors.analysis.trajectory.component_fft` | Dominant FFT frequency and spectral entropy proxy | Standard spectral analysis; used as a diagnostic only, not as hiddenness proof |
| `hidden_attractors.analysis.trajectory.section_points` | Poincare-style section points for trajectory-cloud comparison | Operational section geometry documented in `reporte_unificado_chua_fraccionario.tex`; not a proof of hiddenness |
| `hidden_attractors.analysis.trajectory.cloud_median_distance` | Symmetric nearest-neighbor cloud distance | Local geometric comparison contract documented in the report |
| `hidden_attractors.analysis.trajectory.trajectory_metrics` | Boundedness, range, variance, spectral, section, and cloud diagnostics | Local robustness contract; hiddenness reading follows Leonov--Kuznetsov hidden/self-excited classification |

## Bifurcation And Plots

| Code | Purpose | Reference source |
|---|---|---|
| `hidden_attractors.analysis.bifurcation.bifurcation_points_from_trajectories` | Extract maxima, minima, or samples from parameter scans | Post-processing convention for bifurcation diagrams; continuation should be delegated to tools such as PyDSTool when needed |
| `hidden_attractors.plotting.dynamics.plot_phase_space` | Phase-space view of `t,x,y,z` trajectories | Standard dynamical-systems visualization |
| `hidden_attractors.plotting.dynamics.plot_phase_projections` | `xy`, `xz`, `yz` projections | Standard dynamical-systems visualization |
| `hidden_attractors.plotting.dynamics.plot_time_series` | Time series for selected observables | Standard numerical diagnostics |
| `hidden_attractors.plotting.dynamics.plot_bifurcation_diagram` | Scatter plot of extracted bifurcation points | Post-processing visualization, not numerical continuation |
| `hidden_attractors.plotting.dynamics.plot_lure_transfer_components` | Real and imaginary transfer-function closure panels for a selected Lur'e seed | Integer Chua `q=1` MATLAB verification view and Guan--Xie Example 6 branch |

## Native And Workflow Contracts

| Code | Purpose | Reference source |
|---|---|---|
| `hidden_attractors.native.FractionalChuaBackend` | Wrapper for C/EFORK fractional Chua integration | Caputo fractional model; stage ordering aligned with Ghoreishi--Ghaffari--Saad; full fractional external validation remains separate from the verified `q=1` Chua rerun |
| `hidden_attractors.native.BasinBackend` | Wrapper for basin classification backend | Leonov--Kuznetsov hidden/self-excited classification plus local finite-time basin contract |
| `hidden_attractors.workflows.robustness_overlay` | Overlay trajectories under changes of `h`, `Lm`, and `t_final` | Local robustness contract; robustness does not imply hiddenness |
| `hidden_attractors.workflows.sphere_controls` | Compatibility adapter now sampling inside equilibrium-centred balls | Leonov--Kuznetsov basin criterion; finite-sample numerical control |
| `hidden_attractors.workflows.refined_basin` | Refine unresolved basin cells by trajectory geometry | Local target-reference geometry contract |
| `hidden_attractors.workflows.strict_target_refinement` | Stricter target-reference refinement for unresolved Chua/Danca basin or equilibrium-ball rows | Local finite-time trajectory-similarity contract with negative controls; still numerical evidence, not proof |
| `hidden_attractors.workflows.danca_abm_sphere_controls` | Danca-located reference accreditation with untruncated ABM, followed by ABM/EFORK3 full/truncated-memory ball controls and diagnostics | Danca fractional Chua example plus Diethelm--Ford--Freed ABM predictor-corrector; solver/memory comparison does not replace the official hiddenness protocol |
| `hidden_attractors.protocol_cli` | Canonical stage-envelope CLI for the single official methodology | Local protocol contract; numerical payloads may use corrected C backends |
| `hidden_attractors.workflows.integer_lure` | Reusable order-one Lur'e seed, continuation, final-attractor, and hiddenness controls | Integer Chua `q=1` reference report; Guan--Xie Example 6 displayed-value comparison; locally regenerated evidence package |
| `hidden_attractors.workflows.specs.WorkflowInputSpec` | Shared input contract for reusable CLIs and migrated adapters | Local reproducibility contract: records solver, classifier, target, basin, ball-sampling, and refinement assumptions |
| `hidden_attractors.systems.requirements` | Capability and requirement checklist for applying workflows to new systems | Local library-extension policy; distinguishes vector-field registration from hiddenness evidence workflows |

## Optional External Methods

| Code | Purpose | Reference source |
|---|---|---|
| `hidden_attractors.integrations.compute_complexity_measures(..., backend="nolds")` | Delegate entropy, dimension, Lyapunov, Hurst, and DFA metrics to `nolds` | External `nolds` package; Lyapunov-spectrum methods should cite Benettin et al., "Lyapunov Characteristic Exponents for Smooth Dynamical Systems and for Hamiltonian Systems" |
| `hidden_attractors.integrations.compute_complexity_measures(..., backend="antropy")` | Delegate entropy and fractal measures to `antropy` | External `antropy` package; do not copy implementations into this repo |
| `hidden_attractors.integrations.external_tool_report` | Registry of optional tools | PyDSTool documentation; pyComplexity notebook reference; local adapter policy |

## Internal Transition Dependencies

| Code | Purpose | Reference source |
|---|---|---|
| `tools/legacy/danca2017_chua_abm_replication.py` | Temporary ABM numerical dependency used by the maintained robustness comparison | M. F. Danca, "Hidden Chaotic Attractors in Fractional-Order Systems"; K. Diethelm, N. J. Ford, and A. D. Freed, "A Predictor-Corrector Approach for the Numerical Solution of Fractional Differential Equations" |
| `tools/legacy/chua_initial_cond.py` and DF helpers | Temporary calculation dependencies used by `fractional_report_run`; no public route | Genesio--Tesi seed mechanism; it must not be read as proof of hiddenness |
| `tools/legacy/early_periodicity_filter.py` | Migration adapter tested to prevent pre-continuation periodic rejection | Official `soft_precheck` rule |

## Policy For New Calculation Code

When adding a new calculation module:

1. Add the article title and source in the module docstring or in this table.
2. State whether the method is exact theory, finite-time numerical evidence, or
   an operational diagnostic.
3. If a maintained external package already implements the algorithm, add an
   adapter and citation instead of copying the algorithm.
4. If the method integrates many trajectories, classifies basins, estimates
   Lyapunov exponents, or runs bifurcation sweeps, use or add a C backend rather
   than a heavy Python implementation.
5. Add a small example that writes reproducible figures or CSV/JSON artifacts.
