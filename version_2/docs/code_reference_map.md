# Public Calculation Reference Map

This page maps the supported public calculation interfaces to their scientific
basis and current validation status. It contains no case-development scripts
or editorial infrastructure.

## Independent Dynamical Characterization

These exported functions can be used without running a hidden-attractor
search.

| Public interface | Result | Evidence boundary |
| --- | --- | --- |
| `hidden_attractors.compute_trajectory_metrics` and `trajectory_metrics` | Boundedness, ranges, variance, spectral summaries, section counts, and finite trajectory diagnostics | Tested finite-trajectory characterization; not a proof of chaos or hiddenness. |
| `hidden_attractors.compute_boundedness_metrics` | Finite-window boundedness status and supporting statistics | Numerical diagnostic under the supplied observation window. |
| `hidden_attractors.compute_fft_psd` | FFT-derived PSD arrays and spectral summaries | Standard sampled-signal spectral analysis. |
| `hidden_attractors.detect_poincare_crossings` | Direction-aware Poincare crossings and crossing metadata | Finite geometric sampling of a trajectory. |
| `hidden_attractors.zero_one_test` | 0--1 chaos-test statistic | Finite-series diagnostic; it does not certify chaos or hiddenness by itself. |
| `hidden_attractors.bifurcation_points_from_trajectories` | Maxima, minima, or samples from parameter sweeps | Post-processing for bifurcation diagrams, not a continuation proof. |
| `hidden_attractors.estimate_time_series_lyapunov` | Rosenstein largest-exponent estimate, Eckmann reconstructed spectrum, and Kaplan--Yorke dimension | Fully integrated and tested scalar-series route through the optional `nolds` backend. Results are finite-data, sampling-dependent estimates with parameters, provenance, fit diagnostics, and warnings. |
| `hidden_attractors.kaplan_yorke_dimension` | Kaplan--Yorke dimension from an ordered finite spectrum | Algebraic calculation whose interpretation inherits the evidence limits of the supplied spectrum. |
| `hidden_attractors.compute_lyapunov_spectrum` | State-space Lyapunov calculation selected from the implemented method registry | The returned summary identifies the method and its validation status; the dispatcher does not imply equal evidence for every method. |
| `hidden_attractors.TrajectoryInput`, `PrehistorySpec`, and `AnalysisResult` | Immutable common input/result envelope with explicit sampling, time coordinate, projection, memory, prehistory, solver provenance, and fingerprint | Makes integer/fractional provenance comparable without treating a sampled fractional state vector as the complete hereditary state. |
| `hidden_attractors.ordinal_pattern_distribution` and `permutation_entropy` | Dense Bandt--Pompe ordinal histogram plus finite plug-in Shannon entropy, with declared delay and tie policy | Python/Numba/C parity and a finite Wolfram oracle validate implementation consistency; no entropy-rate, chaos, attraction, or hiddenness claim. |
| `hidden_attractors.correlation_sum_curve`, `fit_correlation_dimension`, and `estimate_correlation_dimension` | Exact finite q=2 pair counts after Theiler exclusion and an OLS slope on a caller-declared radius range | Python/Numba/C parity and a finite Wolfram oracle validate implementation consistency; no automatic scaling-region, fractal-dimension, chaos, or hiddenness claim. |
| `hidden_attractors.smaller_alignment_index`, `generalized_alignment_index`, `linear_dependence_index`, and the three integer alignment facades | Instantaneous and propagated SALI/GALI/LDI for ordinary flows and maps, with variational and multiparticle routes | NumPy/Numba parity plus exact Wolfram map/flow fixtures validate finite integer tangent algebra. Every facade rejects `q != 1`; no automatic chaos, attraction, hiddenness, or fractional-memory claim. |
| `hidden_attractors.integer_covariant_vectors_from_qr_history`, the flow/map/system CLV facades, and `covariant_lyapunov_angles` | Finite-time Ginelli CLV histories and pair/principal-subspace angles for memoryless integer systems | Exact nonnormal map/flow fixtures, NumPy/Numba parity, projective covariance checks, and an independent Wolfram comparison validate the implemented finite `q=1` algebra. They do not prove nonlinear convergence, hyperbolicity, chaos, attraction, hiddenness, or any fractional CLV formulation. |
| `hidden_attractors.fractional.integrate_multi_term_caputo_l1` | Finite positive-coefficient sum of Caputo derivatives with exact duplicate-order canonization | Semantic facade over the combined distributed-order L1 kernel; coefficients are not normalized and the finite Wolfram comparison is not a general stability or convergence proof. |
| `hidden_attractors.fractional.tempered_convolution_quadrature` | Sampled BDF1/BDF2 tempered RL or conjugated-Caputo operator with componentwise orders/tempering | Python/Numba/FFT parity, exact reductions, manufactured refinement, and an independent Wolfram oracle validate finite implementation consistency. It is not an FDE solver, fast-history method, stability theorem, or dynamics classification. |
| `hidden_attractors.fractional.tempered_fast_multistep_history` | Sampled FBDF1/GNGF2 tempered RL or conjugated-Caputo operator with an exact local window and real recurrent old history | Python/Numba/direct parity, complete finite-grid compression calibration, exact `q=1` reductions, and an independent Wolfram oracle validate finite consistency. The reported bound controls compression only; this is not fractional BDF2, an FDE solver, or a dynamics classification. |

The Lyapunov registry contains four callable methods. Their recorded validation
states are:

| Method identifier | Current validation state |
| --- | --- |
| `integer_qr_benettin` | Exact linear controls and internal cross-checks; restricted to `q=1`; no quantitative published-spectrum reproduction. |
| `fractional_variational_abm_qr` | Synthetic numerical validation only. |
| `fractional_cloned_dynamics_abm_gs_published` | Implemented diagnostic with a recorded published-benchmark discrepancy. |
| `fractional_cloned_dynamics_abm_qr` | Implemented numerical-comparison route; no published quantitative-validation claim. |

All four methods return finite-time indicators. No current registry entry
claims complete quantitative published-benchmark validation.

## Models, Systems, And Stability

| Public interface | Result | Evidence boundary |
| --- | --- | --- |
| `hidden_attractors.ChuaParameters`, `rhs_nonsmooth`, and `equilibria_nonsmooth` | Parameterized non-smooth Chua vector field and equilibria | Implemented model equations with algebraic and software validation. |
| `hidden_attractors.jacobian_nonsmooth` | Regional Jacobian with a switching-surface guard | Local linearization; fractional stability interpretation follows Matignon's criterion. |
| `hidden_attractors.register_system`, `get_system`, and `list_systems` | Registration and retrieval of dynamical-system definitions | Stable model-registry API. Registration alone does not establish any dynamical classification. |
| `hidden_attractors.check_system_capability` and `requirements_for` | Capability checks for a selected calculation or workflow | Separates general characterization requirements from the additional evidence required for hidden-attractor localization. |
| `hidden_attractors.load_trajectory_csv` | Portable loading of sampled trajectory data | Stable input route for independent trajectory and time-series analysis. |

## Hidden-Attractor Localization

The public localization route is built from exported generic Lur'e seed
functions and the completed executable integer reference workflow.

Names written as `hidden_attractors.<name>` below, with no intervening module,
are members of the tiered top-level surface. Longer module-qualified names are
narrower experimental or system-specific helpers; their availability does not
promote them into `PUBLIC_API_STABLE` or `PUBLIC_API_EXPERIMENTAL`.

| Public interface | Result | Evidence boundary |
| --- | --- | --- |
| `hidden_attractors.seed_generation.find_integer_lure_harmonic_seed_direct` | Integer Nyquist/DF seed recomputed from the declared rational Lur'e transfer | Module-qualified implementation beneath the primary integer route; stored Mathematica/MATLAB values are regression references, not inputs. |
| `hidden_attractors.find_lure_harmonic_seed` | Describing-function seed for a compatible registered Lur'e system | Initialization calculation; a seed is not hiddenness evidence. |
| `hidden_attractors.integer_lure_seed` and `continue_integer_lure_seed` | Direct integer reference seed and continuation trace | Top-level experimental route; `integer_lure_seed` has no frequency-grid or scan-fallback argument. |
| `hidden_attractors.workflows.find_sign_switching_cycle_seed` and `continue_integer_lure_nonlinearity` | Switching-map seed and explicit source-to-target nonlinearity continuation | Alternative theoretical route; it is not recorded as direct DF success and does not use a frequency grid. |
| `hidden_attractors.integrate_integer_lure` and `final_integer_lure_attractor` | Finite numerical trajectory for the selected continued seed | Solver output under explicit numerical settings. |
| `hidden_attractors.run_integer_lure_hiddenness_controls` | Finite equilibrium-neighborhood controls with conservative labels | Numerical control record; it does not turn finite sampling into a global basin proof. |
| `hidden_attractors.solvers.dop853_q1_integrate` | Adaptive integer-order trajectory with explicit tolerances and output sampling | Memoryless `q=1` solver; not a Caputo method. |
| `hidden_attractors.analysis.lyapunov_adaptive.integer_system_dop853_variational_qr` | Adaptive variational QR spectrum for a registered integer system | Module-qualified finite-time estimate; requires convergence and independent controls. |
| `hidden_attractors.workflows.continue_integer_parameter_path` | State continuation through complete parameter dictionaries | Reconstructs the system and Lur'e declaration at every node; it is an explicit alternative, not a silent scan. |
| `hidden_attractors.verification.calibrate_attractor_reference` and `hidden_attractors.workflows.run_integer_hidden_chaos_controls` | Calibrated chaotic reference plus equilibrium-neighborhood probes | Supports only `hidden_under_tested_neighborhoods`; no global basin proof. |
| `hidden_attractors.verification.candidate_gate.evaluate_candidate_gate` | Joint boundedness, dynamics, robustness, and sampled-hiddenness decision | Combines existing evidence without integrating a trajectory or proving global hiddenness. |
| `hidden_attractors.validate_full_workflow_system` | Preflight validation of model capabilities required by a workflow | Software-contract validation only. |

## MAVPD Repository Route

The case executable is
`examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/run_example.py`.
It is a reproducible validation/search script, not an installed top-level API.
Its current dependency and integrator map is:

| Stage | Main functions | Numerical contract |
| --- | --- | --- |
| Source equations, Hopf algebra, and direct branches | `mavpd_2023_system`, `mavpd_hopf_gamma_boundaries`, `integer_lure_seed` | Algebraic construction and polynomial roots; Nyquist samples are display-only. |
| `lambda` continuation from each direct seed | `continue_integer_lure_seed` | Fixed-step integer `efork_q1_integrate`. |
| Finite chaos trigger at the base parameters | `integer_system_dop853_variational_qr` | Adaptive DOP853 variational QR; it records only that the declared finite threshold was not exceeded. |
| Structured continuation in `xi` and `gamma` | `continue_integer_parameter_path` | Adaptive DOP853, with a freshly built system and Lur'e object at every node. |
| Declared Hopf-offset screen | `integer_system_dop853_variational_qr`, `calibrate_attractor_reference`, `run_integer_hidden_chaos_controls` | DOP853 exponent, cloud, and finite `E0` probes; no blind grid or transition-boundary claim. |
| Strict candidate and all-equilibrium controls | `dop853_q1_integrate`, `run_integer_hidden_chaos_controls` | Adaptive DOP853 under the stored tolerances, horizons, radii, and directions. |
| Cross-integrator and joint decision | `efork_q1_integrate`, `evaluate_candidate_gate` | Fixed-step EFORK cloud control followed by a non-integrating evidence gate. |

The maintained local point is distinct from the published parameter tuples.
Its strongest recorded status is
`chaotic_hidden_under_tested_neighborhoods`, which remains finite numerical
evidence. The existing Julia comparison concerns the periodic `gamma=0.1`
MAVPD audit and cannot be cited as reproduction of this chaotic point.

## Scientific References

The implemented equations and validation boundaries cite the following source
titles:

- “Chaos in Chua's Circuit” for the Chua model family;
- “Hidden Chaotic Attractors in Fractional-Order Systems” as a bibliographic
  basis for the implemented fractional reference model, without claiming a
  complete independent reproduction;
- “Stability Results for Fractional Differential Equations with Applications to Control Processing” for the local fractional stability criterion;
- “Lyapunov Characteristic Exponents for Smooth Dynamical Systems and for Hamiltonian Systems” for the QR/orthonormalization foundation;
- Skokos (2001), Skokos et al. (2004), and Skokos--Bountis--Antonopoulos
  (2007) for SALI/GALI definitions and asymptotic laws;
- Manda--Hillebrand--Skokos (2025) for the multiparticle implementation
  controls, Rolim Sales--Leonel--Antonopoulos (2026) for the SVD/LDI
  formulation, and Ma--Long--Zhu (2016) for dissipative-system cautions;
- Ginelli et al. (2007), Kuptsov--Parlitz (2012), Froyland et al. (2013),
  and Ginelli et al. (2013) for covariant Lyapunov vectors and their numerical
  comparison; Noethen (2021) for Hilbert-space convergence and spectral-gap
  analysis; and du Plessis--Hillebrand--Skokos (2026) for transient-monitoring
  and central-subspace cautions;
- “Existence of Self-Excited and Hidden Attractors in the Modified Autonomous Van Der Pol-Duffing Systems” for the published MAVPD equation family, not for the locally selected chaotic parameter point;
- Rosenstein et al. and Eckmann et al. for scalar time-series reconstruction.

## Evidence Boundary

The library supports hidden-attractor localization and independent dynamical
characterization as separate uses. Every result must be interpreted according
to its returned method, numerical settings, validation status, sampling, and
finite observation window. This map is limited to current exported interfaces
and recorded validation states.
