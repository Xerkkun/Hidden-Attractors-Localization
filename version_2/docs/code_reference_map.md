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

| Public interface | Result | Evidence boundary |
| --- | --- | --- |
| `hidden_attractors.find_integer_lure_harmonic_seed_direct` | Integer Nyquist/DF seed recomputed from the declared rational Lur'e transfer | Primary integer route; stored Mathematica/MATLAB values are regression references, not inputs. |
| `hidden_attractors.find_lure_harmonic_seed` | Describing-function seed for a compatible registered Lur'e system | Initialization calculation; a seed is not hiddenness evidence. |
| `hidden_attractors.integer_lure_seed` and `continue_integer_lure_seed` | Direct integer reference seed and continuation trace | Direct transfer is the default; dense scan fallback must be requested explicitly. |
| `hidden_attractors.integrate_integer_lure` and `final_integer_lure_attractor` | Finite numerical trajectory for the selected continued seed | Solver output under explicit numerical settings. |
| `hidden_attractors.run_integer_lure_hiddenness_controls` | Finite equilibrium-neighborhood controls with conservative labels | Numerical control record; it does not turn finite sampling into a global basin proof. |
| `hidden_attractors.validate_full_workflow_system` | Preflight validation of model capabilities required by a workflow | Software-contract validation only. |

## Scientific References

The implemented equations and validation boundaries cite the following source
titles:

- “Chaos in Chua's Circuit” for the Chua model family;
- “Hidden Chaotic Attractors in Fractional-Order Systems” as a bibliographic
  basis for the implemented fractional reference model, without claiming a
  complete independent reproduction;
- “Stability Results for Fractional Differential Equations with Applications to Control Processing” for the local fractional stability criterion;
- “Lyapunov Characteristic Exponents for Smooth Dynamical Systems and for Hamiltonian Systems” for the QR/orthonormalization foundation;
- Rosenstein et al. and Eckmann et al. for scalar time-series reconstruction.

## Evidence Boundary

The library supports hidden-attractor localization and independent dynamical
characterization as separate uses. Every result must be interpreted according
to its returned method, numerical settings, validation status, sampling, and
finite observation window. This map is limited to current exported interfaces
and recorded validation states.
