# API Reference

This file is the release inventory for the active Python library under `version_2/hidden_attractors`. It is generated from Python AST parsing, so it lists symbols defined in source files without importing modules or compiling native backends.

The inventory intentionally includes public symbols, classes, methods, and private helpers. Private names are documented for auditability only; they are not stable API unless they are exported through `hidden_attractors.__all__` or documented in the public CLI/manual.

## Release API Summary

- Modules with defined symbols: `137`
- Top-level functions: `723`
- Classes: `95`
- Class methods: `104`

## Stable Entry Points

- Public CLI: `hidden-attractors` (`hidden_attractors.cli.main:main`)
- New-user run path: `hidden-attractors run -p chua_integer`, `hidden-attractors run -p chua_fractional`, or `hidden-attractors run -c path/to/config.yaml`
- New-system extension path: `hidden_attractors.systems.ChaoticSystem`, `register_system`, `WorkflowInputSpec`, and explicit Lur'e data when DF/Nyquist workflows are used

## Top-Level Exported Names

- `EXPERIMENTAL`, `INTERNAL`, `LEGACY`, `STABLE`
- `api_tier`, `assert_tier`, `get_tier`, `ChuaParameters`
- `chua_parameters`, `chua_arctan_wu2023_parameters`, `chua_nonsmooth_parameters`, `equilibria_arctan`
- `equilibria_nonsmooth`, `jacobian_arctan`, `jacobian_nonsmooth`, `rhs_arctan`
- `rhs_nonsmooth`, `ChaoticSystem`, `LureSystem`, `check_system_capability`
- `get_system`, `known_workflows`, `list_systems`, `register_system`
- `requirements_for`, `CLASS_LABELS`, `TARGET_CLASS_IDS`, `class_label`
- `is_target_class`, `CandidateRecord`, `load_final_candidate_records`, `load_trajectory_csv`
- `LyapunovResult`, `RobustnessCase`, `integer_system_lyapunov_exponents`, `trajectory_metrics`
- `trajectory_metrics_for_system`, `HarmonicSeed`, `find_harmonic_seed`, `find_lure_harmonic_seed`
- `find_lure_omega_gain_candidates`, `find_omega_gain_candidates`, `validate_fractional_order`, `BasinSliceSpec`
- `DestinationClassifierSpec`, `FullWorkflowContract`, `ContinuationPlan`, `ContinuationTrace`
- `DynamicReference`, `FINAL_LABELS`, `HiddennessTestResult`, `IntegratorSpec`
- `NumericalContract`, `OFFICIAL_STAGE_ORDER`, `ParameterSweepSpec`, `RobustnessCaseSpec`
- `RobustnessVerdict`, `PROTOCOL_VERSION`, `PostContinuationDecision`, `SEED_FAMILIES`
- `SoftPrecheckResult`, `SphereControlSpec`, `StageEnvelope`, `StrictRefinementSpec`
- `TargetReferenceSpec`, `TrajectoryDiagnosticsSpec`, `WorkflowInputSpec`, `UnifiedSeedRecord`
- `continue_integer_lure_seed`, `final_integer_lure_attractor`, `integer_lure_seed`, `integrate_integer_lure`
- `run_integer_lure_hiddenness_controls`, `validate_full_workflow_system`, `load_config`, `save_effective_config`
- `run_attractor_only_workflow`, `run_bifurcation_workflow`, `run_basin_workflow`, `run_simple_workflow`

## Public Workflow Families

| Family | Main modules | Use |
| --- | --- | --- |
| System registry | `hidden_attractors.systems`, `hidden_attractors.models` | Define or retrieve vector fields, equilibria, Jacobians, parameters, and Lur'e splits. |
| Seed generation | `hidden_attractors.seed_generation`, `hidden_attractors.lure` | Build DF/Nyquist/Lur'e seeds; seed output is heuristic evidence only. |
| Integration | `hidden_attractors.integrations`, `hidden_attractors.solvers`, `hidden_attractors.native` | Run integer and Caputo trajectories under explicit solver and memory contracts. |
| Continuation and workflows | `hidden_attractors.continuation`, `hidden_attractors.workflows` | Transport seeds, run staged protocols, robustness, hiddenness, basins, reporting, and examples. |
| Verification | `hidden_attractors.verification`, `hidden_attractors.basins` | Classify equilibria, probe neighborhoods, label contacts, and audit hiddenness claims. |
| Diagnostics | `hidden_attractors.analysis`, `hidden_attractors.diagnostics` | Compute finite-time spectra, zero-one, Lyapunov, Poincare, boundedness, and method comparisons; diagnostics do not certify hiddenness. |
| Plotting and IO | `hidden_attractors.plotting`, `hidden_attractors.io` | Export canonical figures and machine-readable CSV/JSON metadata. |

## Complete Defined Symbols

### `hidden_attractors._stability`

Source: `version_2/hidden_attractors/_stability.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `api_tier` | 64 | `public/module` | `api_tier(tier: str) -> Callable[[_F], _F]` | Stamp ``__api_tier__`` on a callable or class. |
| `function` | `get_tier` | 106 | `public/module` | `get_tier(obj: Any) -> str \| None` | Return the tier string stamped on *obj*, or ``None`` if absent. |
| `function` | `assert_tier` | 134 | `public/module` | `assert_tier(obj: Any, expected: str) -> None` | Raise :exc:`AssertionError` if *obj* does not carry the expected tier. |

### `hidden_attractors.analysis.bifurcation`

Source: `version_2/hidden_attractors/analysis/bifurcation.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `BifurcationPoint` | 26 | `public/module` | `class BifurcationPoint(object)` | One observable value associated with one parameter value. |
| `method` | `BifurcationPoint.as_dict` | 50 | `public/module` | `as_dict(self) -> dict[str, float \| int \| str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `observable_column` | 60 | `public/module` | `observable_column(observable: str \| int) -> int` | Return the trajectory column index for an observable label. |
| `function` | `trajectory_tail` | 99 | `public/module` | `trajectory_tail(traj: np.ndarray, *, t_start: float \| None=None) -> np.ndarray` | Return the post-transient rows of a trajectory. |
| `function` | `local_extrema` | 128 | `public/module` | `local_extrema(values: Sequence[float], *, mode: str='maxima') -> np.ndarray` | Return indices of local extrema in a one-dimensional series. |
| `function` | `_scan_item` | 171 | `internal` | `_scan_item(item: Mapping[str, Any] \| tuple[float, np.ndarray], parameter_key: str) -> tuple[float, np.ndarray]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `bifurcation_points_from_trajectories` | 182 | `public/module` | `bifurcation_points_from_trajectories(scans: Iterable[Mapping[str, Any] \| tuple[float, np.ndarray]], *, parameter_key: str='parameter', observable: str \| int='x', t_start: float \| None=None, mode: str='maxima', max_points_per_parameter: int=250) -> list[BifurcationPoint]` | Extract bifurcation-diagram points from a parameter scan. |
| `function` | `bifurcation_summary` | 262 | `public/module` | `bifurcation_summary(points: Sequence[BifurcationPoint]) -> dict[str, float \| int]` | Return summary statistics for a set of bifurcation points. |

### `hidden_attractors.analysis.boundedness`

Source: `version_2/hidden_attractors/analysis/boundedness.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_coordinate_payload` | 23 | `internal` | `_coordinate_payload(values: np.ndarray) -> list[float]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `compute_boundedness_metrics` | 27 | `public/module` | `compute_boundedness_metrics(times: Sequence[float], trajectory: Sequence[Sequence[float]], burn_time: float, norm: str='euclidean', divergence_radius: float \| None=None, growth_window_fraction: float=0.2) -> dict[str, Any]` | Compute conservative post-transient boundedness metrics. |

### `hidden_attractors.analysis.integrated_chaos_validator`

Source: `version_2/hidden_attractors/analysis/integrated_chaos_validator.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `method_is_applicable` | 36 | `public/module` | `method_is_applicable(method: LyapunovMethodInfo, q: float) -> bool` | Return whether a registered method may be compared at ``q``. |
| `function` | `method_registry_rows` | 53 | `public/module` | `method_registry_rows(registry: Mapping[str, LyapunovMethodInfo]=LYAPUNOV_METHODS) -> list[dict[str, Any]]` | Return JSON-ready metadata for the implemented comparison methods. |
| `function` | `classify_lambda_max` | 68 | `public/module` | `classify_lambda_max(value: float \| None, *, near_zero: float=0.02) -> str` | Classify one finite-time largest exponent without promoting a proof. |
| `function` | `normalize_lyapunov_case_evidence` | 80 | `public/module` | `normalize_lyapunov_case_evidence(*, case_id: str, q: float, f4_integer_rows: list[dict[str, Any]] \| None=None, registry: Mapping[str, LyapunovMethodInfo]=LYAPUNOV_METHODS) -> list[dict[str, Any]]` | Build per-case method evidence. |
| `function` | `integrate_case_evidence` | 142 | `public/module` | `integrate_case_evidence(*, case_id: str, boundedness_status: str \| None, zero_one_status: str \| None, psd_fft_status: str \| None, poincare_status: str \| None, lyapunov_evidence: list[dict[str, Any]], f4_status: str) -> dict[str, Any]` | Apply explicit conservative F6 rules to one case. |
| `function` | `_sign_pattern` | 231 | `internal` | `_sign_pattern(spectrum: list[float] \| None) -> list[str] \| None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_decision` | 237 | `internal` | `_decision(case_id: str, status: str, reason: str) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |

### `hidden_attractors.analysis.lyapunov`

Source: `version_2/hidden_attractors/analysis/lyapunov.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `LyapunovResult` | 80 | `public/module` | `class LyapunovResult(object)` | Finite-time Lyapunov exponent estimate. |
| `function` | `finite_difference_jacobian` | 140 | `public/module` | `finite_difference_jacobian(rhs: Callable[[np.ndarray], np.ndarray], state: np.ndarray, *, eps: float=1e-06) -> np.ndarray` | Estimate the Jacobian of *rhs* by central finite differences. |
| `function` | `integer_lyapunov_exponents` | 186 | `public/module` | `integer_lyapunov_exponents(rhs: Callable[[np.ndarray], np.ndarray], jacobian: Callable[[np.ndarray], np.ndarray] \| None, x0: np.ndarray, *, h: float, t_final: float, t_burn: float=0.0, reorthonormalize_every: int=10, jacobian_eps: float=1e-06, div_threshold: float \| None=None, q: float=1.0) -> LyapunovResult` | Estimate integer-order Lyapunov exponents by QR reorthonormalisation. |
| `function` | `integer_qr_benettin_lyapunov_exponents` | 372 | `public/module` | `integer_qr_benettin_lyapunov_exponents(rhs: Callable[[np.ndarray], np.ndarray], jacobian: Callable[[np.ndarray], np.ndarray] \| None, x0: np.ndarray, *, h: float, t_final: float, t_burn: float=0.0, reorthonormalize_every: int=10, jacobian_eps: float=1e-06, div_threshold: float \| None=None, q: float=1.0) -> LyapunovResult` | Canonical F0 entry point for integer-order QR-Benettin Lyapunov exponents. |
| `function` | `_infer_system_order` | 471 | `internal` | `_infer_system_order(system: object) -> float \| None` | Attempt to infer the fractional order *q* from a system object. |
| `function` | `integer_system_lyapunov_exponents` | 522 | `public/module` | `integer_system_lyapunov_exponents(system: ChaoticSystem, x0: np.ndarray, *, h: float, t_final: float, t_burn: float=0.0, reorthonormalize_every: int=10, jacobian_eps: float=1e-06, div_threshold: float \| None=None) -> LyapunovResult` | Estimate Lyapunov exponents for a registered integer-order system. |

### `hidden_attractors.analysis.lyapunov_api`

Source: `version_2/hidden_attractors/analysis/lyapunov_api.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `LyapunovComputationRequest` | 60 | `public/module` | `class LyapunovComputationRequest(object)` | Structured request for a Lyapunov spectrum computation. |
| `class` | `LyapunovComputationSummary` | 132 | `public/module` | `class LyapunovComputationSummary(object)` | Result of a ``compute_lyapunov_spectrum`` call. |
| `function` | `validate_lyapunov_method_request` | 170 | `public/module` | `validate_lyapunov_method_request(request: LyapunovComputationRequest) -> tuple[bool, str, tuple[str, ...]]` | Validate a :class:`LyapunovComputationRequest`. |
| `function` | `compute_lyapunov_spectrum` | 353 | `public/module` | `compute_lyapunov_spectrum(*, system: object \| None=None, rhs: Callable[[np.ndarray], np.ndarray] \| None=None, jacobian: Callable[[np.ndarray], np.ndarray] \| None=None, x0: np.ndarray, q: float, method: str, h: float, t_final: float, t_burn: float=0.0, reorthonormalization_time: float \| None=None, reorthonormalize_every: int \| None=None, jacobian_eps: float=1e-06, div_threshold: float \| None=None, memory_mode: str='not_applicable', memory_window: int \| None=None, **extra: object) -> LyapunovComputationSummary` | Compute the Lyapunov spectrum using a named method. |

### `hidden_attractors.analysis.lyapunov_cloned`

Source: `version_2/hidden_attractors/analysis/lyapunov_cloned.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `ClonedDynamicsResult` | 37 | `public/module` | `class ClonedDynamicsResult(object)` | Finite-time spectrum returned by :func:`compute_cloned_dynamics_spectrum`. |
| `function` | `_modified_gram_schmidt` | 56 | `internal` | `_modified_gram_schmidt(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]` | Return orthonormal columns and residual norms. |
| `function` | `_classical_gram_schmidt` | 72 | `internal` | `_classical_gram_schmidt(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]` | Return classical Gram-Schmidt columns for diagnostic comparisons. |
| `function` | `_orthonormalize` | 89 | `internal` | `_orthonormalize(vectors: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `compute_cloned_dynamics_spectrum` | 102 | `public/module` | `compute_cloned_dynamics_spectrum(rhs: Callable, x0: np.ndarray, orders: float \| list[float] \| tuple[float, ...] \| np.ndarray, h: float, t_clone: float, n_clones: int \| None, k_blocks: int, delta: float, method: str='gs', memory_protocol: str='published_block_restart', system_id: str \| None=None, parameters: Mapping[str, float] \| None=None, return_history: bool=False, random_seed: int \| None=None, divergence_norm: float \| None=None, integration_mode: str='fractional_abm') -> ClonedDynamicsResult` | Estimate a spectrum from perturbed clones without a Jacobian. |

### `hidden_attractors.analysis.lyapunov_fractional`

Source: `version_2/hidden_attractors/analysis/lyapunov_fractional.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `FractionalVariationalQRConfig` | 121 | `public/module` | `class FractionalVariationalQRConfig(object)` | Configuration for :func:`fractional_variational_abm_qr`. |
| `method` | `FractionalVariationalQRConfig.__post_init__` | 174 | `dunder` | `__post_init__(self) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `pack_extended_state` | 203 | `public/module` | `pack_extended_state(X: np.ndarray, Phi: np.ndarray) -> np.ndarray` | Pack state X (shape n) and variational matrix Φ (shape n×n) into a flat vector Y of shape n + n² (row-major). |
| `function` | `unpack_extended_state` | 219 | `public/module` | `unpack_extended_state(Y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]` | Unpack flat vector Y into state X and variational matrix Φ. |
| `function` | `build_extended_variational_rhs` | 242 | `public/module` | `build_extended_variational_rhs(rhs: Callable[[np.ndarray], np.ndarray], jacobian: Callable[[np.ndarray], np.ndarray] \| None, n: int, jacobian_eps: float=1e-06) -> Callable[[np.ndarray], np.ndarray]` | Build ``G(Y) = [F(X), J(X)Φ]`` for the extended variational system. |
| `function` | `apply_history_aware_qr_transform` | 284 | `public/module` | `apply_history_aware_qr_transform(states_history: list[np.ndarray], rhs_history: list[np.ndarray], rhs_ext: Callable[[np.ndarray], np.ndarray], n: int, current_index: int, qr_epsilon: float=1e-300, memory_start_index: int=0) -> tuple[np.ndarray, float, str]` | Apply a history-aware QR transform to the variational block. |
| `function` | `_caputo_abm_extended_stepwise` | 383 | `internal` | `_caputo_abm_extended_stepwise(rhs_ext: Callable[[np.ndarray], np.ndarray], Y0: np.ndarray, q: float, h: float, n_steps: int, n: int, memory_mode: str='full', memory_window: int \| None=None, qr_callback: Callable \| None=None, div_threshold: float \| None=None) -> tuple[list[np.ndarray], list[np.ndarray], list[float], str]` | Caputo ABM step-by-step integrator for the extended variational system. |
| `function` | `fractional_variational_abm_qr` | 530 | `public/module` | `fractional_variational_abm_qr(rhs: Callable[[np.ndarray], np.ndarray], jacobian: Callable[[np.ndarray], np.ndarray] \| None, x0: np.ndarray, *, q: float, h: float, t_final: float, t_burn: float=0.0, reorthonormalization_time: float \| None=None, reorthonormalize_every: int \| None=None, memory_mode: str='full', memory_window: int \| None=None, jacobian_eps: float=1e-06, div_threshold: float \| None=None, history_aware_qr: bool=True, qr_epsilon: float=1e-300) -> LyapunovResult` | Estimate Caputo fractional-order Lyapunov exponents (F2). |

### `hidden_attractors.analysis.lyapunov_methods`

Source: `version_2/hidden_attractors/analysis/lyapunov_methods.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_dk2018_published_validation_status` | 27 | `internal` | `_dk2018_published_validation_status() -> str` | Read DK2018 evidence conservatively; missing or malformed means pending. |
| `class` | `LyapunovMethodInfo` | 58 | `public/module` | `class LyapunovMethodInfo(object)` | Metadata descriptor for a Lyapunov exponent estimation method. |

### `hidden_attractors.analysis.method_comparison`

Source: `version_2/hidden_attractors/analysis/method_comparison.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `classify_method_row` | 21 | `public/module` | `classify_method_row(method: dict[str, Any]) -> str` | Classify one case-method row under the F7 frozen vocabulary. |
| `function` | `compare_lyapunov_methods` | 41 | `public/module` | `compare_lyapunov_methods(rows: list[dict[str, Any]]) -> tuple[str, list[str]]` | Compare case-specific validated or explicitly acceptable spectra. |
| `function` | `compare_f5_diagnostics` | 68 | `public/module` | `compare_f5_diagnostics(*, boundedness: str \| None, zero_one: str \| None, psd_fft: str \| None, poincare: str \| None) -> tuple[str, str]` | Compare F5 indicators without treating boundedness as chaos evidence. |

### `hidden_attractors.analysis.phase_f_closure`

Source: `version_2/hidden_attractors/analysis/phase_f_closure.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `assess_phase_f_closure` | 66 | `public/module` | `assess_phase_f_closure(*, f4_summary: Mapping[str, Any], f5_summary: Mapping[str, Any], dk2018_summary: Mapping[str, Any], fischer2020_summary: Mapping[str, Any], registry: Mapping[str, LyapunovMethodInfo]=LYAPUNOV_METHODS, f6_summary: Mapping[str, Any] \| None=None, f7_summary: Mapping[str, Any] \| None=None, accepted_fractional_policy_exists: bool=False, fischer_resolution_exists: bool=False, diagnostic_scope_statement_exists: bool=False) -> dict[str, Any]` | Evaluate strict and diagnostic Phase F closure routes. |
| `function` | `build_phase_f_closure_matrix` | 255 | `public/module` | `build_phase_f_closure_matrix(summary: Mapping[str, Any]) -> list[dict[str, Any]]` | Create the auditable criterion matrix for the Phase F assessment. |
| `function` | `_method_application` | 335 | `internal` | `_method_application(case: Mapping[str, Any] \| None, method_id: str) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_phase_f_evidence_summary` | 352 | `internal` | `_phase_f_evidence_summary(cases: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]` | Summarize F6 case evidence with the frozen positive vocabulary. |
| `function` | `_lyapunov_support` | 418 | `internal` | `_lyapunov_support(evidence: Mapping[str, Any]) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_evidence_interpretation` | 427 | `internal` | `_evidence_interpretation(level: str) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_all_f5_cases_match` | 435 | `internal` | `_all_f5_cases_match(summary: Mapping[str, Any], key: str, expected: str) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_f5_values` | 440 | `internal` | `_f5_values(summary: Mapping[str, Any], key: str) -> list[Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_criterion` | 447 | `internal` | `_criterion(criterion_id: str, criterion: str, required_for_strict_closure: bool, passed: Any, evidence_file: str, blocker: str, notes: str) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |

### `hidden_attractors.analysis.poincare`

Source: `version_2/hidden_attractors/analysis/poincare.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `PoincareCrossingResult` | 32 | `public/module` | `class PoincareCrossingResult(object)` | Numerical section crossings and conservative metadata. |
| `function` | `_section_index` | 45 | `internal` | `_section_index(section_variable: int \| str, dimension: int) -> int` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_call_rhs` | 58 | `internal` | `_call_rhs(rhs: Callable[..., Sequence[float]], time: float, state: np.ndarray) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `detect_poincare_crossings` | 66 | `public/module` | `detect_poincare_crossings(times: Sequence[float], trajectory: Sequence[Sequence[float]], *, section_variable: int \| str=0, section_value: float=0.0, direction: str='positive', derivative_mode: str='integer_rhs', rhs: Callable[..., Sequence[float]] \| None=None, interpolation: str='linear', min_crossing_separation: float=0.0, burn_time: float \| None=None, max_points: int \| None=None) -> PoincareCrossingResult` | Detect linearly interpolated section crossings. |
| `function` | `_nearest_neighbor_stats` | 206 | `internal` | `_nearest_neighbor_stats(points: np.ndarray) -> dict[str, float \| None]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `summarize_poincare_points` | 221 | `public/module` | `summarize_poincare_points(points: Sequence[Sequence[float]], *, retained_after_burn: int \| None=None, duplicate_tolerance: float=1e-08) -> dict[str, Any]` | Summarize numerical section geometry without certifying dynamics. |
| `function` | `write_poincare_outputs` | 293 | `public/module` | `write_poincare_outputs(output_dir: str \| Path, result: PoincareCrossingResult, *, metadata: dict[str, Any] \| None=None) -> dict[str, str]` | Write standardized CSV and JSON outputs for one crossing result. |

### `hidden_attractors.analysis.spectral`

Source: `version_2/hidden_attractors/analysis/spectral.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `SpectrumResult` | 25 | `public/module` | `class SpectrumResult(object)` | One component amplitude or power spectrum. |
| `function` | `infer_step` | 49 | `public/module` | `infer_step(times: np.ndarray, fallback: float \| None=None) -> float` | Infer a positive sample step from trajectory times. |
| `function` | `fft_spectrum` | 82 | `public/module` | `fft_spectrum(values: np.ndarray, h: float, *, window: str='hann', component: int=0) -> SpectrumResult` | Return the one-sided FFT amplitude spectrum for one trajectory component. |
| `function` | `psd_welch` | 135 | `public/module` | `psd_welch(values: np.ndarray, h: float, *, nperseg: int=512, overlap: float=0.5, component: int=0) -> SpectrumResult` | Return a simple NumPy Welch power spectral density estimate. |
| `function` | `trajectory_component_spectra` | 185 | `public/module` | `trajectory_component_spectra(trajectory: np.ndarray, *, components: Sequence[int] \| None=None, h: float \| None=None, method: str='fft') -> list[SpectrumResult]` | Compute FFT or Welch PSD spectra for state components in a trajectory. |
| `function` | `_window_weights` | 246 | `internal` | `_window_weights(length: int, window: str) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_prominent_peak_indices` | 254 | `internal` | `_prominent_peak_indices(power: np.ndarray) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `compute_fft_psd` | 261 | `public/module` | `compute_fft_psd(times: Sequence[float], signal: Sequence[float], burn_time: float \| None=None, detrend: bool=True, window: str='hann', normalize_power: bool=True, remove_dc: bool=True) -> dict[str, Any]` | Compute one-sided FFT power metrics and a conservative spectral label. |
| `function` | `spectral_diagnostics_multicoordinate` | 356 | `public/module` | `spectral_diagnostics_multicoordinate(times: Sequence[float], trajectory: Sequence[Sequence[float]], burn_time: float, coordinates: Sequence[str]=_COORDINATE_NAMES, **kwargs: Any) -> dict[str, Any]` | Apply :func:`compute_fft_psd` to selected trajectory coordinates. |

### `hidden_attractors.analysis.trajectory`

Source: `version_2/hidden_attractors/analysis/trajectory.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `RobustnessCase` | 26 | `public/module` | `class RobustnessCase(object)` | Numerical contract for one robustness trajectory. |
| `method` | `RobustnessCase.as_dict` | 58 | `public/module` | `as_dict(self, baseline: 'RobustnessCase \| None'=None) -> Dict[str, float \| str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `default_robustness_cases` | 73 | `public/module` | `default_robustness_cases(q: float=0.9998) -> list[RobustnessCase]` | Return the standard six-case h/Lm/time perturbation set. |
| `function` | `component_fft` | 107 | `public/module` | `component_fft(values: np.ndarray, h: float) -> Tuple[float, float]` | Return the dominant FFT frequency and normalised spectral entropy. |
| `function` | `state_view` | 144 | `public/module` | `state_view(traj: np.ndarray) -> np.ndarray` | Extract state columns from a trajectory array. |
| `function` | `min_distance_to_points` | 168 | `public/module` | `min_distance_to_points(state: np.ndarray, points: Iterable[np.ndarray]) -> float` | Return the Euclidean distance from *state* to the nearest point in *points*. |
| `function` | `system_equilibria` | 191 | `public/module` | `system_equilibria(system: ChaoticSystem, parameters: Dict[str, Any] \| None=None) -> dict[str, np.ndarray]` | Return equilibria from a registered system, raising if none are defined. |
| `function` | `classify_trajectory_against_equilibria` | 218 | `public/module` | `classify_trajectory_against_equilibria(traj: np.ndarray, equilibria: Dict[str, np.ndarray], *, divergence_norm: float=120.0, equilibrium_tol: float=0.001, t_start: float \| None=None) -> Dict[str, Any]` | Classify a trajectory's boundedness and final proximity to equilibria. |
| `function` | `trajectory_metrics_for_system` | 277 | `public/module` | `trajectory_metrics_for_system(traj: np.ndarray, *, system: ChaoticSystem \| None=None, equilibria: Dict[str, np.ndarray] \| None=None, h: float, t_start: float, divergence_norm: float=120.0, equilibrium_tol: float=0.001) -> Dict[str, Any]` | Compute dimension-agnostic trajectory metrics for a registered system. |
| `function` | `trajectory_ranges` | 348 | `public/module` | `trajectory_ranges(traj: np.ndarray) -> Dict[str, float]` | Compute coordinate ranges for the ``x``, ``y``, ``z`` columns. |
| `function` | `tail_view` | 372 | `public/module` | `tail_view(traj: np.ndarray, *, t_start: float) -> np.ndarray` | Return trajectory rows with ``t >= t_start``. |
| `function` | `sample_rows` | 395 | `public/module` | `sample_rows(arr: np.ndarray, max_points: int) -> np.ndarray` | Subsample rows evenly, preserving the first and last rows. |
| `function` | `section_points` | 418 | `public/module` | `section_points(traj: np.ndarray, *, t_start: float, max_points: int, params: ChuaParameters \| None=None) -> np.ndarray` | Compute upward ``x=0`` Poincaré-section points ``(y, z)``. |
| `function` | `cloud_median_distance` | 472 | `public/module` | `cloud_median_distance(a: np.ndarray, b: np.ndarray) -> float` | Symmetric median nearest-neighbour distance between two point clouds. |
| `function` | `min_equilibrium_distance` | 510 | `public/module` | `min_equilibrium_distance(state: np.ndarray, params: ChuaParameters \| None=None) -> float` | Distance from *state* to the nearest non-smooth Chua equilibrium. |
| `function` | `trajectory_metrics` | 531 | `public/module` | `trajectory_metrics(traj: np.ndarray, *, h: float, t_start: float, divergence_norm: float=120.0, equilibrium_tol: float=0.001, max_section_points: int=300, max_cloud_points: int=1000, reference: Dict[str, Any] \| None=None) -> tuple[Dict[str, Any], Dict[str, Any]]` | Compute geometric and spectral diagnostics for one Chua trajectory. |

### `hidden_attractors.analysis.zero_one`

Source: `version_2/hidden_attractors/analysis/zero_one.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_prepare_signal` | 22 | `internal` | `_prepare_signal(signal: Sequence[float], *, detrend: bool, normalize: bool, max_samples: int \| None) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_classify_k` | 47 | `internal` | `_classify_k(value: float) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_k_for_c` | 55 | `internal` | `_k_for_c(signal: np.ndarray, c_value: float) -> float` | Return correlation statistic K_c using modified mean-square displacement. |
| `function` | `zero_one_test` | 78 | `public/module` | `zero_one_test(signal: Sequence[float], n_c: int=100, c_values: Sequence[float] \| None=None, random_seed: int=12345, detrend: bool=True, normalize: bool=True, max_samples: int \| None=None) -> dict[str, Any]` | Compute a median robust 0-1 statistic over reproducible values of ``c``. |
| `function` | `zero_one_multicoordinate` | 125 | `public/module` | `zero_one_multicoordinate(times: Sequence[float], trajectory: Sequence[Sequence[float]], burn_time: float, coordinates: Sequence[str]=_COORDINATE_NAMES, **kwargs: Any) -> dict[str, Any]` | Apply :func:`zero_one_test` to selected post-transient coordinates. |

### `hidden_attractors.basins.classification`

Source: `version_2/hidden_attractors/basins/classification.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `class_label` | 23 | `public/module` | `class_label(class_id: int) -> str` | Return the stable label for a basin-classification integer. |
| `function` | `is_target_class` | 29 | `public/module` | `is_target_class(class_id: int) -> bool` | Return whether a classifier ID is one of the target-attractor classes. |

### `hidden_attractors.candidates`

Source: `version_2/hidden_attractors/candidates.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_float` | 23 | `internal` | `_float(value: Any, default: float=float('nan')) -> float` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `CandidateRecord` | 31 | `public/module` | `class CandidateRecord(object)` | Numerical-attractor candidate used by verification workflows. |
| `method` | `CandidateRecord.to_dict` | 86 | `public/module` | `to_dict(self) -> Dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_vec` | 104 | `internal` | `_vec(value: Sequence[Any] \| None) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `load_lure_survivor` | 110 | `public/module` | `load_lure_survivor(source_dir: str \| Path, candidate_id: str) -> CandidateRecord` | Load a Lur'e continuation survivor from the final q=0.9998 run. |
| `function` | `load_machado_candidate` | 161 | `public/module` | `load_machado_candidate(candidate_id: str, targeted_path: str \| Path, corrida1_path: str \| Path) -> CandidateRecord` | Load one Machado/FDF candidate from the final targeted verification outputs. |
| `function` | `_selection_path` | 210 | `internal` | `_selection_path(source_dir: str \| Path \| None) -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_record_from_selected` | 217 | `internal` | `_record_from_selected(row: Dict[str, Any], source: Path) -> CandidateRecord` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `load_final_candidate_records` | 235 | `public/module` | `load_final_candidate_records(source_dir: str \| Path \| None=None) -> List[CandidateRecord]` | Return the three candidates promoted from the current validated run. |

### `hidden_attractors.cli.basin`

Source: `version_2/hidden_attractors/cli/basin.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `refined` | 13 | `public/module` | `refined(argv: Sequence[str] \| None=None) -> None` | Run the refined basin workflow. |
| `function` | `strict_target_refinement` | 18 | `public/module` | `strict_target_refinement(argv: Sequence[str] \| None=None) -> None` | Run the strict target refinement workflow for basins. |

### `hidden_attractors.cli.bifurcation`

Source: `version_2/hidden_attractors/cli/bifurcation.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `run_bifurcation` | 21 | `public/module` | `run_bifurcation(argv: Sequence[str] \| None=None) -> None` | Run bifurcation parameter sweep workflow. |
| `function` | `plot_bifurcation` | 58 | `public/module` | `plot_bifurcation(argv: Sequence[str] \| None=None) -> None` | Plot bifurcation diagram from CSV data. |
| `function` | `inspect_bifurcation` | 102 | `public/module` | `inspect_bifurcation(argv: Sequence[str] \| None=None) -> None` | Inspect bifurcation summary JSON. |

### `hidden_attractors.cli.chaos_test`

Source: `version_2/hidden_attractors/cli/chaos_test.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `run_zero_one` | 20 | `public/module` | `run_zero_one(argv: Sequence[str] \| None=None) -> None` | Run 0-1 chaos-test diagnostic on a trajectory or configuration. |
| `function` | `inspect_zero_one` | 120 | `public/module` | `inspect_zero_one(argv: Sequence[str] \| None=None) -> None` | Inspect zero_one summary JSON. |

### `hidden_attractors.cli.continuation`

Source: `version_2/hidden_attractors/cli/continuation.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `load_seeds_from_csv` | 27 | `public/module` | `load_seeds_from_csv(csv_path: Path) -> list[dict[str, Any]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_scalar_continuation` | 50 | `public/module` | `run_scalar_continuation(argv: Sequence[str] \| None=None) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_multiparameter_continuation` | 393 | `public/module` | `run_multiparameter_continuation(argv: Sequence[str] \| None=None) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |

### `hidden_attractors.cli.hiddenness`

Source: `version_2/hidden_attractors/cli/hiddenness.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `sphere_controls` | 13 | `public/module` | `sphere_controls(argv: Sequence[str] \| None=None) -> None` | Run the sphere controls validation workflow. |
| `function` | `strict_target_refinement` | 18 | `public/module` | `strict_target_refinement(argv: Sequence[str] \| None=None) -> None` | Run the strict target refinement workflow. |

### `hidden_attractors.cli.inspect`

Source: `version_2/hidden_attractors/cli/inspect.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `list_candidates` | 17 | `public/module` | `list_candidates() -> None` | Print final candidate records using the public package API. |
| `function` | `systems` | 27 | `public/module` | `systems(argv: Sequence[str] \| None=None) -> None` | List or inspect registered chaotic systems. |
| `function` | `workflow_requirements` | 55 | `public/module` | `workflow_requirements(argv: Sequence[str] \| None=None) -> None` | Print required inputs for reusable workflows and system readiness. |

### `hidden_attractors.cli.lyapunov`

Source: `version_2/hidden_attractors/cli/lyapunov.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `compute_lyapunov` | 21 | `public/module` | `compute_lyapunov(argv: Sequence[str] \| None=None) -> None` | Compute Lyapunov exponents of a system from a configuration or preset. |
| `function` | `trajectory_lyapunov_spectrum` | 58 | `public/module` | `trajectory_lyapunov_spectrum(argv: Sequence[str] \| None=None) -> None` | Estimate trajectory-based Lyapunov exponent spectrum using time-series analysis. |
| `function` | `validate_lyapunov` | 117 | `public/module` | `validate_lyapunov(argv: Sequence[str] \| None=None) -> None` | Validate Lyapunov results JSON summary against mathematical criteria. |

### `hidden_attractors.cli.main`

Source: `version_2/hidden_attractors/cli/main.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `dispatch` | 46 | `public/module` | `dispatch(group: str, cmd: str \| None, argv: Sequence[str]) -> None` | Route the command to the appropriate subcommand logic. |
| `function` | `main` | 163 | `public/module` | `main(argv: Sequence[str] \| None=None) -> None` | Main CLI entry point. |

### `hidden_attractors.cli.protocol`

Source: `version_2/hidden_attractors/cli/protocol.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `run_protocol_stage` | 13 | `public/module` | `run_protocol_stage(stage_cmd: str, argv: Sequence[str] \| None=None) -> None` | Delegate protocol stage subcommand directly to protocol_cli. |

### `hidden_attractors.cli.published`

Source: `version_2/hidden_attractors/cli/published.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `danca_abm_sphere_controls` | 12 | `public/module` | `danca_abm_sphere_controls(argv: Sequence[str] \| None=None) -> None` | Run the published Danca ABM sphere controls workflow. |

### `hidden_attractors.cli.report`

Source: `version_2/hidden_attractors/cli/report.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `fractional_run` | 12 | `public/module` | `fractional_run(argv: Sequence[str] \| None=None) -> None` | Run the fractional report run workflow. |

### `hidden_attractors.cli.robustness`

Source: `version_2/hidden_attractors/cli/robustness.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `overlay` | 12 | `public/module` | `overlay(argv: Sequence[str] \| None=None) -> None` | Run the robustness overlay workflow. |

### `hidden_attractors.cli.run`

Source: `version_2/hidden_attractors/cli/run.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `find_example_config` | 35 | `public/module` | `find_example_config(filename: str) -> Path` | Resolve the template configuration path dynamically, copying to cache if not physically present. |
| `function` | `parse_dynamic_overrides` | 72 | `public/module` | `parse_dynamic_overrides(extra_args: List[str]) -> Dict[str, Any]` | Parse dynamic double-dashed --key=val or --key val arguments as dictionary overrides. |
| `function` | `run_cmd` | 115 | `public/module` | `run_cmd(args: argparse.Namespace, extra_args: List[str]) -> None` | Execute the run subcommand. |
| `function` | `init_cmd` | 194 | `public/module` | `init_cmd(args: argparse.Namespace) -> None` | Execute the init subcommand. |
| `function` | `inspect_config_cmd` | 232 | `public/module` | `inspect_config_cmd(args: argparse.Namespace, extra_args: List[str]) -> None` | Execute the inspect-config subcommand. |
| `function` | `validate_bibliography_cmd` | 274 | `public/module` | `validate_bibliography_cmd(args: argparse.Namespace) -> None` | Execute the validate-bibliography subcommand. |
| `function` | `main` | 326 | `public/module` | `main(argv: Sequence[str] \| None=None) -> None` | Main CLI entry point. |

### `hidden_attractors.cli.seed`

Source: `version_2/hidden_attractors/cli/seed.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `compute_rho_H_for_lure` | 30 | `public/module` | `compute_rho_H_for_lure(system: Any, q: float, omega: float, amplitude: float, sigma0: float, gain: float, K: int=10, n_quad: int=1024) -> tuple[float, dict[str, Any]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `search_biased_seeds` | 65 | `public/module` | `search_biased_seeds(system: Any, q: float, wmin: float, wmax: float, nscan: int, A_min: float, A_max: float, sigma0_min: float, sigma0_max: float, config_path: Path, theta: float=0.0) -> list[dict[str, Any]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_seed_generation` | 186 | `public/module` | `run_seed_generation(centered_or_biased: str, argv: Sequence[str] \| None=None) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `lure_centered` | 471 | `public/module` | `lure_centered(argv: Sequence[str] \| None=None) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `lure_biased` | 474 | `public/module` | `lure_biased(argv: Sequence[str] \| None=None) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |

### `hidden_attractors.cli.validate`

Source: `version_2/hidden_attractors/cli/validate.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `validate_contract` | 90 | `public/module` | `validate_contract(argv: Sequence[str] \| None=None) -> None` | Validate numerical validation evidence contract. |
| `function` | `validate_bibliography` | 95 | `public/module` | `validate_bibliography(argv: Sequence[str] \| None=None) -> None` | Validate claims bibliography manifest against bibliographic registry. |
| `function` | `_repo_root` | 147 | `internal` | `_repo_root() -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_git` | 151 | `internal` | `_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_git_ls_files` | 155 | `internal` | `_git_ls_files(root: Path, *patterns: str) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_git_head` | 162 | `internal` | `_git_head(root: Path, short: bool=False) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_check` | 170 | `internal` | `_check(name: str, category: str, ok: bool, details: list[str] \| None=None) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_load_json` | 174 | `internal` | `_load_json(path: Path) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_paper_bib_entries` | 181 | `internal` | `_paper_bib_entries(path: Path) -> int` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_paper_bib_todos` | 187 | `internal` | `_paper_bib_todos(path: Path) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_mojibake_hits` | 197 | `internal` | `_mojibake_hits(root: Path) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_manifest_path_references` | 217 | `internal` | `_manifest_path_references(root: Path, manifest: dict[str, Any], key: str) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_is_policy_markdown_line` | 228 | `internal` | `_is_policy_markdown_line(lines: list[str], index: int) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_string_path_violation` | 248 | `internal` | `_string_path_violation(value: str, *, allow_validation_outputs: bool) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_json_path_hits` | 258 | `internal` | `_json_path_hits(path: Path, root: Path) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_text_path_hits` | 282 | `internal` | `_text_path_hits(path: Path, root: Path) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_promoted_local_path_hits` | 295 | `internal` | `_promoted_local_path_hits(root: Path) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_remaining_work_file_matches` | 312 | `internal` | `_remaining_work_file_matches(path: Path) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `validate_release_readiness` | 323 | `public/module` | `validate_release_readiness(argv: Sequence[str] \| None=None) -> None` | Validate release repository/software readiness without changing science artifacts. |

### `hidden_attractors.cli`

Source: `version_2/hidden_attractors/cli.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `list_candidates` | 13 | `public/module` | `list_candidates() -> None` | Print final candidate records using the public package API. |
| `function` | `systems` | 24 | `public/module` | `systems(argv: Sequence[str] \| None=None) -> None` | List or inspect registered chaotic systems. |
| `function` | `workflow_requirements` | 35 | `public/module` | `workflow_requirements(argv: Sequence[str] \| None=None) -> None` | Print required inputs for reusable workflows and system readiness. |

### `hidden_attractors.continuation.continuation_fractional`

Source: `version_2/hidden_attractors/continuation/continuation_fractional.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_chua_params_from_system` | 9 | `internal` | `_chua_params_from_system(system: Any)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_can_use_native_chua_efork` | 24 | `internal` | `_can_use_native_chua_efork(system: Any, integrator: str, use_c_backend: bool) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_native_chua_continuation_steps` | 33 | `internal` | `_native_chua_continuation_steps(*, system: Any, seed_x0: np.ndarray, k_gain: float, lambda_values: Sequence[float], q: float, h: float, memory_mode: str, memory_window_length: Optional[int], t_transient: float, t_keep: float) -> List[Dict[str, Any]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_fractional_continuation` | 97 | `public/module` | `run_fractional_continuation(system: Any, seed_x0: np.ndarray, k_gain: float, lambda_values: Sequence[float], h: float, memory_mode: str='full', memory_window_length: Optional[int]=None, div_threshold: float=120.0, integrator: str='abm', use_c_backend: bool=True, t_transient: float=30.0, t_keep: float=30.0, history_times: Optional[np.ndarray]=None, history_states: Optional[np.ndarray]=None, early_stop_config: Optional[Dict]=None, equilibria: Optional[List[np.ndarray]]=None, require_c_backend: bool=True, allow_python_fallback: bool=False, q: Optional[float]=None) -> List[Dict[str, Any]]` | Execute fractional-order parameter continuation for parameter eta (lambda_values). |
| `function` | `run_fractional_continuation_abm_monolithic` | 365 | `public/module` | `run_fractional_continuation_abm_monolithic(system: Any, seed_x0: np.ndarray, k_gain: float, lambda_values: Sequence[float], h: float, memory_mode: str='full', memory_window_length: Optional[int]=None, div_threshold: float=120.0, t_transient: float=30.0, t_keep: float=30.0, history_times: Optional[np.ndarray]=None, history_states: Optional[np.ndarray]=None, early_stop_config: Optional[Dict]=None, equilibria: Optional[List[np.ndarray]]=None, q: Optional[float]=None) -> List[Dict[str, Any]]` | Monolithic Python ABM fractional continuation. |
| `function` | `_make_step_dict` | 643 | `internal` | `_make_step_dict(eta: float, x_in: np.ndarray, x_out: np.ndarray, trajectory: np.ndarray, status: str, used_c: bool, rhs_src: str, n_steps: int, t_end: float, max_norm: float, x_in_norm: float, x_out_norm: float, early_stop_reason: str, history_policy: str='full_caputo', carry_state_history: bool=True, carry_derivative_history: bool=False, eta_boundary_policy: str='right_continuous') -> Dict[str, Any]` | Build a standardised continuation step record. |

### `hidden_attractors.continuation.continuation_integer`

Source: `version_2/hidden_attractors/continuation/continuation_integer.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `run_integer_continuation` | 6 | `public/module` | `run_integer_continuation(system: Any, seed_x0: np.ndarray, k_gain: float, lambda_values: Sequence[float], h: float, t_transient: float=30.0, t_keep: float=30.0, div_threshold: float=120.0, integrator: str='efork_q1', early_stop_config: Optional[Dict]=None, equilibria: Optional[List[np.ndarray]]=None) -> List[Dict[str, Any]]` | Execute integer-order parameter continuation for parameter eta (lambda_values). |
| `function` | `_make_step_dict` | 96 | `internal` | `_make_step_dict(eta: float, x_in: np.ndarray, x_out: np.ndarray, trajectory: np.ndarray, status: str, n_steps: int, t_end: float, max_norm: float, x_in_norm: float, x_out_norm: float, early_stop_reason: str) -> Dict[str, Any]` | Build a standardised integer continuation step record. |

### `hidden_attractors.continuation.memory`

Source: `version_2/hidden_attractors/continuation/memory.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `extract_memory_window` | 4 | `public/module` | `extract_memory_window(times: np.ndarray, states: np.ndarray, h: float, memory_mode: str, memory_window_length: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]` | Extract a slice of the trajectory as prehistory. |

### `hidden_attractors.contracts`

Source: `version_2/hidden_attractors/contracts.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `validate_contracts` | 13 | `public/module` | `validate_contracts(config: Dict[str, Any], resolved: bool=False) -> None` | Validate all mathematical and numerical contract keys in the configuration. |

### `hidden_attractors.diagnostics.periodicity`

Source: `version_2/hidden_attractors/diagnostics/periodicity.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_post_transient_segment` | 22 | `internal` | `_post_transient_segment(trajectory: np.ndarray, t_transient: float) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_power_summary` | 31 | `internal` | `_power_summary(values: np.ndarray, h: float) -> tuple[float, float, float]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_window_frequency_drift` | 47 | `internal` | `_window_frequency_drift(values: np.ndarray, h: float, n_windows: int) -> float` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_component_metrics` | 55 | `internal` | `_component_metrics(values: np.ndarray, h: float, config: Mapping[str, Any], name: str) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `classify_post_transient_periodicity` | 84 | `public/module` | `classify_post_transient_periodicity(trajectory: np.ndarray, *, h: float, config: Mapping[str, Any] \| None=None) -> dict[str, Any]` | Label target-system dynamics using FFT/PSD and multi-component agreement. |
| `function` | `promotion_label_after_hiddenness_probe` | 137 | `public/module` | `promotion_label_after_hiddenness_probe(periodicity: Mapping[str, Any], *, target_contacts_from_equilibria: int \| None) -> str` | Prevent a periodic orbit from being promoted as a hidden chaotic attractor. |

### `hidden_attractors.diagnostics.zero_one`

Source: `version_2/hidden_attractors/diagnostics/zero_one.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `run_zero_one_diagnostic` | 23 | `public/module` | `run_zero_one_diagnostic(times: np.ndarray, states: np.ndarray, observable: str, output_dir: Path, *, t_burn: float=0.0, n_c: int=100, c_min: float=0.1, c_max: float=3.04159, seed: int=12345, threshold_chaotic: float=0.85, threshold_regular: float=0.15, system_id: str='unknown', metadata_base: Dict[str, Any] \| None=None) -> Dict[str, Any]` | Execute 0-1 test on a trajectory time-series and write outputs. |
| `function` | `run_zero_one_from_config` | 187 | `public/module` | `run_zero_one_from_config(config: Dict[str, Any]) -> Dict[str, Any]` | Simulate from config and run 0-1 test workflow. |

### `hidden_attractors.integrations.abm`

Source: `version_2/hidden_attractors/integrations/abm.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `eval_rhs` | 42 | `public/module` | `eval_rhs(rhs: Callable, t: float, x: np.ndarray) -> np.ndarray` | Call ``rhs(t, x)`` or ``rhs(x)`` and return a float64 array. |
| `function` | `_python_abm_integrate` | 63 | `internal` | `_python_abm_integrate(rhs: Callable, x0: np.ndarray, q: float, h: float, t_final: float, divergence_norm: Optional[float]=120.0, history_times: Optional[np.ndarray]=None, history_states: Optional[np.ndarray]=None, memory_mode: str='full', memory_window_length: Optional[int]=None, early_stop_config: Optional[dict]=None, equilibria: Optional[List[np.ndarray]]=None, memory_window_steps: Optional[int]=None, memory_window_time: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray, str]` | Pure-Python ABM PECE integrator for Caputo FDEs. |
| `function` | `caputo_abm_integrate` | 292 | `public/module` | `caputo_abm_integrate(rhs: Callable, x0: np.ndarray, q: float, h: float, t_final: float, divergence_norm: Optional[float]=120.0, history_times: Optional[np.ndarray]=None, history_states: Optional[np.ndarray]=None, memory_mode: str='full', memory_window_length: Optional[int]=None, system: Optional[Any]=None, use_c_backend: bool=True, early_stop_config: Optional[dict]=None, equilibria: Optional[List[np.ndarray]]=None, memory_window_steps: Optional[int]=None, memory_window_time: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray, str]` | Integrate a Caputo FDE with the ABM predictor-corrector. |

### `hidden_attractors.integrations.abm_fractional`

Source: `version_2/hidden_attractors/integrations/abm_fractional.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `normalize_component_orders` | 17 | `public/module` | `normalize_component_orders(orders: float \| list[float] \| tuple[float, ...] \| np.ndarray, dimension: int) -> np.ndarray` | Return one validated derivative order per state component. |
| `function` | `classify_component_orders` | 36 | `public/module` | `classify_component_orders(orders: np.ndarray) -> str` | Classify normalized orders for result metadata. |
| `function` | `_eval_rhs` | 47 | `internal` | `_eval_rhs(rhs: Callable, t: float, state: np.ndarray, parameters: Mapping[str, float] \| None) -> np.ndarray` | Evaluate common autonomous and time-dependent RHS signatures. |
| `function` | `_weights_for_order` | 70 | `internal` | `_weights_for_order(q: float, h: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]` | Precompute predictor and corrector history weights for one order. |
| `function` | `integrate_fractional_abm` | 96 | `public/module` | `integrate_fractional_abm(rhs: Callable, x0: np.ndarray, orders: float \| list[float] \| tuple[float, ...] \| np.ndarray, h: float, n_steps: int, parameters: Mapping[str, float] \| None=None, *, memory_protocol: str='block_restart', divergence_norm: float \| None=None) -> tuple[np.ndarray, np.ndarray, str]` | Integrate one ABM block with scalar or component-wise Caputo orders. |

### `hidden_attractors.integrations.adm_wu2023`

Source: `version_2/hidden_attractors/integrations/adm_wu2023.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_gamma_factors` | 70 | `internal` | `_gamma_factors(q: float) -> Dict[str, float]` | Pre-compute all Gamma values needed for the 4th-order ADM coefficients. |
| `function` | `_extract_arctan_parameters` | 82 | `internal` | `_extract_arctan_parameters(params: Any) -> Tuple[float, float, float, float, float, float]` | Return alpha, beta, gamma, a1, a2, rho from either project notation. |
| `function` | `_adm_step` | 112 | `internal` | `_adm_step(x0: float, y0: float, z0: float, alpha: float, beta: float, gamma: float, a1: float, a2: float, rho: float, q: float, h: float, gf: Dict[str, float]) -> Tuple[float, float, float]` | Advance one step of the 4th-order ADM scheme. |
| `function` | `adm_wu2023_integrate` | 249 | `public/module` | `adm_wu2023_integrate(params: Any, x0: np.ndarray, q: float, h: float, N: int, divergence_norm: float=120.0) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]` | Integrate the Wu2023 fractional Chua arctan system using the 4th-order ADM. |
| `function` | `adm_wu2023_integrate_from_config` | 377 | `public/module` | `adm_wu2023_integrate_from_config(config: Dict[str, Any], x0: np.ndarray, label: str='unnamed') -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]` | Wrap :func:`adm_wu2023_integrate` reading params/options from a config dict. |
| `function` | `rhs_chua_arctan` | 410 | `public/module` | `rhs_chua_arctan(x: np.ndarray, params: Any) -> np.ndarray` | Evaluate the RHS vector field F(X) of the Wu2023 system. |

### `hidden_attractors.integrations.efork`

Source: `version_2/hidden_attractors/integrations/efork.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `EFORK3Coefficients` | 36 | `public/module` | `class EFORK3Coefficients(object)` | Coefficients of the explicit three-stage fractional RK method. |
| `function` | `efork3_coefficients` | 49 | `public/module` | `efork3_coefficients(alpha: float) -> EFORK3Coefficients` | Return three-stage EFORK coefficients for ``0 < alpha < 1``. |
| `function` | `_history_term` | 81 | `internal` | `_history_term(t_eval: float, times: np.ndarray, states: np.ndarray, n_local: int, alpha: float, h: float, s_idx: int=0) -> np.ndarray` | Evaluate the discrete Caputo history kernel at ``t_eval``. |
| `function` | `_python_efork3_integrate` | 115 | `internal` | `_python_efork3_integrate(rhs: Callable[[float, np.ndarray], np.ndarray], x0: np.ndarray, q: float, h: float, t_final: float, divergence_norm: Optional[float]=120.0, history_times: Optional[np.ndarray]=None, history_states: Optional[np.ndarray]=None, memory_mode: str='full', memory_window_length: Optional[int]=None, early_stop_config: Optional[dict]=None, equilibria: Optional[List[np.ndarray]]=None) -> Tuple[np.ndarray, np.ndarray, str]` | Pure-Python EFORK-3 Caputo integration (exact published algorithm). |
| `function` | `efork_integrate` | 277 | `public/module` | `efork_integrate(system: Any, x0: np.ndarray, q: float, h: float, t_final: float, memory_mode: str='full', memory_window_length: Optional[int]=None, k: float=0.0, eps: float=1.0, use_c_backend: bool=True, divergence_norm: Optional[float]=None, early_stop_config: Optional[dict]=None, equilibria: Optional[List[np.ndarray]]=None, history_times: Optional[np.ndarray]=None, history_states: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray, str]` | Integrate a Lur'e system using EFORK-3 (Caputo, 0 < q < 1) or Euler for q = 1. |

### `hidden_attractors.integrations.external_tools`

Source: `version_2/hidden_attractors/integrations/external_tools.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `ExternalTool` | 23 | `public/module` | `class ExternalTool(object)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `require_external` | 68 | `public/module` | `require_external(import_name: str, package_name: str \| None=None) -> Any` | Import an optional dependency or raise a clear installation error. |
| `function` | `external_tool_report` | 78 | `public/module` | `external_tool_report() -> list[dict[str, Any]]` | Return documentation-ready metadata for the registered tools. |
| `function` | `available_complexity_backends` | 103 | `public/module` | `available_complexity_backends() -> list[str]` | Return installed optional complexity backends. |
| `function` | `_first_available_backend` | 116 | `internal` | `_first_available_backend(preferred: str \| None=None) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `compute_complexity_measures` | 128 | `public/module` | `compute_complexity_measures(signal: Sequence[float], *, backend: str='auto', sample_rate: float=1.0, measures: Iterable[str] \| None=None) -> dict[str, float]` | Compute scalar complexity measures through optional external libraries. |

### `hidden_attractors.integrations.fractional_c`

Source: `version_2/hidden_attractors/integrations/fractional_c.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `eval_rhs` | 49 | `public/module` | `eval_rhs(rhs: Callable, t: float, x: np.ndarray) -> np.ndarray` | Evaluate ``rhs(t, x)`` or fall back to ``rhs(x)`` for legacy callables. |
| `function` | `_shared_suffix` | 61 | `internal` | `_shared_suffix() -> str` | Return the platform-appropriate shared-library extension. |
| `class` | `GeneralFractionalCBackend` | 74 | `public/module` | `class GeneralFractionalCBackend(object)` | Singleton wrapper for the compiled generic C fractional integrator. |
| `method` | `GeneralFractionalCBackend.__init__` | 86 | `dunder` | `__init__(self) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `GeneralFractionalCBackend.get_instance` | 90 | `public/module` | `get_instance(cls) -> 'GeneralFractionalCBackend'` | Return the singleton, compiling the C library on first call. |
| `function` | `fractional_integrate` | 183 | `public/module` | `fractional_integrate(rhs: Any, x0: np.ndarray, q: float, h: float, t_final: float, method: str, memory_mode: str, memory_window_length: Optional[int]=None, history_times: Optional[np.ndarray]=None, history_states: Optional[np.ndarray]=None, system: Optional[Any]=None, params: Optional[Any]=None, use_c_backend: bool=True, divergence_norm: float=120.0, return_history: bool=False, allow_python_fallback: bool=False, early_stop_config: Optional[dict]=None, equilibria: Optional[List[np.ndarray]]=None) -> Tuple[np.ndarray, np.ndarray, str, dict]` | Integrate a fractional-order ODE system. |

### `hidden_attractors.integrations.general`

Source: `version_2/hidden_attractors/integrations/general.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_native_chua_params` | 17 | `internal` | `_native_chua_params(system: Any)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_try_native_chua_fractional` | 34 | `internal` | `_try_native_chua_fractional(*, system: Optional[Any], x0: np.ndarray, q: float, h: float, t_final: float, integrator: str, memory_mode: str, memory_window_length: Optional[int], divergence_norm: Optional[float]) -> Optional[Tuple[np.ndarray, np.ndarray, str]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `integrate_general` | 68 | `public/module` | `integrate_general(rhs: Callable[[float, np.ndarray], np.ndarray], x0: np.ndarray, q: float, h: float, t_final: float, integrator: str='efork', memory_mode: str='full', memory_window_length: Optional[int]=None, divergence_norm: Optional[float]=120.0, system: Optional[Any]=None, use_c_backend: bool=True, early_stop_config: Optional[dict]=None, equilibria: Optional[List[np.ndarray]]=None) -> Tuple[np.ndarray, np.ndarray, str]` | Unified general solver facade for integrating any system (fractional or integer). |

### `hidden_attractors.integrations.numba_kernels`

Source: `version_2/hidden_attractors/integrations/numba_kernels.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_eval_rhs_njit` | 74 | `internal` | `_eval_rhs_njit(x: np.ndarray, P: np.ndarray, b: np.ndarray, psi_coeff: float, psi_kind: int) -> np.ndarray` | Evaluate the Chua system RHS for 3-dimensional state vector x. |
| `function` | `efork3_q1_integrate_numba` | 122 | `public/module` | `efork3_q1_integrate_numba(x0: np.ndarray, P: np.ndarray, b: np.ndarray, psi_coeff: float, psi_kind: int, h: float, n_steps: int, divergence_norm_hard: float, es_enabled: bool, div_enabled: bool, div_norm_es: float, div_consec_threshold: int, div_growth_factor: float, eq_enabled: bool, eq_tol: float, eq_deriv_tol: float, eq_consec_threshold: int, eq_min_t: float, equilibria: np.ndarray)` | EFORK-3 (q→1 limit) integration loop compiled to native code. |
| `function` | `integrate_efork3_q1_numba` | 292 | `public/module` | `integrate_efork3_q1_numba(system, x0, h: float, t_final: float, divergence_norm=120.0, early_stop_config=None, equilibria=None)` | Python entry point for the Numba-accelerated EFORK-3 q=1.0 integrator. |
| `function` | `benchmark` | 404 | `public/module` | `benchmark(n_steps: int=50000, n_trials: int=5)` | Compare Python-loop vs Numba kernel for a standard Chua integration. |

### `hidden_attractors.integrations.rk4`

Source: `version_2/hidden_attractors/integrations/rk4.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `rk4_integrate` | 14 | `public/module` | `rk4_integrate(rhs: Callable[..., np.ndarray], x0: np.ndarray, h: float, N: int, divergence_norm: float=120.0) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]` | Integrate a system of ODEs using the classical 4th-order Runge-Kutta method. |

### `hidden_attractors.integrations.selector`

Source: `version_2/hidden_attractors/integrations/selector.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_canonical_name` | 45 | `internal` | `_canonical_name(integrator: str) -> str` | Normalize integrator names for internal dispatch. |
| `function` | `validate_integrator_compatibility` | 53 | `public/module` | `validate_integrator_compatibility(integrator: str, q: float) -> str` | Validate and return canonical integrator name. |
| `function` | `get_integrator_fn` | 110 | `public/module` | `get_integrator_fn() -> Callable` | Return the unified ``integrate_general`` function. |
| `function` | `integrate` | 119 | `public/module` | `integrate(rhs: Callable, x0: np.ndarray, q: float, h: float, t_final: float, integrator: str='efork3', memory_mode: str='full', memory_window_length: Optional[int]=None, divergence_norm: Optional[float]=120.0, system: Optional[Any]=None, use_c_backend: bool=True, allow_python_fallback: bool=True, early_stop_config: Optional[dict]=None, equilibria: Optional[List[np.ndarray]]=None) -> Tuple[np.ndarray, np.ndarray, str]` | Validated unified integrator entry point. |

### `hidden_attractors.io`

Source: `version_2/hidden_attractors/io.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `timestamp` | 19 | `public/module` | `timestamp(fmt: str='%Y%m%d_%H%M%S') -> str` | Return a timestamp string for non-overwriting output folder names. |
| `function` | `safe_name` | 42 | `public/module` | `safe_name(text: str) -> str` | Return a filesystem-safe version of *text* with readability preserved. |
| `function` | `json_safe` | 68 | `public/module` | `json_safe(obj: Any) -> Any` | Recursively convert *obj* to types accepted by :func:`json.dumps`. |
| `function` | `write_json` | 114 | `public/module` | `write_json(path: str \| Path, data: Dict[str, Any]) -> None` | Write *data* as JSON, creating parent directories as needed. |
| `function` | `read_json` | 145 | `public/module` | `read_json(path: str \| Path) -> Dict[str, Any]` | Read a JSON object from *path*. |
| `function` | `_csv_value` | 169 | `internal` | `_csv_value(value: Any) -> Any` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `write_csv` | 183 | `public/module` | `write_csv(path: str \| Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str] \| None=None) -> None` | Write a sequence of row dicts to a CSV file with stable field ordering. |
| `function` | `append_csv` | 222 | `public/module` | `append_csv(path: str \| Path, row: Dict[str, Any], fields: Sequence[str]) -> None` | Append one row to a CSV file, writing the header if the file is new. |
| `function` | `read_csv_rows` | 249 | `public/module` | `read_csv_rows(path: str \| Path) -> List[Dict[str, str]]` | Read a CSV file into a list of string dicts. |
| `function` | `load_trajectory_csv` | 271 | `public/module` | `load_trajectory_csv(path: str \| Path, columns: Sequence[str]=('t', 'x', 'y', 'z')) -> np.ndarray` | Load a trajectory CSV into the package-standard column layout. |

### `hidden_attractors.lure.compatibility`

Source: `version_2/hidden_attractors/lure/compatibility.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `LureCompatibilityValidator` | 16 | `public/module` | `class LureCompatibilityValidator(object)` | Validator to classify Lur'e equivalence and calculate splitting details. |
| `method` | `LureCompatibilityValidator.validate` | 20 | `public/module` | `validate(system: ChaoticSystem, config: Any=None) -> dict[str, Any]` | Validate system compatibility with the Lur'e feedback model. |

### `hidden_attractors.lure.decomposition`

Source: `version_2/hidden_attractors/lure/decomposition.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `validate_lure_decomposition` | 4 | `public/module` | `validate_lure_decomposition(system: Any) -> bool` | Validate that the Lur'e decomposition vector field matches evaluation. |

### `hidden_attractors.lure.describing_function`

Source: `version_2/hidden_attractors/lure/describing_function.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `DescribingFunctionResult` | 10 | `public/module` | `class DescribingFunctionResult(object)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `N_quadrature` | 16 | `public/module` | `N_quadrature(A: float, psi_func) -> float` | Evaluate describing function by standard numerical quadrature: N(A) = (2 / (pi * A)) * integral_0^pi psi(A * cos(theta)) * cos(theta) dtheta |
| `function` | `N_segmented_quadrature` | 29 | `public/module` | `N_segmented_quadrature(A: float, psi_func, theta_breaks: List[float]) -> float` | Evaluate describing function by segmented numerical quadrature: Integrate piecewise sections delimited by theta_breaks in [0, pi] to avoid roundoff errors. |
| `function` | `get_describing_function_capabilities` | 48 | `public/module` | `get_describing_function_capabilities(system: Any) -> Dict[str, Any]` | Retrieve capabilities dictionary or define dynamic default maps. |
| `function` | `evaluate_describing_function` | 68 | `public/module` | `evaluate_describing_function(system: Any, A: float, mode: str='auto') -> DescribingFunctionResult` | General evaluation interface resolving closed-form, piecewise or quadrature modes. |
| `function` | `solve_amplitude_from_gain` | 94 | `public/module` | `solve_amplitude_from_gain(system: Any, k: float, A_min: float, A_max: float, mode: str='auto') -> float` | Solve N(A0) - k = 0 using a robust 1D bisection search. |
| `function` | `evaluate_describing_function_batch` | 118 | `public/module` | `evaluate_describing_function_batch(system: Any, A_array: np.ndarray, mode: str='auto') -> np.ndarray` | Evaluate the describing function N(A) for an entire array of amplitudes. |

### `hidden_attractors.lure.nyquist`

Source: `version_2/hidden_attractors/lure/nyquist.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `HarmonicCandidate` | 13 | `public/module` | `class HarmonicCandidate(object)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `CandidateList` | 21 | `public/module` | `class CandidateList(list)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `CandidateList.__init__` | 22 | `dunder` | `__init__(self, *args, **kwargs)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `find_harmonic_candidates` | 26 | `public/module` | `find_harmonic_candidates(system: Any, transfer_mode: str, seed_strategy: str='k_phi', df_residual_tol: float=0.01, omega_min: float=0.01, omega_max: float=20.0, amplitude_min: float=0.01, amplitude_max: float=20.0, grid_size_omega: int=200, grid_size_amplitude: int=200, root_refinement: bool=True, q: Optional[float]=None, describing_function_mode: str='auto', transfer_convention: str='standard', harmonic_condition: str='1_minus_WN', precomputed_W_vals: Optional[np.ndarray]=None, precomputed_omega_grid: Optional[np.ndarray]=None) -> List[Tuple[float, float, float]]` | Find all candidate pairs (A0, omega0, k) solving the harmonic condition. |

### `hidden_attractors.lure.seeds`

Source: `version_2/hidden_attractors/lure/seeds.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_lambda_from_frequency` | 7 | `internal` | `_lambda_from_frequency(omega0: float, q: float, transfer_mode: str) -> complex` | Return the eigenvalue lambda that corresponds to the harmonic frequency. |
| `function` | `build_closed_form_integer_seed` | 30 | `public/module` | `build_closed_form_integer_seed(system: Any, A0: float, omega0: float, k: float, seed_sign_convention: str='kuznetsov') -> Tuple[np.ndarray, np.ndarray]` | Construct the seed using the algebraic closed-form formula for q = 1. |
| `function` | `build_modal_lure_seed` | 61 | `public/module` | `build_modal_lure_seed(system: Any, A0: float, omega0: float, k: float, q: float, transfer_mode: str='auto', theta: float=0.0) -> Tuple[np.ndarray, np.ndarray, complex, complex]` | Construct the harmonic seed from the dominant eigenvector of P0 = P + k*b*r^T. |
| `function` | `build_lure_seed` | 103 | `public/module` | `build_lure_seed(system: Any, A0: float, omega0: float, k: float, seed_sign_convention: str='kuznetsov', q: Optional[float]=None, transfer_mode: str='auto', theta: float=0.0, seed_construction: str='modal') -> Tuple[np.ndarray, np.ndarray]` | Construct the initial state seed X_seed and its symmetric partner -X_seed. |

### `hidden_attractors.lure.transfer`

Source: `version_2/hidden_attractors/lure/transfer.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `validate_fractional_order` | 4 | `public/module` | `validate_fractional_order(q: float) -> float` | Validate that 0 < q <= 1.0. |
| `function` | `W_spectral` | 11 | `public/module` | `W_spectral(lam: complex, P: np.ndarray, b: np.ndarray, r: np.ndarray, transfer_convention: str='standard') -> complex` | Spectral form. |
| `function` | `W_eval` | 24 | `public/module` | `W_eval(omega: float \| np.ndarray, q: float, transfer_mode: str, P: np.ndarray, b: np.ndarray, r: np.ndarray, transfer_convention: str='standard') -> complex \| np.ndarray` | Evaluates the transfer function at frequency omega (supports scalar or numpy array). |
| `function` | `W_precompute_spectral` | 83 | `public/module` | `W_precompute_spectral(P: np.ndarray, b: np.ndarray, r: np.ndarray, transfer_convention: str='standard') -> dict` | Pre-compute and cache the spectral decomposition of ``P``. |
| `function` | `W_eval_from_cache` | 117 | `public/module` | `W_eval_from_cache(omega: 'float \| np.ndarray', q: float, transfer_mode: str, cache: dict) -> 'complex \| np.ndarray'` | Evaluate ``W(jω)`` using a pre-computed spectral cache. |

### `hidden_attractors.models.chua`

Source: `version_2/hidden_attractors/models/chua.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `normalize_chua_model` | 35 | `public/module` | `normalize_chua_model(raw: str \| int \| None='nonsmooth') -> str` | Normalize project aliases for the supported Chua nonlinearities. |
| `class` | `ChuaParameters` | 91 | `public/module` | `class ChuaParameters(object)` | Parameters for the Chua systems used in the project. |
| `method` | `ChuaParameters.__post_init__` | 150 | `dunder` | `__post_init__(self) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `chua_parameters` | 161 | `public/module` | `chua_parameters(*, model: str='nonsmooth', alpha: float=8.4562, beta: float=12.0732, gamma: float=0.0052, m0: float=-0.1768, m1: float=-1.1468, a1: float=0.4, a2: float=-1.5585, rho: float=1.0) -> ChuaParameters` | Build a :class:`ChuaParameters` object with explicit coefficients. |
| `function` | `chua_nonsmooth_parameters` | 226 | `public/module` | `chua_nonsmooth_parameters() -> ChuaParameters` | Return the project-default non-smooth Chua parameters. |
| `function` | `chua_arctan_parameters` | 249 | `public/module` | `chua_arctan_parameters() -> ChuaParameters` | Return the project-default smooth arctan Chua parameters. |
| `function` | `chua_arctan_wu2023_parameters` | 262 | `public/module` | `chua_arctan_wu2023_parameters() -> ChuaParameters` | Return the official smooth Chua coefficients reported by Wu et al. |
| `function` | `nonlinearity_nonsmooth` | 276 | `public/module` | `nonlinearity_nonsmooth(x: float, p: ChuaParameters) -> float` | Evaluate the non-smooth Chua characteristic, which is linear by pieces. |
| `function` | `nonlinearity_arctan` | 302 | `public/module` | `nonlinearity_arctan(x: float, p: ChuaParameters) -> float` | Evaluate the smooth arctan Chua nonlinearity. |
| `function` | `nonlinearity_chua` | 328 | `public/module` | `nonlinearity_chua(x: float, p: ChuaParameters) -> float` | Evaluate the nonlinearity selected by ``p.model``. |
| `function` | `rhs_chua` | 349 | `public/module` | `rhs_chua(state: np.ndarray, p: ChuaParameters \| None=None) -> np.ndarray` | Evaluate the Chua vector field for the selected nonlinearity. |
| `function` | `rhs_nonsmooth` | 374 | `public/module` | `rhs_nonsmooth(state: np.ndarray, p: ChuaParameters \| None=None) -> np.ndarray` | Evaluate the non-smooth Chua vector field for Caputo integrators. |
| `function` | `rhs_arctan` | 417 | `public/module` | `rhs_arctan(state: np.ndarray, p: ChuaParameters \| None=None) -> np.ndarray` | Evaluate the smooth arctan Chua vector field, irrespective of ``m0``/``m1``. |
| `function` | `jacobian_arctan` | 436 | `public/module` | `jacobian_arctan(state: np.ndarray, p: ChuaParameters \| None=None) -> np.ndarray` | Evaluate the analytic Jacobian for ``f(x)=a1*x+a2*atan(rho*x)``. |
| `function` | `equilibria_arctan` | 453 | `public/module` | `equilibria_arctan(p: ChuaParameters \| None=None) -> Dict[str, np.ndarray]` | Compute the origin and symmetric nonzero equilibria for smooth Chua. |
| `function` | `jacobian_nonsmooth` | 503 | `public/module` | `jacobian_nonsmooth(state: np.ndarray, p: ChuaParameters \| None=None) -> np.ndarray` | Evaluate the regional Jacobian of the non-smooth Chua vector field. |
| `function` | `equilibria_nonsmooth` | 542 | `public/module` | `equilibria_nonsmooth(p: ChuaParameters \| None=None) -> Dict[str, np.ndarray]` | Compute the three equilibria of the non-smooth Chua model. |
| `function` | `chua_piecewise_parameters` | 597 | `public/module` | `chua_piecewise_parameters() -> ChuaParameters` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `nonlinearity_piecewise` | 601 | `public/module` | `nonlinearity_piecewise(x: float, p: ChuaParameters) -> float` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `rhs_piecewise` | 605 | `public/module` | `rhs_piecewise(state: np.ndarray, p: ChuaParameters \| None=None) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `jacobian_piecewise` | 609 | `public/module` | `jacobian_piecewise(state: np.ndarray, p: ChuaParameters \| None=None) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `equilibria_piecewise` | 613 | `public/module` | `equilibria_piecewise(p: ChuaParameters \| None=None) -> Dict[str, np.ndarray]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |

### `hidden_attractors.native.backends`

Source: `version_2/hidden_attractors/native/backends.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `_CFractionalLyapunovRequest` | 45 | `internal` | `class _CFractionalLyapunovRequest(ctypes.Structure)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `_CFractionalLyapunovResult` | 63 | `internal` | `class _CFractionalLyapunovResult(ctypes.Structure)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `NativeFractionalVariationalBackend` | 75 | `public/module` | `class NativeFractionalVariationalBackend(object)` | Native C backend for extensive fractional variational LE calculations. |
| `method` | `NativeFractionalVariationalBackend.build` | 83 | `public/module` | `build(cls, output_name: str='fractional_variational_lyapunov') -> 'NativeFractionalVariationalBackend'` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `NativeFractionalVariationalBackend._parameter_vector` | 128 | `internal` | `_parameter_vector(system_id: str, parameters: dict[str, float] \| Any) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `NativeFractionalVariationalBackend.rhs_jacobian` | 135 | `public/module` | `rhs_jacobian(self, system_id: str, parameters: dict[str, float], state: Sequence[float]) -> tuple[np.ndarray, np.ndarray]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `NativeFractionalVariationalBackend.run` | 154 | `public/module` | `run(self, request: FractionalLyapunovRequest) -> FractionalLyapunovResult` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_shared_suffix` | 214 | `internal` | `_shared_suffix() -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `FractionalChuaBackend` | 223 | `public/module` | `class FractionalChuaBackend(object)` | Wrapper for ``chua_frac_backend_lib.c``. |
| `method` | `FractionalChuaBackend.build` | 239 | `public/module` | `build(cls, output_name: str='chua_frac_backend') -> 'FractionalChuaBackend'` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `FractionalChuaBackend.set_nonsmooth_params` | 298 | `public/module` | `set_nonsmooth_params(self, params: ChuaParameters) -> None` | Set the C backend to the non-smooth Chua parameters. |
| `method` | `FractionalChuaBackend.set_piecewise_params` | 304 | `public/module` | `set_piecewise_params(self, params: ChuaParameters) -> None` | Compatibility alias for :meth:`set_nonsmooth_params`. |
| `method` | `FractionalChuaBackend.set_arctan_params` | 309 | `public/module` | `set_arctan_params(self, params: ChuaParameters) -> None` | Set the C backend to a smooth arctan Chua parameterization. |
| `method` | `FractionalChuaBackend.set_params` | 316 | `public/module` | `set_params(self, params: ChuaParameters) -> None` | Dispatch parameter loading according to ``params.model``. |
| `method` | `FractionalChuaBackend.integrate_efork3` | 324 | `public/module` | `integrate_efork3(self, x0: Sequence[float], *, q: float, h: float, Lm: float, t_final: float, k: float=0.0, eps: float=1.0) -> np.ndarray` | Integrate one trajectory and return columns ``t,x,y,z``. |
| `method` | `FractionalChuaBackend.continue_efork3` | 360 | `public/module` | `continue_efork3(self, x0: Sequence[float], *, lambda_values: Sequence[float] \| None=None, eps_values: Sequence[float] \| None=None, q: float, k: float, h: float, Lm: float, t_transient: float, t_keep: float, t_observe: float=0.0, carry_memory: bool=True) -> dict[str, Any]` | Run public ``lambda`` continuation through the native C ABI. |
| `class` | `FullHistoryABMBackend` | 439 | `public/module` | `class FullHistoryABMBackend(object)` | Native ABM backend for the non-smooth Chua system. |
| `method` | `FullHistoryABMBackend.build` | 452 | `public/module` | `build(cls, output_name: str='chua_abm_full_history') -> 'FullHistoryABMBackend'` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `FullHistoryABMBackend.set_nonsmooth_params` | 514 | `public/module` | `set_nonsmooth_params(self, params: ChuaParameters) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `FullHistoryABMBackend.equilibria` | 517 | `public/module` | `equilibria(self) -> dict[str, np.ndarray]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `FullHistoryABMBackend.integrate` | 522 | `public/module` | `integrate(self, x0: Sequence[float], *, q: float, h: float, t_final: float) -> np.ndarray` | Integrate one full-history ABM trajectory as columns ``t,x,y,z``. |
| `method` | `FullHistoryABMBackend.integrate_truncated` | 552 | `public/module` | `integrate_truncated(self, x0: Sequence[float], *, q: float, h: float, Lm: float, t_final: float) -> np.ndarray` | Integrate with a sliding restarted ABM history window of length ``Lm``. |
| `method` | `FullHistoryABMBackend._continue_abm` | 584 | `internal` | `_continue_abm(self, x0: Sequence[float], *, lambda_values: Sequence[float], q: float, k: float, h: float, t_transient: float, t_keep: float, truncated_history: bool, Lm: float \| None) -> dict[str, Any]` | Continue the Lur'e deformation while retaining declared history. |
| `method` | `FullHistoryABMBackend.continue_full_history` | 672 | `public/module` | `continue_full_history(self, x0: Sequence[float], *, lambda_values: Sequence[float], q: float, k: float, h: float, t_transient: float, t_keep: float) -> dict[str, Any]` | Continue with complete causal Caputo history across eta stages. |
| `method` | `FullHistoryABMBackend.continue_truncated_history` | 697 | `public/module` | `continue_truncated_history(self, x0: Sequence[float], *, lambda_values: Sequence[float], q: float, k: float, h: float, Lm: float, t_transient: float, t_keep: float) -> dict[str, Any]` | Continue with an explicit finite restarted memory window ``Lm``. |
| `class` | `BasinBackend` | 725 | `public/module` | `class BasinBackend(object)` | Wrapper for ``chua_basin_lib.c`` classification routines. |
| `method` | `BasinBackend.build` | 731 | `public/module` | `build(cls, output_name: str='chua_basin_backend') -> 'BasinBackend'` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `BasinBackend.set_nonsmooth_params` | 768 | `public/module` | `set_nonsmooth_params(self, params: ChuaParameters) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `BasinBackend.set_piecewise_params` | 772 | `public/module` | `set_piecewise_params(self, params: ChuaParameters) -> None` | Compatibility alias for :meth:`set_nonsmooth_params`. |
| `method` | `BasinBackend.set_arctan_params` | 777 | `public/module` | `set_arctan_params(self, params: ChuaParameters) -> None` | Set the basin backend to a smooth arctan Chua parameterization. |
| `method` | `BasinBackend.set_params` | 784 | `public/module` | `set_params(self, params: ChuaParameters) -> None` | Dispatch parameter loading according to ``params.model``. |
| `method` | `BasinBackend.equilibria` | 792 | `public/module` | `equilibria(self) -> dict[str, np.ndarray]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `BasinBackend.classify_point` | 797 | `public/module` | `classify_point(self, x0: Sequence[float], *, q: float, h: float, Lm: float, t_final: float, t_burn: float, divergence_norm: float=120.0, r_bound: float=60.0, equilibrium_tol: float=0.001, cap_win: int=150, mean_x_gap: float=0.75) -> int` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `FractionalLyapunovBackend` | 833 | `public/module` | `class FractionalLyapunovBackend(object)` | Runner for the native EFORK/Benettin finite-memory diagnostic. |
| `method` | `FractionalLyapunovBackend.build` | 839 | `public/module` | `build(cls, output_name: str='chua_frac_lyapunov_efork_benettin') -> 'FractionalLyapunovBackend'` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `FractionalLyapunovBackend.run` | 850 | `public/module` | `run(self, x0: Sequence[float], *, params: ChuaParameters \| None=None, q: float, h: float, Lm: float, t_burn: float, n_blocks: int, t_block: float, convergence_csv: str \| Path) -> dict[str, Any]` | Execute the native diagnostic and return the reported exponents. |
| `class` | `GeneralFDEBackend` | 905 | `public/module` | `class GeneralFDEBackend(object)` | Wrapper for general FDE solver in C. |
| `method` | `GeneralFDEBackend.build` | 912 | `public/module` | `build(cls, output_name: str='general_fde_solver') -> 'GeneralFDEBackend'` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `GeneralFDEBackend.integrate` | 955 | `public/module` | `integrate(self, rhs: Any, x0: np.ndarray, q: float, h: float, t_final: float, divergence_norm: float=120.0, integrator: str='efork') -> tuple[np.ndarray, np.ndarray, str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |

### `hidden_attractors.native.contracts`

Source: `version_2/hidden_attractors/native/contracts.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `BackendBuildSpec` | 13 | `public/module` | `class BackendBuildSpec(object)` | Native build inputs for a system-specific backend. |
| `class` | `IntegrationRequest` | 23 | `public/module` | `class IntegrationRequest(object)` | Common integration request for integer and fractional backends. |
| `class` | `IntegrationResult` | 35 | `public/module` | `class IntegrationResult(object)` | Common integration result returned by backend adapters. |
| `class` | `FractionalLyapunovRequest` | 44 | `public/module` | `class FractionalLyapunovRequest(object)` | Native-only request for extensive fractional Lyapunov calculations. |
| `class` | `FractionalLyapunovResult` | 63 | `public/module` | `class FractionalLyapunovResult(object)` | Spectrum and provenance returned by the native fractional backend. |
| `class` | `NativeIntegrationBackend` | 77 | `public/module` | `class NativeIntegrationBackend(Protocol)` | Minimal protocol expected from reusable native integration backends. |
| `method` | `NativeIntegrationBackend.integrate` | 80 | `public/module` | `integrate(self, request: IntegrationRequest) -> IntegrationResult` | Run one integration request and return stored trajectory samples. |
| `class` | `NativeLyapunovBackend` | 84 | `public/module` | `class NativeLyapunovBackend(Protocol)` | Protocol for backend Lyapunov estimators. |
| `method` | `NativeLyapunovBackend.lyapunov` | 87 | `public/module` | `lyapunov(self, request: IntegrationRequest, *, t_burn: float, blocks: int) -> Sequence[float]` | Return Lyapunov exponent estimates for the requested system. |

### `hidden_attractors.native.rhs_registry`

Source: `version_2/hidden_attractors/native/rhs_registry.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `ChuaSaturationParamsStruct` | 5 | `public/module` | `class ChuaSaturationParamsStruct(ctypes.Structure)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `ChuaArctanParamsStruct` | 14 | `public/module` | `class ChuaArctanParamsStruct(ctypes.Structure)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `register_c_rhs` | 27 | `public/module` | `register_c_rhs(system_id: str, rhs_getter_name: str, params_builder: Any)` | Register a pre-compiled C RHS by system_id. |
| `function` | `_get_param` | 35 | `internal` | `_get_param(system: Any, key: str, default: float=0.0) -> float` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `build_chua_saturation_params` | 49 | `public/module` | `build_chua_saturation_params(system: Any) -> ctypes.Structure` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `build_chua_arctan_params` | 58 | `public/module` | `build_chua_arctan_params(system: Any) -> ctypes.Structure` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `get_c_rhs_and_params` | 77 | `public/module` | `get_c_rhs_and_params(system: Any, lib: Any) -> Tuple[Optional[int], Optional[ctypes.Structure]]` | Retrieve the function pointer address and the built ctypes parameters structure for a system. |

### `hidden_attractors.parallel`

Source: `version_2/hidden_attractors/parallel.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `CompileResult` | 36 | `public/module` | `class CompileResult(object)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `env_flag` | 45 | `public/module` | `env_flag(name: str, default: bool=False) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `allow_no_openmp` | 52 | `public/module` | `allow_no_openmp() -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `force_single_openmp_thread_env` | 56 | `public/module` | `force_single_openmp_thread_env(env: MutableMapping[str, str] \| None=None) -> MutableMapping[str, str]` | Return an environment where any nested OpenMP runtime is single-threaded. |
| `function` | `force_single_openmp_thread_current_process` | 64 | `public/module` | `force_single_openmp_thread_current_process() -> None` | Apply the worker-side rule for Python multiprocessing tasks. |
| `function` | `distribute_openmp_threads` | 70 | `public/module` | `distribute_openmp_threads(total_threads: int, external_processes: int) -> int` | Threads per external process when several independent processes are launched. |
| `function` | `_brew_libomp_prefix` | 77 | `internal` | `_brew_libomp_prefix() -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_compiler_and_flags` | 118 | `internal` | `_compiler_and_flags(openmp: bool, target_kind: str) -> tuple[str, List[str], List[str]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `build_c_compile_command` | 145 | `public/module` | `build_c_compile_command(source: Path, output: Path, *, target_kind: str, openmp: bool) -> List[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_format_compile_failure` | 158 | `internal` | `_format_compile_failure(cmd: Sequence[str], exc: subprocess.CalledProcessError) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `compile_c_target` | 173 | `public/module` | `compile_c_target(source: str \| Path, output: str \| Path, *, target_kind: str, openmp: bool=True, logger: Callable[[str], None] \| None=None) -> CompileResult` | Compile a C backend according to the repository parallel policy. |
| `function` | `parallel_contract` | 258 | `public/module` | `parallel_contract(*, python_workers: int, omp_threads: int, backend_openmp_active: bool, seed_strategy: str='not_applicable', stage_kind: str) -> Dict[str, object]` | Small serializable contract for logs and JSON summaries. |

### `hidden_attractors.paths`

Source: `version_2/hidden_attractors/paths.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `get_packaged_examples_ref` | 21 | `public/module` | `get_packaged_examples_ref()` | Return a Traversable reference to the packaged examples configs. |
| `function` | `list_packaged_example_configs` | 27 | `public/module` | `list_packaged_example_configs() -> list[str]` | List filenames of all packaged example configuration files. |
| `function` | `get_example_config_resource` | 40 | `public/module` | `get_example_config_resource(filename: str)` | Return a Traversable reference to a specific example configuration file. |
| `function` | `get_packaged_examples_path` | 45 | `public/module` | `get_packaged_examples_path() -> Path` | Return the physical path fallback for local/editable installs when available. |

### `hidden_attractors.plotting.__init__`

Source: `version_2/hidden_attractors/plotting/__init__.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `generate_all_publication_figures` | 44 | `public/module` | `generate_all_publication_figures(*args, **kwargs)` | Lazily load the canonical generator so it can also run as a module. |

### `hidden_attractors.plotting.basin`

Source: `version_2/hidden_attractors/plotting/basin.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_output_path` | 16 | `internal` | `_output_path(path: str \| Path) -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `plot_basin_slices` | 22 | `public/module` | `plot_basin_slices(basin_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], system_id: str, output_dir: str \| Path) -> Dict[str, str]` | Plot multiple basin slices for all planes in basin_data. |
| `function` | `plot_basin_slice_file` | 58 | `public/module` | `plot_basin_slice_file(plane: str, u: np.ndarray, v: np.ndarray, mat: np.ndarray, eq_name: str, system_id: str, output_dir: str \| Path) -> str` | Renders and saves a high-quality 2D basin slice plot. |

### `hidden_attractors.plotting.biased_chua`

Source: `version_2/hidden_attractors/plotting/biased_chua.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `plot_centered_trajectory` | 25 | `public/module` | `plot_centered_trajectory(traj: np.ndarray, outpath: Path, t_burn: float, h: float) -> None` | 4-panel figure: 3D phase space + 2D projections + time series. |
| `function` | `plot_sign_audit` | 69 | `public/module` | `plot_sign_audit(first_arg: Union[List[Dict[str, Any]], np.ndarray], second_arg: Optional[Union[Path, np.ndarray]]=None, third_arg: Optional[np.ndarray]=None, fourth_arg: Optional[Path]=None) -> None` | Plots sign audit. |
| `function` | `plot_attractor_report` | 111 | `public/module` | `plot_attractor_report(states: np.ndarray, info: Union[List[str], str], outpath: Path) -> None` | Legacy or 2x2 format chaos attractor report. |
| `function` | `plot_continuation_metrics` | 149 | `public/module` | `plot_continuation_metrics(history: Union[Dict[str, List[float]], List[Dict[str, Any]]], outpath_or_prefix: Union[Path, str], outpath: Optional[Path]=None) -> None` | Plots continuation metrics (A_obs/A_theo, norm/coords vs eta). |
| `function` | `_get_colour_map` | 201 | `internal` | `_get_colour_map() -> Dict[str, str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `plot_sphere_summary` | 210 | `public/module` | `plot_sphere_summary(eq_name: str, eq_pt: np.ndarray, radius: float, runs: List[Dict], pts: np.ndarray, outpath: Path) -> None` | 3D figure: probe points colored by destination. |
| `function` | `build_hiddenness_heatmap` | 232 | `public/module` | `build_hiddenness_heatmap(records: List[Dict], radii: List[float]) -> tuple[plt.Figure, plt.Axes]` | Build the canonical TARGET-contact heatmap used by every report. |
| `function` | `plot_heatmap_hiddenness` | 282 | `public/module` | `plot_heatmap_hiddenness(records: List[Dict], radii: List[float], outpath: Path) -> None` | Export the canonical TARGET-contact heatmap. |
| `function` | `plot_candidate_report` | 292 | `public/module` | `plot_candidate_report(traj: np.ndarray, params_str: str, verdict: str, outpath: Path, h: float=0.01) -> None` | 7-panel report figure: 3D + 2D projections + time series + FFT + parameters. |
| `function` | `plot_biased_vs_centered` | 373 | `public/module` | `plot_biased_vs_centered(biased_traj: np.ndarray, centered_traj: Optional[np.ndarray], params_str: str, outpath: Path) -> None` | 3D comparison and FFT of the biased attractor vs centered reference. |
| `function` | `plot_mega_summary` | 420 | `public/module` | `plot_mega_summary(candidates: List[Dict], outpath: Path) -> None` | 3D mosaic of all candidates that survived. |

### `hidden_attractors.plotting.bifurcation`

Source: `version_2/hidden_attractors/plotting/bifurcation.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `plot_bifurcation_diagram_styled` | 20 | `public/module` | `plot_bifurcation_diagram_styled(points: Sequence[BifurcationPoint] \| Sequence[dict] \| np.ndarray, output_path: str \| Path, *, parameter_label: str='parameter', observable_label: str='observable', title: str='Bifurcation diagram', system_id: str='chua_fractional', q: float=1.0, integrator: str='unknown', memory_mode: str='unknown', t_final: float=0.0, t_burn: float=0.0) -> str` | Plot styled bifurcation diagram using centralized library APIs. |

### `hidden_attractors.plotting.dynamics`

Source: `version_2/hidden_attractors/plotting/dynamics.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_output_path` | 29 | `internal` | `_output_path(path: str \| Path) -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `plot_phase_space` | 35 | `public/module` | `plot_phase_space(trajectory: np.ndarray, output_path: str \| Path, *, dims: Sequence[str \| int]=('x', 'y', 'z'), title: str \| None='Phase space', max_points: int=5000, color_by_time: bool=True) -> str` | Plot a 2D or 3D phase-space view of a ``t,x,y,z`` trajectory. |
| `function` | `plot_phase_projections` | 74 | `public/module` | `plot_phase_projections(trajectory: np.ndarray, output_path: str \| Path, *, title: str \| None='Phase projections', max_points: int=5000) -> str` | Plot standard ``xy``, ``xz``, and ``yz`` projections. |
| `function` | `plot_time_series` | 102 | `public/module` | `plot_time_series(trajectory: np.ndarray, output_path: str \| Path, *, columns: Sequence[str \| int]=('x', 'y', 'z'), title: str \| None='Time series', max_points: int=6000) -> str` | Plot selected trajectory coordinates against time. |
| `function` | `plot_bifurcation_diagram` | 130 | `public/module` | `plot_bifurcation_diagram(points: Sequence[BifurcationPoint], output_path: str \| Path, *, parameter_label: str='parameter', observable_label: str='observable', title: str='Bifurcation diagram') -> str` | Plot extracted bifurcation points from a parameter scan. |
| `function` | `_state_column` | 156 | `internal` | `_state_column(dim: str \| int) -> int` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `plot_lure_nyquist_describing_function` | 165 | `public/module` | `plot_lure_nyquist_describing_function(system: LureSystem, seed: HarmonicSeed, output_path: str \| Path, *, q: float=1.0, method: str \| None=None, mu: float \| None=None, wmin: float=1e-05, wmax: float=50.0, amin: float=1.0 + 1e-08, amax: float \| None=None, title: str="Lur'e Nyquist/DF closure") -> str` | Plot ``W_q(i omega)`` and ``-1/N(A)`` for any Lur'e system. |
| `function` | `plot_lure_transfer_components` | 215 | `public/module` | `plot_lure_transfer_components(system: LureSystem, seed: HarmonicSeed, output_path: str \| Path, *, q: float=1.0, wmin: float=0.0001, wmax: float=10.2, nscan: int=5000, title: str \| None=None) -> str` | Plot real and imaginary transfer-function closure conditions. |
| `function` | `plot_integer_lure_continuation` | 277 | `public/module` | `plot_integer_lure_continuation(steps: Sequence[IntegerLureContinuationStep], output_path: str \| Path, *, dims: Sequence[int]=(0, 1, 2), title: str \| None="Integer Lur'e continuation", max_points: int=1500) -> str` | Plot epsilon-continuation trajectories for any integer Lur'e system. |
| `function` | `plot_fractional_continuation_phase_story` | 316 | `public/module` | `plot_fractional_continuation_phase_story(steps: Sequence[dict], output_path: str \| Path, *, final_trajectory: np.ndarray \| None=None, seed_effective: Sequence[float] \| None=None, continuation_final: Sequence[float] \| None=None, max_step_points: int=850, max_final_points: int=1800) -> str` | Plot fractional continuation trajectories directly in phase space. |
| `function` | `plot_phase_space_with_reference_points` | 390 | `public/module` | `plot_phase_space_with_reference_points(trajectory: np.ndarray, output_path: str \| Path, *, seed_effective: Sequence[float] \| None=None, continuation_final: Sequence[float] \| None=None, max_points: int=8000) -> str` | Plot a phase-space trajectory with standard seed/final markers. |
| `function` | `plot_integer_hiddenness_controls` | 423 | `public/module` | `plot_integer_hiddenness_controls(target_trajectory: np.ndarray, probes: Sequence[IntegerHiddennessProbe], output_path: str \| Path, *, dims: Sequence[int]=(0, 1, 2), title: str \| None='Integer hiddenness controls', max_target_points: int=2500, max_probe_points: int=180) -> str` | Plot target attractor and sampled equilibrium-neighborhood probes. |
| `function` | `plot_spectrum` | 459 | `public/module` | `plot_spectrum(spectrum: SpectrumResult, output_path: str \| Path, *, title: str \| None=None, x_units: str='rad/s', omega_marker: float \| None=None, marker_label: str \| None=None) -> str` | Plot one reusable FFT/PSD spectrum. |
| `function` | `plot_trajectory_spectra` | 491 | `public/module` | `plot_trajectory_spectra(trajectory: np.ndarray, output_dir: str \| Path, *, method: str='fft', components: Sequence[int] \| None=None, prefix: str='spectrum') -> list[str]` | Write one FFT/PSD figure per trajectory component. |
| `function` | `plot_lyapunov_convergence` | 510 | `public/module` | `plot_lyapunov_convergence(result: LyapunovResult, output_path: str \| Path) -> str` | Plot Lyapunov convergence curves for any dimension. |

### `hidden_attractors.plotting.export`

Source: `version_2/hidden_attractors/plotting/export.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `get_git_commit` | 13 | `public/module` | `get_git_commit()` | Returns the current git commit hash, or 'unknown' if not available. |
| `function` | `export_figure` | 30 | `public/module` | `export_figure(fig, figure_id, kind, metadata_dict, run_id='default_run', report_targets=None)` | Exports a figure to the canonical folder structure. |
| `function` | `intercept_and_export_path` | 115 | `public/module` | `intercept_and_export_path(fig, output_path, kind, metadata_dict=None)` | Helper to intercept savefig calls in older plotting scripts, formatting them, exporting them to the central library figures repository, updating manifests, and writing them back to the originally requested locations. |

### `hidden_attractors.plotting.generate_publication_figures`

Source: `version_2/hidden_attractors/plotting/generate_publication_figures.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `first_harmonic_reconstruction` | 29 | `public/module` | `first_harmonic_reconstruction(traj: np.ndarray, tail_fraction: float=0.85) -> np.ndarray` | Helper to perform first harmonic (linearized) reconstruction of the attractor trajectory. |
| `function` | `downsample` | 48 | `public/module` | `downsample(arr: np.ndarray, max_points: int) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `save_and_close` | 54 | `public/module` | `save_and_close(fig, path: Path)` | Saves the figure in both PNG and PDF formats and closes the plot. |
| `function` | `_read_json` | 68 | `internal` | `_read_json(path: Path) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_resolve_publication_input` | 72 | `internal` | `_resolve_publication_input(base: Path, value: str \| Path) -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_source_label` | 83 | `internal` | `_source_label(path: Path) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_load_csv_trajectory` | 93 | `internal` | `_load_csv_trajectory(path: Path) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_candidate_metadata` | 110 | `internal` | `_candidate_metadata(summary: dict[str, Any], manifest: dict[str, Any], figure_id: str, sources: list[Path], *, kind: str) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_export_candidate_figure` | 137 | `internal` | `_export_candidate_figure(fig: plt.Figure, figure_id: str, summary: dict[str, Any], manifest: dict[str, Any], sources: list[Path], *, kind: str) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_candidate_tail` | 157 | `internal` | `_candidate_tail(summary: dict[str, Any], times: np.ndarray, states: np.ndarray) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_prune_obsolete_report_figures` | 166 | `internal` | `_prune_obsolete_report_figures() -> None` | Remove report assets superseded by the consolidated publication suite. |
| `function` | `_plot_candidate_seed` | 188 | `internal` | `_plot_candidate_seed(summary: dict[str, Any], manifest: dict[str, Any], times: np.ndarray, states: np.ndarray, target_path: Path) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_plot_candidate_transfer` | 222 | `internal` | `_plot_candidate_transfer(summary: dict[str, Any], manifest: dict[str, Any], seed_path: Path) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_continuation_trajectories` | 276 | `internal` | `_continuation_trajectories(continuation_dir: Path) -> list[tuple[float, Path, np.ndarray]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_plot_candidate_linear_and_continuation` | 286 | `internal` | `_plot_candidate_linear_and_continuation(summary: dict[str, Any], manifest: dict[str, Any], times: np.ndarray, states: np.ndarray, target_path: Path, continuation_dir: Path) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_plot_candidate_dynamics` | 373 | `internal` | `_plot_candidate_dynamics(summary: dict[str, Any], manifest: dict[str, Any], times: np.ndarray, states: np.ndarray, target_path: Path) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_plot_candidate_fft` | 419 | `internal` | `_plot_candidate_fft(summary: dict[str, Any], manifest: dict[str, Any], times: np.ndarray, states: np.ndarray, target_path: Path, spectral_path: Path) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_plot_candidate_lyapunov` | 460 | `internal` | `_plot_candidate_lyapunov(summary: dict[str, Any], manifest: dict[str, Any], evidence_path: Path, convergence_path: Path) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_plot_candidate_spheres_and_heatmap` | 497 | `internal` | `_plot_candidate_spheres_and_heatmap(summary: dict[str, Any], manifest: dict[str, Any], matrix_path: Path) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_export_report_heatmap` | 609 | `internal` | `_export_report_heatmap(figure_id: str, records: list[dict[str, Any]], sources: list[Path], *, q: float, system_id: str, status: str, wide: bool=False) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `generate_comparison_report_heatmaps` | 645 | `public/module` | `generate_comparison_report_heatmaps() -> None` | Regenerate report heatmaps 24, 33 and 37 with one visual contract. |
| `function` | `generate_biased_report_dynamics` | 707 | `public/module` | `generate_biased_report_dynamics() -> None` | Regenerate the biased attractor as 3D-only and its normalized FFT. |
| `function` | `generate_centered_report_fft` | 810 | `public/module` | `generate_centered_report_fft() -> None` | Regenerate the centered control FFT with normalized amplitude and omega0. |
| `function` | `generate_report_comparison_assets` | 868 | `public/module` | `generate_report_comparison_assets() -> None` | Regenerate every shared DF/NC comparison asset used by the paper. |
| `function` | `generate_candidate_publication_figures` | 877 | `public/module` | `generate_candidate_publication_figures(candidate_dir: str \| Path) -> None` | Generate the complete c590-style paper suite from one candidate bundle. |
| `function` | `generate_all_publication_figures` | 927 | `public/module` | `generate_all_publication_figures(output_dir: str, config: Dict[str, Any]) -> None` | Core post-processor that parses raw data and configuration from a workflow run and produces vector PDF + high-resolution PNG figures. |
| `function` | `main` | 1364 | `public/module` | `main(argv: list[str] \| None=None) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |

### `hidden_attractors.plotting.generate_unified_report_figures`

Source: `version_2/hidden_attractors/plotting/generate_unified_report_figures.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `ReportFigure` | 41 | `public/module` | `class ReportFigure(object)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_target_subdir` | 108 | `internal` | `_target_subdir(suffix: str) -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_source_candidates` | 112 | `internal` | `_source_candidates(filename: str) -> Iterable[Path]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_write_nonlinearity_comparison` | 124 | `internal` | `_write_nonlinearity_comparison() -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_ensure_c590_publication_figures` | 154 | `internal` | `_ensure_c590_publication_figures() -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_sync_one` | 163 | `internal` | `_sync_one(figure: ReportFigure) -> dict[str, str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_latex_figure_references` | 179 | `internal` | `_latex_figure_references() -> set[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `generate_unified_report_figures` | 194 | `public/module` | `generate_unified_report_figures(*, verify_latex: bool=True) -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `main` | 230 | `public/module` | `main(argv: list[str] \| None=None) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |

### `hidden_attractors.plotting.lyapunov`

Source: `version_2/hidden_attractors/plotting/lyapunov.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `plot_lyapunov_convergence_styled` | 19 | `public/module` | `plot_lyapunov_convergence_styled(result: LyapunovResult, output_path: str \| Path, *, system_id: str='chua_fractional') -> str` | Plot styled Lyapunov convergence curves and export using centralized API. |

### `hidden_attractors.plotting.manifest`

Source: `version_2/hidden_attractors/plotting/manifest.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `load_manifest` | 10 | `public/module` | `load_manifest()` | Loads the JSON manifest from figure_manifest.json. |
| `function` | `save_manifest` | 23 | `public/module` | `save_manifest(entries)` | Saves the list of entries to figure_manifest.json and figure_manifest.csv. |
| `function` | `update_manifest` | 67 | `public/module` | `update_manifest(entry)` | Updates figure_manifest.json and figure_manifest.csv with the new entry. |

### `hidden_attractors.plotting.matignon`

Source: `version_2/hidden_attractors/plotting/matignon.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `classify_equilibrium_stability` | 17 | `public/module` | `classify_equilibrium_stability(system: ChaoticSystem, eq_point: np.ndarray, q: float, tol: float=1e-08) -> Dict[str, Any]` | Classify the local stability of an equilibrium point. |
| `function` | `plot_matignon_equilibria` | 70 | `public/module` | `plot_matignon_equilibria(system: ChaoticSystem, equilibria: Dict[str, np.ndarray], q: float, output_dir: str \| Path) -> str` | Renders the premium Matignon stability plane visualization for all equilibria. |

### `hidden_attractors.plotting.overlays`

Source: `version_2/hidden_attractors/plotting/overlays.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `plot_trajectory_overlay` | 18 | `public/module` | `plot_trajectory_overlay(trajectories: Sequence[np.ndarray], labels: Sequence[str], *, title: str, output_path: str \| Path, max_points: int=1500) -> str` | Plot superposed trajectories in 3D and coordinate projections. |

### `hidden_attractors.plotting.plot_basins`

Source: `version_2/hidden_attractors/plotting/plot_basins.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `plot_basin_slices` | 9 | `public/module` | `plot_basin_slices(basin_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], config: dict, output_dir: str) -> None` | Legacy wrapper for backward compatibility. |
| `function` | `plot_basin_slice_file` | 26 | `public/module` | `plot_basin_slice_file(plane: str, u: np.ndarray, v: np.ndarray, mat: np.ndarray, eq_name: str, config: dict, output_dir: str) -> str` | Renders and saves a high-quality 2D basin slice plot. |

### `hidden_attractors.plotting.plot_continuation`

Source: `version_2/hidden_attractors/plotting/plot_continuation.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `plot_continuation_eta` | 9 | `public/module` | `plot_continuation_eta(cont_steps: List[dict], config: dict, output_dir: str) -> None` | Generate and save numerical continuation plots: eta vs state norm, and eta vs oscillation amplitude. |
| `function` | `plot_continuation_first_last_comparison` | 73 | `public/module` | `plot_continuation_first_last_comparison(cont_steps: List[dict], config: dict, output_dir: str) -> None` | Compare the linearized attractor (first step, lambda=0.0) and the final nonlinear attractor (last step, lambda=1.0) in 3D and 2D overlays. |
| `function` | `plot_continuation_timeseries_comparison` | 141 | `public/module` | `plot_continuation_timeseries_comparison(cont_steps: List[dict], config: dict, output_dir: str) -> None` | Compare the time series of the first state variable x(t) between the first cycle (linearized) and last cycle. |
| `function` | `plot_continuation_progression` | 184 | `public/module` | `plot_continuation_progression(cont_steps: List[dict], config: dict, output_dir: str) -> None` | Plot progression of trajectories at each step of continuation, and trace the path followed by their initial conditions. |
| `function` | `plot_continuation_tracking` | 288 | `public/module` | `plot_continuation_tracking(cont_steps: List[dict], config: dict, output_dir: str) -> None` | Generate tracking plots for the continuation run. |

### `hidden_attractors.plotting.plot_df`

Source: `version_2/hidden_attractors/plotting/plot_df.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `plot_describing_function` | 9 | `public/module` | `plot_describing_function(system: Any, candidates: List[Tuple[float, float, float]], config: dict, output_dir: str) -> None` | Generate and save describing function plot N(A) vs A. |
| `function` | `plot_harmonic_residual_map` | 48 | `public/module` | `plot_harmonic_residual_map(system: Any, candidates: List[Tuple[float, float, float]], config: dict, output_dir: str) -> None` | Generate and save a 2D contour map of the harmonic residual \|1 + N(A)W(iw)\|. |

### `hidden_attractors.plotting.plot_matignon`

Source: `version_2/hidden_attractors/plotting/plot_matignon.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `plot_matignon_equilibria` | 9 | `public/module` | `plot_matignon_equilibria(system: Any, equilibria: Dict[str, np.ndarray], config: dict, output_dir: str) -> str` | Renders the premium Matignon stability plane visualization for all equilibria. |

### `hidden_attractors.plotting.plot_sphere_tests`

Source: `version_2/hidden_attractors/plotting/plot_sphere_tests.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `plot_sphere_test_results` | 9 | `public/module` | `plot_sphere_test_results(eq_name: str, eq_pt: np.ndarray, radius: float, probe_runs: List[Dict[str, Any]], output_dir: str, trajectory_plot_fraction: float=0.25, max_trajectories_to_plot: int=60) -> str` | Renders a premium 3D visualization of neighborhood sphere probes. |

### `hidden_attractors.plotting.plot_trajectories`

Source: `version_2/hidden_attractors/plotting/plot_trajectories.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `plot_attractor_trajectories` | 10 | `public/module` | `plot_attractor_trajectories(trajectory: np.ndarray, equilibria: Dict[str, np.ndarray], config: dict, output_dir: str) -> None` | Generate and save 3D attractor phase space and 2D projection plots. |
| `function` | `plot_flexible_attractor_and_projections` | 25 | `public/module` | `plot_flexible_attractor_and_projections(trajectory: np.ndarray, equilibria: Dict[str, np.ndarray], config: dict, output_dir: str, file_prefix: str) -> None` | Saves a 3D attractor plot and three individual 2D projections (xy, xz, yz) into the designated output directory under the specified prefix. |
| `function` | `plot_timeseries_data` | 116 | `public/module` | `plot_timeseries_data(trajectory: np.ndarray, config: dict, output_dir: str, file_prefix: str) -> None` | Generate time series plots (x, y, z, and combined xyz) and save states as CSV. |
| `function` | `plot_neighborhood_control_spheres` | 184 | `public/module` | `plot_neighborhood_control_spheres(target_trajectory: np.ndarray, probe_results: list, equilibria: Dict[str, np.ndarray], config: dict, output_dir: str, max_target_points: int=5000, max_probe_points: int=500) -> None` | Generate a 3D control spheres plot showing target attractor and probes. |

### `hidden_attractors.plotting.plot_transfer`

Source: `version_2/hidden_attractors/plotting/plot_transfer.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `plot_nyquist_transfer` | 8 | `public/module` | `plot_nyquist_transfer(omega_grid: np.ndarray, w_vals: np.ndarray, candidates: List[Tuple[float, float, float]], config: dict, output_dir: str) -> None` | Generate and save Nyquist plot and real/imag component plots of the transfer function. |

### `hidden_attractors.plotting.registry`

Source: `version_2/hidden_attractors/plotting/registry.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `register_figure` | 3 | `public/module` | `register_figure(figure_id, renderer_fn, kind)` | Registers a figure renderer function. |
| `function` | `get_registered_figures` | 12 | `public/module` | `get_registered_figures()` | Returns the dictionary of all registered figures. |

### `hidden_attractors.plotting.render_all`

Source: `version_2/hidden_attractors/plotting/render_all.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `render_all_plots` | 8 | `public/module` | `render_all_plots(trajectory=None, equilibria=None, basin_grid=None, grid_x=None, grid_y=None, freqs=None, w_evals=None, n_evals=None, candidates=None, eigenvalues=None, config=None, run_id='default_run', report_targets=None)` | Programmatic entry point to render and export all available figures. |

### `hidden_attractors.plotting.renderers`

Source: `version_2/hidden_attractors/plotting/renderers.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `render_attractor` | 9 | `public/module` | `render_attractor(trajectory: np.ndarray, equilibria: Dict[str, np.ndarray], config: dict, run_id: str='default_run', report_targets: List[str]=None) -> Dict[str, str]` | Renders 3D phase space and 2D projections of the attractor. |
| `function` | `render_basin` | 146 | `public/module` | `render_basin(grid_x: np.ndarray, grid_y: np.ndarray, basin_grid: np.ndarray, config: dict, run_id: str='default_run', report_targets: List[str]=None) -> Tuple[str, str]` | Renders the basin of attraction map. |
| `function` | `render_nyquist` | 225 | `public/module` | `render_nyquist(freqs: np.ndarray, w_evals: np.ndarray, n_evals: np.ndarray, candidates: List[Tuple[float, float, float]]=None, config: dict=None, run_id: str='default_run', report_targets: List[str]=None) -> Tuple[str, str]` | Renders describing function / Nyquist plots distinguishing: - W_q(j\omega) transfer function curve - Critical point -1 / N(A) locus - Candidate crossings Saves JSON containing q, omega, lambda, A, N(A), residual, and transfer mode. |
| `function` | `render_matignon` | 324 | `public/module` | `render_matignon(eigenvalues: np.ndarray, q: float, config: dict, run_id: str='default_run', report_targets: List[str]=None) -> Tuple[str, str]` | Renders the Matignon stability complex plane. |

### `hidden_attractors.plotting.style`

Source: `version_2/hidden_attractors/plotting/style.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `apply_library_style` | 4 | `public/module` | `apply_library_style()` | Applies the library's unified visual style rules globally via rcParams. |
| `function` | `apply_axes_style` | 30 | `public/module` | `apply_axes_style(ax, grid=False, is_3d=False)` | Applies style adjustments to an individual axes object. |
| `function` | `get_figsize` | 53 | `public/module` | `get_figsize(kind)` | Returns standard figure size for the given plot kind. |

### `hidden_attractors.plotting.zero_one`

Source: `version_2/hidden_attractors/plotting/zero_one.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `plot_zero_one_phase_styled` | 18 | `public/module` | `plot_zero_one_phase_styled(signal: np.ndarray, c_value: float, output_path: str \| Path, *, system_id: str='chua_fractional') -> str` | Plot p_c vs q_c trajectory to illustrate regular vs chaotic dynamics. |

### `hidden_attractors.protocol_cli`

Source: `version_2/hidden_attractors/protocol_cli.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_read_object` | 47 | `internal` | `_read_object(path: Path \| None) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_numerical_contract` | 56 | `internal` | `_numerical_contract(payload: dict[str, Any]) -> NumericalContract` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_backend_name` | 74 | `internal` | `_backend_name(backend: str) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_run_metadata` | 83 | `internal` | `_run_metadata(args: argparse.Namespace, contract: NumericalContract, data: dict[str, Any]) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_stage_specific_payload` | 127 | `internal` | `_stage_specific_payload(args: argparse.Namespace, data: dict[str, Any], run_metadata: dict[str, Any], contract: NumericalContract) -> tuple[dict[str, Any], dict[str, Any], str \| None]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `make_parser` | 195 | `public/module` | `make_parser() -> argparse.ArgumentParser` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `main` | 210 | `public/module` | `main(argv: Sequence[str] \| None=None) -> int` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |

### `hidden_attractors.references.claims`

Source: `version_2/hidden_attractors/references/claims.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `ClaimType` | 5 | `public/module` | `class ClaimType(str, Enum)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |

### `hidden_attractors.references.validator`

Source: `version_2/hidden_attractors/references/validator.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `validate_claim_references` | 27 | `public/module` | `validate_claim_references(claims: List[Dict[str, Any]], strict: bool=True) -> Dict[str, Any]` | Validate project claims against the REFERENCE_REGISTRY and CLAIM_REFERENCE_MATRIX. |
| `function` | `write_traceability_matrix_markdown` | 164 | `public/module` | `write_traceability_matrix_markdown(result: Dict[str, Any], path: Union[str, Path]) -> None` | Generate and write a clean markdown traceability matrix file. |
| `function` | `validate_bibliography_manifest` | 231 | `public/module` | `validate_bibliography_manifest(manifest_path: Union[str, Path], strict: bool=True) -> Dict[str, Any]` | Load and validate the bibliography manifest file. |

### `hidden_attractors.reproducibility`

Source: `version_2/hidden_attractors/reproducibility.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `LureMetadata` | 40 | `public/module` | `class LureMetadata(object)` | Lure decomposition used to construct a seed. |
| `class` | `SeedMetadata` | 52 | `public/module` | `class SeedMetadata(object)` | Seed identity and construction data. |
| `class` | `NumericalMetadata` | 63 | `public/module` | `class NumericalMetadata(object)` | Numerical contract needed to reproduce one integration. |
| `class` | `ContinuationMetadata` | 75 | `public/module` | `class ContinuationMetadata(object)` | Continuation path and Caputo-memory propagation policy. |
| `class` | `ToleranceMetadata` | 86 | `public/module` | `class ToleranceMetadata(object)` | Numerical tolerances used by decision gates. |
| `class` | `SoftwareMetadata` | 101 | `public/module` | `class SoftwareMetadata(object)` | Software provenance for one integration. |
| `class` | `RunMetadata` | 115 | `public/module` | `class RunMetadata(object)` | Complete reproducibility envelope for a numerical run. |
| `function` | `_git_output` | 136 | `internal` | `_git_output(*args: str) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_dirty_tree_sha256` | 148 | `internal` | `_dirty_tree_sha256(status: str) -> str` | Hash tracked diffs plus untracked file contents for source-tree audit. |
| `function` | `_package_version` | 176 | `internal` | `_package_version() -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `collect_software_metadata` | 183 | `public/module` | `collect_software_metadata() -> SoftwareMetadata` | Collect software provenance without requiring an installed package. |
| `function` | `collect_lure_metadata` | 202 | `public/module` | `collect_lure_metadata(lure: Any, *, transfer_convention: str, harmonic_condition: str) -> LureMetadata` | Serialise a maintained ``LureSystem`` without serialising callbacks. |
| `function` | `collect_seed_metadata` | 222 | `public/module` | `collect_seed_metadata(seed: Mapping[str, Any] \| None, *, source: str) -> SeedMetadata \| None` | Normalise a workflow seed record into the shared metadata schema. |
| `function` | `_normalise_memory_mode` | 247 | `internal` | `_normalise_memory_mode(mode: str) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `collect_run_metadata` | 259 | `public/module` | `collect_run_metadata(*, run_id: str, workflow: str, system: str, q: float, h: float, t_final: float, t_burn: float, memory_mode: str, integrator_name: str, integrator_backend: str, caputo: bool, M: int \| None=None, memory_window_steps: int \| None=None, memory_window_time: float \| None=None, is_full_caputo: bool \| None=None, parameters: Mapping[str, Any] \| None=None, lure: LureMetadata \| Mapping[str, Any] \| None=None, seed: SeedMetadata \| Mapping[str, Any] \| None=None, random_seed: int \| None=None, random_seed_policy: str='not_applicable', provenance: Mapping[str, Any] \| None=None, extra: Mapping[str, Any] \| None=None, continuation: ContinuationMetadata \| Mapping[str, Any] \| None=None, tolerances: ToleranceMetadata \| Mapping[str, Any] \| None=None) -> RunMetadata` | Build the common metadata envelope used by maintained workflows. |
| `function` | `metadata_to_jsonable` | 336 | `public/module` | `metadata_to_jsonable(value: Any) -> Any` | Convert dataclasses, numpy values and paths to JSON-safe values. |
| `function` | `_is_finite_number` | 354 | `internal` | `_is_finite_number(value: Any) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_require_mapping` | 358 | `internal` | `_require_mapping(metadata: Mapping[str, Any], key: str, errors: list[str]) -> Mapping[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `extract_run_metadata` | 370 | `public/module` | `extract_run_metadata(container: Mapping[str, Any] \| None) -> dict[str, Any] \| None` | Read either metadata alias and return the canonical JSON-ready payload. |
| `function` | `validate_run_metadata` | 383 | `public/module` | `validate_run_metadata(metadata: dict[str, Any]) -> list[str]` | Return every missing or malformed base reproducibility field. |
| `function` | `validate_hiddenness_promotion_metadata` | 474 | `public/module` | `validate_hiddenness_promotion_metadata(metadata: dict[str, Any] \| None) -> list[str]` | Validate metadata required for a strong sampled-neighborhood promotion. |
| `function` | `write_run_metadata` | 513 | `public/module` | `write_run_metadata(path: str \| Path, metadata: RunMetadata \| Mapping[str, Any]) -> dict[str, Any]` | Write an auditable JSON metadata file and return its serialised payload. |

### `hidden_attractors.seed_generation.chua`

Source: `version_2/hidden_attractors/seed_generation/chua.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `chua_gain` | 51 | `public/module` | `chua_gain(params: ChuaParameters) -> float` | Return the saturation gain ``m0 - m1`` for the non-smooth Chua model. |
| `function` | `chua_matrices` | 68 | `public/module` | `chua_matrices(params: ChuaParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]` | Return the Lur'e decomposition matrices ``(P, b, c)`` for Chua. |
| `function` | `psi_sigma` | 105 | `public/module` | `psi_sigma(sigma: float, params: ChuaParameters) -> float` | Evaluate the nonlinear residual ``ψ(σ)`` used by the Lur'e form. |
| `function` | `build_linearized_matrix` | 127 | `public/module` | `build_linearized_matrix(params: ChuaParameters, gain: float) -> np.ndarray` | Return the linearised matrix ``P + k b c^T`` for describing-function gain ``k``. |
| `function` | `transfer_function` | 149 | `public/module` | `transfer_function(omega: float, q: float, params: ChuaParameters) -> complex` | Return the fractional Chua transfer function ``W_q(iω) = c^T (P - (iω)^q I)^{-1} b``. |
| `function` | `find_omega_gain_candidates` | 184 | `public/module` | `find_omega_gain_candidates(q: float, params: ChuaParameters \| None=None, *, wmin: float=0.0001, wmax: float=10.0, nscan: int=20000, compatible_only: bool=True) -> list[tuple[float, float]]` | Find all ``(ω, k)`` pairs where ``Im(W_q(iω)) = 0``. |
| `function` | `is_describing_gain_compatible` | 280 | `public/module` | `is_describing_gain_compatible(gain: float, params: ChuaParameters) -> bool` | Return whether *gain* is reachable by the classical describing function. |
| `function` | `describing_function` | 303 | `public/module` | `describing_function(amplitude: float, params: ChuaParameters) -> float` | Classical first-harmonic describing function ``N(A)``. |
| `function` | `solve_amplitude_from_gain` | 350 | `public/module` | `solve_amplitude_from_gain(gain: float, params: ChuaParameters, *, amin: float=1.0 + 1e-09, amax: float=100.0, nscan: int=20000) -> float` | Solve ``N(A) = gain`` for the classical describing function. |
| `function` | `machado_describing_function` | 405 | `public/module` | `machado_describing_function(amplitude: float, params: ChuaParameters, mu: float) -> float` | Auxiliary Machado-family describing function ``N_μ(A) = N(A)^μ``. |
| `function` | `solve_machado_amplitude_from_gain` | 446 | `public/module` | `solve_machado_amplitude_from_gain(gain: float, params: ChuaParameters, mu: float, *, amin: float=1.0 + 1e-09, amax: float=100.0, nscan: int=20000) -> float` | Solve ``N(A)^μ = gain`` for the auxiliary Machado family. |
| `function` | `fourier_coefficients_psi` | 509 | `public/module` | `fourier_coefficients_psi(amplitude: float, sigma0: float, params: ChuaParameters, *, harmonics: int=10, n_quad: int=4096) -> dict[str, object]` | Compute Fourier coefficients of ``ψ(σ0 + A cos(θ))``. |
| `function` | `biased_describing_function` | 582 | `public/module` | `biased_describing_function(amplitude: float, sigma0: float, params: ChuaParameters, *, harmonics: int=10, n_quad: int=4096) -> complex` | Return the biased describing function ``N(A, σ0) = Y_1(A, σ0) / A``. |
| `function` | `reconstruct_biased_lure_seed` | 618 | `public/module` | `reconstruct_biased_lure_seed(*, q: float, params: ChuaParameters \| None=None, amplitude: float, sigma0: float, omega: float, theta: float=0.0, harmonics: int=10, n_quad: int=4096) -> BiasedHarmonicSeed` | Reconstruct a biased Lur'e seed from DC and first-harmonic balance equations. |
| `function` | `build_fractional_seed` | 704 | `public/module` | `build_fractional_seed(q: float, params: ChuaParameters, omega: float, gain: float, amplitude: float, *, theta: float=0.0) -> tuple[np.ndarray, np.ndarray, complex]` | Build the harmonic initial condition for one DF branch. |
| `function` | `find_harmonic_seed` | 762 | `public/module` | `find_harmonic_seed(*, q: float, params: ChuaParameters \| None=None, branch_index: int=0, method: Literal['classic', 'machado']='classic', mu: float=1.0, theta: float=0.0, wmin: float=0.0001, wmax: float=10.0, nscan: int=20000) -> HarmonicSeed` | Locate a DF branch and return a finite harmonic seed. |
| `function` | `format_seed_report` | 855 | `public/module` | `format_seed_report(seed: HarmonicSeed) -> dict[str, object]` | Return a JSON-serialisable summary of a harmonic seed. |

### `hidden_attractors.seed_generation.chua_arctan_wu2023`

Source: `version_2/hidden_attractors/seed_generation/chua_arctan_wu2023.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_normalise_transfer_mode` | 42 | `internal` | `_normalise_transfer_mode(mode: TransferMode \| str) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `transfer_function_arctan_wu2023` | 54 | `public/module` | `transfer_function_arctan_wu2023(omega: float, q: float=0.99, params: ChuaParameters \| None=None, transfer_mode: TransferMode \| str='published_integer_laplace') -> complex` | Return the Wu2023 arctan Chua transfer value. |
| `function` | `find_centered_arctan_wu2023_branches` | 81 | `public/module` | `find_centered_arctan_wu2023_branches(*, q: float=0.99, params: ChuaParameters \| None=None, wmin: float=0.0001, wmax: float=10.0, nscan: int=20000, transfer_mode: TransferMode \| str='published_integer_laplace') -> list[HarmonicSeed]` | Return admissible centered classical-DF branches for the Wu2023 model. |
| `function` | `_complex_pair` | 157 | `internal` | `_complex_pair(value: complex) -> list[float]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `format_arctan_wu2023_seed_report` | 161 | `public/module` | `format_arctan_wu2023_seed_report(*, q: float=0.99, params: ChuaParameters \| None=None, nscan: int=20000, transfer_mode: TransferMode \| str='published_integer_laplace') -> dict[str, object]` | Create JSON-ready branch evidence for the selected Wu2023 seed mode. |

### `hidden_attractors.seed_generation.core`

Source: `version_2/hidden_attractors/seed_generation/core.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `HarmonicSeed` | 31 | `public/module` | `class HarmonicSeed(object)` | Numerical seed produced by the describing-function construction. |
| `class` | `BiasedHarmonicSeed` | 69 | `public/module` | `class BiasedHarmonicSeed(object)` | Seed reconstructed from a biased first-harmonic approximation. |
| `function` | `validate_fractional_order` | 104 | `public/module` | `validate_fractional_order(q: float) -> float` | Validate a Caputo fractional order and return it as a Python float. |
| `function` | `fractional_iomega_power` | 140 | `public/module` | `fractional_iomega_power(omega: float, q: float) -> complex` | Return ``(i ω)^q`` evaluated on the principal branch. |
| `function` | `_bisect_root` | 179 | `internal` | `_bisect_root(func, left: float, right: float, *, maxiter: int=100, xtol: float=1e-12) -> float` | Small dependency-free scalar bisection helper. |
| `function` | `_solve_scalar_gain` | 213 | `internal` | `_solve_scalar_gain(target_gain: float, evaluator, *, amin: float, amax: float, nscan: int) -> float` | Grid + bisection solver for ``evaluator(a) == target_gain``. |

### `hidden_attractors.seed_generation.lure`

Source: `version_2/hidden_attractors/seed_generation/lure.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `build_lure_linearized_matrix` | 49 | `public/module` | `build_lure_linearized_matrix(system: LureSystem, gain: float) -> np.ndarray` | Return the linearised matrix ``A + k b c^T`` for a Lur'e DF gain ``k``. |
| `function` | `lure_transfer_function` | 82 | `public/module` | `lure_transfer_function(omega: float, q: float, system: LureSystem) -> complex` | Return ``c^T (P - (iω)^q I)^{-1} b`` for a generic Lur'e system. |
| `function` | `find_lure_omega_gain_candidates` | 133 | `public/module` | `find_lure_omega_gain_candidates(q: float, system: LureSystem, *, wmin: float=0.0001, wmax: float=10.0, nscan: int=20000, compatible_only: bool=True) -> list[tuple[float, float]]` | Find all ``(ω, k)`` pairs where ``Im(W_q(iω)) = 0`` for a Lur'e system. |
| `function` | `lure_describing_function` | 220 | `public/module` | `lure_describing_function(amplitude: float, system: LureSystem) -> complex` | Evaluate the classical first-harmonic describing function for *system*. |
| `function` | `lure_machado_describing_function` | 247 | `public/module` | `lure_machado_describing_function(amplitude: float, system: LureSystem, mu: float) -> float` | Return the real Machado-family describing function ``N_μ(A) = N(A)^μ``. |
| `function` | `solve_lure_amplitude_from_gain` | 287 | `public/module` | `solve_lure_amplitude_from_gain(gain: float, system: LureSystem, *, method: Literal['classic', 'machado']='classic', mu: float=1.0, amin: float=1.0 + 1e-09, amax: float=100.0, nscan: int=20000) -> float` | Solve the amplitude relation ``N(A) = gain`` for a generic Lur'e system. |
| `function` | `fourier_coefficients_lure` | 347 | `public/module` | `fourier_coefficients_lure(amplitude: float, sigma0: float, system: LureSystem, *, harmonics: int=10, n_quad: int=4096) -> dict[str, object]` | Compute Fourier coefficients of ``ψ(σ0 + A cos(θ))`` for a Lur'e system. |
| `function` | `biased_lure_describing_function` | 413 | `public/module` | `biased_lure_describing_function(amplitude: float, sigma0: float, system: LureSystem, *, harmonics: int=10, n_quad: int=4096) -> complex` | Return the biased Lur'e describing function ``N(A, σ0) = Y_1(A, σ0) / A``. |
| `function` | `reconstruct_biased_lure_seed_from_system` | 449 | `public/module` | `reconstruct_biased_lure_seed_from_system(*, q: float, system: LureSystem, amplitude: float, sigma0: float, omega: float, theta: float=0.0, harmonics: int=10, n_quad: int=4096) -> BiasedHarmonicSeed` | Reconstruct a biased Lur'e seed from DC and first-harmonic balance equations. |
| `function` | `build_lure_fractional_seed` | 533 | `public/module` | `build_lure_fractional_seed(q: float, system: LureSystem, omega: float, gain: float, amplitude: float, *, theta: float=0.0) -> tuple[np.ndarray, np.ndarray, complex]` | Build the harmonic initial state for one Lur'e DF branch. |
| `function` | `find_lure_harmonic_seed` | 588 | `public/module` | `find_lure_harmonic_seed(*, q: float, system: LureSystem, branch_index: int=0, method: Literal['classic', 'machado']='classic', mu: float=1.0, theta: float=0.0, wmin: float=0.0001, wmax: float=10.0, nscan: int=20000) -> HarmonicSeed` | Locate a Lur'e DF branch and return a finite harmonic seed. |

### `hidden_attractors.solvers.efork_published`

Source: `version_2/hidden_attractors/solvers/efork_published.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `EFORK3Coefficients` | 18 | `public/module` | `class EFORK3Coefficients(object)` | Coefficients of the explicit three-stage fractional RK method. |
| `function` | `efork3_coefficients` | 32 | `public/module` | `efork3_coefficients(alpha: float) -> EFORK3Coefficients` | Return three-stage EFORK coefficients for ``0 < alpha < 1``. |
| `function` | `_history_term` | 55 | `internal` | `_history_term(t_eval: float, times: np.ndarray, states: np.ndarray, n: int, alpha: float, h: float) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `efork3_caputo_integrate` | 70 | `public/module` | `efork3_caputo_integrate(rhs: Callable[[float, np.ndarray], np.ndarray], y0: np.ndarray, *, alpha: float, h: float, t_final: float) -> tuple[np.ndarray, np.ndarray]` | Integrate a Caputo problem using the published EFORK-3 formula. |

### `hidden_attractors.solvers.history`

Source: `version_2/hidden_attractors/solvers/history.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `FractionalHistory` | 12 | `public/module` | `class FractionalHistory(object)` | Discrete finite-memory window transported between continuation stages. |
| `method` | `FractionalHistory.memory_points` | 23 | `public/module` | `memory_points(self) -> int` | Number of stored samples in the history window. |
| `method` | `FractionalHistory.dimension` | 29 | `public/module` | `dimension(self) -> int` | State dimension stored in the history window. |
| `method` | `FractionalHistory.as_efork_history` | 34 | `public/module` | `as_efork_history(self) -> np.ndarray` | Return columns ``t,state...`` accepted by continuation wrappers. |
| `method` | `FractionalHistory.from_trajectory` | 40 | `public/module` | `from_trajectory(cls, trajectory: np.ndarray, *, q: float, h: float, memory_length: float, rhs: Callable[[np.ndarray], np.ndarray] \| None=None) -> 'FractionalHistory'` | Extract the last ``ceil(memory_length / h)+1`` samples. |

### `hidden_attractors.solvers.integer`

Source: `version_2/hidden_attractors/solvers/integer.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `efork_q1_step` | 19 | `public/module` | `efork_q1_step(rhs: Callable[[np.ndarray], np.ndarray], state: np.ndarray, h: float) -> np.ndarray` | Advance one integer-order step with the q=1 EFORK-3 coefficients. |
| `function` | `efork_q1_integrate` | 35 | `public/module` | `efork_q1_integrate(rhs: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, *, t_final: float, h: float, div_threshold: float \| None=None) -> tuple[np.ndarray, str]` | Integrate an integer-order trajectory with columns ``t,state...``. |

### `hidden_attractors.systems.base`

Source: `version_2/hidden_attractors/systems/base.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `ChaoticSystem` | 27 | `public/module` | `class ChaoticSystem(object)` | Definition of a dynamical system that can enter package workflows. |
| `method` | `ChaoticSystem.evaluate` | 75 | `public/module` | `evaluate(self, state: State, parameters: Mapping[str, Any] \| None=None) -> np.ndarray` | Evaluate the system vector field at *state*. |
| `method` | `ChaoticSystem.equilibrium_points` | 116 | `public/module` | `equilibrium_points(self, parameters: Mapping[str, Any] \| None=None) -> dict[str, np.ndarray]` | Return known equilibria for the system. |
| `method` | `ChaoticSystem.jacobian_matrix` | 145 | `public/module` | `jacobian_matrix(self, state: State, parameters: Mapping[str, Any] \| None=None) -> np.ndarray` | Evaluate the analytic Jacobian at *state*. |
| `class` | `SystemRegistry` | 180 | `public/module` | `class SystemRegistry(object)` | Mutable registry for built-in and user-defined chaotic systems. |
| `method` | `SystemRegistry.__init__` | 183 | `dunder` | `__init__(self) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `SystemRegistry.normalize_name` | 187 | `public/module` | `normalize_name(name: str) -> str` | Return a lower-case, stripped, underscore-free registry key. |
| `method` | `SystemRegistry.register` | 191 | `public/module` | `register(self, system: ChaoticSystem, *, replace: bool=False) -> ChaoticSystem` | Add *system* to the registry. |
| `method` | `SystemRegistry.get` | 218 | `public/module` | `get(self, name: str) -> ChaoticSystem` | Return a registered system by name. |
| `method` | `SystemRegistry.list_names` | 242 | `public/module` | `list_names(self) -> list[str]` | Return all registered system names in alphabetical order. |
| `method` | `SystemRegistry.values` | 246 | `public/module` | `values(self) -> list[ChaoticSystem]` | Return all registered systems in alphabetical name order. |
| `function` | `register_system` | 262 | `public/module` | `register_system(system: ChaoticSystem, *, replace: bool=False) -> ChaoticSystem` | Register a built-in or user-defined chaotic system. |
| `function` | `get_system` | 295 | `public/module` | `get_system(name: str) -> ChaoticSystem` | Return a registered system by name. |
| `function` | `list_systems` | 323 | `public/module` | `list_systems() -> list[str]` | Return all registered system names in alphabetical order. |

### `hidden_attractors.systems.builtins`

Source: `version_2/hidden_attractors/systems/builtins.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_bisect_root` | 21 | `internal` | `_bisect_root(func, left: float, right: float, *, maxiter: int=100, xtol: float=1e-12) -> float` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_filter_chua_params` | 49 | `internal` | `_filter_chua_params(parameters: Mapping[str, Any]) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_chua_rhs` | 53 | `internal` | `_chua_rhs(state: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_chua_equilibria` | 57 | `internal` | `_chua_equilibria(parameters: Mapping[str, Any]) -> dict[str, np.ndarray]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_chua_jacobian` | 64 | `internal` | `_chua_jacobian(state: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_chua_lure_system` | 80 | `internal` | `_chua_lure_system(parameters: Mapping[str, Any]) -> LureSystem` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `chua_system` | 169 | `public/module` | `chua_system(model: str='nonsmooth') -> ChaoticSystem` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `chua_arctan_wu2023_system` | 212 | `public/module` | `chua_arctan_wu2023_system() -> ChaoticSystem` | Return the named registration for the official Wu et al. |
| `function` | `register_builtin_systems` | 242 | `public/module` | `register_builtin_systems() -> None` | Register built-in systems, replacing stale registrations if reloaded. |

### `hidden_attractors.systems.fischer_benchmarks`

Source: `version_2/hidden_attractors/systems/fischer_benchmarks.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_parameters` | 31 | `internal` | `_parameters(system_id: str, overrides: Mapping[str, float] \| None) -> dict[str, float]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `jerk_rhs` | 38 | `public/module` | `jerk_rhs(state: np.ndarray, parameters: Mapping[str, float] \| None=None) -> np.ndarray` | Jerk benchmark with exponential diode nonlinearity. |
| `function` | `financial_rhs` | 47 | `public/module` | `financial_rhs(state: np.ndarray, parameters: Mapping[str, float] \| None=None) -> np.ndarray` | Nonsmooth financial benchmark; no Jacobian is required. |
| `function` | `four_wing_rhs` | 55 | `public/module` | `four_wing_rhs(state: np.ndarray, parameters: Mapping[str, float] \| None=None) -> np.ndarray` | Four-wing benchmark system. |
| `function` | `get_fischer_benchmark` | 63 | `public/module` | `get_fischer_benchmark(system_id: str) -> tuple[Callable, np.ndarray, dict[str, object]]` | Return RHS, initial state, and metadata for one Fischer benchmark. |

### `hidden_attractors.systems.lure`

Source: `version_2/hidden_attractors/systems/lure.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `LureSystem` | 24 | `public/module` | `class LureSystem(object)` | Representation of the fractional Lur'e system ``D^q x = Ax + b·ψ(c^T x)``. |
| `method` | `LureSystem.__post_init__` | 81 | `dunder` | `__post_init__(self) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `LureSystem.dimension` | 97 | `public/module` | `dimension(self) -> int` | State dimension ``n`` of the Lur'e representation. |
| `method` | `LureSystem.sigma` | 108 | `public/module` | `sigma(self, state: np.ndarray) -> float` | Return the scalar feedback coordinate ``σ = c^T x``. |
| `method` | `LureSystem.evaluate` | 132 | `public/module` | `evaluate(self, state: np.ndarray) -> np.ndarray` | Evaluate the Lur'e vector field ``Ax + b·ψ(c^T x)``. |
| `method` | `LureSystem.is_gain_compatible` | 150 | `public/module` | `is_gain_compatible(self, gain: float) -> bool` | Return whether *gain* can be produced by the describing-function model. |
| `method` | `LureSystem.solve_amplitude` | 169 | `public/module` | `solve_amplitude(self, gain: float) -> float` | Solve the classical describing-function amplitude relation for *gain*. |

### `hidden_attractors.systems.requirements`

Source: `version_2/hidden_attractors/systems/requirements.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `Requirement` | 39 | `public/module` | `class Requirement(object)` | One explicit input required by a workflow. |
| `class` | `CapabilityReport` | 63 | `public/module` | `class CapabilityReport(object)` | Result of checking a registered system against workflow requirements. |
| `method` | `CapabilityReport.as_lines` | 72 | `public/module` | `as_lines(self) -> list[str]` | Return a human-readable report for CLI output. |
| `function` | `requirements_for` | 277 | `public/module` | `requirements_for(workflow: WorkflowName) -> tuple[Requirement, ...]` | Return documented requirements for one workflow. |
| `function` | `known_workflows` | 307 | `public/module` | `known_workflows() -> tuple[WorkflowName, ...]` | Return workflow names understood by the capability checker. |
| `function` | `check_system_capability` | 325 | `public/module` | `check_system_capability(system: ChaoticSystem, workflow: WorkflowName) -> CapabilityReport` | Check which package-level hooks a registered system provides for *workflow*. |

### `hidden_attractors.validation.chua_arctan_wu2023`

Source: `version_2/hidden_attractors/validation/chua_arctan_wu2023.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_complex_pair` | 53 | `internal` | `_complex_pair(value: complex) -> list[float]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_sorted_eigenvalues` | 57 | `internal` | `_sorted_eigenvalues(values: np.ndarray) -> list[complex]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_finite_difference_jacobian` | 61 | `internal` | `_finite_difference_jacobian(state: np.ndarray, step: float=1e-07) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `build_algebra_validation` | 71 | `public/module` | `build_algebra_validation() -> dict[str, Any]` | Return machine-readable equilibria, Jacobian, eigenvalue and Matignon evidence. |
| `function` | `write_algebra_validation` | 151 | `public/module` | `write_algebra_validation(path: Path=DEFAULT_ALGEBRA_OUTPUT) -> dict[str, Any]` | Write the algebra report JSON and return the same dictionary. |
| `function` | `main` | 160 | `public/module` | `main(argv: Sequence[str] \| None=None) -> int` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |

### `hidden_attractors.validation.manifest`

Source: `version_2/hidden_attractors/validation/manifest.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_read_json` | 25 | `internal` | `_read_json(path: Path) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_git_provenance` | 33 | `internal` | `_git_provenance(project_root: Path) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_package_version` | 50 | `internal` | `_package_version() -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `regenerate_validation_manifest` | 57 | `public/module` | `regenerate_validation_manifest(validation_root: Path=DEFAULT_VALIDATION_ROOT, *, contract_path: Path=DEFAULT_CONTRACT, validation_id: str='chua_fractional_validation_evidence', provenance: dict[str, Any] \| None=None, main_system: str='fractional nonsmooth Chua', main_parameters: dict[str, Any] \| None=None) -> dict[str, Any]` | Write ``00_manifest`` using only official summary files as stage truth. |

### `hidden_attractors.validation.nonsmooth`

Source: `version_2/hidden_attractors/validation/nonsmooth.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `NonSmoothNonlinearityValidator` | 12 | `public/module` | `class NonSmoothNonlinearityValidator(object)` | Analyze continuity, Lipschitz property, switching crossings, and regional stability. |
| `method` | `NonSmoothNonlinearityValidator.analyze_nonlinearity` | 16 | `public/module` | `analyze_nonlinearity(system: ChaoticSystem) -> dict[str, Any]` | Analyze system nonlinearity type and continuity properties. |
| `method` | `NonSmoothNonlinearityValidator.jacobian_region` | 56 | `public/module` | `jacobian_region(system: ChaoticSystem, X: np.ndarray) -> np.ndarray` | Evaluate the Jacobian matrix matching the region of state X. |
| `method` | `NonSmoothNonlinearityValidator.detect_switching_crossings` | 87 | `public/module` | `detect_switching_crossings(trajectory: np.ndarray, system: ChaoticSystem) -> dict[str, Any]` | Detect when the trajectory crosses switching surfaces. |
| `method` | `NonSmoothNonlinearityValidator.validate_equilibrium_stability` | 130 | `public/module` | `validate_equilibrium_stability(system: ChaoticSystem, equilibrium: np.ndarray, q: float) -> str` | Validate stability of equilibrium using regional eigenvalues and Matignon criteria. |

### `hidden_attractors.validation.states`

Source: `version_2/hidden_attractors/validation/states.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `AttractorValidationState` | 8 | `public/module` | `class AttractorValidationState(str, Enum)` | Enumeration of strict validation pipeline states. |

### `hidden_attractors.validation.symmetry`

Source: `version_2/hidden_attractors/validation/symmetry.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `SymmetryValidator` | 12 | `public/module` | `class SymmetryValidator(object)` | Validator to detect symmetries and generate symmetric initial states. |
| `method` | `SymmetryValidator.detect_symmetries` | 21 | `public/module` | `detect_symmetries(system: ChaoticSystem, tolerance: float=1e-05) -> list[str]` | Detect which standard symmetries are valid for the system. |
| `method` | `SymmetryValidator.generate_symmetric_seeds` | 54 | `public/module` | `generate_symmetric_seeds(system: ChaoticSystem, seeds: list[dict[str, Any]], tolerance: float=1e-05) -> list[dict[str, Any]]` | Generate symmetric seeds for each seed in list using detected symmetries. |

### `hidden_attractors.validation.wolfram_artifacts`

Source: `version_2/hidden_attractors/validation/wolfram_artifacts.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_sha256` | 20 | `internal` | `_sha256(path: Path) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_json_passed` | 28 | `internal` | `_json_passed(path: Path) -> bool \| None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `WolframArtifactSet` | 36 | `public/module` | `class WolframArtifactSet(object)` | Resolved Wolfram CSV inputs and their generated-output provenance. |
| `method` | `WolframArtifactSet.complete` | 48 | `public/module` | `complete(self) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `WolframArtifactSet.provenance` | 51 | `public/module` | `provenance(self, *, relative_to: Path) -> dict[str, object]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `resolve_wolfram_artifacts` | 75 | `public/module` | `resolve_wolfram_artifacts(validation_root: Path, *, system_id: str=DEFAULT_SYSTEM_ID, generated_output_root: Path \| None=None) -> WolframArtifactSet` | Resolve official Wolfram inputs, preferring generated prefixed outputs. |

### `hidden_attractors.validation_contract`

Source: `version_2/hidden_attractors/validation_contract.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `ValidationIssue` | 31 | `public/module` | `class ValidationIssue(object)` | One contract-check finding. |
| `method` | `ValidationIssue.format` | 38 | `public/module` | `format(self, root: Path) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_read_json` | 46 | `internal` | `_read_json(path: Path) -> tuple[dict[str, Any] \| None, str \| None]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_has_csv_rows` | 58 | `internal` | `_has_csv_rows(path: Path) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_declared_files` | 67 | `internal` | `_declared_files(summary: dict[str, Any]) -> Iterable[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_check_json_fields` | 83 | `internal` | `_check_json_fields(path: Path, fields: Iterable[str], severity: str='ERROR') -> list[ValidationIssue]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_check_manifest_paths` | 100 | `internal` | `_check_manifest_paths(manifest_path: Path, validation_root: Path, pending_slugs: set[str], allow_pending: bool=False) -> list[ValidationIssue]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_final_report_status` | 128 | `internal` | `_final_report_status(manifest_path: Path) -> str \| None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `check_validation_contract` | 141 | `public/module` | `check_validation_contract(contract_path: Path=DEFAULT_CONTRACT, validation_root: Path=DEFAULT_VALIDATION_ROOT, allow_pending: bool=False) -> list[ValidationIssue]` | Return contract issues for a validation evidence tree. |
| `function` | `_build_parser` | 314 | `internal` | `_build_parser() -> argparse.ArgumentParser` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `main` | 322 | `public/module` | `main(argv: Sequence[str] \| None=None) -> int` | Console entry point for the validation contract checker. |

### `hidden_attractors.verification.basins`

Source: `version_2/hidden_attractors/verification/basins.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_classify_point_worker` | 6 | `internal` | `_classify_point_worker(args: Tuple) -> Tuple[int, int, int]` | Helper worker to run a single point simulation in parallel. |
| `function` | `generate_basin_slice` | 68 | `public/module` | `generate_basin_slice(plane: str, system: Any, transfer_mode: str, integrator: str, ref_tail: np.ndarray, stable_eqs: List[np.ndarray], fixed_values: Dict[str, float], extent: float=8.0, grid_n: int=40, center: Tuple[float, float, float]=(0.0, 0.0, 0.0), t_final: float=100.0, t_burn: float=40.0, h: float=0.02, workers: int=1, eq_tol: float=0.5, div_norm: float=120.0, metric: str='centroid_distance', tol: float=0.5, dynamics_mode: str='system', memory_mode: str='full', memory_window_length: Optional[int]=None, x_interval: Optional[List[float]]=None, y_interval: Optional[List[float]]=None, z_interval: Optional[List[float]]=None, around_equilibria: bool=False, local_radius: float=2.0, eq_name: str='global', system_id: str='chua', early_stop_config: Optional[dict]=None, equilibria_dict: Optional[Dict[str, np.ndarray]]=None, q_dynamics_effective: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]` | Generate a 2D basin slice mesh grid and evaluation matrix of classifications, with configurable intervals, centering, and real-time terminal progress printing. |

### `hidden_attractors.verification.candidate_gate`

Source: `version_2/hidden_attractors/verification/candidate_gate.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `normalize_hiddenness_label` | 37 | `public/module` | `normalize_hiddenness_label(label: str) -> str` | Map legacy labels to the canonical status vocabulary. |
| `function` | `normalize_candidate_evidence` | 43 | `public/module` | `normalize_candidate_evidence(evidence: dict[str, Any]) -> dict[str, Any]` | Return a canonical evidence dictionary with stable section defaults. |
| `function` | `_finite` | 88 | `internal` | `_finite(value: Any) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_close_to_one` | 92 | `internal` | `_close_to_one(value: Any) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_checked_conditions` | 96 | `internal` | `_checked_conditions(evidence: dict[str, Any]) -> dict[str, bool]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `missing_candidate_conditions` | 149 | `public/module` | `missing_candidate_conditions(evidence: dict[str, Any]) -> list[str]` | List unmet conditions for the strongest sampled-neighborhood label. |
| `function` | `classify_chaos_evidence` | 156 | `public/module` | `classify_chaos_evidence(evidence: dict[str, Any]) -> dict[str, Any]` | Classify finite-time chaos evidence using the frozen positive vocabulary. |
| `function` | `evaluate_candidate_gate` | 226 | `public/module` | `evaluate_candidate_gate(evidence: dict[str, Any]) -> dict[str, Any]` | Evaluate candidate promotion, hiddenness level, and chaos evidence. |

### `hidden_attractors.verification.classifiers`

Source: `version_2/hidden_attractors/verification/classifiers.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `classify_hiddenness_verdict` | 3 | `public/module` | `classify_hiddenness_verdict(target_hits_from_equilibria: int, equilibria_count: int, unstable_equilibria_count: int, seed_reached_attractor: bool, numerical_failures: int=0) -> str` | Classify the overall hiddenness verdict under the tested numerical contract. |

### `hidden_attractors.verification.equilibria`

Source: `version_2/hidden_attractors/verification/equilibria.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `solve_equilibria` | 5 | `public/module` | `solve_equilibria(system: Any) -> Dict[str, np.ndarray]` | Find all equilibrium points for the given system. |

### `hidden_attractors.verification.hiddenness`

Source: `version_2/hidden_attractors/verification/hiddenness.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `generate_neighborhood_points` | 5 | `public/module` | `generate_neighborhood_points(eq_point: np.ndarray, radius: float, num_samples: int, mode: str='sphere_random', seed: Optional[int]=None) -> np.ndarray` | Generate initial conditions in the neighborhood of an equilibrium point. |
| `function` | `evaluate_target_match` | 64 | `public/module` | `evaluate_target_match(trajectory_tail: np.ndarray, ref_tail: np.ndarray, metric: str='centroid_distance', tolerance: float=0.5, nn_percentile: float=90.0) -> bool` | Evaluate if the trajectory tail coincides with the reference attractor tail. |
| `function` | `run_neighborhood_probe` | 145 | `public/module` | `run_neighborhood_probe(system: Any, x0: np.ndarray, transfer_mode: str, integrator: str, t_final: float, t_burn: float, h: float, ref_tail: np.ndarray, stable_equilibria: List[np.ndarray], equilibrium_tol: float=0.5, divergence_norm: float=120.0, target_match_metric: str='centroid_distance', target_match_tol: float=0.5, dynamics_mode: str='system', memory_mode: str='full', memory_window_length: Optional[int]=None, early_stop_config: Optional[dict]=None, equilibria_dict: Optional[Dict[str, np.ndarray]]=None, q_dynamics_effective: Optional[float]=None) -> Dict[str, Any]` | Integrate a single trajectory from a neighborhood and classify its destination with early stopping. |

### `hidden_attractors.verification.hiddenness_contract`

Source: `version_2/hidden_attractors/verification/hiddenness_contract.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `HiddennessVerificationStatus` | 21 | `public/module` | `class HiddennessVerificationStatus(str, Enum)` | Enumeration of strict hiddenness verification states. |
| `function` | `is_radius_close` | 34 | `public/module` | `is_radius_close(r1: float, r2: float, rtol: float=1e-12, atol: float=1e-15) -> bool` | Tolerant floating-point comparison for radii. |
| `function` | `verify_hiddenness_contract` | 39 | `public/module` | `verify_hiddenness_contract(equilibria: Dict[str, np.ndarray], sphere_summary_records: List[Dict[str, Any]], probe_runs: List[Dict[str, Any]], required_radii: Sequence[float], require_all_equilibria: bool=True, allow_numerical_failures: bool=False, require_candidate_attractor: bool=True, seed_reached_attractor: bool=True, min_ref_tail_points: int=1000, min_probe_tail_points: int=200, ref_tail_size: int=1000, target_match_metric: str='nn_percentile', target_match_tol: float=0.5, target_match_nn_percentile: float=90.0, run_metadata: Mapping[str, Any] \| None=None, reference_was_robust: bool=False, neighborhood_sampling_mode: str='ball', basin_planes: Sequence[str]=()) -> Dict[str, Any]` | Verify operational hiddenness condition: B(A) ∩ U_epsilon(X_i*) = empty. |
| `function` | `_get_methodological_note` | 330 | `internal` | `_get_methodological_note() -> str` | Return the formal warning required by the verification contract. |

### `hidden_attractors.verification.jacobian`

Source: `version_2/hidden_attractors/verification/jacobian.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `compute_jacobian` | 4 | `public/module` | `compute_jacobian(system: Any, x: np.ndarray) -> np.ndarray` | Evaluate the Jacobian matrix at the state x. |

### `hidden_attractors.verification.sphere_tests`

Source: `version_2/hidden_attractors/verification/sphere_tests.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `run_single_sphere_probe` | 10 | `public/module` | `run_single_sphere_probe(payload: Tuple) -> Tuple[int, Dict[str, Any]]` | Worker function to simulate a single sphere initial condition. |
| `function` | `run_sphere_probe_sweep` | 92 | `public/module` | `run_sphere_probe_sweep(system: Any, config: Dict[str, Any], equilibria: Dict[str, np.ndarray], stable_eqs: List[np.ndarray], ref_tail: np.ndarray, output_dir: str, workers: int=1, q_dynamics_effective: Optional[float]=None) -> Dict[str, Any]` | Executes the complete equilibrium neighborhood sphere probe sweep with early stopping. |
| `function` | `_print_and_save_hiddenness_tables` | 330 | `internal` | `_print_and_save_hiddenness_tables(summary_records: List[Dict[str, Any]], output_dir: str) -> None` | Renders the Markdown summary table to console, CSV, and summary files. |

### `hidden_attractors.verification.stability`

Source: `version_2/hidden_attractors/verification/stability.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `classify_equilibrium_stability` | 5 | `public/module` | `classify_equilibrium_stability(system: Any, eq_point: np.ndarray, tol: float=1e-08) -> Dict[str, Any]` | Classify the local stability of an equilibrium point. |

### `hidden_attractors.verification.status_labels`

Source: `version_2/hidden_attractors/verification/status_labels.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `normalize_attractor_status` | 40 | `public/module` | `normalize_attractor_status(label: str) -> str` | Normalize old status labels to canonical ones. |

### `hidden_attractors.workflows.attractor_only`

Source: `version_2/hidden_attractors/workflows/attractor_only.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_compute_diagnostics` | 37 | `internal` | `_compute_diagnostics(times: np.ndarray, states: np.ndarray, t_burn: float, h: float) -> Dict[str, Any]` | Compute basic diagnostic metrics on the post-burn portion of a trajectory. |
| `function` | `_zero_one_test_approx` | 123 | `internal` | `_zero_one_test_approx(x: np.ndarray, n_samples: int=100) -> float` | Approximate 0-1 test for chaos (Gottwald & Melbourne 2004). |
| `function` | `_save_csv` | 165 | `internal` | `_save_csv(path: Path, times: np.ndarray, states: np.ndarray) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_attractor_only_workflow` | 184 | `public/module` | `run_attractor_only_workflow(config: Dict[str, Any]) -> Dict[str, Any]` | Execute the simulate_attractor_only workflow (Ruta B). |

### `hidden_attractors.workflows.basin_runner`

Source: `version_2/hidden_attractors/workflows/basin_runner.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_evaluate_target_match` | 32 | `internal` | `_evaluate_target_match(trajectory_tail: np.ndarray, ref_tail: np.ndarray, metric: str='centroid_distance', tolerance: float=0.5, nn_percentile: float=90.0) -> bool` | Evaluate if the trajectory tail matches the reference attractor. |
| `function` | `_classify_point_worker` | 80 | `internal` | `_classify_point_worker(args: Tuple) -> Tuple[int, int, int]` | Helper worker to run a single point integration and classify its destination. |
| `function` | `run_basin_workflow` | 135 | `public/module` | `run_basin_workflow(config: Dict[str, Any]) -> Dict[str, Any]` | Execute the basin of attraction slice generation and plotting. |

### `hidden_attractors.workflows.biased_chua`

Source: `version_2/hidden_attractors/workflows/biased_chua.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `_LazyPandas` | 58 | `internal` | `class _LazyPandas(object)` | Import pandas only when CSV/DataFrame workflow steps actually need it. |
| `method` | `_LazyPandas.__getattr__` | 61 | `dunder` | `__getattr__(self, name: str) -> Any` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_centered_reference` | 78 | `public/module` | `run_centered_reference(cfg: Dict[str, Any]) -> List[Dict[str, Any]]` | Runs centered describing function search as baseline. |
| `function` | `get_Wq` | 247 | `public/module` | `get_Wq(omega: float, q: float, pmat: np.ndarray, qvec: np.ndarray, rvec: np.ndarray) -> complex` | Compute the fractional transfer function value using the P - lambda I sign convention. |
| `function` | `harmonic_residual_sign_audit` | 274 | `public/module` | `harmonic_residual_sign_audit(W: complex, N1: float) -> Dict[str, float]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `biased_saturation_df` | 286 | `public/module` | `biased_saturation_df(A: float, c: float, g: float, n_theta: int=8192) -> Tuple[float, float]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `biased_saturation_residual` | 297 | `public/module` | `biased_saturation_residual(A: float, c: float, omega: float, params: ChuaParameters, q: float, n_theta: int=8192) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `find_biased_branches` | 321 | `public/module` | `find_biased_branches(params: ChuaParameters, q: float, s2_cfg: Dict[str, Any]) -> List[Dict[str, Any]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `build_biased_seed` | 367 | `public/module` | `build_biased_seed(params: ChuaParameters, q: float, A: float, c: float, omega: float, psi0: float, N1: float) -> Dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_affine_continuation` | 383 | `public/module` | `run_affine_continuation(params: ChuaParameters, q: float, h: float, seed_x0: np.ndarray, A: float, c: float, psi0: float, N1: float, lambda_values: List[float], t_transient: float, t_keep: float, div_threshold: float) -> List[Dict[str, Any]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_biased_df_search` | 480 | `public/module` | `run_biased_df_search(cfg: Dict[str, Any]) -> List[Dict[str, Any]]` | Runs biased DF grid search, continuation, and long integration. |
| `function` | `run_probe` | 698 | `public/module` | `run_probe(system: Any, x0: np.ndarray, ref_tail: np.ndarray, stable_eqs: List[np.ndarray], h3_cfg: Dict) -> Dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_hiddenness_for_candidate` | 734 | `public/module` | `run_hiddenness_for_candidate(candidate: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_hiddenness_verification` | 855 | `public/module` | `run_hiddenness_verification(candidates: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]` | Runs standard hiddenness protocol sweep across survived candidates. |
| `function` | `init_worker` | 887 | `public/module` | `init_worker(m1: float, m0: float, alpha: float, beta: float, gamma: float, ref_tail: np.ndarray, stable_eqs: List[np.ndarray], h4_cfg: Dict[str, Any]) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `worker_run_probe` | 900 | `public/module` | `worker_run_probe(x0: np.ndarray) -> Dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `sample_ball` | 939 | `public/module` | `sample_ball(eq_point: np.ndarray, radius: float, n: int, seed: int) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_extended_hiddenness` | 955 | `public/module` | `run_extended_hiddenness(cfg: Dict[str, Any]) -> Dict[str, Any]` | Runs high density multiprocessing probe up to radius 2.0. |
| `function` | `load_traj` | 1100 | `public/module` | `load_traj(traj_path: Path, t_burn: float=0.0, max_pts: int=15000) -> Optional[np.ndarray]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `write_markdown_report` | 1111 | `public/module` | `write_markdown_report(classif_rows: List[Dict], hid_results: List[Dict], outpath: Path) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_summarize_and_plot` | 1179 | `public/module` | `run_summarize_and_plot(cfg: Dict[str, Any]) -> None` | Reads all past step outputs and generates report assets and Markdown/JSON summaries. |

### `hidden_attractors.workflows.bifurcation`

Source: `version_2/hidden_attractors/workflows/bifurcation.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `run_bifurcation_workflow` | 33 | `public/module` | `run_bifurcation_workflow(config: Dict[str, Any]) -> Dict[str, Any]` | Execute the parameter sweep and bifurcation diagram workflow. |

### `hidden_attractors.workflows.centered_lure_df`

Source: `version_2/hidden_attractors/workflows/centered_lure_df.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `build_eta_grid` | 196 | `public/module` | `build_eta_grid(cont_cfg: dict) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_save_continuation_trace` | 229 | `internal` | `_save_continuation_trace(cont_steps: list, output_dir: str) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_get_lure_matrix` | 296 | `internal` | `_get_lure_matrix(system: Any) -> Any` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_get_lure_input_vector` | 299 | `internal` | `_get_lure_input_vector(system: Any) -> Any` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_get_lure_output_vector` | 302 | `internal` | `_get_lure_output_vector(system: Any) -> Any` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_get_lure_nonlinearity` | 305 | `internal` | `_get_lure_nonlinearity(system: Any) -> Any` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_get_describing_function` | 308 | `internal` | `_get_describing_function(system: Any) -> Any` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_evaluate_rhs` | 311 | `internal` | `_evaluate_rhs(system: Any, x: Any) -> Any` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_effective_q` | 314 | `internal` | `_effective_q(config: dict, system: Any) -> float` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_write_workflow_run_metadata` | 321 | `internal` | `_write_workflow_run_metadata(*, config: dict[str, Any], system: Any, run_id: str, system_id: str, q_dynamics: float, t_final: float, t_burn: float, seed: dict[str, Any] \| None) -> dict[str, Any]` | Persist the common reproducibility envelope for the centered workflow. |
| `function` | `run_workflow_integration` | 369 | `public/module` | `run_workflow_integration(system, x0, q_val, h, t_final, config, equilibria)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `use_c_backend_check` | 393 | `public/module` | `use_c_backend_check(config: Dict[str, Any]) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_centered_lure_df_workflow` | 396 | `public/module` | `run_centered_lure_df_workflow(config: dict) -> dict` | Execute the full 7-phase centered Lur'e describing function workflow with early stopping. |
| `function` | `_build_summary_dict` | 1132 | `internal` | `_build_summary_dict(config: Dict[str, Any], system: Any, equilibria: Dict[str, np.ndarray], unstable_eqs: List[np.ndarray], candidates: List[Tuple[float, float, float]], A0: Optional[float], omega0: Optional[float], k: Optional[float], cont_success: Optional[bool], probe_results: List[Dict[str, Any]], verdict: str, final_traj: Optional[np.ndarray], matched_ev: Optional[complex], target_lam: Optional[complex], modal_res: Optional[float], norm_res: Optional[float], marginal_eqs: Optional[List[np.ndarray]]=None, contract_res: Optional[Dict[str, Any]]=None, bib_res: Optional[Dict[str, Any]]=None) -> Dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_save_summary` | 1301 | `internal` | `_save_summary(summary: Dict[str, Any], output_dir: str) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_print_terminal_table` | 1312 | `internal` | `_print_terminal_table(summary: Dict[str, Any]) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |

### `hidden_attractors.workflows.config_loader`

Source: `version_2/hidden_attractors/workflows/config_loader.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_flatten_hierarchical` | 300 | `internal` | `_flatten_hierarchical(raw: Dict[str, Any]) -> Dict[str, Any]` | Convert new hierarchical YAML schema to internal flat dict. |
| `function` | `_is_hierarchical` | 430 | `internal` | `_is_hierarchical(raw: Dict[str, Any]) -> bool` | Return True if the YAML looks like the new hierarchical schema. |
| `function` | `_detect_and_warn_legacy` | 436 | `internal` | `_detect_and_warn_legacy(raw: Dict[str, Any]) -> None` | Emit deprecation warnings for flat/legacy YAML keys. |
| `function` | `_deep_merge` | 458 | `internal` | `_deep_merge(base: Dict, override: Dict) -> Dict` | Recursively merge *override* into a copy of *base*. |
| `function` | `_apply_defaults` | 469 | `internal` | `_apply_defaults(flat: Dict[str, Any]) -> Dict[str, Any]` | Fill missing keys from _DEFAULTS, merging nested dicts. |
| `function` | `_normalize` | 484 | `internal` | `_normalize(cfg: Dict[str, Any]) -> Dict[str, Any]` | Apply normalization: aliases, type casting, derived keys. |
| `function` | `_cast_nested_floats` | 707 | `internal` | `_cast_nested_floats(d: Dict, keys: list) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_cast_nested_ints` | 713 | `internal` | `_cast_nested_ints(d: Dict, keys: list) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_validate` | 723 | `internal` | `_validate(cfg: Dict[str, Any]) -> None` | Raise ValueError / UserWarning for invalid combinations. |
| `function` | `_resolve_output_dir` | 808 | `internal` | `_resolve_output_dir(cfg: Dict[str, Any]) -> str` | Return (and set in cfg) the resolved output directory. |
| `function` | `_normalize_memory_config` | 822 | `internal` | `_normalize_memory_config(flat: Dict[str, Any]) -> None` | Normalize and infer memory_mode / memory_policy before defaults are applied. |
| `function` | `load_config` | 885 | `public/module` | `load_config(path: str \| Path, allow_legacy: bool=True) -> Dict[str, Any]` | Load, normalize and validate a YAML config file. |
| `function` | `save_effective_config` | 941 | `public/module` | `save_effective_config(cfg: Dict[str, Any], output_dir: Optional[str]=None) -> Path` | Serialize the effective config to ``effective_config.yaml`` in *output_dir*. |
| `function` | `_set_nested` | 978 | `internal` | `_set_nested(cfg: dict, dotted_key: str, value: Any) -> None` | Set a value in a nested dict from a dotted key string. |
| `function` | `apply_cli_overrides` | 1013 | `public/module` | `apply_cli_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]` | Apply CLI override values to a loaded config, then re-validate. |
| `function` | `resolve_seed_transfer_contract` | 1081 | `public/module` | `resolve_seed_transfer_contract(config: Dict[str, Any], system: Any) -> Dict[str, Any]` | Resolve the explicit contract for describing function and transfer function evaluation. |

### `hidden_attractors.workflows.contracts`

Source: `version_2/hidden_attractors/workflows/contracts.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `SeedResult` | 15 | `public/module` | `class SeedResult(object)` | Candidate seed produced by a describing-function stage. |
| `class` | `ContinuationResult` | 24 | `public/module` | `class ContinuationResult(object)` | Output of a numerical continuation stage. |
| `class` | `HiddennessResult` | 33 | `public/module` | `class HiddennessResult(object)` | Evidence summary from equilibrium-neighborhood controls. |
| `class` | `SeedGenerator` | 42 | `public/module` | `class SeedGenerator(Protocol)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `SeedGenerator.__call__` | 43 | `dunder` | `__call__(self, system: ChaoticSystem, contract: NumericalContract) -> SeedResult` | Generate a classical describing-function seed. |
| `class` | `MachadoSeedGenerator` | 47 | `public/module` | `class MachadoSeedGenerator(Protocol)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `MachadoSeedGenerator.__call__` | 48 | `dunder` | `__call__(self, system: ChaoticSystem, contract: NumericalContract, *, mu: float) -> SeedResult` | Generate a Machado-family describing-function seed. |
| `class` | `ContinuationFunction` | 52 | `public/module` | `class ContinuationFunction(Protocol)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `ContinuationFunction.__call__` | 53 | `dunder` | `__call__(self, system: ChaoticSystem, seed: SeedResult, contract: NumericalContract) -> ContinuationResult` | Continue a candidate seed under the selected numerical contract. |
| `class` | `HiddennessVerifier` | 57 | `public/module` | `class HiddennessVerifier(Protocol)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `HiddennessVerifier.__call__` | 58 | `dunder` | `__call__(self, system: ChaoticSystem, candidate: ContinuationResult, contract: NumericalContract) -> HiddennessResult` | Run equilibrium-neighborhood hiddenness controls. |
| `class` | `BasinClassifier` | 62 | `public/module` | `class BasinClassifier(Protocol)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `BasinClassifier.__call__` | 63 | `dunder` | `__call__(self, system: ChaoticSystem, candidate: ContinuationResult, contract: NumericalContract) -> Mapping[str, Any]` | Classify basin evidence or return a documented replacement criterion. |
| `class` | `ReportWriter` | 67 | `public/module` | `class ReportWriter(Protocol)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `ReportWriter.__call__` | 68 | `dunder` | `__call__(self, system: ChaoticSystem, evidence: Mapping[str, Any], contract: NumericalContract) -> Mapping[str, Any]` | Write or return reproducible workflow artifacts. |
| `class` | `FullWorkflowContract` | 73 | `public/module` | `class FullWorkflowContract(object)` | Required hooks for a system to run the full analysis protocol. |
| `function` | `validate_full_workflow_system` | 84 | `public/module` | `validate_full_workflow_system(system: ChaoticSystem, workflow: FullWorkflowContract) -> None` | Validate that ``system`` exposes the mandatory full-workflow pieces. |

### `hidden_attractors.workflows.danca_abm_sphere_controls`

Source: `version_2/hidden_attractors/workflows/danca_abm_sphere_controls.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_float` | 150 | `internal` | `_float(value: Any, default: float=float('nan')) -> float` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_danca_config` | 157 | `internal` | `_danca_config(source_dir: Path) -> DancaChuaConfig` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_coarse_danca_class` | 180 | `internal` | `_coarse_danca_class(row: dict[str, Any], *, mean_x_gap: float) -> int` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_effective_config` | 198 | `internal` | `_effective_config(source: DancaChuaConfig, args: argparse.Namespace) -> DancaChuaConfig` | Return the fresh numerical contract while retaining Danca parameters. |
| `function` | `_runtime_config` | 222 | `internal` | `_runtime_config(cfg: dict[str, Any]) -> DancaChuaConfig` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_located_reference_seed` | 246 | `internal` | `_located_reference_seed(source_dir: Path) -> dict[str, Any]` | Load the seed previously located by an untruncated ABM reference run. |
| `function` | `_solver_cases` | 268 | `internal` | `_solver_cases(args: argparse.Namespace) -> list[dict[str, Any]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_native_params` | 306 | `internal` | `_native_params(dcfg: DancaChuaConfig) -> ChuaParameters` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_case_integrator` | 317 | `internal` | `_case_integrator(case: dict[str, Any], dcfg: DancaChuaConfig) -> Any` | Build the native integrator for one solver/memory cell. |
| `function` | `_write_trajectory` | 339 | `internal` | `_write_trajectory(path: Path, trajectory: np.ndarray) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_first_harmonic_reconstruction` | 347 | `internal` | `_first_harmonic_reconstruction(trajectory: np.ndarray) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_plot_reference_story` | 360 | `internal` | `_plot_reference_story(trajectory: np.ndarray, case_id: str, output_dir: Path) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_plot_system_nyquist` | 398 | `internal` | `_plot_system_nyquist(q: float, case_id: str, output: Path) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `generate_reference_figures` | 422 | `public/module` | `generate_reference_figures(root: Path, case_trajectories: dict[str, np.ndarray], q: float) -> dict[str, Any]` | Write the dynamic, spectral, and Nyquist diagnostics for each case. |
| `function` | `verify_reference` | 446 | `public/module` | `verify_reference(outdir: str \| Path) -> Path` | Accredit the located Danca seed once with untruncated ABM and compare cases. |
| `function` | `_require_verified_reference` | 506 | `internal` | `_require_verified_reference(root: Path, *, wait: bool=False, poll_sec: float=30.0) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `make_plan` | 516 | `public/module` | `make_plan(outdir: str \| Path, args: argparse.Namespace) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_chunk` | 641 | `public/module` | `run_chunk(outdir: str \| Path, chunk_id: int, chunks: int, *, wait_for_reference: bool=False) -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_plot_ball_control_figures` | 718 | `internal` | `_plot_ball_control_figures(root: Path, cfg: dict[str, Any], rows: list[dict[str, str]]) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `aggregate` | 753 | `public/module` | `aggregate(outdir: str \| Path, *, wait: bool=False, poll_sec: float=60.0) -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `refine_after_aggregate` | 839 | `public/module` | `refine_after_aggregate(outdir: str \| Path, *, wait: bool=True, poll_sec: float=60.0) -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `launch` | 892 | `public/module` | `launch(outdir: str \| Path, args: argparse.Namespace) -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `make_parser` | 942 | `public/module` | `make_parser() -> argparse.ArgumentParser` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `main` | 967 | `public/module` | `main(argv: list[str] \| None=None) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |

### `hidden_attractors.workflows.fractional_report_run`

Source: `version_2/hidden_attractors/workflows/fractional_report_run.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_configure_runtime` | 84 | `internal` | `_configure_runtime() -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_legacy_modules` | 92 | `internal` | `_legacy_modules() -> tuple[Any, Any, Any, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_seed_family` | 104 | `internal` | `_seed_family(df_family: str) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_method_label` | 117 | `internal` | `_method_label(df_family: str) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_finite` | 121 | `internal` | `_finite(value: Any, default: float=float('nan')) -> float` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_short_seed` | 129 | `internal` | `_short_seed(row: dict[str, Any]) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_valid_seed_configuration` | 135 | `internal` | `_valid_seed_configuration(row: dict[str, Any]) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_full_history_horizon` | 145 | `internal` | `_full_history_horizon(t_final: float, h: float) -> float` | Return an EFORK storage horizon that cannot truncate a run from t=0. |
| `function` | `_dominant_period_return_ratio` | 151 | `internal` | `_dominant_period_return_ratio(trajectory: np.ndarray, *, h: float, t_start: float, dominant_frequency: float) -> tuple[float, int]` | Return normalized closure error at one dominant sampled period. |
| `function` | `_post_continuation_periodicity` | 178 | `internal` | `_post_continuation_periodicity(trajectory: np.ndarray, *, h: float, t_final: float) -> dict[str, Any]` | Apply the maintained multi-component periodicity classifier after continuation. |
| `function` | `generate_lightweight_df_pool` | 188 | `public/module` | `generate_lightweight_df_pool(outdir: Path, *, biased_lhs_count: int=24, biased_keep_best: int=12) -> tuple[list[dict[str, Any]], dict[str, Any]]` | Generate fresh DF seeds without long trajectory integrations. |
| `function` | `_write_trajectory` | 284 | `internal` | `_write_trajectory(path: Path, traj: np.ndarray, max_rows: int=5000) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_read_trajectory` | 292 | `internal` | `_read_trajectory(path: Path) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_continued_observation` | 299 | `internal` | `_continued_observation(backend: FractionalChuaBackend, row: dict[str, Any], *, h: float, memory_length: float, t_final: float, full_history: bool) -> tuple[dict[str, np.ndarray], float, str, str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `screen_candidates_with_c` | 333 | `public/module` | `screen_candidates_with_c(outdir: Path, candidates: Sequence[dict[str, Any]], *, h: float, memory_length: float, t_final: float, full_history: bool) -> list[dict[str, Any]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `select_top_three` | 437 | `public/module` | `select_top_three(outdir: Path, screened: Sequence[dict[str, Any]], run_id: str, provenance: dict[str, Any], *, branch_id: str) -> list[dict[str, Any]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `generate_candidate_story_figures` | 537 | `public/module` | `generate_candidate_story_figures(branch_root: Path, branch_id: str, selected: Sequence[dict[str, Any]]) -> dict[str, Any]` | Generate interpretive figures for each promoted branch candidate. |
| `function` | `generate_candidate_diagnostic_figures` | 584 | `public/module` | `generate_candidate_diagnostic_figures(branch_root: Path, selected: Sequence[dict[str, Any]], *, trajectory_source: Path) -> dict[str, Any]` | Write FFT, PSD, and centered-DF Nyquist figures for selected candidates. |
| `function` | `_plot_biased_nyquist_df` | 645 | `internal` | `_plot_biased_nyquist_df(lure_system: Any, row: dict[str, Any], output: Path) -> str` | Plot the fixed-bias complex DF closure used by biased candidates. |
| `function` | `run_dynamic_evidence` | 686 | `public/module` | `run_dynamic_evidence(outdir: Path, selected: Sequence[dict[str, Any]], *, h: float, memory_length: float, t_final: float, full_history: bool, trajectory_source: Path) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_plot_spectrum` | 725 | `internal` | `_plot_spectrum(traj: np.ndarray, h: float, output: Path, *, omega0: float \| None=None) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_robustness_evidence` | 747 | `public/module` | `run_robustness_evidence(outdir: Path, selected: Sequence[dict[str, Any]], *, h: float, full_history: bool, memory_length: float) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_plot_robustness_overlay` | 797 | `internal` | `_plot_robustness_overlay(trajectories: Sequence[np.ndarray], labels: Sequence[str], output: Path) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_hiddenness_evidence` | 818 | `public/module` | `run_hiddenness_evidence(outdir: Path, selected: Sequence[dict[str, Any]], *, h: float, full_history: bool, memory_length: float, trajectory_source: Path) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `plot_hiddenness_ball_figures` | 947 | `public/module` | `plot_hiddenness_ball_figures(outdir: Path, selected: Sequence[dict[str, Any]]) -> list[str]` | Render computed equilibrium-ball samples, not placeholder basin cuts. |
| `function` | `run_strict_refinement` | 985 | `public/module` | `run_strict_refinement(outdir: Path, selected: Sequence[dict[str, Any]], *, h: float, full_history: bool, memory_length: float, trajectory_source: Path) -> None` | Reintegrate target hits and compare them with native-C references. |
| `function` | `run_danca_abm_control` | 1064 | `public/module` | `run_danca_abm_control(outdir: Path, *, h: float) -> dict[str, Any]` | Fresh ABM full-history control for the Danca-reported route only. |
| `function` | `_copy_files` | 1135 | `internal` | `_copy_files(source: Path, destination: Path, names: Sequence[str]) -> dict[str, str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_run_numerical_contract` | 1146 | `internal` | `_run_numerical_contract() -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_write_stage_summary` | 1164 | `internal` | `_write_stage_summary(directory: Path, stage: str, status: str, contract: dict[str, Any], *, files: dict[str, str], provenance: dict[str, Any], run_metadata: dict[str, Any], inputs: dict[str, Any] \| None=None, outputs: dict[str, Any] \| None=None, metrics: dict[str, Any] \| None=None, verdict: str \| None=None, state: str \| None=None, state_history: list[str] \| None=None, evidence: dict[str, Any] \| None=None, failed_requirements: list[str] \| None=None, method_scope: str='', warnings: list[str] \| None=None, literature_note: str='') -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `promote_validation` | 1212 | `public/module` | `promote_validation(run_root: Path, run_id: str, provenance: dict[str, Any], df_metadata: dict[str, Any], branch_results: dict[str, dict[str, Any]], danca_summary: dict[str, Any], run_metadata: dict[str, Any]) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `repository_provenance` | 1816 | `public/module` | `repository_provenance() -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_efork_branch` | 1838 | `public/module` | `run_efork_branch(root: Path, candidates: Sequence[dict[str, Any]], *, branch_id: str, full_history: bool, args: argparse.Namespace, run_id: str, provenance: dict[str, Any]) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run` | 1914 | `public/module` | `run(args: argparse.Namespace) -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `make_parser` | 2005 | `public/module` | `make_parser() -> argparse.ArgumentParser` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `main` | 2018 | `public/module` | `main(argv: Sequence[str] \| None=None) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |

### `hidden_attractors.workflows.integer_lure`

Source: `version_2/hidden_attractors/workflows/integer_lure.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `IntegerLureContinuationStep` | 19 | `public/module` | `class IntegerLureContinuationStep(object)` | One public lambda-continuation step from a Lur'e harmonic seed. |
| `method` | `IntegerLureContinuationStep.epsilon` | 30 | `public/module` | `epsilon(self) -> float` | Deprecated alias retained for plots and recorded integer artifacts. |
| `class` | `IntegerHiddennessProbe` | 37 | `public/module` | `class IntegerHiddennessProbe(object)` | One equilibrium-neighborhood probe for an integer-order system. |
| `function` | `require_lure` | 55 | `public/module` | `require_lure(system: ChaoticSystem \| LureSystem) -> LureSystem` | Return a Lur'e representation or raise a workflow-facing error. |
| `function` | `require_equilibria` | 65 | `public/module` | `require_equilibria(system: ChaoticSystem, equilibria: Mapping[str, Sequence[float]] \| None=None) -> dict[str, np.ndarray]` | Return explicit or registered equilibria for hiddenness controls. |
| `function` | `integer_lure_seed` | 77 | `public/module` | `integer_lure_seed(system: ChaoticSystem \| LureSystem, *, branch_index: int=0, method: str='classic', mu: float=1.0, theta: float=0.0, wmin: float=1e-05, wmax: float=50.0, nscan: int=40000) -> HarmonicSeed` | Build an integer-order Lur'e seed using ``s=i*omega``. |
| `function` | `integer_lure_original_rhs` | 109 | `public/module` | `integer_lure_original_rhs(lure: LureSystem)` | Return ``x -> A x + b psi(c^T x)`` for an integer-order Lur'e system. |
| `function` | `integer_lure_lambda_rhs` | 115 | `public/module` | `integer_lure_lambda_rhs(lure: LureSystem, gain: float, lambda_value: float)` | Return the public lambda-family RHS used to reach the target system. |
| `function` | `integer_lure_epsilon_rhs` | 133 | `public/module` | `integer_lure_epsilon_rhs(lure: LureSystem, gain: float, epsilon: float)` | Compatibility alias for historical epsilon terminology. |
| `function` | `integrate_integer_lure` | 139 | `public/module` | `integrate_integer_lure(system: ChaoticSystem \| LureSystem, x0: Sequence[float] \| np.ndarray, *, t_final: float, h: float, div_threshold: float \| None=None) -> tuple[np.ndarray, str]` | Integrate an integer-order Lur'e system from ``x0``. |
| `function` | `continue_integer_lure_seed` | 159 | `public/module` | `continue_integer_lure_seed(system: ChaoticSystem \| LureSystem, seed: HarmonicSeed, *, plan: ContinuationPlan \| None=None, eps_values: Iterable[float] \| None=None, t_transient: float=80.0, t_keep: float=80.0, h: float=0.01, div_threshold: float \| None=None) -> list[IntegerLureContinuationStep]` | Run public lambda continuation from a seed to the original system. |
| `function` | `final_integer_lure_attractor` | 249 | `public/module` | `final_integer_lure_attractor(system: ChaoticSystem \| LureSystem, x0: Sequence[float] \| np.ndarray, *, t_burn: float=120.0, t_keep: float=180.0, h: float=0.01, div_threshold: float \| None=None) -> tuple[np.ndarray, np.ndarray, str]` | Burn in from ``x0`` and return ``(target_seed, kept_trajectory, status)``. |
| `function` | `_tail_states` | 270 | `internal` | `_tail_states(trajectory: np.ndarray, *, t_start: float, max_points: int) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_random_unit_vectors` | 280 | `internal` | `_random_unit_vectors(dimension: int, count: int, rng: np.random.Generator) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_integer_lure_hiddenness_controls` | 287 | `public/module` | `run_integer_lure_hiddenness_controls(system: ChaoticSystem, target_trajectory: np.ndarray, *, equilibria: Mapping[str, Sequence[float]] \| None=None, radii: Sequence[float]=(1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01), samples_per_radius: int=24, t_final: float=500.0, t_burn: float=120.0, h: float=0.01, div_threshold: float=120.0, equilibrium_tol: float=0.001, target_cloud_tol: float=0.08, max_cloud_points: int=1000, random_seed: int=123456789, sampling_mode: str='ball', sample_growth_per_radius: int=0) -> list[IntegerHiddennessProbe]` | Run integer-order hiddenness controls from equilibrium neighborhoods. |
| `function` | `summarize_integer_hiddenness_controls` | 382 | `public/module` | `summarize_integer_hiddenness_controls(probes: Sequence[IntegerHiddennessProbe]) -> dict[str, Any]` | Summarize integer hiddenness controls for reports and CSV exports. |

### `hidden_attractors.workflows.lyapunov`

Source: `version_2/hidden_attractors/workflows/lyapunov.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `run_lyapunov_workflow` | 21 | `public/module` | `run_lyapunov_workflow(config: Dict[str, Any]) -> Dict[str, Any]` | Execute the Lyapunov exponent estimation workflow. |

### `hidden_attractors.workflows.protocol`

Source: `version_2/hidden_attractors/workflows/protocol.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_normalize_hiddenness_label` | 35 | `internal` | `_normalize_hiddenness_label(label: str) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_evaluate_candidate_gate` | 41 | `internal` | `_evaluate_candidate_gate(evidence: dict[str, Any]) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_jsonable` | 136 | `internal` | `_jsonable(value: Any) -> Any` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `NumericalContract` | 149 | `public/module` | `class NumericalContract(object)` | Complete numerical source-of-truth for one Caputo workflow run. |
| `method` | `NumericalContract.effective_transient` | 179 | `public/module` | `effective_transient(self) -> float` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `NumericalContract.validate` | 182 | `public/module` | `validate(self) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `NumericalContract.to_dict` | 200 | `public/module` | `to_dict(self) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `UnifiedSeedRecord` | 208 | `public/module` | `class UnifiedSeedRecord(object)` | Uniform seed record shared by classical Lur'e and Machado/FDF families. |
| `method` | `UnifiedSeedRecord.validate` | 226 | `public/module` | `validate(self) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `UnifiedSeedRecord.to_dict` | 241 | `public/module` | `to_dict(self) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `SoftPrecheckResult` | 246 | `public/module` | `class SoftPrecheckResult(object)` | Soft diagnostic decision made before continuation. |
| `method` | `SoftPrecheckResult.periodic` | 262 | `public/module` | `periodic(cls, candidate_id: str, **kwargs: Any) -> 'SoftPrecheckResult'` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `SoftPrecheckResult.validate` | 272 | `public/module` | `validate(self) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `ContinuationPlan` | 291 | `public/module` | `class ContinuationPlan(object)` | Public continuation interface; lambda=0 starts and lambda=1 targets. |
| `method` | `ContinuationPlan.validate` | 297 | `public/module` | `validate(self) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `ContinuationPlan.uniform` | 308 | `public/module` | `uniform(cls, steps: int, *, internal_parameter: str='lambda') -> 'ContinuationPlan'` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `ContinuationStep` | 316 | `public/module` | `class ContinuationStep(object)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `ContinuationTrace` | 328 | `public/module` | `class ContinuationTrace(object)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `PostContinuationDecision` | 338 | `public/module` | `class PostContinuationDecision(object)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `DynamicReference` | 351 | `public/module` | `class DynamicReference(object)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `RobustnessVerdict` | 368 | `public/module` | `class RobustnessVerdict(object)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `HiddennessTestResult` | 376 | `public/module` | `class HiddennessTestResult(object)` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `HiddennessTestResult.validate` | 391 | `public/module` | `validate(self) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `HiddennessTestResult.promotion_gate` | 417 | `public/module` | `promotion_gate(self) -> dict[str, Any]` | Evaluate and return the promotion gate result payload. |
| `method` | `HiddennessTestResult.promotion_verdict` | 514 | `public/module` | `promotion_verdict(self) -> str` | Return the only candidate label allowed by the available evidence. |
| `class` | `StageEnvelope` | 520 | `public/module` | `class StageEnvelope(object)` | Uniform machine-readable summary emitted by official stages. |
| `method` | `StageEnvelope.__post_init__` | 547 | `dunder` | `__post_init__(self) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `StageEnvelope.validate` | 564 | `public/module` | `validate(self) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `method` | `StageEnvelope.to_dict` | 582 | `public/module` | `to_dict(self) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `validate_global_report_coherence` | 612 | `public/module` | `validate_global_report_coherence(report_data: dict) -> None` | Validate coherence of global report validation metadata states and evidence. |
| `function` | `sample_uniform_ball` | 758 | `public/module` | `sample_uniform_ball(center: Sequence[float], radius: float, count: int, rng: np.random.Generator) -> np.ndarray` | Sample points inside an n-dimensional ball, including interior points. |

### `hidden_attractors.workflows.refined_basin`

Source: `version_2/hidden_attractors/workflows/refined_basin.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_cache_environment` | 117 | `internal` | `_cache_environment() -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_float` | 124 | `internal` | `_float(value: Any, default: float=float('nan')) -> float` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_int` | 131 | `internal` | `_int(value: Any, default: int=0) -> int` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_truthy` | 138 | `internal` | `_truthy(value: Any) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_csv_tokens` | 144 | `internal` | `_csv_tokens(value: Any) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_load_reference_seed` | 148 | `internal` | `_load_reference_seed(source_cfg: dict[str, Any]) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_analysis_start` | 156 | `internal` | `_analysis_start(cfg: dict[str, Any]) -> float` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_score_against` | 162 | `internal` | `_score_against(payload: dict[str, Any], ref: dict[str, Any], weights: dict[str, float]) -> dict[str, float]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_metric_ok` | 192 | `internal` | `_metric_ok(value: float, threshold: float, *, missing_ok: bool=False) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_deterministic_control_directions` | 198 | `internal` | `_deterministic_control_directions() -> dict[str, np.ndarray]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_reference_payloads` | 209 | `internal` | `_reference_payloads(cfg: dict[str, Any], backend: FractionalChuaBackend) -> dict[str, dict[str, Any]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_negative_control_payloads` | 235 | `internal` | `_negative_control_payloads(cfg: dict[str, Any], backend: FractionalChuaBackend) -> dict[str, dict[str, Any]]` | Build local equilibrium-neighborhood controls for target separation. |
| `function` | `make_config` | 292 | `public/module` | `make_config(outdir: str \| Path, args: argparse.Namespace) -> dict[str, Any]` | Prepare a refined-basin run from an existing coarse basin output. |
| `function` | `classify_refined` | 405 | `public/module` | `classify_refined(traj: np.ndarray, cfg: dict[str, Any], refs: dict[str, dict[str, Any]], controls: dict[str, dict[str, Any]] \| None=None) -> dict[str, Any]` | Classify one trajectory using stricter target-reference geometry. |
| `function` | `run_chunk` | 512 | `public/module` | `run_chunk(outdir: str \| Path, chunk_id: int, chunks: int) -> Path` | Reintegrate and refine one chunk of previously unknown cells. |
| `function` | `_counts` | 572 | `internal` | `_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `plot_grid` | 576 | `public/module` | `plot_grid(outdir: Path, cfg: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str` | Plot the refined basin grid. |
| `function` | `aggregate` | 624 | `public/module` | `aggregate(outdir: str \| Path, *, wait: bool=False, poll_sec: float=30.0) -> Path` | Merge refined chunks with the original grid and write final artifacts. |
| `function` | `launch` | 667 | `public/module` | `launch(outdir: str \| Path, args: argparse.Namespace) -> Path` | Launch independent OS processes for chunked refined classification. |
| `function` | `make_parser` | 695 | `public/module` | `make_parser() -> argparse.ArgumentParser` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `main` | 729 | `public/module` | `main(argv: Sequence[str] \| None=None) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |

### `hidden_attractors.workflows.robustness_overlay`

Source: `version_2/hidden_attractors/workflows/robustness_overlay.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_cache_environment` | 79 | `internal` | `_cache_environment() -> None` | Set writable Matplotlib/font cache folders before importing pyplot. |
| `function` | `_case_dicts` | 88 | `internal` | `_case_dicts(cases: Sequence[RobustnessCase]) -> list[dict[str, Any]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_candidate_dicts` | 93 | `internal` | `_candidate_dicts(source_dir: str \| Path, q: float) -> list[dict[str, Any]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `make_config` | 104 | `public/module` | `make_config(outdir: str \| Path, *, source_dir: str \| Path=DEFAULT_SOURCE_DIR, q: float=0.9998, divergence_norm: float=120.0, equilibrium_tol: float=0.001, max_store_points: int=6000, max_metric_points: int=1000, max_section_points: int=300, tail_fraction_start: float=0.5) -> dict[str, Any]` | Create and persist the robustness-overlay workflow configuration. |
| `function` | `save_sampled_trajectory` | 191 | `public/module` | `save_sampled_trajectory(path: str \| Path, traj: np.ndarray, max_points: int) -> int` | Store an evenly subsampled ``t,x,y,z`` trajectory CSV. |
| `function` | `load_trajectory_csv` | 200 | `public/module` | `load_trajectory_csv(path: str \| Path) -> np.ndarray` | Load a sampled trajectory CSV written by :func:`save_sampled_trajectory`. |
| `function` | `run_candidate` | 207 | `public/module` | `run_candidate(outdir: str \| Path, candidate_index: int) -> Path` | Run all robustness cases for one candidate and write metrics CSV. |
| `function` | `plot_candidate` | 280 | `public/module` | `plot_candidate(outdir: str \| Path, cand: dict[str, Any], metric_rows: Sequence[dict[str, str]]) -> str` | Generate the overlay figure for one candidate from saved trajectory CSVs. |
| `function` | `aggregate` | 302 | `public/module` | `aggregate(outdir: str \| Path, *, wait: bool=False) -> Path` | Aggregate candidate metric files and build overlay plots. |
| `function` | `launch_independent_jobs` | 336 | `public/module` | `launch_independent_jobs(outdir: str \| Path, args: argparse.Namespace) -> None` | Launch one independent OS process per candidate plus one aggregator. |
| `function` | `make_parser` | 373 | `public/module` | `make_parser() -> argparse.ArgumentParser` | Return the CLI parser used by the thin compatibility script. |
| `function` | `main` | 393 | `public/module` | `main(argv: Sequence[str] \| None=None) -> None` | CLI entrypoint. |

### `hidden_attractors.workflows.simple_runner`

Source: `version_2/hidden_attractors/workflows/simple_runner.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `run_simple_workflow` | 17 | `public/module` | `run_simple_workflow(config: Dict[str, Any]) -> Dict[str, Any]` | Execute the configured workflow stages in order or dispatch to sub-workflows. |

### `hidden_attractors.workflows.specs`

Source: `version_2/hidden_attractors/workflows/specs.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `class` | `IntegratorSpec` | 33 | `public/module` | `class IntegratorSpec(object)` | Numerical solver contract shared by CLI and legacy wrappers. |
| `method` | `IntegratorSpec.validate` | 57 | `public/module` | `validate(self) -> list[str]` | Return validation errors without running the solver. |
| `class` | `DestinationClassifierSpec` | 80 | `public/module` | `class DestinationClassifierSpec(object)` | Operational destination labels for basin and hiddenness workflows. |
| `method` | `DestinationClassifierSpec.validate` | 92 | `public/module` | `validate(self) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `TargetReferenceSpec` | 99 | `public/module` | `class TargetReferenceSpec(object)` | Candidate attractor reference used for target-hit and refinement logic. |
| `method` | `TargetReferenceSpec.validate` | 109 | `public/module` | `validate(self, *, dimension: int \| None=None) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `SphereControlSpec` | 121 | `public/module` | `class SphereControlSpec(object)` | Equilibrium-neighborhood sampling contract. |
| `method` | `SphereControlSpec.validate` | 136 | `public/module` | `validate(self) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `BasinSliceSpec` | 152 | `public/module` | `class BasinSliceSpec(object)` | Initial-condition grid or section used by basin workflows. |
| `method` | `BasinSliceSpec.validate` | 162 | `public/module` | `validate(self, *, dimension: int \| None=None) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `StrictRefinementSpec` | 180 | `public/module` | `class StrictRefinementSpec(object)` | Geometry thresholds for target-reference refinement. |
| `method` | `StrictRefinementSpec.validate` | 194 | `public/module` | `validate(self) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `class` | `TrajectoryDiagnosticsSpec` | 214 | `public/module` | `class TrajectoryDiagnosticsSpec(object)` | Tail window and observable contract for metrics, spectra, and sections. |
| `method` | `TrajectoryDiagnosticsSpec.validate` | 225 | `public/module` | `validate(self, *, integrator: IntegratorSpec \| None=None) -> list[str]` | Validate the declared post-transient diagnostics window. |
| `class` | `ParameterSweepSpec` | 250 | `public/module` | `class ParameterSweepSpec(object)` | Parameter sweep contract for bifurcation and continuation-like runs. |
| `method` | `ParameterSweepSpec.validate` | 262 | `public/module` | `validate(self) -> list[str]` | Validate the sweep axis without generating trajectories. |
| `class` | `RobustnessCaseSpec` | 287 | `public/module` | `class RobustnessCaseSpec(object)` | One controlled perturbation for robustness workflows. |
| `method` | `RobustnessCaseSpec.validate` | 295 | `public/module` | `validate(self) -> list[str]` | Validate that the robustness case is named and auditable. |
| `class` | `WorkflowInputSpec` | 306 | `public/module` | `class WorkflowInputSpec(object)` | Single auditable input contract for reusable workflows. |
| `method` | `WorkflowInputSpec.validate_for` | 323 | `public/module` | `validate_for(self, features: Sequence[str]) -> list[str]` | Validate only the pieces needed by the requested features. |
| `method` | `WorkflowInputSpec.to_jsonable` | 376 | `public/module` | `to_jsonable(self) -> dict[str, Any]` | Return a JSON-safe dictionary. |
| `function` | `write_workflow_spec` | 395 | `public/module` | `write_workflow_spec(path: str \| Path, spec: WorkflowInputSpec) -> None` | Persist a workflow specification next to run artifacts. |
| `function` | `load_workflow_spec` | 401 | `public/module` | `load_workflow_spec(path: str \| Path) -> Mapping[str, Any]` | Load a JSON workflow spec for legacy adapters or CLIs. |
| `function` | `example_chua_fractional_spec` | 407 | `public/module` | `example_chua_fractional_spec() -> WorkflowInputSpec` | Return a minimal example spec for documentation and tests. |

### `hidden_attractors.workflows.sphere_controls`

Source: `version_2/hidden_attractors/workflows/sphere_controls.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `sphere_fields` | 109 | `public/module` | `sphere_fields() -> list[str]` | Return stable CSV fields for raw sphere-control rows. |
| `function` | `robust_raw_fields` | 115 | `public/module` | `robust_raw_fields() -> list[str]` | Return stable CSV fields for raw robustness rows. |
| `function` | `as_float` | 121 | `public/module` | `as_float(value: Any, default: float=float('nan')) -> float` | Parse a numeric artifact value, returning ``default`` on failure. |
| `function` | `load_requested_candidates` | 130 | `public/module` | `load_requested_candidates(source_dir: str \| Path=DEFAULT_SOURCE_DIR) -> list[dict[str, Any]]` | Load exactly the three final candidates used by this project stage. |
| `function` | `load_equilibria_from_c` | 143 | `public/module` | `load_equilibria_from_c() -> dict[str, np.ndarray]` | Load equilibria from the same C basin backend used for classification. |
| `function` | `robustness_cases` | 149 | `public/module` | `robustness_cases(args: argparse.Namespace) -> list[dict[str, Any]]` | Return the classifier-based robustness contracts. |
| `function` | `make_plan` | 168 | `public/module` | `make_plan(outdir: str \| Path, args: argparse.Namespace) -> dict[str, Any]` | Create ball-neighborhood seeds and persist the run configuration. |
| `function` | `_classify_point` | 289 | `internal` | `_classify_point(backend: BasinBackend, x0: float, y0: float, z0: float, contract: dict[str, Any]) -> int` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_sphere_chunk` | 305 | `public/module` | `run_sphere_chunk(outdir: str \| Path, chunk_id: int, chunks: int) -> Path` | Classify one independent chunk of the sphere-control plan. |
| `function` | `run_robustness` | 344 | `public/module` | `run_robustness(outdir: str \| Path) -> Path` | Run coarse target-class robustness tests from continuation endpoints. |
| `function` | `aggregate` | 400 | `public/module` | `aggregate(outdir: str \| Path, *, wait: bool=False, poll_sec: float=30.0) -> Path` | Aggregate all sphere chunks into summary and decision artifacts. |
| `function` | `launch` | 478 | `public/module` | `launch(outdir: str \| Path, args: argparse.Namespace) -> Path` | Launch independent OS processes for sphere chunks, robustness and aggregation. |
| `function` | `make_parser` | 511 | `public/module` | `make_parser() -> argparse.ArgumentParser` | Return the CLI parser for the compatibility script. |
| `function` | `main` | 546 | `public/module` | `main(argv: list[str] \| None=None) -> None` | CLI entrypoint. |

### `hidden_attractors.workflows.strict_target_refinement`

Source: `version_2/hidden_attractors/workflows/strict_target_refinement.py`

| Kind | Name | Line | Visibility | Signature / Declaration | Documentation |
| --- | --- | ---: | --- | --- | --- |
| `function` | `_float` | 108 | `internal` | `_float(value: Any, default: float=float('nan')) -> float` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_int` | 115 | `internal` | `_int(value: Any, default: int=0) -> int` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_tokens` | 122 | `internal` | `_tokens(value: Any) -> list[str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_truthy` | 126 | `internal` | `_truthy(value: Any) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_seed_from_row` | 132 | `internal` | `_seed_from_row(row: dict[str, Any]) -> np.ndarray` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_analysis_start` | 143 | `internal` | `_analysis_start(contract: dict[str, Any], analysis: dict[str, Any]) -> float` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_score_against` | 147 | `internal` | `_score_against(payload: dict[str, Any], ref: dict[str, Any], weights: dict[str, float]) -> dict[str, float]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_metric_ok` | 172 | `internal` | `_metric_ok(value: float, threshold: float, *, missing_ok: bool=False) -> bool` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_classify_strict` | 178 | `internal` | `_classify_strict(traj: np.ndarray, cfg: dict[str, Any], refs: dict[str, dict[str, Any]], controls: dict[str, dict[str, Any]]) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_project_integrator` | 269 | `internal` | `_project_integrator(cfg: dict[str, Any]) -> tuple[Any, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_danca_integrator` | 286 | `internal` | `_danca_integrator(cfg: dict[str, Any]) -> tuple[Any, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_reference_payloads` | 314 | `internal` | `_reference_payloads(cfg: dict[str, Any], integrate: Any) -> dict[str, dict[str, Any]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_control_payloads` | 334 | `internal` | `_control_payloads(cfg: dict[str, Any], eq_backend: Any, integrate: Any) -> dict[str, dict[str, Any]]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_load_project_candidate_from_sphere` | 366 | `internal` | `_load_project_candidate_from_sphere(source_cfg: dict[str, Any], candidate_id: str) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_contract_and_reference` | 373 | `internal` | `_contract_and_reference(mode: str, source_dir: Path, args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray, str]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_default_source_csv` | 437 | `internal` | `_default_source_csv(mode: str, source_dir: Path) -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_source_label` | 447 | `internal` | `_source_label(row: dict[str, Any], mode: str) -> str` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_normalize_plan_row` | 453 | `internal` | `_normalize_plan_row(row: dict[str, Any], mode: str, candidate_id: str, case_index: int) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `make_config` | 476 | `public/module` | `make_config(outdir: str \| Path, args: argparse.Namespace) -> dict[str, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `_integrator_for_cfg` | 585 | `internal` | `_integrator_for_cfg(cfg: dict[str, Any]) -> tuple[Any, Any]` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `run_chunk` | 591 | `public/module` | `run_chunk(outdir: str \| Path, chunk_id: int, chunks: int) -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `aggregate` | 631 | `public/module` | `aggregate(outdir: str \| Path, *, wait: bool=False, poll_sec: float=60.0) -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `launch` | 690 | `public/module` | `launch(outdir: str \| Path, args: argparse.Namespace) -> Path` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `make_parser` | 716 | `public/module` | `make_parser() -> argparse.ArgumentParser` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |
| `function` | `main` | 778 | `public/module` | `main(argv: list[str] \| None=None) -> None` | Internal helper or undocumented symbol; not a stable public contract unless exported elsewhere. |

## Maintenance Rule

When a new function, method, or class is added under `hidden_attractors`, update this reference before release. If the symbol participates in public workflows, also update `docs/quick_start.md`, `docs/getting_started.md`, `USER_MANUAL.md`, and the unified report summary.

