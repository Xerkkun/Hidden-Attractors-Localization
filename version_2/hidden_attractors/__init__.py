"""Numerical tools for hidden-attractor studies in fractional-order systems.

Stability tiers
---------------
This package organises its public surface into four tiers.  Every sub-module
starts with a ``Stability: <tier>`` line in its docstring.

stable
    Signatures and return types are fixed.  Breaking changes require a version
    bump and a deprecation cycle.
    Modules: ``models``, ``systems``, ``basins``, ``io``, ``candidates``.

experimental
    API is useful and tested but may evolve.  Changes will be noted in a
    changelog entry.
    Modules: ``analysis``, ``fractional``, ``seed_generation``, ``solvers``,
    ``plotting``, ``integrations``, ``workflows``.

internal
    Consumed by workflows and backends; not part of the user-facing surface.
    May change without notice.
    Modules: ``native``, ``parallel``, ``paths``, ``cli``.

legacy
    Compatibility label for historical symbol aliases. No importable
    ``hidden_attractors.legacy`` module is distributed.

See ``docs/api_stability.md`` for guarantees, upgrade-path guidance, and how
to introspect a symbol's tier programmatically.

Background
----------
The package provides reusable dynamical systems, integer solvers, differentiated
fractional-operator contracts, Caputo trajectory solvers, sampled GL/RL,
tempered, variable-order, distributed-order, conformable, Caputo--Fabrizio and
Hadamard-family operators, trajectory diagnostics, plotting, and process-safe
IO helpers.

The package is intentionally conservative: harmonic-balance and describing
function objects are treated as seed generators, while hiddenness and
robustness are always numerical post-checks on the causal Caputo model.
"""

# Stability constants (re-exported for convenience)
from ._stability import (  # noqa: F401
    EXPERIMENTAL,
    INTERNAL,
    LEGACY,
    STABLE,
    api_tier,
    assert_tier,
    get_tier,
)

# Stable API
# models - vector fields, parameters, equilibria
from .models.chua import (
    ChuaParameters,
    chua_arctan_wu2023_parameters,
    chua_nonsmooth_parameters,
    chua_parameters,
    equilibria_arctan,
    equilibria_nonsmooth,
    jacobian_arctan,
    jacobian_nonsmooth,
    rhs_arctan,
    rhs_nonsmooth,
    # Compatibility aliases for recorded runs created with the old label.
    chua_piecewise_parameters,
    equilibria_piecewise,
    jacobian_piecewise,
    rhs_piecewise,
)

# systems - chaotic-system registry and capability checks
from .systems import (
    ChaoticSystem,
    ExpressionSystemDefinition,
    ExpressionValidationError,
    LureSystem,
    compile_expression_system,
    get_system,
    list_systems,
    register_system,
)
from .systems.requirements import check_system_capability, known_workflows, requirements_for

# basins - classification labels
from .basins import CLASS_LABELS, TARGET_CLASS_IDS, class_label, is_target_class

# io - portable trajectory loading
from .io import load_trajectory_csv

# simulation - structured integer/fractional trajectory generation
from .simulation import SimulationResult, simulate, simulate_fractional

# Experimental API
# analysis - trajectory diagnostics and Lyapunov estimates
from .analysis import (
    AlignmentIndexResult,
    AnalysisResult,
    BifurcationPoint,
    CORRELATION_DIMENSION_EVIDENCE_SCOPE,
    CORRELATION_DIMENSION_REFERENCE_DOIS,
    CORRELATION_SUM_NATIVE_AUTO_MIN_PAIRS,
    CorrelationDimensionResult,
    CorrelationSumResult,
    CovariantAngleResult,
    CovariantLyapunovResult,
    CovariantQRHistoryResult,
    OrdinalPatternDistribution,
    PERMUTATION_ENTROPY_EVIDENCE_SCOPE,
    PERMUTATION_ENTROPY_MAX_EMBEDDING_DIMENSION,
    PERMUTATION_ENTROPY_MAX_PATTERN_STATES,
    PERMUTATION_ENTROPY_NATIVE_AUTO_MIN_WINDOWS,
    PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS,
    PERMUTATION_ENTROPY_REFERENCE_DOIS,
    PermutationEntropyResult,
    LyapunovComputationRequest,
    LyapunovComputationSummary,
    LyapunovResult,
    PoincareCrossingResult,
    PrehistorySpec,
    RobustnessCase,
    SpectrumResult,
    TimeSeriesLyapunovResult,
    TrajectoryInput,
    alignment_indices_from_tangent_history,
    bifurcation_points_from_trajectories,
    bifurcation_summary,
    compute_boundedness_metrics,
    compute_fft_psd,
    compute_lyapunov_spectrum,
    compute_trajectory_metrics,
    correlation_sum_curve,
    covariant_lyapunov_angles,
    detect_poincare_crossings,
    estimate_correlation_dimension,
    estimate_time_series_lyapunov,
    fft_spectrum,
    fit_correlation_dimension,
    generalized_alignment_index,
    integer_flow_alignment_indices,
    integer_flow_covariant_lyapunov_vectors,
    integer_covariant_vectors_from_qr_history,
    integer_map_alignment_indices,
    integer_map_covariant_lyapunov_vectors,
    integer_qr_benettin_lyapunov_exponents,
    integer_system_alignment_indices,
    integer_system_covariant_lyapunov_vectors,
    integer_system_lyapunov_exponents,
    kaplan_yorke_dimension,
    linear_dependence_index,
    ordinal_pattern_distribution,
    permutation_entropy,
    permutation_entropy_from_distribution,
    psd_welch,
    smaller_alignment_index,
    trajectory_component_spectra,
    trajectory_metrics,
    trajectory_metrics_for_system,
    validate_lyapunov_method_request,
    zero_one_test,
)

# seed_generation - harmonic-balance seeds (Chua + generic Lur'e)
from .seed_generation import (
    HarmonicSeed,
    find_harmonic_seed,
    find_lure_harmonic_seed,
    find_lure_omega_gain_candidates,
    find_omega_gain_candidates,
    validate_fractional_order,
)

# workflows - high-level reproducible numerical pipelines
from .workflows.contracts import FullWorkflowContract, NumericalContract, validate_full_workflow_system
from .workflows.protocol import (
    FINAL_LABELS,
    OFFICIAL_STAGE_ORDER,
    PROTOCOL_VERSION,
    SEED_FAMILIES,
    ContinuationPlan,
    ContinuationTrace,
    DynamicReference,
    HiddennessTestResult,
    PostContinuationDecision,
    RobustnessVerdict,
    SoftPrecheckResult,
    StageEnvelope,
    UnifiedSeedRecord,
)
from .workflows.specs import (
    BasinSliceSpec,
    DestinationClassifierSpec,
    IntegratorSpec,
    ParameterSweepSpec,
    RobustnessCaseSpec,
    SphereControlSpec,
    StrictRefinementSpec,
    TargetReferenceSpec,
    TrajectoryDiagnosticsSpec,
    WorkflowInputSpec,
)
from .workflows.integer_lure import (
    continue_integer_lure_seed,
    final_integer_lure_attractor,
    integer_lure_seed,
    integrate_integer_lure,
    run_integer_lure_hiddenness_controls,
)
from .workflows.config_loader import load_config, save_effective_config
from .workflows.attractor_only import run_attractor_only_workflow
from .workflows.bifurcation import run_bifurcation_workflow
from .workflows.basin_runner import run_basin_workflow
from .workflows.simple_runner import run_simple_workflow

PUBLIC_API_STABLE = (
    "ChuaParameters",
    "chua_parameters",
    "chua_arctan_wu2023_parameters",
    "chua_nonsmooth_parameters",
    "equilibria_arctan",
    "equilibria_nonsmooth",
    "jacobian_arctan",
    "jacobian_nonsmooth",
    "rhs_arctan",
    "rhs_nonsmooth",
    "ChaoticSystem",
    "LureSystem",
    "check_system_capability",
    "get_system",
    "known_workflows",
    "list_systems",
    "register_system",
    "requirements_for",
    "CLASS_LABELS",
    "TARGET_CLASS_IDS",
    "class_label",
    "is_target_class",
    "load_trajectory_csv",
)

PUBLIC_API_EXPERIMENTAL = (
    "ExpressionSystemDefinition",
    "ExpressionValidationError",
    "SimulationResult",
    "compile_expression_system",
    "simulate",
    "simulate_fractional",
    "AlignmentIndexResult",
    "AnalysisResult",
    "BifurcationPoint",
    "CORRELATION_DIMENSION_EVIDENCE_SCOPE",
    "CORRELATION_DIMENSION_REFERENCE_DOIS",
    "CORRELATION_SUM_NATIVE_AUTO_MIN_PAIRS",
    "CorrelationDimensionResult",
    "CorrelationSumResult",
    "CovariantAngleResult",
    "CovariantLyapunovResult",
    "CovariantQRHistoryResult",
    "OrdinalPatternDistribution",
    "PERMUTATION_ENTROPY_EVIDENCE_SCOPE",
    "PERMUTATION_ENTROPY_MAX_EMBEDDING_DIMENSION",
    "PERMUTATION_ENTROPY_MAX_PATTERN_STATES",
    "PERMUTATION_ENTROPY_NATIVE_AUTO_MIN_WINDOWS",
    "PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS",
    "PERMUTATION_ENTROPY_REFERENCE_DOIS",
    "PermutationEntropyResult",
    "LyapunovComputationRequest",
    "LyapunovComputationSummary",
    "LyapunovResult",
    "PoincareCrossingResult",
    "PrehistorySpec",
    "RobustnessCase",
    "SpectrumResult",
    "TimeSeriesLyapunovResult",
    "TrajectoryInput",
    "alignment_indices_from_tangent_history",
    "bifurcation_points_from_trajectories",
    "bifurcation_summary",
    "compute_boundedness_metrics",
    "compute_fft_psd",
    "compute_lyapunov_spectrum",
    "compute_trajectory_metrics",
    "correlation_sum_curve",
    "covariant_lyapunov_angles",
    "detect_poincare_crossings",
    "estimate_correlation_dimension",
    "estimate_time_series_lyapunov",
    "fft_spectrum",
    "fit_correlation_dimension",
    "generalized_alignment_index",
    "integer_flow_alignment_indices",
    "integer_flow_covariant_lyapunov_vectors",
    "integer_covariant_vectors_from_qr_history",
    "integer_map_alignment_indices",
    "integer_map_covariant_lyapunov_vectors",
    "integer_qr_benettin_lyapunov_exponents",
    "integer_system_alignment_indices",
    "integer_system_covariant_lyapunov_vectors",
    "integer_system_lyapunov_exponents",
    "kaplan_yorke_dimension",
    "linear_dependence_index",
    "ordinal_pattern_distribution",
    "permutation_entropy",
    "permutation_entropy_from_distribution",
    "psd_welch",
    "smaller_alignment_index",
    "trajectory_component_spectra",
    "trajectory_metrics",
    "trajectory_metrics_for_system",
    "validate_lyapunov_method_request",
    "zero_one_test",
    "HarmonicSeed",
    "find_harmonic_seed",
    "find_lure_harmonic_seed",
    "find_lure_omega_gain_candidates",
    "find_omega_gain_candidates",
    "validate_fractional_order",
    "BasinSliceSpec",
    "DestinationClassifierSpec",
    "FullWorkflowContract",
    "ContinuationPlan",
    "ContinuationTrace",
    "DynamicReference",
    "FINAL_LABELS",
    "HiddennessTestResult",
    "IntegratorSpec",
    "NumericalContract",
    "OFFICIAL_STAGE_ORDER",
    "ParameterSweepSpec",
    "RobustnessCaseSpec",
    "RobustnessVerdict",
    "PROTOCOL_VERSION",
    "PostContinuationDecision",
    "SEED_FAMILIES",
    "SoftPrecheckResult",
    "SphereControlSpec",
    "StageEnvelope",
    "StrictRefinementSpec",
    "TargetReferenceSpec",
    "TrajectoryDiagnosticsSpec",
    "WorkflowInputSpec",
    "UnifiedSeedRecord",
    "continue_integer_lure_seed",
    "final_integer_lure_attractor",
    "integer_lure_seed",
    "integrate_integer_lure",
    "run_integer_lure_hiddenness_controls",
    "validate_full_workflow_system",
    "load_config",
    "save_effective_config",
    "run_attractor_only_workflow",
    "run_bifurcation_workflow",
    "run_basin_workflow",
    "run_simple_workflow",
)

PUBLIC_API_TIERS = {
    STABLE: PUBLIC_API_STABLE,
    EXPERIMENTAL: PUBLIC_API_EXPERIMENTAL,
}

# Stamp the declared compatibility tier on every top-level public object.  This
# keeps runtime introspection aligned with PUBLIC_API_STABLE and
# PUBLIC_API_EXPERIMENTAL even when the implementation lives in another module.
for _public_name in PUBLIC_API_STABLE:
    api_tier(STABLE)(globals()[_public_name])
for _public_name in PUBLIC_API_EXPERIMENTAL:
    api_tier(EXPERIMENTAL)(globals()[_public_name])
del _public_name

__all__ = [
    # stability
    "EXPERIMENTAL",
    "INTERNAL",
    "LEGACY",
    "STABLE",
    "api_tier",
    "assert_tier",
    "get_tier",
    "PUBLIC_API_STABLE",
    "PUBLIC_API_EXPERIMENTAL",
    "PUBLIC_API_TIERS",
    # stable: models
    "ChuaParameters",
    "chua_parameters",
    "chua_arctan_wu2023_parameters",
    "chua_nonsmooth_parameters",
    "equilibria_arctan",
    "equilibria_nonsmooth",
    "jacobian_arctan",
    "jacobian_nonsmooth",
    "rhs_arctan",
    "rhs_nonsmooth",
    # stable: systems
    "ChaoticSystem",
    "LureSystem",
    "check_system_capability",
    "get_system",
    "known_workflows",
    "list_systems",
    "register_system",
    "requirements_for",
    # stable: basins
    "CLASS_LABELS",
    "TARGET_CLASS_IDS",
    "class_label",
    "is_target_class",
    # stable: portable IO
    "load_trajectory_csv",
    # experimental: analysis
    "ExpressionSystemDefinition",
    "ExpressionValidationError",
    "SimulationResult",
    "compile_expression_system",
    "simulate",
    "simulate_fractional",
    "AlignmentIndexResult",
    "AnalysisResult",
    "BifurcationPoint",
    "CORRELATION_DIMENSION_EVIDENCE_SCOPE",
    "CORRELATION_DIMENSION_REFERENCE_DOIS",
    "CORRELATION_SUM_NATIVE_AUTO_MIN_PAIRS",
    "CorrelationDimensionResult",
    "CorrelationSumResult",
    "CovariantAngleResult",
    "CovariantLyapunovResult",
    "CovariantQRHistoryResult",
    "OrdinalPatternDistribution",
    "PERMUTATION_ENTROPY_EVIDENCE_SCOPE",
    "PERMUTATION_ENTROPY_MAX_EMBEDDING_DIMENSION",
    "PERMUTATION_ENTROPY_MAX_PATTERN_STATES",
    "PERMUTATION_ENTROPY_NATIVE_AUTO_MIN_WINDOWS",
    "PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS",
    "PERMUTATION_ENTROPY_REFERENCE_DOIS",
    "PermutationEntropyResult",
    "LyapunovComputationRequest",
    "LyapunovComputationSummary",
    "LyapunovResult",
    "PoincareCrossingResult",
    "PrehistorySpec",
    "RobustnessCase",
    "SpectrumResult",
    "TimeSeriesLyapunovResult",
    "TrajectoryInput",
    "alignment_indices_from_tangent_history",
    "bifurcation_points_from_trajectories",
    "bifurcation_summary",
    "compute_boundedness_metrics",
    "compute_fft_psd",
    "compute_lyapunov_spectrum",
    "compute_trajectory_metrics",
    "correlation_sum_curve",
    "covariant_lyapunov_angles",
    "detect_poincare_crossings",
    "estimate_correlation_dimension",
    "estimate_time_series_lyapunov",
    "fft_spectrum",
    "fit_correlation_dimension",
    "generalized_alignment_index",
    "integer_flow_alignment_indices",
    "integer_flow_covariant_lyapunov_vectors",
    "integer_covariant_vectors_from_qr_history",
    "integer_map_alignment_indices",
    "integer_map_covariant_lyapunov_vectors",
    "integer_qr_benettin_lyapunov_exponents",
    "integer_system_alignment_indices",
    "integer_system_covariant_lyapunov_vectors",
    "integer_system_lyapunov_exponents",
    "kaplan_yorke_dimension",
    "linear_dependence_index",
    "ordinal_pattern_distribution",
    "permutation_entropy",
    "permutation_entropy_from_distribution",
    "psd_welch",
    "smaller_alignment_index",
    "trajectory_component_spectra",
    "trajectory_metrics",
    "trajectory_metrics_for_system",
    "validate_lyapunov_method_request",
    "zero_one_test",
    # experimental: seed_generation
    "HarmonicSeed",
    "find_harmonic_seed",
    "find_lure_harmonic_seed",
    "find_lure_omega_gain_candidates",
    "find_omega_gain_candidates",
    "validate_fractional_order",
    # experimental: workflows
    "BasinSliceSpec",
    "DestinationClassifierSpec",
    "FullWorkflowContract",
    "ContinuationPlan",
    "ContinuationTrace",
    "DynamicReference",
    "FINAL_LABELS",
    "HiddennessTestResult",
    "IntegratorSpec",
    "NumericalContract",
    "OFFICIAL_STAGE_ORDER",
    "ParameterSweepSpec",
    "RobustnessCaseSpec",
    "RobustnessVerdict",
    "PROTOCOL_VERSION",
    "PostContinuationDecision",
    "SEED_FAMILIES",
    "SoftPrecheckResult",
    "SphereControlSpec",
    "StageEnvelope",
    "StrictRefinementSpec",
    "TargetReferenceSpec",
    "TrajectoryDiagnosticsSpec",
    "WorkflowInputSpec",
    "UnifiedSeedRecord",
    "continue_integer_lure_seed",
    "final_integer_lure_attractor",
    "integer_lure_seed",
    "integrate_integer_lure",
    "run_integer_lure_hiddenness_controls",
    "validate_full_workflow_system",
    "load_config",
    "save_effective_config",
    "run_attractor_only_workflow",
    "run_bifurcation_workflow",
    "run_basin_workflow",
    "run_simple_workflow",
]
