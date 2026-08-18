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

from importlib import import_module

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
    "AdvancedRecurrenceMatrix",
    "AdvancedRQAResult",
    "AlignmentIndexResult",
    "AnalysisResult",
    "BASIN_ENTROPY_REFERENCE",
    "BasinEntropyResult",
    "BifurcationPoint",
    "CORRELATION_DIMENSION_EVIDENCE_SCOPE",
    "CORRELATION_DIMENSION_REFERENCE_DOIS",
    "CORRELATION_SUM_NATIVE_AUTO_MIN_PAIRS",
    "CorrelationDimensionResult",
    "CorrelationSumResult",
    "CovariantAngleResult",
    "CovariantLyapunovResult",
    "CovariantQRHistoryResult",
    "DelayEstimateResult",
    "FDE_RECONSTRUCTION_CAVEAT",
    "FNNDimensionResult",
    "FalseNearestNeighborsResult",
    "GeneralizedEmbeddingResult",
    "INDEX_LAG_CAVEAT",
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
    "RecurrenceQuantificationResult",
    "SpectrumResult",
    "TimeSeriesLyapunovResult",
    "TrajectoryInput",
    "UNCERTAINTY_REFERENCE",
    "UncertaintyExponentResult",
    "UncertaintyFractionResult",
    "auto_recurrence_matrix",
    "basin_entropy",
    "alignment_indices_from_tangent_history",
    "bifurcation_points_from_trajectories",
    "bifurcation_summary",
    "compute_boundedness_metrics",
    "compute_fft_psd",
    "compute_lyapunov_spectrum",
    "compute_trajectory_metrics",
    "cross_recurrence_matrix",
    "correlation_sum_curve",
    "covariant_lyapunov_angles",
    "detect_poincare_crossings",
    "delay_embedding",
    "estimate_correlation_dimension",
    "estimate_delay_autocorrelation",
    "estimate_delay_mutual_information",
    "estimate_uncertainty_exponent",
    "estimate_time_series_lyapunov",
    "fft_spectrum",
    "fit_correlation_dimension",
    "false_nearest_neighbors",
    "generalized_alignment_index",
    "generalized_delay_embedding",
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
    "joint_recurrence_matrix",
    "linear_dependence_index",
    "ordinal_pattern_distribution",
    "permutation_entropy",
    "permutation_entropy_from_distribution",
    "psd_welch",
    "recurrence_matrix",
    "recurrence_quantification",
    "recurrence_quantification_advanced",
    "smaller_alignment_index",
    "trajectory_component_spectra",
    "trajectory_metrics",
    "trajectory_metrics_for_system",
    "validate_lyapunov_method_request",
    "uncertainty_fraction",
    "zero_one_test",
    "ExternalTool",
    "available_complexity_backends",
    "compute_complexity_measures",
    "external_tool_report",
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

_MODEL_EXPORTS = PUBLIC_API_STABLE[:10]
_SYSTEM_EXPORTS = (
    "ChaoticSystem",
    "LureSystem",
    "get_system",
    "list_systems",
    "register_system",
    "ExpressionSystemDefinition",
    "ExpressionValidationError",
    "compile_expression_system",
)
_REQUIREMENT_EXPORTS = (
    "check_system_capability",
    "known_workflows",
    "requirements_for",
)
_BASIN_EXPORTS = ("CLASS_LABELS", "TARGET_CLASS_IDS", "class_label", "is_target_class")
_SIMULATION_EXPORTS = ("SimulationResult", "simulate", "simulate_fractional")
_INTEGRATION_EXPORTS = (
    "ExternalTool",
    "available_complexity_backends",
    "compute_complexity_measures",
    "external_tool_report",
)
_SEED_EXPORTS = (
    "HarmonicSeed",
    "find_harmonic_seed",
    "find_lure_harmonic_seed",
    "find_lure_omega_gain_candidates",
    "find_omega_gain_candidates",
    "validate_fractional_order",
)
_ANALYSIS_EXPORTS = PUBLIC_API_EXPERIMENTAL[
    PUBLIC_API_EXPERIMENTAL.index("AdvancedRecurrenceMatrix") :
    PUBLIC_API_EXPERIMENTAL.index("zero_one_test") + 1
]
_WORKFLOW_EXPORTS = PUBLIC_API_EXPERIMENTAL[
    PUBLIC_API_EXPERIMENTAL.index("BasinSliceSpec") :
]

# Importing the package must only establish its public contract.  Numerical
# stacks, optional adapters, and workflows are loaded when a symbol is first
# requested.  Besides improving startup time, this prevents import-time JIT or
# backend discovery from becoming a hidden package side effect.
_LAZY_EXPORTS = {
    **{name: ".models.chua" for name in _MODEL_EXPORTS},
    **{name: ".models.chua" for name in (
        "chua_piecewise_parameters",
        "equilibria_piecewise",
        "jacobian_piecewise",
        "rhs_piecewise",
    )},
    **{name: ".systems" for name in _SYSTEM_EXPORTS},
    **{name: ".systems.requirements" for name in _REQUIREMENT_EXPORTS},
    **{name: ".basins" for name in _BASIN_EXPORTS},
    "load_trajectory_csv": ".io",
    **{name: ".simulation" for name in _SIMULATION_EXPORTS},
    **{name: ".analysis" for name in _ANALYSIS_EXPORTS},
    **{name: ".integrations" for name in _INTEGRATION_EXPORTS},
    **{name: ".seed_generation" for name in _SEED_EXPORTS},
    **{name: ".workflows" for name in _WORKFLOW_EXPORTS},
}


def __getattr__(name: str):
    """Resolve a declared top-level export on first access."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    if name in PUBLIC_API_STABLE:
        api_tier(STABLE)(value)
    elif name in PUBLIC_API_EXPERIMENTAL:
        api_tier(EXPERIMENTAL)(value)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy exports in interactive discovery."""

    return sorted(set(globals()) | set(_LAZY_EXPORTS))

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
    "AdvancedRecurrenceMatrix",
    "AdvancedRQAResult",
    "AlignmentIndexResult",
    "AnalysisResult",
    "BASIN_ENTROPY_REFERENCE",
    "BasinEntropyResult",
    "BifurcationPoint",
    "CORRELATION_DIMENSION_EVIDENCE_SCOPE",
    "CORRELATION_DIMENSION_REFERENCE_DOIS",
    "CORRELATION_SUM_NATIVE_AUTO_MIN_PAIRS",
    "CorrelationDimensionResult",
    "CorrelationSumResult",
    "CovariantAngleResult",
    "CovariantLyapunovResult",
    "CovariantQRHistoryResult",
    "DelayEstimateResult",
    "FDE_RECONSTRUCTION_CAVEAT",
    "FNNDimensionResult",
    "FalseNearestNeighborsResult",
    "GeneralizedEmbeddingResult",
    "INDEX_LAG_CAVEAT",
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
    "RecurrenceQuantificationResult",
    "SpectrumResult",
    "TimeSeriesLyapunovResult",
    "TrajectoryInput",
    "UNCERTAINTY_REFERENCE",
    "UncertaintyExponentResult",
    "UncertaintyFractionResult",
    "auto_recurrence_matrix",
    "basin_entropy",
    "alignment_indices_from_tangent_history",
    "bifurcation_points_from_trajectories",
    "bifurcation_summary",
    "compute_boundedness_metrics",
    "compute_fft_psd",
    "compute_lyapunov_spectrum",
    "compute_trajectory_metrics",
    "cross_recurrence_matrix",
    "correlation_sum_curve",
    "covariant_lyapunov_angles",
    "detect_poincare_crossings",
    "delay_embedding",
    "estimate_correlation_dimension",
    "estimate_delay_autocorrelation",
    "estimate_delay_mutual_information",
    "estimate_uncertainty_exponent",
    "estimate_time_series_lyapunov",
    "fft_spectrum",
    "fit_correlation_dimension",
    "false_nearest_neighbors",
    "generalized_alignment_index",
    "generalized_delay_embedding",
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
    "joint_recurrence_matrix",
    "linear_dependence_index",
    "ordinal_pattern_distribution",
    "permutation_entropy",
    "permutation_entropy_from_distribution",
    "psd_welch",
    "recurrence_matrix",
    "recurrence_quantification",
    "recurrence_quantification_advanced",
    "smaller_alignment_index",
    "trajectory_component_spectra",
    "trajectory_metrics",
    "trajectory_metrics_for_system",
    "validate_lyapunov_method_request",
    "uncertainty_fraction",
    "zero_one_test",
    "ExternalTool",
    "available_complexity_backends",
    "compute_complexity_measures",
    "external_tool_report",
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
