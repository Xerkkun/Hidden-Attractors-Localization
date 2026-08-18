"""Trajectory diagnostics and hiddenness-support checks.

Stability: experimental
    Lyapunov, spectral, bifurcation, and trajectory metrics are useful and
    tested.  New diagnostic parameters may be added; function signatures may
    gain optional keyword arguments without a breaking-change warning.
"""

from importlib import import_module
import sys
from types import ModuleType

_EXPORT_GROUPS = {
    ".bifurcation": ("BifurcationPoint", "bifurcation_points_from_trajectories", "bifurcation_summary", "local_extrema"),
    ".lyapunov": ("LyapunovResult", "finite_difference_jacobian", "integer_lyapunov_exponents", "integer_qr_benettin_lyapunov_exponents", "integer_system_lyapunov_exponents"),
    ".lyapunov_adaptive": ("AdaptiveLyapunovResult", "integer_dop853_variational_qr", "integer_system_dop853_variational_qr"),
    ".alignment_indices": ("AlignmentIndexResult", "alignment_indices_from_tangent_history", "generalized_alignment_index", "integer_flow_alignment_indices", "integer_map_alignment_indices", "integer_system_alignment_indices", "linear_dependence_index", "smaller_alignment_index"),
    ".covariant_lyapunov": ("CovariantAngleResult", "CovariantLyapunovResult", "CovariantQRHistoryResult", "covariant_lyapunov_angles", "integer_covariant_vectors_from_qr_history", "integer_flow_covariant_lyapunov_vectors", "integer_map_covariant_lyapunov_vectors", "integer_system_covariant_lyapunov_vectors"),
    ".lyapunov_api": ("LyapunovComputationRequest", "LyapunovComputationSummary", "validate_lyapunov_method_request", "compute_lyapunov_spectrum"),
    ".lyapunov_fractional": ("FractionalVariationalQRConfig", "fractional_variational_abm_qr", "pack_extended_state", "unpack_extended_state"),
    ".lyapunov_cloned": ("ClonedDynamicsResult", "compute_cloned_dynamics_spectrum"),
    ".time_series_lyapunov": ("ECKMANN_METHOD", "ROSENSTEIN_METHOD", "TimeSeriesLyapunovResult", "estimate_time_series_lyapunov", "kaplan_yorke_dimension"),
    ".correlation_dimension": ("CORRELATION_DIMENSION_EVIDENCE_SCOPE", "CORRELATION_DIMENSION_REFERENCE_DOIS", "CORRELATION_SUM_NATIVE_AUTO_MIN_PAIRS", "CorrelationDimensionResult", "CorrelationSumResult", "correlation_sum_curve", "estimate_correlation_dimension", "fit_correlation_dimension"),
    ".contracts": ("AnalysisResult", "PrehistorySpec", "TrajectoryInput"),
    ".permutation_entropy": ("PERMUTATION_ENTROPY_EVIDENCE_SCOPE", "PERMUTATION_ENTROPY_MAX_EMBEDDING_DIMENSION", "PERMUTATION_ENTROPY_MAX_PATTERN_STATES", "PERMUTATION_ENTROPY_NATIVE_AUTO_MIN_WINDOWS", "PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS", "PERMUTATION_ENTROPY_REFERENCE_DOIS", "OrdinalPatternDistribution", "PermutationEntropyResult", "ordinal_pattern_distribution", "permutation_entropy", "permutation_entropy_from_distribution"),
    ".boundedness": ("ALLOWED_BOUNDEDNESS_STATUSES", "compute_boundedness_metrics"),
    ".spectral": ("ALLOWED_SPECTRAL_STATES", "SpectrumResult", "compute_fft_psd", "fft_spectrum", "infer_step", "psd_welch", "spectral_diagnostics_multicoordinate", "trajectory_component_spectra"),
    ".zero_one": ("ALLOWED_ZERO_ONE_STATES", "zero_one_multicoordinate", "zero_one_test"),
    ".poincare": ("ALLOWED_INTERPRETATION_LABELS", "PoincareCrossingResult", "detect_poincare_crossings", "summarize_poincare_points", "write_poincare_outputs"),
    ".trajectory": ("RobustnessCase", "classify_trajectory_against_equilibria", "cloud_median_distance", "component_fft", "compute_trajectory_metrics", "min_distance_to_points", "section_points", "state_view", "system_equilibria", "trajectory_metrics_for_system", "trajectory_metrics"),
    ".delay_embedding": ("DelayEstimateResult", "FDE_RECONSTRUCTION_CAVEAT", "FNNDimensionResult", "FalseNearestNeighborsResult", "GeneralizedEmbeddingResult", "INDEX_LAG_CAVEAT", "estimate_delay_autocorrelation", "estimate_delay_mutual_information", "false_nearest_neighbors", "generalized_delay_embedding"),
    ".recurrence": ("RecurrenceQuantificationResult", "delay_embedding", "recurrence_matrix", "recurrence_quantification"),
    ".recurrence_advanced": ("AdvancedRecurrenceMatrix", "AdvancedRQAResult", "auto_recurrence_matrix", "cross_recurrence_matrix", "joint_recurrence_matrix", "recurrence_quantification_advanced"),
    ".basin_uncertainty": ("BASIN_ENTROPY_REFERENCE", "UNCERTAINTY_REFERENCE", "BasinEntropyResult", "UncertaintyExponentResult", "UncertaintyFractionResult", "basin_entropy", "estimate_uncertainty_exponent", "uncertainty_fraction"),
}
_LAZY_EXPORTS = {
    name: (module_name, name)
    for module_name, names in _EXPORT_GROUPS.items()
    for name in names
}
_LAZY_EXPORTS["TIME_SERIES_LYAPUNOV_EVIDENCE_STATUS"] = (
    ".time_series_lyapunov",
    "EVIDENCE_STATUS",
)


def __getattr__(name: str):
    """Resolve an analysis symbol on first access."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


class _LazyAnalysisModule(ModuleType):
    """Keep same-named public callables stable after direct submodule imports."""

    def __getattribute__(self, name: str):
        value = ModuleType.__getattribute__(self, name)
        namespace = ModuleType.__getattribute__(self, "__dict__")
        target = namespace.get("_LAZY_EXPORTS", {}).get(name)
        if target is not None and isinstance(value, ModuleType):
            module_name, attribute_name = target
            value = getattr(
                import_module(module_name, namespace["__name__"]),
                attribute_name,
            )
            ModuleType.__setattr__(self, name, value)
        return value


sys.modules[__name__].__class__ = _LazyAnalysisModule


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

# ``__all__`` is intentionally the tested high-level analysis surface, mirrored
# by the package-level experimental API. Module-qualified numerical helpers
# remain available to internal workflows without being promoted as independent
# public contracts.
__all__ = [
    "AdvancedRecurrenceMatrix",
    "AdvancedRQAResult",
    "AlignmentIndexResult",
    "BASIN_ENTROPY_REFERENCE",
    "BasinEntropyResult",
    "BifurcationPoint",
    "AnalysisResult",
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
    "LyapunovComputationRequest",
    "LyapunovComputationSummary",
    "LyapunovResult",
    "PoincareCrossingResult",
    "PrehistorySpec",
    "PERMUTATION_ENTROPY_EVIDENCE_SCOPE",
    "PERMUTATION_ENTROPY_MAX_EMBEDDING_DIMENSION",
    "PERMUTATION_ENTROPY_MAX_PATTERN_STATES",
    "PERMUTATION_ENTROPY_NATIVE_AUTO_MIN_WINDOWS",
    "PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS",
    "PERMUTATION_ENTROPY_REFERENCE_DOIS",
    "OrdinalPatternDistribution",
    "PermutationEntropyResult",
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
    "bifurcation_points_from_trajectories",
    "bifurcation_summary",
    "alignment_indices_from_tangent_history",
    "compute_boundedness_metrics",
    "compute_fft_psd",
    "compute_lyapunov_spectrum",
    "compute_trajectory_metrics",
    "detect_poincare_crossings",
    "delay_embedding",
    "correlation_sum_curve",
    "cross_recurrence_matrix",
    "covariant_lyapunov_angles",
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
    "psd_welch",
    "recurrence_matrix",
    "recurrence_quantification",
    "recurrence_quantification_advanced",
    "smaller_alignment_index",
    "ordinal_pattern_distribution",
    "permutation_entropy",
    "permutation_entropy_from_distribution",
    "trajectory_component_spectra",
    "trajectory_metrics",
    "trajectory_metrics_for_system",
    "validate_lyapunov_method_request",
    "uncertainty_fraction",
    "zero_one_test",
]
