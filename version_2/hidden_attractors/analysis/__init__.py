"""Trajectory diagnostics and hiddenness-support checks.

Stability: experimental
    Lyapunov, spectral, bifurcation, and trajectory metrics are useful and
    tested.  New diagnostic parameters may be added; function signatures may
    gain optional keyword arguments without a breaking-change warning.
"""

from .bifurcation import (
    BifurcationPoint,
    bifurcation_points_from_trajectories,
    bifurcation_summary,
    local_extrema,
)
from .lyapunov import (
    LyapunovResult,
    finite_difference_jacobian,
    integer_lyapunov_exponents,
    integer_qr_benettin_lyapunov_exponents,
    integer_system_lyapunov_exponents,
)
from .lyapunov_adaptive import (
    AdaptiveLyapunovResult,
    integer_dop853_variational_qr,
    integer_system_dop853_variational_qr,
)
from .alignment_indices import (
    AlignmentIndexResult,
    alignment_indices_from_tangent_history,
    generalized_alignment_index,
    integer_flow_alignment_indices,
    integer_map_alignment_indices,
    integer_system_alignment_indices,
    linear_dependence_index,
    smaller_alignment_index,
)
from .covariant_lyapunov import (
    CovariantAngleResult,
    CovariantLyapunovResult,
    CovariantQRHistoryResult,
    covariant_lyapunov_angles,
    integer_covariant_vectors_from_qr_history,
    integer_flow_covariant_lyapunov_vectors,
    integer_map_covariant_lyapunov_vectors,
    integer_system_covariant_lyapunov_vectors,
)
from .lyapunov_api import (
    LyapunovComputationRequest,
    LyapunovComputationSummary,
    validate_lyapunov_method_request,
    compute_lyapunov_spectrum,
)
from .lyapunov_fractional import (
    FractionalVariationalQRConfig,
    fractional_variational_abm_qr,
    pack_extended_state,
    unpack_extended_state,
)
from .lyapunov_cloned import ClonedDynamicsResult, compute_cloned_dynamics_spectrum
from .time_series_lyapunov import (
    ECKMANN_METHOD,
    EVIDENCE_STATUS as TIME_SERIES_LYAPUNOV_EVIDENCE_STATUS,
    ROSENSTEIN_METHOD,
    TimeSeriesLyapunovResult,
    estimate_time_series_lyapunov,
    kaplan_yorke_dimension,
)
from .correlation_dimension import (
    CORRELATION_DIMENSION_EVIDENCE_SCOPE,
    CORRELATION_DIMENSION_REFERENCE_DOIS,
    CORRELATION_SUM_NATIVE_AUTO_MIN_PAIRS,
    CorrelationDimensionResult,
    CorrelationSumResult,
    correlation_sum_curve,
    estimate_correlation_dimension,
    fit_correlation_dimension,
)
from .contracts import AnalysisResult, PrehistorySpec, TrajectoryInput
from .permutation_entropy import (
    PERMUTATION_ENTROPY_EVIDENCE_SCOPE,
    PERMUTATION_ENTROPY_MAX_EMBEDDING_DIMENSION,
    PERMUTATION_ENTROPY_MAX_PATTERN_STATES,
    PERMUTATION_ENTROPY_NATIVE_AUTO_MIN_WINDOWS,
    PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS,
    PERMUTATION_ENTROPY_REFERENCE_DOIS,
    OrdinalPatternDistribution,
    PermutationEntropyResult,
    ordinal_pattern_distribution,
    permutation_entropy,
    permutation_entropy_from_distribution,
)
from .boundedness import ALLOWED_BOUNDEDNESS_STATUSES, compute_boundedness_metrics
from .spectral import (
    ALLOWED_SPECTRAL_STATES,
    SpectrumResult,
    compute_fft_psd,
    fft_spectrum,
    infer_step,
    psd_welch,
    spectral_diagnostics_multicoordinate,
    trajectory_component_spectra,
)
from .zero_one import ALLOWED_ZERO_ONE_STATES, zero_one_multicoordinate, zero_one_test
from .poincare import (
    ALLOWED_INTERPRETATION_LABELS,
    PoincareCrossingResult,
    detect_poincare_crossings,
    summarize_poincare_points,
    write_poincare_outputs,
)
from .trajectory import (
    RobustnessCase,
    classify_trajectory_against_equilibria,
    cloud_median_distance,
    component_fft,
    compute_trajectory_metrics,
    min_distance_to_points,
    section_points,
    state_view,
    system_equilibria,
    trajectory_metrics_for_system,
    trajectory_metrics,
)

# ``__all__`` is intentionally the tested high-level analysis surface, mirrored
# by the package-level experimental API. Module-qualified numerical helpers
# remain available to internal workflows without being promoted as independent
# public contracts.
__all__ = [
    "AlignmentIndexResult",
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
    "SpectrumResult",
    "TimeSeriesLyapunovResult",
    "TrajectoryInput",
    "bifurcation_points_from_trajectories",
    "bifurcation_summary",
    "alignment_indices_from_tangent_history",
    "compute_boundedness_metrics",
    "compute_fft_psd",
    "compute_lyapunov_spectrum",
    "compute_trajectory_metrics",
    "detect_poincare_crossings",
    "correlation_sum_curve",
    "covariant_lyapunov_angles",
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
    "psd_welch",
    "smaller_alignment_index",
    "ordinal_pattern_distribution",
    "permutation_entropy",
    "permutation_entropy_from_distribution",
    "trajectory_component_spectra",
    "trajectory_metrics",
    "trajectory_metrics_for_system",
    "validate_lyapunov_method_request",
    "zero_one_test",
]
