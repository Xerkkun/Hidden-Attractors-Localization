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
from .lyapunov_methods import LyapunovMethodInfo, LYAPUNOV_METHODS
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
from .integrated_chaos_validator import (
    ALLOWED_INTEGRATED_STATUSES,
    CASE_Q,
    classify_lambda_max,
    integrate_case_evidence,
    method_is_applicable,
    method_registry_rows,
    normalize_lyapunov_case_evidence,
)
from .method_comparison import (
    ALLOWED_COMPARISON_STATUSES,
    classify_method_row,
    compare_f5_diagnostics,
    compare_lyapunov_methods,
)
from .phase_f_closure import (
    FRACTIONAL_CANDIDATE_IDS,
    PHASE_F_CLOSURE_RULES,
    PHASE_F_STRUCTURED_STATUS,
    assess_phase_f_closure,
    build_phase_f_closure_matrix,
)
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
    min_distance_to_points,
    section_points,
    state_view,
    system_equilibria,
    trajectory_metrics_for_system,
    trajectory_metrics,
)

__all__ = [
    # Bifurcation
    "BifurcationPoint",
    "bifurcation_points_from_trajectories",
    "bifurcation_summary",
    "local_extrema",
    # Lyapunov — F0 (frozen)
    "LyapunovResult",
    "finite_difference_jacobian",
    "integer_lyapunov_exponents",
    "integer_qr_benettin_lyapunov_exponents",
    "integer_system_lyapunov_exponents",
    # Lyapunov — method registry (F0)
    "LyapunovMethodInfo",
    "LYAPUNOV_METHODS",
    # Lyapunov — common API (F1/F2)
    "LyapunovComputationRequest",
    "LyapunovComputationSummary",
    "validate_lyapunov_method_request",
    "compute_lyapunov_spectrum",
    # Lyapunov — F2 fractional variational ABM-QR
    "FractionalVariationalQRConfig",
    "fractional_variational_abm_qr",
    "pack_extended_state",
    "unpack_extended_state",
    # Lyapunov - F3 cloned dynamics ABM GS/QR
    "ClonedDynamicsResult",
    "compute_cloned_dynamics_spectrum",
    # Spectral
    "ALLOWED_BOUNDEDNESS_STATUSES",
    "compute_boundedness_metrics",
    "ALLOWED_SPECTRAL_STATES",
    "SpectrumResult",
    "compute_fft_psd",
    "fft_spectrum",
    "infer_step",
    "psd_welch",
    "spectral_diagnostics_multicoordinate",
    "trajectory_component_spectra",
    # Zero-one diagnostic
    "ALLOWED_ZERO_ONE_STATES",
    "zero_one_multicoordinate",
    "zero_one_test",
    # Integrated diagnostics (F6/F7)
    "ALLOWED_INTEGRATED_STATUSES",
    "CASE_Q",
    "classify_lambda_max",
    "integrate_case_evidence",
    "method_is_applicable",
    "method_registry_rows",
    "normalize_lyapunov_case_evidence",
    "ALLOWED_COMPARISON_STATUSES",
    "classify_method_row",
    "compare_f5_diagnostics",
    "compare_lyapunov_methods",
    # Phase F closure
    "FRACTIONAL_CANDIDATE_IDS",
    "PHASE_F_CLOSURE_RULES",
    "PHASE_F_STRUCTURED_STATUS",
    "assess_phase_f_closure",
    "build_phase_f_closure_matrix",
    # Poincare diagnostic
    "ALLOWED_INTERPRETATION_LABELS",
    "PoincareCrossingResult",
    "detect_poincare_crossings",
    "summarize_poincare_points",
    "write_poincare_outputs",
    # Trajectory
    "RobustnessCase",
    "classify_trajectory_against_equilibria",
    "cloud_median_distance",
    "component_fft",
    "min_distance_to_points",
    "section_points",
    "state_view",
    "system_equilibria",
    "trajectory_metrics_for_system",
    "trajectory_metrics",
]
