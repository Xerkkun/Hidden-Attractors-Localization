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
    "BifurcationPoint",
    "LyapunovComputationRequest",
    "LyapunovComputationSummary",
    "LyapunovResult",
    "PoincareCrossingResult",
    "RobustnessCase",
    "SpectrumResult",
    "TimeSeriesLyapunovResult",
    "bifurcation_points_from_trajectories",
    "bifurcation_summary",
    "compute_boundedness_metrics",
    "compute_fft_psd",
    "compute_lyapunov_spectrum",
    "compute_trajectory_metrics",
    "detect_poincare_crossings",
    "estimate_time_series_lyapunov",
    "fft_spectrum",
    "integer_qr_benettin_lyapunov_exponents",
    "integer_system_lyapunov_exponents",
    "kaplan_yorke_dimension",
    "psd_welch",
    "trajectory_component_spectra",
    "trajectory_metrics",
    "trajectory_metrics_for_system",
    "validate_lyapunov_method_request",
    "zero_one_test",
]
