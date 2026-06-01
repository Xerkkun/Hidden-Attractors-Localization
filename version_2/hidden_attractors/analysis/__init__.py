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
from .spectral import SpectrumResult, fft_spectrum, infer_step, psd_welch, trajectory_component_spectra
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
    "SpectrumResult",
    "fft_spectrum",
    "infer_step",
    "psd_welch",
    "trajectory_component_spectra",
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
