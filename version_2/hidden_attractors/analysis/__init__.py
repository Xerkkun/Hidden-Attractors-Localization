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
    integer_system_lyapunov_exponents,
)
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
    "BifurcationPoint",
    "LyapunovResult",
    "RobustnessCase",
    "SpectrumResult",
    "bifurcation_points_from_trajectories",
    "bifurcation_summary",
    "classify_trajectory_against_equilibria",
    "cloud_median_distance",
    "component_fft",
    "finite_difference_jacobian",
    "fft_spectrum",
    "infer_step",
    "integer_lyapunov_exponents",
    "integer_system_lyapunov_exponents",
    "local_extrema",
    "min_distance_to_points",
    "psd_welch",
    "section_points",
    "state_view",
    "system_equilibria",
    "trajectory_metrics_for_system",
    "trajectory_metrics",
    "trajectory_component_spectra",
]
