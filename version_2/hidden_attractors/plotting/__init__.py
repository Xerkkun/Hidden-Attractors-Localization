"""Plotting helpers for numerical-attractor workflows.

Stability: experimental
    Plot functions may gain new keyword arguments (colors, axes, figure size)
    as the gallery grows.  Existing call signatures will not be broken without
    notice.
"""

from .dynamics import (
    plot_integer_hiddenness_controls,
    plot_integer_lure_continuation,
    plot_lyapunov_convergence,
    plot_lure_nyquist_describing_function,
    plot_bifurcation_diagram,
    plot_phase_projections,
    plot_phase_space,
    plot_spectrum,
    plot_time_series,
    plot_trajectory_spectra,
)
from .overlays import plot_trajectory_overlay

__all__ = [
    "plot_integer_hiddenness_controls",
    "plot_integer_lure_continuation",
    "plot_lyapunov_convergence",
    "plot_lure_nyquist_describing_function",
    "plot_bifurcation_diagram",
    "plot_phase_projections",
    "plot_phase_space",
    "plot_spectrum",
    "plot_time_series",
    "plot_trajectory_spectra",
    "plot_trajectory_overlay",
]
