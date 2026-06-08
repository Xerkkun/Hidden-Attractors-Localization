"""Plotting helpers for numerical-attractor workflows.

Stability: experimental
"""

from .dynamics import (
    plot_integer_hiddenness_controls,
    plot_integer_lure_continuation,
    plot_lyapunov_convergence,
    plot_lure_nyquist_describing_function,
    plot_lure_transfer_components,
    plot_bifurcation_diagram,
    plot_phase_projections,
    plot_phase_space,
    plot_spectrum,
    plot_time_series,
    plot_trajectory_spectra,
)
from .overlays import plot_trajectory_overlay
from .basin import plot_basin_slices, plot_basin_slice_file
from .matignon import plot_matignon_equilibria, classify_equilibrium_stability

# Migrated functions
from .plot_transfer import plot_nyquist_transfer
from .plot_df import plot_describing_function, plot_harmonic_residual_map
from .plot_continuation import (
    plot_continuation_eta,
    plot_continuation_first_last_comparison,
    plot_continuation_timeseries_comparison,
    plot_continuation_progression,
    plot_continuation_tracking,
)
from .plot_trajectories import (
    plot_attractor_trajectories,
    plot_flexible_attractor_and_projections,
    plot_timeseries_data,
    plot_neighborhood_control_spheres,
)
from .plot_basins import plot_basin_slice_file as plot_basin_slice_file_migrated
from .plot_matignon import plot_matignon_equilibria as plot_matignon_equilibria_migrated
from .plot_sphere_tests import plot_sphere_test_results
from .generate_publication_figures import generate_all_publication_figures

# Unified plotting API
from .style import apply_library_style, apply_axes_style, get_figsize
from .export import export_figure
from .renderers import render_attractor, render_basin, render_nyquist, render_matignon
from .render_all import render_all_plots

__all__ = [
    "plot_integer_hiddenness_controls",
    "plot_integer_lure_continuation",
    "plot_lyapunov_convergence",
    "plot_lure_nyquist_describing_function",
    "plot_lure_transfer_components",
    "plot_bifurcation_diagram",
    "plot_phase_projections",
    "plot_phase_space",
    "plot_spectrum",
    "plot_time_series",
    "plot_trajectory_spectra",
    "plot_trajectory_overlay",
    "plot_basin_slices",
    "plot_basin_slice_file",
    "plot_matignon_equilibria",
    "classify_equilibrium_stability",
    
    # Migrated
    "plot_nyquist_transfer",
    "plot_describing_function",
    "plot_harmonic_residual_map",
    "plot_continuation_eta",
    "plot_continuation_first_last_comparison",
    "plot_continuation_timeseries_comparison",
    "plot_continuation_progression",
    "plot_continuation_tracking",
    "plot_attractor_trajectories",
    "plot_flexible_attractor_and_projections",
    "plot_timeseries_data",
    "plot_neighborhood_control_spheres",
    "plot_sphere_test_results",
    "generate_all_publication_figures",

    # Unified API
    "apply_library_style",
    "apply_axes_style",
    "get_figsize",
    "export_figure",
    "render_attractor",
    "render_basin",
    "render_nyquist",
    "render_matignon",
    "render_all_plots",
]

