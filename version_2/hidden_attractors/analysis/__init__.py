"""Trajectory diagnostics and hiddenness-support checks."""

from .bifurcation import (
    BifurcationPoint,
    bifurcation_points_from_trajectories,
    bifurcation_summary,
    local_extrema,
)
from .trajectory import (
    RobustnessCase,
    cloud_median_distance,
    component_fft,
    section_points,
    trajectory_metrics,
)

__all__ = [
    "BifurcationPoint",
    "RobustnessCase",
    "bifurcation_points_from_trajectories",
    "bifurcation_summary",
    "cloud_median_distance",
    "component_fft",
    "local_extrema",
    "section_points",
    "trajectory_metrics",
]
