"""Trajectory diagnostics and hiddenness-support checks."""

from .trajectory import (
    RobustnessCase,
    cloud_median_distance,
    component_fft,
    section_points,
    trajectory_metrics,
)

__all__ = ["RobustnessCase", "cloud_median_distance", "component_fft", "section_points", "trajectory_metrics"]
