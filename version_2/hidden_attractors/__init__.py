"""Numerical tools for hidden-attractor studies in fractional-order systems.

The package collects reusable pieces that were previously spread across
experiment scripts: Chua models, Caputo/EFORK native backends, trajectory
diagnostics, candidate loading, plotting, and process-safe IO helpers.

The package is intentionally conservative: harmonic-balance and describing
function objects are treated as seed generators, while hiddenness and
robustness are always numerical post-checks on the causal Caputo model.
"""

from .analysis import RobustnessCase, trajectory_metrics
from .basins import CLASS_LABELS, TARGET_CLASS_IDS, class_label, is_target_class
from .candidates import CandidateRecord, load_final_candidate_records
from .io import load_trajectory_csv
from .models.chua import ChuaParameters, chua_parameters, chua_piecewise_parameters, equilibria_piecewise, rhs_piecewise
from .seed_generation import HarmonicSeed, find_harmonic_seed, find_omega_gain_candidates, validate_fractional_order

__all__ = [
    "CLASS_LABELS",
    "CandidateRecord",
    "ChuaParameters",
    "HarmonicSeed",
    "RobustnessCase",
    "TARGET_CLASS_IDS",
    "class_label",
    "chua_piecewise_parameters",
    "chua_parameters",
    "equilibria_piecewise",
    "find_harmonic_seed",
    "find_omega_gain_candidates",
    "is_target_class",
    "load_trajectory_csv",
    "load_final_candidate_records",
    "rhs_piecewise",
    "trajectory_metrics",
    "validate_fractional_order",
]
