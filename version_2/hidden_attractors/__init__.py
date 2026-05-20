"""Numerical tools for hidden-attractor studies in fractional-order systems.

The package collects reusable pieces that were previously spread across
experiment scripts: Chua models, Caputo/EFORK native backends, trajectory
diagnostics, candidate loading, plotting, and process-safe IO helpers.

The package is intentionally conservative: harmonic-balance and describing
function objects are treated as seed generators, while hiddenness and
robustness are always numerical post-checks on the causal Caputo model.
"""

from .analysis import (
    LyapunovResult,
    RobustnessCase,
    integer_system_lyapunov_exponents,
    trajectory_metrics,
    trajectory_metrics_for_system,
)
from .basins import CLASS_LABELS, TARGET_CLASS_IDS, class_label, is_target_class
from .candidates import CandidateRecord, load_final_candidate_records
from .io import load_trajectory_csv
from .models.chua import ChuaParameters, chua_parameters, chua_piecewise_parameters, equilibria_piecewise, rhs_piecewise
from .seed_generation import (
    HarmonicSeed,
    find_harmonic_seed,
    find_lure_harmonic_seed,
    find_lure_omega_gain_candidates,
    find_omega_gain_candidates,
    validate_fractional_order,
)
from .systems import ChaoticSystem, LureSystem, get_system, list_systems, register_system
from .systems.requirements import check_system_capability, known_workflows, requirements_for
from .workflows.contracts import FullWorkflowContract, NumericalContract, validate_full_workflow_system
from .workflows.specs import (
    BasinSliceSpec,
    DestinationClassifierSpec,
    IntegratorSpec,
    ParameterSweepSpec,
    RobustnessCaseSpec,
    SphereControlSpec,
    StrictRefinementSpec,
    TargetReferenceSpec,
    TrajectoryDiagnosticsSpec,
    WorkflowInputSpec,
)
from .workflows.integer_lure import (
    continue_integer_lure_seed,
    final_integer_lure_attractor,
    integer_lure_seed,
    integrate_integer_lure,
    run_integer_lure_hiddenness_controls,
)

__all__ = [
    "CLASS_LABELS",
    "BasinSliceSpec",
    "CandidateRecord",
    "ChaoticSystem",
    "ChuaParameters",
    "DestinationClassifierSpec",
    "HarmonicSeed",
    "FullWorkflowContract",
    "IntegratorSpec",
    "LureSystem",
    "LyapunovResult",
    "NumericalContract",
    "ParameterSweepSpec",
    "RobustnessCaseSpec",
    "RobustnessCase",
    "SphereControlSpec",
    "StrictRefinementSpec",
    "TARGET_CLASS_IDS",
    "TargetReferenceSpec",
    "TrajectoryDiagnosticsSpec",
    "WorkflowInputSpec",
    "class_label",
    "check_system_capability",
    "chua_piecewise_parameters",
    "chua_parameters",
    "continue_integer_lure_seed",
    "equilibria_piecewise",
    "final_integer_lure_attractor",
    "find_harmonic_seed",
    "find_lure_harmonic_seed",
    "find_lure_omega_gain_candidates",
    "find_omega_gain_candidates",
    "get_system",
    "integer_lure_seed",
    "integer_system_lyapunov_exponents",
    "integrate_integer_lure",
    "is_target_class",
    "known_workflows",
    "list_systems",
    "load_trajectory_csv",
    "load_final_candidate_records",
    "register_system",
    "requirements_for",
    "rhs_piecewise",
    "run_integer_lure_hiddenness_controls",
    "trajectory_metrics",
    "trajectory_metrics_for_system",
    "validate_full_workflow_system",
    "validate_fractional_order",
]
