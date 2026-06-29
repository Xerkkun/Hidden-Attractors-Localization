"""Numerical tools for hidden-attractor studies in fractional-order systems.

Stability tiers
---------------
This package organises its public surface into four tiers.  Every sub-module
starts with a ``Stability: <tier>`` line in its docstring.

stable
    Signatures and return types are fixed.  Breaking changes require a version
    bump and a deprecation cycle.
    Modules: ``models``, ``systems``, ``basins``, ``io``, ``candidates``.

experimental
    API is useful and tested but may evolve.  Changes will be noted in a
    changelog entry.
    Modules: ``analysis``, ``seed_generation``, ``solvers``, ``plotting``,
    ``integrations``, ``workflows``.

internal
    Consumed by workflows and backends; not part of the user-facing surface.
    May change without notice.
    Modules: ``native``, ``parallel``, ``paths``, ``cli``.

legacy
    Frozen compatibility facade over historical scripts.  No new features.
    Module: ``legacy``.

See ``docs/api_stability.md`` for guarantees, upgrade-path guidance, and how
to introspect a symbol's tier programmatically.

Background
----------
The package collects reusable pieces that were previously spread across
experiment scripts: Chua models, Caputo/EFORK native backends, trajectory
diagnostics, candidate loading, plotting, and process-safe IO helpers.

The package is intentionally conservative: harmonic-balance and describing
function objects are treated as seed generators, while hiddenness and
robustness are always numerical post-checks on the causal Caputo model.
"""

# Stability constants (re-exported for convenience)
from ._stability import (  # noqa: F401
    EXPERIMENTAL,
    INTERNAL,
    LEGACY,
    STABLE,
    api_tier,
    assert_tier,
    get_tier,
)

# Stable API
# models - vector fields, parameters, equilibria
from .models.chua import (
    ChuaParameters,
    chua_arctan_wu2023_parameters,
    chua_nonsmooth_parameters,
    chua_parameters,
    equilibria_arctan,
    equilibria_nonsmooth,
    jacobian_arctan,
    jacobian_nonsmooth,
    rhs_arctan,
    rhs_nonsmooth,
    # Compatibility aliases for recorded runs created with the old label.
    chua_piecewise_parameters,
    equilibria_piecewise,
    jacobian_piecewise,
    rhs_piecewise,
)

# systems - chaotic-system registry and capability checks
from .systems import ChaoticSystem, LureSystem, get_system, list_systems, register_system
from .systems.requirements import check_system_capability, known_workflows, requirements_for

# basins - classification labels
from .basins import CLASS_LABELS, TARGET_CLASS_IDS, class_label, is_target_class

# io / candidates - filesystem helpers and reference-output loaders
from .candidates import CandidateRecord, load_final_candidate_records
from .io import load_trajectory_csv

# Experimental API
# analysis - trajectory diagnostics and Lyapunov estimates
from .analysis import (
    LyapunovResult,
    RobustnessCase,
    integer_system_lyapunov_exponents,
    trajectory_metrics,
    trajectory_metrics_for_system,
)

# seed_generation - harmonic-balance seeds (Chua + generic Lur'e)
from .seed_generation import (
    HarmonicSeed,
    find_harmonic_seed,
    find_lure_harmonic_seed,
    find_lure_omega_gain_candidates,
    find_omega_gain_candidates,
    validate_fractional_order,
)

# workflows - high-level reproducible numerical pipelines
from .workflows.contracts import FullWorkflowContract, NumericalContract, validate_full_workflow_system
from .workflows.protocol import (
    FINAL_LABELS,
    OFFICIAL_STAGE_ORDER,
    PROTOCOL_VERSION,
    SEED_FAMILIES,
    ContinuationPlan,
    ContinuationTrace,
    DynamicReference,
    HiddennessTestResult,
    PostContinuationDecision,
    RobustnessVerdict,
    SoftPrecheckResult,
    StageEnvelope,
    UnifiedSeedRecord,
)
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
from .workflows.config_loader import load_config, save_effective_config
from .workflows.attractor_only import run_attractor_only_workflow
from .workflows.bifurcation import run_bifurcation_workflow
from .workflows.basin_runner import run_basin_workflow
from .workflows.simple_runner import run_simple_workflow

PUBLIC_API_STABLE = (
    "ChuaParameters",
    "chua_parameters",
    "chua_arctan_wu2023_parameters",
    "chua_nonsmooth_parameters",
    "equilibria_arctan",
    "equilibria_nonsmooth",
    "jacobian_arctan",
    "jacobian_nonsmooth",
    "rhs_arctan",
    "rhs_nonsmooth",
    "ChaoticSystem",
    "LureSystem",
    "check_system_capability",
    "get_system",
    "known_workflows",
    "list_systems",
    "register_system",
    "requirements_for",
    "CLASS_LABELS",
    "TARGET_CLASS_IDS",
    "class_label",
    "is_target_class",
    "CandidateRecord",
    "load_final_candidate_records",
    "load_trajectory_csv",
)

PUBLIC_API_EXPERIMENTAL = (
    "LyapunovResult",
    "RobustnessCase",
    "integer_system_lyapunov_exponents",
    "trajectory_metrics",
    "trajectory_metrics_for_system",
    "HarmonicSeed",
    "find_harmonic_seed",
    "find_lure_harmonic_seed",
    "find_lure_omega_gain_candidates",
    "find_omega_gain_candidates",
    "validate_fractional_order",
    "BasinSliceSpec",
    "DestinationClassifierSpec",
    "FullWorkflowContract",
    "ContinuationPlan",
    "ContinuationTrace",
    "DynamicReference",
    "FINAL_LABELS",
    "HiddennessTestResult",
    "IntegratorSpec",
    "NumericalContract",
    "OFFICIAL_STAGE_ORDER",
    "ParameterSweepSpec",
    "RobustnessCaseSpec",
    "RobustnessVerdict",
    "PROTOCOL_VERSION",
    "PostContinuationDecision",
    "SEED_FAMILIES",
    "SoftPrecheckResult",
    "SphereControlSpec",
    "StageEnvelope",
    "StrictRefinementSpec",
    "TargetReferenceSpec",
    "TrajectoryDiagnosticsSpec",
    "WorkflowInputSpec",
    "UnifiedSeedRecord",
    "continue_integer_lure_seed",
    "final_integer_lure_attractor",
    "integer_lure_seed",
    "integrate_integer_lure",
    "run_integer_lure_hiddenness_controls",
    "validate_full_workflow_system",
    "load_config",
    "save_effective_config",
    "run_attractor_only_workflow",
    "run_bifurcation_workflow",
    "run_basin_workflow",
    "run_simple_workflow",
)

PUBLIC_API_TIERS = {
    STABLE: PUBLIC_API_STABLE,
    EXPERIMENTAL: PUBLIC_API_EXPERIMENTAL,
}
__all__ = [
    # stability
    "EXPERIMENTAL",
    "INTERNAL",
    "LEGACY",
    "STABLE",
    "api_tier",
    "assert_tier",
    "get_tier",
    "PUBLIC_API_STABLE",
    "PUBLIC_API_EXPERIMENTAL",
    "PUBLIC_API_TIERS",
    # stable: models
    "ChuaParameters",
    "chua_parameters",
    "chua_arctan_wu2023_parameters",
    "chua_nonsmooth_parameters",
    "equilibria_arctan",
    "equilibria_nonsmooth",
    "jacobian_arctan",
    "jacobian_nonsmooth",
    "rhs_arctan",
    "rhs_nonsmooth",
    # stable: systems
    "ChaoticSystem",
    "LureSystem",
    "check_system_capability",
    "get_system",
    "known_workflows",
    "list_systems",
    "register_system",
    "requirements_for",
    # stable: basins
    "CLASS_LABELS",
    "TARGET_CLASS_IDS",
    "class_label",
    "is_target_class",
    # stable: io / candidates
    "CandidateRecord",
    "load_final_candidate_records",
    "load_trajectory_csv",
    # experimental: analysis
    "LyapunovResult",
    "RobustnessCase",
    "integer_system_lyapunov_exponents",
    "trajectory_metrics",
    "trajectory_metrics_for_system",
    # experimental: seed_generation
    "HarmonicSeed",
    "find_harmonic_seed",
    "find_lure_harmonic_seed",
    "find_lure_omega_gain_candidates",
    "find_omega_gain_candidates",
    "validate_fractional_order",
    # experimental: workflows
    "BasinSliceSpec",
    "DestinationClassifierSpec",
    "FullWorkflowContract",
    "ContinuationPlan",
    "ContinuationTrace",
    "DynamicReference",
    "FINAL_LABELS",
    "HiddennessTestResult",
    "IntegratorSpec",
    "NumericalContract",
    "OFFICIAL_STAGE_ORDER",
    "ParameterSweepSpec",
    "RobustnessCaseSpec",
    "RobustnessVerdict",
    "PROTOCOL_VERSION",
    "PostContinuationDecision",
    "SEED_FAMILIES",
    "SoftPrecheckResult",
    "SphereControlSpec",
    "StageEnvelope",
    "StrictRefinementSpec",
    "TargetReferenceSpec",
    "TrajectoryDiagnosticsSpec",
    "WorkflowInputSpec",
    "UnifiedSeedRecord",
    "continue_integer_lure_seed",
    "final_integer_lure_attractor",
    "integer_lure_seed",
    "integrate_integer_lure",
    "run_integer_lure_hiddenness_controls",
    "validate_full_workflow_system",
    "load_config",
    "save_effective_config",
    "run_attractor_only_workflow",
    "run_bifurcation_workflow",
    "run_basin_workflow",
    "run_simple_workflow",
]
