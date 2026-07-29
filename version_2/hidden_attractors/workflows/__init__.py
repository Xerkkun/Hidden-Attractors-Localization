"""High-level reproducible workflows built on the library primitives.

Stability: experimental
    Workflow specs and entry points carry narrower compatibility guarantees
    than the stable model and registry interfaces. Changes are recorded in
    the changelog and follow the package deprecation policy.
"""

from .contracts import (
    ContinuationResult,
    FullWorkflowContract,
    HiddennessResult,
    NumericalContract,
    SeedResult,
    validate_full_workflow_system,
)
from .integer_lure import (
    IntegerHiddennessProbe,
    IntegerLureContinuationStep,
    continue_integer_lure_seed,
    final_integer_lure_attractor,
    integer_lure_seed,
    integrate_integer_lure,
    run_integer_lure_hiddenness_controls,
    summarize_integer_hiddenness_controls,
)
from .protocol import (
    FINAL_LABELS,
    OFFICIAL_STAGE_ORDER,
    PROTOCOL_VERSION,
    ROBUSTNESS_VERDICTS,
    SCHEMA_VERSION,
    SEED_FAMILIES,
    ContinuationPlan,
    ContinuationStep,
    ContinuationTrace,
    DynamicReference,
    HiddennessTestResult,
    PostContinuationDecision,
    RobustnessVerdict,
    SoftPrecheckResult,
    StageEnvelope,
    UnifiedSeedRecord,
    sample_uniform_ball,
)
from .config_loader import load_config, save_effective_config
from .attractor_only import run_attractor_only_workflow
from .bifurcation import run_bifurcation_workflow
from .basin_runner import run_basin_workflow
from .simple_runner import run_simple_workflow
from .specs import (
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
    example_workflow_spec,
    load_workflow_spec,
    write_workflow_spec,
)

__all__ = [
    "ContinuationPlan",
    "ContinuationResult",
    "ContinuationStep",
    "ContinuationTrace",
    "DynamicReference",
    "FINAL_LABELS",
    "FullWorkflowContract",
    "HiddennessResult",
    "HiddennessTestResult",
    "BasinSliceSpec",
    "DestinationClassifierSpec",
    "IntegratorSpec",
    "IntegerHiddennessProbe",
    "IntegerLureContinuationStep",
    "NumericalContract",
    "OFFICIAL_STAGE_ORDER",
    "ParameterSweepSpec",
    "PROTOCOL_VERSION",
    "PostContinuationDecision",
    "ROBUSTNESS_VERDICTS",
    "RobustnessCaseSpec",
    "RobustnessVerdict",
    "SCHEMA_VERSION",
    "SEED_FAMILIES",
    "SeedResult",
    "SoftPrecheckResult",
    "SphereControlSpec",
    "StageEnvelope",
    "StrictRefinementSpec",
    "TargetReferenceSpec",
    "TrajectoryDiagnosticsSpec",
    "WorkflowInputSpec",
    "UnifiedSeedRecord",
    "continue_integer_lure_seed",
    "example_workflow_spec",
    "final_integer_lure_attractor",
    "integer_lure_seed",
    "integrate_integer_lure",
    "load_workflow_spec",
    "run_integer_lure_hiddenness_controls",
    "sample_uniform_ball",
    "summarize_integer_hiddenness_controls",
    "validate_full_workflow_system",
    "write_workflow_spec",
    "load_config",
    "save_effective_config",
    "run_attractor_only_workflow",
    "run_bifurcation_workflow",
    "run_basin_workflow",
    "run_simple_workflow",
]
