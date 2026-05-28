"""High-level reproducible workflows built on the library primitives.

Stability: experimental
    Workflow specs and entry points may evolve as new numerical stages are
    added.  Changes are noted in changelog entries; existing workflow scripts
    will not be silently broken between minor versions.
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
from .robustness_overlay import aggregate as aggregate_robustness_overlay
from .robustness_overlay import launch_independent_jobs as launch_robustness_overlay
from .robustness_overlay import run_candidate as run_robustness_overlay_candidate
from .refined_basin import aggregate as aggregate_refined_basin
from .refined_basin import launch as launch_refined_basin
from .sphere_controls import aggregate as aggregate_sphere_controls
from .sphere_controls import launch as launch_sphere_controls
from .sphere_controls import run_sphere_chunk
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
    example_chua_fractional_spec,
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
    "aggregate_refined_basin",
    "aggregate_robustness_overlay",
    "aggregate_sphere_controls",
    "continue_integer_lure_seed",
    "example_chua_fractional_spec",
    "final_integer_lure_attractor",
    "integer_lure_seed",
    "integrate_integer_lure",
    "launch_refined_basin",
    "launch_robustness_overlay",
    "launch_sphere_controls",
    "load_workflow_spec",
    "run_robustness_overlay_candidate",
    "run_sphere_chunk",
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
