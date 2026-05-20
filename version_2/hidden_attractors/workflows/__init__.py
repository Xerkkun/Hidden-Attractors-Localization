"""High-level reproducible workflows built on the library primitives."""

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
from .robustness_overlay import aggregate as aggregate_robustness_overlay
from .robustness_overlay import launch_independent_jobs as launch_robustness_overlay
from .robustness_overlay import run_candidate as run_robustness_overlay_candidate
from .refined_basin import aggregate as aggregate_refined_basin
from .refined_basin import launch as launch_refined_basin
from .sphere_controls import aggregate as aggregate_sphere_controls
from .sphere_controls import launch as launch_sphere_controls
from .sphere_controls import run_sphere_chunk
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
    "ContinuationResult",
    "FullWorkflowContract",
    "HiddennessResult",
    "BasinSliceSpec",
    "DestinationClassifierSpec",
    "IntegratorSpec",
    "IntegerHiddennessProbe",
    "IntegerLureContinuationStep",
    "NumericalContract",
    "ParameterSweepSpec",
    "RobustnessCaseSpec",
    "SeedResult",
    "SphereControlSpec",
    "StrictRefinementSpec",
    "TargetReferenceSpec",
    "TrajectoryDiagnosticsSpec",
    "WorkflowInputSpec",
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
    "summarize_integer_hiddenness_controls",
    "validate_full_workflow_system",
    "write_workflow_spec",
]
