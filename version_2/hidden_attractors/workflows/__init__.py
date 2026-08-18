"""High-level reproducible workflows built on the library primitives.

Stability: experimental
    Workflow specs and entry points carry narrower compatibility guarantees
    than the stable model and registry interfaces. Changes are recorded in
    the changelog and follow the package deprecation policy.
"""

from importlib import import_module

_EXPORT_GROUPS = {
    ".contracts": ("ContinuationResult", "FullWorkflowContract", "HiddennessResult", "NumericalContract", "SeedResult", "validate_full_workflow_system"),
    ".integer_lure": ("IntegerHiddennessProbe", "IntegerLureContinuationStep", "continue_integer_lure_seed", "final_integer_lure_attractor", "integer_lure_seed", "integrate_integer_lure", "run_integer_lure_hiddenness_controls", "summarize_integer_hiddenness_controls"),
    ".integer_hidden_chaos": ("IntegerHiddenChaosProbe", "IntegerParameterContinuationStep", "continue_integer_parameter_path", "deterministic_unit_directions", "equilibrium_stability_records", "run_integer_hidden_chaos_controls", "summarize_integer_hidden_chaos_controls"),
    ".switching_lure": ("NonlinearityContinuationStep", "SwitchingMapSeed", "continue_integer_lure_nonlinearity", "find_sign_switching_cycle_seed", "integer_lure_nonlinearity_homotopy_rhs", "sign_nonlinearity"),
    ".protocol": ("FINAL_LABELS", "OFFICIAL_STAGE_ORDER", "PROTOCOL_VERSION", "ROBUSTNESS_VERDICTS", "SCHEMA_VERSION", "SEED_FAMILIES", "ContinuationPlan", "ContinuationStep", "ContinuationTrace", "DynamicReference", "HiddennessTestResult", "PostContinuationDecision", "RobustnessVerdict", "SoftPrecheckResult", "StageEnvelope", "UnifiedSeedRecord", "sample_uniform_ball"),
    ".config_loader": ("load_config", "save_effective_config"),
    ".attractor_only": ("run_attractor_only_workflow",),
    ".bifurcation": ("run_bifurcation_workflow",),
    ".basin_runner": ("run_basin_workflow",),
    ".simple_runner": ("run_simple_workflow",),
    ".geometric_topological_campaign": ("CAMPAIGN_PROTOCOL_VERSION", "CAMPAIGN_SCHEMA_VERSION", "DEFAULT_B0_B2_BUDGETS", "CampaignArtifactPaths", "CampaignBudget", "CampaignManifest", "EdgeRunContext", "append_edge_tracking_result", "initialize_campaign_artifacts", "run_edge_tracking_and_record"),
    ".specs": ("BasinSliceSpec", "DestinationClassifierSpec", "IntegratorSpec", "ParameterSweepSpec", "RobustnessCaseSpec", "SphereControlSpec", "StrictRefinementSpec", "TargetReferenceSpec", "TrajectoryDiagnosticsSpec", "WorkflowInputSpec", "example_workflow_spec", "load_workflow_spec", "write_workflow_spec"),
}
_LAZY_EXPORTS = {
    name: module_name
    for module_name, names in _EXPORT_GROUPS.items()
    for name in names
}


def __getattr__(name: str):
    """Resolve a workflow symbol on first access."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

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
    "IntegerHiddenChaosProbe",
    "IntegerParameterContinuationStep",
    "IntegerLureContinuationStep",
    "NumericalContract",
    "NonlinearityContinuationStep",
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
    "SwitchingMapSeed",
    "TargetReferenceSpec",
    "TrajectoryDiagnosticsSpec",
    "WorkflowInputSpec",
    "UnifiedSeedRecord",
    "continue_integer_lure_seed",
    "continue_integer_parameter_path",
    "continue_integer_lure_nonlinearity",
    "example_workflow_spec",
    "final_integer_lure_attractor",
    "find_sign_switching_cycle_seed",
    "integer_lure_seed",
    "integer_lure_nonlinearity_homotopy_rhs",
    "integrate_integer_lure",
    "deterministic_unit_directions",
    "equilibrium_stability_records",
    "load_workflow_spec",
    "run_integer_lure_hiddenness_controls",
    "run_integer_hidden_chaos_controls",
    "sample_uniform_ball",
    "summarize_integer_hiddenness_controls",
    "summarize_integer_hidden_chaos_controls",
    "sign_nonlinearity",
    "validate_full_workflow_system",
    "write_workflow_spec",
    "load_config",
    "save_effective_config",
    "run_attractor_only_workflow",
    "run_bifurcation_workflow",
    "run_basin_workflow",
    "run_simple_workflow",
    "CAMPAIGN_PROTOCOL_VERSION",
    "CAMPAIGN_SCHEMA_VERSION",
    "DEFAULT_B0_B2_BUDGETS",
    "CampaignArtifactPaths",
    "CampaignBudget",
    "CampaignManifest",
    "EdgeRunContext",
    "append_edge_tracking_result",
    "initialize_campaign_artifacts",
    "run_edge_tracking_and_record",
]
