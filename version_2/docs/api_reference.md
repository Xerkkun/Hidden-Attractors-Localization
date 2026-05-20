# API Reference

This is the current stable public surface. Scripts in `tools/legacy/` are not
preferred public API, but they are available through `hidden_attractors.legacy`
and installable compatibility commands.

## Top-Level Imports

```python
from hidden_attractors import (
    BasinSliceSpec,
    ChuaParameters,
    CandidateRecord,
    ChaoticSystem,
    DestinationClassifierSpec,
    HarmonicSeed,
    IntegratorSpec,
    LureSystem,
    NumericalContract,
    ParameterSweepSpec,
    RobustnessCaseSpec,
    RobustnessCase,
    SphereControlSpec,
    StrictRefinementSpec,
    TargetReferenceSpec,
    TrajectoryDiagnosticsSpec,
    WorkflowInputSpec,
    check_system_capability,
    chua_parameters,
    chua_piecewise_parameters,
    continue_integer_lure_seed,
    equilibria_piecewise,
    find_harmonic_seed,
    find_lure_harmonic_seed,
    find_lure_omega_gain_candidates,
    find_omega_gain_candidates,
    get_system,
    integer_lure_seed,
    integer_system_lyapunov_exponents,
    known_workflows,
    list_systems,
    load_trajectory_csv,
    load_final_candidate_records,
    register_system,
    requirements_for,
    rhs_piecewise,
    run_integer_lure_hiddenness_controls,
    trajectory_metrics,
    trajectory_metrics_for_system,
    validate_fractional_order,
)
```

## Models

- `hidden_attractors.models.ChuaParameters`
- `hidden_attractors.models.chua_parameters`
- `hidden_attractors.models.chua_piecewise_parameters`
- `hidden_attractors.models.equilibria_piecewise`
- `hidden_attractors.models.rhs_piecewise`

## Systems

- `hidden_attractors.systems.ChaoticSystem`
- `hidden_attractors.systems.LureSystem`
- `hidden_attractors.systems.register_system`
- `hidden_attractors.systems.get_system`
- `hidden_attractors.systems.list_systems`
- `hidden_attractors.systems.known_workflows`
- `hidden_attractors.systems.requirements_for`
- `hidden_attractors.systems.check_system_capability`

The system registry is the extension point for adding new chaotic systems.
Built-ins currently include `chua-piecewise` and `chua-arctan`.
`ChaoticSystem` can expose a vector field, parameters, equilibria, Jacobian,
workflow names, and a manual `LureSystem`.  `LureSystem` is mandatory for the
full Nyquist/DF route.

`requirements_for` and `check_system_capability` document which extra inputs
are needed before a registered system can be used in sphere controls, basin
cuts, strict refinement, continuation, Lyapunov, or full hiddenness protocols.

## Seed Generation

- `hidden_attractors.seed_generation.HarmonicSeed`
- `hidden_attractors.seed_generation.find_harmonic_seed`
- `hidden_attractors.seed_generation.find_lure_harmonic_seed`
- `hidden_attractors.seed_generation.find_lure_omega_gain_candidates`
- `hidden_attractors.seed_generation.find_omega_gain_candidates`
- `hidden_attractors.seed_generation.reconstruct_biased_lure_seed`
- `hidden_attractors.seed_generation.reconstruct_biased_lure_seed_from_system`
- `hidden_attractors.seed_generation.describing_function`
- `hidden_attractors.seed_generation.machado_describing_function`
- `hidden_attractors.seed_generation.lure_describing_function`
- `hidden_attractors.seed_generation.lure_machado_describing_function`

These helpers perform light seed construction and algebraic diagnostics only.
The `lure_*` helpers are generic for systems in manually supplied Lur'e form.
The Chua-named helpers are compatibility wrappers.

## Candidates

- `hidden_attractors.candidates.CandidateRecord`
- `hidden_attractors.candidates.load_final_candidate_records`

## Analysis

- `hidden_attractors.analysis.BifurcationPoint`
- `hidden_attractors.analysis.bifurcation_points_from_trajectories`
- `hidden_attractors.analysis.bifurcation_summary`
- `hidden_attractors.analysis.local_extrema`
- `hidden_attractors.analysis.RobustnessCase`
- `hidden_attractors.analysis.default_robustness_cases`
- `hidden_attractors.analysis.trajectory_metrics`
- `hidden_attractors.analysis.trajectory_metrics_for_system`
- `hidden_attractors.analysis.cloud_median_distance`
- `hidden_attractors.analysis.integer_system_lyapunov_exponents`
- `hidden_attractors.analysis.integer_lyapunov_exponents`
- `hidden_attractors.analysis.trajectory_component_spectra`
- `hidden_attractors.analysis.fft_spectrum`
- `hidden_attractors.analysis.psd_welch`

## Plotting

- `hidden_attractors.plotting.plot_phase_space`
- `hidden_attractors.plotting.plot_phase_projections`
- `hidden_attractors.plotting.plot_time_series`
- `hidden_attractors.plotting.plot_bifurcation_diagram`
- `hidden_attractors.plotting.plot_trajectory_overlay`
- `hidden_attractors.plotting.plot_lure_nyquist_describing_function`
- `hidden_attractors.plotting.plot_integer_lure_continuation`
- `hidden_attractors.plotting.plot_integer_hiddenness_controls`
- `hidden_attractors.plotting.plot_trajectory_spectra`
- `hidden_attractors.plotting.plot_spectrum`
- `hidden_attractors.plotting.plot_lyapunov_convergence`

## IO

- `hidden_attractors.io.load_trajectory_csv`
- `hidden_attractors.io.read_csv_rows`
- `hidden_attractors.io.write_csv`
- `hidden_attractors.io.read_json`
- `hidden_attractors.io.write_json`

## External Integrations

- `hidden_attractors.integrations.external_tool_report`
- `hidden_attractors.integrations.available_complexity_backends`
- `hidden_attractors.integrations.compute_complexity_measures`
- `hidden_attractors.integrations.require_external`

The integration layer delegates optional complexity calculations to external
libraries instead of duplicating their algorithms.

## Basins

- `hidden_attractors.basins.CLASS_LABELS`
- `hidden_attractors.basins.TARGET_CLASS_IDS`
- `hidden_attractors.basins.class_label`
- `hidden_attractors.basins.is_target_class`

## Native Backends

- `hidden_attractors.native.FractionalChuaBackend`
- `hidden_attractors.native.BasinBackend`
- `hidden_attractors.native.NativeIntegrationBackend`
- `hidden_attractors.native.NativeLyapunovBackend`
- `hidden_attractors.native.IntegrationRequest`
- `hidden_attractors.native.IntegrationResult`

Current compiled C implementations are Chua-specific.  The native contracts are
the reusable interface for adding system-specific integer or fractional C
engines, including Lyapunov estimators.

## Solver Contracts

- `hidden_attractors.solvers.FractionalHistory`
- `hidden_attractors.solvers.efork_q1_integrate`
- `hidden_attractors.solvers.efork_q1_step`

`FractionalHistory` stores EFORK-compatible memory windows. It does not run a
heavy Python integrator.

## Workflows

- `hidden_attractors.workflows.robustness_overlay`
- `hidden_attractors.workflows.sphere_controls`
- `hidden_attractors.workflows.refined_basin`
- `hidden_attractors.workflows.unified_chua`
- `hidden_attractors.workflows.integer_lure`
- `hidden_attractors.workflows.contracts.FullWorkflowContract`
- `hidden_attractors.workflows.specs.WorkflowInputSpec`
- `hidden_attractors.workflows.specs.IntegratorSpec`
- `hidden_attractors.workflows.specs.DestinationClassifierSpec`
- `hidden_attractors.workflows.specs.TargetReferenceSpec`
- `hidden_attractors.workflows.specs.SphereControlSpec`
- `hidden_attractors.workflows.specs.BasinSliceSpec`
- `hidden_attractors.workflows.specs.StrictRefinementSpec`
- `hidden_attractors.workflows.specs.TrajectoryDiagnosticsSpec`
- `hidden_attractors.workflows.specs.ParameterSweepSpec`
- `hidden_attractors.workflows.specs.RobustnessCaseSpec`
- `hidden_attractors.workflows.specs.write_workflow_spec`
- `hidden_attractors.workflows.specs.load_workflow_spec`

Use the workflow modules for reusable Python calls and `tools/cli/` or console
entry points for command-line execution. The unified Chua workflow is available
as `hidden-attractors-unified-chua` and replaces manual `$env:HIDDEN_ATTRACTORS_*`
setup for new runs.

`WorkflowInputSpec` is the shared contract for maintained CLIs and migrated
legacy scripts. It records solver, memory, classifier, target-reference,
sphere, basin, strict-refinement, trajectory-diagnostics, parameter-sweep, and
robustness-case inputs so new systems can be audited without copying
Chua/Danca-specific scripts.

`integer_lure` generalizes the Chua integer example to other integer-order
systems in Lur'e form: seed generation, continuation, final trajectory,
hiddenness controls, reusable figures, and Lyapunov smoke estimates.

## Legacy Facade

- `hidden_attractors.legacy.legacy_script_names`
- `hidden_attractors.legacy.legacy_script_path`
- `hidden_attractors.legacy.run_legacy_script`

Command form:

```bash
hidden-attractors-legacy --list
hidden-attractors-legacy extended-search --help
hidden-attractors-danca2017 --help
hidden-attractors-nyquist-pipeline --help
```

Use these commands for reproducibility while migrating reusable logic into
`hidden_attractors/`. New legacy behavior should build or load a
`WorkflowInputSpec` and write the effective spec next to output artifacts.
