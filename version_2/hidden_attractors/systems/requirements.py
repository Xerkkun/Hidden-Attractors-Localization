"""Capability checks for adapting workflows to new chaotic systems.

The package is intended to grow beyond the historical Chua/Danca scripts.  This
module records, in code, which mathematical ingredients must be supplied by a
new system before a workflow can be treated as reusable rather than
system-specific.

Validity warning:
    These checks verify that required hooks are present.  They do not certify
    that the chosen model, solver, or classification thresholds are
    scientifically adequate for a publication claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from .base import ChaoticSystem


WorkflowName = Literal[
    "equilibria",
    "matignon",
    "describing-function",
    "continuation",
    "sphere-controls",
    "basin",
    "strict-refinement",
    "robustness",
    "bifurcation",
    "trajectory-diagnostics",
    "lyapunov",
    "full-hiddenness-protocol",
]


@dataclass(frozen=True)
class Requirement:
    """One explicit input required by a workflow.

    Attributes:
        key:
            Stable identifier used by documentation and CLIs.
        description:
            What the user must provide for a new system.
        add_where:
            File or package area where this information should be added.
        why:
            Mathematical reason the input is required.
        optional:
            Whether the workflow can run with a documented fallback.
    """

    key: str
    description: str
    add_where: str
    why: str
    optional: bool = False


@dataclass(frozen=True)
class CapabilityReport:
    """Result of checking a registered system against workflow requirements."""

    system_name: str
    workflow: WorkflowName
    ok: bool
    missing: tuple[Requirement, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_lines(self) -> list[str]:
        """Return a human-readable report for CLI output."""

        status = "ok" if self.ok else "missing-required-inputs"
        lines = [f"system={self.system_name}", f"workflow={self.workflow}", f"status={status}"]
        for item in self.missing:
            lines.append(f"missing.{item.key}={item.description} | add_where={item.add_where} | why={item.why}")
        for warning in self.warnings:
            lines.append(f"warning={warning}")
        return lines


_BASE_REQUIREMENTS: Mapping[WorkflowName, tuple[Requirement, ...]] = {
    "equilibria": (
        Requirement(
            "equilibria",
            "Function returning named equilibrium points for the active parameter set.",
            "hidden_attractors/systems/builtins.py or a user extension calling register_system(...)",
            "Hiddenness tests and equilibrium-neighborhood controls must know where local basins are sampled.",
        ),
    ),
    "matignon": (
        Requirement(
            "jacobian",
            "Analytic or validated numerical Jacobian at equilibria.",
            "ChaoticSystem(..., jacobian=...) or a workflow-specific stability adapter",
            "Fractional local stability uses eigenvalue arguments; integer stability uses the same local linearization.",
        ),
    ),
    "describing-function": (
        Requirement(
            "lure",
            "Manual Lur'e split A,b,c,psi and describing-function convention.",
            "ChaoticSystem(..., lure=LureSystem(...))",
            "Harmonic-balance seeds require a documented feedback representation; it cannot be inferred safely.",
        ),
    ),
    "continuation": (
        Requirement(
            "integrator",
            "Trajectory integrator adapter with explicit order, step, memory/history policy, and output columns.",
            "hidden_attractors/workflows/specs.py WorkflowInputSpec or a system-specific workflow wrapper",
            "Continuation transports numerical states or memory windows; solver semantics must be explicit.",
        ),
    ),
    "sphere-controls": (
        Requirement(
            "equilibria",
            "Named equilibria to center the spheres.",
            "ChaoticSystem.equilibria",
            "Sphere controls test whether equilibrium neighborhoods intersect a target basin.",
        ),
        Requirement(
            "integrator",
            "Integrator/classifier for each sphere initial condition.",
            "WorkflowInputSpec.integrator plus DestinationClassifierSpec",
            "The workflow must classify destinations under a recorded numerical contract.",
        ),
        Requirement(
            "target-reference",
            "Candidate attractor reference or target-label policy.",
            "WorkflowInputSpec.target_reference",
            "Target hits are meaningful only relative to an explicitly chosen candidate.",
        ),
    ),
    "basin": (
        Requirement(
            "basin-slice",
            "Plane/grid definition, fixed coordinates, and destination classifier.",
            "WorkflowInputSpec.basin",
            "A basin plot is a sampled initial-condition set, not an intrinsic object without a cut.",
        ),
    ),
    "strict-refinement": (
        Requirement(
            "trajectory-integrator",
            "Integrator returning full or sampled trajectories after transients.",
            "WorkflowInputSpec.integrator",
            "Strict refinement compares tail clouds, ranges, spectra, and optional Poincare sections.",
        ),
        Requirement(
            "target-reference",
            "Positive reference trajectory/seed and symmetry policy for negative reference.",
            "WorkflowInputSpec.target_reference",
            "Similarity scores require target geometry, not only a class label.",
        ),
        Requirement(
            "negative-controls",
            "Equilibrium-neighborhood controls or a documented reason they are disabled.",
            "WorkflowInputSpec.strict_refinement.negative_control_equilibria and controls_enabled",
            "False target matches are reduced by comparing against local equilibrium-neighborhood behavior.",
            optional=True,
        ),
    ),
    "robustness": (
        Requirement(
            "integrator",
            "Base solver contract plus allowed perturbations of h, q, memory, horizon, or parameters.",
            "WorkflowInputSpec.integrator and WorkflowInputSpec.robustness_cases",
            "Robustness compares the same candidate under controlled numerical/model changes.",
        ),
        Requirement(
            "target-reference",
            "Candidate seed or reference trajectory to compare across cases.",
            "WorkflowInputSpec.target_reference",
            "Persistence is only meaningful relative to a named candidate or reference trajectory.",
        ),
        Requirement(
            "trajectory-window",
            "Post-transient sampling window and trajectory metrics.",
            "WorkflowInputSpec.trajectory_diagnostics",
            "Robustness should compare tails after a declared burn-in, not arbitrary transient segments.",
        ),
    ),
    "bifurcation": (
        Requirement(
            "integrator",
            "Integrator contract for each parameter value.",
            "WorkflowInputSpec.integrator",
            "A bifurcation diagram is a repeated numerical experiment with one solver contract.",
        ),
        Requirement(
            "sweep-parameter",
            "Parameter name, range, step/list, and continuation or independent-seed policy.",
            "WorkflowInputSpec.parameter_sweep",
            "The sweep variable and seed transport policy define the sampled diagram.",
        ),
        Requirement(
            "observable",
            "Observable, extrema/sampling rule, and post-transient window.",
            "WorkflowInputSpec.trajectory_diagnostics",
            "The plotted points depend on which component and tail extractor are used.",
        ),
    ),
    "trajectory-diagnostics": (
        Requirement(
            "trajectory-window",
            "Burn-in, retained time interval, state columns, and optional section/spectrum parameters.",
            "WorkflowInputSpec.trajectory_diagnostics",
            "Metrics, FFTs, sections, and cloud distances must use the same declared tail window.",
        ),
    ),
    "lyapunov": (
        Requirement(
            "variational-or-tangent-policy",
            "Variational equations, tangent approximation, or external estimator contract.",
            "analysis/lyapunov.py adapter or workflow-specific metadata",
            "Lyapunov estimates for fractional systems are approximate unless the tangent/history treatment is explicit.",
        ),
    ),
    "full-hiddenness-protocol": (
        Requirement(
            "equilibria",
            "Named equilibria for the active parameter set.",
            "ChaoticSystem.equilibria",
            "Hidden/self-excited distinction is defined relative to equilibrium neighborhoods.",
        ),
        Requirement(
            "jacobian",
            "Analytic or validated numerical Jacobian for local stability classification.",
            "ChaoticSystem.jacobian or a workflow-specific stability adapter",
            "Equilibrium controls should be interpreted together with local stability data.",
        ),
        Requirement(
            "integrator",
            "Numerical solver with explicit order and memory/history policy.",
            "WorkflowInputSpec.integrator",
            "All evidence must share one numerical contract unless a comparison is intentionally documented.",
        ),
        Requirement(
            "target-reference",
            "Candidate seed/reference attractor and target-label criterion.",
            "WorkflowInputSpec.target_reference",
            "The verifier must know what it means to reach the candidate.",
        ),
        Requirement(
            "destination-classifier",
            "Finite-time classifier for target, other attractor, infinity, equilibrium, and unknown outcomes.",
            "WorkflowInputSpec.classifier",
            "Equilibrium-neighborhood tests require a declared destination policy.",
        ),
        Requirement(
            "equilibrium-neighborhood-controls",
            "Sphere or alternative local sampler around every relevant equilibrium.",
            "WorkflowInputSpec.sphere_controls or a documented workflow-specific neighborhood sampler",
            "Hidden/self-excited evidence is defined by whether equilibrium neighborhoods reach the candidate.",
        ),
        Requirement(
            "trajectory-window",
            "Post-transient window used for target comparison and diagnostic plots.",
            "WorkflowInputSpec.trajectory_diagnostics",
            "Hiddenness evidence must state which part of the trajectory is compared.",
        ),
    ),
}


def requirements_for(workflow: WorkflowName) -> tuple[Requirement, ...]:
    """Return documented requirements for one workflow."""

    return _BASE_REQUIREMENTS[workflow]


def known_workflows() -> tuple[WorkflowName, ...]:
    """Return workflow names understood by the capability checker."""

    return tuple(_BASE_REQUIREMENTS)


def check_system_capability(system: ChaoticSystem, workflow: WorkflowName) -> CapabilityReport:
    """Check package-level hooks available directly on ``system``.

    Solver and candidate-reference checks are intentionally reported as missing
    unless supplied through a workflow spec.  This prevents a registered RHS
    from being mistaken for a complete hiddenness protocol.
    """

    missing: list[Requirement] = []
    warnings: list[str] = []
    for requirement in requirements_for(workflow):
        key = requirement.key
        if key == "equilibria" and system.equilibria is None:
            missing.append(requirement)
        elif key == "jacobian" and system.jacobian is None:
            missing.append(requirement)
        elif key == "lure" and system.lure is None:
            missing.append(requirement)
        elif key in {
            "integrator",
            "target-reference",
            "basin-slice",
            "trajectory-integrator",
            "destination-classifier",
            "equilibrium-neighborhood-controls",
            "trajectory-window",
            "sweep-parameter",
            "observable",
            "variational-or-tangent-policy",
        }:
            missing.append(requirement)
        elif key == "negative-controls" and system.equilibria is None:
            missing.append(requirement)
    if workflow == "matignon" and system.jacobian is not None:
        warnings.append("Fractional Matignon checks require q in the workflow spec; integer stability uses the order-one criterion.")
    return CapabilityReport(
        system_name=system.name,
        workflow=workflow,
        ok=not any(not item.optional for item in missing),
        missing=tuple(missing),
        warnings=tuple(warnings),
    )


__all__ = [
    "CapabilityReport",
    "Requirement",
    "WorkflowName",
    "check_system_capability",
    "known_workflows",
    "requirements_for",
]
