"""Small command-line helpers for installed users.

Stability: internal
    These functions back the ``hidden-attractors-*`` console scripts.  The
    command-line interface is the stable contract; the Python function
    signatures may change between versions.
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .candidates import load_final_candidate_records
from .systems import check_system_capability, get_system, known_workflows, list_systems, requirements_for
from .workflows.specs import example_chua_fractional_spec


def list_candidates() -> None:
    """Print final candidate records using the public package API."""

    for record in load_final_candidate_records():
        print(
            f"{record.candidate_id} | route={record.route} | "
            f"q={record.q:.4f} | start={record.robust_start.tolist()} | "
            f"seed={record.seed.tolist()}"
        )


def systems(argv: Sequence[str] | None = None) -> None:
    """List or inspect registered chaotic systems."""

    parser = argparse.ArgumentParser(description="Inspect registered chaotic systems.")
    parser.add_argument("--system", help="System name to inspect.")
    parser.add_argument("--state", help="Comma-separated state where the vector field is evaluated.")
    parser.add_argument("--equilibria", action="store_true", help="Print known equilibria for the selected system.")
    args = parser.parse_args(argv)
    if not args.system:
        for name in list_systems():
            system = get_system(name)
            tags = ",".join(system.tags)
            print(f"{name} | dim={system.dimension} | tags={tags} | {system.description}")
        return
    system = get_system(args.system)
    print(f"name={system.name}")
    print(f"dimension={system.dimension}")
    print(f"description={system.description}")
    if system.workflows:
        for name, command in system.workflows.items():
            print(f"workflow.{name}={command}")
    if args.state:
        state = [float(part.strip()) for part in args.state.split(",") if part.strip()]
        print(f"rhs={system.evaluate(state).tolist()}")
    if args.equilibria:
        for name, point in system.equilibrium_points().items():
            print(f"equilibrium.{name}={point.tolist()}")


def workflow_requirements(argv: Sequence[str] | None = None) -> None:
    """Print required inputs for reusable workflows and system readiness."""

    parser = argparse.ArgumentParser(description="Inspect reusable workflow requirements for new systems.")
    parser.add_argument("--workflow", choices=list(known_workflows()), help="Workflow to inspect.")
    parser.add_argument("--system", help="Optional registered system to check.")
    parser.add_argument("--example-spec", action="store_true", help="Print a JSON-like example WorkflowInputSpec.")
    args = parser.parse_args(argv)
    if args.example_spec:
        print(json.dumps(example_chua_fractional_spec().to_jsonable(), indent=2, sort_keys=True))
        return
    workflows = [args.workflow] if args.workflow else list(known_workflows())
    for workflow in workflows:
        print(f"[{workflow}]")
        for req in requirements_for(workflow):
            optional = "optional" if req.optional else "required"
            print(f"{req.key} ({optional})")
            print(f"  description: {req.description}")
            print(f"  add_where: {req.add_where}")
            print(f"  why: {req.why}")
        if args.system:
            report = check_system_capability(get_system(args.system), workflow)
            print("  system_check:")
            for line in report.as_lines():
                print(f"    {line}")
