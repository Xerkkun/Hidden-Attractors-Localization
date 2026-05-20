"""Small command-line helpers for installed users."""

from __future__ import annotations

import argparse
from typing import Sequence

from .candidates import load_final_candidate_records
from .systems import get_system, list_systems


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
