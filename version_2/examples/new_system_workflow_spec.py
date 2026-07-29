#!/usr/bin/env python3
"""Register a system and write an explicit characterization specification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hidden_attractors.systems import ChaoticSystem, check_system_capability, get_system, register_system
from hidden_attractors.workflows import (
    IntegratorSpec,
    TrajectoryDiagnosticsSpec,
    WorkflowInputSpec,
    write_workflow_spec,
)


def lorenz_rhs(state: np.ndarray, p: Mapping[str, Any]) -> np.ndarray:
    """Right-hand side for the integer-order Lorenz 63 vector field."""

    x, y, z = state
    sigma = float(p["sigma"])
    rho = float(p["rho"])
    beta = float(p["beta"])
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], dtype=float)


def lorenz_equilibria(p: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return named Lorenz equilibria for the active parameter set."""

    rho = float(p["rho"])
    beta = float(p["beta"])
    if rho <= 1.0:
        return {"E0": np.zeros(3, dtype=float)}
    scale = float(np.sqrt(beta * (rho - 1.0)))
    return {
        "E0": np.zeros(3, dtype=float),
        "E+": np.array([scale, scale, rho - 1.0], dtype=float),
        "E-": np.array([-scale, -scale, rho - 1.0], dtype=float),
    }


def lorenz_jacobian(state: np.ndarray, p: Mapping[str, Any]) -> np.ndarray:
    """Analytic Jacobian used by local stability and Lyapunov workflows."""

    x, y, z = state
    sigma = float(p["sigma"])
    rho = float(p["rho"])
    beta = float(p["beta"])
    return np.array(
        [
            [-sigma, sigma, 0.0],
            [rho - z, -1.0, -x],
            [y, x, -beta],
        ],
        dtype=float,
    )


def register_lorenz63() -> None:
    """Register Lorenz 63 as an example integer-order chaotic system."""

    register_system(
        ChaoticSystem(
            name="lorenz63",
            dimension=3,
            rhs=lorenz_rhs,
            equilibria=lorenz_equilibria,
            jacobian=lorenz_jacobian,
            parameters={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
            description="Example integer-order Lorenz 63 system.",
            tags=("example", "integer-order"),
        ),
        replace=True,
    )


def build_spec() -> WorkflowInputSpec:
    """Build a self-contained specification for trajectory characterization."""

    return WorkflowInputSpec(
        system_name="lorenz63",
        dimension=3,
        parameters={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        integrator=IntegratorSpec(
            implementation="example.integrate_lorenz63",
            order_kind="integer",
            h=0.01,
            t_final=100.0,
            t_burn=20.0,
            memory_policy="not_applicable",
        ),
        trajectory_diagnostics=TrajectoryDiagnosticsSpec(
            retained_time_start=20.0,
            retained_time_end=100.0,
            observables=("x", "y", "z"),
            extrema_observable="z",
            spectrum_observable="x",
        ),
        notes="Example contract for finite-time trajectory characterization.",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/example_lorenz63_workflow_spec.json"),
        help="Where to write the example WorkflowInputSpec JSON.",
    )
    args = parser.parse_args(argv)

    register_lorenz63()
    system = get_system("lorenz63")
    print(f"registered={system.name}")
    for workflow in ("equilibria", "characterization"):
        report = check_system_capability(system, workflow)
        print(f"[{workflow}] {report.as_lines()[-1] if report.warnings else report.as_lines()[2]}")

    spec = build_spec()
    errors = spec.validate_for(("trajectory-diagnostics",))
    if errors:
        raise SystemExit("\n".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_workflow_spec(args.output, spec)
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
