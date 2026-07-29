"""Closed partial validation for the Danca 2017 fractional Chua case.

The publication discloses the equations, parameters, fractional order,
ABM/full-history method, and equilibrium-neighbourhood radius. It does not
disclose the initial condition of the reported attractor. This validator
therefore checks only the published algebraic and numerical-method contract.
It does not run trajectories or certify chaos or hiddenness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hidden_attractors.io import write_json
from hidden_attractors.paths import PROJECT_ROOT


LEGACY_ROOT = PROJECT_ROOT / "tools" / "legacy"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from danca2017_chua_abm_replication import (  # noqa: E402
    DancaChuaConfig,
    local_jacobian,
    solve_equilibria,
    validate_config,
)


REFERENCE_PATH = PROJECT_ROOT / "validation" / "references" / "danca2017_expected.json"
FULL_HISTORY_POLICY = "full_caputo_history_no_finite_memory_truncation"


def _load_reference(path: str | Path = REFERENCE_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("case_id") != "danca2017_chua_fractional_saturation":
        raise ValueError("unexpected Danca reference case")
    return payload


def _config_from_reference(reference: dict[str, Any]) -> DancaChuaConfig:
    parameters = reference["parameters"]
    return DancaChuaConfig(
        alpha=float(parameters["alpha"]),
        beta=float(parameters["beta"]),
        gamma_chua=float(parameters["gamma"]),
        m0=float(parameters["m0"]),
        m1=float(parameters["m1"]),
    )


def _solver_cases(_args: argparse.Namespace | None = None) -> list[dict[str, Any]]:
    """Return only the method contract disclosed by the publication."""

    return [
        {
            "solver_case_id": "abm_full_history",
            "solver": "abm",
            "backend": "python_abm_full_history",
            "history_policy": FULL_HISTORY_POLICY,
            "memory_length": None,
            "reference_role": "published_method_contract_only",
            "dynamics_executed": False,
        }
    ]


def build_validation_summary(
    reference_path: str | Path = REFERENCE_PATH,
) -> dict[str, Any]:
    """Validate the published parameter and equilibrium records."""

    reference = _load_reference(reference_path)
    config = _config_from_reference(reference)
    validate_config(config)

    computed = solve_equilibria(config.params())
    expected = {
        label: np.asarray(point, dtype=float)
        for label, point in reference["equilibria"].items()
    }
    labels_match = set(computed) == set(expected)
    residuals = {
        label: float(np.linalg.norm(computed[label] - expected[label]))
        for label in sorted(set(computed) & set(expected))
    }
    max_residual = max(residuals.values(), default=float("inf"))

    theta = float(config.q) * np.pi / 2.0
    stability = {}
    for label, point in computed.items():
        eigenvalues = np.linalg.eigvals(local_jacobian(config.params(), point))
        margins = [float(abs(np.angle(value)) - theta) for value in eigenvalues]
        stability[label] = {
            "matignon_class": "stable" if all(value > 0.0 for value in margins) else "unstable",
            "minimum_argument_margin": float(min(margins)),
        }

    initial_condition_missing = reference.get("initial_conditions_from_paper") is None
    passed = bool(labels_match and max_residual <= 1.0e-12 and initial_condition_missing)
    return {
        "schema_version": "1.0",
        "artifact_role": "published_case_partial_validation",
        "case_id": reference["case_id"],
        "status": "passed" if passed else "failed",
        "validated_scope": [
            "published_parameters",
            "published_equilibria",
            "published_fractional_order",
            "published_abm_full_history_method",
        ],
        "numerical_contract": {
            "q": float(config.q),
            "h": float(config.h),
            "t_final": float(config.t_final),
            "t_burn": float(config.transient),
            "equilibrium_neighbourhood_radius": float(config.delta),
            "solver_cases": _solver_cases(),
        },
        "parameters": config.params(),
        "equilibria": {
            label: [float(value) for value in point]
            for label, point in computed.items()
        },
        "equilibrium_residuals": residuals,
        "max_equilibrium_residual": max_residual,
        "matignon_stability": stability,
        "dynamics": {
            "executed": False,
            "reason": "published_initial_condition_not_disclosed",
        },
        "claims": {
            "chaos_certified": False,
            "hiddenness_certified": False,
            "full_reproduction": False,
        },
    }


def run_validation(output_dir: str | Path) -> Path:
    """Write the deterministic partial-validation summary."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "danca2017_partial_validation_summary.json"
    write_json(path, build_validation_summary())
    return path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the disclosed Danca 2017 parameters, equilibria, and "
            "ABM/full-history contract without running undisclosed dynamics."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT.parent / "outputs" / "danca2017_partial_validation"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    print(run_validation(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
