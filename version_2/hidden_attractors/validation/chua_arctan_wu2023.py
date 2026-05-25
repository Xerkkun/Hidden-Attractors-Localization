"""Algebraic validation evidence for ``fractional_chua_arctan_wu2023``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..models.chua import (
    chua_arctan_wu2023_parameters,
    equilibria_arctan,
    jacobian_arctan,
    rhs_arctan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALGEBRA_OUTPUT = (
    PROJECT_ROOT
    / "validation"
    / "reference_cases"
    / "fractional_chua_arctan_wu2023"
    / "01_algebra"
    / "chua_arctan_wu2023_algebra.json"
)
Q_WU2023 = 0.99
EXPECTED_EQUILIBRIA = {
    "E0": np.array([0.0, 0.0, 0.0]),
    "E+": np.array([0.60967911698, 2.6247941849e-4, -0.60941663756]),
    "E-": np.array([-0.60967911698, -2.6247941849e-4, 0.60941663756]),
}
EXPECTED_EIGENVALUES = {
    "E0": np.array(
        [
            2.335992046121269 + 0.0j,
            -1.0004421730606339 + 2.438870245845308j,
            -1.0004421730606339 - 2.438870245845308j,
        ]
    ),
    "E+": np.array(
        [
            -3.649223882578855 + 0.0j,
            0.20653022914273128 + 2.7072975063472904j,
            0.20653022914273128 - 2.7072975063472904j,
        ]
    ),
}


def _complex_pair(value: complex) -> list[float]:
    return [float(np.real(value)), float(np.imag(value))]


def _sorted_eigenvalues(values: np.ndarray) -> list[complex]:
    return sorted((complex(value) for value in values), key=lambda value: (round(value.real, 12), round(value.imag, 12)))


def _finite_difference_jacobian(state: np.ndarray, step: float = 1.0e-7) -> np.ndarray:
    params = chua_arctan_wu2023_parameters()
    matrix = np.zeros((3, 3), dtype=float)
    for column in range(3):
        shift = np.zeros(3)
        shift[column] = step
        matrix[:, column] = (rhs_arctan(state + shift, params) - rhs_arctan(state - shift, params)) / (2.0 * step)
    return matrix


def build_algebra_validation() -> dict[str, Any]:
    """Return machine-readable equilibria, Jacobian, eigenvalue and Matignon evidence."""

    params = chua_arctan_wu2023_parameters()
    equilibria = equilibria_arctan(params)
    threshold = Q_WU2023 * np.pi / 2.0
    records: dict[str, Any] = {}
    passed = set(equilibria) == {"E0", "E+", "E-"}
    for name, state in equilibria.items():
        jacobian = jacobian_arctan(state, params)
        eigenvalues = _sorted_eigenvalues(np.linalg.eigvals(jacobian))
        expected_state = EXPECTED_EQUILIBRIA[name]
        expected_eigenvalues = EXPECTED_EIGENVALUES["E0" if name == "E0" else "E+"]
        expected_sorted = _sorted_eigenvalues(expected_eigenvalues)
        modes = []
        for value in eigenvalues:
            margin = abs(float(np.angle(value))) - threshold
            modes.append(
                {
                    "eigenvalue": _complex_pair(value),
                    "abs_argument": abs(float(np.angle(value))),
                    "matignon_margin": margin,
                    "stable_mode": bool(margin > 0.0),
                }
            )
        residual = float(np.linalg.norm(rhs_arctan(state, params)))
        equilibrium_error = float(np.linalg.norm(state - expected_state))
        eigenvalue_error = max(abs(left - right) for left, right in zip(eigenvalues, expected_sorted))
        finite_difference_error = float(np.linalg.norm(jacobian - _finite_difference_jacobian(state)))
        passed = passed and residual < 1.0e-9 and equilibrium_error < 1.0e-9 and eigenvalue_error < 1.0e-9
        records[name] = {
            "state": state.tolist(),
            "expected_state": expected_state.tolist(),
            "expected_state_error_norm": equilibrium_error,
            "equilibrium_residual_norm": residual,
            "jacobian": jacobian.tolist(),
            "finite_difference_jacobian_error_norm": finite_difference_error,
            "eigenvalues": [_complex_pair(value) for value in eigenvalues],
            "expected_eigenvalue_max_abs_error": float(eigenvalue_error),
            "matignon": {
                "q": Q_WU2023,
                "threshold_radians": threshold,
                "modes": modes,
                "equilibrium_stable": bool(all(mode["stable_mode"] for mode in modes)),
                "classification": "locally_asymptotically_stable" if all(mode["stable_mode"] for mode in modes) else "unstable",
            },
        }
    return {
        "schema_version": "1.0",
        "case_id": "fractional_chua_arctan_wu2023",
        "source": "Wu et al. 2023",
        "stage": "algebraic_validation",
        "status": "passed" if passed else "failed",
        "official_parameters": {
            "model": params.model,
            "alpha": params.alpha,
            "beta": params.beta,
            "gamma": params.gamma,
            "m": params.a1,
            "n": params.a1 + params.a2,
            "a1": params.a1,
            "a2": params.a2,
            "rho": params.rho,
            "q": Q_WU2023,
            "h": 0.01,
            "N": 10000,
        },
        "equilibria": records,
        "matignon_summary": {
            "threshold_radians": threshold,
            "stable_equilibria": [name for name, row in records.items() if row["matignon"]["equilibrium_stable"]],
            "unstable_equilibria": [name for name, row in records.items() if not row["matignon"]["equilibrium_stable"]],
        },
        "scientific_boundary": {
            "hidden_verified": False,
            "reason": "Algebra, local stability and seed generation do not test basins around E0, E+ and E-.",
        },
    }


def write_algebra_validation(path: Path = DEFAULT_ALGEBRA_OUTPUT) -> dict[str, Any]:
    """Write the algebra report JSON and return the same dictionary."""

    report = build_algebra_validation()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ALGEBRA_OUTPUT)
    args = parser.parse_args(argv)
    report = write_algebra_validation(args.output)
    print(json.dumps({"output": str(args.output), "status": report["status"]}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
