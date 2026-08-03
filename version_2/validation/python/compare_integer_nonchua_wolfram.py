"""Cross-check non-Chua integer Wolfram derivations against the Python library.

The Wolfram scripts start from the cited source models.  This comparator is a
second implementation: it reads only their exported JSON artifacts and asks
the library to recompute matrices, transfer samples, direct branches, and the
route-specific initial seed.  It never reads the comparative LaTeX report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hidden_attractors.seed_generation import (
    find_integer_lure_omega_gain_candidates_direct,
)
from hidden_attractors.systems.kalman_fitts import kalman_fitts_2019_system
from hidden_attractors.systems.modified_van_der_pol_duffing import (
    mavpd_2023_system,
    mavpd_hopf_gamma_boundaries,
)
from hidden_attractors.systems.pll_lead_lag import pll_lead_lag_2015_system
from hidden_attractors.workflows.integer_lure import integer_lure_seed
from hidden_attractors.workflows.pll_lead_lag import exact_zero_gain_running_cycle
from hidden_attractors.workflows.switching_lure import find_sign_switching_cycle_seed


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "validation" / "outputs" / "wolfram"
SPECTRAL_SAMPLE = 0.7 + 1.3j


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _standard_transfer(system, spectral: complex) -> complex:
    lure = system.lure
    assert lure is not None
    return complex(
        np.asarray(lure.output_vector, dtype=float)
        @ np.linalg.solve(
            spectral * np.eye(lure.dimension) - np.asarray(lure.matrix, dtype=float),
            np.asarray(lure.input_vector, dtype=float),
        )
    )


def _complex_pair(values: list[float]) -> complex:
    return complex(float(values[0]), float(values[1]))


def _matrix_checks(summary: dict[str, Any], system) -> dict[str, float | bool]:
    lure = system.lure
    assert lure is not None
    numeric = summary["lure_form_numeric"]
    p_diff = float(np.max(np.abs(np.asarray(numeric["P"], dtype=float) - lure.matrix)))
    b_diff = float(
        np.max(np.abs(np.asarray(numeric["b"], dtype=float) - lure.input_vector))
    )
    r_diff = float(
        np.max(np.abs(np.asarray(numeric["r"], dtype=float) - lure.output_vector))
    )
    return {
        "P_max_diff": p_diff,
        "b_max_diff": b_diff,
        "r_max_diff": r_diff,
        "passed": max(p_diff, b_diff, r_diff) <= 1.0e-12,
    }


def compare_kalman(base: Path) -> dict[str, Any]:
    system_id = "kalman_fitts_integer"
    directory = base / system_id
    symbolic = _load(directory / f"{system_id}_symbolic_summary.json")
    seeds = _load(directory / f"{system_id}_seed_data.json")
    system = kalman_fitts_2019_system()

    direct = next(row for row in seeds if row["route"] == "direct_integer_transfer")
    pairs = find_integer_lure_omega_gain_candidates_direct(
        system.lure, wmin=0.0, wmax=20.0, compatible_only=False
    )
    if len(pairs) != 1:
        raise AssertionError(f"expected one Kalman direct pair, got {pairs!r}")
    omega_py, gain_py = pairs[0]

    switching_w = next(
        row for row in seeds if row["route"] == "exact_sign_switching_point_map"
    )
    switching_py = find_sign_switching_cycle_seed(
        system,
        [-4.0, -4.0, 0.0, -4.0],
        max_crossings=400,
        max_return_period=8,
        convergence_window=4,
        convergence_tolerance=1.0e-10,
        bracket_step=0.02,
        max_crossing_time=20.0,
        root_tolerance=1.0e-12,
    )

    sample = symbolic["transfer"]["numeric_sample"]
    transfer_w = _complex_pair(sample["W_standard"])
    transfer_py = _standard_transfer(system, SPECTRAL_SAMPLE)
    result = {
        "system_id": system_id,
        "report_input_used": symbolic["report_input_used"],
        "matrices": _matrix_checks(symbolic, system),
        "transfer_sample_diff": abs(transfer_w - transfer_py),
        "omega0_diff": abs(float(direct["omega0"]) - omega_py),
        "gain_diff": abs(float(direct["k"]) - gain_py),
        "switching_seed_max_diff": float(
            np.max(
                np.abs(
                    np.asarray(switching_w["seed"], dtype=float)
                    - np.asarray(switching_py.seed, dtype=float)
                )
            )
        ),
        "switching_period_match": int(switching_w["return_period"])
        == switching_py.return_period,
    }
    result["passed"] = bool(
        result["report_input_used"] is False
        and result["matrices"]["passed"]
        and result["transfer_sample_diff"] <= 1.0e-12
        and result["omega0_diff"] <= 1.0e-10
        and result["gain_diff"] <= 1.0e-10
        and result["switching_seed_max_diff"] <= 1.0e-7
        and result["switching_period_match"]
    )
    return result


def compare_mavpd(base: Path) -> dict[str, Any]:
    system_id = "mavpd_integer"
    directory = base / system_id
    symbolic = _load(directory / f"{system_id}_symbolic_summary.json")
    rows = _load(directory / f"{system_id}_seed_data.json")
    validation = _load(directory / f"{system_id}_validation_summary.json")
    base_system = mavpd_2023_system({"xi": 3.1})
    sample_diffs: list[float] = []
    for sample in symbolic["transfer"]["numeric_samples"]:
        system = mavpd_2023_system({"xi": float(sample["xi"])})
        sample_diffs.append(
            abs(_complex_pair(sample["W_standard"]) - _standard_transfer(system, SPECTRAL_SAMPLE))
        )

    branch_results: list[dict[str, Any]] = []
    for row in rows:
        system = mavpd_2023_system({"xi": float(row["xi"])})
        seed_py = integer_lure_seed(
            system,
            branch_index=int(row["branch"]),
            method="classic",
            theta=float(row["phase"]),
            wmin=1.0e-5,
            wmax=50.0,
        )
        branch_results.append(
            {
                "xi": float(row["xi"]),
                "branch": int(row["branch"]),
                "phase": float(row["phase"]),
                "omega0_diff": abs(float(row["omega0"]) - seed_py.omega),
                "gain_diff": abs(float(row["k"]) - seed_py.gain),
                "amplitude_diff": abs(float(row["a0"]) - seed_py.amplitude),
                "seed_max_diff": float(
                    np.max(
                        np.abs(
                            np.asarray(row["seed"], dtype=float)
                            - np.asarray(seed_py.seed, dtype=float)
                        )
                    )
                ),
            }
        )
    worst_scalar = max(
        max(item["omega0_diff"], item["gain_diff"], item["amplitude_diff"])
        for item in branch_results
    )
    worst_seed = max(item["seed_max_diff"] for item in branch_results)
    stability = symbolic["nonzero_equilibrium_stability"]
    xi_target = float(stability["xi_target"])
    hopf_python = mavpd_hopf_gamma_boundaries(
        {"xi": xi_target, "delta": 100.0, "rho": 200.0, "gamma": 0.1}
    )
    hopf_wolfram = tuple(float(value) for value in stability["positive_gamma_boundaries"])
    hopf_diff = max(abs(left - right) for left, right in zip(hopf_python, hopf_wolfram, strict=True))
    candidate_gamma = float(stability["candidate_gamma"])
    candidate_system = mavpd_2023_system(
        {"xi": xi_target, "delta": 100.0, "rho": 200.0, "gamma": candidate_gamma}
    )
    candidate_equilibrium = candidate_system.equilibrium_points()["E+"]
    eigen_python = sorted(
        np.linalg.eigvals(candidate_system.jacobian_matrix(candidate_equilibrium)),
        key=lambda value: (float(np.real(value)), float(np.imag(value))),
    )
    eigen_wolfram = sorted(
        (_complex_pair(value) for value in stability["candidate_Eplus_Eminus_eigenvalues"]),
        key=lambda value: (float(np.real(value)), float(np.imag(value))),
    )
    eigen_diff = max(abs(left - right) for left, right in zip(eigen_python, eigen_wolfram, strict=True))
    validation_names = {str(item["name"]) for item in validation["tests"]}
    result = {
        "system_id": system_id,
        "report_input_used": symbolic["report_input_used"],
        "matrices_xi_3p1": _matrix_checks(symbolic, base_system),
        "transfer_sample_max_diff": max(sample_diffs),
        "branch_results": branch_results,
        "branch_scalar_max_diff": worst_scalar,
        "seed_max_diff": worst_seed,
        "hopf_boundaries_max_diff": hopf_diff,
        "candidate_equilibrium_eigenvalues_max_diff": eigen_diff,
        "boundary_derived_from_source_equations": stability["boundary_derived_from_source_equations"],
        "candidate_parameter_tuple_derived_algebraically": stability[
            "candidate_parameter_tuple_derived_algebraically"
        ],
        "wolfram_validation_passed": validation["passed"],
        "required_wolfram_checks_present": {
            "nonzero_characteristic_polynomial_derived",
            "routh_hopf_polynomial_derived",
            "numerically_selected_candidate_Eplus_Eminus_hurwitz",
        }.issubset(validation_names),
    }
    result["passed"] = bool(
        result["report_input_used"] is False
        and result["matrices_xi_3p1"]["passed"]
        and result["transfer_sample_max_diff"] <= 1.0e-12
        and worst_scalar <= 1.0e-10
        and worst_seed <= 1.0e-7
        and hopf_diff <= 1.0e-12
        and eigen_diff <= 1.0e-10
        and result["boundary_derived_from_source_equations"] is True
        and result["candidate_parameter_tuple_derived_algebraically"] is False
        and result["wolfram_validation_passed"] is True
        and result["required_wolfram_checks_present"] is True
    )
    return result


def compare_pll(base: Path) -> dict[str, Any]:
    system_id = "pll_lead_lag_integer"
    directory = base / system_id
    symbolic = _load(directory / f"{system_id}_symbolic_summary.json")
    seeds = _load(directory / f"{system_id}_seed_data.json")
    row = seeds[0]
    system = pll_lead_lag_2015_system()
    cycle = exact_zero_gain_running_cycle(system.parameters)
    sample = symbolic["transfer"]["numeric_sample"]
    transfer_w = _complex_pair(sample["G_standard"])
    transfer_py = _standard_transfer(system, SPECTRAL_SAMPLE)
    result = {
        "system_id": system_id,
        "report_input_used": symbolic["report_input_used"],
        "matrices": _matrix_checks(symbolic, system),
        "transfer_sample_diff": abs(transfer_w - transfer_py),
        "section_x_diff": abs(float(row["section_x"]) - cycle.section_x),
        "section_velocity_diff": abs(
            float(row["section_velocity"]) - cycle.section_velocity
        ),
        "period_diff": abs(float(row["period"]) - cycle.period),
        "multiplier_diff": abs(float(row["multiplier"]) - cycle.multiplier),
    }
    result["passed"] = bool(
        result["report_input_used"] is False
        and result["matrices"]["passed"]
        and result["transfer_sample_diff"] <= 1.0e-12
        and max(
            result["section_x_diff"],
            result["section_velocity_diff"],
            result["period_diff"],
            result["multiplier_diff"],
        )
        <= 1.0e-10
    )
    return result


def compare_all(base: Path = DEFAULT_BASE, *, write_outputs: bool = False) -> dict[str, Any]:
    comparisons = [compare_kalman(base), compare_mavpd(base), compare_pll(base)]
    summary = {
        "validation_scope": "independent_wolfram_to_python_integer_nonchua_consistency",
        "report_input_used": False,
        "comparisons": comparisons,
        "passed": all(item["passed"] for item in comparisons),
    }
    if write_outputs:
        for item in comparisons:
            directory = base / item["system_id"]
            _write(directory / f"{item['system_id']}_python_consistency_summary.json", item)
        _write(base / "integer_nonchua_python_consistency_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()
    summary = compare_all(args.base.resolve(), write_outputs=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
