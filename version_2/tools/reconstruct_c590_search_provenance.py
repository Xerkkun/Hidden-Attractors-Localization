"""Reconstruct the deterministic parameter and seed provenance of c590.

The original search combined two PCG64 parameter banks with dynamical
screening.  This audit utility regenerates the random draws exactly and records
the numerical filters that selected the global centre, local candidate c590,
and the final fractional seed.  It does not rerun the expensive trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "arctan_hidden_candidate_search"
    / "c590_q09999_seed9_candidate_20260623"
    / "search_provenance.json"
)

PARAMETER_NAMES = ("alpha", "beta", "gamma", "a1", "a2", "rho")
GLOBAL_INDEX = 1731
LOCAL_INDEX = 590

EXPECTED_GLOBAL_PARAMETERS = {
    "alpha": 18.485729510399246,
    "beta": 21.96695211004372,
    "gamma": 0.005274610555721088,
    "a1": 0.02629749304876286,
    "a2": -3.208777924503481,
    "rho": 1.949261203458447,
}
EXPECTED_GLOBAL_SEED = np.array(
    [7.079605327144233, 0.5079327670652387, -14.49039352666749],
    dtype=float,
)
EXPECTED_C590_PARAMETERS = {
    "alpha": 21.849356906616716,
    "beta": 19.081840840860202,
    "gamma": 0.007378011979156531,
    "a1": 0.04228979343578827,
    "a2": -3.3367815123026694,
    "rho": 1.7984259332820332,
}
EXPECTED_C590_INTEGER_SEED = np.array(
    [7.6733768928786095, 0.5079327670652387, -14.49039352666749],
    dtype=float,
)
EXPECTED_FINAL_SEED = np.array(
    [5.864244979081692, 1.5847111486491057, 3.2155806477633915],
    dtype=float,
)
EXPECTED_EXTRACTED_SEEDS = np.array(
    [
        [-2.729473337164626, 2.1102962827455594, 10.432403330832878],
        [1.086782248440764, -2.57728725411782, 0.5540523821057417],
        [-2.1175469987001168, 2.372607407573507, 5.375571214892048],
        [-4.963397803649754, 0.24695513953718384, 17.24106885861425],
        [2.573553666760516, -2.2074465263950245, -9.598640684855509],
        [-1.3443209405873457, 2.394579416841281, -1.5861791996125314],
        [2.5970851435812365, -1.9066187889449309, -4.043299382162537],
        [4.445217479749246, -0.589499420498435, -13.200382065644083],
        [6.6792007983871535, 1.8624013079380815, -9.432621685715773],
        [5.864244979081692, 1.5847111486491057, 3.2155806477633915],
        [2.1070652812673774, -1.6959277503489347, 5.205928761559128],
        [4.188027775831749, 2.8128803633627006, -1.7614566460862728],
        [-6.976264361536463, -2.009821008822811, 13.134260959465786],
        [-3.27274740849852, 0.9735766277700553, -2.50103439150134],
        [-2.861358092116564, 1.8356042430512796, 7.247421429834667],
        [-5.211832850603234, -0.16296250009100868, 13.703379396402225],
    ],
    dtype=float,
)


def generate_global_bank() -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Regenerate the 2,400-case integer-order coexistence bank."""

    rng = np.random.default_rng(2026062304)
    count = 2400
    parameters = {
        "alpha": rng.uniform(5.0, 22.0, count),
        "beta": rng.uniform(6.0, 35.0, count),
        "gamma": 10.0 ** rng.uniform(-3.0, -0.5, count),
        "a1": rng.uniform(-0.1, 1.0, count),
        "a2": rng.uniform(-3.5, -0.4, count),
        "rho": rng.uniform(0.4, 2.8, count),
    }
    seeds = np.zeros((count, 3), dtype=float)
    seeds[:, 0] = 13.8 / parameters["rho"]
    seeds[:, 1] = rng.uniform(0.2, 1.4, count)
    seeds[:, 2] = rng.uniform(-35.0, -8.0, count)
    return parameters, seeds


def generate_local_bank(
    centre: dict[str, float],
    centre_seed: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Regenerate the 1,000 log-normal/additive perturbations around case 1731."""

    rng = np.random.default_rng(2026062305)
    count = 1000
    parameters = {
        "alpha": centre["alpha"] * np.exp(rng.normal(0.0, 0.18, count)),
        "beta": centre["beta"] * np.exp(rng.normal(0.0, 0.18, count)),
        "gamma": centre["gamma"] * np.exp(rng.normal(0.0, 0.35, count)),
        "a1": centre["a1"] + rng.normal(0.0, 0.12, count),
        "a2": centre["a2"] * np.exp(rng.normal(0.0, 0.15, count)),
        "rho": centre["rho"] * np.exp(rng.normal(0.0, 0.15, count)),
    }
    seeds = np.repeat(
        np.asarray(centre_seed, dtype=float)[None, :],
        count,
        axis=0,
    )
    # Preserve the operation order of the archived inline search exactly.
    # Reassociating this expression as x_b * (rho_b / rho_i) changes one or
    # two ulps for 411 rows and is enough to alter finite-time diagnostics for
    # weakly chaotic rows such as c603 and c709.
    seeds[:, 0] = (
        float(centre_seed[0])
        * centre["rho"]
        / parameters["rho"]
    )
    return parameters, seeds


def _row(parameters: dict[str, np.ndarray], index: int) -> dict[str, float]:
    return {name: float(parameters[name][index]) for name in PARAMETER_NAMES}


def _maximum_error(
    actual_parameters: dict[str, float],
    expected_parameters: dict[str, float],
    actual_seed: np.ndarray,
    expected_seed: np.ndarray,
) -> float:
    parameter_errors = [
        abs(actual_parameters[name] - expected_parameters[name])
        for name in PARAMETER_NAMES
    ]
    seed_errors = np.abs(np.asarray(actual_seed) - np.asarray(expected_seed))
    return float(max(parameter_errors + seed_errors.tolist()))


def reconstruct() -> dict[str, Any]:
    global_bank, global_seeds = generate_global_bank()
    global_parameters = _row(global_bank, GLOBAL_INDEX)
    global_seed = global_seeds[GLOBAL_INDEX]
    global_error = _maximum_error(
        global_parameters,
        EXPECTED_GLOBAL_PARAMETERS,
        global_seed,
        EXPECTED_GLOBAL_SEED,
    )

    local_bank, local_seeds = generate_local_bank(global_parameters, global_seed)
    c590_parameters = _row(local_bank, LOCAL_INDEX)
    c590_seed = local_seeds[LOCAL_INDEX]
    c590_error = _maximum_error(
        c590_parameters,
        EXPECTED_C590_PARAMETERS,
        c590_seed,
        EXPECTED_C590_INTEGER_SEED,
    )

    tolerance = 5.0e-15
    if global_error > tolerance or c590_error > tolerance:
        raise RuntimeError(
            "The regenerated PCG64 search bank does not match the recorded c590 "
            f"provenance: global_error={global_error}, local_error={c590_error}."
        )

    return {
        "schema_version": "1.0",
        "candidate_id": "chua_arctan_c590_q09999_seed9",
        "status": "exact_parameter_and_seed_provenance_verified",
        "random_generator": "NumPy Generator(PCG64)",
        "global_exploration": {
            "rng_seed": 2026062304,
            "cases": 2400,
            "parameter_domains": {
                "alpha": "U(5,22)",
                "beta": "U(6,35)",
                "gamma": "10^U(-3,-0.5)",
                "a1": "U(-0.1,1.0)",
                "a2": "U(-3.5,-0.4)",
                "rho": "U(0.4,2.8)",
            },
            "target_seed_rule": "[13.8/rho, U(0.2,1.4), U(-35,-8)]",
            "equilibrium_probe_seed": [0.001, 0.0, 0.0],
            "dynamics": {
                "order": 1.0,
                "integrator": "RK4",
                "h": 0.01,
                "t_final": 200.0,
                "t_burn": 100.0,
                "tail_stride_steps": 10,
            },
            "basin_separation_rule": (
                "min(D90(probe,target),D90(probe,-target)) "
                "> 2.5*D90_calibration"
            ),
            "selection": {
                "zero_based_index": GLOBAL_INDEX,
                "reason": (
                    "highest dynamical score among 61 bounded, distinct, "
                    "nonperiodic-inconclusive cases, followed by the "
                    "equilibrium-stability audit"
                ),
                "parameters": global_parameters,
                "seed": global_seed.tolist(),
            },
        },
        "local_exploration": {
            "rng_seed": 2026062305,
            "cases": 1000,
            "centre_global_index": GLOBAL_INDEX,
            "perturbations": {
                "alpha": "alpha_b*exp(N(0,0.18))",
                "beta": "beta_b*exp(N(0,0.18))",
                "gamma": "gamma_b*exp(N(0,0.35))",
                "a1": "a1_b+N(0,0.12)",
                "a2": "a2_b*exp(N(0,0.15))",
                "rho": "rho_b*exp(N(0,0.15))",
            },
            "target_seed_rule": (
                "[x_b*rho_b/rho, y_b, z_b], using the selected global seed"
            ),
            "equilibrium_probe_seed": [0.001, 0.0, 0.0],
            "dynamics": {
                "order": 1.0,
                "integrator": "RK4",
                "h": 0.01,
                "t_final": 240.0,
                "t_burn": 120.0,
                "tail_stride_steps": 10,
            },
            "screening": {
                "distinct_nontrivial_cases": 148,
                "nonperiodic_inconclusive_cases": 87,
                "regular_periodic_cases": 61,
                "c590_score_rank_among_distinct_nontrivial": 10,
                "variational_candidates": 30,
                "c590_variational_largest_exponent": 0.4699043683192531,
                "selection_rule": (
                    "maximum positive variational exponent among the audited "
                    "short list subject to bounded and distinct E0 response, "
                    "unstable E0, and stable E+ and E-"
                ),
            },
            "selection": {
                "zero_based_index": LOCAL_INDEX,
                "label": "c590",
                "parameters": c590_parameters,
                "integer_order_seed": c590_seed.tolist(),
            },
        },
        "fractional_refinement": {
            "operator": "Caputo",
            "integrator": "ABM predictor-corrector",
            "memory_mode": "full",
            "order_scan": [0.9995, 0.9998, 0.9999, 0.99995],
            "selected_order": 0.9999,
            "source_trajectory": {
                "initial_seed": c590_seed.tolist(),
                "h": 0.005,
                "t_final": 300.0,
                "t_burn": 150.0,
            },
            "seed_extraction": {
                "count": 16,
                "rule": (
                    "linspace(searchsorted(t,150),len(t)-1,16,dtype=int)"
                ),
                "source_indices": [
                    30001,
                    32000,
                    34000,
                    36000,
                    38000,
                    40000,
                    42000,
                    44000,
                    46000,
                    48000,
                    50000,
                    52000,
                    54000,
                    56000,
                    58000,
                    60000,
                ],
                "source_times": [
                    150.00499999993608,
                    159.99999999992698,
                    169.9999999999179,
                    179.9999999999088,
                    189.9999999998997,
                    199.9999999998906,
                    209.9999999998815,
                    219.99999999987241,
                    229.99999999986332,
                    239.99999999985423,
                    249.99999999984513,
                    259.99999999983606,
                    269.99999999982697,
                    279.9999999998179,
                    289.9999999998088,
                    299.9999999997997,
                ],
                "source_states": EXPECTED_EXTRACTED_SEEDS.tolist(),
                "bounded_cross_step_indices": [5, 8, 9, 10, 13, 15],
                "selected_zero_based_index": 9,
                "selected_source_time": 239.99999999985423,
                "selected_source_time_nominal": 240.0,
                "selected_seed": EXPECTED_FINAL_SEED.tolist(),
                "selection_rule": (
                    "bounded at h=0.0025,0.005,0.01 through the long-horizon "
                    "audit, then maximum robust resampled 0-1 support at the "
                    "two refined steps"
                ),
            },
            "post_selection_step_audit": {
                "h_values": [0.0025, 0.005, 0.01],
                "t_final": 300.0,
                "t_burn": 150.0,
                "resampled_zero_one_medians": [
                    0.865843,
                    0.885089,
                    0.740058,
                ],
            },
        },
        "verification": {
            "global_max_abs_error": global_error,
            "local_max_abs_error": c590_error,
            "absolute_tolerance": tolerance,
        },
        "scope": (
            "This artifact reconstructs the exact stochastic draws and records "
            "the archived screening contract. Dynamical and basin results remain "
            "independently reproducible from the fixed parameters and seeds."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    payload = reconstruct()
    if not args.check_only:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(output)
    print(json.dumps(payload["verification"], indent=2))


if __name__ == "__main__":
    main()
