"""Rerun the recovered discovery route from the integer bank to Caputo seed9.

The global and local searches below are executable transcriptions of the
historical vectorized scripts.  They retain the original NumPy PCG64 streams,
two simultaneous initial conditions per parameter row, fixed-step RK4,
tail-sampling convention, nearest-neighbour separation test, and calls to the
library's ``_diagnose`` and ``_matignon`` routines.

The variational stage is likewise reconstructed from the historical
top-30-inconclusive RK4/Benettin calculation.  The final stage reproduces the
full-memory Caputo order scan, step audits, 16-seed refinement, and seed9
selection. None of the expensive stages is run unless ``--execute`` is
supplied.

Examples
--------
Inspect every recovered numerical contract without writing files::

    python tools/rerun_c590_discovery.py --stage all --dry-run

Persist the exact PCG64 banks::

    python tools/rerun_c590_discovery.py --stage banks --execute

Rerun one expensive integer stage::

    python tools/rerun_c590_discovery.py --stage integer-global --execute
    python tools/rerun_c590_discovery.py --stage integer-local --execute

Rerun the complete recovered integer route::

    python tools/rerun_c590_discovery.py --stage integer-route --execute

Rerun every recovered integer and Caputo stage::

    python tools/rerun_c590_discovery.py --stage all --execute
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hidden_attractors.analysis.zero_one import zero_one_multicoordinate
from hidden_attractors.integrations.general import integrate_general
from tools.arctan_hidden_screen import _build_system, _diagnose, _matignon
from tools.reconstruct_c590_search_provenance import (
    EXPECTED_C590_INTEGER_SEED,
    EXPECTED_C590_PARAMETERS,
    EXPECTED_EXTRACTED_SEEDS,
    EXPECTED_FINAL_SEED,
    EXPECTED_GLOBAL_PARAMETERS,
    EXPECTED_GLOBAL_SEED,
    GLOBAL_INDEX,
    LOCAL_INDEX,
    PARAMETER_NAMES,
    generate_global_bank,
    generate_local_bank,
    reconstruct,
)


DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "arctan_hidden_candidate_search"
    / "c590_discovery_rerun"
)

EXECUTABLE_STAGES = (
    "banks",
    "integer-global",
    "integer-local",
    "variational-shortlist",
)
STAGE_ORDER = (*EXECUTABLE_STAGES, "caputo-seed9")
STAGE_CHOICES = ("all", "integer-route", *STAGE_ORDER)


@dataclass(frozen=True)
class IntegerSearchContract:
    """Exact numerical settings recovered for one vectorized integer search."""

    name: str
    cases: int
    h: float
    steps: int
    burn_steps: int
    stride: int
    progress_every: int
    calibration_split: int
    recorded_index: int
    zero_one_samples: tuple[int, int, int]

    @property
    def t_final(self) -> float:
        return self.steps * self.h

    @property
    def t_burn(self) -> float:
        return self.burn_steps * self.h

    @property
    def sampled_h(self) -> float:
        return self.h * self.stride

    @property
    def tail_rows(self) -> int:
        return len(range(self.burn_steps, self.steps, self.stride))

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": 1.0,
            "integrator": "fixed_step_vectorized_RK4",
            "cases": self.cases,
            "simultaneous_initial_conditions_per_case": 2,
            "h": self.h,
            "steps": self.steps,
            "t_final": self.t_final,
            "burn_steps": self.burn_steps,
            "t_burn": self.t_burn,
            "tail_stride_steps": self.stride,
            "sampled_h": self.sampled_h,
            "tail_rows": self.tail_rows,
            "calibration_split": self.calibration_split,
            "zero_one_samples": list(self.zero_one_samples),
            "recorded_zero_based_index": self.recorded_index,
            "time_stamp_convention": (
                "Historical convention: the first stored post-update state is "
                "labelled t_burn and subsequent rows use h*stride."
            ),
        }


GLOBAL_CONTRACT = IntegerSearchContract(
    name="integer_global",
    cases=2400,
    h=0.01,
    steps=20_000,
    burn_steps=10_000,
    stride=10,
    progress_every=5_000,
    calibration_split=500,
    recorded_index=GLOBAL_INDEX,
    zero_one_samples=(500, 800, 1000),
)

LOCAL_CONTRACT = IntegerSearchContract(
    name="integer_local",
    cases=1000,
    h=0.01,
    steps=24_000,
    burn_steps=12_000,
    stride=10,
    progress_every=6_000,
    calibration_split=600,
    recorded_index=LOCAL_INDEX,
    zero_one_samples=(600, 900, 1200),
)

VARIATIONAL_CONTRACT = {
    "order": 1.0,
    "integrator": "fixed_step_vectorized_RK4_Benettin_QR",
    "source_rows": (
        "top 30 rows with screen_label=inconclusive_nonperiodic, sorted by "
        "the archived diagnostic score in descending order"
    ),
    "candidates": 30,
    "h": 0.005,
    "state_burn_steps": 40_000,
    "state_burn_time": 200.0,
    "measurement_steps": 60_000,
    "measurement_time": 300.0,
    "reorthonormalize_every_steps": 10,
    "recorded_c590_largest_exponent": 0.4699043683192531,
}

CAPUTO_CONTRACT = {
    "operator": "Caputo",
    "integrator": "ABM predictor-corrector",
    "memory_mode": "full",
    "use_c_backend": True,
    "early_stop": False,
    "divergence_norm": 300.0,
    "parameters": EXPECTED_C590_PARAMETERS,
    "integer_seed": EXPECTED_C590_INTEGER_SEED.tolist(),
    "q_scan": {
        "q_values": [0.9995, 0.9998, 0.9999, 0.99995],
        "h": 0.005,
        "t_final": 300.0,
        "t_burn": 150.0,
        "zero_one_samples": [2000, 4000, 8000],
    },
    "resampled_zero_one": {
        "q_values": [0.9997, 0.9998, 0.9999],
        "q_09997_t_final": 260.0,
        "q_09997_t_burn": 130.0,
        "h": 0.005,
        "sample_intervals": [0.5, 0.75, 1.0, 1.5],
        "n_c": 100,
    },
    "integer_seed_h_audit": {
        "q": 0.9999,
        "h_values": [0.0025, 0.005, 0.01],
        "t_final": 300.0,
        "t_burn": 150.0,
        "bounded_norm": 50.0,
        "sample_intervals": [0.5, 0.75, 1.0, 1.5],
        "n_c": 100,
        "recorded_robust_bounded": False,
        "recorded_robust_chaos_screen": False,
    },
    "seed_extraction": {
        "source_q": 0.9999,
        "source_h": 0.005,
        "source_t_start": 150.0,
        "count": 16,
        "rule": (
            "linspace(searchsorted(times,150),len(times)-1,16,dtype=int)"
        ),
        "recorded_source_indices": [
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
        "recorded_source_times": [
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
        "recorded_source_states": EXPECTED_EXTRACTED_SEEDS.tolist(),
    },
    "seed_refinement": {
        "q": 0.9999,
        "h": 0.0025,
        "t_final": 180.0,
        "t_burn": 90.0,
        "bounded_norm": 50.0,
        "recorded_bounded_indices": [
            0,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            15,
        ],
    },
    "cross_step": {
        "additional_h_values": [0.005, 0.01],
        "t_final": 180.0,
        "t_burn": 90.0,
        "recorded_passing_indices": [5, 8, 9, 10, 13, 15],
    },
    "seed5_long_audit": {
        "seed_index": 5,
        "h_values": [0.0025, 0.005, 0.01],
        "t_final": 300.0,
        "t_burn": 150.0,
        "sample_intervals": [0.5, 0.75, 1.0, 1.5],
        "n_c": 100,
        "K_threshold": 0.8,
        "recorded_result": "rejected",
    },
    "survivor_long_audit": {
        "seed_indices": [8, 9, 10, 13, 15],
        "h_values": [0.0025, 0.005, 0.01],
        "t_final": 300.0,
        "t_burn": 150.0,
        "sample_intervals": [0.5, 0.75, 1.0, 1.5],
        "n_c": 80,
        "K_threshold": 0.8,
        "selection_rule": (
            "bounded at h=0.0025,0.005,0.01, then K_median>0.8 at "
            "the two refined steps h=0.0025 and h=0.005"
        ),
        "recorded_passing_indices": [9],
    },
    "final_target": {
        "seed_index": 9,
        "seed": EXPECTED_FINAL_SEED.tolist(),
        "q": 0.9999,
        "h": 0.005,
        "t_final": 300.0,
        "t_burn": 150.0,
    },
}


class MissingDiscoveryContract(RuntimeError):
    """Raised when a requested stage still lacks a recovered contract."""


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _require_new_outputs(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite an earlier reconstruction. Select a new "
            f"--output-dir. Existing paths: {joined}"
        )


def _stage_plan(provenance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    global_contract = provenance["global_exploration"]
    local_contract = provenance["local_exploration"]
    fractional_contract = provenance["fractional_refinement"]
    return {
        "banks": {
            "status": "executable",
            "cost": "inexpensive",
            "action": (
                "Regenerate and persist the exact 2,400-row global and "
                "1,000-row local NumPy Generator(PCG64) banks."
            ),
            "contract": {
                "generator": provenance["random_generator"],
                "global_rng_seed": global_contract["rng_seed"],
                "global_cases": global_contract["cases"],
                "local_rng_seed": local_contract["rng_seed"],
                "local_cases": local_contract["cases"],
                "absolute_tolerance": provenance["verification"][
                    "absolute_tolerance"
                ],
            },
            "outputs": ["search_provenance.json", "search_banks.npz"],
            "missing_contract": [],
        },
        "integer-global": {
            "status": "executable",
            "cost": "expensive",
            "depends_on": ["banks"],
            "action": (
                "Run 2,400 parameter rows, each with the biased target seed and "
                "the E0 probe [1e-3,0,0], using vectorized q=1 RK4."
            ),
            "contract": GLOBAL_CONTRACT.as_dict(),
            "parameter_domains": global_contract["parameter_domains"],
            "separation_rule": global_contract["basin_separation_rule"],
            "recorded_selection": global_contract["selection"],
            "outputs": [
                "integer_global/initial_bank.npz",
                "integer_global/screen.json",
                "integer_global/summary.json",
                "integer_global/recorded_selection_i1731.npz",
            ],
            "missing_contract": [],
        },
        "integer-local": {
            "status": "executable",
            "cost": "expensive",
            "depends_on": ["integer-global"],
            "action": (
                "Run the 1,000 PCG64 perturbations around global row 1731 and "
                "recover local row c590 with the archived q=1 RK4 screen."
            ),
            "contract": LOCAL_CONTRACT.as_dict(),
            "perturbations": local_contract["perturbations"],
            "recorded_screening": local_contract["screening"],
            "recorded_selection": local_contract["selection"],
            "outputs": [
                "integer_local/initial_bank.npz",
                "integer_local/screen.json",
                "integer_local/summary.json",
                "integer_local/recorded_selection_c590.npz",
            ],
            "missing_contract": [],
        },
        "variational-shortlist": {
            "status": "executable",
            "cost": "expensive",
            "depends_on": ["integer-local"],
            "action": (
                "Select the 30 highest-scoring inconclusive local rows and run "
                "the recovered vectorized RK4/Benettin tangent audit."
            ),
            "contract": VARIATIONAL_CONTRACT,
            "recorded_selection_rule": local_contract["screening"][
                "selection_rule"
            ],
            "outputs": [
                "variational_shortlist/variational_shortlist.json",
                "variational_shortlist/variational_shortlist.npz",
            ],
            "missing_contract": [],
        },
        "caputo-seed9": {
            "status": "executable",
            "cost": "very_expensive",
            "depends_on": ["variational-shortlist"],
            "action": (
                "Repeat the full-memory Caputo q scan, resampled 0-1 audit, "
                "16-seed refinement, short and long cross-step filters, and "
                "the final seed9 target."
            ),
            "contract": CAPUTO_CONTRACT,
            "recorded_provenance": {
                "operator": fractional_contract["operator"],
                "integrator": fractional_contract["integrator"],
                "memory_mode": fractional_contract["memory_mode"],
                "order_scan": fractional_contract["order_scan"],
                "seed_extraction": fractional_contract["seed_extraction"],
                "post_selection_step_audit": fractional_contract[
                    "post_selection_step_audit"
                ],
            },
            "outputs": [
                "caputo_seed9/q_scan/*.npz",
                "caputo_seed9/q_scan/summary.json",
                "caputo_seed9/resampled_zero_one.json",
                "caputo_seed9/integer_seed_h_audit/*.npz",
                "caputo_seed9/seed_refinement/*.npz",
                "caputo_seed9/cross_step/*.npz",
                "caputo_seed9/seed5_long_audit/*.npz",
                "caputo_seed9/survivor_long_audit/*.npz",
                "caputo_seed9/target.npz",
                "caputo_seed9/summary.json",
            ],
            "missing_contract": [],
        },
    }


def _selected_stage_names(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return STAGE_ORDER
    if stage == "integer-route":
        return EXECUTABLE_STAGES
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage: {stage}")
    return (stage,)


def build_plan(stage: str = "all") -> dict[str, Any]:
    """Verify the exact random banks and return a non-mutating execution plan."""

    selected = _selected_stage_names(stage)
    provenance = reconstruct()
    stages = _stage_plan(provenance)
    blocked = [
        name for name in selected if stages[name]["status"].startswith("blocked")
    ]
    return {
        "schema_version": "2.0",
        "candidate_id": provenance["candidate_id"],
        "mode": "dry_run_plan",
        "requested_stage": stage,
        "writes_performed": False,
        "bank_verification": provenance["verification"],
        "stages": [{"stage": name, **stages[name]} for name in selected],
        "overall_status": (
            "ready_to_execute"
            if not blocked
            else "partial_blocked_by_explicit_contract_boundary"
        ),
        "blocked_stages": blocked,
        "scientific_boundary": (
            "The global, local, integer variational, and full-memory Caputo "
            "calculations are executable recovered contracts. Their recorded "
            "c590 and seed9 values are regression targets, not substitutes "
            "for the rerun results."
        ),
    }


def _parameter_row(
    parameters: dict[str, np.ndarray],
    index: int,
) -> dict[str, float]:
    return {
        name: float(parameters[name][index])
        for name in PARAMETER_NAMES
    }


def _vector_field(
    states: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> np.ndarray:
    values = np.empty_like(states)
    phi = (
        parameters["a1"][None, :] * states[:, :, 0]
        + parameters["a2"][None, :]
        * np.arctan(parameters["rho"][None, :] * states[:, :, 0])
    )
    values[:, :, 0] = parameters["alpha"][None, :] * (
        states[:, :, 1] - states[:, :, 0] - phi
    )
    values[:, :, 1] = (
        states[:, :, 0] - states[:, :, 1] + states[:, :, 2]
    )
    values[:, :, 2] = (
        -parameters["beta"][None, :] * states[:, :, 1]
        - parameters["gamma"][None, :] * states[:, :, 2]
    )
    return values


def _integrate_vectorized_rk4(
    initial_states: np.ndarray,
    parameters: dict[str, np.ndarray],
    contract: IntegerSearchContract,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the recovered vectorized RK4 and historical tail sampling."""

    states = np.asarray(initial_states, dtype=float).copy()
    tail: list[np.ndarray] = []
    h = contract.h
    for step in range(contract.steps):
        k1 = _vector_field(states, parameters)
        k2 = _vector_field(states + 0.5 * h * k1, parameters)
        k3 = _vector_field(states + 0.5 * h * k2, parameters)
        k4 = _vector_field(states + h * k3, parameters)
        states += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if (
            step >= contract.burn_steps
            and (step - contract.burn_steps) % contract.stride == 0
        ):
            tail.append(states.copy())
        if (step + 1) % contract.progress_every == 0:
            print(
                f"{contract.name}: RK4 step {step + 1}/{contract.steps}",
                file=sys.stderr,
                flush=True,
            )
    sampled_states = np.asarray(tail)
    times = (
        np.arange(len(sampled_states), dtype=float) * contract.sampled_h
        + contract.t_burn
    )
    return times, sampled_states


def _initial_states(
    target_seeds: np.ndarray,
) -> np.ndarray:
    target = np.asarray(target_seeds, dtype=float)
    states = np.zeros((2, len(target), 3), dtype=float)
    states[0] = target
    states[1, :, 0] = 1.0e-3
    return states


def _screen_integer_tail(
    times: np.ndarray,
    sampled_states: np.ndarray,
    parameters: dict[str, np.ndarray],
    target_seeds: np.ndarray,
    contract: IntegerSearchContract,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], np.ndarray, np.ndarray]]]:
    """Apply the exact archived cKDTree separation and diagnostic screen."""

    rows: list[dict[str, Any]] = []
    saved: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    for index in range(contract.cases):
        target = sampled_states[:, 0, index]
        probe = sampled_states[:, 1, index]
        calibration = float(
            np.quantile(
                cKDTree(target[: contract.calibration_split]).query(
                    target[contract.calibration_split :]
                )[0],
                0.9,
            )
        )
        target_distance = float(
            np.quantile(cKDTree(target).query(probe)[0], 0.9)
        )
        symmetric_distance = float(
            np.quantile(cKDTree(-target).query(probe)[0], 0.9)
        )
        target_range = np.ptp(target, axis=0)
        probe_range = np.ptp(probe, axis=0)
        row: dict[str, Any] = {
            "i": index,
            **_parameter_row(parameters, index),
            "seed": target_seeds[index].tolist(),
            "target_range": target_range.tolist(),
            "probe_range": probe_range.tolist(),
            "calibration_nn90": calibration,
            "probe_target_nn90": target_distance,
            "probe_symmetric_nn90": symmetric_distance,
            "distinct": (
                min(target_distance, symmetric_distance)
                > 2.5 * max(calibration, 1.0e-6)
            ),
        }
        if float(target_range.max()) > 0.1 and row["distinct"]:
            row.update(
                _diagnose(
                    times,
                    target,
                    h=contract.sampled_h,
                    t_burn=contract.t_burn,
                    zero_one_samples=contract.zero_one_samples,
                )
            )
            if str(row.get("screen_label", "")).startswith(
                ("strong_", "chaos_")
            ):
                row["local"] = _matignon(
                    _parameter_row(parameters, index),
                    1.0,
                )
                saved.append((row, target, probe))
        rows.append(row)
    return rows, saved


def _save_ranked_candidate_trajectories(
    stage_dir: Path,
    times: np.ndarray,
    candidates: Sequence[
        tuple[dict[str, Any], np.ndarray, np.ndarray]
    ],
) -> list[str]:
    ranked = sorted(
        candidates,
        key=lambda item: float(item[0].get("score", -99.0)),
        reverse=True,
    )
    paths: list[str] = []
    for rank, (row, target, probe) in enumerate(ranked[:30]):
        path = stage_dir / f"candidate_{rank:02d}_i{row['i']}.npz"
        np.savez_compressed(
            path,
            times=times,
            states=target,
            probe_states=probe,
        )
        paths.append(path.name)
    return paths


def _screen_counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    nontrivial_distinct = [
        row
        for row in rows
        if row["distinct"] and max(row["target_range"]) > 0.1
    ]
    labels: dict[str, int] = {}
    for row in nontrivial_distinct:
        label = str(row.get("screen_label", "not_diagnosed"))
        labels[label] = labels.get(label, 0) + 1
    return {
        "all_rows": len(rows),
        "distinct_rows": sum(bool(row["distinct"]) for row in rows),
        "distinct_nontrivial_rows": len(nontrivial_distinct),
        "screen_labels_among_distinct_nontrivial": labels,
    }


def _rank_of(
    rows: Sequence[dict[str, Any]],
    index: int,
    *,
    label: str | None = None,
) -> int | None:
    eligible = [
        row
        for row in rows
        if row["distinct"]
        and max(row["target_range"]) > 0.1
        and (label is None or row.get("screen_label") == label)
    ]
    eligible.sort(
        key=lambda row: float(row.get("score", -99.0)),
        reverse=True,
    )
    for rank, row in enumerate(eligible, start=1):
        if int(row["i"]) == index:
            return rank
    return None


def _portable_path(path: Path, base: Path) -> str:
    """Return a path relative to an artifact root with POSIX separators."""

    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _aggregate_stage_regression(
    results: Sequence[dict[str, Any]],
) -> tuple[str, bool]:
    """Propagate explicit child-stage regression mismatches."""

    matched = all(
        result.get("recorded_regression_matched") is not False
        and "mismatch" not in str(result.get("status", ""))
        for result in results
    )
    return (
        (
            "completed_recorded_regression_matched"
            if matched
            else "completed_recorded_regression_mismatch"
        ),
        matched,
    )


def execute_banks(output_dir: Path) -> dict[str, Any]:
    """Persist the two exact PCG64 banks and their verified provenance."""

    output_dir = output_dir.resolve()
    provenance_path = output_dir / "search_provenance.json"
    bank_path = output_dir / "search_banks.npz"
    _require_new_outputs((provenance_path, bank_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = reconstruct()
    global_parameters, global_seeds = generate_global_bank()
    local_parameters, local_seeds = generate_local_bank(
        EXPECTED_GLOBAL_PARAMETERS,
        EXPECTED_GLOBAL_SEED,
    )
    np.savez_compressed(
        bank_path,
        parameter_names=np.asarray(PARAMETER_NAMES),
        global_parameters=np.column_stack(
            [global_parameters[name] for name in PARAMETER_NAMES]
        ),
        global_target_seeds=global_seeds,
        local_parameters=np.column_stack(
            [local_parameters[name] for name in PARAMETER_NAMES]
        ),
        local_target_seeds=local_seeds,
    )
    _write_json(provenance_path, provenance)
    return {
        "stage": "banks",
        "status": "completed",
        "provenance_json": str(provenance_path),
        "banks_npz": str(bank_path),
        "verification": provenance["verification"],
    }


def _execute_integer_search(
    output_dir: Path,
    contract: IntegerSearchContract,
) -> dict[str, Any]:
    stage_dir = output_dir.resolve() / contract.name
    primary_paths = (
        stage_dir / "initial_bank.npz",
        stage_dir / "screen.json",
        stage_dir / "summary.json",
    )
    _require_new_outputs(primary_paths)
    stage_dir.mkdir(parents=True, exist_ok=True)

    provenance = reconstruct()
    if contract is GLOBAL_CONTRACT:
        parameters, target_seeds = generate_global_bank()
        recorded_parameters = EXPECTED_GLOBAL_PARAMETERS
        recorded_seed = EXPECTED_GLOBAL_SEED
        recorded_npz_name = f"recorded_selection_i{GLOBAL_INDEX}.npz"
    elif contract is LOCAL_CONTRACT:
        parameters, target_seeds = generate_local_bank(
            EXPECTED_GLOBAL_PARAMETERS,
            EXPECTED_GLOBAL_SEED,
        )
        recorded_parameters = EXPECTED_C590_PARAMETERS
        recorded_seed = EXPECTED_C590_INTEGER_SEED
        recorded_npz_name = "recorded_selection_c590.npz"
    else:
        raise ValueError(f"unsupported integer contract: {contract.name}")

    parameter_matrix = np.column_stack(
        [parameters[name] for name in PARAMETER_NAMES]
    )
    np.savez_compressed(
        stage_dir / "initial_bank.npz",
        parameter_names=np.asarray(PARAMETER_NAMES),
        parameters=parameter_matrix,
        target_seeds=target_seeds,
        equilibrium_probe_seeds=_initial_states(target_seeds)[1],
    )

    times, sampled_states = _integrate_vectorized_rk4(
        _initial_states(target_seeds),
        parameters,
        contract,
    )
    rows, saved = _screen_integer_tail(
        times,
        sampled_states,
        parameters,
        target_seeds,
        contract,
    )
    _write_json(stage_dir / "screen.json", rows)

    recorded_row = rows[contract.recorded_index]
    recorded_path = stage_dir / recorded_npz_name
    np.savez_compressed(
        recorded_path,
        times=times,
        states=sampled_states[:, 0, contract.recorded_index],
        probe_states=sampled_states[:, 1, contract.recorded_index],
        parameters=np.asarray(
            [recorded_parameters[name] for name in PARAMETER_NAMES],
            dtype=float,
        ),
        seed=np.asarray(recorded_seed, dtype=float),
    )
    candidate_paths = _save_ranked_candidate_trajectories(
        stage_dir,
        times,
        saved,
    )

    counts = _screen_counts(rows)
    recorded_checks: dict[str, Any] = {
        "zero_based_index": contract.recorded_index,
        "parameters_match_bank": all(
            recorded_row[name] == recorded_parameters[name]
            for name in PARAMETER_NAMES
        ),
        "seed_match_bank": bool(
            np.array_equal(recorded_row["seed"], recorded_seed)
        ),
        "screen_label": recorded_row.get("screen_label"),
        "score": recorded_row.get("score"),
        "rank_among_distinct_nontrivial": _rank_of(
            rows,
            contract.recorded_index,
        ),
        "rank_among_inconclusive_nonperiodic": _rank_of(
            rows,
            contract.recorded_index,
            label="inconclusive_nonperiodic",
        ),
    }
    if contract is GLOBAL_CONTRACT:
        expected_checks = {
            "recorded_index": GLOBAL_INDEX,
            "expected_inconclusive_nonperiodic_rows": 61,
            "expected_inconclusive_rank": 1,
        }
        observed_match = (
            counts["screen_labels_among_distinct_nontrivial"].get(
                "inconclusive_nonperiodic",
                0,
            )
            == 61
            and recorded_checks["rank_among_inconclusive_nonperiodic"] == 1
        )
    else:
        local_screening = provenance["local_exploration"]["screening"]
        expected_checks = {
            "recorded_index": LOCAL_INDEX,
            "expected_distinct_nontrivial_rows": local_screening[
                "distinct_nontrivial_cases"
            ],
            "expected_inconclusive_nonperiodic_rows": local_screening[
                "nonperiodic_inconclusive_cases"
            ],
            "expected_regular_periodic_rows": local_screening[
                "regular_periodic_cases"
            ],
            "expected_c590_score_rank": local_screening[
                "c590_score_rank_among_distinct_nontrivial"
            ],
        }
        observed_match = (
            counts["distinct_nontrivial_rows"]
            == local_screening["distinct_nontrivial_cases"]
            and counts["screen_labels_among_distinct_nontrivial"].get(
                "inconclusive_nonperiodic",
                0,
            )
            == local_screening["nonperiodic_inconclusive_cases"]
            and counts["screen_labels_among_distinct_nontrivial"].get(
                "regular_periodic_rejected",
                0,
            )
            == local_screening["regular_periodic_cases"]
            and recorded_checks["rank_among_distinct_nontrivial"]
            == local_screening["c590_score_rank_among_distinct_nontrivial"]
        )

    summary = {
        "schema_version": "2.0",
        "stage": contract.name,
        "status": (
            "completed_recorded_regression_matched"
            if observed_match
            else "completed_recorded_regression_mismatch"
        ),
        "contract": contract.as_dict(),
        "diagnostic_implementation": (
            "tools.arctan_hidden_screen._diagnose"
        ),
        "stability_implementation": (
            "tools.arctan_hidden_screen._matignon"
        ),
        "separation_rule": (
            "min(probe_target_nn90,probe_symmetric_nn90) "
            "> 2.5*max(calibration_nn90,1e-6)"
        ),
        "counts": counts,
        "recorded_selection": recorded_checks,
        "expected_regression": expected_checks,
        "recorded_regression_matched": observed_match,
        "outputs": {
            "initial_bank": "initial_bank.npz",
            "screen": "screen.json",
            "recorded_selection_trajectory": recorded_path.name,
            "ranked_candidate_trajectories": candidate_paths,
        },
        "scientific_boundary": (
            "This is an integer-order exploratory screening result. The "
            "diagnostic label and score do not by themselves certify a hidden "
            "attractor or a fractional-order attractor."
        ),
    }
    _write_json(stage_dir / "summary.json", summary)
    return summary


def execute_integer_global(output_dir: Path) -> dict[str, Any]:
    """Run the recovered 2,400-row global integer search."""

    return _execute_integer_search(output_dir, GLOBAL_CONTRACT)


def execute_integer_local(output_dir: Path) -> dict[str, Any]:
    """Run the recovered 1,000-row local integer refinement."""

    return _execute_integer_search(output_dir, LOCAL_CONTRACT)


def _variational_field_and_tangent(
    states: np.ndarray,
    tangent_frames: np.ndarray | None,
    parameters: dict[str, np.ndarray],
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    phi = (
        parameters["a1"] * states[:, 0]
        + parameters["a2"]
        * np.arctan(parameters["rho"] * states[:, 0])
    )
    field = np.column_stack(
        (
            parameters["alpha"] * (
                states[:, 1] - states[:, 0] - phi
            ),
            states[:, 0] - states[:, 1] + states[:, 2],
            -parameters["beta"] * states[:, 1]
            - parameters["gamma"] * states[:, 2],
        )
    )
    if tangent_frames is None:
        return field

    count = len(states)
    derivative = (
        parameters["a1"]
        + parameters["a2"]
        * parameters["rho"]
        / (1.0 + (parameters["rho"] * states[:, 0]) ** 2)
    )
    jacobian = np.zeros((count, 3, 3), dtype=float)
    jacobian[:, 0, 0] = -parameters["alpha"] * (
        1.0 + derivative
    )
    jacobian[:, 0, 1] = parameters["alpha"]
    jacobian[:, 1] = [1.0, -1.0, 1.0]
    jacobian[:, 2, 1] = -parameters["beta"]
    jacobian[:, 2, 2] = -parameters["gamma"]
    return field, jacobian @ tangent_frames


def _rk4_variational_shortlist(
    states: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the exact archived state burn and Benettin/QR measurement."""

    states = np.asarray(states, dtype=float).copy()
    h = float(VARIATIONAL_CONTRACT["h"])
    burn_steps = int(VARIATIONAL_CONTRACT["state_burn_steps"])
    measurement_steps = int(VARIATIONAL_CONTRACT["measurement_steps"])
    qr_every = int(
        VARIATIONAL_CONTRACT["reorthonormalize_every_steps"]
    )

    for step in range(burn_steps):
        k1 = _variational_field_and_tangent(states, None, parameters)
        k2 = _variational_field_and_tangent(
            states + 0.5 * h * k1,
            None,
            parameters,
        )
        k3 = _variational_field_and_tangent(
            states + 0.5 * h * k2,
            None,
            parameters,
        )
        k4 = _variational_field_and_tangent(
            states + h * k3,
            None,
            parameters,
        )
        states += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if (step + 1) % 10_000 == 0:
            print(
                f"variational-shortlist: burn step "
                f"{step + 1}/{burn_steps}",
                file=sys.stderr,
                flush=True,
            )

    count = len(states)
    frames = np.repeat(np.eye(3)[None, :, :], count, axis=0)
    sums = np.zeros((count, 3), dtype=float)
    for step in range(measurement_steps):
        f1, v1 = _variational_field_and_tangent(
            states,
            frames,
            parameters,
        )
        f2, v2 = _variational_field_and_tangent(
            states + 0.5 * h * f1,
            frames + 0.5 * h * v1,
            parameters,
        )
        f3, v3 = _variational_field_and_tangent(
            states + 0.5 * h * f2,
            frames + 0.5 * h * v2,
            parameters,
        )
        f4, v4 = _variational_field_and_tangent(
            states + h * f3,
            frames + h * v3,
            parameters,
        )
        states += h * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0
        frames += h * (v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0
        if (step + 1) % qr_every == 0:
            orthonormal, upper = np.linalg.qr(frames)
            sums += np.log(
                np.maximum(
                    np.abs(
                        np.diagonal(upper, axis1=1, axis2=2)
                    ),
                    1.0e-300,
                )
            )
            frames = orthonormal
        if (step + 1) % 10_000 == 0:
            print(
                f"variational-shortlist: measurement step "
                f"{step + 1}/{measurement_steps}",
                file=sys.stderr,
                flush=True,
            )
    exponents = sums / float(
        VARIATIONAL_CONTRACT["measurement_time"]
    )
    return exponents, states, frames


def execute_variational_shortlist(
    output_dir: Path,
    local_screen: Path | None = None,
) -> dict[str, Any]:
    """Run the recovered top-30 integer RK4/Benettin audit."""

    output_dir = output_dir.resolve()
    screen_path = (
        local_screen.resolve()
        if local_screen is not None
        else output_dir / "integer_local" / "screen.json"
    )
    if not screen_path.exists():
        raise FileNotFoundError(
            "The variational stage requires the integer-local screen. "
            f"Missing: {screen_path}"
        )

    stage_dir = output_dir / "variational_shortlist"
    json_path = stage_dir / "variational_shortlist.json"
    npz_path = stage_dir / "variational_shortlist.npz"
    _require_new_outputs((json_path, npz_path))
    stage_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(screen_path.read_text(encoding="utf-8"))
    shortlist = sorted(
        [
            row
            for row in rows
            if row.get("screen_label")
            == "inconclusive_nonperiodic"
        ],
        key=lambda row: float(row.get("score", -9.0)),
        reverse=True,
    )[: int(VARIATIONAL_CONTRACT["candidates"])]
    if len(shortlist) != int(VARIATIONAL_CONTRACT["candidates"]):
        raise MissingDiscoveryContract(
            "The local screen does not contain the 30 "
            "inconclusive_nonperiodic rows required by the recovered "
            f"variational contract; found {len(shortlist)}."
        )

    parameters = {
        name: np.asarray([row[name] for row in shortlist], dtype=float)
        for name in PARAMETER_NAMES
    }
    initial_states = np.asarray(
        [row["seed"] for row in shortlist],
        dtype=float,
    )
    exponents, final_states, final_frames = (
        _rk4_variational_shortlist(initial_states, parameters)
    )

    result_rows: list[dict[str, Any]] = []
    for row, exponent in zip(shortlist, exponents):
        parameter_row = {
            name: float(row[name]) for name in PARAMETER_NAMES
        }
        matignon = _matignon(parameter_row, 1.0)
        stability = matignon["stability"]
        equilibrium_pattern = (
            stability.get("E0") == "unstable"
            and stability.get("E+") == "stable"
            and stability.get("E-") == "stable"
        )
        result_rows.append(
            {
                "source_i": int(row["i"]),
                "source_score": float(row["score"]),
                "source_zero_one_K_robust_median": row.get(
                    "zero_one_K_robust_median"
                ),
                "parameters": parameter_row,
                "seed": row["seed"],
                "source_distinct": bool(row["distinct"]),
                "source_target_nontrivial": (
                    max(row["target_range"]) > 0.1
                ),
                "matignon": matignon,
                "equilibrium_stability_pattern_eligible": (
                    equilibrium_pattern
                ),
                "integer_variational_exponents": exponent.tolist(),
                "largest_integer_variational_exponent": float(
                    exponent[0]
                ),
            }
        )

    ranking = sorted(
        result_rows,
        key=lambda row: row[
            "largest_integer_variational_exponent"
        ],
        reverse=True,
    )
    eligible = [
        row
        for row in ranking
        if row["source_distinct"]
        and row["source_target_nontrivial"]
        and row["equilibrium_stability_pattern_eligible"]
    ]
    selected = eligible[0] if eligible else None
    c590 = next(
        (row for row in result_rows if row["source_i"] == LOCAL_INDEX),
        None,
    )
    c590_exponent = (
        None
        if c590 is None
        else c590["largest_integer_variational_exponent"]
    )
    recorded_exponent = float(
        VARIATIONAL_CONTRACT[
            "recorded_c590_largest_exponent"
        ]
    )
    recorded_match = bool(
        selected is not None
        and selected["source_i"] == LOCAL_INDEX
        and c590_exponent is not None
        and np.isclose(
            c590_exponent,
            recorded_exponent,
            rtol=0.0,
            atol=1.0e-10,
        )
    )

    np.savez_compressed(
        npz_path,
        candidate_indices=np.asarray(
            [row["i"] for row in shortlist],
            dtype=int,
        ),
        parameter_names=np.asarray(PARAMETER_NAMES),
        parameters=np.column_stack(
            [parameters[name] for name in PARAMETER_NAMES]
        ),
        initial_states=initial_states,
        exponents=exponents,
        final_states=final_states,
        final_tangent_frames=final_frames,
    )
    payload = {
        "schema_version": "2.0",
        "stage": "variational_shortlist",
        "status": (
            "completed_recorded_regression_matched"
            if recorded_match
            else "completed_recorded_regression_mismatch"
        ),
        "source_screen": _portable_path(screen_path, output_dir),
        "contract": VARIATIONAL_CONTRACT,
        "shortlist_rule": (
            "screen_label == inconclusive_nonperiodic; descending score; "
            "first 30 rows"
        ),
        "selection_rule": (
            "maximum largest integer variational exponent among bounded, "
            "distinct rows with unstable E0 and stable E+ and E-"
        ),
        "rows": result_rows,
        "ranking_by_largest_exponent": [
            row["source_i"] for row in ranking
        ],
        "eligible_ranking": [row["source_i"] for row in eligible],
        "selected_source_i": (
            None if selected is None else selected["source_i"]
        ),
        "c590": {
            "present_in_shortlist": c590 is not None,
            "computed_largest_exponent": c590_exponent,
            "recorded_largest_exponent": recorded_exponent,
            "absolute_error": (
                None
                if c590_exponent is None
                else abs(c590_exponent - recorded_exponent)
            ),
        },
        "recorded_regression_matched": recorded_match,
        "outputs": {
            "json": json_path.name,
            "npz": npz_path.name,
        },
        "scientific_boundary": (
            "The Benettin result is an integer-order finite-time variational "
            "audit used for discovery. Fractional-order classification and "
            "equilibrium-neighborhood hiddenness tests are separate stages."
        ),
    }
    _write_json(json_path, payload)
    return payload


def _number_tag(value: float) -> str:
    return f"{value:.10g}".replace("-", "m").replace(".", "p")


def _integrate_caputo(
    seed: np.ndarray,
    *,
    q: float,
    h: float,
    t_final: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Integrate one recovered full-memory Caputo case."""

    parameters = {
        name: float(EXPECTED_C590_PARAMETERS[name])
        for name in PARAMETER_NAMES
    }
    system = _build_system(parameters, q)
    return integrate_general(
        lambda _time, state, active=system: active.evaluate(state),
        np.asarray(seed, dtype=float),
        q=q,
        h=h,
        t_final=t_final,
        integrator="abm",
        memory_mode="full",
        system=system,
        use_c_backend=True,
        divergence_norm=300.0,
        early_stop_config={"enabled": False},
    )


def _caputo_case_metrics(
    times: np.ndarray,
    states: np.ndarray,
    status: str,
    *,
    q: float,
    h: float,
    t_final: float,
    t_burn: float,
    seed: np.ndarray,
) -> dict[str, Any]:
    tail = states[times >= t_burn]
    return {
        "q": q,
        "h": h,
        "status": status,
        "end_time": float(times[-1]),
        "max_norm": float(np.max(np.linalg.norm(states, axis=1))),
        "range": (
            np.ptp(tail, axis=0).tolist() if len(tail) else []
        ),
        "t_final": t_final,
        "t_burn": t_burn,
        "seed": np.asarray(seed, dtype=float).tolist(),
    }


def _resampled_zero_one(
    times: np.ndarray,
    states: np.ndarray,
    *,
    h: float,
    t_burn: float,
    n_c: int,
) -> list[dict[str, Any]]:
    tail_mask = times >= t_burn
    tail_times = times[tail_mask]
    tail_states = states[tail_mask]
    rows: list[dict[str, Any]] = []
    for interval in (0.5, 0.75, 1.0, 1.5):
        stride = round(interval / h)
        sampled_times = tail_times[::stride]
        sampled_states = tail_states[::stride]
        if len(sampled_times) < 100:
            continue
        result = zero_one_multicoordinate(
            sampled_times,
            sampled_states,
            float(sampled_times[0]),
            n_c=n_c,
            max_samples=None,
        )
        rows.append(
            {
                "sample_interval": interval,
                "stride_steps": stride,
                "samples": len(sampled_times),
                "K_global_median": float(
                    result["K_global_median"]
                ),
                "state_global": result["state_global"],
            }
        )
    return rows


def _integrate_and_store_caputo(
    path: Path,
    seed: np.ndarray,
    *,
    q: float,
    h: float,
    t_final: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    times, states, status = _integrate_caputo(
        seed,
        q=q,
        h=h,
        t_final=t_final,
    )
    np.savez_compressed(
        path,
        times=times,
        states=states,
        seed=np.asarray(seed, dtype=float),
        q=np.asarray(q),
        h=np.asarray(h),
    )
    return times, states, status


def _bounded_case(
    row: dict[str, Any],
    *,
    norm_limit: float = 50.0,
) -> bool:
    return (
        row["status"] == "ok"
        and float(row["max_norm"]) < norm_limit
    )


def _robust_zero_one_case(
    row: dict[str, Any],
    *,
    norm_limit: float = 50.0,
    k_threshold: float = 0.8,
) -> bool:
    return (
        _bounded_case(row, norm_limit=norm_limit)
        and row["K_median"] is not None
        and float(row["K_median"]) > k_threshold
    )


def execute_caputo_seed9(output_dir: Path) -> dict[str, Any]:
    """Run the complete recovered full-memory Caputo seed9 route."""

    output_dir = output_dir.resolve()
    stage_dir = output_dir / "caputo_seed9"
    _require_new_outputs((stage_dir,))
    stage_dir.mkdir(parents=True)

    parameters = {
        name: float(EXPECTED_C590_PARAMETERS[name])
        for name in PARAMETER_NAMES
    }
    integer_seed = np.asarray(
        EXPECTED_C590_INTEGER_SEED,
        dtype=float,
    )

    # 1. Recovered q scan (historical script 1414).
    q_scan_dir = stage_dir / "q_scan"
    q_scan_dir.mkdir()
    q_scan_data: dict[
        float,
        tuple[np.ndarray, np.ndarray, str, Path],
    ] = {}
    q_scan_rows: list[dict[str, Any]] = []
    for q in CAPUTO_CONTRACT["q_scan"]["q_values"]:
        q_value = float(q)
        trajectory_path = q_scan_dir / f"q{_number_tag(q_value)}.npz"
        times, states, status = _integrate_and_store_caputo(
            trajectory_path,
            integer_seed,
            q=q_value,
            h=0.005,
            t_final=300.0,
        )
        tail = states[times >= 150.0]
        diagnostics = _diagnose(
            times,
            states,
            h=0.005,
            t_burn=150.0,
            zero_one_samples=(2000, 4000, 8000),
        )
        row = {
            **_caputo_case_metrics(
                times,
                states,
                status,
                q=q_value,
                h=0.005,
                t_final=300.0,
                t_burn=150.0,
                seed=integer_seed,
            ),
            "range": np.ptp(tail, axis=0).tolist(),
            **diagnostics,
            "matignon": _matignon(parameters, q_value),
            "trajectory": trajectory_path.name,
        }
        q_scan_rows.append(row)
        q_scan_data[q_value] = (
            times,
            states,
            status,
            trajectory_path,
        )
        print(
            f"caputo-seed9 q scan: q={q_value} "
            f"{diagnostics['screen_label']}",
            file=sys.stderr,
            flush=True,
        )
    q_scan_payload = {
        "schema_version": "2.0",
        "stage": "caputo_q_scan",
        "contract": CAPUTO_CONTRACT["q_scan"],
        "rows": q_scan_rows,
        "recorded_selected_q": 0.9999,
    }
    _write_json(q_scan_dir / "summary.json", q_scan_payload)

    # 2. Supplemental q=0.9997 trajectory and the recovered resampled
    #    0-1 audit for q=0.9997, 0.9998, and 0.9999.
    neighbour_dir = stage_dir / "q_neighbour"
    neighbour_dir.mkdir()
    neighbour_path = neighbour_dir / "q0p9997.npz"
    neighbour_times, neighbour_states, neighbour_status = (
        _integrate_and_store_caputo(
            neighbour_path,
            integer_seed,
            q=0.9997,
            h=0.005,
            t_final=260.0,
        )
    )
    neighbour_diagnostics = _diagnose(
        neighbour_times,
        neighbour_states,
        h=0.005,
        t_burn=130.0,
        zero_one_samples=(2000, 4000, 8000),
    )
    _write_json(
        neighbour_dir / "summary.json",
        {
            "schema_version": "2.0",
            "stage": "caputo_q_neighbour",
            **_caputo_case_metrics(
                neighbour_times,
                neighbour_states,
                neighbour_status,
                q=0.9997,
                h=0.005,
                t_final=260.0,
                t_burn=130.0,
                seed=integer_seed,
            ),
            **neighbour_diagnostics,
            "matignon": _matignon(parameters, 0.9997),
            "trajectory": neighbour_path.name,
        },
    )
    resampled_sources = {
        0.9997: (
            neighbour_times,
            neighbour_states,
            130.0,
            str(neighbour_path.relative_to(stage_dir)),
        ),
        0.9998: (
            q_scan_data[0.9998][0],
            q_scan_data[0.9998][1],
            150.0,
            str(q_scan_data[0.9998][3].relative_to(stage_dir)),
        ),
        0.9999: (
            q_scan_data[0.9999][0],
            q_scan_data[0.9999][1],
            150.0,
            str(q_scan_data[0.9999][3].relative_to(stage_dir)),
        ),
    }
    resampled_rows = []
    for q_value, (
        times,
        states,
        t_burn,
        source_path,
    ) in resampled_sources.items():
        resampled_rows.append(
            {
                "q": q_value,
                "h": 0.005,
                "t_burn": t_burn,
                "source_trajectory": source_path,
                "rows": _resampled_zero_one(
                    times,
                    states,
                    h=0.005,
                    t_burn=t_burn,
                    n_c=100,
                ),
            }
        )
    resampled_payload = {
        "schema_version": "2.0",
        "stage": "resampled_zero_one",
        "contract": CAPUTO_CONTRACT["resampled_zero_one"],
        "rows": resampled_rows,
    }
    _write_json(
        stage_dir / "resampled_zero_one.json",
        resampled_payload,
    )

    # 3. Three-step audit of the integer c590 seed. The h=0.005 result is
    #    reused from the q scan because every numerical input is identical.
    h_audit_dir = stage_dir / "integer_seed_h_audit"
    h_audit_dir.mkdir()
    h_audit_data: dict[
        float,
        tuple[np.ndarray, np.ndarray, str, Path],
    ] = {
        0.005: (
            q_scan_data[0.9999][0],
            q_scan_data[0.9999][1],
            q_scan_data[0.9999][2],
            q_scan_data[0.9999][3],
        )
    }

    def run_integer_seed_h(
        h_value: float,
    ) -> tuple[float, np.ndarray, np.ndarray, str, Path]:
        path = h_audit_dir / f"h{_number_tag(h_value)}.npz"
        times, states, status = _integrate_and_store_caputo(
            path,
            integer_seed,
            q=0.9999,
            h=h_value,
            t_final=300.0,
        )
        return h_value, times, states, status, path

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_integer_seed_h, h_value)
            for h_value in (0.0025, 0.01)
        ]
        for future in as_completed(futures):
            h_value, times, states, status, path = future.result()
            h_audit_data[h_value] = (
                times,
                states,
                status,
                path,
            )

    h_audit_rows: list[dict[str, Any]] = []
    for h_value in (0.0025, 0.005, 0.01):
        times, states, status, path = h_audit_data[h_value]
        zero_one_rows = (
            _resampled_zero_one(
                times,
                states,
                h=h_value,
                t_burn=150.0,
                n_c=100,
            )
            if status == "ok"
            else []
        )
        k_values = [
            float(row["K_global_median"])
            for row in zero_one_rows
        ]
        row = {
            **_caputo_case_metrics(
                times,
                states,
                status,
                q=0.9999,
                h=h_value,
                t_final=300.0,
                t_burn=150.0,
                seed=integer_seed,
            ),
            "K_by_dt": {
                str(item["sample_interval"]): item[
                    "K_global_median"
                ]
                for item in zero_one_rows
            },
            "K_median": (
                float(np.median(k_values)) if k_values else None
            ),
            "matignon": _matignon(parameters, 0.9999),
            "trajectory": str(path.relative_to(stage_dir)),
            "reused_from_q_scan": h_value == 0.005,
        }
        row["bounded"] = _bounded_case(
            row,
            norm_limit=float(
                CAPUTO_CONTRACT["integer_seed_h_audit"][
                    "bounded_norm"
                ]
            ),
        )
        h_audit_rows.append(row)
    robust_bounded = all(row["bounded"] for row in h_audit_rows)
    robust_chaos_screen = all(
        row["K_median"] is not None
        and float(row["K_median"]) > 0.8
        for row in h_audit_rows
    )
    h_audit_match = (
        robust_bounded
        == bool(
            CAPUTO_CONTRACT["integer_seed_h_audit"][
                "recorded_robust_bounded"
            ]
        )
        and robust_chaos_screen
        == bool(
            CAPUTO_CONTRACT["integer_seed_h_audit"][
                "recorded_robust_chaos_screen"
            ]
        )
    )
    h_audit_payload = {
        "schema_version": "2.0",
        "stage": "integer_seed_h_audit",
        "contract": CAPUTO_CONTRACT["integer_seed_h_audit"],
        "rows": h_audit_rows,
        "robust_bounded": robust_bounded,
        "robust_chaos_screen": robust_chaos_screen,
        "recorded_regression_matched": h_audit_match,
    }
    _write_json(h_audit_dir / "summary.json", h_audit_payload)

    # 4. Extract 16 equally indexed tail states from the q=0.9999, h=0.005
    #    source trajectory, then screen each at h=0.0025 through T=180.
    source_times = q_scan_data[0.9999][0]
    source_states = q_scan_data[0.9999][1]
    source_indices = np.linspace(
        np.searchsorted(source_times, 150.0),
        len(source_times) - 1,
        16,
        dtype=int,
    )
    expected_source_indices = np.asarray(
        CAPUTO_CONTRACT["seed_extraction"][
            "recorded_source_indices"
        ],
        dtype=int,
    )
    expected_source_times = np.asarray(
        CAPUTO_CONTRACT["seed_extraction"][
            "recorded_source_times"
        ],
        dtype=float,
    )
    source_grid_match = bool(
        np.array_equal(source_indices, expected_source_indices)
        and np.allclose(
            source_times[source_indices],
            expected_source_times,
            rtol=0.0,
            atol=5.0e-12,
        )
    )
    if not source_grid_match:
        raise RuntimeError(
            "The q=0.9999 source trajectory does not reproduce the recorded "
            "16-seed index/time grid."
        )
    seeds = np.asarray(
        [source_states[index].copy() for index in source_indices],
        dtype=float,
    )
    source_states_error = float(
        np.max(np.abs(seeds - EXPECTED_EXTRACTED_SEEDS))
    )
    source_states_match = source_states_error <= 5.0e-12
    seed9_exact = bool(
        np.array_equal(seeds[9], EXPECTED_FINAL_SEED)
    )
    if not source_states_match or not seed9_exact:
        raise RuntimeError(
            "The q=0.9999 source trajectory does not reproduce the 16 "
            "recorded historical states exactly enough: "
            f"max_abs_error={source_states_error}, "
            f"seed9_exact={seed9_exact}."
        )
    seed_dir = stage_dir / "seed_refinement"
    seed_dir.mkdir()
    np.savez_compressed(
        seed_dir / "seeds.npz",
        source_indices=source_indices,
        source_times=source_times[source_indices],
        seeds=seeds,
    )

    def run_seed_refinement(
        seed_index: int,
    ) -> tuple[dict[str, Any], Path]:
        path = seed_dir / f"seed{seed_index:02d}.npz"
        times, states, status = _integrate_and_store_caputo(
            path,
            seeds[seed_index],
            q=0.9999,
            h=0.0025,
            t_final=180.0,
        )
        row = {
            "i": seed_index,
            "source_index": int(source_indices[seed_index]),
            "source_time": float(
                source_times[source_indices[seed_index]]
            ),
            **_caputo_case_metrics(
                times,
                states,
                status,
                q=0.9999,
                h=0.0025,
                t_final=180.0,
                t_burn=90.0,
                seed=seeds[seed_index],
            ),
            "trajectory": path.name,
        }
        return row, path

    seed_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(run_seed_refinement, seed_index)
            for seed_index in range(16)
        ]
        for future in as_completed(futures):
            row, _path = future.result()
            seed_rows.append(row)
            print(
                f"caputo-seed9 seed refinement: "
                f"seed{row['i']:02d} {row['status']}",
                file=sys.stderr,
                flush=True,
            )
    seed_rows.sort(key=lambda row: int(row["i"]))
    bounded_indices = [
        int(row["i"]) for row in seed_rows if _bounded_case(row)
    ]
    expected_bounded = list(
        CAPUTO_CONTRACT["seed_refinement"][
            "recorded_bounded_indices"
        ]
    )
    seed_payload = {
        "schema_version": "2.0",
        "stage": "seed_refinement",
        "contract": CAPUTO_CONTRACT["seed_refinement"],
        "source_trajectory": str(
            q_scan_data[0.9999][3].relative_to(stage_dir)
        ),
        "source_indices": source_indices.tolist(),
        "source_times": source_times[source_indices].tolist(),
        "recorded_source_indices": expected_source_indices.tolist(),
        "recorded_source_times": expected_source_times.tolist(),
        "source_grid_regression_matched": source_grid_match,
        "recorded_source_states": EXPECTED_EXTRACTED_SEEDS.tolist(),
        "source_states_max_abs_error": source_states_error,
        "source_states_regression_matched": source_states_match,
        "seed9_exact_match": seed9_exact,
        "rows": seed_rows,
        "bounded_indices": bounded_indices,
        "recorded_bounded_indices": expected_bounded,
        "recorded_regression_matched": (
            bounded_indices == expected_bounded
        ),
    }
    _write_json(seed_dir / "summary.json", seed_payload)
    if bounded_indices != expected_bounded:
        raise RuntimeError(
            "The 16-seed refinement did not reproduce the recorded bounded "
            f"indices: observed={bounded_indices}, expected={expected_bounded}."
        )

    # 5. Complete the T=180 cross-step filter at h=0.005 and h=0.01.
    cross_dir = stage_dir / "cross_step"
    cross_dir.mkdir()

    def run_cross_step(
        seed_index: int,
        h_value: float,
    ) -> dict[str, Any]:
        path = (
            cross_dir
            / f"seed{seed_index:02d}_h{_number_tag(h_value)}.npz"
        )
        times, states, status = _integrate_and_store_caputo(
            path,
            seeds[seed_index],
            q=0.9999,
            h=h_value,
            t_final=180.0,
        )
        return {
            "i": seed_index,
            **_caputo_case_metrics(
                times,
                states,
                status,
                q=0.9999,
                h=h_value,
                t_final=180.0,
                t_burn=90.0,
                seed=seeds[seed_index],
            ),
            "trajectory": path.name,
        }

    cross_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                run_cross_step,
                seed_index,
                h_value,
            )
            for seed_index in bounded_indices
            for h_value in (0.005, 0.01)
        ]
        for future in as_completed(futures):
            cross_rows.append(future.result())
    cross_rows.sort(key=lambda row: (int(row["i"]), float(row["h"])))
    cross_passing = [
        seed_index
        for seed_index in bounded_indices
        if all(
            _bounded_case(row)
            for row in cross_rows
            if int(row["i"]) == seed_index
        )
    ]
    expected_cross_passing = list(
        CAPUTO_CONTRACT["cross_step"][
            "recorded_passing_indices"
        ]
    )
    cross_payload = {
        "schema_version": "2.0",
        "stage": "cross_step",
        "contract": CAPUTO_CONTRACT["cross_step"],
        "rows": cross_rows,
        "passing_all_h": cross_passing,
        "seeds": {
            str(index): seeds[index].tolist()
            for index in cross_passing
        },
        "recorded_passing_indices": expected_cross_passing,
        "recorded_regression_matched": (
            cross_passing == expected_cross_passing
        ),
    }
    _write_json(cross_dir / "summary.json", cross_payload)
    if cross_passing != expected_cross_passing:
        raise RuntimeError(
            "The short cross-step filter did not reproduce the recorded "
            f"indices: observed={cross_passing}, "
            f"expected={expected_cross_passing}."
        )

    # 6. Reproduce the separate long seed5 rejection with n_c=100.
    seed5_dir = stage_dir / "seed5_long_audit"
    seed5_dir.mkdir()

    def run_long_case(
        destination: Path,
        seed_index: int,
        h_value: float,
        n_c: int,
    ) -> dict[str, Any]:
        path = (
            destination
            / f"seed{seed_index:02d}_h{_number_tag(h_value)}.npz"
        )
        times, states, status = _integrate_and_store_caputo(
            path,
            seeds[seed_index],
            q=0.9999,
            h=h_value,
            t_final=300.0,
        )
        zero_one_rows = (
            _resampled_zero_one(
                times,
                states,
                h=h_value,
                t_burn=150.0,
                n_c=n_c,
            )
            if status == "ok"
            and float(np.max(np.linalg.norm(states, axis=1))) < 50.0
            else []
        )
        k_values = [
            float(row["K_global_median"])
            for row in zero_one_rows
        ]
        return {
            "i": seed_index,
            **_caputo_case_metrics(
                times,
                states,
                status,
                q=0.9999,
                h=h_value,
                t_final=300.0,
                t_burn=150.0,
                seed=seeds[seed_index],
            ),
            "K_values": k_values,
            "K_median": (
                float(np.median(k_values)) if k_values else None
            ),
            "trajectory": path.name,
        }

    seed5_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                run_long_case,
                seed5_dir,
                5,
                h_value,
                100,
            )
            for h_value in (0.0025, 0.005, 0.01)
        ]
        for future in as_completed(futures):
            seed5_rows.append(future.result())
    seed5_rows.sort(key=lambda row: float(row["h"]))
    seed5_robust = all(
        _robust_zero_one_case(row) for row in seed5_rows
    )
    seed5_payload = {
        "schema_version": "2.0",
        "stage": "seed5_long_audit",
        "contract": CAPUTO_CONTRACT["seed5_long_audit"],
        "seed": seeds[5].tolist(),
        "rows": seed5_rows,
        "robust_bounded_and_zero_one": seed5_robust,
        "recorded_result": "rejected",
        "recorded_regression_matched": not seed5_robust,
    }
    _write_json(seed5_dir / "summary.json", seed5_payload)
    if seed5_robust:
        raise RuntimeError(
            "The long seed5 audit did not reproduce its recorded rejection."
        )

    # 7. Audit the five surviving seed indices through T=300 with n_c=80.
    survivor_indices = [
        index for index in cross_passing if index != 5
    ]
    expected_survivors = list(
        CAPUTO_CONTRACT["survivor_long_audit"]["seed_indices"]
    )
    if survivor_indices != expected_survivors:
        raise RuntimeError(
            "The post-seed5 survivor set differs from the recorded contract: "
            f"observed={survivor_indices}, expected={expected_survivors}."
        )
    long_dir = stage_dir / "survivor_long_audit"
    long_dir.mkdir()
    long_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                run_long_case,
                long_dir,
                seed_index,
                h_value,
                80,
            )
            for seed_index in survivor_indices
            for h_value in (0.0025, 0.005, 0.01)
        ]
        for future in as_completed(futures):
            long_rows.append(future.result())
    long_rows.sort(key=lambda row: (int(row["i"]), float(row["h"])))
    long_bounded = [
        seed_index
        for seed_index in survivor_indices
        if all(
            _bounded_case(row)
            for row in long_rows
            if int(row["i"]) == seed_index
        )
    ]
    long_passing = [
        seed_index
        for seed_index in long_bounded
        if all(
            row["K_median"] is not None
            and float(row["K_median"]) > 0.8
            for row in long_rows
            if int(row["i"]) == seed_index
            and float(row["h"]) in (0.0025, 0.005)
        )
    ]
    expected_long_passing = list(
        CAPUTO_CONTRACT["survivor_long_audit"][
            "recorded_passing_indices"
        ]
    )
    long_payload = {
        "schema_version": "2.0",
        "stage": "survivor_long_audit",
        "contract": CAPUTO_CONTRACT["survivor_long_audit"],
        "rows": long_rows,
        "bounded_at_all_three_h": long_bounded,
        "passing": long_passing,
        "seeds": {
            str(index): seeds[index].tolist()
            for index in long_passing
        },
        "recorded_passing_indices": expected_long_passing,
        "recorded_regression_matched": (
            long_passing == expected_long_passing
        ),
    }
    _write_json(long_dir / "summary.json", long_payload)
    if long_passing != expected_long_passing:
        raise RuntimeError(
            "The long three-step filter did not reproduce seed9 as the "
            f"unique passing seed: observed={long_passing}, "
            f"expected={expected_long_passing}."
        )

    # 8. Materialize the selected seed9 target from the identical h=0.005
    #    long-audit trajectory.
    selected_seed = seeds[9]
    selected_seed_error = float(
        np.max(np.abs(selected_seed - EXPECTED_FINAL_SEED))
    )
    selected_source = (
        long_dir / f"seed09_h{_number_tag(0.005)}.npz"
    )
    with np.load(selected_source) as source:
        target_times = source["times"].copy()
        target_states = source["states"].copy()
    target_path = stage_dir / "target.npz"
    np.savez_compressed(
        target_path,
        times=target_times,
        states=target_states,
        seed=selected_seed,
        q=np.asarray(0.9999),
        h=np.asarray(0.005),
    )
    target_tail = target_states[target_times >= 150.0]
    target_payload = {
        "schema_version": "2.0",
        "stage": "seed9_target",
        "contract": CAPUTO_CONTRACT["final_target"],
        "seed": selected_seed.tolist(),
        "recorded_seed": EXPECTED_FINAL_SEED.tolist(),
        "seed_max_abs_error": selected_seed_error,
        "status": next(
            row["status"]
            for row in long_rows
            if int(row["i"]) == 9
            and np.isclose(float(row["h"]), 0.005)
        ),
        "end_time": float(target_times[-1]),
        "max_norm": float(
            np.max(np.linalg.norm(target_states, axis=1))
        ),
        "range": np.ptp(target_tail, axis=0).tolist(),
        "trajectory": target_path.name,
        "reused_from": str(selected_source.relative_to(stage_dir)),
        "recorded_regression_matched": (
            selected_seed_error <= 5.0e-12
        ),
    }
    _write_json(stage_dir / "target_summary.json", target_payload)

    overall_match = all(
        (
            seed_payload["recorded_regression_matched"],
            h_audit_payload["recorded_regression_matched"],
            cross_payload["recorded_regression_matched"],
            seed5_payload["recorded_regression_matched"],
            long_payload["recorded_regression_matched"],
            target_payload["recorded_regression_matched"],
        )
    )
    payload = {
        "schema_version": "2.0",
        "stage": "caputo-seed9",
        "status": (
            "completed_recorded_regression_matched"
            if overall_match
            else "completed_recorded_regression_mismatch"
        ),
        "contract": CAPUTO_CONTRACT,
        "parameters": parameters,
        "integer_seed": integer_seed.tolist(),
        "q_scan": q_scan_payload,
        "resampled_zero_one": resampled_payload,
        "integer_seed_h_audit": h_audit_payload,
        "seed_refinement": {
            "bounded_indices": bounded_indices,
            "recorded_regression_matched": seed_payload[
                "recorded_regression_matched"
            ],
        },
        "cross_step": {
            "passing_all_h": cross_passing,
            "recorded_regression_matched": cross_payload[
                "recorded_regression_matched"
            ],
        },
        "seed5_long_audit": {
            "robust_bounded_and_zero_one": seed5_robust,
            "recorded_regression_matched": seed5_payload[
                "recorded_regression_matched"
            ],
        },
        "survivor_long_audit": {
            "bounded_at_all_three_h": long_bounded,
            "passing": long_passing,
            "recorded_regression_matched": long_payload[
                "recorded_regression_matched"
            ],
        },
        "target": target_payload,
        "recorded_regression_matched": overall_match,
        "scientific_boundary": (
            "This route reproduces the finite-time full-memory Caputo "
            "selection of seed9 and its tested step robustness. The separate "
            "equilibrium-neighborhood probe supplies the hiddenness evidence."
        ),
    }
    _write_json(stage_dir / "summary.json", payload)
    return payload


def execute_stage(
    stage: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    local_screen: Path | None = None,
) -> dict[str, Any]:
    """Execute one recovered stage or the complete executable integer route."""

    if stage == "banks":
        return execute_banks(output_dir)
    if stage == "integer-global":
        return execute_integer_global(output_dir)
    if stage == "integer-local":
        return execute_integer_local(output_dir)
    if stage == "variational-shortlist":
        return execute_variational_shortlist(
            output_dir,
            local_screen=local_screen,
        )
    if stage == "caputo-seed9":
        return execute_caputo_seed9(output_dir)
    if stage == "integer-route":
        results = [
            execute_banks(output_dir),
            execute_integer_global(output_dir),
            execute_integer_local(output_dir),
            execute_variational_shortlist(
                output_dir,
                local_screen=local_screen,
            ),
        ]
        status, recorded_match = _aggregate_stage_regression(results)
        return {
            "stage": "integer-route",
            "status": status,
            "recorded_regression_matched": recorded_match,
            "results": results,
        }
    if stage == "all":
        results = [
            execute_banks(output_dir),
            execute_integer_global(output_dir),
            execute_integer_local(output_dir),
            execute_variational_shortlist(
                output_dir,
                local_screen=local_screen,
            ),
            execute_caputo_seed9(output_dir),
        ]
        status, recorded_match = _aggregate_stage_regression(results)
        return {
            "stage": "all",
            "status": status,
            "recorded_regression_matched": recorded_match,
            "results": results,
        }
    raise ValueError(f"unknown stage: {stage}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=STAGE_CHOICES,
        default="all",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the verified plan and perform no writes (default).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Execute the requested recovered stage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--local-screen",
        type=Path,
        default=None,
        help=(
            "Optional integer-local screen.json for the variational stage; "
            "defaults to <output-dir>/integer_local/screen.json."
        ),
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if not args.execute:
        print(
            json.dumps(
                build_plan(args.stage),
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )
        )
        return
    try:
        result = execute_stage(
            args.stage,
            args.output_dir,
            local_screen=args.local_screen,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        MissingDiscoveryContract,
        RuntimeError,
        ValueError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
