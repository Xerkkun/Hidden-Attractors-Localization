#!/usr/bin/env python3
"""Build traceable representative probe trajectories for Paper 07 figures.

The validated probe tables retain initial conditions and classifications but not
the complete state histories.  This script deterministically selects nine probes
from each Paper 07 case, reintegrates them under the original numerical
contracts, verifies every stored classification and metric, and writes the full
trajectories together with a reproducibility manifest.

The arctangent case is truncated after the first completed radius with target
contacts (r=1).  The previously executed r=2 audit remains untouched and is not
part of the declared Paper 07 experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve()
VERSION2 = HERE.parents[1]
REPOSITORY = VERSION2.parent
if str(VERSION2) not in sys.path:
    sys.path.insert(0, str(VERSION2))

from hidden_attractors.integrations.fractional_c import (  # noqa: E402
    fractional_integrate,
)
from hidden_attractors.integrations.general import integrate_general  # noqa: E402
from hidden_attractors.systems import get_system  # noqa: E402


DEFAULT_OUTPUT = (
    VERSION2
    / "validation"
    / "paper07_chua"
    / "evidence"
    / "probe_story_trajectories"
)

NONSMOOTH_PROBE_IDS = (
    0,
    8800,
    8801,
    12350,
    13407,
    17600,
    17602,
    21163,
    22330,
)

ARCTAN_SELECTION = (
    ("hiddenness_r100_rows.csv", "E0|r00|d194"),
    ("hiddenness_r100_rows.csv", "E0|r00|d228"),
    ("hiddenness_scaled_rows.csv", "E+|r00|d000"),
    ("hiddenness_scaled_rows.csv", "E+|r00|d001"),
    ("hiddenness_scaled_rows.csv", "E-|r00|d000"),
    ("hiddenness_scaled_rows.csv", "E-|r00|d001"),
    ("hiddenness_r100_rows.csv", "E0|r00|d000"),
    ("hiddenness_r100_rows.csv", "E+|r00|d000"),
    ("hiddenness_r100_rows.csv", "E-|r00|d000"),
)

ARCTAN_INCLUDED_FILES = (
    "hiddenness_scaled_rows.csv",
    "hiddenness_r003_rows.csv",
    "hiddenness_r010_rows.csv",
    "hiddenness_r030_rows.csv",
    "hiddenness_r100_rows.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for trajectory CSV files and manifest.json.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY.resolve()).as_posix()


def evenly_spaced(states: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum).astype(int)
    return values[indices]


def fixed_random_sample(states: np.ndarray, maximum: int = 2_000) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if len(values) <= maximum:
        return values
    indices = np.random.default_rng(0).choice(
        len(values),
        maximum,
        replace=False,
    )
    return values[indices]


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def assert_text(name: str, actual: Any, expected: Any) -> None:
    if str(actual) != str(expected):
        raise RuntimeError(f"{name}: reproduced={actual!r}, stored={expected!r}")


def assert_scalar(
    name: str,
    actual: float,
    expected: Any,
    *,
    atol: float = 1.0e-11,
) -> None:
    expected_value = float(expected)
    if np.isnan(expected_value):
        if not np.isnan(float(actual)):
            raise RuntimeError(f"{name}: expected NaN, reproduced={actual!r}")
        return
    if not np.isclose(
        float(actual),
        expected_value,
        rtol=1.0e-9,
        atol=atol,
    ):
        raise RuntimeError(
            f"{name}: reproduced={float(actual):.17g}, "
            f"stored={expected_value:.17g}"
        )


def save_trajectory(
    output_dir: Path,
    stem: str,
    times: np.ndarray,
    states: np.ndarray,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.csv"
    pd.DataFrame(
        np.column_stack((times, states)),
        columns=["t", "x", "y", "z"],
    ).to_csv(path, index=False)
    return path


def output_record(path: Path) -> dict[str, Any]:
    return {
        "path": repository_path(path),
        "rows": int(sum(1 for _ in path.open("r", encoding="utf-8")) - 1),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def nonsmooth_story(output_root: Path) -> dict[str, Any]:
    evidence_dir = (
        VERSION2
        / "validation"
        / "paper07_chua"
        / "evidence"
        / "nonsmooth_corrected"
        / "extended_first_contact_clean"
    )
    contract_path = evidence_dir / "extended_numerical_contract.json"
    rows_path = evidence_dir / "extended_probe_runs.csv"
    target_path = evidence_dir / "target_cloud_nn_sample.csv"
    contract = read_json(contract_path)
    rows = pd.read_csv(rows_path)
    if rows["probe_id"].duplicated().any():
        raise RuntimeError("The non-smooth probe table has duplicated probe_id values.")
    selected = rows.set_index("probe_id").loc[list(NONSMOOTH_PROBE_IDS)].copy()

    parameters = {
        key: float(value)
        for key, value in contract["system"]["parameters"].items()
    }
    base_system = get_system("chua-nonsmooth")
    system = replace(
        base_system,
        parameters={**dict(base_system.parameters), **parameters},
    )
    equilibrium_names = ("E0", "E+", "E-")
    equilibria = [
        np.asarray(contract["system"]["equilibria"][name], dtype=float)
        for name in equilibrium_names
    ]
    target_cloud = pd.read_csv(target_path)[["x", "y", "z"]].to_numpy(dtype=float)
    target_tree = cKDTree(target_cloud)
    numerical = contract["integrator"]
    classification = contract["classification"]
    early_stop = dict(numerical["early_stopping"])
    q = float(numerical["q"])
    h = float(numerical["h"])
    t_final = float(numerical["t_final"])
    t_burn = float(numerical["t_burn"])
    eq_tol = float(classification["equilibrium_tol"])
    match_tol = float(classification["target_match_tol"])
    match_percentile = float(classification["target_match_percentile"])

    case_dir = output_root / "nonsmooth"
    entries: list[dict[str, Any]] = []
    for row in selected.itertuples():
        initial = np.asarray([row.x0, row.y0, row.z0], dtype=float)
        times, states, status, info = fractional_integrate(
            rhs=lambda _time, state: system.rhs(state, system.parameters),
            x0=initial,
            q=q,
            h=h,
            t_final=t_final,
            method="abm",
            memory_mode="full",
            system=system,
            use_c_backend=True,
            divergence_norm=float(numerical["hard_divergence_norm"]),
            allow_python_fallback=False,
            early_stop_config=early_stop,
            equilibria=equilibria,
        )
        if not len(states):
            raise RuntimeError(f"Empty non-smooth trajectory for probe {row.Index}.")

        final = np.asarray(states[-1], dtype=float)
        eq_distances = [float(np.linalg.norm(final - state)) for state in equilibria]
        score = float("nan")
        if status in {"diverged", "diverged_early", "nonfinite_solution"}:
            destination = "divergence"
        elif status == "converged_equilibrium_early":
            destination = "stable_equilibrium"
        elif status != "ok":
            destination = "numerical_failure"
        elif min(eq_distances) <= eq_tol:
            destination = "stable_equilibrium"
        else:
            burn_index = int(np.ceil(t_burn / h))
            tail = (
                states[burn_index:]
                if len(states) > burn_index
                else states[int(len(states) / 2) :]
            )
            distances, _ = target_tree.query(fixed_random_sample(tail), k=1, workers=1)
            score = float(np.percentile(distances, match_percentile))
            destination = (
                "target_attractor" if score <= match_tol else "other_attractor"
            )

        assert_text("non-smooth status", status, row.status)
        assert_text("non-smooth destination", destination, row.destination)
        assert_scalar("non-smooth end time", times[-1], row.end_time)
        assert_scalar(
            "non-smooth maximum norm",
            np.linalg.norm(states, axis=1).max(),
            row.max_norm,
        )
        assert_scalar("non-smooth final x", final[0], row.final_x)
        assert_scalar("non-smooth final y", final[1], row.final_y)
        assert_scalar("non-smooth final z", final[2], row.final_z)
        assert_scalar(
            "non-smooth equilibrium distance",
            min(eq_distances),
            row.min_final_equilibrium_distance,
        )
        assert_scalar("non-smooth NN score", score, row.nn_percentile_score)
        if int(len(states)) != int(row.steps):
            raise RuntimeError(
                f"non-smooth steps: reproduced={len(states)}, stored={row.steps}"
            )
        if not bool(info.get("used_c_backend", False)):
            raise RuntimeError(f"Probe {row.Index} did not use the native backend.")

        stem = f"probe_{int(row.Index):05d}_{str(row.equilibrium).replace('+', 'p').replace('-', 'm')}"
        trajectory_path = save_trajectory(case_dir, stem, times, states)
        entries.append(
            {
                "source_table": repository_path(rows_path),
                "probe_id": int(row.Index),
                "equilibrium": str(row.equilibrium),
                "radius": float(row.radius),
                "initial_state": initial.tolist(),
                "status": str(status),
                "category": {
                    "target_attractor": "TARGET",
                    "stable_equilibrium": "EQUILIBRIUM",
                    "divergence": "DIVERGENCE",
                    "other_attractor": "OTHER",
                }[destination],
                "destination": destination,
                "trajectory": output_record(trajectory_path),
            }
        )

    return {
        "selection_rule": (
            "Sort by probe_id and retain the first row of every observed "
            "(destination, equilibrium) stratum."
        ),
        "selected_probe_ids": list(NONSMOOTH_PROBE_IDS),
        "selected_probe_id_sha256": hashlib.sha256(
            (",".join(map(str, NONSMOOTH_PROBE_IDS)) + "\n").encode("utf-8")
        ).hexdigest(),
        "source_contract": {
            "path": repository_path(contract_path),
            "sha256": sha256_file(contract_path),
        },
        "source_rows": {
            "path": repository_path(rows_path),
            "sha256": sha256_file(rows_path),
        },
        "target_cloud": {
            "path": repository_path(target_path),
            "sha256": sha256_file(target_path),
        },
        "numerical_contract": {
            "operator": "Caputo",
            "integrator": "ABM-PECE",
            "memory_mode": "full",
            "q": q,
            "h": h,
            "t_final": t_final,
            "t_burn": t_burn,
            "native_backend_required": True,
            "early_stopping": early_stop,
        },
        "entries": entries,
    }


def arctan_story(output_root: Path) -> dict[str, Any]:
    probe_dir = (
        VERSION2 / "validation" / "chua_fractional_arctan_c590"
    )
    target_path = (
        VERSION2
        / "validation"
        / "paper07_chua"
        / "evidence"
        / "c590_reconstruction"
        / "caputo_seed9"
        / "target.npz"
    )
    summary_path = (
        VERSION2
        / "outputs"
        / "arctan_hidden_candidate_search"
        / "c590_q09999_seed9_candidate_20260623"
        / "candidate_summary_r100.json"
    )
    probe_run_config_path = probe_dir / "hiddenness_r100_run_config.json"
    source_summary = read_json(summary_path)
    probe_run_config = read_json(probe_run_config_path)
    parameters = {
        key: float(source_summary["parameters"][key])
        for key in ("alpha", "beta", "gamma", "a1", "a2", "rho")
    }
    q = float(source_summary["q"])
    numerical = source_summary["numerical_contract"]
    h = float(probe_run_config["h"])
    t_final = float(probe_run_config["t_final"])
    t_burn = float(probe_run_config["t_burn"])
    target_burn = float(numerical["target_t_burn"])
    equilibrium_names = ("E0", "E+", "E-")
    equilibria_dict = source_summary["hiddenness_evidence"]["equilibria"]
    equilibria = [
        np.asarray(equilibria_dict[name], dtype=float) for name in equilibrium_names
    ]
    contact_threshold = float(
        source_summary["hiddenness_evidence"]["contact_threshold"]
    )

    with np.load(target_path, allow_pickle=False) as archive:
        target_times = np.asarray(archive["times"], dtype=float)
        target_states = np.asarray(archive["states"], dtype=float)
    target_cloud = evenly_spaced(
        target_states[target_times >= target_burn],
        6_000,
    )
    target_tree = cKDTree(target_cloud)
    symmetric_tree = cKDTree(-target_cloud)

    base_system = get_system("chua-arctan")
    merged = {
        **dict(base_system.parameters),
        **parameters,
        "model": "arctan",
        "q": q,
        "system_id": "chua_fractional_arctan",
    }
    system = replace(base_system, parameters=merged)
    early_stop = {
        "enabled": True,
        "divergence_enabled": False,
        "equilibrium_enabled": True,
        "equilibrium_min_time": 0.5 * t_burn,
        "equilibrium_tol": 1.0e-6,
        "equilibrium_derivative_tol": 1.0e-4,
        "equilibrium_consecutive_steps": 50,
    }

    included_paths = [probe_dir / name for name in ARCTAN_INCLUDED_FILES]
    included_frames = []
    for path in included_paths:
        frame = pd.read_csv(path)
        frame["_source_file"] = path.name
        included_frames.append(frame)
    full_rows = pd.concat(included_frames, ignore_index=True)
    radii = sorted(float(value) for value in full_rows["radius"].unique())
    contact_radii = sorted(
        float(value)
        for value in full_rows.loc[
            full_rows["contact"].map(as_bool), "radius"
        ].unique()
    )
    if contact_radii != [1.0] or radii[-1] != 1.0:
        raise RuntimeError(
            f"Unexpected arctangent first-contact cutoff: radii={radii}, "
            f"contact_radii={contact_radii}"
        )

    case_dir = output_root / "arctan"
    entries: list[dict[str, Any]] = []
    for source_name, probe_id in ARCTAN_SELECTION:
        source_path = probe_dir / source_name
        source = pd.read_csv(source_path)
        matches = source[source["probe_id"].eq(probe_id)]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one row for {(source_name, probe_id)}, found {len(matches)}."
            )
        row = matches.iloc[0]
        initial = np.asarray(json.loads(row["initial_state"]), dtype=float)
        times, states, status = integrate_general(
            lambda _time, state: system.evaluate(state),
            initial,
            q=q,
            h=h,
            t_final=t_final,
            integrator="abm",
            memory_mode="full",
            memory_window_length=None,
            system=system,
            use_c_backend=True,
            divergence_norm=120.0,
            early_stop_config=early_stop,
            equilibria=equilibria,
        )
        finite = bool(len(states) and np.all(np.isfinite(states)))
        tail = states[times >= t_burn] if finite else np.empty((0, 3))
        tail = evenly_spaced(tail, 6_000) if len(tail) else tail
        if len(tail):
            target_nn90 = float(
                np.percentile(target_tree.query(tail, k=1)[0], 90)
            )
            symmetric_nn90 = float(
                np.percentile(symmetric_tree.query(tail, k=1)[0], 90)
            )
            minimum_nn90 = min(target_nn90, symmetric_nn90)
        else:
            target_nn90 = float("nan")
            symmetric_nn90 = float("nan")
            minimum_nn90 = float("nan")
        contact = bool(
            status == "ok"
            and np.isfinite(minimum_nn90)
            and minimum_nn90 <= contact_threshold
        )
        if contact:
            outcome = "TARGET"
        elif status == "converged_equilibrium_early":
            outcome = "EQUILIBRIUM"
        elif status in {"diverged", "diverged_early"}:
            outcome = "DIVERGED"
        elif status == "ok":
            outcome = "OTHER"
        else:
            outcome = "NUMERICAL_FAILURE"

        assert_text("arctangent status", status, row["status"])
        assert_text("arctangent outcome", outcome, row["outcome"])
        if contact != as_bool(row["contact"]):
            raise RuntimeError(
                f"arctangent contact: reproduced={contact}, stored={row['contact']}"
            )
        if finite != as_bool(row["finite"]):
            raise RuntimeError(
                f"arctangent finite: reproduced={finite}, stored={row['finite']}"
            )
        assert_scalar("arctangent end time", times[-1], row["end_time"])
        assert_scalar(
            "arctangent maximum norm",
            np.linalg.norm(states, axis=1).max(),
            row["max_norm"],
        )
        stored_range = np.asarray(json.loads(row["range"]), dtype=float)
        reproduced_range = np.ptp(tail, axis=0) if len(tail) else np.zeros(3)
        for coordinate, actual, expected in zip(
            "xyz", reproduced_range, stored_range
        ):
            assert_scalar(f"arctangent range {coordinate}", actual, expected)
        assert_scalar("arctangent target NN90", target_nn90, row["target_nn90"])
        assert_scalar(
            "arctangent symmetric NN90",
            symmetric_nn90,
            row["symmetric_nn90"],
        )
        assert_scalar(
            "arctangent minimum NN90",
            minimum_nn90,
            row["minimum_nn90"],
        )

        safe_probe = (
            probe_id.replace("|", "_")
            .replace("+", "p")
            .replace("-", "m")
        )
        stem = f"{Path(source_name).stem}_{safe_probe}"
        trajectory_path = save_trajectory(case_dir, stem, times, states)
        entries.append(
            {
                "source_table": repository_path(source_path),
                "probe_id": probe_id,
                "equilibrium": str(row["equilibrium"]),
                "radius": float(row["radius"]),
                "initial_state": initial.tolist(),
                "status": str(status),
                "category": outcome,
                "contact": contact,
                "trajectory": output_record(trajectory_path),
            }
        )

    outcome_counts = {
        str(key): int(value)
        for key, value in full_rows["outcome"].value_counts().sort_index().items()
    }
    return {
        "selection_rule": (
            "At r<=1 retain the first two TARGET rows by direction_index, all "
            "four EQUILIBRIUM rows, and the first r=1 OTHER row from each "
            "equilibrium. Source filename is part of the identity because "
            "probe_id restarts in each independent radius run."
        ),
        "first_contact_cutoff": {
            "rule": (
                "Process radii in ascending order; after the first radius with "
                "TARGET contacts, finish that radius and exclude larger radii."
            ),
            "first_contact_radius": 1.0,
            "included_radii": radii,
            "included_tests": int(len(full_rows)),
            "included_outcome_counts": outcome_counts,
            "included_source_tables": [
                {
                    "path": repository_path(path),
                    "sha256": sha256_file(path),
                }
                for path in included_paths
            ],
            "post_cutoff_source_excluded": repository_path(
                probe_dir / "hiddenness_r200_rows.csv"
            ),
        },
        "source_summary": {
            "path": repository_path(summary_path),
            "sha256": sha256_file(summary_path),
        },
        "target_trajectory": {
            "path": repository_path(target_path),
            "sha256": sha256_file(target_path),
            "classification_burn": target_burn,
            "classification_points": int(len(target_cloud)),
        },
        "probe_run_config": {
            "path": repository_path(probe_run_config_path),
            "sha256": sha256_file(probe_run_config_path),
        },
        "numerical_contract": {
            "operator": "Caputo",
            "integrator": "ABM predictor-corrector",
            "memory_mode": "full",
            "q": q,
            "h": h,
            "t_final": t_final,
            "t_burn": t_burn,
            "divergence_norm": 120.0,
            "early_stopping": early_stop,
        },
        "entries": entries,
    }


def iter_output_paths(manifest: dict[str, Any]) -> Iterable[Path]:
    for case in ("nonsmooth", "arctan"):
        for entry in manifest[case]["entries"]:
            yield REPOSITORY / entry["trajectory"]["path"]


def main() -> int:
    args = parse_args()
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "generated_by": repository_path(HERE),
        "purpose": (
            "Full representative state histories for the Paper 07 spatial "
            "probe overview. Every trajectory is reintegrated and checked "
            "against its stored classification and metrics."
        ),
        "nonsmooth": nonsmooth_story(output_root),
        "arctan": arctan_story(output_root),
    }
    manifest["outputs_sha256"] = {
        repository_path(path): sha256_file(path)
        for path in iter_output_paths(manifest)
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")
    print(
        json.dumps(
            {
                "nonsmooth_entries": len(manifest["nonsmooth"]["entries"]),
                "arctan_entries": len(manifest["arctan"]["entries"]),
                "arctan_cutoff": manifest["arctan"]["first_contact_cutoff"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
