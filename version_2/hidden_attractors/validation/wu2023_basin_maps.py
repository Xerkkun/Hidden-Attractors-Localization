"""Traceable finite-destination slices for Wu's declared local ADM contract.

The map dispatches the actual method selected by its JSON contract.  Local
ADM does not accumulate a Caputo history.  A bounded response and the sign of
its terminal mean do not establish target identity, chaos, or hiddenness.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .._time_grid import exact_fixed_step_count
from ..integrations.adm_wu2023 import adm_wu2023_integrate
from ..models.chua import ChuaParameters, equilibria_arctan, rhs_arctan
from ..verification.terminal_policy import completion_tail, terminal_equilibrium


SCHEMA_VERSION = "wu_finite_maps_v1"
CLASS_LABELS = {
    0: "equilibrium_terminal",
    1: "bounded_positive_mean",
    2: "bounded_negative_mean",
    3: "threshold_crossed",
    4: "inconclusive",
    5: "numerical_failure",
    6: "bounded_near_zero_mean",
}
PLANE_COMPONENTS = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError(f"{name} must be finite" + (" and positive." if positive else "."))
    return number


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def validate_map_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete experiment before an output folder is claimed.

    Numerical values are never inferred from an unrelated default model.
    ``basin_map`` supplies map-only settings; the original numerical horizon,
    order, step, ADM order and memory policy remain authoritative.
    """
    if not isinstance(config, Mapping):
        raise ValueError("The config must be a JSON object.")
    if config.get("system") != "fractional_chua_arctan_wu2023" or config.get("model") != "arctan":
        raise ValueError("Expected fractional_chua_arctan_wu2023 with model arctan.")
    raw_params = config.get("parameters")
    raw_numerical = config.get("numerical_contract")
    raw_map = config.get("basin_map", {})
    if not isinstance(raw_params, Mapping) or not isinstance(raw_numerical, Mapping) or not isinstance(raw_map, Mapping):
        raise ValueError("parameters, numerical_contract and basin_map must be objects.")
    required = ("alpha", "beta", "gamma", "rho")
    params = {key: _finite_number(raw_params.get(key), f"parameters.{key}") for key in required}
    a1 = raw_params.get("a1", raw_params.get("m"))
    a2 = raw_params.get("a2")
    if a2 is None and raw_params.get("n") is not None and a1 is not None:
        a2 = _finite_number(raw_params["n"], "parameters.n") - _finite_number(a1, "parameters.a1")
    params.update(a1=_finite_number(a1, "parameters.a1/m"), a2=_finite_number(a2, "parameters.a2/(n-m)"))
    for alias, expected in (("m", params["a1"]), ("n", params["a1"] + params["a2"])):
        if alias in raw_params and not math.isclose(_finite_number(raw_params[alias], f"parameters.{alias}"), expected, rel_tol=1e-14, abs_tol=1e-14):
            raise ValueError(f"parameters.{alias} conflicts with a1/a2; no silent alias conversion.")
    ChuaParameters(model="arctan", **params)
    numerical = dict(raw_numerical)
    for key in ("q", "h", "t_final", "divergence_norm"):
        numerical[key] = _finite_number(raw_numerical.get(key), f"numerical_contract.{key}", positive=True)
    if numerical["q"] > 1.0:
        raise ValueError("numerical_contract.q must satisfy 0 < q <= 1.")
    numerical["N"] = _integer(raw_numerical.get("N"), "numerical_contract.N")
    if exact_fixed_step_count(numerical["h"], numerical["t_final"], caller=SCHEMA_VERSION) != numerical["N"]:
        raise ValueError("numerical_contract.N disagrees with h and t_final.")
    method = raw_numerical.get("integrator")
    if method == "adm_wu2023":
        if (raw_numerical.get("adm_order") != 4 or isinstance(raw_numerical.get("adm_order"), bool)
                or raw_numerical.get("backend") != "adm_local_reproduction"
                or raw_numerical.get("memory_policy") != "none_local_adm"
                or raw_numerical.get("memory_length") is not None
                or raw_numerical.get("caputo_history_accumulated") is not False):
            raise ValueError("ADM requires order=4, backend=adm_local_reproduction, memory_policy=none_local_adm, memory_length=null and caputo_history_accumulated=false.")
    elif method in {"efork", "efork3"}:
        if raw_numerical.get("backend") not in {"python", "caputo_diagnostic"}:
            raise ValueError("The explicit EFORK diagnostic requires backend=python or caputo_diagnostic.")
        if raw_numerical.get("memory_policy") != "full" or raw_numerical.get("caputo_history_accumulated") is not True:
            raise ValueError("The explicit EFORK diagnostic requires memory_policy=full and caputo_history_accumulated=true.")
        length = _finite_number(raw_numerical.get("memory_length"), "numerical_contract.memory_length", positive=True)
        if length < numerical["t_final"]:
            raise ValueError("This EFORK diagnostic requires memory_length >= t_final (full history).")
        numerical["integrator"] = "efork3"
    else:
        raise ValueError("integrator must explicitly select adm_wu2023 or efork3.")
    points = _integer(raw_map.get("points", 31), "basin_map.points", minimum=2)
    limit = _finite_number(raw_map.get("limit", 1.2), "basin_map.limit", positive=True)
    fixed = _finite_number(raw_map.get("fixed_coordinate", 0.0), "basin_map.fixed_coordinate")
    planes = raw_map.get("planes", ["xy"])
    if isinstance(planes, str) or not isinstance(planes, Sequence) or not planes or any(p not in PLANE_COMPONENTS for p in planes) or len(set(planes)) != len(planes):
        raise ValueError("basin_map.planes must be a nonempty unique list drawn from xy, xz, yz.")
    burn = _finite_number(raw_map.get("t_burn", numerical["t_final"] / 2.0), "basin_map.t_burn")
    if burn < 0.0 or burn >= numerical["t_final"]:
        raise ValueError("basin_map.t_burn must satisfy 0 <= t_burn < t_final.")
    settings = {
        "points": points, "limit": limit, "fixed_coordinate": fixed, "planes": list(planes), "t_burn": burn,
        "equilibrium_tolerance": _finite_number(raw_map.get("equilibrium_tolerance", 0.03), "basin_map.equilibrium_tolerance", positive=True),
        "terminal_window_points": _integer(raw_map.get("terminal_window_points", 90), "basin_map.terminal_window_points", minimum=3),
        "equilibrium_span_tolerance": _finite_number(raw_map.get("equilibrium_span_tolerance", 0.06), "basin_map.equilibrium_span_tolerance", positive=True),
        "mean_zero_tolerance": _finite_number(raw_map.get("mean_zero_tolerance", 1e-12), "basin_map.mean_zero_tolerance", positive=True),
    }
    post_burn = numerical["N"] - int(math.ceil(burn / numerical["h"])) + 1
    if post_burn < settings["terminal_window_points"]:
        raise ValueError("The declared terminal window needs more post-burn samples; extend the horizon or explicitly reduce terminal_window_points.")
    return {"parameters": params, "numerical_contract": numerical, "basin_map": settings}


def integrate_map_point(validated: Mapping[str, Any], initial: np.ndarray):
    """Dispatch ADM directly; an explicit EFORK diagnostic uses its own path."""
    numerical = validated["numerical_contract"]
    params = validated["parameters"]
    if numerical["integrator"] == "adm_wu2023":
        return adm_wu2023_integrate(
            params, initial, q=numerical["q"], h=numerical["h"], N=numerical["N"],
            divergence_norm=numerical["divergence_norm"],
        )
    from ..integrations.selector import integrate
    model_params = ChuaParameters(model="arctan", **params)
    times, states, status = integrate(
        lambda state: rhs_arctan(state, model_params), initial,
        q=numerical["q"], h=numerical["h"], t_final=numerical["t_final"],
        integrator="efork3", memory_mode="full", divergence_norm=numerical["divergence_norm"],
        use_c_backend=False, early_stop_config={"enabled": False},
    )
    return times, states, status, {"integrator": "efork3", "backend": "python", "caputo_history_accumulated": True}


def classify_map_trajectory(times, states, status: str, validated, equilibria) -> dict[str, Any]:
    """Use a complete terminal window; early visits never establish an EQ."""
    time_array = np.asarray(times, dtype=float)
    state_array = np.asarray(states, dtype=float)
    numerical, settings = validated["numerical_contract"], validated["basin_map"]
    result = {"equilibrium_name": None, "is_reference_target": False, "hidden_verified": False,
              "classification_reason": None}
    if (time_array.ndim != 1 or state_array.ndim != 2 or state_array.shape != (len(time_array), 3) or len(time_array) == 0
            or not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(state_array))
            or status == "nonfinite" or status.startswith("solver_exception")):
        return {**result, "class_id": 5, "class_label": CLASS_LABELS[5]}
    norms = np.linalg.norm(state_array, axis=1)
    if np.any(norms > numerical["divergence_norm"]):
        return {**result, "class_id": 3, "class_label": CLASS_LABELS[3],
                "classification_reason": "observed_norm_threshold_crossing"}
    if status in {"diverged", "diverged_early"}:
        return {**result, "class_id": 4, "class_label": CLASS_LABELS[4],
                "classification_reason": "reported_divergence_without_observed_crossing"}
    count = settings["terminal_window_points"]
    if status != "ok":
        return {**result, "class_id": 4, "class_label": CLASS_LABELS[4]}
    try:
        _, eligible = completion_tail(time_array, state_array, numerical["t_final"], settings["t_burn"], min_samples=count)
    except ValueError:
        return {**result, "class_id": 4, "class_label": CLASS_LABELS[4]}
    terminal = eligible[-count:]
    eq_names = list(equilibria)
    eq_index, _ = terminal_equilibrium(terminal, [equilibria[name] for name in eq_names],
                                      settings["equilibrium_tolerance"], settings["equilibrium_span_tolerance"])
    # The map deliberately requires every terminal point inside the same EQ
    # ball in addition to the shared final/q95/span controls.
    if eq_index is not None:
        name = eq_names[eq_index]
        if np.all(np.linalg.norm(terminal - equilibria[name], axis=1) <= settings["equilibrium_tolerance"]):
            return {**result, "class_id": 0, "class_label": CLASS_LABELS[0], "equilibrium_name": name}
    mean_x = float(np.mean(terminal[:, 0]))
    code = 1 if mean_x > settings["mean_zero_tolerance"] else 2 if mean_x < -settings["mean_zero_tolerance"] else 6
    return {**result, "class_id": code, "class_label": CLASS_LABELS[code]}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _implementation_paths(validated) -> list[Path]:
    """List method-specific Python sources used by the declared route."""
    root = Path(__file__).resolve().parents[2]
    relatives = [
        "hidden_attractors/validation/wu2023_basin_maps.py",
        "hidden_attractors/verification/terminal_policy.py",
        "hidden_attractors/models/chua.py", "hidden_attractors/_run_output.py",
        "hidden_attractors/_time_grid.py",
    ]
    if validated["numerical_contract"]["integrator"] == "adm_wu2023":
        relatives.append("hidden_attractors/integrations/adm_wu2023.py")
    else:
        relatives.extend([
            "hidden_attractors/_rhs.py", "hidden_attractors/integrations/selector.py",
            "hidden_attractors/integrations/general.py", "hidden_attractors/integrations/efork.py",
            "hidden_attractors/integrations/_history.py",
        ])
        if validated["numerical_contract"]["q"] == 1.0:
            relatives.append("hidden_attractors/integrations/_q1_coefficients.py")
    return [root / relative for relative in relatives]


def _snapshot_sources(output: Path, validated) -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[2]
    records = []
    for source in _implementation_paths(validated):
        raw = source.read_bytes()
        snapshot = output / "source_snapshot" / source.relative_to(root)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open("xb") as stream:
            stream.write(raw)
        records.append({"path": str(source), "relative_path": source.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "snapshot_path": snapshot.relative_to(output).as_posix()})
    return records


def run_basin_maps(config_path: Path, output_dir: Path, *, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run slices, saving every trajectory and a hash-linked map manifest.

    Invalid contracts produce no output. A nonempty destination is rejected
    before integration and writing; interrupted folders are preserved.
    """
    config_path = Path(config_path).resolve()
    raw = config_path.read_bytes()
    config = json.loads(raw)
    if overrides:
        if not isinstance(config, dict):
            raise ValueError("The config must be a JSON object.")
        config.setdefault("basin_map", {}).update(dict(overrides))
    validated = validate_map_contract(config)
    from .._run_output import RUN_MARKER, claim_output_directory, complete_output_claim
    claim = claim_output_directory(Path(output_dir), run_kind=SCHEMA_VERSION)
    output = claim.path
    _write_json(output / "input_config.json", json.loads(raw))
    _write_json(output / "effective_contract.json", validated)
    sources = _snapshot_sources(output, validated)
    trajectory_dir = output / "trajectories"
    trajectory_dir.mkdir()
    params = ChuaParameters(model="arctan", **validated["parameters"])
    equilibria = equilibria_arctan(params)
    settings = validated["basin_map"]
    axis = np.linspace(-settings["limit"], settings["limit"], settings["points"])
    all_rows = []
    for plane in settings["planes"]:
        horizontal, vertical, fixed = PLANE_COMPONENTS[plane]
        labels = np.empty((len(axis), len(axis)), dtype=np.int16)
        rows = []
        for row_index, vertical_value in enumerate(axis):
            for column_index, horizontal_value in enumerate(axis):
                initial = np.full(3, settings["fixed_coordinate"], dtype=float)
                initial[horizontal], initial[vertical] = horizontal_value, vertical_value
                point_id = f"{plane}_{row_index:04d}_{column_index:04d}"
                times, states, status, info = integrate_map_point(validated, initial)
                classification = classify_map_trajectory(times, states, status, validated, equilibria)
                labels[row_index, column_index] = classification["class_id"]
                data_path = trajectory_dir / f"{point_id}.npz"
                with data_path.open("xb") as stream:
                    np.savez_compressed(stream, times=times, states=states, initial=initial)
                info_path = trajectory_dir / f"{point_id}.json"
                _write_json(info_path, {"point_id": point_id, "status": status, "classification": classification, "integrator_info": info,
                                       "trajectory_sha256": _sha256(data_path), "effective_contract_sha256": _sha256(output / "effective_contract.json")})
                row = {"point_id": point_id, "plane": plane, "row": row_index, "column": column_index,
                       "x0": float(initial[0]), "y0": float(initial[1]), "z0": float(initial[2]),
                       **classification, "integration_status": status, "samples": len(times), "time_reached": float(times[-1]),
                       "trajectory_file": data_path.relative_to(output).as_posix(), "trajectory_sha256": _sha256(data_path),
                       "metadata_file": info_path.relative_to(output).as_posix(), "metadata_sha256": _sha256(info_path)}
                rows.append(row)
                all_rows.append(row)
        with (output / f"map_{plane}.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (output / f"map_{plane}.npz").open("xb") as stream:
            np.savez_compressed(stream, axis=axis, class_ids=labels, initial_conditions=np.array([[r["x0"], r["y0"], r["z0"]] for r in rows]).reshape(len(axis), len(axis), 3))
        _render_map(output / f"map_{plane}.png", labels, axis, plane, fixed, validated)
    files = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": _sha256(p)}
             for p in sorted(output.rglob("*")) if p.is_file() and p.name != RUN_MARKER]
    changed_sources = [entry["relative_path"] for entry in sources if _sha256(Path(entry["path"])) != entry["sha256"]]
    if changed_sources:
        raise RuntimeError(f"Source files changed during the run: {changed_sources}; partial evidence is preserved, choose a new output directory.")
    manifest = {
        "schema_version": SCHEMA_VERSION, "case_id": config["system"], "status": "completed",
        "config_source": str(config_path), "config_source_sha256": hashlib.sha256(raw).hexdigest(),
        "overrides": dict(overrides or {}), "effective_contract": validated,
        "method": validated["numerical_contract"]["integrator"], "classifier_policy": "terminal_window_mean_v1",
        "reference_contact_enabled": False, "hidden_verified": False,
        "claim_scope": "Finite operational destinations under the declared method; mean-sign classes do not establish an attractor identity, chaos or hiddenness. Local ADM has no accumulated Caputo history.",
        "points_total": len(all_rows), "class_counts": dict(Counter(row["class_label"] for row in all_rows)),
        "equilibria": {name: value.tolist() for name, value in equilibria.items()},
        "source_hashes": sources,
        "source_hash_scope": "Method-specific Python numerical/classification dependencies snapshotted before integration; dependency package versions remain environment metadata.",
        "environment": {"numpy": np.__version__}, "files": files,
        "hash_scope": "All scientific artifacts except this manifest; mutable run-ownership marker excluded.",
    }
    _write_json(output / "manifest.json", manifest)
    complete_output_claim(claim)
    return manifest


def _render_map(path: Path, labels, axis, plane: str, fixed: int, validated) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    colors = ["#3b82f6", "#10b981", "#a855f7", "#1e293b", "#eab308", "#f97316", "#cbd5e1"]
    figure, axes = plt.subplots(figsize=(9.0, 6.4))
    image = axes.imshow(labels, origin="lower", extent=[axis[0], axis[-1], axis[0], axis[-1]],
                        interpolation="nearest", cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(-.5, 7.5), 7))
    bar = figure.colorbar(image, ax=axes, ticks=np.arange(7), shrink=.8)
    bar.ax.set_yticklabels(list(CLASS_LABELS.values()), fontsize=8)
    axes.set_xlabel(f"{plane[0]}(0)")
    axes.set_ylabel(f"{plane[1]}(0)")
    numerical, settings = validated["numerical_contract"], validated["basin_map"]
    axes.set_title(f"Wu2023 finite destinations: {plane}, {'xyz'[fixed]}0={settings['fixed_coordinate']:g}\n"
                   f"{numerical['integrator']}; q={numerical['q']:g}, h={numerical['h']:g}, T={numerical['t_final']:g}")
    figure.text(.02, .025, "Operational terminal classes; no reference-contact, hiddenness or chaos proof.\n"
                + ("Local ADM: no accumulated Caputo history." if numerical["integrator"] == "adm_wu2023" else "Explicit full-history EFORK diagnostic."), fontsize=8)
    figure.tight_layout(rect=(0, .08, 1, 1))
    figure.savefig(path, dpi=160)
    plt.close(figure)


__all__ = ["CLASS_LABELS", "classify_map_trajectory", "integrate_map_point", "run_basin_maps", "validate_map_contract"]
