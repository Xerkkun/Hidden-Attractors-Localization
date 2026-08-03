"""Reproduce a structured integer MAVPD hidden-chaos search.

The script starts from the declared equations and the direct integer Lur'e
transfer condition.  A parameter continuation is enabled only after neither
direct base branch passes the declared finite-time Lyapunov chaos screen.
Frequency samples are used only to draw a Nyquist curve; they are never used
to obtain the seed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import yaml

from hidden_attractors.analysis.boundedness import compute_boundedness_metrics
from hidden_attractors.analysis.lyapunov_adaptive import (
    AdaptiveLyapunovResult,
    integer_system_dop853_variational_qr,
)
from hidden_attractors.analysis.poincare import (
    detect_poincare_crossings,
    summarize_poincare_points,
)
from hidden_attractors.analysis.spectral import spectral_diagnostics_multicoordinate
from hidden_attractors.analysis.trajectory import sample_rows
from hidden_attractors.analysis.zero_one import zero_one_test
from hidden_attractors.plotting.dynamics import plot_lure_nyquist_describing_function
from hidden_attractors.plotting.export import (
    promote_local_figure_pairs_batch,
    save_figure_pair_local,
)
from hidden_attractors.reproducibility import (
    ContinuationMetadata,
    collect_lure_metadata,
    collect_run_metadata,
    metadata_to_jsonable,
)
from hidden_attractors.seed_generation import lure_transfer_function
from hidden_attractors.solvers.integer import dop853_q1_integrate, efork_q1_integrate
from hidden_attractors.systems.modified_van_der_pol_duffing import (
    MAVPD_2023_DOI,
    mavpd_2023_system,
    mavpd_hopf_gamma_boundaries,
    mavpd_nonzero_equilibrium_characteristic_coefficients,
)
from hidden_attractors.verification.attractor_reference import (
    AttractorReferenceCalibration,
    calibrate_attractor_reference,
    classify_cloud_against_reference,
)
from hidden_attractors.verification.candidate_gate import evaluate_candidate_gate
from hidden_attractors.workflows.integer_hidden_chaos import (
    IntegerHiddenChaosProbe,
    continue_integer_parameter_path,
    deterministic_unit_directions,
    equilibrium_stability_records,
    run_integer_hidden_chaos_controls,
    summarize_integer_hidden_chaos_controls,
)
from hidden_attractors.workflows.integer_lure import (
    continue_integer_lure_seed,
    integer_lure_seed,
)
from hidden_attractors.workflows.protocol import ContinuationPlan


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "reproducibility.yaml"

SCIENTIFIC_SOURCE_FIXED_FILES = (
    "examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/run_example.py",
    "examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/reproducibility.yaml",
    "validation/wolfram/cases/mavpd_integer.wl",
)


def _scientific_source_files() -> tuple[str, ...]:
    """Discover the maintained source set afresh so newly added modules are detected."""

    return tuple(
        sorted(
            {
                *SCIENTIFIC_SOURCE_FIXED_FILES,
                *(
                    path.relative_to(ROOT).as_posix()
                    for path in (ROOT / "hidden_attractors").rglob("*.py")
                ),
            }
        )
    )


def _scientific_source_snapshot() -> dict[str, Any]:
    """Hash the exact maintained sources that determine this experiment."""

    file_hashes: dict[str, str] = {}
    for relative in _scientific_source_files():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"scientific source is missing: {relative}")
        file_hashes[relative] = sha256(path.read_bytes()).hexdigest()
    bundle_material = "".join(
        f"{relative}\0{digest}\n" for relative, digest in sorted(file_hashes.items())
    ).encode("utf-8")
    return {
        "algorithm": "sha256",
        "bundle_sha256": sha256(bundle_material).hexdigest(),
        "files": file_hashes,
    }


def _assert_scientific_sources_unchanged(expected: Mapping[str, Any], *, phase: str) -> None:
    """Abort before promotion when a maintained scientific source changed mid-run."""

    current = _scientific_source_snapshot()
    if current["bundle_sha256"] != expected["bundle_sha256"]:
        raise RuntimeError(
            "scientific source bundle changed during "
            f"{phase}: expected {expected['bundle_sha256']}, got {current['bundle_sha256']}"
        )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    names = list(fieldnames or (materialized[0].keys() if materialized else ()))
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
            writer.writeheader()
            for row in materialized:
                writer.writerow({key: _csv_value(row.get(key)) for key in names})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _config_sha256(config: Mapping[str, Any]) -> str:
    material = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(material).hexdigest()


def _runtime_environment() -> dict[str, str]:
    """Return the numerical runtime identity required by a resumable checkpoint."""

    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }


def _new_run_status(
    output: Path,
    config: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    created = datetime.now(timezone.utc)
    run_id = (
        f"{config['case_id']}-{'quick-smoke' if config.get('quick_mode') else 'full'}-"
        f"{created.strftime('%Y%m%dT%H%M%S%fZ')}-{source_snapshot['bundle_sha256'][:12]}-"
        f"{uuid4().hex[:12]}"
    )
    status = {
        "status": "in_progress",
        "run_id": run_id,
        "created_at_utc": created.isoformat(),
        "quick_mode": bool(config.get("quick_mode")),
        "config_sha256": _config_sha256(config),
        "runtime_environment": _runtime_environment(),
        "scientific_source_snapshot": dict(source_snapshot),
        "completed_phases": [],
        "artifacts": {},
    }
    stale_manifest = output / "run_manifest.json"
    if stale_manifest.exists():
        stale_manifest.unlink()
    _write_json(output / "run_status.json", status)
    return status


def _record_run_phase(
    output: Path,
    status: dict[str, Any],
    phase: str,
    relative_paths: Sequence[str],
) -> None:
    artifact_hashes = status.setdefault("artifacts", {})
    for relative in relative_paths:
        path = output / relative
        if not path.is_file():
            raise FileNotFoundError(f"phase {phase!r} did not produce required artifact: {relative}")
        artifact_hashes[relative] = sha256(path.read_bytes()).hexdigest()
    phases = status.setdefault("completed_phases", [])
    if phase not in phases:
        phases.append(phase)
    status["last_completed_phase"] = phase
    status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output / "run_status.json", status)


def _verify_recorded_artifacts(
    output: Path,
    status: Mapping[str, Any],
    required_paths: Sequence[str],
) -> None:
    recorded = status.get("artifacts", {})
    for relative in required_paths:
        path = output / relative
        expected = recorded.get(relative) if isinstance(recorded, Mapping) else None
        if not path.is_file() or not isinstance(expected, str):
            raise RuntimeError(f"cannot resume: {relative} is not bound to the interrupted run")
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"cannot resume: artifact hash mismatch for {relative}")


def _save_figure_pair(figure: Any, png_path: Path) -> None:
    """Save a title-free report figure as both PNG and vector PDF."""

    save_figure_pair_local(figure, png_path)


FIGURE_DATA_SOURCES = {
    "00_nyquist_direct_seed": ["00_system_contract.json"],
    "03_continuation_screen": ["03_candidate_screening.csv"],
    "04_candidate_phase_portraits": ["04_candidate_trajectory.csv"],
    "04_candidate_time_series": ["04_candidate_trajectory.csv"],
    "05_lyapunov_convergence": ["05_lyapunov_convergence.csv"],
    "05_poincare_section": ["05_poincare_section.csv"],
    "05_normalized_fft_power": ["05_normalized_fft_power.csv"],
    "07_hiddenness_outcomes": ["07_hiddenness_probes.csv"],
}


def _finalize_figure_manifest(
    output: Path,
    config: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
    *,
    run_id: str,
    promote: bool,
) -> list[dict[str, Any]]:
    """Validate and bind every local pair before any optional global promotion."""

    figures = output / "figures"
    pairs = []
    for figure_id, data_sources in FIGURE_DATA_SOURCES.items():
        png_path = figures / f"{figure_id}.png"
        pdf_path = figures / f"{figure_id}.pdf"
        if not png_path.is_file() or not pdf_path.is_file():
            raise FileNotFoundError(f"required local figure pair is incomplete: {figure_id}")
        pairs.append((figure_id, data_sources, png_path, pdf_path))

    rows = []
    for figure_id, data_sources, png_path, pdf_path in pairs:
        parameters = (
            dict(config["system"]["base_parameters"])
            if figure_id.startswith("00_")
            else dict(candidate_parameters)
        )
        metadata = {
            "caption_key": f"fig_{figure_id}",
            "source_script": "examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/run_example.py",
            "source_function": "run_contract/run_diagnostics/run_hiddenness",
            "data_sources": data_sources,
            "system_id": "modified_van_der_pol_duffing",
            "q": 1.0,
            "parameters": parameters,
            "integrator": "display_only" if figure_id.startswith("00_") else "DOP853",
            "memory_mode": "not_applicable_integer_q1",
            "t_final": float(config["candidate_trajectory"]["t_final"]),
            "t_burn": float(config["candidate_trajectory"]["t_burn"]),
            "scientific_source_bundle_sha256": source_snapshot["bundle_sha256"],
            "quick_smoke_only": not promote,
        }
        rows.append(
            {
                "figure_id": figure_id,
                "run_id": run_id,
                "local_png": str(png_path.relative_to(output)).replace("\\", "/"),
                "local_pdf": str(pdf_path.relative_to(output)).replace("\\", "/"),
                "png_sha256": sha256(png_path.read_bytes()).hexdigest(),
                "pdf_sha256": sha256(pdf_path.read_bytes()).hexdigest(),
                "global_promotion_requested": bool(promote),
                "promoted_to_global_manifest": False,
                "central_paths": None,
                "metadata": metadata,
            }
        )
    _write_json(figures / "figure_manifest.json", {"figures": rows})
    return rows


def _restore_local_snapshots(snapshots: Mapping[Path, bytes | None]) -> None:
    """Restore local transaction records if the global batch validator fails."""

    for path, previous in snapshots.items():
        if previous is None:
            path.unlink(missing_ok=True)
            continue
        temporary = path.with_name(f".{path.name}.rollback.tmp")
        try:
            temporary.write_bytes(previous)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _promote_completed_figures(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    manifest: dict[str, Any],
    run_status: dict[str, Any],
    source_snapshot: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Promote one completed run as a global batch with local-record rollback."""

    persisted_status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    if persisted_status.get("status") != "complete" or persisted_status.get("quick_mode") is not False:
        raise RuntimeError("global figure promotion requires a completed full run")
    if persisted_status.get("run_id") != run_status.get("run_id"):
        raise RuntimeError("global figure promotion run_id differs from the completed run")
    _verify_recorded_artifacts(output, persisted_status, tuple(persisted_status["artifacts"]))
    _assert_scientific_sources_unchanged(source_snapshot, phase="pre-promotion verification")

    local_figure_manifest = output / "figures" / "figure_manifest.json"
    run_manifest_path = output / "run_manifest.json"
    status_path = output / "run_status.json"
    timings_path = output / "phase_timings.csv"
    receipt_path = output / "figures" / "global_promotion_receipt.json"
    snapshots = {
        local_figure_manifest: local_figure_manifest.read_bytes(),
        run_manifest_path: run_manifest_path.read_bytes(),
        status_path: status_path.read_bytes(),
        timings_path: timings_path.read_bytes(),
        receipt_path: receipt_path.read_bytes() if receipt_path.exists() else None,
    }
    promoted_rows: list[dict[str, Any]] = []
    updated_manifest = dict(manifest)

    promotions = [
        {
            "png_path": output / str(row["local_png"]),
            "kind": "mavpd_integer_hidden_chaos",
            "metadata_dict": dict(row["metadata"]),
            "export_targets": ["mavpd_integer_hidden_chaos_report"],
        }
        for row in rows
    ]
    promotion_started = time.perf_counter()

    def validate_and_record(**transaction: Any) -> bool:
        nonlocal promoted_rows, updated_manifest
        try:
            _assert_scientific_sources_unchanged(source_snapshot, phase="global figure promotion")
            promoted_pairs = tuple(transaction["promoted_pairs"])
            if len(promoted_pairs) != len(rows):
                raise RuntimeError("global promotion returned an incomplete figure batch")
            promotion_seconds = time.perf_counter() - promotion_started
            promoted_rows = []
            for row, (central_pdf, central_png) in zip(rows, promoted_pairs, strict=True):
                promoted_rows.append(
                    {
                        **dict(row),
                        "promoted_to_global_manifest": True,
                        "central_paths": {"pdf": str(central_pdf), "png": str(central_png)},
                    }
                )
            receipt = {
                "status": "committed",
                "run_id": run_status["run_id"],
                "scientific_source_bundle_sha256": source_snapshot["bundle_sha256"],
                "figure_count": len(promoted_rows),
                "figure_ids": [row["figure_id"] for row in promoted_rows],
                "seconds": promotion_seconds,
                "global_manifest_paths": [str(path) for path in transaction["manifest_paths"]],
            }
            updated_timings = []
            gate_timing_found = False
            total_timing_found = False
            for timing in manifest["timings"]:
                timing_row = dict(timing)
                phase = str(timing_row["phase"])
                if phase.startswith("candidate_gate"):
                    timing_row["seconds"] = float(timing_row["seconds"]) + promotion_seconds
                    gate_timing_found = True
                elif phase.startswith("total"):
                    timing_row["seconds"] = float(timing_row["seconds"]) + promotion_seconds
                    total_timing_found = True
                updated_timings.append(timing_row)
            if not gate_timing_found or not total_timing_found:
                raise RuntimeError("run manifest lacks gate/total timing rows required for promotion")
            updated_manifest = {
                **manifest,
                "figures": promoted_rows,
                "global_figure_promotion": receipt,
                "timings": updated_timings,
            }
            _write_json(local_figure_manifest, {"figures": promoted_rows})
            _write_json(receipt_path, receipt)
            _write_rows(timings_path, updated_timings)
            _write_json(run_manifest_path, updated_manifest)
            _record_run_phase(
                output,
                run_status,
                "global_figure_promotion",
                (
                    "figures/figure_manifest.json",
                    "figures/global_promotion_receipt.json",
                    "phase_timings.csv",
                    "run_manifest.json",
                ),
            )
            _assert_scientific_sources_unchanged(source_snapshot, phase="post-promotion record")
        except Exception:
            _restore_local_snapshots(snapshots)
            raise
        return True

    promote_local_figure_pairs_batch(
        promotions,
        run_id=str(run_status["run_id"]),
        validator=validate_and_record,
    )
    return promoted_rows, updated_manifest


def _load_persisted_pre_resume_timings(path: Path) -> list[dict[str, Any]]:
    """Load immutable phase measurements instead of reconstructing them from mtimes."""

    if not path.is_file():
        raise FileNotFoundError(
            "cannot resume without the phase_timings.csv written by the interrupted run"
        )
    with path.open("r", newline="", encoding="utf-8") as handle:
        materialized = list(csv.DictReader(handle))
    excluded_prefixes = ("hiddenness", "candidate_gate", "total")
    rows: list[dict[str, Any]] = []
    for raw in materialized:
        phase = str(raw.get("phase", "")).strip()
        if not phase or phase.startswith(excluded_prefixes):
            continue
        seconds = float(raw["seconds"])
        if not np.isfinite(seconds) or seconds < 0.0:
            raise ValueError(f"invalid persisted duration for phase {phase!r}: {seconds}")
        rows.append(
            {
                "phase": phase,
                "seconds": seconds,
                "timing_source": raw.get("timing_source")
                or "persisted_perf_counter_from_interrupted_run",
            }
        )
    if not rows:
        raise ValueError("phase_timings.csv contains no resumable completed phase")
    return rows


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, list, tuple, dict)):
        return json.dumps(_jsonable(value), separators=(",", ":"))
    return value


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def load_config(*, quick: bool = False) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if quick:
        config = _deep_update(config, config.get("quick", {}))
    config["quick_mode"] = bool(quick)
    return config


def _canonical_output_dir(config: Mapping[str, Any]) -> Path:
    return (ROOT / str(config["outputs"]["output_dir"])).resolve()


def _assert_isolated_working_output(config: Mapping[str, Any], output: Path) -> None:
    """Prevent partial workflows from writing into or around canonical evidence."""

    canonical = _canonical_output_dir(config)
    resolved = output.resolve()
    if resolved == canonical or resolved in canonical.parents or canonical in resolved.parents:
        raise ValueError(
            "working output must be isolated from the canonical evidence directory; "
            "run in tmp/staging and promote only after validating a complete run"
        )


def _assert_fresh_run_output(output: Path) -> None:
    """Reject a nonempty staging directory so artifacts from runs cannot mix."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"run output is not empty: {output}")


def _output_dir(config: Mapping[str, Any], override: str | Path | None) -> Path:
    if override is not None:
        return Path(override).resolve()
    if config.get("quick_mode") is True:
        return ROOT / "tmp" / "mavpd_integer_hidden_chaos_quick_smoke"
    return ROOT / "tmp" / "mavpd_integer_hidden_chaos_full_staging"


def _seed_record(seed: Any) -> dict[str, Any]:
    system = None
    return {
        "branch_index": int(seed.branch_index),
        "phase": 0.0,
        "omega0": float(seed.omega),
        "k": float(seed.gain),
        "a0": float(seed.amplitude),
        "seed": seed.seed,
        "eigenvector": seed.eigenvector,
        "matched_eigenvalue": seed.matched_eigenvalue,
        "method": seed.method,
        "search_route": seed.search_route,
        "frequency_grid_used": False,
        "published_table_used": False,
    }


def _hopf_payload(parameters: Mapping[str, float]) -> dict[str, Any]:
    boundaries = mavpd_hopf_gamma_boundaries(parameters)
    high = max(boundaries)
    a1, a2, a3 = mavpd_nonzero_equilibrium_characteristic_coefficients(
        {**parameters, "gamma": high}
    )
    return {
        "derivation": (
            "At E±, p(lambda)=lambda^3+a1 lambda^2+a2 lambda+a3; "
            "with g=2 delta gamma, a1*a2=a3 becomes "
            "xi*g^2+(xi^2-delta)*g+xi*(rho-delta)=0."
        ),
        "gamma_boundaries": boundaries,
        "selected_high_gamma_boundary": high,
        "boundary_interpretation": (
            "Routh-Hurwitz imaginary-pair crossing candidate; transversality "
            "and Hopf nondegeneracy are not asserted"
        ),
        "coefficients_at_selected_boundary": {"a1": a1, "a2": a2, "a3": a3},
        "routh_hurwitz_residual_at_boundary": a1 * a2 - a3,
        "values_derived_from_equations": True,
    }


def run_contract(
    config: Mapping[str, Any],
    output: Path,
    *,
    scientific_source_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_snapshot = dict(scientific_source_snapshot or _scientific_source_snapshot())
    parameters = dict(config["system"]["base_parameters"])
    system = mavpd_2023_system(parameters)
    assert system.lure is not None
    states = [np.zeros(3), np.array([0.37, -0.21, 0.58]), np.array([-0.9, 0.2, -0.4])]
    lure_residuals = [float(np.linalg.norm(system.evaluate(state) - system.lure.evaluate(state))) for state in states]
    point = states[1]
    step = 1.0e-7
    numerical_jacobian = np.column_stack(
        [
            (system.evaluate(point + step * direction) - system.evaluate(point - step * direction))
            / (2.0 * step)
            for direction in np.eye(3)
        ]
    )
    seeds = [
        integer_lure_seed(
            system,
            branch_index=branch,
            theta=float(config["primary_route"]["phase"]),
            wmin=float(config["primary_route"]["admissible_omega_min"]),
            wmax=float(config["primary_route"]["admissible_omega_max"]),
        )
        for branch in config["primary_route"]["branch_order"]
    ]
    seed_rows = [_seed_record(seed) for seed in seeds]
    transfer_checks = []
    for seed in seeds:
        value = lure_transfer_function(float(seed.omega), 1.0, system.lure)
        transfer_checks.append(
            {
                "branch_index": seed.branch_index,
                "omega0": float(seed.omega),
                "W_iomega": value,
                "imaginary_residual": abs(value.imag),
                "closure_residual": abs(value.real + 1.0 / seed.gain),
                "describing_function_residual": abs(float(system.lure.describing_function(seed.amplitude)) - seed.gain),
            }
        )
    equilibria = []
    for name, state in system.equilibrium_points().items():
        equilibria.append(
            {
                "name": name,
                "state": state,
                "rhs_residual": float(np.linalg.norm(system.evaluate(state))),
                "eigenvalues": np.linalg.eigvals(system.jacobian_matrix(state)),
            }
        )
    target_parameters = {
        **parameters,
        "xi": float(config["alternative_parameter_continuation"]["xi_stop"]),
    }
    payload = {
        "case_id": config["case_id"],
        "source_doi": MAVPD_2023_DOI,
        "source_scope": "published_model_equations_only",
        "candidate_parameter_set_published": False,
        "scientific_source_snapshot": source_snapshot,
        "equations": [
            "y1'=delta*gamma*y1+delta*y2-delta*y1^3",
            "y2'=y1-xi*y2-y3",
            "y3'=rho*y2",
        ],
        "q": 1.0,
        "base_parameters": parameters,
        "lure": {
            "A": system.lure.matrix,
            "b": system.lure.input_vector,
            "c": system.lure.output_vector,
            "psi": "sigma^3",
            "describing_function": "N(a)=3*a^2/4",
            "max_field_residual": max(lure_residuals),
        },
        "jacobian_residual": float(np.linalg.norm(system.jacobian_matrix(point) - numerical_jacobian)),
        "equilibria": equilibria,
        "direct_seed_records": seed_rows,
        "transfer_checks": transfer_checks,
        "hopf_boundary_at_xi_target": _hopf_payload(target_parameters),
        "frequency_grid_used_for_seed": False,
        "fallback_frequency_scan_used": False,
        "report_values_used_as_search_input": False,
        "mathematica_validation": "validation/wolfram/cases/mavpd_integer.wl",
    }
    _write_json(output / "00_system_contract.json", payload)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    visual = config["visualization"]
    plot_lure_nyquist_describing_function(
        system.lure,
        seeds[0],
        figures / "00_nyquist_direct_seed.png",
        q=1.0,
        wmin=float(visual["nyquist_frequency_min"]),
        wmax=float(visual["nyquist_frequency_max"]),
        amin=float(visual["nyquist_amplitude_min"]),
        amax=float(visual["nyquist_amplitude_max"]),
        title="",
        local_pair_only=True,
    )
    return {"system": system, "seeds": seeds, "payload": payload}


def _lyapunov_payload(result: AdaptiveLyapunovResult) -> dict[str, Any]:
    return {
        "method": result.method_id,
        "status": result.status,
        "exponents": result.exponents,
        "sum_exponents": result.sum_exponents,
        "final_state": result.final_state,
        "t_accumulate": result.accumulated_time,
        "metadata": result.metadata,
        "finite_time_local": result.finite_time_local,
        "does_not_prove_chaos_alone": True,
    }


def _candidate_cloud(
    system: Any,
    state: np.ndarray,
    *,
    duration: float,
    burn: float,
    sample_step: float,
    max_step: float,
    div_threshold: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    trajectory, status = dop853_q1_integrate(
        lambda value: system.evaluate(value),
        state,
        t_final=duration,
        h=sample_step,
        rtol=2.0e-10,
        atol=2.0e-12,
        max_step=max_step,
        div_threshold=float(div_threshold),
    )
    if status != "ok":
        raise RuntimeError(f"candidate cloud integration failed: {status}")
    midpoint = burn + 0.5 * (duration - burn)
    clouds = [
        trajectory[(trajectory[:, 0] >= burn) & (trajectory[:, 0] < midpoint), 1:],
        trajectory[trajectory[:, 0] >= midpoint, 1:],
    ]
    return trajectory, clouds


def run_search(
    config: Mapping[str, Any],
    output: Path,
    contract: Mapping[str, Any],
    *,
    scientific_source_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_snapshot = dict(scientific_source_snapshot or _scientific_source_snapshot())
    primary = config["primary_route"]
    primary_screen = primary["chaos_screen"]
    div_threshold = float(config["numerical_safety"]["divergence_norm"])
    base_system = contract["system"]
    lambda_plan = ContinuationPlan.uniform(int(primary["lambda_nodes"]), internal_parameter="epsilon")
    branch_results = []
    branch_steps = {}
    for seed in contract["seeds"]:
        steps = continue_integer_lure_seed(
            base_system,
            seed,
            plan=lambda_plan,
            t_transient=float(primary["t_transient"]),
            t_keep=float(primary["t_keep"]),
            h=float(primary["h"]),
            div_threshold=div_threshold,
        )
        if not steps or steps[-1].status != "ok":
            raise RuntimeError(f"lambda continuation failed for branch {seed.branch_index}")
        branch_steps[int(seed.branch_index)] = steps
        short_le = integer_system_dop853_variational_qr(
            base_system,
            steps[-1].x_out,
            t_burn=float(primary_screen["t_burn"]),
            t_accumulate=float(primary_screen["t_accumulate"]),
            qr_interval=float(primary_screen["qr_interval"]),
            rtol=float(primary_screen["rtol"]),
            atol=float(primary_screen["atol"]),
            max_step=float(primary_screen["max_step"]),
            div_threshold=div_threshold,
        )
        branch_results.append(
            {
                "branch_index": seed.branch_index,
                "omega0": float(seed.omega),
                "lambda_nodes_completed": len(steps),
                "final_lambda": steps[-1].lambda_value,
                "final_state": steps[-1].x_out,
                "lyapunov": _lyapunov_payload(short_le),
                "chaotic_at_base": bool(
                    short_le.status == "ok"
                    and short_le.exponents[0] > float(primary_screen["positive_threshold"])
                ),
            }
        )
    primary_failed = not any(row["chaotic_at_base"] for row in branch_results)
    if not primary_failed:
        raise RuntimeError("the declared alternative cannot run because a direct base branch qualified as chaotic")
    alternative = config["alternative_parameter_continuation"]
    source_branch_rule = str(alternative["source_branch_selection_rule"])
    if source_branch_rule != "lowest direct harmonic frequency among successful direct base branches":
        raise ValueError(f"unsupported alternative source-branch rule: {source_branch_rule}")
    successful_base_branches = [
        row for row in branch_results if row["lyapunov"]["status"] == "ok"
    ]
    if not successful_base_branches:
        raise RuntimeError("no successful direct base branch is available for the alternative route")
    source_branch = min(
        successful_base_branches,
        key=lambda row: float(row["omega0"]),
    )
    source_branch_index = int(source_branch["branch_index"])
    _write_json(
        output / "01_direct_seed_and_lambda_continuation.json",
        {
            "primary_route": "direct_integer_lure",
            "frequency_grid_used": False,
            "branches": branch_results,
            "alternative_triggered": primary_failed,
            "trigger": primary["trigger_for_alternative"],
            "alternative_source_branch_index": source_branch_index,
            "alternative_source_branch_selection_rule": source_branch_rule,
        },
    )

    screening = alternative["screening"]
    screening_lyapunov = screening["lyapunov"]
    screening_reference = screening["reference"]
    screening_probes = screening["probes"]
    base_parameters = dict(config["system"]["base_parameters"])
    xi_start = float(alternative["xi_start"])
    xi_stop = float(alternative["xi_stop"])
    xi_step = float(alternative["xi_step"])
    if not np.isclose(xi_start, float(base_parameters["xi"]), rtol=0.0, atol=1.0e-12):
        raise ValueError("alternative xi_start must equal the base-system xi")
    if xi_step == 0.0:
        raise ValueError("alternative xi_step must be nonzero")
    xi_intervals = int(round((xi_stop - xi_start) / xi_step))
    if xi_intervals < 1 or not np.isclose(
        xi_start + xi_intervals * xi_step,
        xi_stop,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("xi_start, xi_stop, and xi_step must define an exact directed path")
    xi_values = [xi_start + index * xi_step for index in range(1, xi_intervals + 1)]
    xi_path = [
        {**base_parameters, "xi": float(round(value, 10))}
        for value in xi_values
    ]
    xi_steps = continue_integer_parameter_path(
        mavpd_2023_system,
        xi_path,
        branch_steps[source_branch_index][-1].x_out,
        t_burn=float(alternative["node_t_burn"]),
        t_keep=float(alternative["node_t_keep"]),
        sample_step=float(alternative["sample_step"]),
        rtol=float(alternative["rtol"]),
        atol=float(alternative["atol"]),
        max_step=float(alternative["max_step"]),
        div_threshold=div_threshold,
    )
    if not xi_steps or xi_steps[-1].status != "ok":
        raise RuntimeError("xi continuation failed")

    target_base = {**base_parameters, "xi": xi_stop}
    gamma_h = max(mavpd_hopf_gamma_boundaries(target_base))
    offsets = [float(value) for value in alternative["hopf_offsets"]]
    largest_gamma = gamma_h + max(offsets)
    gamma_step = float(alternative["gamma_base_step"])
    gamma_start = float(base_parameters["gamma"]) + gamma_step
    gamma_values = list(np.arange(gamma_start, largest_gamma, gamma_step))
    gamma_values.extend(gamma_h + offset for offset in offsets)
    gamma_values = sorted({float(value) for value in gamma_values if value > 0.1 + 1.0e-12 and value <= largest_gamma + 1.0e-12})
    gamma_path = [{**target_base, "gamma": value} for value in gamma_values]
    gamma_steps = continue_integer_parameter_path(
        mavpd_2023_system,
        gamma_path,
        xi_steps[-1].x_out,
        t_burn=float(alternative["node_t_burn"]),
        t_keep=float(alternative["node_t_keep"]),
        sample_step=float(alternative["sample_step"]),
        rtol=float(alternative["rtol"]),
        atol=float(alternative["atol"]),
        max_step=float(alternative["max_step"]),
        div_threshold=div_threshold,
    )
    if not gamma_steps or gamma_steps[-1].status != "ok":
        raise RuntimeError("gamma continuation failed")
    continuation_rows = []
    for stage, steps in (("xi", xi_steps), ("gamma", gamma_steps)):
        for node in steps:
            continuation_rows.append(
                {
                    "stage": stage,
                    "node_index": node.node_index,
                    "xi": node.parameters["xi"],
                    "gamma": node.parameters["gamma"],
                    "x_in": node.x_in,
                    "x_out": node.x_out,
                    "status": node.status,
                    "system_rebuilt_at_node": True,
                    "lure_rebuilt_at_node": True,
                }
            )
    _write_rows(output / "02_parameter_continuation.csv", continuation_rows)

    screens = []
    screen_probe_rows: list[dict[str, Any]] = []
    selected_states: dict[float, np.ndarray] = {}
    for offset in offsets:
        target_gamma = gamma_h + offset
        node = min(gamma_steps, key=lambda item: abs(float(item.parameters["gamma"]) - target_gamma))
        if abs(float(node.parameters["gamma"]) - target_gamma) > 1.0e-11:
            raise RuntimeError("an exact Hopf-offset node is missing from the continuation")
        system = mavpd_2023_system(node.parameters)
        stability = equilibrium_stability_records(system)
        ep_margin = -max(
            row["spectral_abscissa"] for row in stability if row["equilibrium"] in {"E+", "E-"}
        )
        screen_le = integer_system_dop853_variational_qr(
            system,
            node.x_out,
            t_burn=float(screening_lyapunov["t_burn"]),
            t_accumulate=float(screening_lyapunov["t_accumulate"]),
            qr_interval=float(screening_lyapunov["qr_interval"]),
            rtol=float(screening_lyapunov["rtol"]),
            atol=float(screening_lyapunov["atol"]),
            max_step=float(screening_lyapunov["max_step"]),
            div_threshold=div_threshold,
        )
        _trajectory, clouds = _candidate_cloud(
            system,
            node.x_out,
            duration=float(screening_reference["duration"]),
            burn=float(screening_reference["burn"]),
            sample_step=float(screening_reference["sample_step"]),
            max_step=float(screening_reference["max_step"]),
            div_threshold=div_threshold,
        )
        calibration = calibrate_attractor_reference(
            clouds,
            safety_factor=float(screening_reference["safety_factor"]),
            max_points=int(screening_reference["max_points"]),
        )
        direction_count = int(screening_probes["directions"])
        axes = (
            np.vstack((np.eye(3)[0], -np.eye(3)[0]))
            if direction_count == 2
            else deterministic_unit_directions(3, direction_count)
        )
        screen_radii = tuple(float(value) for value in screening_probes["radii"])
        screen_equilibria = tuple(str(value) for value in screening_probes["equilibrium_names"])
        probes = run_integer_hidden_chaos_controls(
            system,
            clouds,
            calibration,
            equilibrium_names=screen_equilibria,
            radii=screen_radii,
            directions=axes,
            samples_per_radius=len(axes),
            sampling_mode=str(screening_probes["sampling_mode"]),
            t_burn=float(screening_probes["t_burn"]),
            t_keep=float(screening_probes["t_keep"]),
            sample_step=float(screening_probes["sample_step"]),
            rtol=float(screening_probes["rtol"]),
            atol=float(screening_probes["atol"]),
            max_step=float(screening_probes["max_step"]),
            div_threshold=div_threshold,
            equilibrium_tol=float(screening_probes["equilibrium_tol"]),
            equilibrium_tail_span_tol=float(screening_probes["equilibrium_tail_span_tol"]),
            max_cloud_points=int(screening_reference["max_points"]),
        )
        probe_summary = summarize_integer_hidden_chaos_controls(
            probes,
            required_equilibrium_names=screen_equilibria,
            declared_equilibrium_names=tuple(system.equilibrium_points()),
        )
        screen_probe_rows.extend(
            {
                "hopf_offset": offset,
                "gamma": target_gamma,
                "reference_acceptance_threshold": calibration.acceptance_threshold,
                **_probe_row(probe, contract="candidate_screen_E0"),
            }
            for probe in probes
        )
        row = {
            "hopf_offset": offset,
            "gamma": target_gamma,
            "equilibrium_stability_margin": ep_margin,
            "lambda_1": float(screen_le.exponents[0]),
            "lambda_2": float(screen_le.exponents[1]),
            "lambda_3": float(screen_le.exponents[2]),
            "lyapunov_status": screen_le.status,
            "E0_probe_count": probe_summary["n_probes"],
            "E0_target_hits": probe_summary["target_hits"],
            "E0_ambiguous": probe_summary["ambiguous"],
            "eligible_hidden_chaos_screen": bool(
                screen_le.status == "ok"
                and screen_le.exponents[0] > float(screening_lyapunov["positive_threshold"])
                and probe_summary["target_hits"] == 0
                and probe_summary["ambiguous"] == 0
                and probe_summary["numerical_failures"] == 0
            ),
        }
        screens.append(row)
        selected_states[offset] = node.x_out.copy()
    eligible = [row for row in screens if row["eligible_hidden_chaos_screen"]]
    if not eligible:
        raise RuntimeError("no Hopf-offset node passed the declared finite screen")
    selected = max(eligible, key=lambda row: row["hopf_offset"])
    selected_offset = float(selected["hopf_offset"])
    _write_rows(output / "03_candidate_screening.csv", screens)
    _write_rows(output / "03_candidate_screening_probes.csv", screen_probe_rows)
    _write_json(
        output / "03_candidate_screening_contract.json",
        {
            "contract": screening,
            "candidate_states_source": "02_parameter_continuation.csv x_out at each exact Hopf-offset node",
            "probe_initial_conditions": "03_candidate_screening_probes.csv",
            "finite_sample_only": True,
        },
    )
    _write_json(
        output / "03_candidate_selection.json",
        {
            "hopf_boundary": gamma_h,
            "selection_rule": alternative["selection_rule"],
            "selected": selected,
            "selected_candidate_initial_state": selected_states[selected_offset],
            "parameter_provenance": "gamma_selected_by_local_continuation_at_declared_xi_endpoint_not_a_published_parameter_tuple",
            "xi_endpoint_provenance": "declared_local_continuation_endpoint_not_selected_by_candidate_screen",
            "alternative_source_branch_index": source_branch_index,
            "alternative_source_branch_selection_rule": source_branch_rule,
            "scientific_source_snapshot": source_snapshot,
            "no_frequency_sweep": True,
            "primary_route_chaos_screen_failed_before_alternative": primary_failed,
        },
    )
    return {
        "gamma_h": gamma_h,
        "selected": selected,
        "selected_state": np.asarray(selected_states[selected_offset], dtype=float),
        "selected_system": mavpd_2023_system(
            {**target_base, "gamma": float(selected["gamma"])}
        ),
        "screens": screens,
        "direct_seed": next(
            seed for seed in contract["seeds"] if int(seed.branch_index) == source_branch_index
        ),
        "lambda_values": lambda_plan.lambda_values,
    }


def _kaplan_yorke(exponents: Sequence[float]) -> float:
    ordered = np.sort(np.asarray(exponents, dtype=float))[::-1]
    total = 0.0
    for index, value in enumerate(ordered):
        if total + value < 0.0:
            return float(index + total / abs(value))
        total += value
    return float(len(ordered))


def _plot_diagnostics(
    output: Path,
    trajectory: np.ndarray,
    lyapunov: AdaptiveLyapunovResult,
    poincare_points: np.ndarray,
    spectrum: Mapping[str, Any],
    screens: Sequence[Mapping[str, Any]],
    *,
    time_series_tail: float,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    states = trajectory[:, 1:]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    for axis, (i, j, xlabel, ylabel) in zip(
        axes,
        ((0, 1, "$y_1$", "$y_2$"), (0, 2, "$y_1$", "$y_3$"), (1, 2, "$y_2$", "$y_3$")),
        strict=True,
    ):
        axis.plot(states[:, i], states[:, j], lw=0.28, color="#1f5a94")
        axis.set(xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    _save_figure_pair(fig, figures / "04_candidate_phase_portraits.png")
    plt.close(fig)

    mask = trajectory[:, 0] >= trajectory[-1, 0] - float(time_series_tail)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.0), sharex=True)
    for index, axis in enumerate(axes):
        axis.plot(trajectory[mask, 0], trajectory[mask, index + 1], lw=0.7)
        axis.set_ylabel(f"$y_{index + 1}$")
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    _save_figure_pair(fig, figures / "04_candidate_time_series.png")
    plt.close(fig)

    fig, (axis, zoom) = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
    for index in range(lyapunov.convergence.shape[1]):
        axis.plot(lyapunov.times, lyapunov.convergence[:, index], label=fr"$\lambda_{index + 1}$")
    axis.axhline(0.0, color="black", lw=0.8)
    axis.set(ylabel="finite-time exponent")
    axis.legend()
    for index in range(min(2, lyapunov.convergence.shape[1])):
        zoom.plot(lyapunov.times, lyapunov.convergence[:, index], label=fr"$\lambda_{index + 1}$")
    zoom.axhline(0.0, color="black", lw=0.8)
    zoom.set(xlabel="accumulation time", ylabel="expanded scale")
    zoom.legend()
    fig.tight_layout()
    _save_figure_pair(fig, figures / "05_lyapunov_convergence.png")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    if poincare_points.size:
        axis.scatter(poincare_points[:, 0], poincare_points[:, 1], s=6, alpha=0.7)
    axis.set(xlabel="$y_1$ at $y_2=0$", ylabel="$y_3$ at $y_2=0$")
    fig.tight_layout()
    _save_figure_pair(fig, figures / "05_poincare_section.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.2), sharex=True)
    for axis, name in zip(axes, ("x", "y", "z"), strict=True):
        values = spectrum["coordinate_results"][name]
        frequencies = np.asarray(values["frequencies"], dtype=float)
        power = np.asarray(values["power"], dtype=float)
        keep = frequencies > 0.0
        axis.semilogy(frequencies[keep], np.maximum(power[keep], 1.0e-18), lw=0.7)
        axis.set_ylabel(f"$y_{{{('x','y','z').index(name)+1}}}$")
    axes[-1].set_xlabel("frequency (cycles per unit time)")
    fig.tight_layout()
    _save_figure_pair(fig, figures / "05_normalized_fft_power.png")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.4, 4.7))
    offsets = [float(row["hopf_offset"]) for row in screens]
    lle = [float(row["lambda_1"]) for row in screens]
    margin = [float(row["equilibrium_stability_margin"]) for row in screens]
    axis.plot(offsets, lle, "o-", label=r"$\lambda_1$")
    axis.plot(offsets, margin, "s-", label="stability margin of $E_\\pm$")
    axis.axhline(0.0, color="black", lw=0.8)
    axis.set(xlabel=r"$\gamma-\gamma_H$", ylabel="rate")
    axis.legend()
    fig.tight_layout()
    _save_figure_pair(fig, figures / "03_continuation_screen.png")
    plt.close(fig)


def run_diagnostics(config: Mapping[str, Any], output: Path, search: Mapping[str, Any]) -> dict[str, Any]:
    system = search["selected_system"]
    state = np.asarray(search["selected_state"], dtype=float)
    trajectory_cfg = config["candidate_trajectory"]
    div_threshold = float(config["numerical_safety"]["divergence_norm"])
    trajectory, status = dop853_q1_integrate(
        lambda value: system.evaluate(value),
        state,
        t_final=float(trajectory_cfg["t_final"]),
        h=float(trajectory_cfg["sample_step"]),
        rtol=float(trajectory_cfg["rtol"]),
        atol=float(trajectory_cfg["atol"]),
        max_step=float(trajectory_cfg["max_step"]),
        div_threshold=div_threshold,
    )
    if status != "ok":
        raise RuntimeError(f"strict candidate trajectory failed: {status}")
    _write_rows(
        output / "04_candidate_trajectory.csv",
        [
            {"time": row[0], "y1": row[1], "y2": row[2], "y3": row[3]}
            for row in trajectory
        ],
    )
    burn_time = float(trajectory_cfg["t_burn"])
    boundedness = compute_boundedness_metrics(
        trajectory[:, 0], trajectory[:, 1:], burn_time, divergence_radius=div_threshold
    )
    windows = [tuple(map(float, item)) for item in trajectory_cfg["reference_windows"]]
    reference_clouds = [
        trajectory[(trajectory[:, 0] >= start) & (trajectory[:, 0] < stop), 1:]
        for start, stop in windows
    ]
    negative_controls = [
        np.repeat(value[None, :], int(config["hiddenness"]["negative_control_points"]), axis=0)
        for value in system.equilibrium_points().values()
    ]
    hidden_cfg = config["hiddenness"]
    calibration = calibrate_attractor_reference(
        reference_clouds,
        negative_control_clouds=negative_controls,
        max_points=int(hidden_cfg["reference_max_points"]),
        safety_factor=float(hidden_cfg["reference_safety_factor"]),
        ambiguity_fraction=float(hidden_cfg["reference_ambiguity_fraction"]),
    )

    lyapunov_cfg = config["chaos_diagnostics"]["lyapunov"]
    lyapunov = integer_system_dop853_variational_qr(
        system,
        state,
        t_burn=float(lyapunov_cfg["t_burn"]),
        t_accumulate=float(lyapunov_cfg["t_accumulate"]),
        qr_interval=float(lyapunov_cfg["qr_interval"]),
        rtol=float(lyapunov_cfg["rtol"]),
        atol=float(lyapunov_cfg["atol"]),
        max_step=float(lyapunov_cfg["max_step"]),
        div_threshold=div_threshold,
    )
    control_cfg = config["chaos_diagnostics"]["lyapunov_control"]
    control_state = state + np.asarray(control_cfg["perturbation"], dtype=float)
    lyapunov_control = integer_system_dop853_variational_qr(
        system,
        control_state,
        t_burn=float(control_cfg["t_burn"]),
        t_accumulate=float(control_cfg["t_accumulate"]),
        qr_interval=float(control_cfg["qr_interval"]),
        rtol=float(control_cfg["rtol"]),
        atol=float(control_cfg["atol"]),
        max_step=float(control_cfg["max_step"]),
        div_threshold=div_threshold,
    )
    if lyapunov.status != "ok" or lyapunov_control.status != "ok":
        raise RuntimeError("a Lyapunov refinement failed")

    section_cfg = config["chaos_diagnostics"]["poincare"]
    section = detect_poincare_crossings(
        trajectory[:, 0],
        trajectory[:, 1:],
        section_variable=int(section_cfg["section_variable"]),
        section_value=float(section_cfg["section_value"]),
        direction=str(section_cfg["direction"]),
        rhs=lambda _time, value: system.evaluate(value),
        derivative_mode="integer_rhs",
        min_crossing_separation=float(section_cfg["min_crossing_separation"]),
        burn_time=burn_time,
    )
    section_points = section.points[:, [0, 2]] if section.points.size else np.empty((0, 2))
    section_summary = summarize_poincare_points(
        section_points,
        duplicate_tolerance=float(section_cfg["duplicate_tolerance"]),
    )
    _write_rows(
        output / "05_poincare_section.csv",
        [
            {"time": time_value, "y1": point[0], "y3": point[1]}
            for time_value, point in zip(section.crossing_times, section_points, strict=True)
        ],
    )

    zero_cfg = config["chaos_diagnostics"]["zero_one"]
    post = trajectory[trajectory[:, 0] >= burn_time]
    flow_stride_rows = []
    for stride in zero_cfg["flow_strides"]:
        values = post[:: int(stride), 1]
        if len(values) < 100:
            flow_stride_rows.append(
                {
                    "series": "flow_y1",
                    "stride": int(stride),
                    "effective_sample_step": float(trajectory_cfg["sample_step"]) * int(stride),
                    "samples": len(values),
                    "K": float("nan"),
                    "state": "insufficient_samples",
                }
            )
            continue
        result = zero_one_test(
            values,
            n_c=int(zero_cfg["n_c"]),
            random_seed=int(zero_cfg["random_seed"]),
            max_samples=None,
        )
        flow_stride_rows.append(
            {
                "series": "flow_y1",
                "stride": int(stride),
                "effective_sample_step": float(trajectory_cfg["sample_step"]) * int(stride),
                "samples": len(values),
                "K": result["K"],
                "state": result["state"],
            }
        )
    return_results = []
    for coordinate in zero_cfg["return_coordinates"]:
        result = zero_one_test(
            section.points[:, int(coordinate)],
            n_c=int(zero_cfg["n_c"]),
            random_seed=int(zero_cfg["random_seed"]),
            max_samples=None,
        )
        return_results.append({"coordinate": int(coordinate), **result})
    _write_rows(output / "05_zero_one_stride_sensitivity.csv", flow_stride_rows)
    _write_json(output / "05_zero_one_return_map.json", return_results)

    spectrum = spectral_diagnostics_multicoordinate(
        trajectory[:, 0], trajectory[:, 1:], burn_time, coordinates=("x", "y", "z")
    )
    spectrum_rows = []
    spectrum_summary = {"state_global": spectrum["state_global"], "coordinate_results": {}}
    for coordinate, result in spectrum["coordinate_results"].items():
        spectrum_summary["coordinate_results"][coordinate] = {
            key: value for key, value in result.items() if key not in {"frequencies", "power"}
        }
        for frequency, power in zip(result["frequencies"], result["power"], strict=True):
            spectrum_rows.append({"coordinate": coordinate, "frequency": frequency, "normalized_fft_power": power})
    _write_rows(output / "05_normalized_fft_power.csv", spectrum_rows)

    efork_cfg = config["robustness"]["efork_crosscheck"]
    efork_time = min(float(efork_cfg["t_final_cap"]), float(trajectory_cfg["t_final"]))
    efork, efork_status = efork_q1_integrate(
        lambda value: system.evaluate(value),
        state,
        t_final=efork_time,
        h=float(efork_cfg["h"]),
        div_threshold=div_threshold,
    )
    efork_tail = efork[
        efork[:, 0]
        >= min(burn_time, float(efork_cfg["tail_burn_fraction"]) * efork_time),
        1:,
    ]
    efork_match = classify_cloud_against_reference(efork_tail, reference_clouds, calibration)
    retained_states = trajectory[trajectory[:, 0] >= burn_time, 1:]
    mean_divergence = float(np.mean(
        float(system.parameters["delta"]) * float(system.parameters["gamma"])
        - float(system.parameters["xi"])
        - 3.0 * float(system.parameters["delta"]) * retained_states[:, 0] ** 2
    ))
    diagnostics = {
        "candidate_parameters": dict(system.parameters),
        "boundedness": {key: value for key, value in boundedness.items() if key != "norm_timeseries"},
        "reference_calibration": calibration,
        "lyapunov": _lyapunov_payload(lyapunov),
        "lyapunov_control": _lyapunov_payload(lyapunov_control),
        "kaplan_yorke_dimension": _kaplan_yorke(lyapunov.exponents),
        "mean_vector_field_divergence": mean_divergence,
        "lyapunov_sum_minus_mean_divergence": lyapunov.sum_exponents - mean_divergence,
        "zero_one": {
            "flow_stride_sensitivity": flow_stride_rows,
            "return_map_results": return_results,
            "interpretation": "flow sampling is stride-sensitive; return-map 0-1 is the applicable supporting test",
        },
        "poincare": {**section_summary, "section_metadata": section.section_metadata},
        "spectrum": {
            **spectrum_summary,
            "gate_applicable": False,
            "interpretation": "FFT is supporting-only and its line dominance does not veto a positive variational exponent",
            "normalized_fft_power_not_welch_psd": True,
        },
        "efork_crosscheck": {
            "status": efork_status,
            "h": float(efork_cfg["h"]),
            "t_final": efork_time,
            "tail_burn_fraction": float(efork_cfg["tail_burn_fraction"]),
            "target_match": efork_match,
        },
        "finite_time_only": True,
    }
    _write_json(output / "05_chaos_diagnostics.json", diagnostics)
    _write_rows(
        output / "05_lyapunov_convergence.csv",
        [
            {"time": time_value, **{f"lambda_{index + 1}": row[index] for index in range(len(row))}}
            for time_value, row in zip(lyapunov.times, lyapunov.convergence, strict=True)
        ],
    )
    stability = equilibrium_stability_records(system)
    _write_json(output / "06_equilibrium_stability.json", stability)
    _plot_diagnostics(
        output,
        trajectory,
        lyapunov,
        section_points,
        spectrum,
        search["screens"],
        time_series_tail=float(config["visualization"]["time_series_tail"]),
    )
    return {
        "trajectory": trajectory,
        "reference_clouds": reference_clouds,
        "calibration": calibration,
        "boundedness": boundedness,
        "lyapunov": lyapunov,
        "lyapunov_control": lyapunov_control,
        "zero_one_return_results": return_results,
        "spectrum": spectrum,
        "poincare": section_summary,
        "stability": stability,
        "efork_status": efork_status,
        "efork_match": efork_match,
        "diagnostics": diagnostics,
    }


def _probe_row(probe: IntegerHiddenChaosProbe, *, contract: str) -> dict[str, Any]:
    return {
        "contract": contract,
        "sample_id": probe.sample_id,
        "equilibrium": probe.equilibrium,
        "radius": probe.radius,
        "direction_id": probe.direction_id,
        "sampling_mode": probe.sampling_mode,
        "x0": probe.x0,
        "status": probe.status,
        "destination": probe.destination,
        "target_classification": probe.target_classification,
        "target_distance_norm": probe.target_distance_norm,
        "target_hit": probe.target_hit,
        "ambiguous": probe.ambiguous,
        "tail_span": probe.tail_span,
        "closest_equilibrium": probe.closest_equilibrium,
        "closest_equilibrium_distance": probe.closest_equilibrium_distance,
    }


def _probe_cell_coverage(
    probes: Sequence[IntegerHiddenChaosProbe],
    *,
    equilibrium_names: Sequence[str],
    radii: Sequence[float],
    expected_per_cell: int,
) -> dict[str, Any]:
    """Audit every equilibrium-radius cell, including duplicate direction IDs."""

    cells = []
    for equilibrium in equilibrium_names:
        for radius in radii:
            matching = [
                probe
                for probe in probes
                if probe.equilibrium == equilibrium
                and np.isclose(float(probe.radius), float(radius), rtol=1.0e-12, atol=1.0e-15)
            ]
            unique_directions = {str(probe.direction_id) for probe in matching}
            completed = sum(probe.status == "ok" for probe in matching)
            cells.append(
                {
                    "equilibrium": str(equilibrium),
                    "radius": float(radius),
                    "expected": int(expected_per_cell),
                    "recorded": len(matching),
                    "completed": int(completed),
                    "unique_direction_ids": len(unique_directions),
                    "complete": bool(
                        len(matching) == int(expected_per_cell)
                        and completed == int(expected_per_cell)
                        and len(unique_directions) == int(expected_per_cell)
                    ),
                }
            )
    return {
        "expected_cells": len(tuple(equilibrium_names)) * len(tuple(radii)),
        "expected_per_cell": int(expected_per_cell),
        "cells": cells,
        "complete": bool(cells) and all(cell["complete"] for cell in cells),
    }


def run_hiddenness(config: Mapping[str, Any], output: Path, search: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    hidden = config["hiddenness"]
    system = search["selected_system"]
    div_threshold = float(config["numerical_safety"]["divergence_norm"])
    directions = deterministic_unit_directions(system.dimension, int(hidden["directions"]))
    main = run_integer_hidden_chaos_controls(
        system,
        diagnostics["reference_clouds"],
        diagnostics["calibration"],
        radii=tuple(float(value) for value in hidden["radii"]),
        directions=directions,
        samples_per_radius=int(hidden["directions"]),
        sampling_mode=str(hidden["sampling_mode"]),
        random_seed=int(hidden["random_seed"]),
        t_burn=float(hidden["t_burn"]),
        t_keep=float(hidden["t_keep"]),
        sample_step=float(hidden["sample_step"]),
        rtol=float(hidden["rtol"]),
        atol=float(hidden["atol"]),
        max_step=float(hidden["max_step"]),
        div_threshold=div_threshold,
        equilibrium_tol=float(hidden["equilibrium_tol"]),
        equilibrium_tail_span_tol=float(hidden["equilibrium_tail_span_tol"]),
        max_cloud_points=int(hidden["reference_max_points"]),
    )
    e0 = system.equilibrium_points()["E0"]
    eigenvalues, eigenvectors = np.linalg.eig(system.jacobian_matrix(e0))
    unstable_index = int(np.argmax(np.real(eigenvalues)))
    unstable_direction = np.real(eigenvectors[:, unstable_index])
    unstable_direction /= np.linalg.norm(unstable_direction)
    targeted = run_integer_hidden_chaos_controls(
        system,
        diagnostics["reference_clouds"],
        diagnostics["calibration"],
        equilibrium_names=("E0",),
        radii=tuple(float(value) for value in hidden["targeted_E0_unstable_direction_radii"]),
        directions=np.vstack((unstable_direction, -unstable_direction)),
        samples_per_radius=2,
        sampling_mode="sphere",
        random_seed=int(hidden["random_seed"]),
        t_burn=float(hidden["t_burn"]),
        t_keep=float(hidden["t_keep"]),
        sample_step=float(hidden["sample_step"]),
        rtol=float(hidden["rtol"]),
        atol=float(hidden["atol"]),
        max_step=float(hidden["max_step"]),
        div_threshold=div_threshold,
        equilibrium_tol=float(hidden["equilibrium_tol"]),
        equilibrium_tail_span_tol=float(hidden["equilibrium_tail_span_tol"]),
        max_cloud_points=int(hidden["reference_max_points"]),
    )
    declared_equilibria = tuple(system.equilibrium_points())
    main_summary = summarize_integer_hidden_chaos_controls(
        main,
        required_equilibrium_names=declared_equilibria,
        declared_equilibrium_names=declared_equilibria,
    )
    targeted_summary = summarize_integer_hidden_chaos_controls(
        targeted,
        required_equilibrium_names=("E0",),
        declared_equilibrium_names=declared_equilibria,
    )
    main_coverage = _probe_cell_coverage(
        main,
        equilibrium_names=declared_equilibria,
        radii=tuple(float(value) for value in hidden["radii"]),
        expected_per_cell=int(hidden["directions"]),
    )
    targeted_coverage = _probe_cell_coverage(
        targeted,
        equilibrium_names=("E0",),
        radii=tuple(float(value) for value in hidden["targeted_E0_unstable_direction_radii"]),
        expected_per_cell=2,
    )
    combined = {
        "main": main_summary,
        "targeted_E0_unstable_direction": {
            **targeted_summary,
            "unstable_eigenvalue": eigenvalues[unstable_index],
            "unstable_direction": unstable_direction,
        },
        "n_probes": len(main) + len(targeted),
        "target_hits": main_summary["target_hits"] + targeted_summary["target_hits"],
        "ambiguous": main_summary["ambiguous"] + targeted_summary["ambiguous"],
        "numerical_failures": main_summary["numerical_failures"] + targeted_summary["numerical_failures"],
        "coverage_by_equilibrium_radius": {
            "main": main_coverage,
            "targeted_E0_unstable_direction": targeted_coverage,
            "complete": bool(main_coverage["complete"] and targeted_coverage["complete"]),
        },
        "sampled_hiddenness_status": (
            "hidden_under_tested_neighborhoods"
            if main_summary["sampled_hiddenness_status"] == "hidden_under_tested_neighborhoods"
            and targeted_summary["sampled_hiddenness_status"] == "hidden_under_tested_neighborhoods"
            else "self_excited_under_tested_neighborhoods"
            if main_summary["target_hits"] + targeted_summary["target_hits"] > 0
            else "inconclusive"
        ),
        "finite_sample_only": True,
        "global_hiddenness_proved": False,
    }
    rows = [_probe_row(probe, contract="main_3x3xN") for probe in main]
    rows.extend(_probe_row(probe, contract="targeted_E0_unstable") for probe in targeted)
    _write_rows(output / "07_hiddenness_probes.csv", rows)
    _write_json(output / "07_hiddenness_summary.json", combined)
    initial_rows = []
    for row in rows:
        x0 = json.loads(row["x0"]) if isinstance(row["x0"], str) else np.asarray(row["x0"], dtype=float)
        initial_rows.append(
            {
                "contract": row["contract"],
                "sample_id": row["sample_id"],
                "equilibrium": row["equilibrium"],
                "radius": row["radius"],
                "direction_id": row["direction_id"],
                "y1": x0[0],
                "y2": x0[1],
                "y3": x0[2],
            }
        )
    _write_rows(output / "07_hiddenness_initial_conditions.csv", initial_rows)

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    labels = list(main_summary["by_equilibrium"])
    targeted_by_equilibrium = targeted_summary["by_equilibrium"]
    equilibrium_destinations = [
        main_summary["by_equilibrium"][label]["equilibrium_destinations"]
        + targeted_by_equilibrium.get(label, {}).get("equilibrium_destinations", 0)
        for label in labels
    ]
    target_hits = [
        main_summary["by_equilibrium"][label]["target_hits"]
        + targeted_by_equilibrium.get(label, {}).get("target_hits", 0)
        for label in labels
    ]
    x_positions = np.arange(len(labels))
    axis.bar(x_positions - 0.18, equilibrium_destinations, 0.36, label="equilibrium destinations")
    axis.bar(x_positions + 0.18, target_hits, 0.36, label="target contacts")
    axis.set_xticks(x_positions, labels)
    axis.set(ylabel="probe count")
    axis.legend()
    fig.tight_layout()
    _save_figure_pair(fig, output / "figures" / "07_hiddenness_outcomes.png")
    plt.close(fig)
    return {"main": main, "targeted": targeted, "summary": combined}


def run_gate(
    config: Mapping[str, Any],
    output: Path,
    search: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    hiddenness: Mapping[str, Any],
    *,
    run_id: str,
    scientific_source_snapshot: Mapping[str, Any] | None = None,
    allow_scientific_promotion: bool = True,
) -> dict[str, Any]:
    source_snapshot = dict(scientific_source_snapshot or _scientific_source_snapshot())
    system = search["selected_system"]
    direct_seed = search["direct_seed"]
    candidate_initial_state = np.asarray(search["selected_state"], dtype=float)
    trajectory = diagnostics["trajectory"]
    boundedness = diagnostics["boundedness"]
    stability = diagnostics["stability"]
    hidden_summary = hiddenness["summary"]
    return_k = [float(row["K"]) for row in diagnostics["zero_one_return_results"]]
    main_probes = tuple(hiddenness["main"])
    targeted_probes = tuple(hiddenness["targeted"])
    required_radii = tuple(float(value) for value in config["hiddenness"]["radii"])
    completed_main_probes = tuple(probe for probe in main_probes if probe.status == "ok")
    tested_radii = sorted({float(probe.radius) for probe in completed_main_probes})
    expected_main_count = (
        len(system.equilibrium_points())
        * len(required_radii)
        * int(config["hiddenness"]["directions"])
    )
    expected_targeted_count = 2 * len(config["hiddenness"]["targeted_E0_unstable_direction_radii"])
    main_contract_complete = bool(
        len(main_probes) == expected_main_count
        and len(completed_main_probes) == expected_main_count
        and hidden_summary["main"]["tested_all_declared_equilibria"]
        and hidden_summary["coverage_by_equilibrium_radius"]["main"]["complete"]
    )
    targeted_contract_complete = bool(
        len(targeted_probes) == expected_targeted_count
        and all(probe.status == "ok" for probe in targeted_probes)
        and hidden_summary["coverage_by_equilibrium_radius"]["targeted_E0_unstable_direction"]["complete"]
    )
    strict_meta = diagnostics["lyapunov"].metadata
    control_meta = diagnostics["lyapunov_control"].metadata
    positive_tol = float(config["candidate_gate"]["lyapunov_positive_tol"])
    strict_positive = bool(
        diagnostics["lyapunov"].status == "ok"
        and diagnostics["lyapunov"].exponents[0] > positive_tol
    )
    control_positive = bool(
        diagnostics["lyapunov_control"].status == "ok"
        and diagnostics["lyapunov_control"].exponents[0] > positive_tol
    )
    tested_h = bool(
        strict_meta.get("max_step") != control_meta.get("max_step")
        and strict_meta.get("rtol") != control_meta.get("rtol")
    )
    tested_t_final = bool(
        strict_meta.get("t_accumulate_requested") != control_meta.get("t_accumulate_requested")
        and strict_meta.get("t_accumulate_completed") == strict_meta.get("t_accumulate_requested")
        and control_meta.get("t_accumulate_completed") == control_meta.get("t_accumulate_requested")
    )
    tested_integrator = bool(
        strict_meta.get("solver_method") == "DOP853"
        and diagnostics["efork_status"] == "ok"
    )
    integrator_match = bool(
        diagnostics["efork_match"]["classification"]
        == "same_attractor_under_calibrated_cloud_test"
    )
    robustness_consistent = bool(
        strict_positive
        and control_positive
        and tested_h
        and tested_t_final
        and tested_integrator
        and integrator_match
    )
    metadata = collect_run_metadata(
        run_id=str(run_id),
        workflow="integer_hidden_chaos_search",
        system=system.name,
        q=1.0,
        h=float(config["candidate_trajectory"]["sample_step"]),
        t_final=float(config["candidate_trajectory"]["t_final"]),
        t_burn=float(config["candidate_trajectory"]["t_burn"]),
        memory_mode="not_applicable",
        integrator_name="DOP853",
        integrator_backend="python",
        caputo=False,
        is_full_caputo=False,
        parameters=system.parameters,
        lure=collect_lure_metadata(
            system.lure,
            transfer_convention="c^T (P-s I)^(-1) b; direct polynomial roots",
            harmonic_condition="Im W(i omega)=0 and Re W(i omega)=-1/k",
        ),
        seed={
            "candidate_id": "mavpd-chaos-continuation-endpoint",
            "family": "continuation_endpoint_from_theoretical_lure_seed",
            "x0": candidate_initial_state,
            "source": "selected_parameter_continuation_endpoint",
            "parameters": {
                "source_branch_index": int(direct_seed.branch_index),
                "theoretical_harmonic_seed": direct_seed.seed,
                "omega0": direct_seed.omega,
                "k": direct_seed.gain,
                "a0": direct_seed.amplitude,
            },
        },
        random_seed=int(config["hiddenness"]["random_seed"]),
        random_seed_policy="fixed_reproducible",
        continuation=ContinuationMetadata(
            used=True,
            eta_path=tuple(float(value) for value in search["lambda_values"]),
            continuation_mode="integer",
            memory_window_propagated=None,
            final_eta=1.0,
        ),
        provenance={
            "source_doi": MAVPD_2023_DOI,
            "source_scope": "published_model_equations_only",
            "candidate_parameter_set": "gamma_selected_at_declared_xi_continuation_endpoint",
            "candidate_parameter_set_published": False,
            "scientific_source_snapshot": source_snapshot,
            "frequency_grid_used_for_search": False,
            "alternative_triggered_after_direct_chaos_screen_failure": True,
        },
        extra={"candidate_parameters": dict(system.parameters)},
        tolerances=config["candidate_gate"],
    )
    post_states = trajectory[trajectory[:, 0] >= float(config["candidate_trajectory"]["t_burn"]), 1:]
    evidence = {
        "run_metadata": metadata_to_jsonable(metadata),
        "equilibria": {
            "all_found": len(stability) == 3,
            "max_residual": max(float(row["rhs_residual"]) for row in stability),
        },
        "matignon": {"all_classified": all(row["stability"] != "marginal_or_inconclusive" for row in stability), "q": 1.0},
        "seed": {
            "localized": True,
            "method": "continuation",
            "source": "candidate_initial_state_is_selected_parameter_continuation_endpoint",
            "theoretical_seed_source": "direct_integer_transfer_from_declared_equations",
        },
        "continuation": {
            "used": True,
            "eta_path": list(search["lambda_values"]),
            "continuation_mode": "integer",
            "memory_window_propagated": None,
            "final_eta": 1.0,
        },
        "trajectory": {
            "bounded": boundedness["boundedness_status"] == "bounded_candidate",
            "nontrivial": float(np.max(np.var(post_states, axis=0))) > 1.0e-8,
            "finite_fraction": boundedness["finite_fraction"],
            "post_transient_rows": boundedness["post_transient_rows"],
            "minimum_post_transient_length": 1000,
        },
        "robustness": {
            "tested_h": tested_h,
            "tested_memory": False,
            "memory_applicable": bool(config["robustness"]["memory_applicable"]),
            "tested_t_final": tested_t_final,
            "tested_integrator": tested_integrator,
            "integrator_match": integrator_match,
            "consistent": robustness_consistent,
        },
        "hiddenness": {
            "tested_all_equilibria": hidden_summary["main"]["tested_all_declared_equilibria"],
            "tested_radii": tested_radii,
            "required_radii": list(required_radii),
            "target_hits_from_equilibria": hidden_summary["target_hits"],
            "basin_intersection_detected": hidden_summary["target_hits"] > 0,
            "basin_controls_complete": bool(
                main_contract_complete
                and targeted_contract_complete
                and hidden_summary["ambiguous"] == 0
                and hidden_summary["sampled_hiddenness_status"] != "inconclusive"
            ),
            "coverage_by_equilibrium_radius_complete": hidden_summary[
                "coverage_by_equilibrium_radius"
            ]["complete"],
            "numerical_failures": hidden_summary["numerical_failures"],
        },
        "lyapunov": {
            "exponents": diagnostics["lyapunov"].exponents,
            "method_status": (
                "internal_controls_passed"
                if robustness_consistent
                else "internal_controls_incomplete"
            ),
        },
        "zero_one": {
            "K": float(np.median(return_k)),
            "state": (
                "zero_one_chaotic_candidate"
                if np.median(return_k)
                >= float(config["candidate_gate"]["zero_one_chaos_threshold"])
                else "zero_one_inconclusive"
            ),
            "gate_applicable": True,
            "series": "Poincare return sequence",
        },
        "spectrum": {
            "state_global": diagnostics["spectrum"]["state_global"],
            "gate_applicable": False,
            "reason": "supporting FFT only; line dominance is not a regularity proof",
        },
        "poincare": {**diagnostics["poincare"], "gate_applicable": True},
        "tolerances": dict(config["candidate_gate"]),
    }
    gate = evaluate_candidate_gate(evidence)
    if not allow_scientific_promotion:
        gate = {
            **gate,
            "quick_smoke_only": True,
            "scientific_promotion_allowed": False,
            "promotion_allowed": False,
            "hiddenness_promotion_allowed": False,
            "chaotic_hidden_promotion_allowed": False,
            "hidden_chaos_status": "quick_smoke_not_promotable",
        }
    _write_json(output / "08_robustness_matrix.json", evidence["robustness"])
    _write_json(output / "09_candidate_gate.json", {"gate": gate, "evidence": evidence})
    if allow_scientific_promotion and not gate["chaotic_hidden_promotion_allowed"]:
        raise RuntimeError(f"joint hidden-chaos gate did not pass: {gate}")
    return gate


def run_all(*, quick: bool = False, output_override: str | Path | None = None) -> dict[str, Any]:
    config = load_config(quick=quick)
    output = _output_dir(config, output_override)
    _assert_isolated_working_output(config, output)
    _assert_fresh_run_output(output)
    output.mkdir(parents=True, exist_ok=True)
    source_snapshot = _scientific_source_snapshot()
    run_status = _new_run_status(output, config, source_snapshot)
    timings = []
    started = time.perf_counter()

    phase_start = time.perf_counter()
    contract = run_contract(config, output, scientific_source_snapshot=source_snapshot)
    _assert_scientific_sources_unchanged(source_snapshot, phase="contract")
    timings.append({"phase": "contract", "seconds": time.perf_counter() - phase_start, "timing_source": "perf_counter"})
    _write_rows(output / "phase_timings.csv", timings)
    _record_run_phase(
        output,
        run_status,
        "contract",
        (
            "00_system_contract.json",
            "figures/00_nyquist_direct_seed.png",
            "figures/00_nyquist_direct_seed.pdf",
            "phase_timings.csv",
        ),
    )
    phase_start = time.perf_counter()
    search = run_search(
        config,
        output,
        contract,
        scientific_source_snapshot=source_snapshot,
    )
    _assert_scientific_sources_unchanged(source_snapshot, phase="search")
    timings.append({"phase": "search", "seconds": time.perf_counter() - phase_start, "timing_source": "perf_counter"})
    _write_rows(output / "phase_timings.csv", timings)
    _record_run_phase(
        output,
        run_status,
        "search",
        (
            "01_direct_seed_and_lambda_continuation.json",
            "02_parameter_continuation.csv",
            "03_candidate_screening.csv",
            "03_candidate_screening_probes.csv",
            "03_candidate_screening_contract.json",
            "03_candidate_selection.json",
            "phase_timings.csv",
        ),
    )
    phase_start = time.perf_counter()
    diagnostics = run_diagnostics(config, output, search)
    _assert_scientific_sources_unchanged(source_snapshot, phase="diagnostics")
    timings.append({"phase": "diagnostics", "seconds": time.perf_counter() - phase_start, "timing_source": "perf_counter"})
    _write_rows(output / "phase_timings.csv", timings)
    _record_run_phase(
        output,
        run_status,
        "diagnostics",
        (
            "04_candidate_trajectory.csv",
            "05_chaos_diagnostics.json",
            "05_lyapunov_convergence.csv",
            "05_poincare_section.csv",
            "05_zero_one_stride_sensitivity.csv",
            "05_zero_one_return_map.json",
            "05_normalized_fft_power.csv",
            "06_equilibrium_stability.json",
            "figures/03_continuation_screen.png",
            "figures/03_continuation_screen.pdf",
            "figures/04_candidate_phase_portraits.png",
            "figures/04_candidate_phase_portraits.pdf",
            "figures/04_candidate_time_series.png",
            "figures/04_candidate_time_series.pdf",
            "figures/05_lyapunov_convergence.png",
            "figures/05_lyapunov_convergence.pdf",
            "figures/05_poincare_section.png",
            "figures/05_poincare_section.pdf",
            "figures/05_normalized_fft_power.png",
            "figures/05_normalized_fft_power.pdf",
            "phase_timings.csv",
        ),
    )
    phase_start = time.perf_counter()
    hiddenness = run_hiddenness(config, output, search, diagnostics)
    _assert_scientific_sources_unchanged(source_snapshot, phase="hiddenness")
    timings.append({"phase": "hiddenness", "seconds": time.perf_counter() - phase_start, "timing_source": "perf_counter"})
    _write_rows(output / "phase_timings.csv", timings)
    _record_run_phase(
        output,
        run_status,
        "hiddenness",
        (
            "07_hiddenness_probes.csv",
            "07_hiddenness_summary.json",
            "07_hiddenness_initial_conditions.csv",
            "figures/07_hiddenness_outcomes.png",
            "figures/07_hiddenness_outcomes.pdf",
            "phase_timings.csv",
        ),
    )
    phase_start = time.perf_counter()
    gate = run_gate(
        config,
        output,
        search,
        diagnostics,
        hiddenness,
        run_id=run_status["run_id"],
        scientific_source_snapshot=source_snapshot,
        allow_scientific_promotion=not quick,
    )
    _assert_scientific_sources_unchanged(source_snapshot, phase="candidate gate")
    figure_manifest = _finalize_figure_manifest(
        output,
        config,
        source_snapshot,
        search["selected_system"].parameters,
        run_id=run_status["run_id"],
        promote=not quick,
    )
    _assert_scientific_sources_unchanged(source_snapshot, phase="figure finalization")
    timings.append({"phase": "candidate_gate", "seconds": time.perf_counter() - phase_start, "timing_source": "perf_counter"})
    timings.append({"phase": "total", "seconds": time.perf_counter() - started, "timing_source": "perf_counter"})
    _write_rows(output / "phase_timings.csv", timings)
    _record_run_phase(
        output,
        run_status,
        "candidate_gate_and_figures",
        (
            "08_robustness_matrix.json",
            "09_candidate_gate.json",
            "figures/figure_manifest.json",
            "phase_timings.csv",
        ),
    )
    manifest = {
        "case_id": config["case_id"],
        "run_id": run_status["run_id"],
        "config_sha256": run_status["config_sha256"],
        "runtime_environment": run_status["runtime_environment"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "quick_mode": quick,
        "candidate_parameters": dict(search["selected_system"].parameters),
        "candidate_parameter_provenance": "gamma_selected_at_declared_xi_continuation_endpoint_not_a_published_parameter_tuple",
        "xi_endpoint_provenance": "declared_local_continuation_endpoint_not_selected_by_candidate_screen",
        "alternative_source_branch_index": int(search["direct_seed"].branch_index),
        "alternative_source_branch_selection_rule": config["alternative_parameter_continuation"]["source_branch_selection_rule"],
        "scientific_source_snapshot": source_snapshot,
        "hopf_boundary": search["gamma_h"],
        "selected_hopf_offset": search["selected"]["hopf_offset"],
        "lyapunov_exponents": diagnostics["lyapunov"].exponents,
        "hiddenness": hiddenness["summary"],
        "candidate_gate": gate,
        "figures": figure_manifest,
        "frequency_grid_used_for_search": False,
        "global_proof_claimed": False,
        "timings": timings,
    }
    _write_json(output / "run_manifest.json", manifest)
    _record_run_phase(output, run_status, "manifest", ("run_manifest.json", "phase_timings.csv"))
    run_status["status"] = "complete"
    run_status["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output / "run_status.json", run_status)
    if not quick:
        figure_manifest, manifest = _promote_completed_figures(
            output,
            figure_manifest,
            manifest,
            run_status,
            source_snapshot,
        )
    return manifest


def _adaptive_result_from_json(payload: Mapping[str, Any]) -> AdaptiveLyapunovResult:
    exponents = np.asarray(payload["exponents"], dtype=float)
    final_state = np.asarray(payload.get("final_state", np.zeros_like(exponents)), dtype=float)
    return AdaptiveLyapunovResult(
        exponents=exponents,
        times=np.empty(0, dtype=float),
        convergence=np.empty((0, len(exponents)), dtype=float),
        status=str(payload["status"]),
        final_state=final_state,
        accumulated_time=float(payload.get("t_accumulate", 0.0)),
        metadata=dict(payload.get("metadata", {})),
    )


def resume_validated_candidate(
    *,
    output_override: str | Path | None = None,
    launcher_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Resume hiddenness after search/diagnostic artifacts passed validation."""

    if launcher_timeout_seconds is not None and launcher_timeout_seconds <= 0.0:
        raise ValueError("launcher_timeout_seconds must be positive when provided")
    config = load_config(quick=False)
    output = _output_dir(config, output_override)
    _assert_isolated_working_output(config, output)
    source_snapshot = _scientific_source_snapshot()
    status_path = output / "run_status.json"
    if not status_path.is_file():
        raise FileNotFoundError("cannot resume without run_status.json from the interrupted full run")
    run_status = json.loads(status_path.read_text(encoding="utf-8"))
    if run_status.get("status") != "in_progress" or run_status.get("quick_mode") is not False:
        raise RuntimeError("resume requires an in-progress, non-quick run status")
    if run_status.get("config_sha256") != _config_sha256(config):
        raise RuntimeError("cannot resume: resolved configuration digest differs from the interrupted run")
    if run_status.get("runtime_environment") != _runtime_environment():
        raise RuntimeError("cannot resume: Python/NumPy/SciPy runtime differs from the interrupted run")
    recorded_source = run_status.get("scientific_source_snapshot", {})
    if recorded_source.get("bundle_sha256") != source_snapshot["bundle_sha256"]:
        raise RuntimeError("cannot resume: scientific source bundle differs from the interrupted run")
    if "diagnostics" not in run_status.get("completed_phases", []):
        raise RuntimeError("cannot resume before the diagnostics phase was recorded complete")
    resumable_artifacts = (
        "00_system_contract.json",
        "01_direct_seed_and_lambda_continuation.json",
        "02_parameter_continuation.csv",
        "03_candidate_screening.csv",
        "03_candidate_screening_probes.csv",
        "03_candidate_screening_contract.json",
        "03_candidate_selection.json",
        "04_candidate_trajectory.csv",
        "05_chaos_diagnostics.json",
        "05_lyapunov_convergence.csv",
        "05_poincare_section.csv",
        "05_zero_one_stride_sensitivity.csv",
        "05_zero_one_return_map.json",
        "05_normalized_fft_power.csv",
        "06_equilibrium_stability.json",
        "phase_timings.csv",
    )
    resumable_figure_ids = tuple(
        figure_id for figure_id in FIGURE_DATA_SOURCES if not figure_id.startswith("07_")
    )
    resumable_artifacts += tuple(
        f"figures/{figure_id}.{suffix}"
        for figure_id in resumable_figure_ids
        for suffix in ("png", "pdf")
    )
    _verify_recorded_artifacts(output, run_status, resumable_artifacts)
    contract_json = json.loads((output / "00_system_contract.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "03_candidate_selection.json").read_text(encoding="utf-8"))
    for artifact_name, payload in (("00_system_contract.json", contract_json), ("03_candidate_selection.json", selection)):
        recorded_snapshot = payload.get("scientific_source_snapshot", {})
        if recorded_snapshot.get("bundle_sha256") != source_snapshot["bundle_sha256"]:
            raise RuntimeError(
                f"cannot resume: {artifact_name} was produced by a different scientific source bundle"
            )
    diagnostic_json = json.loads((output / "05_chaos_diagnostics.json").read_text(encoding="utf-8"))
    stability = json.loads((output / "06_equilibrium_stability.json").read_text(encoding="utf-8"))
    trajectory_values = np.genfromtxt(
        output / "04_candidate_trajectory.csv", delimiter=",", names=True, dtype=float
    )
    trajectory = np.column_stack(
        tuple(trajectory_values[name] for name in ("time", "y1", "y2", "y3"))
    )
    selected_state = np.asarray(selection["selected_candidate_initial_state"], dtype=float)
    if not np.allclose(trajectory[0, 1:], selected_state, rtol=0.0, atol=1.0e-13):
        raise RuntimeError("cannot resume: trajectory initial state differs from selected continuation endpoint")
    selected_path_rows = []
    with (output / "02_parameter_continuation.csv").open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("stage") == "gamma"
                and np.isclose(float(row["xi"]), float(config["alternative_parameter_continuation"]["xi_stop"]), rtol=0.0, atol=1.0e-12)
                and np.isclose(float(row["gamma"]), float(selection["selected"]["gamma"]), rtol=0.0, atol=1.0e-12)
            ):
                selected_path_rows.append(row)
    if len(selected_path_rows) != 1:
        raise RuntimeError("cannot resume: selected parameter node is not unique in 02_parameter_continuation.csv")
    path_state = np.asarray(json.loads(selected_path_rows[0]["x_out"]), dtype=float)
    if not np.allclose(path_state, selected_state, rtol=0.0, atol=1.0e-13):
        raise RuntimeError("cannot resume: selected state differs from its parameter-continuation node")
    if not (
        np.isclose(float(trajectory[0, 0]), 0.0, rtol=0.0, atol=1.0e-15)
        and np.isclose(
            float(trajectory[-1, 0]),
            float(config["candidate_trajectory"]["t_final"]),
            rtol=0.0,
            atol=1.0e-10,
        )
    ):
        raise RuntimeError("cannot resume: candidate trajectory time span differs from the full contract")
    parameters = dict(config["system"]["base_parameters"])
    parameters.update(
        {
            "xi": float(config["alternative_parameter_continuation"]["xi_stop"]),
            "gamma": float(selection["selected"]["gamma"]),
        }
    )
    system = mavpd_2023_system(parameters)
    windows = [tuple(map(float, item)) for item in config["candidate_trajectory"]["reference_windows"]]
    reference_clouds = [
        trajectory[(trajectory[:, 0] >= start) & (trajectory[:, 0] < stop), 1:]
        for start, stop in windows
    ]
    negative_controls = [
        np.repeat(value[None, :], int(config["hiddenness"]["negative_control_points"]), axis=0)
        for value in system.equilibrium_points().values()
    ]
    hidden_cfg = config["hiddenness"]
    calibration = calibrate_attractor_reference(
        reference_clouds,
        negative_control_clouds=negative_controls,
        max_points=int(hidden_cfg["reference_max_points"]),
        safety_factor=float(hidden_cfg["reference_safety_factor"]),
        ambiguity_fraction=float(hidden_cfg["reference_ambiguity_fraction"]),
    )
    recorded_calibration = diagnostic_json["reference_calibration"]
    if not np.isclose(
        calibration.acceptance_threshold,
        float(recorded_calibration["acceptance_threshold"]),
        rtol=1.0e-12,
        atol=1.0e-15,
    ):
        raise RuntimeError("recomputed reference calibration differs from the validated diagnostic")
    burn_time = float(config["candidate_trajectory"]["t_burn"])
    boundedness = compute_boundedness_metrics(
        trajectory[:, 0],
        trajectory[:, 1:],
        burn_time,
        divergence_radius=float(config["numerical_safety"]["divergence_norm"]),
    )
    diagnostics = {
        "trajectory": trajectory,
        "reference_clouds": reference_clouds,
        "calibration": calibration,
        "boundedness": boundedness,
        "lyapunov": _adaptive_result_from_json(diagnostic_json["lyapunov"]),
        "lyapunov_control": _adaptive_result_from_json(diagnostic_json["lyapunov_control"]),
        "zero_one_return_results": diagnostic_json["zero_one"]["return_map_results"],
        "spectrum": {"state_global": diagnostic_json["spectrum"]["state_global"]},
        "poincare": diagnostic_json["poincare"],
        "stability": stability,
        "efork_status": diagnostic_json["efork_crosscheck"]["status"],
        "efork_match": diagnostic_json["efork_crosscheck"]["target_match"],
    }
    base_system = mavpd_2023_system(config["system"]["base_parameters"])
    source_branch_index = int(selection["alternative_source_branch_index"])
    direct_seed = integer_lure_seed(
        base_system,
        branch_index=source_branch_index,
        theta=float(config["primary_route"]["phase"]),
        wmin=float(config["primary_route"]["admissible_omega_min"]),
        wmax=float(config["primary_route"]["admissible_omega_max"]),
    )
    lambda_values = ContinuationPlan.uniform(
        int(config["primary_route"]["lambda_nodes"]), internal_parameter="epsilon"
    ).lambda_values
    search = {
        "selected_system": system,
        "selected": selection["selected"],
        "gamma_h": float(selection["hopf_boundary"]),
        "direct_seed": direct_seed,
        "selected_state": selected_state,
        "lambda_values": lambda_values,
    }
    timing_rows = _load_persisted_pre_resume_timings(output / "phase_timings.csv")
    phase_start = time.perf_counter()
    hiddenness = run_hiddenness(config, output, search, diagnostics)
    _assert_scientific_sources_unchanged(source_snapshot, phase="resumed hiddenness")
    timing_rows.append(
        {"phase": "hiddenness_resumed", "seconds": time.perf_counter() - phase_start, "timing_source": "perf_counter"}
    )
    _write_rows(output / "phase_timings.csv", timing_rows)
    _record_run_phase(
        output,
        run_status,
        "hiddenness_resumed",
        (
            "07_hiddenness_probes.csv",
            "07_hiddenness_summary.json",
            "07_hiddenness_initial_conditions.csv",
            "figures/07_hiddenness_outcomes.png",
            "figures/07_hiddenness_outcomes.pdf",
            "phase_timings.csv",
        ),
    )
    phase_start = time.perf_counter()
    gate = run_gate(
        config,
        output,
        search,
        diagnostics,
        hiddenness,
        run_id=run_status["run_id"],
        scientific_source_snapshot=source_snapshot,
    )
    _assert_scientific_sources_unchanged(source_snapshot, phase="resumed candidate gate")
    figure_manifest = _finalize_figure_manifest(
        output,
        config,
        source_snapshot,
        system.parameters,
        run_id=run_status["run_id"],
        promote=True,
    )
    _assert_scientific_sources_unchanged(source_snapshot, phase="resumed figure finalization")
    timing_rows.append(
        {"phase": "candidate_gate_resumed", "seconds": time.perf_counter() - phase_start, "timing_source": "perf_counter"}
    )
    timing_rows.append(
        {
            "phase": "total_recorded_phase_seconds",
            "seconds": sum(float(row["seconds"]) for row in timing_rows),
            "timing_source": "sum_of_persisted_and_resumed_phase_timers_not_wall_clock",
        }
    )
    _write_rows(output / "phase_timings.csv", timing_rows)
    _record_run_phase(
        output,
        run_status,
        "candidate_gate_and_figures_resumed",
        (
            "08_robustness_matrix.json",
            "09_candidate_gate.json",
            "figures/figure_manifest.json",
            "phase_timings.csv",
        ),
    )
    manifest = {
        "case_id": config["case_id"],
        "run_id": run_status["run_id"],
        "config_sha256": run_status["config_sha256"],
        "runtime_environment": run_status["runtime_environment"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resumed_from_persisted_phase_record": True,
        "launcher_timeout_seconds": launcher_timeout_seconds,
        "candidate_parameters": dict(system.parameters),
        "candidate_parameter_provenance": "gamma_selected_at_declared_xi_continuation_endpoint_not_a_published_parameter_tuple",
        "xi_endpoint_provenance": "declared_local_continuation_endpoint_not_selected_by_candidate_screen",
        "alternative_source_branch_index": source_branch_index,
        "alternative_source_branch_selection_rule": selection["alternative_source_branch_selection_rule"],
        "scientific_source_snapshot": source_snapshot,
        "hopf_boundary": search["gamma_h"],
        "selected_hopf_offset": search["selected"]["hopf_offset"],
        "lyapunov_exponents": diagnostics["lyapunov"].exponents,
        "hiddenness": hiddenness["summary"],
        "candidate_gate": gate,
        "figures": figure_manifest,
        "frequency_grid_used_for_search": False,
        "global_proof_claimed": False,
        "timings": timing_rows,
    }
    _write_json(output / "run_manifest.json", manifest)
    _record_run_phase(output, run_status, "manifest_resumed", ("run_manifest.json", "phase_timings.csv"))
    run_status["status"] = "complete"
    run_status["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output / "run_status.json", run_status)
    figure_manifest, manifest = _promote_completed_figures(
        output,
        figure_manifest,
        manifest,
        run_status,
        source_snapshot,
    )
    return manifest


def run_contract_only(*, quick: bool = True, output_override: str | Path | None = None) -> dict[str, Any]:
    config = load_config(quick=quick)
    output = (
        Path(output_override).resolve()
        if output_override is not None
        else ROOT
        / "tmp"
        / ("mavpd_integer_hidden_chaos_contract_quick" if quick else "mavpd_integer_hidden_chaos_contract_full")
    )
    _assert_isolated_working_output(config, output)
    output.mkdir(parents=True, exist_ok=True)
    source_snapshot = _scientific_source_snapshot()
    result = run_contract(config, output, scientific_source_snapshot=source_snapshot)
    _assert_scientific_sources_unchanged(source_snapshot, phase="contract-only run")
    return result["payload"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="short smoke configuration; not publication evidence")
    parser.add_argument("--contract-only", action="store_true", help="derive and validate equations/seeds only")
    parser.add_argument(
        "--resume-validated-candidate",
        action="store_true",
        help="resume final hiddenness from completed search and diagnostic artifacts",
    )
    parser.add_argument(
        "--launcher-timeout-seconds",
        type=float,
        default=None,
        help="optional externally measured launcher timeout, recorded only in resume metadata",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="explicit output directory")
    args = parser.parse_args(argv)
    if args.quick and args.resume_validated_candidate:
        parser.error("--quick cannot be combined with --resume-validated-candidate")
    if args.resume_validated_candidate:
        manifest = resume_validated_candidate(
            output_override=args.output_dir,
            launcher_timeout_seconds=args.launcher_timeout_seconds,
        )
        print(json.dumps(_jsonable(manifest), indent=2))
    elif args.contract_only:
        payload = run_contract_only(quick=args.quick, output_override=args.output_dir)
        print(json.dumps(_jsonable({"status": "ok", "contract": payload["case_id"]}), indent=2))
    else:
        manifest = run_all(quick=args.quick, output_override=args.output_dir)
        print(json.dumps(_jsonable(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
