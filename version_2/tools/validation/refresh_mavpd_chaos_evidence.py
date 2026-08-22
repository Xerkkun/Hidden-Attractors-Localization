#!/usr/bin/env python3
"""Refresh the MAVPD hidden-chaos report evidence from one validated full run.

This script deliberately uses only the Python standard library.  It does not
run simulations or modify scientific evidence: it validates an already
completed run, then copies report assets and derives LaTeX fragments from the
recorded JSON/CSV fields.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence
from uuid import uuid4


DEFAULT_REPORT_DATE = "3 de agosto de 2026"
CASE_ID = "mavpd_integer_hidden_chaos"

RUN_ID_PATTERN = re.compile(
    rf"^{CASE_ID}-full-(?P<timestamp>[0-9]{{8}}T[0-9]{{12}}Z)-"
    r"(?P<source>[0-9a-f]{12})-(?P<nonce>[0-9a-f]{12})$"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DIRECT_COMPLETED_PHASES = (
    "contract",
    "search",
    "diagnostics",
    "hiddenness",
    "candidate_gate_and_figures",
    "manifest",
    "global_figure_promotion",
)
DIRECT_TIMING_PHASES = (
    "contract",
    "search",
    "diagnostics",
    "hiddenness",
    "candidate_gate",
    "total",
)
ALL_FIGURE_IDS = (
    "00_nyquist_direct_seed",
    "03_continuation_screen",
    "04_candidate_phase_portraits",
    "04_candidate_time_series",
    "05_lyapunov_convergence",
    "05_poincare_section",
    "05_normalized_fft_power",
    "07_hiddenness_outcomes",
)
EXPECTED_SCREEN_OFFSETS = (
    Decimal("0.002"),
    Decimal("0.003"),
    Decimal("0.005"),
    Decimal("0.008"),
    Decimal("0.010"),
    Decimal("0.012"),
    Decimal("0.015"),
)
EXPECTED_CANDIDATE_PARAMETERS = {
    "xi": Decimal("2.85"),
    "delta": Decimal("100"),
    "rho": Decimal("200"),
}

FIGURE_ASSETS: tuple[tuple[str, str], ...] = (
    ("03_continuation_screen", "mavpd_chaos_screen"),
    ("04_candidate_phase_portraits", "mavpd_chaos_phase_portraits"),
    ("04_candidate_time_series", "mavpd_chaos_time_series"),
    ("05_lyapunov_convergence", "mavpd_chaos_lyapunov"),
    ("05_poincare_section", "mavpd_chaos_poincare"),
    ("05_normalized_fft_power", "mavpd_chaos_fft"),
    ("07_hiddenness_outcomes", "mavpd_chaos_hiddenness"),
)

REQUIRED_LEDGER_PATHS: tuple[str, ...] = (
    "00_system_contract.json",
    "03_candidate_screening.csv",
    "03_candidate_screening_probes.csv",
    "03_candidate_screening_contract.json",
    "03_candidate_selection.json",
    "05_chaos_diagnostics.json",
    "05_zero_one_stride_sensitivity.csv",
    "05_zero_one_return_map.json",
    "07_hiddenness_probes.csv",
    "07_hiddenness_summary.json",
    "08_robustness_matrix.json",
    "09_candidate_gate.json",
    "phase_timings.csv",
    "run_manifest.json",
    "figures/figure_manifest.json",
    "figures/global_promotion_receipt.json",
) + tuple(
    f"figures/{figure_id}.{extension}"
    for figure_id in ALL_FIGURE_IDS
    for extension in ("png", "pdf")
)


class EvidenceRefreshError(RuntimeError):
    """Raised when a run cannot be used as report evidence."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _decode_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_float=Decimal,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise EvidenceRefreshError(f"cannot decode valid JSON for {label}: {exc}") from exc


def _read_json(path: Path) -> Any:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise EvidenceRefreshError(f"cannot read valid JSON from {path}: {exc}") from exc
    return _decode_json(payload, str(path))


def _decode_csv(payload: bytes, label: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceRefreshError(f"cannot decode UTF-8 CSV for {label}: {exc}") from exc
    reader = csv.DictReader(text.splitlines())
    if not reader.fieldnames:
        raise EvidenceRefreshError(f"CSV has no header: {label}")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise EvidenceRefreshError(f"CSV has duplicate columns: {label}")
    rows = [dict(row) for row in reader]
    if not rows:
        raise EvidenceRefreshError(f"CSV has no data rows: {label}")
    if any(value is None for row in rows for value in row.values()):
        raise EvidenceRefreshError(f"CSV contains a short or malformed row: {label}")
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise EvidenceRefreshError(f"cannot read CSV from {path}: {exc}") from exc
    return _decode_csv(payload, str(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise EvidenceRefreshError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceRefreshError(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EvidenceRefreshError(f"{label} must be a JSON array")
    return value


def _path_get(value: Any, path: str) -> Any:
    current = value
    for component in path.split("."):
        current_map = _mapping(current, path)
        if component not in current_map:
            raise EvidenceRefreshError(f"missing required field {path}")
        current = current_map[component]
    return current


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceRefreshError(f"{label} must be a non-empty string")
    return value


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise EvidenceRefreshError(f"{label} must be numeric, not boolean")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:
        raise EvidenceRefreshError(f"{label} is not numeric: {value!r}") from exc
    if not result.is_finite():
        raise EvidenceRefreshError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str) -> int:
    number = _decimal(value, label)
    if number != number.to_integral_value():
        raise EvidenceRefreshError(f"{label} must be an integer")
    result = int(number)
    if result < 0:
        raise EvidenceRefreshError(f"{label} must be non-negative")
    return result


def _number_text(value: Any, label: str) -> str:
    number = _decimal(value, label)
    if number == 0:
        return "0"
    return format(number, "f")


def _boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise EvidenceRefreshError(f"{label} must be exactly true or false")


def _close_decimal(
    left: Any,
    right: Any,
    label: str,
    *,
    atol: Decimal = Decimal("1e-12"),
    rtol: Decimal = Decimal("1e-10"),
) -> None:
    left_number = _decimal(left, f"{label} left value")
    right_number = _decimal(right, f"{label} right value")
    tolerance = atol + rtol * max(abs(left_number), abs(right_number))
    if abs(left_number - right_number) > tolerance:
        raise EvidenceRefreshError(
            f"inconsistent {label}: expected {left_number}, found {right_number}"
        )


def _required_sha256(value: Any, label: str) -> str:
    text = _required_string(value, label)
    if not SHA256_PATTERN.fullmatch(text):
        raise EvidenceRefreshError(f"{label} is not a lowercase SHA-256")
    return text


def _required_run_id(value: Any, source_bundle: str) -> str:
    run_id = _required_string(value, "run_status.run_id")
    match = RUN_ID_PATTERN.fullmatch(run_id)
    if match is None:
        raise EvidenceRefreshError("run_status.run_id does not match the full-run identifier contract")
    if match.group("source") != source_bundle[:12]:
        raise EvidenceRefreshError("run_id source prefix differs from the scientific-source bundle")
    return run_id


def _validated_report_date(value: Any) -> str:
    text = _required_string(value, "report date")
    if any(character in text for character in "\r\n"):
        raise EvidenceRefreshError("report date must be a single line")
    if len(text) > 120:
        raise EvidenceRefreshError("report date is unreasonably long")
    return text


def _same(expected: Any, actual: Any, label: str) -> None:
    if expected != actual:
        raise EvidenceRefreshError(
            f"inconsistent {label}: expected {expected!r}, found {actual!r}"
        )


def _bundle_from_snapshot(snapshot: Any, label: str) -> str:
    snapshot_map = _mapping(snapshot, label)
    if snapshot_map.get("algorithm") != "sha256":
        raise EvidenceRefreshError(f"{label}.algorithm must be 'sha256'")
    files = _mapping(snapshot_map.get("files"), f"{label}.files")
    if not files:
        raise EvidenceRefreshError(f"{label}.files must not be empty")
    normalized_files: dict[str, str] = {}
    for raw_path, raw_digest in files.items():
        path = _required_string(raw_path, f"{label}.files path")
        if "\\" in path or path.startswith("/") or ".." in Path(path).parts:
            raise EvidenceRefreshError(f"{label}.files contains a non-portable path: {path!r}")
        normalized_files[path] = _required_sha256(raw_digest, f"{label}.files[{path!r}]")
    material = "".join(
        f"{path}\0{digest}\n" for path, digest in sorted(normalized_files.items())
    ).encode("utf-8")
    computed = hashlib.sha256(material).hexdigest()
    recorded = _required_sha256(snapshot_map.get("bundle_sha256"), f"{label}.bundle_sha256")
    if computed != recorded:
        raise EvidenceRefreshError(f"{label}.bundle_sha256 does not match its file-hash map")
    return recorded


def _load_ledger_artifacts_once(
    run_dir: Path,
    status: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, str]]:
    """Read every report-relevant bound artifact once and retain those exact bytes."""

    artifacts = _mapping(status.get("artifacts"), "run_status.artifacts")
    payloads: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    for relative in REQUIRED_LEDGER_PATHS:
        expected = _required_sha256(
            artifacts.get(relative), f"run_status.artifacts[{relative!r}]"
        )
        path = run_dir / Path(relative)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise EvidenceRefreshError(f"required run artifact is unreadable: {relative}: {exc}") from exc
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise EvidenceRefreshError(
                f"run artifact hash mismatch for {relative}: expected {expected}, found {actual}"
            )
        payloads[relative] = payload
        hashes[relative] = actual
    return payloads, hashes


def _validate_figure_records(
    run_id: str,
    source_bundle: str,
    figure_manifest: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
) -> dict[str, Mapping[str, Any]]:
    rows = _sequence(figure_manifest.get("figures"), "figures/figure_manifest.json.figures")
    manifest_rows = _sequence(manifest.get("figures"), "run_manifest.figures")

    def index_rows(raw_rows: Sequence[Any], label: str) -> dict[str, Mapping[str, Any]]:
        indexed: dict[str, Mapping[str, Any]] = {}
        for raw in raw_rows:
            row = _mapping(raw, label)
            figure_id = _required_string(row.get("figure_id"), f"{label}.figure_id")
            if figure_id in indexed:
                raise EvidenceRefreshError(f"duplicate figure_id {figure_id!r} in {label}")
            indexed[figure_id] = row
        return indexed

    local_index = index_rows(rows, "local figure manifest")
    run_index = index_rows(manifest_rows, "run manifest figures")
    expected_ids = set(ALL_FIGURE_IDS)
    if set(local_index) != expected_ids or set(run_index) != expected_ids:
        raise EvidenceRefreshError(
            "full-run figure manifests must contain exactly the eight maintained figure IDs"
        )
    for figure_id in ALL_FIGURE_IDS:
        row = local_index[figure_id]
        run_row = run_index[figure_id]
        _same(row, run_row, f"local/run manifest row for {figure_id}")
        for record, label in ((row, "local figure manifest"), (run_row, "run manifest figures")):
            _same(run_id, record.get("run_id"), f"{label} run_id for {figure_id}")
            metadata = _mapping(record.get("metadata"), f"{label} metadata for {figure_id}")
            _same(
                source_bundle,
                metadata.get("scientific_source_bundle_sha256"),
                f"{label} source bundle for {figure_id}",
            )
            if metadata.get("quick_smoke_only") is not False:
                raise EvidenceRefreshError(f"figure {figure_id} is marked as quick-smoke evidence")
            if record.get("promoted_to_global_manifest") is not True:
                raise EvidenceRefreshError(f"figure {figure_id} was not committed by the full-run promotion")
            if record.get("global_promotion_requested") is not True:
                raise EvidenceRefreshError(f"figure {figure_id} lacks a full-run promotion request")
            central_paths = _mapping(record.get("central_paths"), f"{label} central_paths for {figure_id}")
            for extension in ("png", "pdf"):
                _required_string(
                    central_paths.get(extension),
                    f"{label} central {extension} path for {figure_id}",
                )
        for extension in ("png", "pdf"):
            relative = f"figures/{figure_id}.{extension}"
            _same(relative, row.get(f"local_{extension}"), f"local {extension} path for {figure_id}")
            _same(relative, run_row.get(f"local_{extension}"), f"run {extension} path for {figure_id}")
            actual_hash = artifact_hashes[relative]
            _same(actual_hash, row.get(f"{extension}_sha256"), f"local {extension} hash for {figure_id}")
            _same(actual_hash, run_row.get(f"{extension}_sha256"), f"run {extension} hash for {figure_id}")
    return local_index


def _validate_and_load(run_dir: Path) -> dict[str, Any]:
    status_path = run_dir / "run_status.json"
    try:
        status_bytes = status_path.read_bytes()
    except OSError as exc:
        raise EvidenceRefreshError(f"cannot read run_status.json: {exc}") from exc
    status = _mapping(_decode_json(status_bytes, "run_status.json"), "run_status.json")
    if status.get("status") != "complete":
        raise EvidenceRefreshError("run_status.status must be 'complete'")
    if status.get("quick_mode") is not False:
        raise EvidenceRefreshError("quick/smoke runs cannot refresh report evidence")
    source_bundle = _bundle_from_snapshot(
        status.get("scientific_source_snapshot"), "run_status.scientific_source_snapshot"
    )
    run_id = _required_run_id(status.get("run_id"), source_bundle)
    config_sha256 = _required_sha256(status.get("config_sha256"), "run_status.config_sha256")
    runtime_environment = _mapping(
        status.get("runtime_environment"), "run_status.runtime_environment"
    )
    completed_at_utc = _required_string(
        status.get("completed_at_utc"), "run_status.completed_at_utc"
    )
    completed_phases = tuple(
        _required_string(value, "run_status.completed_phases entry")
        for value in _sequence(status.get("completed_phases"), "run_status.completed_phases")
    )
    if completed_phases != DIRECT_COMPLETED_PHASES:
        raise EvidenceRefreshError(
            "report timing evidence requires the exact uninterrupted full-run phase sequence"
        )
    if status.get("last_completed_phase") != "global_figure_promotion":
        raise EvidenceRefreshError("last_completed_phase must be global_figure_promotion")

    artifact_bytes, artifact_hashes = _load_ledger_artifacts_once(run_dir, status)

    def load_json(relative: str, label: str) -> Any:
        return _decode_json(artifact_bytes[relative], label)

    def load_csv(relative: str, label: str) -> list[dict[str, str]]:
        return _decode_csv(artifact_bytes[relative], label)

    manifest = _mapping(load_json("run_manifest.json", "run_manifest.json"), "run_manifest.json")
    contract = _mapping(load_json("00_system_contract.json", "00_system_contract.json"), "00_system_contract.json")
    selection = _mapping(load_json("03_candidate_selection.json", "03_candidate_selection.json"), "03_candidate_selection.json")
    diagnostics = _mapping(load_json("05_chaos_diagnostics.json", "05_chaos_diagnostics.json"), "05_chaos_diagnostics.json")
    return_map = _sequence(load_json("05_zero_one_return_map.json", "05_zero_one_return_map.json"), "05_zero_one_return_map.json")
    hiddenness = _mapping(load_json("07_hiddenness_summary.json", "07_hiddenness_summary.json"), "07_hiddenness_summary.json")
    robustness = _mapping(load_json("08_robustness_matrix.json", "08_robustness_matrix.json"), "08_robustness_matrix.json")
    gate_payload = _mapping(load_json("09_candidate_gate.json", "09_candidate_gate.json"), "09_candidate_gate.json")
    figure_manifest = _mapping(load_json("figures/figure_manifest.json", "figures/figure_manifest.json"), "figures/figure_manifest.json")
    receipt = _mapping(load_json("figures/global_promotion_receipt.json", "figures/global_promotion_receipt.json"), "figures/global_promotion_receipt.json")
    screening = load_csv("03_candidate_screening.csv", "03_candidate_screening.csv")
    screening_probes = load_csv(
        "03_candidate_screening_probes.csv", "03_candidate_screening_probes.csv"
    )
    screening_contract = _mapping(
        load_json("03_candidate_screening_contract.json", "03_candidate_screening_contract.json"),
        "03_candidate_screening_contract.json",
    )
    stride_sensitivity = load_csv(
        "05_zero_one_stride_sensitivity.csv", "05_zero_one_stride_sensitivity.csv"
    )
    hiddenness_probes = load_csv("07_hiddenness_probes.csv", "07_hiddenness_probes.csv")
    timings = load_csv("phase_timings.csv", "phase_timings.csv")

    if manifest.get("quick_mode") is not False:
        raise EvidenceRefreshError("run_manifest marks the run as quick/smoke")
    if manifest.get("case_id") != CASE_ID:
        raise EvidenceRefreshError(f"run_manifest.case_id must be {CASE_ID!r}")
    _same(run_id, manifest.get("run_id"), "run_id in run_manifest")
    _same(config_sha256, manifest.get("config_sha256"), "config_sha256")
    _same(runtime_environment, manifest.get("runtime_environment"), "runtime environment")
    for snapshot, label in (
        (manifest.get("scientific_source_snapshot"), "run_manifest.scientific_source_snapshot"),
        (contract.get("scientific_source_snapshot"), "system contract scientific_source_snapshot"),
        (selection.get("scientific_source_snapshot"), "candidate selection scientific_source_snapshot"),
    ):
        _same(source_bundle, _bundle_from_snapshot(snapshot, label), f"source bundle in {label}")

    gate = _mapping(gate_payload.get("gate"), "09_candidate_gate.gate")
    for key in ("chaotic_hidden_promotion_allowed", "hiddenness_promotion_allowed", "promotion_allowed"):
        if gate.get(key) is not True:
            raise EvidenceRefreshError(f"candidate gate field {key} is not true")
    if gate.get("hidden_chaos_status") != "chaotic_hidden_under_tested_neighborhoods":
        raise EvidenceRefreshError("candidate gate has the wrong finite hidden-chaos label")
    if gate.get("quick_smoke_only") is True:
        raise EvidenceRefreshError("candidate gate is marked quick-smoke only")
    if _sequence(gate.get("warnings"), "candidate gate warnings"):
        raise EvidenceRefreshError("candidate gate contains warnings")
    if _sequence(gate.get("missing_conditions"), "candidate gate missing_conditions"):
        raise EvidenceRefreshError("candidate gate contains missing conditions")
    manifest_gate = _mapping(manifest.get("candidate_gate"), "run_manifest.candidate_gate")
    _same(gate, manifest_gate, "candidate gate in run_manifest")

    run_metadata = _mapping(_path_get(gate_payload, "evidence.run_metadata"), "candidate gate run_metadata")
    _same(run_id, run_metadata.get("run_id"), "run_id in candidate-gate evidence")
    gate_source_snapshot = _path_get(run_metadata, "provenance.scientific_source_snapshot")
    _same(
        source_bundle,
        _bundle_from_snapshot(gate_source_snapshot, "candidate gate source snapshot"),
        "source bundle in candidate-gate evidence",
    )

    if receipt.get("status") != "committed":
        raise EvidenceRefreshError("global figure promotion receipt is not committed")
    _same(run_id, receipt.get("run_id"), "run_id in global promotion receipt")
    _same(source_bundle, receipt.get("scientific_source_bundle_sha256"), "source bundle in global promotion receipt")
    if _integer(receipt.get("figure_count"), "global receipt figure_count") != len(ALL_FIGURE_IDS):
        raise EvidenceRefreshError("global promotion receipt must bind all eight figures")
    receipt_ids = tuple(
        _required_string(value, "global receipt figure_id")
        for value in _sequence(receipt.get("figure_ids"), "global receipt figure_ids")
    )
    if receipt_ids != ALL_FIGURE_IDS:
        raise EvidenceRefreshError("global promotion receipt has the wrong ordered figure IDs")
    _decimal(receipt.get("seconds"), "global promotion seconds")
    _sequence(receipt.get("global_manifest_paths"), "global promotion manifest paths")
    manifest_receipt = _mapping(manifest.get("global_figure_promotion"), "run_manifest.global_figure_promotion")
    _same(receipt, manifest_receipt, "global promotion receipt in run_manifest")

    figure_index = _validate_figure_records(
        run_id, source_bundle, figure_manifest, manifest, artifact_hashes
    )
    return {
        "status": status,
        "status_bytes": status_bytes,
        "manifest": manifest,
        "contract": contract,
        "selection": selection,
        "diagnostics": diagnostics,
        "return_map": return_map,
        "hiddenness": hiddenness,
        "robustness": robustness,
        "gate": gate,
        "gate_payload": gate_payload,
        "receipt": receipt,
        "figure_index": figure_index,
        "screening": screening,
        "screening_probes": screening_probes,
        "screening_contract": screening_contract,
        "stride_sensitivity": stride_sensitivity,
        "hiddenness_probes": hiddenness_probes,
        "timings": timings,
        "run_id": run_id,
        "source_bundle": source_bundle,
        "config_sha256": config_sha256,
        "completed_at_utc": completed_at_utc,
        "artifact_bytes": artifact_bytes,
        "artifact_hashes": artifact_hashes,
    }


def _validate_consistent_metrics(data: Mapping[str, Any]) -> None:
    manifest = _mapping(data["manifest"], "run_manifest")
    contract = _mapping(data["contract"], "system contract")
    selection = _mapping(data["selection"], "candidate selection")
    diagnostics = _mapping(data["diagnostics"], "diagnostics")
    hiddenness = _mapping(data["hiddenness"], "hiddenness")
    gate_payload = _mapping(data["gate_payload"], "candidate gate payload")

    manifest_parameters = _mapping(manifest.get("candidate_parameters"), "manifest candidate_parameters")
    diagnostic_parameters = _mapping(diagnostics.get("candidate_parameters"), "diagnostic candidate_parameters")
    selected = _mapping(selection.get("selected"), "candidate selection selected")
    for key in ("xi", "gamma", "delta", "rho"):
        expected = _decimal(manifest_parameters.get(key), f"manifest candidate {key}")
        _same(expected, _decimal(diagnostic_parameters.get(key), f"diagnostic candidate {key}"), f"candidate {key}")
        if key == "gamma":
            _same(expected, _decimal(selected.get(key), "selected gamma"), "selected gamma")
    for key, expected in EXPECTED_CANDIDATE_PARAMETERS.items():
        _same(expected, _decimal(manifest_parameters.get(key), f"candidate {key}"), f"declared candidate {key}")

    hopf = _decimal(selection.get("hopf_boundary"), "selection hopf_boundary")
    _same(hopf, _decimal(manifest.get("hopf_boundary"), "manifest hopf_boundary"), "Hopf boundary")
    contract_hopf = _decimal(
        _path_get(contract, "hopf_boundary_at_xi_target.selected_high_gamma_boundary"),
        "contract high Hopf boundary",
    )
    _close_decimal(hopf, contract_hopf, "contract/selection Hopf boundary")
    selected_offset = _decimal(selected.get("hopf_offset"), "selected Hopf offset")
    selected_gamma = _decimal(selected.get("gamma"), "selected gamma")
    _close_decimal(selected_gamma, hopf + selected_offset, "gamma = gamma_H + selected offset")
    _close_decimal(manifest.get("selected_hopf_offset"), selected_offset, "manifest selected Hopf offset")
    if contract.get("frequency_grid_used_for_seed") is not False:
        raise EvidenceRefreshError("system contract records a frequency grid in the direct seed")
    if contract.get("fallback_frequency_scan_used") is not False:
        raise EvidenceRefreshError("system contract records a fallback frequency scan")
    if contract.get("report_values_used_as_search_input") is not False:
        raise EvidenceRefreshError("system contract records report values as search inputs")
    if selection.get("no_frequency_sweep") is not True:
        raise EvidenceRefreshError("candidate selection does not explicitly exclude a frequency sweep")
    if selection.get("primary_route_chaos_screen_failed_before_alternative") is not True:
        raise EvidenceRefreshError("alternative route lacks its declared primary-route failure trigger")
    if manifest.get("frequency_grid_used_for_search") is not False:
        raise EvidenceRefreshError("run manifest records a frequency grid in the search")

    strict_payload = _mapping(diagnostics.get("lyapunov"), "strict Lyapunov payload")
    control_payload = _mapping(diagnostics.get("lyapunov_control"), "control Lyapunov payload")
    strict_exponents = _sequence(strict_payload.get("exponents"), "strict exponents")
    control_exponents = _sequence(control_payload.get("exponents"), "control exponents")
    if len(strict_exponents) != 3 or len(control_exponents) != 3:
        raise EvidenceRefreshError("strict and control Lyapunov spectra must each have three exponents")
    strict = tuple(_decimal(value, "strict exponent") for value in strict_exponents)
    control = tuple(_decimal(value, "control exponent") for value in control_exponents)
    if strict_payload.get("status") != "ok" or control_payload.get("status") != "ok":
        raise EvidenceRefreshError("strict/control Lyapunov integrations must both have status ok")
    if strict[0] <= Decimal("0.02") or control[0] <= Decimal("0.02"):
        raise EvidenceRefreshError("strict/control largest Lyapunov exponents must both exceed 0.02")
    manifest_exponents = _sequence(manifest.get("lyapunov_exponents"), "manifest Lyapunov exponents")
    _same(
        strict,
        tuple(_decimal(value, "manifest exponent") for value in manifest_exponents),
        "strict Lyapunov spectrum",
    )
    if strict[0] + strict[1] <= 0 or sum(strict) >= 0:
        raise EvidenceRefreshError("strict spectrum is incompatible with the reported Kaplan-Yorke branch")
    expected_ky = Decimal(2) + (strict[0] + strict[1]) / abs(strict[2])
    _close_decimal(
        diagnostics.get("kaplan_yorke_dimension"),
        expected_ky,
        "Kaplan-Yorke dimension",
        atol=Decimal("1e-10"),
    )
    mean_divergence = _decimal(
        diagnostics.get("mean_vector_field_divergence"), "mean vector-field divergence"
    )
    _close_decimal(
        diagnostics.get("lyapunov_sum_minus_mean_divergence"),
        sum(strict) - mean_divergence,
        "Lyapunov/divergence residual",
        atol=Decimal("1e-9"),
    )
    if diagnostics.get("finite_time_only") is not True:
        raise EvidenceRefreshError("diagnostics must be explicitly finite-time only")

    return_map = _sequence(data["return_map"], "return-map results")
    diagnostic_return_map = _sequence(
        _path_get(diagnostics, "zero_one.return_map_results"), "diagnostic return-map results"
    )
    if len(return_map) != 2:
        raise EvidenceRefreshError("exactly two declared Poincare return-map 0-1 results are required")
    _same(return_map, diagnostic_return_map, "return-map 0-1 results")
    return_coordinates: set[int] = set()
    return_k: list[Decimal] = []
    for index, raw in enumerate(return_map):
        row = _mapping(raw, f"return-map result {index}")
        coordinate = _integer(row.get("coordinate"), f"return-map result {index} coordinate")
        return_coordinates.add(coordinate)
        k_value = _decimal(row.get("K"), f"return-map result {index} K")
        if not Decimal("0.7") <= k_value <= Decimal("1.000000000001"):
            raise EvidenceRefreshError("return-map 0-1 K is outside the declared chaotic range")
        if row.get("state") != "zero_one_chaotic_candidate":
            raise EvidenceRefreshError("return-map 0-1 state is not chaotic-candidate")
        return_k.append(k_value)
    if return_coordinates != {0, 2}:
        raise EvidenceRefreshError("return-map 0-1 results must use coordinates 0 and 2")

    stride_rows = data["stride_sensitivity"]
    if len(stride_rows) != 10:
        raise EvidenceRefreshError("flow 0-1 sensitivity must contain ten declared strides")
    stride_counts = {"chaotic": 0, "regular": 0, "inconclusive": 0}
    for row in stride_rows:
        state = row.get("state")
        if state == "zero_one_chaotic_candidate":
            stride_counts["chaotic"] += 1
        elif state == "zero_one_regular_candidate":
            stride_counts["regular"] += 1
        elif state in {"zero_one_inconclusive", "insufficient_samples"}:
            stride_counts["inconclusive"] += 1
        else:
            raise EvidenceRefreshError(f"unexpected flow 0-1 state: {state!r}")
    if stride_counts != {"chaotic": 3, "regular": 1, "inconclusive": 6}:
        raise EvidenceRefreshError(
            f"flow 0-1 state counts differ from the report claim: {stride_counts}"
        )

    efork = _mapping(diagnostics.get("efork_crosscheck"), "EFORK cross-check")
    if efork.get("status") != "ok":
        raise EvidenceRefreshError("EFORK cross-check did not complete")
    efork_match = _mapping(efork.get("target_match"), "EFORK target match")
    if efork_match.get("classification") != "same_attractor_under_calibrated_cloud_test":
        raise EvidenceRefreshError("EFORK did not reach the calibrated candidate cloud")

    main_count = _integer(_path_get(hiddenness, "main.n_probes"), "main probe count")
    targeted_count = _integer(
        _path_get(hiddenness, "targeted_E0_unstable_direction.n_probes"),
        "targeted probe count",
    )
    total_count = _integer(hiddenness.get("n_probes"), "total probe count")
    if (main_count, targeted_count, total_count) != (108, 4, 112):
        raise EvidenceRefreshError("hiddenness counts must be exactly 108 main + 4 targeted = 112")
    hits = _integer(hiddenness.get("target_hits"), "target hits")
    if hits != 0:
        raise EvidenceRefreshError("validated hidden candidate must have zero target contacts")
    for key in ("ambiguous", "numerical_failures"):
        if _integer(hiddenness.get(key), f"hiddenness {key}") != 0:
            raise EvidenceRefreshError(f"validated hidden candidate has nonzero {key}")
    if hiddenness.get("sampled_hiddenness_status") != "hidden_under_tested_neighborhoods":
        raise EvidenceRefreshError("hiddenness summary does not have the finite-neighborhood hidden label")
    if hiddenness.get("finite_sample_only") is not True or hiddenness.get("global_hiddenness_proved") is not False:
        raise EvidenceRefreshError("hiddenness summary has the wrong finite/global evidence boundary")
    if _path_get(hiddenness, "coverage_by_equilibrium_radius.complete") is not True:
        raise EvidenceRefreshError("hiddenness coverage by equilibrium/radius is incomplete")
    if _path_get(hiddenness, "main.tested_all_declared_equilibria") is not True:
        raise EvidenceRefreshError("main hiddenness contract did not test all declared equilibria")
    _same(hiddenness, manifest.get("hiddenness"), "hiddenness summary in run_manifest")

    probe_rows = data["hiddenness_probes"]
    required_probe_columns = {
        "contract",
        "sample_id",
        "equilibrium",
        "status",
        "destination",
        "target_hit",
        "ambiguous",
        "closest_equilibrium",
    }
    if not probe_rows or not required_probe_columns.issubset(probe_rows[0]):
        raise EvidenceRefreshError("hiddenness probe CSV lacks report-required columns")
    if len(probe_rows) != 112:
        raise EvidenceRefreshError("hiddenness probe CSV must contain exactly 112 rows")
    contract_counts = {"main_3x3xN": 0, "targeted_E0_unstable": 0}
    seen_probe_ids: set[tuple[str, int]] = set()
    for index, row in enumerate(probe_rows, start=2):
        contract_name = row["contract"]
        if contract_name not in contract_counts:
            raise EvidenceRefreshError(f"unexpected hiddenness contract at row {index}")
        contract_counts[contract_name] += 1
        sample_id = _integer(row["sample_id"], f"hiddenness sample_id row {index}")
        identity = (contract_name, sample_id)
        if identity in seen_probe_ids:
            raise EvidenceRefreshError(f"duplicate hiddenness probe identity at row {index}")
        seen_probe_ids.add(identity)
        if row["status"] != "ok" or _boolean(row["target_hit"], f"target_hit row {index}"):
            raise EvidenceRefreshError(f"hiddenness probe row {index} is not a successful non-contact")
        if _boolean(row["ambiguous"], f"ambiguous row {index}"):
            raise EvidenceRefreshError(f"hiddenness probe row {index} is ambiguous")
        destination = row["destination"]
        if not destination.startswith("equilibrium_"):
            raise EvidenceRefreshError(
                f"all 112 report probes must have equilibrium destinations; row {index} has {destination!r}"
            )
        destination_equilibrium = destination.removeprefix("equilibrium_")
        if destination_equilibrium != row["closest_equilibrium"]:
            raise EvidenceRefreshError(f"hiddenness destination/closest equilibrium mismatch at row {index}")
    if contract_counts != {"main_3x3xN": 108, "targeted_E0_unstable": 4}:
        raise EvidenceRefreshError(f"hiddenness probe contract counts are wrong: {contract_counts}")

    screening = data["screening"]
    required_columns = {
        "hopf_offset",
        "gamma",
        "equilibrium_stability_margin",
        "lambda_1",
        "lambda_2",
        "lambda_3",
        "lyapunov_status",
        "E0_probe_count",
        "E0_target_hits",
        "E0_ambiguous",
        "eligible_hidden_chaos_screen",
    }
    if not required_columns.issubset(screening[0]):
        missing = sorted(required_columns.difference(screening[0]))
        raise EvidenceRefreshError(f"candidate screening CSV lacks columns: {missing}")
    screening_probes = data["screening_probes"]
    grouped_probes: dict[Decimal, list[Mapping[str, str]]] = {}
    for row in screening_probes:
        offset = _decimal(row.get("hopf_offset"), "screening-probe Hopf offset")
        grouped_probes.setdefault(offset, []).append(row)
    observed_offsets = tuple(_decimal(row["hopf_offset"], "screening Hopf offset") for row in screening)
    if observed_offsets != EXPECTED_SCREEN_OFFSETS:
        raise EvidenceRefreshError("candidate screening does not contain the exact ordered offset contract")
    eligible_rows = []
    for row in screening:
        offset = _decimal(row["hopf_offset"], "screening Hopf offset")
        gamma = _decimal(row["gamma"], "screening gamma")
        _close_decimal(gamma, hopf + offset, f"screening gamma at offset {offset}")
        lambda_one = _decimal(row["lambda_1"], "screening lambda_1")
        _decimal(row.get("lambda_2"), "screening lambda_2")
        _decimal(row.get("lambda_3"), "screening lambda_3")
        probes = grouped_probes.get(offset, [])
        if len(probes) != 12:
            raise EvidenceRefreshError(f"screening offset {offset} must have exactly 12 probe rows")
        probe_hits = sum(_boolean(probe.get("target_hit"), "screening probe target_hit") for probe in probes)
        probe_ambiguous = sum(_boolean(probe.get("ambiguous"), "screening probe ambiguous") for probe in probes)
        probe_failures = sum(probe.get("status") != "ok" for probe in probes)
        _same(12, _integer(row["E0_probe_count"], "screening E0 probe count"), "screening probe count")
        _same(probe_hits, _integer(row["E0_target_hits"], "screening target hits"), "screening target-hit count")
        _same(probe_ambiguous, _integer(row["E0_ambiguous"], "screening ambiguous count"), "screening ambiguous count")
        expected_eligible = bool(
            row["lyapunov_status"] == "ok"
            and lambda_one > Decimal("0.5")
            and probe_hits == 0
            and probe_ambiguous == 0
            and probe_failures == 0
        )
        actual_eligible = _boolean(row["eligible_hidden_chaos_screen"], "screening eligibility")
        if actual_eligible != expected_eligible:
            raise EvidenceRefreshError(f"screening eligibility is inconsistent at offset {offset}")
        if actual_eligible:
            eligible_rows.append(row)
    if not eligible_rows:
        raise EvidenceRefreshError("candidate screening contains no eligible row")
    selected_rows = [
        row for row in eligible_rows
        if _decimal(row["hopf_offset"], "screening Hopf offset") == selected_offset
    ]
    if len(selected_rows) != 1:
        raise EvidenceRefreshError("selected Hopf offset does not identify exactly one eligible screening row")
    largest_eligible = max(
        _decimal(row["hopf_offset"], "eligible Hopf offset") for row in eligible_rows
    )
    _same(largest_eligible, selected_offset, "declared largest-eligible-offset selection rule")
    _same(
        _decimal(selected_rows[0]["gamma"], "screening selected gamma"),
        _decimal(selected.get("gamma"), "selected gamma"),
        "selected gamma in screening",
    )
    for key in required_columns:
        if key in {
            "hopf_offset",
            "gamma",
            "equilibrium_stability_margin",
            "lambda_1",
            "lambda_2",
            "lambda_3",
            "E0_probe_count",
            "E0_target_hits",
            "E0_ambiguous",
        }:
            _close_decimal(selected_rows[0][key], selected.get(key), f"selected screening field {key}")
        elif key in {"eligible_hidden_chaos_screen"}:
            _same(
                _boolean(selected_rows[0][key], f"selected row {key}"),
                _boolean(selected.get(key), f"selection selected {key}"),
                f"selected screening field {key}",
            )
        else:
            _same(selected_rows[0][key], selected.get(key), f"selected screening field {key}")
    screening_contract = _mapping(data["screening_contract"], "screening contract")
    if screening_contract.get("finite_sample_only") is not True:
        raise EvidenceRefreshError("candidate screening contract is not explicitly finite-sample only")

    evidence = _mapping(gate_payload.get("evidence"), "candidate-gate evidence")
    gate_lyapunov = tuple(
        _decimal(value, "gate Lyapunov exponent")
        for value in _sequence(_path_get(evidence, "lyapunov.exponents"), "gate Lyapunov exponents")
    )
    _same(strict, gate_lyapunov, "diagnostic/gate Lyapunov spectrum")
    gate_hiddenness = _mapping(evidence.get("hiddenness"), "gate hiddenness evidence")
    for field, expected in (
        ("target_hits_from_equilibria", 0),
        ("numerical_failures", 0),
    ):
        _same(expected, _integer(gate_hiddenness.get(field), f"gate hiddenness {field}"), f"gate hiddenness {field}")
    if gate_hiddenness.get("basin_controls_complete") is not True or gate_hiddenness.get("coverage_by_equilibrium_radius_complete") is not True:
        raise EvidenceRefreshError("candidate gate does not bind complete basin/coverage controls")
    expected_gate_k = sum(return_k) / Decimal(len(return_k))
    _close_decimal(_path_get(evidence, "zero_one.K"), expected_gate_k, "gate return-map median K")
    _same(data["robustness"], evidence.get("robustness"), "robustness matrix/gate evidence")

    timing_rows = data["timings"]
    phases = tuple(row.get("phase", "") for row in timing_rows)
    if phases != DIRECT_TIMING_PHASES:
        raise EvidenceRefreshError("phase_timings.csv is not an exact uninterrupted full-run timing table")
    timing_seconds: dict[str, Decimal] = {}
    for row in timing_rows:
        phase = row["phase"]
        seconds = _decimal(row.get("seconds"), f"{phase} timing")
        if seconds < 0:
            raise EvidenceRefreshError(f"{phase} timing must be non-negative")
        if row.get("timing_source") != "perf_counter":
            raise EvidenceRefreshError(f"{phase} timing must come from perf_counter")
        timing_seconds[phase] = seconds
    if timing_seconds["total"] < sum(
        timing_seconds[phase] for phase in DIRECT_TIMING_PHASES[:-1]
    ):
        raise EvidenceRefreshError("total timing is smaller than the sum of recorded phases")
    manifest_timings = _sequence(manifest.get("timings"), "run manifest timings")
    if len(manifest_timings) != len(timing_rows):
        raise EvidenceRefreshError("run manifest timing count differs from phase_timings.csv")
    for csv_row, raw_manifest_row in zip(timing_rows, manifest_timings, strict=True):
        manifest_row = _mapping(raw_manifest_row, "run manifest timing row")
        _same(csv_row["phase"], manifest_row.get("phase"), "timing phase")
        _same(csv_row["timing_source"], manifest_row.get("timing_source"), "timing source")
        _close_decimal(csv_row["seconds"], manifest_row.get("seconds"), f"{csv_row['phase']} timing")


def _timing_value(rows: Sequence[Mapping[str, str]], prefix: str, label: str) -> str:
    matches = [row for row in rows if row.get("phase", "") == prefix]
    if len(matches) != 1:
        raise EvidenceRefreshError(f"expected exactly one {label} timing row, found {len(matches)}")
    seconds = matches[0].get("seconds", "")
    if _decimal(seconds, f"{label} timing") < 0:
        raise EvidenceRefreshError(f"{label} timing must be non-negative")
    return _number_text(seconds, f"{label} timing")


def _macro_values(data: Mapping[str, Any]) -> list[tuple[str, str]]:
    manifest = _mapping(data["manifest"], "run manifest")
    selection = _mapping(data["selection"], "candidate selection")
    diagnostics = _mapping(data["diagnostics"], "diagnostics")
    hiddenness = _mapping(data["hiddenness"], "hiddenness")
    parameters = _mapping(manifest.get("candidate_parameters"), "candidate parameters")
    strict = _sequence(_path_get(diagnostics, "lyapunov.exponents"), "strict exponents")
    control = _sequence(_path_get(diagnostics, "lyapunov_control.exponents"), "control exponents")
    return_map = _sequence(data["return_map"], "return-map results")
    timings = data["timings"]

    values = [
        ("MAVPDChaosXi", _number_text(parameters.get("xi"), "candidate xi")),
        ("MAVPDChaosHopf", _number_text(selection.get("hopf_boundary"), "Hopf boundary")),
        ("MAVPDChaosGamma", _number_text(parameters.get("gamma"), "candidate gamma")),
        ("MAVPDChaosLEOne", _number_text(strict[0], "strict lambda_1")),
        ("MAVPDChaosLETwo", _number_text(strict[1], "strict lambda_2")),
        ("MAVPDChaosLEThree", _number_text(strict[2], "strict lambda_3")),
        ("MAVPDChaosLEControlOne", _number_text(control[0], "control lambda_1")),
        (
            "MAVPDChaosDivergenceResidual",
            _number_text(
                diagnostics.get("lyapunov_sum_minus_mean_divergence"),
                "Lyapunov/divergence residual",
            ),
        ),
        ("MAVPDChaosKY", _number_text(diagnostics.get("kaplan_yorke_dimension"), "Kaplan-Yorke dimension")),
        ("MAVPDChaosPoincareCount", str(_integer(_path_get(diagnostics, "poincare.crossing_count"), "Poincare crossing count"))),
        (
            "MAVPDChaosReturnKOne",
            _number_text(_mapping(return_map[0], "return-map result 0").get("K"), "return K 1"),
        ),
        (
            "MAVPDChaosReturnKTwo",
            _number_text(_mapping(return_map[1], "return-map result 1").get("K"), "return K 2"),
        ),
        ("MAVPDChaosMainProbes", str(_integer(_path_get(hiddenness, "main.n_probes"), "main probes"))),
        (
            "MAVPDChaosTargetedProbes",
            str(_integer(_path_get(hiddenness, "targeted_E0_unstable_direction.n_probes"), "targeted probes")),
        ),
        ("MAVPDChaosTotalProbes", str(_integer(hiddenness.get("n_probes"), "total probes"))),
        ("MAVPDChaosHits", str(_integer(hiddenness.get("target_hits"), "target hits"))),
        ("MAVPDChaosContractTime", _timing_value(timings, "contract", "contract")),
        ("MAVPDChaosSearchTime", _timing_value(timings, "search", "search")),
        ("MAVPDChaosDiagnosticsTime", _timing_value(timings, "diagnostics", "diagnostics")),
        ("MAVPDChaosHiddennessTime", _timing_value(timings, "hiddenness", "hiddenness")),
        ("MAVPDChaosGateTime", _timing_value(timings, "candidate_gate", "candidate gate")),
        ("MAVPDChaosTotalTime", _timing_value(timings, "total", "total")),
    ]
    if len(values) != 22 or len({name for name, _value in values}) != 22:
        raise AssertionError("the MAVPD report contract must contain exactly 22 unique macros")
    return values


def _render_generated_tex(data: Mapping[str, Any]) -> bytes:
    lines = [
        "% Generado por refresh_mavpd_chaos_evidence.py; no editar a mano.",
        f"% run_id: {_latex_escape(str(data['run_id']))}",
        f"% scientific_source_bundle_sha256: {data['source_bundle']}",
        f"\\renewcommand{{\\FechaCorteDatos}}{{{_latex_escape(str(data['report_date']))}}}",
        "",
    ]
    lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in _macro_values(data))
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _screening_interpretation(row: Mapping[str, str], *, selected: bool) -> str:
    eligible = row["eligible_hidden_chaos_screen"].strip().lower() == "true"
    if selected:
        return "Elegible y seleccionada por la regla declarada."
    if eligible:
        return "Elegible bajo el cribado finito; no seleccionada por la regla."
    if row["lyapunov_status"].strip().lower() != "ok":
        return "No elegible: el diagnóstico finito de Lyapunov no se completó."
    if _integer(row["E0_target_hits"], "screening target hits") > 0:
        return "No elegible: hubo contactos en los sondeos preliminares."
    if _integer(row["E0_ambiguous"], "screening ambiguous count") > 0:
        return "No elegible: hubo resultados ambiguos en los sondeos preliminares."
    return "No elegible bajo el cribado finito declarado."


def _render_screening_rows(data: Mapping[str, Any]) -> bytes:
    selected = _mapping(_mapping(data["selection"], "selection").get("selected"), "selected row")
    selected_offset = _decimal(selected.get("hopf_offset"), "selected Hopf offset")
    lines = [
        "% Generado desde todas las filas de 03_candidate_screening.csv.",
        "% Columnas: hopf_offset; lambda_1; E0_target_hits/E0_probe_count; lectura finita.",
    ]
    for row in data["screening"]:
        offset = _decimal(row["hopf_offset"], "screening Hopf offset")
        is_selected = (
            offset == selected_offset
            and row["eligible_hidden_chaos_screen"].strip().lower() == "true"
        )
        hits = _integer(row["E0_target_hits"], "screening target hits")
        probes = _integer(row["E0_probe_count"], "screening probe count")
        if hits > probes:
            raise EvidenceRefreshError("screening target hits exceed the probe count")
        fields = (
            format(offset, ".3f"),
            format(_decimal(row["lambda_1"], "screening lambda_1"), ".6f")
            .rstrip("0")
            .rstrip("."),
            f"{hits}/{probes}",
            _screening_interpretation(row, selected=is_selected),
        )
        lines.append(" & ".join(_latex_escape(field) for field in fields) + r" \\")
    # Keep the booktabs rule inside the input fragment.  TeX otherwise resumes
    # the alignment after \input and can treat a following \bottomrule in the
    # parent file as cell content instead of an inter-row \noalign command.
    lines.append(r"\bottomrule")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def _provenance_payload(
    run_dir: Path,
    repo_root: Path,
    data: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    authoritative_validation: Mapping[str, Any],
) -> dict[str, Any]:
    assets = []
    for figure_id, destination_stem in FIGURE_ASSETS:
        formats: dict[str, Any] = {}
        for extension in ("png", "pdf"):
            source_relative = f"figures/{figure_id}.{extension}"
            destination_relative = f"assets/{destination_stem}.{extension}"
            digest = source_hashes[source_relative]
            formats[extension] = {
                "source": {"path": source_relative, "sha256": digest},
                "destination": {"path": destination_relative, "sha256": digest},
            }
        assets.append({"figure_id": figure_id, "formats": formats})
    try:
        portable_run_dir = run_dir.relative_to(repo_root).as_posix()
        source_path_kind = "repo_relative"
    except ValueError:
        portable_run_dir = run_dir.as_posix()
        source_path_kind = "absolute_outside_repo"
    return {
        "schema_version": "2.0",
        "case_id": CASE_ID,
        "report_date": data["report_date"],
        "run_id": data["run_id"],
        "config_sha256": data["config_sha256"],
        "scientific_source_bundle_sha256": data["source_bundle"],
        "completed_at_utc": data["completed_at_utc"],
        "source_run_directory": portable_run_dir,
        "source_run_directory_kind": source_path_kind,
        "run_status_sha256": hashlib.sha256(data["status_bytes"]).hexdigest(),
        "run_manifest_sha256": data["artifact_hashes"]["run_manifest.json"],
        "authoritative_validation": {
            "validator": authoritative_validation.get("validator"),
            "ok": authoritative_validation.get("ok"),
            "active_promotion_checked": authoritative_validation.get("active_promotion_checked"),
            "active_promotion_required": authoritative_validation.get("active_promotion_required"),
        },
        "refresh_transaction": {
            "exclusive_lock": True,
            "all_outputs_staged_before_commit": True,
            "rollback_on_commit_or_postverification_failure": True,
            "filesystem_batch_atomic": False,
        },
        "assets": assets,
    }


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _write_fsynced(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _verify_output_payloads(outputs: Mapping[Path, bytes]) -> None:
    for destination, payload in outputs.items():
        actual = _sha256(destination)
        expected = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise EvidenceRefreshError(f"post-write verification failed for {destination}")


@contextmanager
def _exclusive_refresh_lock(report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    lock_path = report_dir / ".mavpd_chaos_refresh.lock"
    token = f"pid={os.getpid()} token={uuid4().hex}\n"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise EvidenceRefreshError(
            f"another MAVPD report refresh holds the lock: {lock_path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(token)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            if lock_path.read_text(encoding="utf-8") == token:
                lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def _commit_transaction(report_dir: Path, outputs: Mapping[Path, bytes]) -> None:
    """Stage every output, commit under a lock, and roll back verified failures."""

    resolved_report = report_dir.resolve()
    with _exclusive_refresh_lock(resolved_report):
        staging = Path(tempfile.mkdtemp(prefix=".mavpd-refresh-stage-", dir=resolved_report))
        backups = Path(tempfile.mkdtemp(prefix=".mavpd-refresh-backup-", dir=resolved_report))
        relative_paths: dict[Path, Path] = {}
        committed: list[Path] = []
        had_previous: dict[Path, bool] = {}
        preserve_backups = False
        try:
            for destination, payload in outputs.items():
                resolved_destination = destination.resolve(strict=False)
                if not _is_within(resolved_destination, resolved_report):
                    raise EvidenceRefreshError(
                        f"report output escapes the report directory: {destination}"
                    )
                if destination.is_symlink():
                    raise EvidenceRefreshError(f"refusing to replace symlinked report output: {destination}")
                relative = resolved_destination.relative_to(resolved_report)
                relative_paths[destination] = relative
                staged_path = staging / relative
                _write_fsynced(staged_path, payload)
                if hashlib.sha256(staged_path.read_bytes()).hexdigest() != hashlib.sha256(payload).hexdigest():
                    raise EvidenceRefreshError(f"staging verification failed for {destination}")

            for destination, relative in relative_paths.items():
                had_previous[destination] = destination.exists()
                if destination.exists():
                    if not destination.is_file():
                        raise EvidenceRefreshError(f"report output path is not a regular file: {destination}")
                    _write_fsynced(backups / relative, destination.read_bytes())

            _write_fsynced(
                backups / "transaction.json",
                _json_bytes(
                    {
                        "schema_version": "1.0",
                        "state": "backups_complete_before_commit",
                        "outputs": [
                            {
                                "path": relative_paths[destination].as_posix(),
                                "had_previous": had_previous[destination],
                            }
                            for destination in outputs
                        ],
                    }
                ),
            )

            try:
                for destination, relative in relative_paths.items():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staging / relative, destination)
                    committed.append(destination)
                _verify_output_payloads(outputs)
            except Exception as commit_error:
                rollback_errors: list[str] = []
                for destination in reversed(committed):
                    try:
                        if had_previous[destination]:
                            os.replace(backups / relative_paths[destination], destination)
                        else:
                            destination.unlink(missing_ok=True)
                    except Exception as rollback_error:  # continue restoring every path
                        rollback_errors.append(f"{destination}: {rollback_error}")
                if rollback_errors:
                    preserve_backups = True
                    raise EvidenceRefreshError(
                        "report refresh failed and rollback was incomplete: "
                        + "; ".join(rollback_errors)
                        + f"; recovery backups retained at {backups}"
                    ) from commit_error
                if isinstance(commit_error, EvidenceRefreshError):
                    raise
                raise EvidenceRefreshError(f"report refresh transaction failed: {commit_error}") from commit_error
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if not preserve_backups:
                shutil.rmtree(backups, ignore_errors=True)


def _load_authoritative_validator(repo_root: Path):
    validator_path = repo_root / "validation" / "python" / "validate_mavpd_integer_hidden_chaos_run.py"
    if not validator_path.is_file():
        raise EvidenceRefreshError(f"authoritative MAVPD validator is missing: {validator_path}")
    module_name = f"_mavpd_run_validator_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, validator_path)
    if spec is None or spec.loader is None:
        raise EvidenceRefreshError(f"cannot load authoritative MAVPD validator: {validator_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EvidenceRefreshError(f"cannot import authoritative MAVPD validator: {exc}") from exc
    validator = getattr(module, "validate_run", None)
    if not callable(validator):
        raise EvidenceRefreshError("authoritative MAVPD validator lacks callable validate_run")
    return validator


def _run_authoritative_validation(
    run_dir: Path,
    repo_root: Path,
    scientific_source_root: Path,
) -> Mapping[str, Any]:
    validator = _load_authoritative_validator(repo_root)
    try:
        raw_summary = validator(
            run_dir,
            repo_root,
            scientific_source_root,
            require_active_promotion=True,
        )
    except Exception as exc:
        raise EvidenceRefreshError(f"authoritative MAVPD validation raised: {exc}") from exc
    summary = _mapping(raw_summary, "authoritative MAVPD validation summary")
    if summary.get("ok") is not True:
        errors = summary.get("errors")
        raise EvidenceRefreshError(
            "authoritative MAVPD validation failed: "
            + json.dumps(errors, ensure_ascii=False, default=str)
        )
    if summary.get("active_promotion_required") is not True or summary.get("active_promotion_checked") is not True:
        raise EvidenceRefreshError("authoritative validation did not verify the active global promotion")
    return summary


def refresh_evidence(
    run_dir: str | Path,
    report_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    scientific_source_root: str | Path | None = None,
    report_date: str = DEFAULT_REPORT_DATE,
) -> dict[str, Any]:
    """Validate ``run_dir`` and transactionally refresh report-only MAVPD evidence."""

    resolved_run = Path(run_dir).expanduser().resolve()
    if not resolved_run.is_dir():
        raise EvidenceRefreshError(f"run directory does not exist: {resolved_run}")
    resolved_report = Path(report_dir).expanduser().resolve()
    if resolved_report.exists() and not resolved_report.is_dir():
        raise EvidenceRefreshError(f"report path is not a directory: {resolved_report}")
    resolved_repo = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    resolved_source = (
        Path(scientific_source_root).expanduser().resolve()
        if scientific_source_root is not None
        else resolved_repo
    )
    if not resolved_repo.is_dir() or not resolved_source.is_dir():
        raise EvidenceRefreshError("repo_root and scientific_source_root must be existing directories")
    validated_date = _validated_report_date(report_date)

    authoritative_validation = _run_authoritative_validation(
        resolved_run, resolved_repo, resolved_source
    )
    data = _validate_and_load(resolved_run)
    _same(data["run_id"], authoritative_validation.get("run_id"), "authoritative run_id")
    _same(
        data["source_bundle"],
        authoritative_validation.get("scientific_source_bundle_sha256"),
        "authoritative scientific-source bundle",
    )
    _same(
        data["config_sha256"],
        authoritative_validation.get("config_sha256"),
        "authoritative config SHA-256",
    )
    _validate_consistent_metrics(data)
    data["report_date"] = validated_date

    source_hashes: dict[str, str] = {}
    outputs: dict[Path, bytes] = {}
    for figure_id, destination_stem in FIGURE_ASSETS:
        for extension in ("png", "pdf"):
            relative = f"figures/{figure_id}.{extension}"
            payload = data["artifact_bytes"][relative]
            digest = data["artifact_hashes"][relative]
            if hashlib.sha256(payload).hexdigest() != digest:
                raise EvidenceRefreshError(f"cached figure bytes/hash mismatch for {relative}")
            source_hashes[relative] = digest
            outputs[resolved_report / "assets" / f"{destination_stem}.{extension}"] = payload

    provenance = _provenance_payload(
        resolved_run,
        resolved_repo,
        data,
        source_hashes,
        authoritative_validation,
    )
    outputs[resolved_report / "assets" / "mavpd_chaos_assets_provenance.json"] = _json_bytes(provenance)
    outputs[resolved_report / "mavpd_chaos_generated.tex"] = _render_generated_tex(data)
    outputs[resolved_report / "mavpd_chaos_screening_rows.tex"] = _render_screening_rows(data)
    try:
        _commit_transaction(resolved_report, outputs)
    except EvidenceRefreshError:
        raise
    except Exception as exc:
        raise EvidenceRefreshError(f"report refresh transaction failed: {exc}") from exc
    return {
        "status": "refreshed",
        "run_id": data["run_id"],
        "scientific_source_bundle_sha256": data["source_bundle"],
        "report_dir": str(resolved_report),
        "asset_pairs": len(FIGURE_ASSETS),
        "mavpd_macro_count": 22,
        "screening_rows": len(data["screening"]),
        "report_date": validated_date,
        "authoritative_validation": "passed_with_active_promotion",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh MAVPD hidden-chaos report evidence from one validated full run."
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="completed MAVPD full-run directory")
    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
        help="explicit report directory to refresh",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repository root used by the authoritative run/promotion validator",
    )
    parser.add_argument(
        "--scientific-source-root",
        type=Path,
        default=None,
        help="root whose scientific sources must match the completed run snapshot",
    )
    parser.add_argument(
        "--report-date",
        default=DEFAULT_REPORT_DATE,
        help=f"explicit report evidence date (default: {DEFAULT_REPORT_DATE})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = refresh_evidence(
            args.run_dir,
            args.report_dir,
            repo_root=args.repo_root,
            scientific_source_root=args.scientific_source_root,
            report_date=args.report_date,
        )
    except EvidenceRefreshError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
