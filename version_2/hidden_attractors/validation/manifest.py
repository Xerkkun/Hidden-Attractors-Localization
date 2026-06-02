"""Regenerate the official validation manifest from stage summaries."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from ..paths import CONFIGS, PROJECT_ROOT
from ..reproducibility import validate_hiddenness_promotion_metadata


CLOSED_STAGE_STATUSES = frozenset({"completed", "passed_python_wolfram"})
DEFAULT_CONTRACT = CONFIGS / "validation_contract.json"
DEFAULT_VALIDATION_ROOT = PROJECT_ROOT / "validation"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _git_provenance(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return {"repository_commit": "working_tree", "working_tree_dirty": True}
    return {"repository_commit": commit, "working_tree_dirty": bool(status)}


def _package_version() -> str:
    try:
        return metadata.version("hidden-attractors-fo")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def regenerate_validation_manifest(
    validation_root: Path = DEFAULT_VALIDATION_ROOT,
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    validation_id: str = "chua_fractional_validation_evidence",
    provenance: dict[str, Any] | None = None,
    main_system: str = "fractional nonsmooth Chua",
    main_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``00_manifest`` using only official summary files as stage truth."""

    validation_root = Path(validation_root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = _read_json(contract_path)
    if not contract:
        raise ValueError(f"Cannot read validation contract: {contract_path}")

    manifest_dir = validation_root / contract.get("manifest", {}).get("directory", "00_manifest")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    previous = _read_json(manifest_dir / "validation_manifest.json")
    actual_provenance = _git_provenance(PROJECT_ROOT)
    if provenance:
        actual_provenance.update(provenance)

    stages: dict[str, str] = {}
    stage_statuses: dict[str, str] = {}
    stage_evidence_scopes: dict[str, str] = {}
    pending_stages: list[str] = []
    hidden_verified_flag = False
    for stage in contract.get("stages", []):
        stage_id = stage["id"]
        slug = stage["slug"]
        summary_name = stage["summary"]
        summary_path = validation_root / stage_id / summary_name
        if summary_path.exists():
            stages[slug] = summary_path.relative_to(validation_root).as_posix()
            summary = _read_json(summary_path)
            status = str(summary.get("status", "missing_status"))
            scope = summary.get("evidence_scope", {})
            stage_evidence_scopes[slug] = (
                str(scope.get("classification", "current_or_unspecified"))
                if isinstance(scope, dict)
                else "current_or_unspecified"
            )
            if slug == "hiddenness_tests":
                evidence = summary.get("evidence", {})
                requested_hidden_verified = bool(
                    summary.get("hidden_verified", False)
                    or (evidence.get("hidden_verified", False) if isinstance(evidence, dict) else False)
                )
                metadata_errors = validate_hiddenness_promotion_metadata(summary.get("run_metadata"))
                hidden_verified_flag = bool(requested_hidden_verified and not metadata_errors)
        else:
            stages[slug] = "pending"
            status = "missing_summary"
            stage_evidence_scopes[slug] = "missing_summary"
        stage_statuses[slug] = status
        if status not in CLOSED_STAGE_STATUSES:
            pending_stages.append(slug)

    parameters = main_parameters
    if parameters is None:
        previous_parameters = previous.get("main_parameters")
        parameters = previous_parameters if isinstance(previous_parameters, dict) else {}

    report_config = contract.get("final_report", {})
    report_source = str(report_config.get("source", "validation/final_validation_report.tex"))
    report_compiled = str(report_config.get("compiled", "validation/final_validation_report.pdf"))
    source_path = Path(report_source)
    compiled_path = Path(report_compiled)
    source_path = source_path if source_path.is_absolute() else PROJECT_ROOT / source_path
    compiled_path = compiled_path if compiled_path.is_absolute() else PROJECT_ROOT / compiled_path
    report_files_exist = source_path.exists() and compiled_path.exists()
    if pending_stages:
        final_report_status = "pending_full_protocol"
    elif report_files_exist:
        final_report_status = "completed"
    else:
        final_report_status = "pending_final_report_generation"

    # Determine overall state from the stages' statuses
    overall_state = "candidate_attractor"
    if stage_statuses.get("seed_generation") in ("completed", "passed"):
        overall_state = "seed_found"
    if stage_statuses.get("continuation") in ("completed", "passed"):
        overall_state = "candidate_attractor"
    if stage_statuses.get("dynamic_reference") in ("completed", "passed"):
        overall_state = "chaotic_candidate"
    if stage_statuses.get("robustness") in ("completed", "passed"):
        overall_state = "hidden_compatible"
    if "hiddenness_tests" in stage_statuses:
        if hidden_verified_flag:
            overall_state = "hidden_verified"
        else:
            overall_state = "hidden_compatible"

    from ..references.validator import validate_bibliography_manifest
    claims_manifest_path = PROJECT_ROOT / "references" / "claims_manifest.yaml"
    bib_res = validate_bibliography_manifest(claims_manifest_path, strict=False)

    manifest = {
        "validation_id": validation_id,
        "repository_commit": actual_provenance["repository_commit"],
        "package_version": _package_version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "main_system": main_system,
        "main_parameters": parameters,
        "schema_version": str(contract.get("schema_version", "1.0")),
        "protocol_version": str(contract.get("protocol_version", "caputo_hidden_attractors_v1")),
        "state": overall_state,
        "stages": stages,
        "stage_statuses": stage_statuses,
        "stage_evidence_scopes": stage_evidence_scopes,
        "pending_stages": pending_stages,
        "failed_or_incomplete_stages": pending_stages,
        "final_report": {
            "status": final_report_status,
            "source": report_source,
            "compiled": report_compiled,
        },
        "bibliographic_validation": {
            "status": bib_res["bibliographic_validation_status"],
            "claims_total": bib_res["claims_total"],
            "references_used": bib_res["references_used"],
            "missing_claim_references": [c.get("claim_id") for c in bib_res.get("claims_missing_references", [])],
            "traceability_manifest": claims_manifest_path.relative_to(PROJECT_ROOT).as_posix() if claims_manifest_path.exists() else "references/claims_manifest.yaml"
        },
        "dirty": bool(actual_provenance.get("working_tree_dirty", False)),
    }
    for key in ("working_tree_dirty", "working_tree_diff_sha256", "implementation_sources_sha256"):
        if key in actual_provenance:
            manifest[key] = actual_provenance[key]

    with (manifest_dir / "validation_manifest.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    with (manifest_dir / "environment.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps({"python": sys.version, "platform": platform.platform()}, indent=2) + "\n")
    with (manifest_dir / "software_versions.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps({"numpy": np.__version__, "matplotlib": matplotlib.__version__}, indent=2) + "\n")
    return manifest


__all__ = [
    "CLOSED_STAGE_STATUSES",
    "DEFAULT_CONTRACT",
    "DEFAULT_VALIDATION_ROOT",
    "regenerate_validation_manifest",
]
