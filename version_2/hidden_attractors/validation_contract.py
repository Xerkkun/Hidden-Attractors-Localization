"""Validation evidence contract checker.

Stability: internal
    The console command is the supported interface. The Python data structures
    may change as the validation package matures.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from hidden_attractors.reproducibility import (
    validate_hiddenness_promotion_metadata,
    validate_run_metadata,
)
from hidden_attractors.workflows.protocol import validate_global_report_coherence


DEFAULT_CONTRACT = Path("configs") / "validation_contract.json"
DEFAULT_VALIDATION_ROOT = Path("validation")
FIGURE_SUFFIXES = {".png", ".pdf"}


@dataclass(frozen=True)
class ValidationIssue:
    """One contract-check finding."""

    severity: str
    path: Path
    message: str

    def format(self, root: Path) -> str:
        try:
            rel = self.path.relative_to(root)
        except ValueError:
            rel = self.path
        return f"{self.severity}: {rel}: {self.message}"


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(data, dict):
        return None, "JSON root must be an object"
    return data, None


def _has_csv_rows(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return any(row for row in csv.reader(handle))
    except UnicodeDecodeError:
        with path.open("r", newline="") as handle:
            return any(row for row in csv.reader(handle))


def _declared_files(summary: dict[str, Any]) -> Iterable[str]:
    files = summary.get("files")
    if isinstance(files, dict):
        for value in files.values():
            if isinstance(value, str):
                yield value
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        yield item
    elif isinstance(files, list):
        for item in files:
            if isinstance(item, str):
                yield item


def _check_json_fields(path: Path, fields: Iterable[str], severity: str = "ERROR") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    data, error = _read_json(path)
    if error:
        return [ValidationIssue(severity, path, error)]
    assert data is not None
    for field in fields:
        if field not in data:
            issues.append(ValidationIssue(severity, path, f"missing required field '{field}'"))
    if path.name.endswith("_validation_summary.json") or path.name.endswith("_summary.json"):
        for declared in _declared_files(data):
            declared_path = (path.parent / declared).resolve()
            if not declared_path.exists():
                issues.append(ValidationIssue(severity, declared_path, "declared evidence path does not exist"))
    return issues


def _check_manifest_paths(
    manifest_path: Path,
    validation_root: Path,
    pending_slugs: set[str],
    allow_pending: bool = False,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    data, error = _read_json(manifest_path)
    if error:
        return [ValidationIssue("ERROR", manifest_path, error)]
    assert data is not None
    stages = data.get("stages", {})
    if not isinstance(stages, dict):
        return [ValidationIssue("ERROR", manifest_path, "field 'stages' must be an object")]
    for stage_name, relative_path in stages.items():
        if not isinstance(relative_path, str):
            issues.append(ValidationIssue("ERROR", manifest_path, f"stage '{stage_name}' path must be a string"))
            continue
        if relative_path == "pending":
            continue
        target = validation_root / relative_path
        is_pending = stage_name in pending_slugs or allow_pending
        severity = "WARNING" if is_pending else "ERROR"
        if not target.exists():
            issues.append(ValidationIssue(severity, target, f"manifest path for stage '{stage_name}' does not exist"))
    return issues


def _final_report_status(manifest_path: Path) -> str | None:
    if not manifest_path.exists():
        return None
    data, error = _read_json(manifest_path)
    if error or data is None:
        return None
    final_report = data.get("final_report")
    if isinstance(final_report, dict) and isinstance(final_report.get("status"), str):
        return final_report["status"]
    status = data.get("final_report_status")
    return status if isinstance(status, str) else None


def check_validation_contract(
    contract_path: Path = DEFAULT_CONTRACT,
    validation_root: Path = DEFAULT_VALIDATION_ROOT,
    allow_pending: bool = False,
) -> list[ValidationIssue]:
    """Return contract issues for a validation evidence tree."""

    contract_path = contract_path.resolve()
    project_root = contract_path.parent.parent
    validation_root = (project_root / validation_root).resolve() if not validation_root.is_absolute() else validation_root
    issues: list[ValidationIssue] = []

    contract, error = _read_json(contract_path)
    if error:
        return [ValidationIssue("ERROR", contract_path, error)]
    assert contract is not None

    if not validation_root.exists():
        return [ValidationIssue("ERROR", validation_root, "validation root does not exist")]

    manifest_info = contract.get("manifest", {})
    manifest_dir = validation_root / manifest_info.get("directory", "00_manifest")
    if not manifest_dir.exists():
        issues.append(ValidationIssue("ERROR", manifest_dir, "manifest directory does not exist"))

    pending_slugs: set[str] = set()
    manifest_path = manifest_dir / "validation_manifest.json"
    if manifest_path.exists():
        data, _ = _read_json(manifest_path)
        if data is not None:
            # Legacy field migrations for backward compatibility
            migrated = False
            if "final_label" in data and "state" not in data:
                data["state"] = data["final_label"]
                issues.append(ValidationIssue("WARNING", manifest_path, "Migrated legacy 'final_label' to 'state'"))
                warnings.warn(f"{manifest_path}: Migrated legacy 'final_label' to 'state'", DeprecationWarning)
                migrated = True
            if "final_report_status" in data and "state" not in data:
                data["state"] = data["final_report_status"]
                issues.append(ValidationIssue("WARNING", manifest_path, "Migrated legacy 'final_report_status' to 'state'"))
                warnings.warn(f"{manifest_path}: Migrated legacy 'final_report_status' to 'state'", DeprecationWarning)
                migrated = True

            # Extract pending stages
            stages = data.get("stages", {})
            if isinstance(stages, dict):
                for k, v in stages.items():
                    if v == "pending":
                        pending_slugs.add(k)
            pending_stages = data.get("pending_stages", [])
            if isinstance(pending_stages, list):
                for item in pending_stages:
                    if isinstance(item, str):
                        pending_slugs.add(item)

            # Global report coherence check
            try:
                # Also read stage summaries to populate evidence if not already present
                combined_data = dict(data)
                stage_statuses = data.get("stage_statuses", {})
                
                # Check for hiddenness_tests or diagnostics to extract evidence/outputs
                for stage_slug, rel_path in stages.items():
                    if rel_path != "pending":
                        stage_path = validation_root / rel_path
                        if stage_path.exists():
                            stage_summary, _ = _read_json(stage_path)
                            if stage_summary:
                                # merge outputs, metrics or files into combined_data to provide complete context
                                for key in ("outputs", "metrics", "evidence", "files", "run_metadata"):
                                    if key in stage_summary:
                                        if key == "run_metadata":
                                            combined_data[key] = stage_summary[key]
                                            continue
                                        if key not in combined_data:
                                            combined_data[key] = {}
                                        if isinstance(combined_data[key], dict) and isinstance(stage_summary[key], dict):
                                            combined_data[key].update(stage_summary[key])
                
                validate_global_report_coherence(combined_data)
            except ValueError as exc:
                issues.append(ValidationIssue("ERROR", manifest_path, f"Global report coherence check failed: {exc}"))

    for file_name in manifest_info.get("required_files", []):
        path = manifest_dir / file_name
        if not path.exists():
            issues.append(ValidationIssue("ERROR", path, "required manifest file is missing"))
            continue
        if path.suffix == ".json":
            fields = manifest_info.get("required_fields", []) if file_name == "validation_manifest.json" else []
            issues.extend(_check_json_fields(path, fields))

    if manifest_path.exists():
        issues.extend(_check_manifest_paths(manifest_path, validation_root, pending_slugs, allow_pending))

    summary_required = contract.get("summary_required_fields", [])
    for stage in contract.get("stages", []):
        if not isinstance(stage, dict):
            issues.append(ValidationIssue("ERROR", validation_root, "stage entry must be an object"))
            continue
        stage_id = stage.get("id")
        slug = stage.get("slug")
        summary_name = stage.get("summary")
        if not isinstance(stage_id, str):
            issues.append(ValidationIssue("ERROR", validation_root, "stage is missing string field 'id'"))
            continue
        stage_dir = validation_root / stage_id
        if not isinstance(slug, str):
            issues.append(ValidationIssue("ERROR", stage_dir, "stage is missing string field 'slug'"))

        is_pending = slug in pending_slugs or stage_id in pending_slugs or allow_pending
        severity = "WARNING" if is_pending else "ERROR"

        if not stage_dir.exists():
            issues.append(ValidationIssue(severity, stage_dir, "stage directory does not exist"))
            continue
        if not isinstance(summary_name, str):
            issues.append(ValidationIssue(severity, stage_dir, "stage is missing string field 'summary'"))
        else:
            summary_path = stage_dir / summary_name
            if not summary_path.exists():
                issues.append(ValidationIssue(severity, summary_path, "stage summary JSON is missing"))
            else:
                issues.extend(_check_json_fields(summary_path, summary_required, severity=severity))
                summary_data, _ = _read_json(summary_path)
                if isinstance(summary_data, dict) and isinstance(summary_data.get("run_metadata"), dict):
                    metadata_errors = validate_run_metadata(summary_data["run_metadata"])
                    if summary_data.get("metadata_validation_errors") != metadata_errors:
                        issues.append(
                            ValidationIssue(
                                severity,
                                summary_path,
                                "metadata_validation_errors does not match validate_run_metadata(run_metadata)",
                            )
                        )
                    for error in metadata_errors:
                        issues.append(ValidationIssue(severity, summary_path, f"invalid run_metadata: {error}"))
                    if summary_data.get("verdict") in {"hidden_verified", "hidden_verified_only_if_full_protocol_passed", "hiddenness_supported_under_tested_neighborhoods"}:
                        for error in validate_hiddenness_promotion_metadata(summary_data["run_metadata"]):
                            issues.append(ValidationIssue(severity, summary_path, f"invalid hidden_verified metadata: {error}"))
        for file_name in stage.get("expected_evidence", []):
            if not isinstance(file_name, str):
                issues.append(ValidationIssue(severity, stage_dir, "expected evidence entries must be strings"))
                continue
            path = stage_dir / file_name
            if not path.exists():
                issues.append(ValidationIssue(severity, path, "expected evidence file is missing"))
                continue
            if path.suffix.lower() == ".csv" and not _has_csv_rows(path):
                issues.append(ValidationIssue(severity, path, "CSV file is empty"))
            if path.suffix.lower() in FIGURE_SUFFIXES and path.stat().st_size == 0:
                issues.append(ValidationIssue(severity, path, "figure file is empty"))

    final_report = contract.get("final_report", {})
    source = final_report.get("source")
    compiled = final_report.get("compiled")
    if isinstance(source, str) and isinstance(compiled, str):
        source_path = project_root / source
        compiled_path = project_root / compiled
        report_status = _final_report_status(manifest_path)
        is_report_pending = report_status is not None and report_status.startswith("pending")
        if not (source_path.exists() and compiled_path.exists()) and not is_report_pending and not allow_pending:
            issues.append(
                ValidationIssue(
                    "ERROR",
                    source_path,
                    "final report source/PDF missing and manifest does not mark final_report.status='pending'",
                )
            )

    return issues


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check validation evidence against configs/validation_contract.json.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT, help="Path to validation_contract.json.")
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT, help="Path to validation root.")
    parser.add_argument("--allow-pending", action="store_true", help="Allow pending stages to report WARNING instead of ERROR.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for hidden-attractors-check-validation."""
    import warnings
    warnings.warn(
        "Deprecated: use 'hidden-attractors validate contract ...'",
        DeprecationWarning,
        stacklevel=2
    )
    print("Deprecated: use 'hidden-attractors validate contract ...'")
    parser = _build_parser()
    args = parser.parse_args(argv)
    issues = check_validation_contract(args.contract, args.validation_root, args.allow_pending)
    root = args.contract.resolve().parent.parent
    if not issues:
        print("validation contract: passed")
        return 0
    for issue in issues:
        print(issue.format(root))
    return 1 if any(issue.severity == "ERROR" for issue in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
