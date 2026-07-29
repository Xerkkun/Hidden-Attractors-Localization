"""Synchronize and verify the closed numerical evidence for paper 07.

This command inventories only the fixed, result-level artifacts retained in
``validation`` and records a SHA-256 manifest. Verification uses only tracked
closed validation evidence.

Examples
--------
Update the tracked package after completing the fixed validation runs::

    python validation/paper07_chua/scripts/sync_paper07_evidence.py --sync

Verify the tracked package in a clean checkout::

    python validation/paper07_chua/scripts/sync_paper07_evidence.py --verify

Also confirm byte-for-byte parity with locally retained workflow outputs::

    python validation/paper07_chua/scripts/sync_paper07_evidence.py --verify --verify-sources
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VERSION_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = VERSION_ROOT.parent
PACKAGE_ROOT = VERSION_ROOT / "validation" / "paper07_chua"
EVIDENCE_ROOT = PACKAGE_ROOT / "evidence"
MANIFEST_PATH = PACKAGE_ROOT / "evidence_manifest.json"

NONSMOOTH_SOURCE = EVIDENCE_ROOT / "nonsmooth_corrected"
C590_HIDDENNESS_VALIDATION = (
    VERSION_ROOT / "validation" / "chua_fractional_arctan_c590"
)


@dataclass(frozen=True)
class ArtifactSpec:
    """One source-to-tracked-destination projection."""

    group: str
    source: Path
    destination: Path


def _spec(
    group: str,
    source_root: Path,
    source_relative: str,
    destination_root: Path,
    destination_relative: str | None = None,
) -> ArtifactSpec:
    destination_relative = destination_relative or source_relative
    return ArtifactSpec(
        group=group,
        source=source_root / source_relative,
        destination=destination_root / destination_relative,
    )


def artifact_specs() -> tuple[ArtifactSpec, ...]:
    """Return the complete, intentionally finite validation evidence set."""

    nonsmooth_destination = EVIDENCE_ROOT / "nonsmooth_corrected"
    probe_story_destination = EVIDENCE_ROOT / "probe_story_trajectories"

    nonsmooth_files = (
        "candidate_and_reference.json",
        "continuation_stages.csv",
        "numerical_contract.json",
        "probe_runs.csv",
        "probe_summary.csv",
        "result.json",
        "target_reproduction_runs.csv",
        "extended_first_contact_clean/extended_numerical_contract.json",
        "extended_first_contact_clean/extended_probe_plan.csv",
        "extended_first_contact_clean/extended_probe_runs.csv",
        "extended_first_contact_clean/extended_probe_summary.csv",
        "extended_first_contact_clean/extended_result.json",
        "extended_first_contact_clean/target_cloud_nn_sample.csv",
    )
    hiddenness_files = (
        "hiddenness_scaled_rows.csv",
        "scaled_hiddenness_run_config.json",
        "hiddenness_r003_rows.csv",
        "hiddenness_r003_run_config.json",
        "hiddenness_r010_rows.csv",
        "hiddenness_r010_run_config.json",
        "hiddenness_r030_rows.csv",
        "hiddenness_r030_run_config.json",
        "hiddenness_r100_rows.csv",
        "hiddenness_r100_run_config.json",
        "hiddenness_r200_rows.csv",
        "hiddenness_r200_run_config.json",
    )
    probe_story_files = (
        "arctan/hiddenness_r100_rows_E0_r00_d000.csv",
        "arctan/hiddenness_r100_rows_E0_r00_d194.csv",
        "arctan/hiddenness_r100_rows_E0_r00_d228.csv",
        "arctan/hiddenness_r100_rows_Em_r00_d000.csv",
        "arctan/hiddenness_r100_rows_Ep_r00_d000.csv",
        "arctan/hiddenness_scaled_rows_Em_r00_d000.csv",
        "arctan/hiddenness_scaled_rows_Em_r00_d001.csv",
        "arctan/hiddenness_scaled_rows_Ep_r00_d000.csv",
        "arctan/hiddenness_scaled_rows_Ep_r00_d001.csv",
        "nonsmooth/probe_00000_E0.csv",
        "nonsmooth/probe_08800_Ep.csv",
        "nonsmooth/probe_08801_Ep.csv",
        "nonsmooth/probe_12350_Ep.csv",
        "nonsmooth/probe_13407_Ep.csv",
        "nonsmooth/probe_17600_Em.csv",
        "nonsmooth/probe_17602_Em.csv",
        "nonsmooth/probe_21163_Em.csv",
        "nonsmooth/probe_22330_Em.csv",
    )

    specs = [
        _spec(
            "nonsmooth_corrected",
            NONSMOOTH_SOURCE,
            relative,
            nonsmooth_destination,
        )
        for relative in nonsmooth_files
    ]
    specs.extend(
        _spec(
            "c590_hiddenness_rows",
            C590_HIDDENNESS_VALIDATION,
            relative,
            C590_HIDDENNESS_VALIDATION,
        )
        for relative in hiddenness_files
    )
    specs.extend(
        _spec(
            "probe_story_trajectories",
            probe_story_destination,
            relative,
            probe_story_destination,
        )
        for relative in probe_story_files
    )
    return tuple(specs)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _repository_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"artifact path escapes repository root: {path}"
        ) from exc


def _validate_scientific_contracts(nonsmooth_root: Path) -> None:
    """Reject an incomplete or scientifically different validation result."""

    extended_result = _read_json(
        nonsmooth_root
        / "extended_first_contact_clean"
        / "extended_result.json"
    )
    termination = extended_result.get("termination", {})
    destinations = extended_result.get("destination_counts", {})
    expected_extended = {
        "status": "stopped_after_first_contact_radius",
        "samples_total": 17_400,
        "first_contact_radius": 0.3,
        "first_contact_hits": 37,
        "numerical_failure": 0,
    }
    observed_extended = {
        "status": extended_result.get("status"),
        "samples_total": extended_result.get("samples_total"),
        "first_contact_radius": termination.get("first_contact_radius"),
        "first_contact_hits": termination.get("first_contact_hits"),
        "numerical_failure": destinations.get("numerical_failure"),
    }
    if observed_extended != expected_extended:
        raise ValueError(
            "the nonsmooth extended run is not the canonical first-contact "
            f"result: observed={observed_extended}"
        )


def validate_source_contracts() -> None:
    """Refuse to publish an incomplete or scientifically different run."""

    _validate_scientific_contracts(NONSMOOTH_SOURCE)


def _prune_evidence_directory(expected: Iterable[Path]) -> list[str]:
    """Remove stale files only from the dedicated tracked evidence subtree."""

    expected_resolved = {path.resolve() for path in expected}
    removed: list[str] = []
    if not EVIDENCE_ROOT.exists():
        return removed
    for path in sorted(EVIDENCE_ROOT.rglob("*"), reverse=True):
        if path.is_file() and path.resolve() not in expected_resolved:
            removed.append(_repository_relative(path))
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    return sorted(removed)


def _manifest_payload(specs: Iterable[ArtifactSpec]) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    group_bytes: Counter[str] = Counter()
    for spec in specs:
        size = spec.destination.stat().st_size
        group_counts[spec.group] += 1
        group_bytes[spec.group] += size
        artifacts.append(
            {
                "group": spec.group,
                "source_path": _repository_relative(spec.source),
                "path": _repository_relative(spec.destination),
                "size_bytes": size,
                "sha256": sha256_file(spec.destination),
            }
        )
    artifacts.sort(key=lambda item: item["path"])
    groups = {
        group: {
            "files": group_counts[group],
            "size_bytes": group_bytes[group],
        }
        for group in sorted(group_counts)
    }
    return {
        "schema_version": "1.0",
        "package_id": "paper07_chua_tracked_numerical_evidence",
        "path_base": "repository_root",
        "generated_by": "version_2/validation/paper07_chua/scripts/sync_paper07_evidence.py",
        "scientific_scope": (
            "Fixed result-level evidence for the corrected nonsmooth probes "
            "and the radius-limited c590 hiddenness rows, together with the "
            "finite representative trajectories used by the spatial probe "
            "figures."
        ),
        "artifact_count": len(artifacts),
        "total_size_bytes": sum(item["size_bytes"] for item in artifacts),
        "groups": groups,
        "artifacts": artifacts,
    }


def sync_package() -> dict[str, Any]:
    """Copy the selected artifacts and write their tracked hash manifest."""

    specs = artifact_specs()
    missing = [spec.source for spec in specs if not spec.source.is_file()]
    if missing:
        paths = ", ".join(_repository_relative(path) for path in missing)
        raise FileNotFoundError(f"missing source artifacts: {paths}")
    validate_source_contracts()

    for spec in specs:
        spec.destination.parent.mkdir(parents=True, exist_ok=True)
        if spec.source.resolve() != spec.destination.resolve():
            shutil.copy2(spec.source, spec.destination)
    removed = _prune_evidence_directory(
        spec.destination
        for spec in specs
        if spec.destination.is_relative_to(EVIDENCE_ROOT)
    )

    payload = _manifest_payload(specs)
    payload["pruned_stale_files"] = removed
    MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    verify_package(verify_sources=True)
    return payload


def verify_package(*, verify_sources: bool = False) -> dict[str, Any]:
    """Verify manifest completeness, tracked hashes, and optional source parity."""

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"missing evidence manifest: {MANIFEST_PATH}")
    manifest = _read_json(MANIFEST_PATH)
    specs = artifact_specs()
    expected_by_path = {
        _repository_relative(spec.destination): spec for spec in specs
    }
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise ValueError("evidence manifest artifacts must be a list")
    recorded_by_path = {
        str(record.get("path")): record for record in records
    }
    if set(recorded_by_path) != set(expected_by_path):
        missing = sorted(set(expected_by_path) - set(recorded_by_path))
        extra = sorted(set(recorded_by_path) - set(expected_by_path))
        raise ValueError(
            f"manifest inventory mismatch: missing={missing}, extra={extra}"
        )

    for path_text, spec in expected_by_path.items():
        record = recorded_by_path[path_text]
        expected_source = _repository_relative(spec.source)
        if record.get("group") != spec.group:
            raise ValueError(f"group mismatch for {path_text}")
        if record.get("source_path") != expected_source:
            raise ValueError(f"source path mismatch for {path_text}")
        if not spec.destination.is_file():
            raise FileNotFoundError(f"missing tracked artifact: {path_text}")
        size = spec.destination.stat().st_size
        digest = sha256_file(spec.destination)
        if record.get("size_bytes") != size:
            raise ValueError(f"size mismatch for {path_text}")
        if record.get("sha256") != digest:
            raise ValueError(f"SHA-256 mismatch for {path_text}")
        if verify_sources:
            if not spec.source.is_file():
                raise FileNotFoundError(
                    f"missing local source artifact: {expected_source}"
                )
            if spec.source.stat().st_size != size:
                raise ValueError(f"source size mismatch for {path_text}")
            if sha256_file(spec.source) != digest:
                raise ValueError(f"source SHA-256 mismatch for {path_text}")

    expected_evidence = {
        spec.destination.resolve()
        for spec in specs
        if spec.destination.is_relative_to(EVIDENCE_ROOT)
    }
    actual_evidence = (
        {
            path.resolve()
            for path in EVIDENCE_ROOT.rglob("*")
            if path.is_file()
        }
        if EVIDENCE_ROOT.exists()
        else set()
    )
    if actual_evidence != expected_evidence:
        missing = sorted(
            _repository_relative(path)
            for path in expected_evidence - actual_evidence
        )
        extra = sorted(
            _repository_relative(path)
            for path in actual_evidence - expected_evidence
        )
        raise ValueError(
            f"tracked evidence subtree mismatch: missing={missing}, "
            f"extra={extra}"
        )

    total_size = sum(
        spec.destination.stat().st_size for spec in specs
    )
    if manifest.get("artifact_count") != len(specs):
        raise ValueError("manifest artifact_count is stale")
    if manifest.get("total_size_bytes") != total_size:
        raise ValueError("manifest total_size_bytes is stale")
    group_counts: Counter[str] = Counter()
    group_bytes: Counter[str] = Counter()
    for spec in specs:
        group_counts[spec.group] += 1
        group_bytes[spec.group] += spec.destination.stat().st_size
    expected_groups = {
        group: {
            "files": group_counts[group],
            "size_bytes": group_bytes[group],
        }
        for group in sorted(group_counts)
    }
    if manifest.get("groups") != expected_groups:
        raise ValueError("manifest group inventory is stale")

    _validate_scientific_contracts(
        EVIDENCE_ROOT / "nonsmooth_corrected"
    )
    return {
        "status": "verified",
        "manifest": _repository_relative(MANIFEST_PATH),
        "artifact_count": len(specs),
        "total_size_bytes": total_size,
        "source_parity_verified": verify_sources,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--sync",
        action="store_true",
        help="Copy canonical artifacts from ignored outputs into validation.",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="Verify the tracked package without modifying it (default).",
    )
    parser.add_argument(
        "--verify-sources",
        action="store_true",
        help="Also compare every tracked artifact with its local source.",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    try:
        if args.sync:
            sync_payload = sync_package()
            result = {
                "status": "synchronized_and_verified",
                "manifest": _repository_relative(MANIFEST_PATH),
                "artifact_count": sync_payload["artifact_count"],
                "total_size_bytes": sync_payload["total_size_bytes"],
                "pruned_stale_files": sync_payload[
                    "pruned_stale_files"
                ],
                "source_parity_verified": True,
            }
        else:
            result = verify_package(
                verify_sources=args.verify_sources
            )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
