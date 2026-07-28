"""Synchronize and verify the tracked numerical evidence for paper 07.

The long-running workflows write into ignored ``outputs`` directories.  This
command projects only the compact, publication-relevant artifacts into
``validation`` and records a SHA-256 manifest.  Verification of the tracked
package does not require the ignored source outputs to be present.

Examples
--------
Update the tracked package after completing both numerical workflows::

    python tools/sync_paper07_evidence.py --sync

Verify the tracked package in a clean checkout::

    python tools/sync_paper07_evidence.py --verify

Also confirm byte-for-byte parity with locally retained workflow outputs::

    python tools/sync_paper07_evidence.py --verify --verify-sources
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


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = VERSION_ROOT.parent
PACKAGE_ROOT = VERSION_ROOT / "validation" / "paper07_chua"
EVIDENCE_ROOT = PACKAGE_ROOT / "evidence"
MANIFEST_PATH = PACKAGE_ROOT / "evidence_manifest.json"

C590_RECONSTRUCTION_SOURCE = (
    VERSION_ROOT / "outputs" / "paper07_c590_discovery_reconstruction"
)
C590_HIDDENNESS_SOURCE = (
    VERSION_ROOT
    / "outputs"
    / "arctan_hidden_candidate_search"
    / "c590_q09999_seed9_candidate_20260623"
)
NONSMOOTH_SOURCE = (
    REPOSITORY_ROOT / "outputs" / "paper07_nonsmooth_corrected"
)
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
    """Return the complete, intentionally finite publication evidence set."""

    c590_destination = EVIDENCE_ROOT / "c590_reconstruction"
    nonsmooth_destination = EVIDENCE_ROOT / "nonsmooth_corrected"

    c590_files = (
        "search_provenance.json",
        "search_banks.npz",
        "integer_global/screen.json",
        "integer_global/summary.json",
        "integer_global/recorded_selection_i1731.npz",
        "integer_local/screen.json",
        "integer_local/summary.json",
        "integer_local/recorded_selection_c590.npz",
        "variational_shortlist/variational_shortlist.json",
        "variational_shortlist/variational_shortlist.npz",
        "caputo_seed9/summary.json",
        "caputo_seed9/resampled_zero_one.json",
        "caputo_seed9/target.npz",
        "caputo_seed9/target_summary.json",
        "caputo_seed9/q_scan/summary.json",
        "caputo_seed9/q_neighbour/summary.json",
        "caputo_seed9/integer_seed_h_audit/summary.json",
        "caputo_seed9/seed_refinement/seeds.npz",
        "caputo_seed9/seed_refinement/summary.json",
        "caputo_seed9/cross_step/summary.json",
        "caputo_seed9/seed5_long_audit/summary.json",
        "caputo_seed9/survivor_long_audit/summary.json",
    )
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

    specs = [
        _spec(
            "c590_reconstruction",
            C590_RECONSTRUCTION_SOURCE,
            relative,
            c590_destination,
        )
        for relative in c590_files
    ]
    specs.extend(
        _spec(
            "nonsmooth_corrected",
            NONSMOOTH_SOURCE,
            relative,
            nonsmooth_destination,
        )
        for relative in nonsmooth_files
    )
    specs.extend(
        _spec(
            "c590_hiddenness_rows",
            C590_HIDDENNESS_SOURCE,
            relative,
            C590_HIDDENNESS_VALIDATION,
        )
        for relative in hiddenness_files
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


def _validate_scientific_contracts(
    c590_root: Path,
    nonsmooth_root: Path,
) -> None:
    """Reject incomplete or scientifically different evidence directories."""

    global_summary = _read_json(
        c590_root / "integer_global" / "summary.json"
    )
    local_summary = _read_json(
        c590_root / "integer_local" / "summary.json"
    )
    variational_summary = _read_json(
        c590_root
        / "variational_shortlist"
        / "variational_shortlist.json"
    )
    caputo_summary = _read_json(
        c590_root / "caputo_seed9" / "summary.json"
    )
    for name, payload in (
        ("integer_global", global_summary),
        ("integer_local", local_summary),
        ("variational_shortlist", variational_summary),
        ("caputo_seed9", caputo_summary),
    ):
        if payload.get("recorded_regression_matched") is not True:
            raise ValueError(
                f"{name} does not match the archived c590 regression; "
                "the tracked package was not updated"
            )

    local_counts = local_summary.get("counts", {})
    local_labels = local_counts.get(
        "screen_labels_among_distinct_nontrivial", {}
    )
    expected_local = {
        "all_rows": 1_000,
        "distinct_nontrivial_rows": 148,
        "inconclusive_nonperiodic": 87,
        "regular_periodic_rejected": 61,
    }
    observed_local = {
        "all_rows": local_counts.get("all_rows"),
        "distinct_nontrivial_rows": local_counts.get(
            "distinct_nontrivial_rows"
        ),
        "inconclusive_nonperiodic": local_labels.get(
            "inconclusive_nonperiodic"
        ),
        "regular_periodic_rejected": local_labels.get(
            "regular_periodic_rejected"
        ),
    }
    if observed_local != expected_local:
        raise ValueError(
            "integer_local does not preserve the archived 87/61 "
            f"classification: observed={observed_local}"
        )
    selection = local_summary.get("recorded_selection", {})
    if (
        selection.get("zero_based_index") != 590
        or selection.get("rank_among_distinct_nontrivial") != 10
    ):
        raise ValueError(
            "integer_local does not preserve the archived c590 selection"
        )

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
    """Refuse to publish known incomplete or scientifically different runs."""

    _validate_scientific_contracts(
        C590_RECONSTRUCTION_SOURCE,
        NONSMOOTH_SOURCE,
    )


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
        "generated_by": "version_2/tools/sync_paper07_evidence.py",
        "scientific_scope": (
            "Compact evidence for the reported c590 reconstruction, the "
            "corrected nonsmooth probes, and the promoted c590 hiddenness "
            "rows. Checkpoints, figures, exploratory trajectories, and "
            "superseded attempts are intentionally excluded."
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
        EVIDENCE_ROOT / "c590_reconstruction",
        EVIDENCE_ROOT / "nonsmooth_corrected",
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
