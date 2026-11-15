from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation.paper07_chua.scripts import sync_paper07_evidence as evidence_module

from validation.paper07_chua.scripts.summarize_c590_hiddenness import (
    CANONICAL_ROW_FILES,
    DEFAULT_SOURCE_DIR,
    DEFAULT_VALIDATION_DIR,
)
from validation.paper07_chua.scripts.sync_paper07_evidence import (
    EVIDENCE_ROOT,
    PACKAGE_ROOT,
    artifact_specs,
    canonical_artifact_metadata,
    declared_evidence_destinations,
    restored_artifact_records,
    verify_restored_artifacts,
    verify_package,
)


def test_paper07_evidence_inventory_is_finite_and_compact() -> None:
    specs = artifact_specs()
    destinations = [spec.destination.resolve() for spec in specs]

    assert len(specs) == 43
    assert len(destinations) == len(set(destinations))
    assert {spec.group for spec in specs} == {
        "c590_hiddenness_rows",
        "nonsmooth_corrected",
        "probe_story_trajectories",
    }

    forbidden_tokens = (
        "checkpoint",
        "trajectory_full",
        "trajectory_posttransient",
    )
    forbidden_suffixes = {".png", ".pdf"}
    for spec in specs:
        destination_text = spec.destination.as_posix().lower()
        assert not any(token in destination_text for token in forbidden_tokens)
        assert spec.destination.suffix.lower() not in forbidden_suffixes


def test_manifest_hashes_verify_without_ignored_outputs() -> None:
    result = verify_package()

    assert result["status"] == "verified"
    assert result["artifact_count"] == 43
    assert result["total_size_bytes"] > 0
    assert result["source_parity_verified"] is False
    assert result["restored_artifact_count"] == 4
    assert result["total_declared_artifact_count"] == 47


def test_manifest_text_hashes_are_line_ending_portable(tmp_path) -> None:
    lf = tmp_path / "artifact.json"
    crlf = tmp_path / "artifact-crlf.json"
    lf.write_bytes(b'{\n  "value": 1\n}\n')
    crlf.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')

    assert canonical_artifact_metadata(lf) == canonical_artifact_metadata(crlf)


def test_compact_case_manifest_references_existing_evidence() -> None:
    manifest = json.loads(
        (PACKAGE_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    repository_root = PACKAGE_ROOT.parents[2]

    for case in manifest["cases"].values():
        for relative in case["canonical_evidence"]:
            assert (repository_root / relative).is_file(), relative


def test_dedicated_evidence_subtree_contains_only_declared_files() -> None:
    declared = declared_evidence_destinations()
    actual = {
        path.resolve()
        for path in EVIDENCE_ROOT.rglob("*")
        if path.is_file()
    }

    assert actual == declared


def _restored_sandbox(tmp_path: Path, monkeypatch) -> tuple[Path, list[Path]]:
    """Copy only historical recovered bytes, never mutate the tracked inputs."""
    records = restored_artifact_records()
    for record in records:
        original = evidence_module.REPOSITORY_ROOT / record["path"]
        target = tmp_path / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
    manifest = tmp_path / "restored_evidence_manifest.json"
    manifest.write_bytes(evidence_module.RESTORED_MANIFEST_PATH.read_bytes())
    root = tmp_path / "version_2/validation/paper07_chua/evidence"
    monkeypatch.setattr(evidence_module, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(evidence_module, "EVIDENCE_ROOT", root)
    monkeypatch.setattr(evidence_module, "RESTORED_MANIFEST_PATH", manifest)
    return root, [tmp_path / record["path"] for record in records]


def test_restored_evidence_is_a_separate_exact_historical_admission() -> None:
    original = json.loads((PACKAGE_ROOT / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert original["artifact_count"] == 43
    restored = verify_restored_artifacts()
    assert restored == {"artifact_count": 4, "total_size_bytes": 1711042}
    assert not (EVIDENCE_ROOT / "probe_story_trajectories/manifest.json").exists()


def test_restored_artifact_byte_mutation_is_rejected(tmp_path, monkeypatch) -> None:
    _, paths = _restored_sandbox(tmp_path, monkeypatch)
    binary = next(path for path in paths if path.suffix == ".npz")
    modified = bytearray(binary.read_bytes())
    modified[-1] ^= 1
    binary.write_bytes(modified)
    with pytest.raises(ValueError, match="restored SHA-256 mismatch"):
        verify_restored_artifacts()


def test_evidence_pruning_preserves_explicitly_restored_inputs(tmp_path, monkeypatch) -> None:
    root, recovered = _restored_sandbox(tmp_path, monkeypatch)
    unexpected = root / "undeclared.tmp"
    unexpected.write_bytes(b"not an admitted historical artifact")
    removed = evidence_module._prune_evidence_directory(declared_evidence_destinations())
    assert removed == [unexpected.relative_to(tmp_path).as_posix()]
    assert not unexpected.exists()
    assert all(path.is_file() for path in recovered)
    assert verify_restored_artifacts()["artifact_count"] == 4


def test_c590_summarizer_defaults_to_tracked_row_level_evidence() -> None:
    assert DEFAULT_SOURCE_DIR == DEFAULT_VALIDATION_DIR
    assert all(
        (DEFAULT_SOURCE_DIR / name).is_file()
        for name in CANONICAL_ROW_FILES
    )
    assert all(
        (DEFAULT_SOURCE_DIR / name.replace("_rows.csv", "_run_config.json"))
        .is_file()
        if name != "hiddenness_scaled_rows.csv"
        else (DEFAULT_SOURCE_DIR / "scaled_hiddenness_run_config.json").is_file()
        for name in CANONICAL_ROW_FILES
    )
