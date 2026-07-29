from __future__ import annotations

import json

from validation.paper07_chua.scripts.summarize_c590_hiddenness import (
    CANONICAL_ROW_FILES,
    DEFAULT_SOURCE_DIR,
    DEFAULT_VALIDATION_DIR,
)
from validation.paper07_chua.scripts.sync_paper07_evidence import (
    EVIDENCE_ROOT,
    PACKAGE_ROOT,
    artifact_specs,
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


def test_compact_case_manifest_references_existing_evidence() -> None:
    manifest = json.loads(
        (PACKAGE_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    repository_root = PACKAGE_ROOT.parents[2]

    for case in manifest["cases"].values():
        for relative in case["canonical_evidence"]:
            assert (repository_root / relative).is_file(), relative


def test_dedicated_evidence_subtree_contains_only_declared_files() -> None:
    declared = {
        spec.destination.resolve()
        for spec in artifact_specs()
        if spec.destination.is_relative_to(EVIDENCE_ROOT)
    }
    actual = {
        path.resolve()
        for path in EVIDENCE_ROOT.rglob("*")
        if path.is_file()
    }

    assert actual == declared


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
