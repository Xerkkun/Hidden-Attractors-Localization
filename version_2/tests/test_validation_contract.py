"""Tests for validation evidence contract checking."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from hidden_attractors.validation_contract import check_validation_contract


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _case_root(name: str) -> Path:
    root = Path("__codex_dir_test") / f"{name}_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    return root.resolve()


def test_validation_contract_accepts_complete_minimal_tree() -> None:
    tmp_path = _case_root("validation_contract_complete")
    try:
        _assert_complete_minimal_tree(tmp_path)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def _assert_complete_minimal_tree(tmp_path: Path) -> None:
    contract = {
        "output_root": "validation",
        "manifest": {
            "directory": "00_manifest",
            "required_files": ["validation_manifest.json", "environment.json", "software_versions.json"],
            "required_fields": [
                "validation_id",
                "repository_commit",
                "package_version",
                "python_version",
                "platform",
                "main_system",
                "main_parameters",
                "stages",
            ],
        },
        "summary_required_fields": ["stage", "status", "files"],
        "stages": [
            {
                "id": "01_algebra",
                "slug": "algebra",
                "summary": "algebra_validation_summary.json",
                "expected_evidence": ["algebra_validation.md", "equilibria_summary.csv", "matignon_margins.png"],
            }
        ],
        "final_report": {
            "source": "validation/final_validation_report.tex",
            "compiled": "validation/final_validation_report.pdf",
        },
    }
    configs = tmp_path / "configs"
    validation = tmp_path / "validation"
    manifest_dir = validation / "00_manifest"
    stage_dir = validation / "01_algebra"
    configs.mkdir()
    manifest_dir.mkdir(parents=True)
    stage_dir.mkdir()
    contract_path = configs / "validation_contract.json"
    _write_json(contract_path, contract)
    _write_json(
        manifest_dir / "validation_manifest.json",
        {
            "validation_id": "test",
            "repository_commit": "abc",
            "package_version": "0.1.0",
            "python_version": "3.12",
            "platform": "test",
            "main_system": "fractional nonsmooth Chua",
            "main_parameters": {},
            "stages": {"algebra": "01_algebra/algebra_validation_summary.json"},
            "final_report": {"status": "pending"},
        },
    )
    _write_json(manifest_dir / "environment.json", {})
    _write_json(manifest_dir / "software_versions.json", {})
    (stage_dir / "algebra_validation.md").write_text("# Algebra\n", encoding="utf-8")
    (stage_dir / "equilibria_summary.csv").write_text("eq_id,residual\nE0,0\n", encoding="utf-8")
    (stage_dir / "matignon_margins.png").write_bytes(b"fake-image")
    _write_json(
        stage_dir / "algebra_validation_summary.json",
        {
            "stage": "algebra",
            "status": "passed",
            "files": {
                "equilibria_summary": "equilibria_summary.csv",
                "matignon_plot": "matignon_margins.png",
            },
        },
    )

    assert check_validation_contract(contract_path, validation) == []


def test_validation_contract_reports_missing_stage_slug_and_files() -> None:
    tmp_path = _case_root("validation_contract_missing")
    try:
        _assert_missing_stage_slug_and_files(tmp_path)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def _assert_missing_stage_slug_and_files(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    validation = tmp_path / "validation"
    configs.mkdir()
    validation.mkdir()
    contract_path = configs / "validation_contract.json"
    _write_json(
        contract_path,
        {
            "manifest": {"directory": "00_manifest", "required_files": [], "required_fields": []},
            "stages": [{"id": "01_algebra", "summary": "algebra_validation_summary.json"}],
        },
    )

    messages = [issue.message for issue in check_validation_contract(contract_path, validation)]

    assert "stage directory does not exist" in messages
    assert "stage is missing string field 'slug'" in messages
