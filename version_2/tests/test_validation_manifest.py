from __future__ import annotations

import json
from pathlib import Path

from hidden_attractors.validation.manifest import regenerate_validation_manifest


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_regenerate_manifest_uses_real_summary_statuses(tmp_path: Path) -> None:
    contract_path = tmp_path / "configs" / "validation_contract.json"
    validation = tmp_path / "validation"
    _write_json(
        contract_path,
        {
            "schema_version": "1.0",
            "protocol_version": "test_protocol",
            "manifest": {"directory": "00_manifest"},
            "stages": [
                {"id": "01_closed", "slug": "closed", "summary": "closed_validation_summary.json"},
                {"id": "02_partial", "slug": "partial", "summary": "partial_validation_summary.json"},
                {"id": "03_missing", "slug": "missing", "summary": "missing_validation_summary.json"},
            ],
            "final_report": {
                "source": "validation/final_validation_report.tex",
                "compiled": "validation/final_validation_report.pdf",
            },
        },
    )
    _write_json(validation / "01_closed" / "closed_validation_summary.json", {"status": "completed"})
    _write_json(
        validation / "02_partial" / "partial_validation_summary.json",
        {"status": "completed_with_lyapunov_pending"},
    )

    manifest = regenerate_validation_manifest(validation, contract_path=contract_path, validation_id="test")

    assert manifest["stages"] == {
        "closed": "01_closed/closed_validation_summary.json",
        "partial": "02_partial/partial_validation_summary.json",
        "missing": "pending",
    }
    assert manifest["stage_statuses"] == {
        "closed": "completed",
        "partial": "completed_with_lyapunov_pending",
        "missing": "missing_summary",
    }
    assert manifest["stage_evidence_scopes"] == {
        "closed": "current_or_unspecified",
        "partial": "current_or_unspecified",
        "missing": "missing_summary",
    }
    assert manifest["pending_stages"] == ["partial", "missing"]
    assert manifest["failed_or_incomplete_stages"] == ["partial", "missing"]
    assert manifest["final_report"]["status"] == "pending_full_protocol"
    assert (validation / "00_manifest" / "environment.json").exists()
    assert (validation / "00_manifest" / "software_versions.json").exists()
