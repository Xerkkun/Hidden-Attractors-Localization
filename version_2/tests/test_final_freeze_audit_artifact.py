"""Test verifying the existence and contents of the final freeze audit artifacts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "validation" / "python" / "run_final_freeze_audit.py"


def _load_freeze_audit_module():
    spec = importlib.util.spec_from_file_location("run_final_freeze_audit", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_freeze_audit_artifacts_exist_and_are_valid() -> None:
    stdout_path = ROOT / "validation" / "freeze_audit" / "final_freeze_pytest_stdout.txt"
    summary_path = ROOT / "validation" / "freeze_audit" / "final_freeze_pytest_summary.json"

    assert SCRIPT_PATH.is_file(), f"Audit script not found at {SCRIPT_PATH}"
    assert stdout_path.is_file(), f"Audit stdout log not found at {stdout_path}"
    assert summary_path.is_file(), f"Audit summary JSON not found at {summary_path}"

    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "stage" in payload
    assert "status" in payload
    assert "generated_at_utc" in payload or "audit_timestamp_utc" in payload
    assert "git_commit" in payload
    assert "working_tree_dirty" in payload
    assert "stages" in payload
    assert "freeze_ready" in payload
    assert isinstance(payload["freeze_ready"], bool), "freeze_ready must be a boolean"

    if payload.get("working_tree_dirty") is True:
        assert payload.get("historical_artifact") is True
        assert "historical_note" in payload
        assert payload.get("output_policy") == "historical_promoted_artifact"

    module = _load_freeze_audit_module()
    assert module.validate_freeze_summary(payload) == []

    if payload["freeze_ready"]:
        assert payload["status"] == "passed"
        assert payload["git_commit"] != "unknown"
        assert all(stage["exit_code"] == 0 for stage in payload["stages"].values())
        assert all(stage["passed"] is True for stage in payload["stages"].values())
    else:
        assert "reason" in payload or any(not stage["passed"] for stage in payload["stages"].values())
        if payload.get("reason") == "audit_in_progress":
            assert not payload["freeze_ready"]


def test_dirty_current_freeze_audit_cannot_be_ready_without_historical_marker() -> None:
    module = _load_freeze_audit_module()
    payload = {
        "stage": "final_freeze_audit",
        "status": "passed",
        "git_commit": "abc123",
        "working_tree_dirty": True,
        "git_diff_sha256": "deadbeef",
        "stages": {"all_tests": {"exit_code": 0, "passed": True}},
        "freeze_ready": True,
    }
    errors = module.validate_freeze_summary(payload)
    assert "dirty_tree_requires_historical_artifact" in errors
    assert "freeze_ready_true_with_policy_errors" in errors


def test_freeze_audit_script_supports_current_scratch_output_path() -> None:
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "--output-dir" in text
    assert "--require-clean" in text
    assert "validation_outputs/freeze_audit_current" in text