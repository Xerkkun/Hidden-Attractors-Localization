"""Test verifying the existence and contents of the final freeze audit artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_freeze_audit_artifacts_exist_and_are_valid() -> None:
    script_path = ROOT / "validation" / "python" / "run_final_freeze_audit.py"
    stdout_path = ROOT / "validation" / "freeze_audit" / "final_freeze_pytest_stdout.txt"
    summary_path = ROOT / "validation" / "freeze_audit" / "final_freeze_pytest_summary.json"

    # Verify the script exists
    assert script_path.is_file(), f"Audit script not found at {script_path}"

    # Verify the outputs exist (or exist as placeholders/final files)
    assert stdout_path.is_file(), f"Audit stdout log not found at {stdout_path}"
    assert summary_path.is_file(), f"Audit summary JSON not found at {summary_path}"

    # Verify the summary JSON parses correctly and contains the expected structure
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    
    assert "stage" in payload
    assert "status" in payload
    assert "generated_at_utc" in payload or "audit_timestamp_utc" in payload
    assert "git_commit" in payload
    assert "working_tree_dirty" in payload
    assert "stages" in payload
    assert "freeze_ready" in payload
    assert isinstance(payload["freeze_ready"], bool), "freeze_ready must be a boolean"

    if payload["freeze_ready"]:
        assert payload["status"] == "passed"
        assert payload["git_commit"] != "unknown"
        assert all(stage["exit_code"] == 0 for stage in payload["stages"].values())
        assert all(stage["passed"] is True for stage in payload["stages"].values())
    else:
        # If freeze_ready is false, must have a reason field or at least one failed stage (or empty stages for placeholders)
        assert "reason" in payload or any(not stage["passed"] for stage in payload["stages"].values())
        if payload.get("reason") == "audit_in_progress":
            # Reject placeholder with freeze_ready = true
            assert not payload["freeze_ready"]
