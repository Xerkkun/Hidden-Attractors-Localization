#!/usr/bin/env python3
"""Run all tests for the final freeze audit and collect results."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hidden_attractors.reproducibility import collect_software_metadata


AUDIT_DIR = PROJECT_ROOT / "validation" / "freeze_audit"
SUMMARY_PATH = AUDIT_DIR / "final_freeze_pytest_summary.json"
STDOUT_PATH = AUDIT_DIR / "final_freeze_pytest_stdout.txt"


def write_placeholder_summary(commit: str, dirty: bool, diff_hash: str | None) -> None:
    """Write a temporary summary file to satisfy bootstrapping tests."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    placeholder = {
        "stage": "final_freeze_audit",
        "status": "running",
        "freeze_ready": False,
        "reason": "audit_in_progress",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "git_diff_sha256": diff_hash,
        "stages": {},
    }
    SUMMARY_PATH.write_text(json.dumps(placeholder, indent=2) + "\n", encoding="utf-8")
    if not STDOUT_PATH.exists():
        STDOUT_PATH.write_text("placeholder stdout\n", encoding="utf-8")


def run_stage(name: str, args: list[str]) -> tuple[int, str]:
    """Run one pytest command and capture output."""
    cmd = [sys.executable, "-m", "pytest"] + args
    print(f"Running stage {name} via: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    stdout_and_stderr = f"=== STAGE: {name} ===\n"
    stdout_and_stderr += f"Command: {' '.join(cmd)}\n"
    stdout_and_stderr += f"Exit code: {result.returncode}\n\n"
    stdout_and_stderr += "--- STDOUT ---\n"
    stdout_and_stderr += result.stdout
    stdout_and_stderr += "\n--- STDERR ---\n"
    stdout_and_stderr += result.stderr
    stdout_and_stderr += "\n\n"
    return result.returncode, stdout_and_stderr


def main() -> None:
    # 1. Collect metadata
    metadata = collect_software_metadata()
    commit = metadata.git_commit
    dirty = metadata.working_tree_dirty
    diff_hash = metadata.git_diff_sha256

    print(f"Git commit: {commit}")
    print(f"Working tree dirty: {dirty}")
    if dirty:
        print(f"Git diff SHA256: {diff_hash}")

    # 2. Write placeholder to prevent test failures on missing/invalid file
    write_placeholder_summary(commit, dirty, diff_hash)

    # 3. Define the stages to run
    stages_to_run = {
        "test_scientific_software_algebra": ["tests/test_scientific_software_algebra.py", "-q"],
        "test_scientific_software_integrators": ["tests/test_scientific_software_integrators.py", "-q"],
        "test_scientific_software_hiddenness": ["tests/test_scientific_software_hiddenness.py", "-q"],
        "test_reproducibility_metadata": ["tests/test_reproducibility_metadata.py", "-q"],
        "test_candidate_gate": ["tests/test_candidate_gate.py", "-q"],
        "test_freeze_audit_json_outputs": ["tests/test_freeze_audit_json_outputs.py", "-q"],
        "regression_tests": ["-m", "regression", "-q"],
        "all_tests": ["-q"],
    }

    results = {}
    combined_log = ""
    overall_passed = True

    # 4. Execute all stages
    for name, args in stages_to_run.items():
        code, log = run_stage(name, args)
        passed = (code == 0)
        if not passed:
            overall_passed = False
        results[name] = {
            "command": "python -m pytest " + " ".join(args),
            "exit_code": code,
            "passed": passed,
        }
        combined_log += log

    # 5. Write final files
    STDOUT_PATH.write_text(combined_log, encoding="utf-8")

    freeze_ready = overall_passed and (commit != "unknown")
    status = "passed" if freeze_ready else "failed"
    reason = None if freeze_ready else ("git_commit_unknown" if commit == "unknown" else "pytest_stage_failed")

    summary_payload = {
        "stage": "final_freeze_audit",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "git_diff_sha256": diff_hash,
        "stages": results,
        "freeze_ready": freeze_ready,
    }
    if reason:
        summary_payload["reason"] = reason

    SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    print(f"Audit completed. Freeze ready: {freeze_ready}")
    sys.exit(0 if freeze_ready else 1)


if __name__ == "__main__":
    main()
