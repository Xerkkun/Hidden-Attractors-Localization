#!/usr/bin/env python3
"""Run all tests for the final freeze audit and collect results.

The committed files under ``validation/freeze_audit`` are promoted release
evidence. For a current local audit, pass ``--output-dir`` so regenerated logs
land in a scratch location such as ``validation_outputs/freeze_audit_current``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hidden_attractors.reproducibility import collect_software_metadata


DEFAULT_AUDIT_DIR = PROJECT_ROOT / "validation" / "freeze_audit"
SUMMARY_FILENAME = "final_freeze_pytest_summary.json"
STDOUT_FILENAME = "final_freeze_pytest_stdout.txt"


def _display_command(args: list[str]) -> str:
    return "python -m pytest " + " ".join(args)


def validate_freeze_summary(payload: dict) -> list[str]:
    """Return policy errors for a freeze-audit summary payload."""
    errors: list[str] = []
    if payload.get("git_commit") == "unknown":
        errors.append("git_commit_unknown")
    if payload.get("working_tree_dirty") is True and payload.get("historical_artifact") is not True:
        errors.append("dirty_tree_requires_historical_artifact")
    if payload.get("freeze_ready") is True and errors:
        errors.append("freeze_ready_true_with_policy_errors")
    return errors


def write_placeholder_summary(
    audit_dir: Path,
    commit: str,
    dirty: bool,
    diff_hash: str | None,
    *,
    historical: bool,
) -> None:
    """Write a temporary summary file to satisfy bootstrapping tests."""
    audit_dir.mkdir(parents=True, exist_ok=True)
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
    if historical:
        placeholder["historical_artifact"] = True
    (audit_dir / SUMMARY_FILENAME).write_text(json.dumps(placeholder, indent=2) + "\n", encoding="utf-8")
    stdout_path = audit_dir / STDOUT_FILENAME
    if not stdout_path.exists():
        stdout_path.write_text("placeholder stdout\n", encoding="utf-8")


def run_stage(name: str, args: list[str]) -> tuple[int, str]:
    """Run one pytest command and capture output."""
    cmd = [sys.executable, "-m", "pytest"] + args
    display_cmd = _display_command(args)
    temp_root = PROJECT_ROOT.parent / ".pytest_tmp" / "freeze_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"TMP": str(temp_root), "TEMP": str(temp_root), "TMPDIR": str(temp_root)})
    print(f"Running stage {name} via: {display_cmd}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=PROJECT_ROOT,
    )
    stdout_and_stderr = f"=== STAGE: {name} ===\n"
    stdout_and_stderr += f"Command: {display_cmd}\n"
    stdout_and_stderr += f"Exit code: {result.returncode}\n\n"
    stdout_and_stderr += "--- STDOUT ---\n"
    stdout_and_stderr += result.stdout or ""
    stdout_and_stderr += "\n--- STDERR ---\n"
    stdout_and_stderr += result.stderr or ""
    stdout_and_stderr += "\n\n"
    return result.returncode, stdout_and_stderr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_AUDIT_DIR,
        help="Directory for summary/log outputs. Use validation_outputs/freeze_audit_current for local reruns.",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="Fail the audit if the working tree is dirty.",
    )
    parser.add_argument(
        "--historical-artifact",
        action="store_true",
        help="Mark the written summary as a historical artifact. Do not use for current release regeneration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    summary_path = audit_dir / SUMMARY_FILENAME
    stdout_path = audit_dir / STDOUT_FILENAME

    metadata = collect_software_metadata()
    commit = metadata.git_commit
    dirty = metadata.working_tree_dirty
    diff_hash = metadata.git_diff_sha256

    print(f"Git commit: {commit}")
    print(f"Working tree dirty: {dirty}")
    if dirty:
        print(f"Git diff SHA256: {diff_hash}")

    write_placeholder_summary(audit_dir, commit, dirty, diff_hash, historical=args.historical_artifact)

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

    for name, stage_args in stages_to_run.items():
        code, log = run_stage(name, stage_args)
        passed = code == 0
        if not passed:
            overall_passed = False
        results[name] = {
            "command": _display_command(stage_args),
            "exit_code": code,
            "passed": passed,
        }
        combined_log += log

    stdout_path.write_text(combined_log, encoding="utf-8")

    clean_tree_ok = (not dirty) or args.historical_artifact
    if args.require_clean and dirty:
        clean_tree_ok = False
    freeze_ready = overall_passed and (commit != "unknown") and clean_tree_ok
    status = "passed" if freeze_ready else "failed"
    if freeze_ready:
        reason = None
    elif commit == "unknown":
        reason = "git_commit_unknown"
    elif dirty and not args.historical_artifact:
        reason = "working_tree_dirty"
    elif args.require_clean and dirty:
        reason = "working_tree_dirty"
    else:
        reason = "pytest_stage_failed"

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
        "output_policy": "historical_promoted_artifact" if args.historical_artifact else "current_clean_audit",
    }
    if args.historical_artifact:
        summary_payload["historical_artifact"] = True
    if reason:
        summary_payload["reason"] = reason
    policy_errors = validate_freeze_summary(summary_payload)
    if policy_errors:
        summary_payload["policy_errors"] = policy_errors

    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    print(f"Audit completed. Freeze ready: {freeze_ready}")
    sys.exit(0 if freeze_ready else 1)


if __name__ == "__main__":
    main()