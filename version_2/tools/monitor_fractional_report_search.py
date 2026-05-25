#!/usr/bin/env python3
"""Monitor independent fractional-report searches and launch one expansion.

This tool does not change numerical acceptance rules and does not promote
validation artifacts. It waits for exploratory runs, records a readable
summary, and launches increasingly fine fallback searches only after every
coarser permitted search has completed without valid candidates.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STILL_ACTIVE = 259


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_id_from_job(job: dict[str, Any]) -> str:
    command = str(job.get("command", ""))
    match = re.search(r"--run-id\s+(\S+)", command)
    if match:
        return match.group(1)
    raise ValueError(f"Job has no --run-id: {command}")


def step_size_from_job(job: dict[str, Any]) -> float:
    command = str(job.get("command", ""))
    match = re.search(r"--h\s+(\S+)", command)
    if match:
        return float(match.group(1))
    raise ValueError(f"Job has no --h: {command}")


def is_running(pid: int) -> bool:
    if sys.platform != "win32":
        try:
            import os

            os.kill(int(pid), 0)
            return True
        except OSError:
            return False
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return int(code.value) == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def load_jobs(manifest_path: Path) -> list[dict[str, Any]]:
    payload = read_json(manifest_path)
    return list(payload if isinstance(payload, list) else payload.get("launched", []))


def inspect_job(job: dict[str, Any]) -> dict[str, Any]:
    run_id = run_id_from_job(job)
    h = step_size_from_job(job)
    approved_step = h in {0.01, 0.005, 0.001}
    run_root = OUTPUTS / run_id
    metadata_path = run_root / "run_metadata.json"
    stderr_path = Path(str(job["stderr"]))
    stdout_path = Path(str(job["stdout"]))
    metadata = read_json(metadata_path) if metadata_path.exists() else None
    branch_counts: dict[str, int] = {}
    selected_ids: dict[str, list[str]] = {}
    if metadata:
        for branch_id in ("full_history", "finite_window"):
            selected = metadata["branches"][branch_id].get("selected", [])
            branch_counts[branch_id] = len(selected)
            selected_ids[branch_id] = [str(row["candidate_id"]) for row in selected]
    completed_with_selection = bool(
        metadata
        and all(branch_counts.get(branch_id, 0) == 3 for branch_id in ("full_history", "finite_window"))
    )
    running = is_running(int(job["pid"])) if not metadata else False
    if completed_with_selection and approved_step:
        status = "valid_completed"
    elif completed_with_selection:
        status = "preliminary_requires_approved_h_confirmation"
    elif metadata:
        status = "completed_no_valid_selection"
    else:
        status = "running" if running else "failed"
    return {
        "worker": str(job.get("worker", "")),
        "pid": int(job["pid"]),
        "run_id": run_id,
        "run_root": str(run_root),
        "h": h,
        "approved_step": approved_step,
        "status": status,
        "metadata": str(metadata_path) if metadata else "",
        "full_history_selected": branch_counts.get("full_history"),
        "finite_window_selected": branch_counts.get("finite_window"),
        "selected_ids": selected_ids,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "stdout_bytes": stdout_path.stat().st_size if stdout_path.exists() else 0,
        "stderr_bytes": stderr_path.stat().st_size if stderr_path.exists() else 0,
    }


def launch_fallback(output_dir: Path, spec: dict[str, str]) -> dict[str, Any]:
    label = str(spec["label"])
    run_id = f"candidate_nonperiodic_overnight_{label}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = output_dir / f"{label}.out.log"
    stderr_path = output_dir / f"{label}.err.log"
    cmd = [
        sys.executable,
        "-m",
        "hidden_attractors.workflows.fractional_report_run",
        "--run-id",
        run_id,
        "--h",
        str(spec["h"]),
        "--memory-length",
        str(spec["memory_length"]),
        "--t-final",
        str(spec["t_final"]),
        "--biased-lhs-count",
        str(spec["biased_lhs_count"]),
        "--biased-keep-best",
        str(spec["biased_keep_best"]),
        "--skip-validation-promotion",
    ]
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=stdout,
            stderr=stderr,
            **kwargs,
        )
    job = {
        "worker": f"overnight_{label}",
        "pid": int(proc.pid),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "command": subprocess.list2cmdline(cmd),
    }
    write_json(output_dir / f"{label}_launch_manifest.json", [job])
    return job


def render_summary(output_dir: Path, inspections: list[dict[str, Any]], fallbacks_launched: list[str]) -> None:
    valid = [row for row in inspections if row["status"] == "valid_completed"]
    lines = [
        "# Overnight Candidate Search Status",
        "",
        f"- Updated: `{now_text()}`",
        f"- Valid completed runs: `{len(valid)}`",
        f"- Fallbacks launched: `{', '.join(fallbacks_launched) if fallbacks_launched else 'none'}`",
        "- Acceptance rule: post-continuation nonperiodic selection remains mandatory.",
        "- Accepted integration steps for new candidate claims: `h=0.01`, `h=0.005`, or `h=0.001`.",
        "- Policy: `h=0.005` starts only after every `h=0.01` run fails; `h=0.001` starts only after all previous runs fail.",
        "- Validation promotion: disabled for exploratory parallel runs.",
        "",
        "| Worker | h | Status | Full selected | Window selected | Run root |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in inspections:
        lines.append(
            f"| {row['worker']} | {row['h']} | {row['status']} | {row['full_history_selected'] if row['full_history_selected'] is not None else ''} "
            f"| {row['finite_window_selected'] if row['finite_window_selected'] is not None else ''} | `{row['run_root']}` |"
        )
    if valid:
        lines.extend(["", "## Candidate Runs"])
        for row in valid:
            lines.append(f"- `{row['run_id']}`: `{json.dumps(row['selected_ids'], ensure_ascii=True)}`")
        lines.append("")
        lines.append("These runs contain generated figures and candidate evidence under outputs/; promote only after review.")
    else:
        lines.extend(
            [
                "",
                "No completed run with three nonperiodic selections in both memory branches has been found yet.",
            ]
        )
    (output_dir / "overnight_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output_dir / "overnight_status.json",
        {"updated": now_text(), "fallbacks_launched": fallbacks_launched, "jobs": inspections},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor official exploratory fractional-report candidate searches.")
    parser.add_argument("--manifest", required=True, help="Manifest JSON for current independent processes.")
    parser.add_argument("--output-dir", required=True, help="Directory for monitoring summaries and any expansion logs.")
    parser.add_argument("--poll-sec", type=float, default=120.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_jobs = load_jobs(Path(args.manifest).resolve())
    fallback_specs = [
        {
            "label": "fallback_h0005_lm40",
            "h": "0.005",
            "memory_length": "40",
            "t_final": "300",
            "biased_lhs_count": "192",
            "biased_keep_best": "48",
        },
        {
            "label": "fallback_h0001_lm40",
            "h": "0.001",
            "memory_length": "40",
            "t_final": "300",
            "biased_lhs_count": "96",
            "biased_keep_best": "24",
        },
    ]
    fallback_jobs: list[dict[str, Any]] = []
    fallbacks_launched: list[str] = []
    while True:
        jobs = list(base_jobs) + list(fallback_jobs)
        inspections = [inspect_job(job) for job in jobs]
        render_summary(output_dir, inspections, fallbacks_launched)
        if any(row["status"] == "valid_completed" for row in inspections):
            return
        if all(
            row["status"]
            in {"completed_no_valid_selection", "failed", "preliminary_requires_approved_h_confirmation"}
            for row in inspections
        ):
            if len(fallback_jobs) < len(fallback_specs):
                spec = fallback_specs[len(fallback_jobs)]
                fallback_jobs.append(launch_fallback(output_dir, spec))
                fallbacks_launched.append(spec["label"])
                continue
            return
        time.sleep(max(10.0, float(args.poll_sec)))


if __name__ == "__main__":
    main()
