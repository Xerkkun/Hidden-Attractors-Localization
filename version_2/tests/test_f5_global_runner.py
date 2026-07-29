"""End-to-end fast contract for the global F5 runner."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_global_f5_runner_fast_reuses_poincare_and_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_root = tmp_path / "validation"
    shutil.copytree(
        ROOT / "validation" / "chaos_validation",
        validation_root / "chaos_validation",
    )
    monkeypatch.setenv("HIDDEN_ATTRACTORS_VALIDATION_ROOT", str(validation_root))
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "validation" / "python" / "run_f5_dynamics_diagnostics.py"),
            "--all",
            "--use-existing-poincare",
            "--fast",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    diagnostics = validation_root / "chaos_validation" / "dynamics_diagnostics"
    for relative in (
        "boundedness/boundedness_diagnostics_summary.json",
        "zero_one/zero_one_diagnostics_summary.json",
        "psd_fft/psd_fft_diagnostics_summary.json",
        "poincare/poincare_diagnostics_summary.json",
    ):
        assert (diagnostics / relative).is_file()
    summary = json.loads((diagnostics / "f5_diagnostics_summary.json").read_text(encoding="utf-8"))
    assert summary["final_f5_status"] == "f5_diagnostics_structured_outputs_ready"
    assert summary["certifications"] == {"chaos_verified": False, "hidden_verified": False}
    assert summary["combined_interpretation"]["status"] == "diagnostics_only_not_certification"
