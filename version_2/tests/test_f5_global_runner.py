"""End-to-end fast contract for the global F5 runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "validation" / "chaos_validation" / "dynamics_diagnostics"


def test_global_f5_runner_fast_reuses_poincare_and_writes_outputs() -> None:
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
    for relative in (
        "boundedness/boundedness_diagnostics_summary.json",
        "zero_one/zero_one_diagnostics_summary.json",
        "psd_fft/psd_fft_diagnostics_summary.json",
        "poincare/poincare_diagnostics_summary.json",
    ):
        assert (DIAGNOSTICS / relative).is_file()
    summary = json.loads((DIAGNOSTICS / "f5_diagnostics_summary.json").read_text(encoding="utf-8"))
    assert summary["final_f5_status"] == "f5_diagnostics_structured_outputs_ready"
    assert summary["certifications"] == {"chaos_verified": False, "hidden_verified": False}
    assert summary["combined_interpretation"]["status"] == "diagnostics_only_not_certification"
