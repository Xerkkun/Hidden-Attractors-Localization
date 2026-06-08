# -*- coding: utf-8 -*-
import subprocess
import sys
from pathlib import Path
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "chua_nonsmooth_biased_hidden_attractor"
RUN_SCRIPT = EXAMPLE_DIR / "run_example.py"

def test_workflow_imports():
    """Verify that all BDF step functions can be imported from the library."""
    from hidden_attractors.workflows.biased_chua import (
        run_centered_reference,
        run_biased_df_search,
        run_hiddenness_verification,
        run_extended_hiddenness,
        run_summarize_and_plot,
    )
    assert run_centered_reference is not None
    assert run_biased_df_search is not None
    assert run_hiddenness_verification is not None
    assert run_extended_hiddenness is not None
    assert run_summarize_and_plot is not None

def test_run_example_help():
    """Verify that run_example.py parses options and shows help successfully."""
    cmd = [sys.executable, str(RUN_SCRIPT), "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "run_example.py" in res.stdout or "PASO" in res.stdout or "argument" in res.stdout

def test_run_example_step5_quick():
    """Verify executing Step 5 through the orchestrator runs without error."""
    # Step 5 is safe to run standalone as a smoke test because it only summarizes/plots
    cmd = [sys.executable, str(RUN_SCRIPT), "--quick", "--steps", "5"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    # Exits successfully or handles missing files cleanly without crashing
    assert res.returncode == 0
