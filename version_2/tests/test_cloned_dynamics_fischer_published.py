"""Opt-in long Fischer 2020 benchmark execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PYTHON = ROOT / "validation" / "python"
if str(VALIDATION_PYTHON) not in sys.path:
    sys.path.insert(0, str(VALIDATION_PYTHON))

from cloned_dynamics_benchmarks import load_benchmark_case, run_benchmark_row  # noqa: E402


BENCHMARKS = ROOT / "validation" / "lyapunov_benchmarks" / "fractional_cloned_dynamics_abm_gs_published"


@pytest.mark.slow
@pytest.mark.published
@pytest.mark.skipif(
    os.environ.get("RUN_PUBLISHED_CLONED") != "1",
    reason="Set RUN_PUBLISHED_CLONED=1 to run Fischer 2020 cloned dynamics benchmarks",
)
@pytest.mark.parametrize(
    "filename",
    [
        "fischer2020_jerk_commensurate.yaml",
        "fischer2020_financial_commensurate.yaml",
        "fischer2020_four_wing_commensurate.yaml",
        "fischer2020_jerk_incommensurate.yaml",
        "fischer2020_financial_incommensurate.yaml",
        "fischer2020_four_wing_incommensurate.yaml",
    ],
)
def test_fischer_published_first_row(filename: str) -> None:
    case = load_benchmark_case(BENCHMARKS / filename)
    record, _ = run_benchmark_row(case, case["expected"]["rows"][0])
    assert record["sign_match"] is True
    assert record["abs_error"][0] < 0.15
    assert record["status"] in {
        "published_benchmark_passed_quantitative",
        "published_benchmark_passed_sign_pattern",
    }
