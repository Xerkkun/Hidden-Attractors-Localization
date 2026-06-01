"""Opt-in exhaustive Fischer 2020 cloned-dynamics benchmark execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PYTHON = ROOT / "validation" / "python"
if str(VALIDATION_PYTHON) not in sys.path:
    sys.path.insert(0, str(VALIDATION_PYTHON))

from cloned_dynamics_benchmarks import run_benchmark_row  # noqa: E402


BENCHMARKS = ROOT / "validation" / "lyapunov_benchmarks" / "fractional_cloned_dynamics_abm_gs_published"


def _published_rows() -> list[tuple[str, int, dict, dict]]:
    rows = []
    for path in sorted(BENCHMARKS.glob("*.yaml")):
        case = yaml.safe_load(path.read_text(encoding="utf-8"))
        for row_index, row in enumerate(case["expected"]["rows"]):
            rows.append((path.name, row_index, case, row))
    return rows


@pytest.mark.slow
@pytest.mark.published
@pytest.mark.skipif(
    os.environ.get("RUN_PUBLISHED_CLONED_ALL") != "1",
    reason="Set RUN_PUBLISHED_CLONED_ALL=1 to run all Fischer 2020 rows",
)
@pytest.mark.parametrize("filename,row_index,case,row", _published_rows())
def test_fischer_published_all_rows(filename: str, row_index: int, case: dict, row: dict) -> None:
    record, _ = run_benchmark_row(case, row, case_file=filename, row_index=row_index)
    assert record["sign_match"] is True
    assert record["abs_error"][0] < case["expected"]["tolerance_abs_initial"]
    assert record["status"] in {
        "published_benchmark_passed_quantitative",
        "published_benchmark_passed_sign_pattern",
    }
