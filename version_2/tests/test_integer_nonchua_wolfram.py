"""Reproducibility checks for the non-Chua integer Wolfram validators."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PYTHON = ROOT / "validation" / "python"
if str(VALIDATION_PYTHON) not in sys.path:
    sys.path.insert(0, str(VALIDATION_PYTHON))

from compare_integer_nonchua_wolfram import compare_all  # noqa: E402


CASES = {
    "kalman_fitts_integer": "10.1016/j.ifacol.2019.11.747",
    "mavpd_integer": "10.3390/math11030591",
    "pll_lead_lag_integer": "10.1109/ICUMT.2015.7382409",
}


@pytest.mark.hygiene
@pytest.mark.parametrize("system_id,doi", CASES.items())
def test_nonchua_wolfram_case_is_source_anchored(system_id: str, doi: str) -> None:
    path = ROOT / "validation" / "wolfram" / "cases" / f"{system_id}.wl"
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    assert doi in text
    assert "main.tex" not in lowered
    assert "local_reports" not in lowered
    assert "comparacion_dynamicalsystems" not in lowered
    assert '"report_input_used" -> false' in lowered


@pytest.mark.wolfram
def test_generated_nonchua_wolfram_outputs_match_python() -> None:
    base = ROOT / "validation" / "outputs" / "wolfram"
    required = [
        base / system_id / f"{system_id}_validation_summary.json"
        for system_id in CASES
    ]
    if not all(path.exists() for path in required):
        pytest.skip(
            "non-Chua Wolfram outputs are absent; run "
            "python validation/python/run_wolfram_validations.py --all"
        )
    summary = compare_all(base)
    assert summary["report_input_used"] is False
    assert summary["passed"] is True
    assert all(item["passed"] for item in summary["comparisons"])


@pytest.mark.hygiene
def test_nonchua_wolfram_summaries_do_not_claim_hiddenness() -> None:
    base = ROOT / "validation" / "outputs" / "wolfram"
    for system_id in CASES:
        path = base / system_id / f"{system_id}_validation_summary.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        serialized = json.dumps(payload, sort_keys=True).lower()
        assert payload["report_input_used"] is False
        assert payload["passed"] is True
        assert "hidden_verified" not in serialized
        assert "chaos_verified" not in serialized
