"""Official Wolfram artifact resolution and algebra-stage promotion tests."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from hidden_attractors.validation import REQUIRED_WOLFRAM_ARTIFACTS, resolve_wolfram_artifacts


SYSTEM_ID = "chua_fractional_saturation"


@pytest.fixture
def artifact_root() -> Path:
    root = Path(__file__).resolve().parents[1] / "outputs" / "test_artifacts" / f"wolfram_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write_generated_artifacts(root: Path, *, passed: bool) -> Path:
    out = root / "outputs" / "wolfram" / SYSTEM_ID
    out.mkdir(parents=True)
    for suffix in REQUIRED_WOLFRAM_ARTIFACTS.values():
        (out / f"{SYSTEM_ID}_{suffix}").write_text("header\n", encoding="utf-8")
    for suffix in ("validation_summary.json", "python_consistency_summary.json"):
        (out / f"{SYSTEM_ID}_{suffix}").write_text(
            json.dumps({"passed": passed}) + "\n",
            encoding="utf-8",
        )
    return out


def test_official_algebra_fails_or_pending_without_wolfram_artifacts(artifact_root: Path) -> None:
    artifacts = resolve_wolfram_artifacts(artifact_root)
    assert artifacts.complete is False
    assert artifacts.source_kind == "missing"


def test_promoted_wolfram_artifacts_are_consumed_by_algebra_validation(artifact_root: Path) -> None:
    out = _write_generated_artifacts(artifact_root, passed=True)
    artifacts = resolve_wolfram_artifacts(artifact_root)
    assert artifacts.complete is True
    assert artifacts.summaries_pass is True
    assert artifacts.source_kind == "generated_prefixed_outputs"
    assert artifacts.files["wolfram_jacobians.csv"] == out / f"{SYSTEM_ID}_jacobians.csv"


def test_algebra_summary_promotes_only_when_wolfram_summaries_pass(artifact_root: Path) -> None:
    _write_generated_artifacts(artifact_root, passed=False)
    artifacts = resolve_wolfram_artifacts(artifact_root)
    assert artifacts.complete is True
    assert artifacts.summaries_pass is False


def test_algebra_summary_does_not_claim_hiddenness_or_chaos() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = (
        root / "validation" / "02_algebraic_validation" / "algebraic_validation_validation_summary.json"
    ).read_text(encoding="utf-8").lower()
    assert "hidden_verified" not in summary
    assert "chaos_verified" not in summary
