"""Conservative promotion checks for the DK2018 reproduction lane."""

from __future__ import annotations

import csv
import shutil
import sys
import uuid
from pathlib import Path

import pytest


_VALIDATION_PYTHON = Path(__file__).resolve().parents[1] / "validation" / "python"
if str(_VALIDATION_PYTHON) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_PYTHON))

from update_lyapunov_method_validation_status import REQUIRED_CASES, promote  # noqa: E402


@pytest.fixture
def artifact_root() -> Path:
    root = Path(__file__).resolve().parents[1] / "outputs" / "test_artifacts" / f"promotion_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write_runtime(root: Path, *, statuses: dict[str, str]) -> Path:
    runtime = root / "runtime"
    convergence = runtime / "convergence"
    convergence.mkdir(parents=True)
    with (runtime / "benchmark_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "status", "numerical_route", "execution_contract"],
        )
        writer.writeheader()
        for case_id, status in statuses.items():
            writer.writerow(
                {
                    "case_id": case_id,
                    "status": status,
                    "numerical_route": "native_c",
                    "execution_contract": "dk2018_block_restart_abm_gs",
                }
            )
            (convergence / f"{case_id}.csv").write_text("time,lambda_0,lambda_1,lambda_2\n1,0,0,0\n", encoding="utf-8")
    return runtime


def test_promotion_rejects_missing_case(artifact_root: Path) -> None:
    only_case = next(iter(REQUIRED_CASES))
    runtime = _write_runtime(artifact_root, statuses={only_case: "published_benchmark_passed_quantitative"})
    with pytest.raises(RuntimeError, match="missing cases"):
        promote(runtime, artifact_root / "official")


def test_promotion_rejects_smoke_result(artifact_root: Path) -> None:
    runtime = _write_runtime(
        artifact_root,
        statuses={case_id: "published_benchmark_smoke_passed" for case_id in REQUIRED_CASES},
    )
    with pytest.raises(RuntimeError, match="status is"):
        promote(runtime, artifact_root / "official")


def test_promotion_accepts_two_native_quantitative_results(artifact_root: Path) -> None:
    runtime = _write_runtime(
        artifact_root,
        statuses={case_id: "published_benchmark_passed_quantitative" for case_id in REQUIRED_CASES},
    )
    official = artifact_root / "official"
    summary = promote(runtime, official)
    assert summary["status"] == "published_quantitative_validated"
    assert summary["certifications"]["chaos_certified_by_this_pipeline"] is False
    assert summary["certifications"]["hiddenness_certified_by_this_pipeline"] is False
    assert (official / "published_benchmark_results.csv").exists()
    assert (official / "manifest.json").exists()
    assert len(list((official / "convergence").glob("*.csv"))) == 2
