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
from record_dk2018_published_validation_run import record  # noqa: E402


@pytest.fixture
def artifact_root(tmp_path) -> Path:
    root = tmp_path / f"promotion_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    yield root


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


@pytest.mark.unit
def test_promotion_rejects_missing_case(artifact_root: Path) -> None:
    only_case = next(iter(REQUIRED_CASES))
    runtime = _write_runtime(artifact_root, statuses={only_case: "published_benchmark_passed_quantitative"})
    with pytest.raises(RuntimeError, match="missing cases"):
        promote(runtime, artifact_root / "official")


@pytest.mark.unit
def test_promotion_rejects_smoke_result(artifact_root: Path) -> None:
    runtime = _write_runtime(
        artifact_root,
        statuses={case_id: "published_benchmark_smoke_passed" for case_id in REQUIRED_CASES},
    )
    with pytest.raises(RuntimeError, match="status is"):
        promote(runtime, artifact_root / "official")


@pytest.mark.unit
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


@pytest.mark.unit
def test_record_long_run_keeps_discrepancy_pending_without_promoting_csv(artifact_root: Path) -> None:
    runtime = artifact_root / "runtime"
    runtime.mkdir()
    with (runtime / "benchmark_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "status",
                "validation_run_class",
                "numerical_route",
                "execution_contract",
                "computed_exponents",
                "expected_exponents",
                "absolute_differences",
                "absolute_tolerance",
                "failing_components",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "published_dk2018_lorenz_q0985",
                "status": "published_benchmark_passed_quantitative",
                "validation_run_class": "published_quantitative_long",
                "numerical_route": "native_c",
                "execution_contract": "dk2018_block_restart_abm_gs",
                "computed_exponents": "[-0.0027, -0.0868, -1.6224]",
                "expected_exponents": "[-0.0026, -0.087, -1.6225]",
                "absolute_differences": "[0.0001, 0.0002, 0.0001]",
                "absolute_tolerance": "0.05",
                "failing_components": "[]",
            }
        )
        writer.writerow(
            {
                "case_id": "published_dk2018_rabinovich_fabrikant_q0999",
                "status": "published_benchmark_failed",
                "validation_run_class": "published_quantitative_long",
                "numerical_route": "native_c",
                "execution_contract": "dk2018_block_restart_abm_gs",
                "computed_exponents": "[0.0611, 0.0039, -1.8325]",
                "expected_exponents": "[0.0749, 0.0018, -2.085]",
                "absolute_differences": "[0.0138, 0.0021, 0.2525]",
                "absolute_tolerance": "0.05",
                "failing_components": "['lambda_3']",
            }
        )
    official = artifact_root / "official"
    summary = record(runtime, official)
    assert summary["status"] == "published_benchmarks_pending_reproduced_discrepancy"
    assert summary["local_full_history_qr_status"] == "published_benchmarks_pending_separate_contract_required"
    assert not (official / "published_benchmark_results.csv").exists()
