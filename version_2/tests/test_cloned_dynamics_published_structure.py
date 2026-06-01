"""Structure checks for Fischer 2020 published benchmark YAMLs."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "validation" / "lyapunov_benchmarks" / "fractional_cloned_dynamics_abm_gs_published"
FILES = {
    "fischer2020_jerk_commensurate.yaml",
    "fischer2020_jerk_incommensurate.yaml",
    "fischer2020_financial_commensurate.yaml",
    "fischer2020_financial_incommensurate.yaml",
    "fischer2020_four_wing_commensurate.yaml",
    "fischer2020_four_wing_incommensurate.yaml",
}


def test_six_published_yaml_files_exist() -> None:
    assert {path.name for path in BENCHMARKS.glob("*.yaml")} == FILES


def test_published_yaml_files_contain_complete_extracted_values() -> None:
    for filename in FILES:
        path = BENCHMARKS / filename
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["method_id"] == "fractional_cloned_dynamics_abm_gs_published"
        assert data["benchmark_type"] == "published"
        assert data["reference"]["doi"] == "10.1016/j.apnum.2020.03.027"
        assert data["reference"]["data_complete"] is True
        assert data["integration"]["memory_protocol"] == "published_block_restart"
        assert data["expected"]["rows"]
        assert data["expected"]["sign_pattern_required"] is True
        for row in data["expected"]["rows"]:
            assert len(row["lyapunov"]) == 3
            assert "K01" in row
        text = path.read_text(encoding="utf-8")
        assert "chaos_verified" not in text
        assert "hidden_verified" not in text


def test_qr_variant_yaml_files_are_experimental_comparisons() -> None:
    qr_dir = BENCHMARKS.parent / "fractional_cloned_dynamics_abm_qr"
    assert {path.name for path in qr_dir.glob("*.yaml")} == FILES
    for path in qr_dir.glob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["method_id"] == "fractional_cloned_dynamics_abm_qr"
        assert data["benchmark_type"] == "internal_variant_against_published_data"
        assert data["expected"]["validation_level"] == "experimental_comparison"
        assert "not the published algorithm" in data["note"]
