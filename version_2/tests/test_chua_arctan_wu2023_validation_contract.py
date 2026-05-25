"""Configuration and algebra JSON checks for the isolated Wu2023 case."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from hidden_attractors.validation.chua_arctan_wu2023 import write_algebra_validation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "chua_arctan_wu2023_caputo.json"


def test_wu2023_config_loads_reported_initial_conditions_and_memory_policies() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["system"] == "fractional_chua_arctan_wu2023"
    assert config["numerical_contract"]["q"] == 0.99
    assert config["numerical_contract"]["h"] == 0.01
    assert config["numerical_contract"]["N"] == 10000
    assert config["numerical_contract"]["memory_policy_options"] == ["full_history", "finite_memory"]
    assert config["numerical_contract"]["memory_length"] == 40.0
    assert config["initial_conditions_reported"]["x0_plus"] == [13.8, 0.7093, -19.8768]
    assert config["initial_conditions_reported"]["x0_minus"] == [-13.8, -0.7093, 19.8768]
    assert config["hiddenness_protocol"]["requires_all_equilibria"] == ["E0", "E+", "E-"]
    assert config["hiddenness_protocol"]["hidden_verified_default"] is False


def test_algebra_writer_creates_isolated_json_without_hidden_verified_label() -> None:
    output_dir = ROOT / "__codex_dir_test" / f"wu2023_{uuid.uuid4().hex}"
    output_path = output_dir / "chua_arctan_wu2023_algebra.json"
    try:
        report = write_algebra_validation(output_path)
        disk = json.loads(output_path.read_text(encoding="utf-8"))

        assert report["status"] == "passed"
        assert set(disk["equilibria"]) == {"E0", "E+", "E-"}
        assert disk["scientific_boundary"]["hidden_verified"] is False
        assert max(row["equilibrium_residual_norm"] for row in disk["equilibria"].values()) < 1.0e-9
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
