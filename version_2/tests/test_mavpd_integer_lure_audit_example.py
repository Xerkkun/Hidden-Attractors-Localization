"""Contracts for the reproducible integer MAVPD audit example."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "modified_van_der_pol_duffing_integer_lure_audit"
SCRIPT = EXAMPLE / "run_example.py"
CONFIG = EXAMPLE / "reproducibility.yaml"
PROBES = EXAMPLE / "input" / "mavpd_integer_hiddenness_initial_conditions.csv"


def _module():
    spec = importlib.util.spec_from_file_location("mavpd_integer_audit_example", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_yaml_separates_blind_search_from_table_one_posthoc_audit() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert cfg["system"]["q"] == 1.0
    assert cfg["system"]["primary"]["xi"] == 3.1
    assert cfg["system"]["negative_control"]["xi"] == 3.5
    assert cfg["direct_route"]["route"] == "direct_integer_transfer"
    assert cfg["direct_route"]["frequency_grid_used"] is False
    assert cfg["direct_route"]["branch_order"] == [0, 1]
    assert len(cfg["continuation"]["lambda_values"]) == 21
    assert cfg["continuation"]["h"] == 0.002
    assert cfg["posthoc_table_1"]["used_as_search_input"] is False
    assert cfg["fallback_negative"]["trigger"] == "finite_target_contact_blocks_hiddenness"
    assert "probe_radius" not in cfg["negative_audit"]
    assert "probe_direction_ids" not in cfg["negative_audit"]
    assert cfg["negative_audit"]["h"] == cfg["hiddenness"]["h"]
    assert cfg["negative_audit"]["probe_t_burn"] == cfg["hiddenness"]["t_burn"]
    assert (
        cfg["negative_audit"]["probe_t_burn"]
        + cfg["negative_audit"]["probe_t_keep"]
        == cfg["hiddenness"]["t_final"]
    )


@pytest.mark.unit
def test_shared_probe_csv_is_literal_108_condition_contract() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rows = PROBES.read_text(encoding="utf-8").splitlines()
    digest = hashlib.sha256(PROBES.read_bytes()).hexdigest()

    assert len(rows) == 109
    assert rows[0] == "sample_id,equilibrium,radius,direction_id,y1,y2,y3"
    assert digest == cfg["hiddenness"]["expected_sha256"]
    assert cfg["hiddenness"]["expected_rows"] == 108


@pytest.mark.unit
def test_direct_route_recomputes_every_primary_branch_with_branch_zero_first() -> None:
    module = _module()
    cfg = module.load_config(quick=True)
    system = module.build_system(cfg, "primary")
    payload, entries = module.derive_direct_seed_records(system, cfg)

    assert payload["frequency_grid_used"] is False
    assert payload["fallback_used"] is False
    actual_pairs = np.asarray(payload["omega_gain_pairs_all"], dtype=float)
    expected_pairs = np.asarray(
        [
            (10.597523031056207, 0.13970159481860317),
            (13.344755734228839, 0.35187905034269),
        ],
        dtype=float,
    )
    assert actual_pairs.shape == (2, 2)
    np.testing.assert_allclose(
        actual_pairs,
        expected_pairs,
        rtol=0.0,
        atol=2.0e-12,
    )
    assert [(entry["branch_index"], entry["phase"]) for entry in entries] == [
        (0, 0.0),
        (0, pytest.approx(3.141592653589793)),
        (1, 0.0),
        (1, pytest.approx(3.141592653589793)),
    ]
    assert all(entry["published_table_used"] is False for entry in entries)


@pytest.mark.unit
def test_quick_negative_audit_uses_the_same_probe_time_contract_as_primary() -> None:
    module = _module()
    cfg = module.load_config(quick=True)

    assert cfg["negative_audit"]["h"] == cfg["hiddenness"]["h"]
    assert cfg["negative_audit"]["probe_t_burn"] == cfg["hiddenness"]["t_burn"]
    assert (
        cfg["negative_audit"]["probe_t_burn"]
        + cfg["negative_audit"]["probe_t_keep"]
        == cfg["hiddenness"]["t_final"]
    )


@pytest.mark.integration
def test_contract_only_run_writes_to_requested_temporary_directory(tmp_path: Path) -> None:
    module = _module()
    manifest = module.run_selected(
        ["contract"],
        quick=True,
        output_override=tmp_path,
    )

    contract = tmp_path / "00_system_contract.json"
    assert contract.is_file()
    assert manifest["q"] == 1.0
    assert manifest["frequency_grid_used_for_search"] is False
    assert manifest["table_1_used_as_search_input"] is False
