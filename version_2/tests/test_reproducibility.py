from __future__ import annotations

import json

from hidden_attractors.protocol_cli import main as protocol_main
from hidden_attractors.reproducibility import (
    collect_seed_metadata,
    metadata_to_jsonable,
    validate_hiddenness_promotion_metadata,
    validate_run_metadata,
    write_run_metadata,
)


def test_write_run_metadata_records_validation_errors(tmp_path, valid_run_metadata) -> None:
    payload = write_run_metadata(tmp_path / "run_metadata.json", valid_run_metadata)
    saved = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert payload == saved
    assert saved["metadata_validation_errors"] == []
    assert validate_run_metadata(saved) == []


def test_hidden_verified_metadata_requires_full_caputo(valid_run_metadata) -> None:
    metadata = metadata_to_jsonable(valid_run_metadata)
    metadata["numerical_contract"]["memory"]["mode"] = "finite_window"
    metadata["numerical_contract"]["memory"]["is_full_caputo"] = False
    errors = validate_hiddenness_promotion_metadata(metadata)
    assert "hidden_verified requires numerical_contract.memory.is_full_caputo=true" in errors


def test_fixed_random_seed_policy_requires_integer_seed(valid_run_metadata) -> None:
    metadata = metadata_to_jsonable(valid_run_metadata)
    metadata["random_seed"] = None
    assert "random_seed must be an integer when random_seed_policy=fixed_reproducible" in validate_run_metadata(metadata)


def test_seed_metadata_accepts_maintained_legacy_seed_alias() -> None:
    seed = collect_seed_metadata(
        {"candidate_id": "c1", "family": "lure_classical_centered", "seed": [1.0, 2.0, 3.0]},
        source="pytest",
    )
    assert seed is not None
    assert seed.x0 == [1.0, 2.0, 3.0]


def test_protocol_cli_degrades_strong_label_without_lure_and_seed_metadata(tmp_path) -> None:
    contract_path = tmp_path / "contract.json"
    payload_path = tmp_path / "payload.json"
    output_path = tmp_path / "hiddenness_summary.json"
    contract_path.write_text(
        json.dumps(
            {
                "q": 0.9998,
                "h": 0.01,
                "t_final": 100.0,
                "t_transient": 20.0,
                "backend": "efork_c",
                "memory_policy": "full_history",
            }
        ),
        encoding="utf-8",
    )
    payload_path.write_text(
        json.dumps(
            {
                "verdict": "hidden_verified",
                "state": "hidden_verified",
                "hiddenness_test_result": {
                    "tested_equilibria": ["E0"],
                    "tested_radii": [0.01],
                    "neighborhood_sampling_mode": "ball",
                    "target_contacts": 0,
                    "numerical_failures": 0,
                    "basin_planes": ["xy_close", "xy_large", "xz_close", "xz_large", "yz_close", "yz_large"],
                    "reference_was_robust": True,
                },
            }
        ),
        encoding="utf-8",
    )
    assert protocol_main(
        [
            "hiddenness",
            "--contract",
            str(contract_path),
            "--payload",
            str(payload_path),
            "--output",
            str(output_path),
        ]
    ) == 0
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["verdict"] == "compatible_with_hiddenness_under_tested_radii"
    assert summary["state"] == "hidden_compatible"
    assert (tmp_path / "run_metadata.json").exists()
