from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


VERSION_ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = VERSION_ROOT / "docs" / "assets" / "generated_plot_catalog"
MANIFEST_PATH = CATALOG_ROOT / "catalog_results.json"


@pytest.mark.hygiene
@pytest.mark.plotting
def test_plot_catalog_uses_named_real_system_outputs() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["total_public_callables"] == 33
    assert manifest["requested"] == 33
    assert manifest["successful"] == 33
    assert manifest["failed"] == 0
    assert manifest["scientific_evidence"] is False
    assert manifest["input_data_are_real_numerical_outputs"] is True
    assert manifest["catalog_plot_alone_certifies_scientific_claim"] is False

    provenance = manifest["numerical_input_provenance"]
    assert provenance["system_id"] == "chua-nonsmooth"
    assert provenance["source_kind"] == "canonical_library_reintegration"
    assert provenance["claim_scope"] == (
        "reproducible_real_system_numerical_example_not_new_validation_evidence"
    )
    assert provenance["lure_frequency_domain"] == {
        "transfer_convention": "opposite_sign",
        "transfer_definition": "W_code(s)=c^T(P-sI)^(-1)b",
        "harmonic_condition": "1_plus_WN",
        "closure": "W_code(i*omega_0)=-1/N(A_0)",
    }

    rows = manifest["results"]
    assert len(rows) == 33
    assert len({row["function"] for row in rows}) == 33
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["system_id"] == "chua-nonsmooth" for row in rows)
    assert all(
        row["data_policy"] == "real_named_system_numerical_output" for row in rows
    )


@pytest.mark.hygiene
@pytest.mark.plotting
def test_plot_catalog_records_every_generated_png_and_hash() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    records = [
        output
        for row in manifest["results"]
        for output in row["produced_outputs"]
    ]

    assert len(records) == 62
    assert len({record["catalog_png"] for record in records}) == 62
    assert sum(record["output_index"] == 1 for record in records) == 33

    disk_pngs = sorted((CATALOG_ROOT / "examples").glob("*.png"))
    assert len(disk_pngs) == 62

    for record in records:
        path = VERSION_ROOT / record["catalog_png"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]

