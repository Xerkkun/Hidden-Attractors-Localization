from __future__ import annotations

import csv
from pathlib import Path

from validation.paper07_chua.scripts.summarize_c590_hiddenness import (
    DEFAULT_SOURCE_DIR,
    aggregate_probe_rows,
    build_public_validation_summary,
    read_probe_rows,
    summarize,
    write_outputs,
    write_public_outputs,
)


def test_c590_canonical_probe_counts_preserve_status_semantics() -> None:
    summary = summarize()

    assert summary["total_tests"] == 13_500
    assert summary["total_contacts"] == 610
    assert summary["status_counts"] == {
        "converged_equilibrium_early": 4,
        "diverged": 131,
        "ok": 13_365,
    }
    assert len(summary["source_file_manifest"]) == 6
    assert sum(
        entry["rows"] for entry in summary["source_file_manifest"]
    ) == 13_500
    r200_source = next(
        entry
        for entry in summary["source_file_manifest"]
        if entry["path"].endswith("hiddenness_r200_rows.csv")
    )
    assert r200_source["rows"] == 2_700
    assert r200_source["status_counts"] == {"diverged": 131, "ok": 2_569}
    assert len(r200_source["sha256"]) == 64

    local = summary["local_probe_summary"]
    assert local["tests"] == 8_400
    assert local["contacts"] == 0
    assert local["status_counts"] == {
        "converged_equilibrium_early": 4,
        "ok": 8_396,
    }

    macro = summary["macro_probe_summary"]
    assert macro["tests"] == 5_100
    assert macro["contacts"] == 610
    assert macro["status_counts"] == {"diverged": 131, "ok": 4_969}

    radius_two = next(
        entry for entry in summary["summary_by_radius"] if entry["radius"] == 2.0
    )
    assert radius_two["tests"] == 2_700
    assert radius_two["contacts"] == 588
    assert radius_two["status_counts"] == {"diverged": 131, "ok": 2_569}
    assert radius_two["finite_counts"] == {"false": 0, "true": 2_700}


def test_detailed_grouping_keeps_equilibrium_status_and_contact() -> None:
    rows = read_probe_rows(
        [
            DEFAULT_SOURCE_DIR / "hiddenness_r200_rows.csv",
        ]
    )
    aggregate = aggregate_probe_rows(rows)
    groups = aggregate["summary_by_radius_equilibrium_status_contact"]

    eplus_diverged = next(
        entry
        for entry in groups
        if entry["equilibrium"] == "E+"
        and entry["status"] == "diverged"
        and entry["contact"] is False
    )
    eplus_target = next(
        entry
        for entry in groups
        if entry["equilibrium"] == "E+"
        and entry["status"] == "ok"
        and entry["contact"] is True
    )
    assert eplus_diverged == {
        "radius": 2.0,
        "equilibrium": "E+",
        "status": "diverged",
        "contact": False,
        "finite": True,
        "tests": 37,
    }
    assert eplus_target["tests"] == 150


def test_writer_emits_status_columns_and_detailed_groups(tmp_path: Path) -> None:
    summary = summarize()
    write_outputs(tmp_path, summary)

    with (tmp_path / "summary_by_radius.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    radius_two = next(row for row in rows if row["radius"] == "2.0")
    assert radius_two["ok"] == "2569"
    assert radius_two["diverged"] == "131"
    assert radius_two["full_horizon"] == "2569"

    detailed = tmp_path / "summary_by_radius_equilibrium_status_contact.csv"
    assert detailed.is_file()
    with detailed.open(newline="", encoding="utf-8") as handle:
        detailed_rows = list(csv.DictReader(handle))
    assert any(
        row["radius"] == "2.0"
        and row["equilibrium"] == "E0"
        and row["status"] == "diverged"
        and row["tests"] == "67"
        for row in detailed_rows
    )


def test_public_projection_preserves_status_counts(tmp_path: Path) -> None:
    source_summary = summarize()
    public_summary = build_public_validation_summary(
        source_summary,
        {
            "case_id": "chua_fractional_arctan",
            "source_case_id": "chua_fractional_arctan_c590",
            "claim_scope": "local radii <= 0.3 only",
        },
    )
    write_public_outputs(tmp_path, public_summary, local_max_radius=0.3)

    assert public_summary["case_id"] == "chua_fractional_arctan"
    assert public_summary["macro_probe_summary"]["status_counts"] == {
        "diverged": 131,
        "ok": 4_969,
    }
    with (tmp_path / "hiddenness_decisions.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        decisions = list(csv.DictReader(handle))
    radius_two_ezero = next(
        row
        for row in decisions
        if row["radius"] == "2.0" and row["equilibrium"] == "E0"
    )
    assert radius_two_ezero["diverged"] == "67"
    assert radius_two_ezero["claim_role"] == "macro_audit"
