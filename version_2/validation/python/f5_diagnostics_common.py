#!/usr/bin/env python3
"""Shared paths and conservative helpers for F5 runners."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS_ROOT = PROJECT_ROOT / "validation" / "chaos_validation" / "dynamics_diagnostics"
CASE_IDS = [
    "chua_integer_q1_reference",
    "danca2017_chua_fractional_saturation_q09998",
    "wu2023_chua_fractional_arctan_q099",
]
CERTIFICATIONS = {"chaos_verified": False, "hidden_verified": False}
INVARIANTS = {
    "single_indicator_cannot_certify_chaos": True,
    "diagnostics_cannot_certify_hiddenness": True,
    "poincare_cannot_certify_caputo_periodic_orbits": True,
}


def relative(path: Path) -> str:
    return path.relative_to(DIAGNOSTICS_ROOT).as_posix()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        data = value.tolist() if isinstance(value, np.ndarray) else value
        return json.dumps(data, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in values:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


def write_case_readme(path: Path, case_id: str, diagnostic: str) -> None:
    path.write_text(
        f"# {case_id}: {diagnostic}\n\n"
        "Standardized finite-time numerical diagnostic output. This artifact\n"
        "does not certify chaos or hiddenness.\n",
        encoding="utf-8",
    )


def aggregate_state(states: list[str], *, regular: str, chaotic: str, inconclusive: str) -> str:
    if not states:
        return inconclusive
    if chaotic in states:
        return chaotic
    if all(state == regular for state in states):
        return regular
    return inconclusive


__all__ = [
    "CASE_IDS",
    "CERTIFICATIONS",
    "DIAGNOSTICS_ROOT",
    "INVARIANTS",
    "aggregate_state",
    "relative",
    "write_case_readme",
    "write_csv",
    "write_json",
]
