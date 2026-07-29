"""Summarize the promoted c590 hiddenness probes from tracked canonical rows.

This command performs no numerical integration.  It reads the six retained
probe tables, validates their row-level schema, and regenerates the JSON and
CSV summaries used by the c590 validation lane.  The default source is inside
``validation`` so the aggregation also works in a clean repository checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VALIDATION_DIR = ROOT / "validation" / "chua_fractional_arctan_c590"
DEFAULT_SOURCE_DIR = DEFAULT_VALIDATION_DIR
DEFAULT_PUBLIC_VALIDATION_DIR = ROOT / "validation" / "chua_fractional_arctan"
CANONICAL_ROW_FILES = (
    "hiddenness_scaled_rows.csv",
    "hiddenness_r003_rows.csv",
    "hiddenness_r010_rows.csv",
    "hiddenness_r030_rows.csv",
    "hiddenness_r100_rows.csv",
    "hiddenness_r200_rows.csv",
)
REQUIRED_COLUMNS = {
    "probe_id",
    "equilibrium",
    "radius",
    "status",
    "contact",
    "outcome",
    "finite",
}
EQUILIBRIUM_ORDER = {"E+": 0, "E-": 1, "E0": 2}


@dataclass(frozen=True)
class ProbeRow:
    """Normalized fields needed by the validation aggregation."""

    source_file: str
    probe_id: str
    radius: float
    equilibrium: str
    status: str
    contact: bool
    outcome: str
    finite: bool


def _parse_bool(value: str, *, field: str, source: Path, line: int) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(
        f"{source}:{line}: {field} must be True or False, got {value!r}"
    )


def read_probe_rows(paths: Sequence[Path]) -> list[ProbeRow]:
    """Read and validate the row-level evidence from one or more CSV files."""

    rows: list[ProbeRow] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError(
                    f"{path}: missing required columns {sorted(missing)}"
                )
            for line, raw in enumerate(reader, start=2):
                try:
                    radius = float(raw["radius"])
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line}: invalid radius {raw['radius']!r}"
                    ) from exc
                if radius <= 0.0:
                    raise ValueError(f"{path}:{line}: radius must be positive")

                equilibrium = raw["equilibrium"].strip()
                status = raw["status"].strip()
                outcome = raw["outcome"].strip()
                probe_id = raw["probe_id"].strip()
                if not equilibrium or not status or not outcome or not probe_id:
                    raise ValueError(
                        f"{path}:{line}: probe_id, equilibrium, status, and "
                        "outcome must be non-empty"
                    )

                contact = _parse_bool(
                    raw["contact"], field="contact", source=path, line=line
                )
                finite = _parse_bool(
                    raw["finite"], field="finite", source=path, line=line
                )
                if contact and outcome != "TARGET":
                    raise ValueError(
                        f"{path}:{line}: contact=True requires outcome=TARGET"
                    )

                rows.append(
                    ProbeRow(
                        source_file=path.name,
                        probe_id=probe_id,
                        radius=radius,
                        equilibrium=equilibrium,
                        status=status,
                        contact=contact,
                        outcome=outcome,
                        finite=finite,
                    )
                )
    if not rows:
        raise ValueError("no probe rows were read")
    return rows


def _ordered_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _display_source_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.parent.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_source_file_manifest(
    source_paths: Sequence[Path],
    rows: Sequence[ProbeRow],
) -> list[dict[str, Any]]:
    """Record hashes and row counts for the exact CSV evidence consumed."""

    manifest: list[dict[str, Any]] = []
    for path in source_paths:
        file_rows = [row for row in rows if row.source_file == path.name]
        summary = _count_summary(file_rows)
        manifest.append(
            {
                "path": _display_source_path(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": summary["tests"],
                "radii": sorted({row.radius for row in file_rows}),
                "contacts": summary["contacts"],
                "status_counts": summary["status_counts"],
            }
        )
    return manifest


def _count_summary(rows: Sequence[ProbeRow]) -> dict[str, Any]:
    statuses = Counter(row.status for row in rows)
    outcomes = Counter(row.outcome for row in rows)
    contacts = Counter(row.contact for row in rows)
    finite = Counter(row.finite for row in rows)
    return {
        "tests": len(rows),
        "contacts": contacts[True],
        "contact_counts": {
            "false": contacts[False],
            "true": contacts[True],
        },
        "finite": finite[True],
        "finite_counts": {
            "false": finite[False],
            "true": finite[True],
        },
        "full_horizon": statuses["ok"],
        "status_counts": _ordered_counter(statuses),
        "outcome_counts": _ordered_counter(outcomes),
    }


def _decision(*, radius: float, contacts: int, local_max_radius: float) -> str:
    if contacts == 0:
        return "no_contact_detected"
    if radius > local_max_radius:
        return "macro_radius_contact_detected"
    return "local_radius_contact_detected"


def aggregate_probe_rows(
    rows: Sequence[ProbeRow],
    *,
    local_max_radius: float = 0.3,
) -> dict[str, Any]:
    """Aggregate rows by radius, equilibrium, status, and contact."""

    radii = sorted({row.radius for row in rows})
    by_radius: list[dict[str, Any]] = []
    by_radius_equilibrium: list[dict[str, Any]] = []
    by_radius_equilibrium_status_contact: list[dict[str, Any]] = []

    for radius in radii:
        radius_rows = [row for row in rows if row.radius == radius]
        radius_summary = {"radius": radius, **_count_summary(radius_rows)}
        radius_summary["decision"] = _decision(
            radius=radius,
            contacts=radius_summary["contacts"],
            local_max_radius=local_max_radius,
        )
        by_radius.append(radius_summary)

        equilibria = sorted(
            {row.equilibrium for row in radius_rows},
            key=lambda value: (EQUILIBRIUM_ORDER.get(value, 99), value),
        )
        for equilibrium in equilibria:
            equilibrium_rows = [
                row for row in radius_rows if row.equilibrium == equilibrium
            ]
            equilibrium_summary = {
                "radius": radius,
                "equilibrium": equilibrium,
                **_count_summary(equilibrium_rows),
            }
            equilibrium_summary["decision"] = _decision(
                radius=radius,
                contacts=equilibrium_summary["contacts"],
                local_max_radius=local_max_radius,
            )
            by_radius_equilibrium.append(equilibrium_summary)

            grouped: Counter[tuple[str, bool, bool]] = Counter(
                (row.status, row.contact, row.finite) for row in equilibrium_rows
            )
            for (status, contact, finite), tests in sorted(
                grouped.items(),
                key=lambda item: (item[0][0], item[0][1], item[0][2]),
            ):
                by_radius_equilibrium_status_contact.append(
                    {
                        "radius": radius,
                        "equilibrium": equilibrium,
                        "status": status,
                        "contact": contact,
                        "finite": finite,
                        "tests": tests,
                    }
                )

    local_rows = [row for row in rows if row.radius <= local_max_radius]
    macro_rows = [row for row in rows if row.radius > local_max_radius]
    return {
        "overall": _count_summary(rows),
        "local": {
            "maximum_radius": local_max_radius,
            "radii": sorted({row.radius for row in local_rows}),
            **_count_summary(local_rows),
        },
        "macro": {
            "radii": sorted({row.radius for row in macro_rows}),
            **_count_summary(macro_rows),
        },
        "summary_by_radius": by_radius,
        "summary_by_radius_equilibrium": by_radius_equilibrium,
        "summary_by_radius_equilibrium_status_contact": (
            by_radius_equilibrium_status_contact
        ),
    }


def build_validation_summary(
    base: dict[str, Any],
    aggregate: dict[str, Any],
    *,
    source_paths: Sequence[Path],
    source_file_manifest: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Merge row-derived counts into the retained validation metadata."""

    payload = dict(base)
    overall = aggregate["overall"]
    local = aggregate["local"]
    macro = aggregate["macro"]
    payload["schema_version"] = "1.1"
    payload["source_rows"] = [
        _display_source_path(path) for path in source_paths
    ]
    payload["source_file_manifest"] = list(source_file_manifest)
    payload["aggregation_contract"] = {
        "script": "version_2/validation/paper07_chua/scripts/summarize_c590_hiddenness.py",
        "simulation_performed": False,
        "source_of_truth": "canonical_probe_csv_rows",
        "grouping_dimensions": [
            "radius",
            "equilibrium",
            "status",
            "contact",
            "finite",
        ],
        "status_semantics": {
            "ok": "integration_reached_the_recorded_full_horizon",
            "converged_equilibrium_early": (
                "integration_terminated_after_the_recorded_equilibrium_criterion"
            ),
            "diverged": (
                "integration_reached_the_recorded_divergence_threshold"
            ),
        },
    }
    payload["total_tests"] = overall["tests"]
    payload["total_contacts"] = overall["contacts"]
    payload["total_finite"] = overall["finite"]
    payload["status_counts"] = overall["status_counts"]
    payload["outcome_counts"] = overall["outcome_counts"]
    payload["zero_contact_local_radii"] = local["radii"]
    payload["zero_contact_max_radius"] = local["maximum_radius"]
    payload["zero_contact_tests"] = local["tests"]
    payload["zero_contact_contacts"] = local["contacts"]
    payload["local_probe_summary"] = local
    payload["macro_review_radii"] = macro["radii"]
    payload["macro_review_tests"] = macro["tests"]
    payload["macro_review_contacts"] = macro["contacts"]
    payload["macro_probe_summary"] = macro
    payload["summary_by_radius"] = aggregate["summary_by_radius"]
    payload["summary_by_radius_equilibrium"] = (
        aggregate["summary_by_radius_equilibrium"]
    )
    payload["summary_by_radius_equilibrium_status_contact"] = (
        aggregate["summary_by_radius_equilibrium_status_contact"]
    )
    return payload


def _counter_value(entry: dict[str, Any], status: str) -> int:
    return int(entry["status_counts"].get(status, 0))


def _write_summary_by_radius(
    path: Path,
    entries: Iterable[dict[str, Any]],
) -> None:
    fieldnames = (
        "radius",
        "tests",
        "contacts",
        "finite",
        "full_horizon",
        "ok",
        "converged_equilibrium_early",
        "diverged",
        "decision",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "radius": entry["radius"],
                    "tests": entry["tests"],
                    "contacts": entry["contacts"],
                    "finite": entry["finite"],
                    "full_horizon": entry["full_horizon"],
                    "ok": _counter_value(entry, "ok"),
                    "converged_equilibrium_early": _counter_value(
                        entry, "converged_equilibrium_early"
                    ),
                    "diverged": _counter_value(entry, "diverged"),
                    "decision": entry["decision"],
                }
            )


def _write_summary_by_radius_equilibrium(
    path: Path,
    entries: Iterable[dict[str, Any]],
) -> None:
    fieldnames = (
        "radius",
        "equilibrium",
        "tests",
        "contacts",
        "finite",
        "full_horizon",
        "ok",
        "converged_equilibrium_early",
        "diverged",
        "decision",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "radius": entry["radius"],
                    "equilibrium": entry["equilibrium"],
                    "tests": entry["tests"],
                    "contacts": entry["contacts"],
                    "finite": entry["finite"],
                    "full_horizon": entry["full_horizon"],
                    "ok": _counter_value(entry, "ok"),
                    "converged_equilibrium_early": _counter_value(
                        entry, "converged_equilibrium_early"
                    ),
                    "diverged": _counter_value(entry, "diverged"),
                    "decision": entry["decision"],
                }
            )


def _write_detailed_groups(
    path: Path,
    entries: Iterable[dict[str, Any]],
) -> None:
    fieldnames = (
        "radius",
        "equilibrium",
        "status",
        "contact",
        "finite",
        "tests",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)


def write_outputs(validation_dir: Path, summary: dict[str, Any]) -> None:
    """Write the canonical JSON and three directly derived CSV summaries."""

    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_summary_by_radius(
        validation_dir / "summary_by_radius.csv",
        summary["summary_by_radius"],
    )
    _write_summary_by_radius_equilibrium(
        validation_dir / "summary_by_radius_equilibrium.csv",
        summary["summary_by_radius_equilibrium"],
    )
    _write_detailed_groups(
        validation_dir / "summary_by_radius_equilibrium_status_contact.csv",
        summary["summary_by_radius_equilibrium_status_contact"],
    )


def build_public_validation_summary(
    source_summary: dict[str, Any],
    public_base: dict[str, Any],
) -> dict[str, Any]:
    """Project the source package into the canonical public validation package."""

    payload = dict(source_summary)
    source_keys = set(source_summary)
    for key, value in public_base.items():
        if key not in source_keys or key in {
            "case_id",
            "claim_boundary",
            "claim_label",
            "claim_scope",
        }:
            payload[key] = value
    payload["schema_version"] = "1.1"
    payload["source_case_id"] = source_summary["case_id"]
    payload["derived_from_validation_package"] = (
        "version_2/validation/chua_fractional_arctan_c590/"
    )
    return payload


def _write_public_decisions(
    path: Path,
    entries: Iterable[dict[str, Any]],
    *,
    local_max_radius: float,
) -> None:
    fieldnames = (
        "radius",
        "equilibrium",
        "tests",
        "contacts",
        "finite",
        "full_horizon",
        "ok",
        "converged_equilibrium_early",
        "diverged",
        "decision",
        "claim_role",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "radius": entry["radius"],
                    "equilibrium": entry["equilibrium"],
                    "tests": entry["tests"],
                    "contacts": entry["contacts"],
                    "finite": entry["finite"],
                    "full_horizon": entry["full_horizon"],
                    "ok": _counter_value(entry, "ok"),
                    "converged_equilibrium_early": _counter_value(
                        entry, "converged_equilibrium_early"
                    ),
                    "diverged": _counter_value(entry, "diverged"),
                    "decision": entry["decision"],
                    "claim_role": (
                        "local_claim"
                        if entry["radius"] <= local_max_radius
                        else "macro_audit"
                    ),
                }
            )


def _write_public_detailed_groups(
    path: Path,
    entries: Iterable[dict[str, Any]],
    *,
    local_max_radius: float,
) -> None:
    fieldnames = (
        "radius",
        "equilibrium",
        "status",
        "contact",
        "finite",
        "tests",
        "claim_role",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    **entry,
                    "claim_role": (
                        "local_claim"
                        if entry["radius"] <= local_max_radius
                        else "macro_audit"
                    ),
                }
            )


def write_public_outputs(
    validation_dir: Path,
    summary: dict[str, Any],
    *,
    local_max_radius: float,
) -> None:
    """Write the public package directly derived from the c590 source package."""

    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "hiddenness_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_summary_by_radius(
        validation_dir / "summary_by_radius.csv",
        summary["summary_by_radius"],
    )
    _write_public_decisions(
        validation_dir / "hiddenness_decisions.csv",
        summary["summary_by_radius_equilibrium"],
        local_max_radius=local_max_radius,
    )
    _write_public_detailed_groups(
        validation_dir / "hiddenness_decisions_status_contact.csv",
        summary["summary_by_radius_equilibrium_status_contact"],
        local_max_radius=local_max_radius,
    )


def summarize(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    validation_dir: Path = DEFAULT_VALIDATION_DIR,
    local_max_radius: float = 0.3,
) -> dict[str, Any]:
    """Build the validation summary without executing any simulations."""

    source_paths = [source_dir / name for name in CANONICAL_ROW_FILES]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing canonical c590 row files: " + ", ".join(missing)
        )
    base_path = validation_dir / "validation_summary.json"
    base = (
        json.loads(base_path.read_text(encoding="utf-8"))
        if base_path.is_file()
        else {}
    )
    rows = read_probe_rows(source_paths)
    aggregate = aggregate_probe_rows(rows, local_max_radius=local_max_radius)
    source_file_manifest = build_source_file_manifest(source_paths, rows)
    return build_validation_summary(
        base,
        aggregate,
        source_paths=source_paths,
        source_file_manifest=source_file_manifest,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing the six canonical c590 probe CSV files.",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
        help="Directory where the derived JSON and CSV summaries are written.",
    )
    parser.add_argument(
        "--local-max-radius",
        type=float,
        default=0.3,
        help="Largest radius included in the local-evidence partition.",
    )
    parser.add_argument(
        "--public-validation-dir",
        type=Path,
        default=DEFAULT_PUBLIC_VALIDATION_DIR,
        help="Directory containing the public validation projection.",
    )
    parser.add_argument(
        "--skip-public-package",
        action="store_true",
        help="Write only the source c590 validation package.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    summary = summarize(
        source_dir=args.source_dir,
        validation_dir=args.validation_dir,
        local_max_radius=args.local_max_radius,
    )
    write_outputs(args.validation_dir, summary)
    public_path: str | None = None
    if not args.skip_public_package:
        public_summary_path = (
            args.public_validation_dir / "hiddenness_validation_summary.json"
        )
        public_base = (
            json.loads(public_summary_path.read_text(encoding="utf-8"))
            if public_summary_path.is_file()
            else {}
        )
        public_summary = build_public_validation_summary(summary, public_base)
        write_public_outputs(
            args.public_validation_dir,
            public_summary,
            local_max_radius=args.local_max_radius,
        )
        public_path = str(public_summary_path)
    print(
        json.dumps(
            {
                "validation_summary": str(
                    args.validation_dir / "validation_summary.json"
                ),
                "public_validation_summary": public_path,
                "tests": summary["total_tests"],
                "contacts": summary["total_contacts"],
                "status_counts": summary["status_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
