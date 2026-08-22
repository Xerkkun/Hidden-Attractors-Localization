"""Migrate the canonical MAVPD promotion pointers to portable POSIX paths.

This is a one-purpose, fail-closed metadata migration.  It permits exactly 36
path leaves: 18 promotion pointers and their 18 copies in ``run_manifest``.
It rebinds their three derived SHA-256 entries in ``run_status.json`` and does
not alter numerical or scientific values.  Check mode is the default; pass
``--write`` explicitly to apply the validated transaction.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


VERSION_ROOT = Path(__file__).resolve().parents[2]
CASE_RELATIVE = Path("validation/reference_cases/mavpd_integer_hidden_chaos")
FIGURE_MANIFEST_RELATIVE = CASE_RELATIVE / "figures/figure_manifest.json"
PROMOTION_RECEIPT_RELATIVE = CASE_RELATIVE / "figures/global_promotion_receipt.json"
RUN_MANIFEST_RELATIVE = CASE_RELATIVE / "run_manifest.json"
RUN_STATUS_RELATIVE = CASE_RELATIVE / "run_status.json"

RUN_ID = (
    "mavpd_integer_hidden_chaos-full-20260803T110818509512Z-"
    "f1d10f792a3a-1e2527c6895d"
)
FIGURE_IDS = (
    "00_nyquist_direct_seed",
    "03_continuation_screen",
    "04_candidate_phase_portraits",
    "04_candidate_time_series",
    "05_lyapunov_convergence",
    "05_poincare_section",
    "05_normalized_fft_power",
    "07_hiddenness_outcomes",
)

JsonPath = tuple[str | int, ...]


def _figure_portable_path(figure_id: str, suffix: str) -> str:
    return (
        f"library_figures/by_run/{RUN_ID}/{suffix}/"
        f"{figure_id}.{suffix}"
    )


FIGURE_FIELD_TARGETS: tuple[tuple[int, str, str], ...] = tuple(
    (index, suffix, _figure_portable_path(figure_id, suffix))
    for index, figure_id in enumerate(FIGURE_IDS)
    for suffix in ("pdf", "png")
)

PATH_TARGETS: tuple[tuple[Path, JsonPath, str], ...] = tuple(
    (
        FIGURE_MANIFEST_RELATIVE,
        ("figures", index, "central_paths", suffix),
        expected,
    )
    for index, suffix, expected in FIGURE_FIELD_TARGETS
) + (
    (
        PROMOTION_RECEIPT_RELATIVE,
        ("global_manifest_paths", 0),
        "library_figures/manifests/figure_manifest.json",
    ),
    (
        PROMOTION_RECEIPT_RELATIVE,
        ("global_manifest_paths", 1),
        "library_figures/manifests/figure_manifest.csv",
    ),
) + tuple(
    (
        RUN_MANIFEST_RELATIVE,
        ("figures", index, "central_paths", suffix),
        expected,
    )
    for index, suffix, expected in FIGURE_FIELD_TARGETS
) + (
    (
        RUN_MANIFEST_RELATIVE,
        ("global_figure_promotion", "global_manifest_paths", 0),
        "library_figures/manifests/figure_manifest.json",
    ),
    (
        RUN_MANIFEST_RELATIVE,
        ("global_figure_promotion", "global_manifest_paths", 1),
        "library_figures/manifests/figure_manifest.csv",
    ),
)

LEDGER_TARGETS = {
    FIGURE_MANIFEST_RELATIVE: "figures/figure_manifest.json",
    PROMOTION_RECEIPT_RELATIVE: "figures/global_promotion_receipt.json",
    RUN_MANIFEST_RELATIVE: "run_manifest.json",
}


class PathMigrationError(RuntimeError):
    """Raised before writing when the canonical migration contract is not met."""


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PathMigrationError(f"cannot read canonical JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise PathMigrationError(f"canonical JSON must be an object: {path.name}")
    return payload


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _get(value: Any, path: JsonPath) -> Any:
    cursor = value
    for part in path:
        if isinstance(part, int):
            if not isinstance(cursor, list) or part < 0 or part >= len(cursor):
                raise PathMigrationError(f"missing allowlisted JSON path: {_render_path(path)}")
            cursor = cursor[part]
        else:
            if not isinstance(cursor, Mapping) or part not in cursor:
                raise PathMigrationError(f"missing allowlisted JSON path: {_render_path(path)}")
            cursor = cursor[part]
    return cursor


def _set(value: Any, path: JsonPath, replacement: Any) -> None:
    cursor = value
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = replacement


def _render_path(path: JsonPath) -> str:
    rendered = ""
    for part in path:
        rendered += f"[{part}]" if isinstance(part, int) else ("." if rendered else "") + part
    return rendered


def _changed_leaf_paths(before: Any, after: Any, prefix: JsonPath = ()) -> set[JsonPath]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        if set(before) != set(after):
            return {prefix}
        changed: set[JsonPath] = set()
        for key in before:
            changed.update(_changed_leaf_paths(before[key], after[key], (*prefix, str(key))))
        return changed
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return {prefix}
        changed: set[JsonPath] = set()
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            changed.update(_changed_leaf_paths(left, right, (*prefix, index)))
        return changed
    return {prefix} if type(before) is not type(after) or before != after else set()


def _portable_value(raw: Any, expected: str, *, version_root: Path) -> str:
    if raw == expected:
        return expected
    transitional = f"outputs/{expected}"
    if raw == transitional:
        return expected
    if not isinstance(raw, str) or not raw:
        raise PathMigrationError("allowlisted path is not a non-empty string")
    source = Path(raw)
    if not source.is_absolute():
        raise PathMigrationError("allowlisted path is neither canonical relative text nor absolute")
    root = version_root.resolve()
    resolved = source.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise PathMigrationError("absolute promotion path resolves outside the repository") from exc
    if relative.as_posix() != transitional:
        raise PathMigrationError("absolute promotion path does not match its exact allowlisted target")
    return expected


def _assert_canonical_shape(
    figure_manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> None:
    rows = _get(figure_manifest, ("figures",))
    run_rows = _get(run_manifest, ("figures",))
    if not isinstance(rows, list) or not isinstance(run_rows, list):
        raise PathMigrationError("figure manifests must contain figure lists")
    if tuple(row.get("figure_id") for row in rows if isinstance(row, Mapping)) != FIGURE_IDS:
        raise PathMigrationError("canonical figure manifest IDs or order changed")
    if tuple(row.get("figure_id") for row in run_rows if isinstance(row, Mapping)) != FIGURE_IDS:
        raise PathMigrationError("run manifest figure IDs or order changed")
    if receipt.get("run_id") != RUN_ID or tuple(receipt.get("figure_ids", ())) != FIGURE_IDS:
        raise PathMigrationError("canonical promotion receipt identity changed")
    if run_manifest.get("run_id") != RUN_ID:
        raise PathMigrationError("canonical run manifest identity changed")
    run_promotion = run_manifest.get("global_figure_promotion")
    if not isinstance(run_promotion, Mapping) or run_promotion.get("run_id") != RUN_ID:
        raise PathMigrationError("run manifest promotion identity changed")


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_transaction(payloads: Mapping[Path, bytes]) -> None:
    snapshots = {path: path.read_bytes() for path in payloads}
    replaced: list[Path] = []
    try:
        for path, payload in payloads.items():
            _atomic_replace(path, payload)
            replaced.append(path)
    except Exception as exc:
        rollback_errors: list[str] = []
        for path in reversed(replaced):
            try:
                _atomic_replace(path, snapshots[path])
            except Exception as rollback_exc:  # pragma: no cover - catastrophic filesystem failure
                rollback_errors.append(f"{path.name}: {rollback_exc}")
        detail = f"; rollback errors: {'; '.join(rollback_errors)}" if rollback_errors else ""
        raise PathMigrationError(f"portable-path migration failed and was rolled back{detail}") from exc


def migrate(*, version_root: Path = VERSION_ROOT, write: bool = False) -> dict[str, Any]:
    """Validate and optionally apply the exact canonical path migration."""

    root = Path(version_root).resolve()
    files = {
        FIGURE_MANIFEST_RELATIVE: root / FIGURE_MANIFEST_RELATIVE,
        PROMOTION_RECEIPT_RELATIVE: root / PROMOTION_RECEIPT_RELATIVE,
        RUN_MANIFEST_RELATIVE: root / RUN_MANIFEST_RELATIVE,
        RUN_STATUS_RELATIVE: root / RUN_STATUS_RELATIVE,
    }
    original_bytes = {relative: path.read_bytes() for relative, path in files.items()}
    figure_manifest = _read_json_object(files[FIGURE_MANIFEST_RELATIVE])
    receipt = _read_json_object(files[PROMOTION_RECEIPT_RELATIVE])
    run_manifest = _read_json_object(files[RUN_MANIFEST_RELATIVE])
    run_status = _read_json_object(files[RUN_STATUS_RELATIVE])
    _assert_canonical_shape(figure_manifest, receipt, run_manifest)

    if run_status.get("run_id") != RUN_ID:
        raise PathMigrationError("canonical run status identity changed")
    artifacts = run_status.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PathMigrationError("run status artifact ledger is missing")
    for relative, ledger_key in LEDGER_TARGETS.items():
        if artifacts.get(ledger_key) != _digest(original_bytes[relative]):
            raise PathMigrationError(f"artifact ledger mismatch before migration: {ledger_key}")

    migrated = {
        FIGURE_MANIFEST_RELATIVE: deepcopy(figure_manifest),
        PROMOTION_RECEIPT_RELATIVE: deepcopy(receipt),
        RUN_MANIFEST_RELATIVE: deepcopy(run_manifest),
    }
    for relative, path, expected in PATH_TARGETS:
        current = _get(migrated[relative], path)
        _set(
            migrated[relative],
            path,
            _portable_value(current, expected, version_root=root),
        )

    allowed_by_file = {
        relative: {path for target_relative, path, _expected in PATH_TARGETS if target_relative == relative}
        for relative in LEDGER_TARGETS
    }
    changed_by_file: dict[Path, set[JsonPath]] = {}
    for relative, before in (
        (FIGURE_MANIFEST_RELATIVE, figure_manifest),
        (PROMOTION_RECEIPT_RELATIVE, receipt),
        (RUN_MANIFEST_RELATIVE, run_manifest),
    ):
        after = migrated[relative]
        changed = _changed_leaf_paths(before, after)
        if not changed <= allowed_by_file[relative]:
            raise PathMigrationError(f"migration changed a non-allowlisted field in {relative.as_posix()}")
        scrubbed_before = deepcopy(before)
        scrubbed_after = deepcopy(after)
        for path in allowed_by_file[relative]:
            _set(scrubbed_before, path, "<portable-path>")
            _set(scrubbed_after, path, "<portable-path>")
        if scrubbed_before != scrubbed_after:
            raise PathMigrationError(f"non-path JSON semantics changed in {relative.as_posix()}")
        changed_by_file[relative] = changed

    # Count by file because the local manifest and run manifest deliberately
    # contain matching JSON-path shapes.
    changed_count = sum(len(paths) for paths in changed_by_file.values())
    if changed_count not in {0, len(PATH_TARGETS)}:
        raise PathMigrationError("canonical bundle is partially migrated; expected all 36 paths or none")
    if changed_count == len(PATH_TARGETS) and any(
        changed_by_file[relative] != allowed_by_file[relative] for relative in LEDGER_TARGETS
    ):
        raise PathMigrationError("changed path set differs from the exact 36-field allowlist")

    migrated_bytes = {
        relative: _json_bytes(payload) for relative, payload in migrated.items()
    }
    migrated_status = deepcopy(run_status)
    migrated_artifacts = migrated_status["artifacts"]
    for relative, ledger_key in LEDGER_TARGETS.items():
        migrated_artifacts[ledger_key] = _digest(migrated_bytes[relative])
    status_changes = _changed_leaf_paths(run_status, migrated_status)
    allowed_status_changes = {
        ("artifacts", ledger_key) for ledger_key in LEDGER_TARGETS.values()
    }
    expected_status_changes = allowed_status_changes if changed_count else set()
    if status_changes != expected_status_changes:
        raise PathMigrationError("run status changes differ from the three derived hash bindings")
    migrated_status_bytes = _json_bytes(migrated_status)

    writes = {
        files[FIGURE_MANIFEST_RELATIVE]: migrated_bytes[FIGURE_MANIFEST_RELATIVE],
        files[PROMOTION_RECEIPT_RELATIVE]: migrated_bytes[PROMOTION_RECEIPT_RELATIVE],
        files[RUN_MANIFEST_RELATIVE]: migrated_bytes[RUN_MANIFEST_RELATIVE],
        files[RUN_STATUS_RELATIVE]: migrated_status_bytes,
    }
    if write and changed_count:
        _write_transaction(writes)
        for path, expected_bytes in writes.items():
            if path.read_bytes() != expected_bytes:
                raise PathMigrationError(f"post-write verification failed: {path.name}")

    return {
        "mode": "write" if write else "check",
        "changed_path_fields": changed_count,
        "path_allowlist_size": len(PATH_TARGETS),
        "derived_ledger_fields": len(allowed_status_changes),
        "files": {
            relative.as_posix(): {
                "sha256_before": _digest(original_bytes[relative]),
                "sha256_after": _digest(
                    migrated_status_bytes
                    if relative == RUN_STATUS_RELATIVE
                    else migrated_bytes.get(relative, original_bytes[relative])
                ),
            }
            for relative in (
                FIGURE_MANIFEST_RELATIVE,
                PROMOTION_RECEIPT_RELATIVE,
                RUN_MANIFEST_RELATIVE,
                RUN_STATUS_RELATIVE,
            )
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="apply the exact validated migration; default is read-only check mode",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = migrate(write=bool(args.write))
    except (OSError, PathMigrationError) as exc:
        print(f"portable-path migration refused: {exc}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
