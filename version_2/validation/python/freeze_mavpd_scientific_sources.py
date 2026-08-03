"""Create and verify an immutable MAVPD scientific-source snapshot.

The file set and bundle digest intentionally match
``examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/run_example.py``.
Only Python's standard library is used so the snapshot can be frozen before a
scientific environment is installed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
MANIFEST_NAME = "snapshot_manifest.json"
README_NAME = "README.md"
SCIENTIFIC_SOURCE_FIXED_FILES = (
    "examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/run_example.py",
    "examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/reproducibility.yaml",
    "validation/wolfram/cases/mavpd_integer.wl",
)


class SnapshotError(RuntimeError):
    """Raised when a source tree or immutable snapshot violates the contract."""


def _absolute(path: Path) -> Path:
    """Return an absolute path without following a final symlink/reparse point."""

    return Path(os.path.abspath(os.fspath(path)))


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _is_link_or_reparse(info: os.stat_result) -> bool:
    attributes = int(getattr(info, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return stat.S_ISLNK(info.st_mode) or bool(attributes & reparse_flag)


def _lstat(path: Path, *, label: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as error:
        raise SnapshotError(f"cannot inspect {label} {path}: {error}") from error


def _require_real_directory(path: Path, *, label: str) -> None:
    info = _lstat(path, label=label)
    if _is_link_or_reparse(info):
        raise SnapshotError(f"{label} must not be a symlink or reparse point: {path}")
    if not stat.S_ISDIR(info.st_mode):
        raise SnapshotError(f"{label} is not a directory: {path}")


def _require_regular_file(path: Path, *, label: str) -> os.stat_result:
    info = _lstat(path, label=label)
    if _is_link_or_reparse(info):
        raise SnapshotError(f"{label} must not be a symlink or reparse point: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise SnapshotError(f"{label} is not a regular file: {path}")
    return info


def _resolved_within(root: Path, path: Path, *, label: str) -> Path:
    try:
        root_resolved = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as error:
        raise SnapshotError(f"{label} escapes its root: {path}") from error
    return resolved


def _walk_regular_tree(root: Path) -> list[Path]:
    """Walk without following links and reject every special tree entry."""

    _require_real_directory(root, label="tree root")
    paths: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        _require_real_directory(directory, label="tree directory")
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise SnapshotError(f"cannot enumerate tree directory {directory}: {error}") from error
        for entry in entries:
            path = Path(entry.path)
            info = _lstat(path, label="tree entry")
            if _is_link_or_reparse(info):
                raise SnapshotError(f"tree entry must not be a symlink or reparse point: {path}")
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
            elif stat.S_ISREG(info.st_mode):
                paths.append(path)
            else:
                raise SnapshotError(f"tree entry is not a regular file or directory: {path}")
    return sorted(paths)


def _safe_relative(relative: str) -> PurePosixPath:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise SnapshotError(f"invalid portable relative path: {relative!r}")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or pure == PurePosixPath(".") or ".." in pure.parts:
        raise SnapshotError(f"invalid portable relative path: {relative!r}")
    return pure


def _local_path(root: Path, relative: str) -> Path:
    pure = _safe_relative(relative)
    return root.joinpath(*pure.parts)


def discover_scientific_source_files(source_root: Path) -> tuple[str, ...]:
    """Discover exactly the maintained source set used by the MAVPD runner."""

    source_root = _absolute(source_root)
    _require_real_directory(source_root, label="source root")
    hidden_root = source_root / "hidden_attractors"
    _require_real_directory(hidden_root, label="hidden_attractors source directory")

    relatives = set(SCIENTIFIC_SOURCE_FIXED_FILES)
    for path in _walk_regular_tree(hidden_root):
        _resolved_within(source_root, path, label="scientific source")
        if path.suffix == ".py":
            relatives.add(path.relative_to(source_root).as_posix())

    for relative in sorted(relatives):
        path = _local_path(source_root, relative)
        _require_regular_file(path, label="scientific source")
        _resolved_within(source_root, path, label="scientific source")
    return tuple(sorted(relatives))


def _read_regular_file(root: Path, relative: str) -> bytes:
    path = _local_path(root, relative)
    before = _require_regular_file(path, label="scientific source")
    _resolved_within(root, path, label="scientific source")

    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SnapshotError(f"cannot open scientific source {path}: {error}") from error
    try:
        opened = os.fstat(descriptor)
        if _is_link_or_reparse(opened) or not stat.S_ISREG(opened.st_mode):
            raise SnapshotError(f"scientific source is not an opened regular file: {path}")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise SnapshotError(f"scientific source changed identity while opening: {relative}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)

    after = _require_regular_file(path, label="scientific source")
    if (
        (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise SnapshotError(f"scientific source drifted while being read: {relative}")
    return b"".join(chunks)


def _bundle_digest(file_hashes: Mapping[str, str]) -> str:
    material = "".join(
        f"{relative}\0{digest}\n" for relative, digest in sorted(file_hashes.items())
    ).encode("utf-8")
    return sha256(material).hexdigest()


def capture_scientific_source_state(source_root: Path) -> dict[str, Any]:
    """Hash one dynamically discovered source state with the runner algorithm."""

    source_root = _absolute(source_root)
    files: dict[str, dict[str, Any]] = {}
    for relative in discover_scientific_source_files(source_root):
        payload = _read_regular_file(source_root, relative)
        files[relative] = {
            "size_bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
        }
    hashes = {relative: record["sha256"] for relative, record in files.items()}
    return {
        "algorithm": "sha256",
        "bundle_sha256": _bundle_digest(hashes),
        "files": files,
    }


def _state_signature(state: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    files = state.get("files")
    if not isinstance(files, Mapping):
        raise SnapshotError("source state has no file map")
    signature: dict[str, tuple[int, str]] = {}
    for relative, record in files.items():
        if not isinstance(relative, str) or not isinstance(record, Mapping):
            raise SnapshotError("source state contains an invalid file record")
        try:
            size = int(record["size_bytes"])
            digest = str(record["sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise SnapshotError(f"invalid source state record for {relative}") from error
        signature[relative] = (size, digest)
    return signature


def _copy_sources(source_root: Path, snapshot_root: Path, before: Mapping[str, Any]) -> None:
    for relative in sorted(_state_signature(before)):
        payload = _read_regular_file(source_root, relative)
        destination = _local_path(snapshot_root, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise SnapshotError(f"cannot copy scientific source {relative}: {error}") from error


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _readme(bundle: str) -> str:
    return f"""# Immutable MAVPD scientific-source snapshot

This directory contains the exact maintained source set used by the integer
modified Van der Pol--Duffing hidden-chaos runner. Its SHA-256 bundle is
`{bundle}`.

Treat this entire directory as immutable. Run the experiment from this tree,
so its local `hidden_attractors` package, example runner, YAML contract, and
Wolfram case are the only scientific sources imported or read. Do not mix the
snapshot with modules from a mutable checkout through `PYTHONPATH`.

From this directory, the full example command is:

```text
python -m examples.modified_van_der_pol_duffing_integer_hidden_chaos_search.run_example --output-dir <fresh-output-directory>
```

The module form is deliberate: with the snapshot as the working directory it
puts this frozen root ahead of any editable installation on the import path.

Before execution, verify this snapshot with the external freezing tool:

```text
python <mutable-checkout>/validation/python/freeze_mavpd_scientific_sources.py --snapshot-root <this-directory> --verify-only
```

`snapshot_manifest.json` records every scientific file's relative path, byte
size, SHA-256 digest, and the bundle digest. Verification rejects edits,
missing or extra files, and symlink/reparse substitutions. The README and
manifest are metadata and are not part of the runner's scientific bundle.
"""


def _manifest_payload(source_root: Path, snapshot_root: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "source_root": str(source_root.resolve(strict=True)),
        "snapshot_root": str(snapshot_root),
        "algorithm": "sha256",
        "bundle_sha256": state["bundle_sha256"],
        "file_count": len(state["files"]),
        "files": state["files"],
    }


def _load_manifest(snapshot_root: Path) -> dict[str, Any]:
    path = snapshot_root / MANIFEST_NAME
    _require_regular_file(path, label="snapshot manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SnapshotError(f"cannot read snapshot manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SnapshotError("snapshot manifest root must be an object")
    return payload


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SnapshotError(f"unsupported snapshot schema version: {manifest.get('schema_version')!r}")
    if manifest.get("algorithm") != "sha256":
        raise SnapshotError("snapshot manifest algorithm must be sha256")
    for field in ("created_at_utc", "source_root", "snapshot_root"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise SnapshotError(f"snapshot manifest field {field!r} must be a nonempty string")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SnapshotError("snapshot manifest files must be a nonempty object")
    if manifest.get("file_count") != len(files):
        raise SnapshotError("snapshot manifest file_count does not match files")

    normalized: dict[str, dict[str, Any]] = {}
    for relative, record in files.items():
        _safe_relative(relative)
        if not isinstance(record, dict):
            raise SnapshotError(f"manifest record is not an object: {relative}")
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise SnapshotError(f"invalid manifest byte size: {relative}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise SnapshotError(f"invalid manifest SHA-256 digest: {relative}")
        try:
            int(digest, 16)
        except ValueError as error:
            raise SnapshotError(f"invalid manifest SHA-256 digest: {relative}") from error
        normalized[relative] = {"size_bytes": size, "sha256": digest.lower()}
    bundle = manifest.get("bundle_sha256")
    if not isinstance(bundle, str) or len(bundle) != 64:
        raise SnapshotError("snapshot manifest has an invalid bundle digest")
    return normalized


def verify_snapshot(snapshot_root: Path, source_root: Path | None = None) -> dict[str, Any]:
    """Verify an existing immutable snapshot, optionally against a live source root."""

    snapshot_root = _absolute(snapshot_root)
    _require_real_directory(snapshot_root, label="snapshot root")
    all_files = _walk_regular_tree(snapshot_root)
    manifest = _load_manifest(snapshot_root)
    manifest_files = _validate_manifest_shape(manifest)
    captured = capture_scientific_source_state(snapshot_root)
    captured_signature = _state_signature(captured)
    manifest_signature = {
        relative: (record["size_bytes"], record["sha256"])
        for relative, record in manifest_files.items()
    }
    if captured_signature != manifest_signature:
        raise SnapshotError("snapshot files do not match the manifest sizes and hashes")
    if captured["bundle_sha256"] != manifest.get("bundle_sha256"):
        raise SnapshotError("snapshot bundle digest does not match the manifest")

    allowed = set(manifest_files) | {README_NAME, MANIFEST_NAME}
    actual = {path.relative_to(snapshot_root).as_posix() for path in all_files}
    if actual != allowed:
        missing = sorted(allowed - actual)
        extra = sorted(actual - allowed)
        raise SnapshotError(f"snapshot tree is not exact; missing={missing}, extra={extra}")
    _require_regular_file(snapshot_root / README_NAME, label="snapshot README")

    source_matches: bool | None = None
    if source_root is not None:
        current = capture_scientific_source_state(_absolute(source_root))
        source_matches = _state_signature(current) == captured_signature
        if not source_matches:
            raise SnapshotError("current source root does not match the immutable snapshot")

    return {
        "status": "verified",
        "snapshot_root": str(snapshot_root),
        "file_count": len(captured_signature),
        "bundle_sha256": captured["bundle_sha256"],
        "matches_current_source": source_matches,
    }


def freeze_snapshot(source_root: Path, snapshot_root: Path) -> dict[str, Any]:
    """Freeze a fresh tree and publish it only after all three hash maps agree."""

    source_root = _absolute(source_root)
    snapshot_root = _absolute(snapshot_root)
    _require_real_directory(source_root, label="source root")
    if _lexists(snapshot_root):
        raise SnapshotError(f"snapshot target already exists: {snapshot_root}")
    if snapshot_root == source_root:
        raise SnapshotError("snapshot target must differ from the source root")

    hidden_root = (source_root / "hidden_attractors").resolve(strict=True)
    try:
        snapshot_root.resolve(strict=False).relative_to(hidden_root)
    except ValueError:
        pass
    else:
        raise SnapshotError("snapshot target must not be inside hidden_attractors")

    snapshot_root.parent.mkdir(parents=True, exist_ok=True)
    _require_real_directory(snapshot_root.parent, label="snapshot parent")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{snapshot_root.name}.tmp-", dir=snapshot_root.parent)
    )
    published = False
    try:
        before = capture_scientific_source_state(source_root)
        _copy_sources(source_root, staging, before)
        copied = capture_scientific_source_state(staging)
        after = capture_scientific_source_state(source_root)
        signatures = (
            _state_signature(before),
            _state_signature(copied),
            _state_signature(after),
        )
        if not (signatures[0] == signatures[1] == signatures[2]):
            raise SnapshotError(
                "scientific source drift detected: before-copy, copied, and after-copy maps differ"
            )
        bundles = (before["bundle_sha256"], copied["bundle_sha256"], after["bundle_sha256"])
        if len(set(bundles)) != 1:
            raise SnapshotError("scientific source bundle changed while freezing")

        _atomic_write(staging / README_NAME, _readme(before["bundle_sha256"]).encode("utf-8"))
        manifest = _manifest_payload(source_root, snapshot_root, before)
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _atomic_write(staging / MANIFEST_NAME, manifest_bytes)
        verify_snapshot(staging)

        if _lexists(snapshot_root):
            raise SnapshotError(f"snapshot target appeared during publication: {snapshot_root}")
        os.replace(staging, snapshot_root)
        published = True
        return verify_snapshot(snapshot_root, source_root)
    finally:
        if not published and _lexists(staging):
            shutil.rmtree(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="repository root; required when creating and optional for strict verification",
    )
    parser.add_argument("--snapshot-root", type=Path, required=True, help="snapshot directory")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing snapshot without creating or changing files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.verify_only and args.source_root is None:
        parser.error("--source-root is required unless --verify-only is used")
    try:
        if args.verify_only:
            result = verify_snapshot(args.snapshot_root, args.source_root)
        else:
            result = freeze_snapshot(args.source_root, args.snapshot_root)
    except (OSError, SnapshotError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
