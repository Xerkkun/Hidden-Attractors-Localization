"""Focused tests for the pure-stdlib MAVPD source-freezing tool."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "python" / "freeze_mavpd_scientific_sources.py"


def _module():
    spec = importlib.util.spec_from_file_location("freeze_mavpd_scientific_sources", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    _write(root / "hidden_attractors" / "__init__.py", "VALUE = 1\n")
    _write(root / "hidden_attractors" / "nested" / "model.py", "MODEL = 'mavpd'\n")
    _write(root / "hidden_attractors" / "notes.txt", "not in the bundle\n")
    _write(
        root
        / "examples"
        / "modified_van_der_pol_duffing_integer_hidden_chaos_search"
        / "run_example.py",
        "print('runner')\n",
    )
    _write(
        root
        / "examples"
        / "modified_van_der_pol_duffing_integer_hidden_chaos_search"
        / "reproducibility.yaml",
        "mode: full\n",
    )
    _write(root / "validation" / "wolfram" / "cases" / "mavpd_integer.wl", "xi = 57/20;\n")
    _write(root / "unrelated.py", "NOT_SCIENTIFIC = True\n")
    return root


def _expected_hashes(module, root: Path) -> dict[str, str]:
    return {
        relative: sha256(root.joinpath(*relative.split("/")).read_bytes()).hexdigest()
        for relative in module.discover_scientific_source_files(root)
    }


def test_freeze_copies_exact_runner_set_and_writes_verifiable_manifest(
    tmp_path: Path, source_root: Path
) -> None:
    module = _module()
    target = tmp_path / "snapshot"
    result = module.freeze_snapshot(source_root, target)

    manifest = json.loads((target / module.MANIFEST_NAME).read_text(encoding="utf-8"))
    expected = _expected_hashes(module, source_root)
    recorded = {relative: record["sha256"] for relative, record in manifest["files"].items()}
    assert recorded == expected
    assert manifest["file_count"] == len(expected) == 5
    material = "".join(f"{relative}\0{digest}\n" for relative, digest in sorted(expected.items()))
    assert manifest["bundle_sha256"] == sha256(material.encode("utf-8")).hexdigest()
    assert result["bundle_sha256"] == manifest["bundle_sha256"]
    assert result["matches_current_source"] is True
    assert not (target / "hidden_attractors" / "notes.txt").exists()
    assert not (target / "unrelated.py").exists()
    readme = (target / module.README_NAME).read_text(encoding="utf-8")
    assert "Treat this entire directory as immutable" in readme
    assert (
        "python -m examples.modified_van_der_pol_duffing_integer_hidden_chaos_search.run_example "
        "--output-dir <fresh-output-directory>"
    ) in readme
    assert "--mode full" not in readme
    assert " --output <" not in readme
    independent = module.verify_snapshot(target)
    assert independent["status"] == "verified"
    assert independent["matches_current_source"] is None


@pytest.mark.parametrize("nonempty", [False, True])
def test_freeze_rejects_every_existing_target(
    tmp_path: Path, source_root: Path, nonempty: bool
) -> None:
    module = _module()
    target = tmp_path / "snapshot"
    target.mkdir()
    if nonempty:
        _write(target / "keep.txt", "do not replace\n")

    with pytest.raises(module.SnapshotError, match="already exists"):
        module.freeze_snapshot(source_root, target)
    if nonempty:
        assert (target / "keep.txt").read_text(encoding="utf-8") == "do not replace\n"


def test_freeze_rejects_a_fixed_source_that_is_not_a_file(
    tmp_path: Path, source_root: Path
) -> None:
    module = _module()
    fixed = (
        source_root
        / "examples"
        / "modified_van_der_pol_duffing_integer_hidden_chaos_search"
        / "run_example.py"
    )
    fixed.unlink()
    fixed.mkdir()

    with pytest.raises(module.SnapshotError, match="not a regular file"):
        module.freeze_snapshot(source_root, tmp_path / "snapshot")


def test_freeze_detects_source_drift_and_does_not_publish(
    tmp_path: Path, source_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    target = tmp_path / "snapshot"
    original = module._copy_sources

    def copy_then_drift(root, staging, before):
        original(root, staging, before)
        path = root / "hidden_attractors" / "nested" / "model.py"
        path.write_text("MODEL = 'changed-during-copy'\n", encoding="utf-8")

    monkeypatch.setattr(module, "_copy_sources", copy_then_drift)
    with pytest.raises(module.SnapshotError, match="drift detected"):
        module.freeze_snapshot(source_root, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".snapshot.tmp-*"))


def test_verify_rejects_tampering_and_current_source_mismatch(
    tmp_path: Path, source_root: Path
) -> None:
    module = _module()
    target = tmp_path / "snapshot"
    module.freeze_snapshot(source_root, target)
    frozen = target / "hidden_attractors" / "nested" / "model.py"
    frozen.write_text("MODEL = 'tampered'\n", encoding="utf-8")

    with pytest.raises(module.SnapshotError, match="do not match the manifest"):
        module.verify_snapshot(target)

    frozen.write_text("MODEL = 'mavpd'\n", encoding="utf-8")
    live = source_root / "hidden_attractors" / "nested" / "model.py"
    live.write_text("MODEL = 'new-checkout-version'\n", encoding="utf-8")
    with pytest.raises(module.SnapshotError, match="current source root does not match"):
        module.verify_snapshot(target, source_root)


def test_verify_rejects_extra_files(tmp_path: Path, source_root: Path) -> None:
    module = _module()
    target = tmp_path / "snapshot"
    module.freeze_snapshot(source_root, target)
    _write(target / "unexpected.txt", "contamination\n")

    with pytest.raises(module.SnapshotError, match="snapshot tree is not exact"):
        module.verify_snapshot(target)


def test_symlink_source_and_snapshot_entries_are_rejected(
    tmp_path: Path, source_root: Path
) -> None:
    module = _module()
    outside = tmp_path / "outside.py"
    _write(outside, "OUTSIDE = True\n")
    linked = source_root / "hidden_attractors" / "linked.py"
    try:
        os.symlink(outside, linked)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks unavailable on this platform: {error}")

    with pytest.raises(module.SnapshotError, match="symlink or reparse"):
        module.freeze_snapshot(source_root, tmp_path / "snapshot")

    linked.unlink()
    target = tmp_path / "snapshot"
    module.freeze_snapshot(source_root, target)
    snapshot_link = target / "extra-link.py"
    os.symlink(outside, snapshot_link)
    with pytest.raises(module.SnapshotError, match="symlink or reparse"):
        module.verify_snapshot(target)


def test_cli_verify_only_is_read_only(tmp_path: Path, source_root: Path, capsys) -> None:
    module = _module()
    target = tmp_path / "snapshot"
    module.freeze_snapshot(source_root, target)
    before = {
        path.relative_to(target).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in target.rglob("*")
        if path.is_file()
    }

    assert module.main(["--snapshot-root", str(target), "--verify-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "verified"
    after = {
        path.relative_to(target).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in target.rglob("*")
        if path.is_file()
    }
    assert after == before
