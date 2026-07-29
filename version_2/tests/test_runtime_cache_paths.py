from __future__ import annotations

from pathlib import Path

import pytest

from hidden_attractors import paths


def test_default_cache_falls_back_when_platform_directory_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")
    monkeypatch.delenv("HIDDEN_ATTRACTORS_CACHE_DIR", raising=False)
    monkeypatch.setattr(paths, "NATIVE_CACHE", blocked_parent / "native")
    monkeypatch.setattr(paths.tempfile, "gettempdir", lambda: str(tmp_path / "temp"))

    cache = paths.get_native_cache()

    assert cache == (tmp_path / "temp" / "hidden-attractors-fo" / "native").resolve()
    assert cache.is_dir()


def test_explicit_unwritable_cache_is_not_silently_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("HIDDEN_ATTRACTORS_CACHE_DIR", str(blocked_parent))
    monkeypatch.setattr(paths, "NATIVE_CACHE", blocked_parent / "native")

    with pytest.raises(OSError, match="Configured cache directory is not writable"):
        paths.get_native_cache()
