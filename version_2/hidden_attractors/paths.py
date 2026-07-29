"""Canonical repository paths used by examples and workflows.

Stability: internal
    Path constants consumed by loaders and workflows.  If the repository
    layout changes, these constants change with it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIGS = PROJECT_ROOT / "configs"


def _runtime_output_root() -> Path:
    configured = os.environ.get("HIDDEN_ATTRACTORS_OUTPUT_DIR")
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve() / "outputs"


def _runtime_cache_root() -> Path:
    configured = os.environ.get("HIDDEN_ATTRACTORS_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]).expanduser().resolve() / "hidden-attractors-fo"
    if os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]).expanduser().resolve() / "hidden-attractors-fo"
    return Path.home().resolve() / ".cache" / "hidden-attractors-fo"


OUTPUTS = _runtime_output_root()
_CACHE_ROOT = _runtime_cache_root()
NATIVE_CACHE = _CACHE_ROOT / "native"
RUNTIME_CACHE = _CACHE_ROOT / "runtime"


def _ensure_writable_cache(directory: Path) -> Path:
    """Create *directory* or use a process-temporary cache when defaults are blocked.

    An explicitly configured ``HIDDEN_ATTRACTORS_CACHE_DIR`` is never silently
    replaced: an unusable configured path raises a clear error.  Platform
    defaults may be unavailable in restricted containers, so those fall back
    to the operating-system temporary directory.
    """

    probe = directory / f".write-probe-{os.getpid()}"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"")
        probe.unlink()
        return directory
    except OSError as exc:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        if os.environ.get("HIDDEN_ATTRACTORS_CACHE_DIR"):
            raise OSError(f"Configured cache directory is not writable: {directory}") from exc

    fallback = (
        Path(tempfile.gettempdir()).expanduser().resolve()
        / "hidden-attractors-fo"
        / directory.name
    )
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def get_native_cache() -> Path:
    """Return a writable directory for compiled native backends."""

    return _ensure_writable_cache(NATIVE_CACHE)


def get_runtime_cache() -> Path:
    """Return a writable directory for transient runtime resources."""

    return _ensure_writable_cache(RUNTIME_CACHE)


def get_packaged_examples_ref():
    """Return a Traversable reference to the packaged examples configs."""
    import importlib.resources
    return importlib.resources.files("hidden_attractors").joinpath("configs", "examples")


def list_packaged_example_configs() -> list[str]:
    """List filenames of all packaged example configuration files."""
    try:
        ref = get_packaged_examples_ref()
        return [f.name for f in ref.iterdir() if f.is_file() and f.name.endswith(".yaml")]
    except Exception:
        # Fallback to local files if iterdir fails
        local_dir = PACKAGE_ROOT / "configs" / "examples"
        if local_dir.exists():
            return [f.name for f in local_dir.glob("*.yaml")]
        return []


def get_example_config_resource(filename: str):
    """Return a Traversable reference to a specific example configuration file."""
    return get_packaged_examples_ref().joinpath(filename)


def get_packaged_examples_path() -> Path:
    """Return the physical path fallback for local/editable installs when available.

    Warning: This returns a local filesystem path which might not exist in zipped installations.
    For zipped or non-editable installs, use get_packaged_examples_ref() or get_example_config_resource().
    """
    p = PACKAGE_ROOT / "configs" / "examples"
    if p.exists():
        return p
    # Fallback to current working directory templates if present
    p2 = Path.cwd() / "configs" / "examples"
    if p2.exists():
        return p2
    return p
