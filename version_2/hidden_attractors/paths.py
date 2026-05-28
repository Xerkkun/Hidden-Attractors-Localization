"""Canonical repository paths used by examples and workflows.

Stability: internal
    Path constants consumed by loaders and workflows.  If the repository
    layout changes, these constants change with it.
"""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
OUTPUTS = PROJECT_ROOT / "outputs"
CONFIGS = PROJECT_ROOT / "configs"
NATIVE_CACHE = PROJECT_ROOT / ".runtime_native"
RUNTIME_CACHE = PROJECT_ROOT / ".runtime_cache"


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


