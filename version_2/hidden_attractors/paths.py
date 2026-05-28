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


def get_packaged_examples_path() -> Path:
    """Return the absolute path of the packaged examples directory using importlib.resources."""
    import importlib.resources
    try:
        # Modern Python (3.9+)
        ref = importlib.resources.files("hidden_attractors") / "configs" / "examples"
        return Path(str(ref))
    except Exception:
        return PACKAGE_ROOT / "configs" / "examples"


