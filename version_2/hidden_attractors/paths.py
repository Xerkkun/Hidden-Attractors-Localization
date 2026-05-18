"""Canonical repository paths used by examples and workflows."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
OUTPUTS = PROJECT_ROOT / "outputs"
CONFIGS = PROJECT_ROOT / "configs"
NATIVE_CACHE = PROJECT_ROOT / ".runtime_native"
RUNTIME_CACHE = PROJECT_ROOT / ".runtime_cache"

