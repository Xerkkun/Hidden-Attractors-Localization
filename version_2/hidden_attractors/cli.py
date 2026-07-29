"""Thin compatibility wrapper for legacy CLI helpers.

Deprecated: use the unified `hidden-attractors` CLI instead.
"""

from __future__ import annotations

import warnings
from typing import Sequence
from .cli import inspect


def list_candidates(argv: Sequence[str] | None = None) -> None:
    """Print candidate records from an explicit portable JSON source."""
    warnings.warn(
        "Deprecated: use 'hidden-attractors inspect candidates'",
        DeprecationWarning,
        stacklevel=2
    )
    print("Deprecated: use 'hidden-attractors inspect candidates'")
    inspect.list_candidates(argv)


def systems(argv: Sequence[str] | None = None) -> None:
    """List or inspect registered chaotic systems."""
    warnings.warn(
        "Deprecated: use 'hidden-attractors inspect systems'",
        DeprecationWarning,
        stacklevel=2
    )
    print("Deprecated: use 'hidden-attractors inspect systems'")
    inspect.systems(argv)


def workflow_requirements(argv: Sequence[str] | None = None) -> None:
    """Print required inputs for reusable workflows and system readiness."""
    warnings.warn(
        "Deprecated: use 'hidden-attractors inspect workflow-requirements'",
        DeprecationWarning,
        stacklevel=2
    )
    print("Deprecated: use 'hidden-attractors inspect workflow-requirements'")
    inspect.workflow_requirements(argv)
