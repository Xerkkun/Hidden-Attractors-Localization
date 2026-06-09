"""CLI commands for hiddenness verification workflows.

Stability: internal
"""

from __future__ import annotations

from typing import Sequence
from ..workflows.sphere_controls import main as sphere_controls_main
from ..workflows.strict_target_refinement import main as strict_target_refinement_main


def sphere_controls(argv: Sequence[str] | None = None) -> None:
    """Run the sphere controls validation workflow."""
    sphere_controls_main(list(argv) if argv is not None else None)


def strict_target_refinement(argv: Sequence[str] | None = None) -> None:
    """Run the strict target refinement workflow."""
    strict_target_refinement_main(list(argv) if argv is not None else None)
