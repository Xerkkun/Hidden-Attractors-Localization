"""CLI commands for basin of attraction workflows.

Stability: internal
"""

from __future__ import annotations

from typing import Sequence
from ..workflows.refined_basin import main as refined_basin_main
from ..workflows.strict_target_refinement import main as strict_target_refinement_main


def refined(argv: Sequence[str] | None = None) -> None:
    """Run the refined basin workflow."""
    refined_basin_main(argv)


def strict_target_refinement(argv: Sequence[str] | None = None) -> None:
    """Run the strict target refinement workflow for basins."""
    strict_target_refinement_main(list(argv) if argv is not None else None)
