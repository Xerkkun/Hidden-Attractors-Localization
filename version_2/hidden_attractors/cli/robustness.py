"""CLI commands for robustness verification workflows.

Stability: internal
"""

from __future__ import annotations

from typing import Sequence
from ..workflows.robustness_overlay import main as robustness_overlay_main


def overlay(argv: Sequence[str] | None = None) -> None:
    """Run the robustness overlay workflow."""
    robustness_overlay_main(argv)
