"""CLI commands for replicating/validating published workflows.

Stability: internal
"""

from __future__ import annotations

from typing import Sequence
from ..workflows.danca_abm_sphere_controls import main as danca_abm_sphere_controls_main


def danca_abm_sphere_controls(argv: Sequence[str] | None = None) -> None:
    """Run the published Danca ABM sphere controls workflow."""
    danca_abm_sphere_controls_main(list(argv) if argv is not None else None)
