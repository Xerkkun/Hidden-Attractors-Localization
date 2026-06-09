"""CLI commands for fractional reports and publication figures.

Stability: internal
"""

from __future__ import annotations

from typing import Sequence
from ..workflows.fractional_report_run import main as fractional_report_run_main


def fractional_run(argv: Sequence[str] | None = None) -> None:
    """Run the fractional report run workflow."""
    fractional_report_run_main(argv)
