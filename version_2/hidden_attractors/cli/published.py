"""CLI commands for replicating/validating published workflows.

Stability: internal
"""

from __future__ import annotations

from typing import Sequence


def danca_abm_sphere_controls(argv: Sequence[str] | None = None) -> None:
    """Run the published Danca ABM sphere controls workflow.

    The underlying workflow requires a source checkout with ``tools/legacy``
    (Danca ABM helpers); it is not part of the PyPI public runtime.
    The import is deferred to call-time so that ``hidden-attractors --help``
    works in a clean wheel installation.
    """
    from ..workflows.danca_abm_sphere_controls import main as danca_abm_sphere_controls_main  # noqa: PLC0415
    danca_abm_sphere_controls_main(list(argv) if argv is not None else None)
