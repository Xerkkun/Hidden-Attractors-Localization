"""CLI commands for official Caputo hidden-attractor protocol stages.

Stability: internal
"""

from __future__ import annotations

import sys
from typing import Sequence
from ..protocol_cli import main as protocol_cli_main


def run_protocol_stage(stage_cmd: str, argv: Sequence[str] | None = None) -> None:
    """Delegate protocol stage subcommand directly to protocol_cli."""
    args = [stage_cmd]
    if argv:
        args.extend(argv)
    sys.exit(protocol_cli_main(args))
