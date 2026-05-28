"""Legacy adapter for the old /src CLI.

This module redirects calls to the new, package-maintained CLI in version_2.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    print("\n" + "!" * 80)
    print(" WARNING: The legacy /src CLI is deprecated and will be removed in a future version.")
    print(" Please use the new 'hidden-attractors' CLI from the formal package instead.")
    print(" Redirecting execution to hidden_attractors.cli.run...")
    print("!" * 80 + "\n")

    # Add version_2 path to sys.path if not present
    workspace_root = Path(__file__).resolve().parents[2]
    version_2_path = str(workspace_root / "version_2")
    if version_2_path not in sys.path:
        sys.path.insert(0, version_2_path)

    # Import the new CLI main
    try:
        from hidden_attractors.cli.run import main as new_main
    except ImportError as e:
        print(f"Error: Could not import hidden_attractors package. Ensure version_2 is on sys.path.")
        print(f"ImportError: {e}")
        sys.exit(1)

    # Adapt arguments: new CLI uses subcommand 'run' for simulations
    new_argv = list(argv)
    if not new_argv or new_argv[0] not in ("run", "init", "inspect-config"):
        new_argv.insert(0, "run")

    new_main(new_argv)


if __name__ == "__main__":
    main()
