#!/usr/bin/env python3
"""Compatibility CLI for top-candidate sphere controls and robustness.

The reusable implementation lives in
``hidden_attractors.workflows.sphere_controls``.  The CLI name and output
artifact names are preserved for reproducibility with previous runs.
"""

from hidden_attractors.workflows.sphere_controls import main


if __name__ == "__main__":
    main()
