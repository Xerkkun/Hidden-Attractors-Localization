#!/usr/bin/env python3
"""Compatibility CLI for C/EFORK robustness-overlay trajectories.

The reusable implementation lives in
``hidden_attractors.workflows.robustness_overlay`` so that examples and future
experiments do not duplicate JSON/CSV, C-backend, metrics, or plotting code.
"""

from hidden_attractors.workflows.robustness_overlay import main


if __name__ == "__main__":
    main()
