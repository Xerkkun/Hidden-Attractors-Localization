#!/usr/bin/env python3
"""Run repository-only release-readiness checks.

This wrapper is deliberately outside the importable PyPI package. It requires
the complete tagged repository, including release metadata and workflows.
"""

from __future__ import annotations

import sys

from readiness import validate_release_readiness


if __name__ == "__main__":
    validate_release_readiness(sys.argv[1:])
