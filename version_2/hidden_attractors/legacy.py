"""Installable command facade for historical research scripts.

The scripts in ``tools.legacy`` are not the preferred public API, but some
research workflows still depend on them.  This module gives them the same
installed-command shape as the newer ``hidden_attractors`` workflows while the
reusable parts are migrated into package modules.
"""

from __future__ import annotations

import argparse
import importlib.resources as resources
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class LegacyScript:
    """Metadata for one historical script exposed through the facade."""

    slug: str
    filename: str
    summary: str


LEGACY_SCRIPTS: tuple[LegacyScript, ...] = (
    LegacyScript("audit-q", "audit_and_homogenize_q.py", "Audit q values in historical outputs."),
    LegacyScript("biased-describing-function", "biased_describing_function.py", "Historical biased describing-function search."),
    LegacyScript("basin-compare", "chua_basin_comparison_h001.py", "Compare Danca/project basin classifications."),
    LegacyScript("initial-conditions", "chua_initial_cond.py", "Historical Chua initial-condition and DF utilities."),
    LegacyScript("corrida1-refined", "corrida1_refined_verification.py", "Corrida 1 refined candidate verification."),
    LegacyScript("danca2017", "danca2017_chua_abm_replication.py", "Danca 2017 ABM replication workflow."),
    LegacyScript("debug-branch1", "debug_branch1_failures.py", "Diagnostics for branch-1 failures."),
    LegacyScript("equilibria-analysis", "equilibria_analysis.py", "Historical equilibrium analysis helpers."),
    LegacyScript("formal-nyquist", "formal_nyquist_df_chua.py", "Formal Nyquist/describing-function diagnostics."),
    LegacyScript("harmonic-diagnostics", "harmonic_diagnostics.py", "Harmonic-balance diagnostic outputs."),
    LegacyScript("launch-danca2017", "launch_danca2017_jobs.py", "macOS launchd helper for Danca jobs."),
    LegacyScript("adaptive-contact", "lure_adaptive_contact_test.py", "Adaptive Lure equilibrium-contact tests."),
    LegacyScript("biased-continuation", "lure_biased_multiparam_continuation.py", "Biased Lure multiparameter continuation."),
    LegacyScript("biased-search", "lure_biased_multiparam_search.py", "Biased Lure multiparameter search."),
    LegacyScript("candidate-manifest", "lure_candidate_manifest.py", "Build or inspect Lure candidate manifests."),
    LegacyScript("refined-route", "lure_refined_route.py", "Refined Lure route analysis."),
    LegacyScript("rhoh-diagnostics", "lure_rhoH_diagnostics.py", "rho_H diagnostics for Lure candidates."),
    LegacyScript("robustness-controls", "lure_robustness_and_control_tests.py", "Lure robustness and directed controls."),
    LegacyScript("machado-targeted", "machado_targeted_verification.py", "Machado targeted verification workflow."),
    LegacyScript("multiparameter-continuation", "multiparameter_continuation.py", "Historical multiparameter continuation helper."),
    LegacyScript("plot-lure-3d", "plot_lure_hiddenness_3d.py", "3D Lure hiddenness plotting helper."),
    LegacyScript("plot-sphere-geometry", "plot_top3_sphere_geometry.py", "Top-candidate sphere geometry plots."),
    LegacyScript("positive-x-basin", "positive_x_basin_sweep.py", "Positive-x basin sweep workflow."),
    LegacyScript("extended-search", "run_extended_search.py", "Extended non-smooth fractional Chua search."),
    LegacyScript("hidden-verify", "run_hidden_verify_frac_hybrid.py", "Hybrid hiddenness verification workflow."),
    LegacyScript("seed-cloud", "seed_cloud_search.py", "Seed-cloud search helper."),
    LegacyScript("integer-chua", "unified_chua_integer_pipeline.py", "Unified integer Chua pipeline."),
    LegacyScript("nyquist-pipeline", "unified_nyquist_hidden_pipeline.py", "Unified fractional Chua Nyquist pipeline."),
    LegacyScript("validate-chua", "validate_chua_piecewise_case.py", "Validate piecewise Chua backend command shape."),
)

_BY_SLUG = {item.slug: item for item in LEGACY_SCRIPTS}


def legacy_script_names() -> list[str]:
    """Return all installable legacy script slugs."""

    return sorted(_BY_SLUG)


def legacy_script_path(slug: str) -> Path:
    """Return the installed filesystem path for a legacy script slug."""

    try:
        item = _BY_SLUG[slug]
    except KeyError as exc:
        known = ", ".join(legacy_script_names())
        raise ValueError(f"Unknown legacy script '{slug}'. Known scripts: {known}") from exc
    return Path(resources.files("tools.legacy") / item.filename)


def run_legacy_script(slug: str, args: Sequence[str] | None = None, *, check: bool = False) -> subprocess.CompletedProcess[str]:
    """Run one historical script with the current Python interpreter."""

    script = legacy_script_path(slug)
    cmd = [sys.executable, str(script), *(args or ())]
    env = None
    return subprocess.run(cmd, check=check, text=True, env=env)


def _print_list() -> None:
    width = max(len(item.slug) for item in LEGACY_SCRIPTS)
    for item in LEGACY_SCRIPTS:
        print(f"{item.slug:<{width}}  {item.summary}")


def main(argv: Sequence[str] | None = None) -> None:
    """Console entry point for ``hidden-attractors-legacy``."""

    parser = argparse.ArgumentParser(description="Run packaged historical Hidden Attractors scripts.")
    parser.add_argument("script", nargs="?", help="Legacy script slug to run.")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the legacy script.")
    parser.add_argument("--list", action="store_true", help="List available legacy script slugs.")
    args = parser.parse_args(argv)
    if args.list or not args.script:
        _print_list()
        return
    proc = run_legacy_script(args.script, args.script_args)
    raise SystemExit(proc.returncode)


def _entry(slug: str, argv: Sequence[str] | None = None) -> None:
    proc = run_legacy_script(slug, sys.argv[1:] if argv is None else argv)
    raise SystemExit(proc.returncode)


def extended_search(argv: Sequence[str] | None = None) -> None:
    _entry("extended-search", argv)


def danca2017(argv: Sequence[str] | None = None) -> None:
    _entry("danca2017", argv)


def nyquist_pipeline(argv: Sequence[str] | None = None) -> None:
    _entry("nyquist-pipeline", argv)


def integer_chua(argv: Sequence[str] | None = None) -> None:
    _entry("integer-chua", argv)


def hidden_verify(argv: Sequence[str] | None = None) -> None:
    _entry("hidden-verify", argv)


def basin_compare(argv: Sequence[str] | None = None) -> None:
    _entry("basin-compare", argv)


def positive_x_basin(argv: Sequence[str] | None = None) -> None:
    _entry("positive-x-basin", argv)


def machado_targeted(argv: Sequence[str] | None = None) -> None:
    _entry("machado-targeted", argv)
