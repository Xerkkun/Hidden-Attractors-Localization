"""Maintain the canonical figure bundle for the unified Chua report.

This report-specific utility does not run the scientific simulations.  It
regenerates only the analytic nonlinearity comparison, reuses maintained
publication-figure builders when needed, copies already retained figures into
the canonical report bundle, and verifies every LaTeX figure reference.

Unlike the retired script, LaTeX discovery follows ``\\input``/``\\include``
recursively so report sections can remain modular.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from hidden_attractors.plotting.export import export_figure


VERSION2_ROOT = Path(__file__).resolve().parents[1]
REPORT_TEX = VERSION2_ROOT / "docs" / "reporte_unificado_chua_fraccionario.tex"
SOURCE_REPORT_DIR = VERSION2_ROOT / "library_figures" / "by_report" / "df_nc_chua"
TARGET_REPORT_DIR = (
    VERSION2_ROOT / "library_figures" / "by_report" / "unified_chua_fractional"
)
C590_CANDIDATE_DIR = (
    VERSION2_ROOT
    / "outputs"
    / "arctan_hidden_candidate_search"
    / "c590_q09999_seed9_candidate_20260623"
)

_FIGURE_PATTERN = re.compile(
    r"\\(?:reportinclude|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}"
)
_INPUT_PATTERN = re.compile(r"\\(?:input|include)\{([^}]+)\}")


def _resolve_tex_input(parent: Path, raw_value: str) -> Path:
    candidate = parent / raw_value.strip()
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".tex")
    return candidate.resolve()


def _collect_tex_documents(
    root: Path,
    *,
    _seen: set[Path] | None = None,
) -> tuple[tuple[Path, str], ...]:
    """Return the root and all existing recursively included TeX documents."""

    resolved = root.resolve()
    seen = set() if _seen is None else _seen
    if resolved in seen:
        return ()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing LaTeX source: {resolved}")
    seen.add(resolved)
    text = resolved.read_text(encoding="utf-8")
    documents: list[tuple[Path, str]] = [(resolved, text)]
    for match in _INPUT_PATTERN.finditer(text):
        raw_value = match.group(1).strip()
        if raw_value.startswith("#"):
            continue
        child = _resolve_tex_input(resolved.parent, raw_value)
        if child.exists():
            documents.extend(_collect_tex_documents(child, _seen=seen))
    return tuple(documents)


def _latex_figure_references(report_tex: Path = REPORT_TEX) -> dict[str, str]:
    """Map each referenced figure filename to its first declaring TeX source."""

    references: dict[str, str] = {}
    for source, text in _collect_tex_documents(report_tex):
        try:
            source_label = source.relative_to(VERSION2_ROOT).as_posix()
        except ValueError:
            source_label = str(source)
        for match in _FIGURE_PATTERN.finditer(text):
            value = match.group(1).strip()
            if value.startswith("#"):
                continue
            path = Path(value)
            if path.suffix.lower() not in {".pdf", ".png"}:
                continue
            references.setdefault(path.name, source_label)
    return references


def _target_path(filename: str) -> Path:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix not in {"pdf", "png"}:
        raise ValueError(f"Unsupported report figure extension: {filename}")
    return TARGET_REPORT_DIR / suffix / filename


def _source_candidates(filename: str) -> Iterable[Path]:
    suffix = Path(filename).suffix.lower().lstrip(".")
    yield TARGET_REPORT_DIR / suffix / filename
    yield SOURCE_REPORT_DIR / suffix / filename
    yield VERSION2_ROOT / "library_figures" / "current" / suffix / filename
    by_run = VERSION2_ROOT / "library_figures" / "by_run"
    if by_run.exists():
        for run_dir in sorted(by_run.iterdir(), key=lambda path: path.name):
            yield run_dir / suffix / filename


def _write_nonlinearity_comparison() -> None:
    x = np.linspace(-3.0, 3.0, 1200)
    m0, m1 = -0.1768, -1.1468
    a1, a2, rho = 0.4, -1.5585, 1.0
    nonsmooth = m1 * x + 0.5 * (m0 - m1) * (
        np.abs(x + 1.0) - np.abs(x - 1.0)
    )
    arctan = a1 * x + a2 * np.arctan(rho * x)

    fig, ax = plt.subplots(figsize=(6.9, 4.2), dpi=300)
    ax.plot(x, nonsmooth, color="#0f766e", lw=1.45, label="Chua no suave")
    ax.plot(x, arctan, color="#7c3aed", lw=1.45, label="Chua arctan Wu2023")
    ax.axvline(-1.0, color="#94a3b8", lw=0.8, ls="--")
    ax.axvline(1.0, color="#94a3b8", lw=0.8, ls="--")
    ax.axhline(0.0, color="#475569", lw=0.7)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")
    ax.grid(True, color="#e2e8f0", lw=0.6)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    pdf_path, png_path = export_figure(
        fig,
        "chua_nonlinearity_piecewise_vs_arctan",
        "model_reference",
        {
            "caption_key": "chua_nonlinearity_piecewise_vs_arctan",
            "source_script": (
                "tools/unified_report_assets.py"
            ),
            "source_function": "_write_nonlinearity_comparison",
            "data_sources": ["analytic_piecewise_chua", "analytic_arctan_chua"],
            "system_id": "chua_comparison",
            "q": "not_applicable",
            "parameters": {"m0": m0, "m1": m1, "a1": a1, "a2": a2, "rho": rho},
            "integrator": "not_applicable",
            "memory_mode": "not_applicable",
            "t_final": 0.0,
            "t_burn": 0.0,
        },
        run_id="unified_chua_fractional_report",
        export_targets=["unified_chua_fractional", "df_nc_chua"],
    )
    for report_directory in (TARGET_REPORT_DIR, SOURCE_REPORT_DIR):
        for source_path, suffix in ((pdf_path, "pdf"), (png_path, "png")):
            destination = report_directory / suffix / source_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
    plt.close(fig)


def _ensure_c590_publication_figures(references: Iterable[str]) -> None:
    c590_references = [name for name in references if "arctan_c590" in name]
    if not c590_references or all(_target_path(name).exists() for name in c590_references):
        return
    from validation.python.generate_publication_figures import (
        generate_candidate_publication_figures,
    )

    generate_candidate_publication_figures(C590_CANDIDATE_DIR)


def _sync_one(filename: str, latex_source: str) -> dict[str, str]:
    destination = _target_path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for source in _source_candidates(filename):
        if source.exists():
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            return {
                "filename": filename,
                "latex_source": latex_source,
                "source": source.relative_to(VERSION2_ROOT).as_posix(),
                "destination": destination.relative_to(VERSION2_ROOT).as_posix(),
            }
    raise FileNotFoundError(f"Missing retained source for report figure: {filename}")


def generate_unified_report_figures(*, verify_latex: bool = True) -> Path:
    """Synchronize the retained bundle and return its JSON manifest path."""

    references = _latex_figure_references()
    if "chua_nonlinearity_piecewise_vs_arctan.pdf" in references:
        _write_nonlinearity_comparison()
    _ensure_c590_publication_figures(references)
    rows = [_sync_one(name, source) for name, source in references.items()]

    missing_files = [name for name in references if not _target_path(name).exists()]
    if verify_latex and missing_files:
        raise RuntimeError(f"Missing unified-report figure files: {missing_files}")

    manifest = {
        "report": REPORT_TEX.relative_to(VERSION2_ROOT).as_posix(),
        "source_script": (
            "tools/unified_report_assets.py"
        ),
        "target_report": TARGET_REPORT_DIR.relative_to(VERSION2_ROOT).as_posix(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "recursive_latex_discovery": True,
        "figure_count": len(rows),
        "figures": rows,
    }
    manifest_path = TARGET_REPORT_DIR / "unified_report_figure_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate and verify figures for the unified Chua report."
    )
    parser.add_argument(
        "--no-verify-latex",
        action="store_true",
        help="Synchronize without the final missing-file assertion.",
    )
    args = parser.parse_args(argv)
    manifest_path = generate_unified_report_figures(
        verify_latex=not args.no_verify_latex
    )
    print(f"[Unified Report Figures] manifest={manifest_path}")


if __name__ == "__main__":
    main()
