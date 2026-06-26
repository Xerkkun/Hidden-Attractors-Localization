"""Editorial figure bundle for the unified Chua report.

This module is intentionally report-specific: it gathers the canonical figures
used by ``docs/reporte_unificado_chua_fraccionario.tex`` into a single report
asset directory and verifies that every figure referenced by the LaTeX source is
present. It does not run simulations; it only regenerates figures from existing
workflow outputs when a maintained plotting routine already supports that.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from .export import export_figure


VERSION2_ROOT = Path(__file__).resolve().parents[2]
REPORT_TEX = VERSION2_ROOT / "docs" / "reporte_unificado_chua_fraccionario.tex"
SOURCE_REPORT_DIR = VERSION2_ROOT / "library_figures" / "by_report" / "df_nc_chua"
TARGET_REPORT_DIR = VERSION2_ROOT / "library_figures" / "by_report" / "unified_chua_fractional"
C590_CANDIDATE_DIR = (
    VERSION2_ROOT
    / "outputs"
    / "arctan_hidden_candidate_search"
    / "c590_q09999_seed9_candidate_20260623"
)


@dataclass(frozen=True)
class ReportFigure:
    filename: str
    section: str
    source: str = "df_nc_chua"


REPORT_FIGURES: tuple[ReportFigure, ...] = (
    ReportFigure("chua_nonlinearity_piecewise_vs_arctan.pdf", "modelos"),
    ReportFigure("matignon_complex_plane.pdf", "fundamentos"),
    ReportFigure("fig01_nyquist_df.pdf", "chua_entero"),
    ReportFigure("fig01c_transfer_real_imag.pdf", "chua_entero"),
    ReportFigure("fig02d_continuation_story.pdf", "chua_entero"),
    ReportFigure("fig03_final_attractor.pdf", "chua_entero"),
    ReportFigure("fig05b_hiddenness_overview.pdf", "chua_entero"),
    ReportFigure("fig11a_fft_x.pdf", "chua_entero"),
    ReportFigure("fig11d_psd_x.pdf", "chua_entero"),
    ReportFigure("fig13_lyapunov_convergence.pdf", "chua_entero"),
    ReportFigure("danca_frac_phase_3d.pdf", "chua_fraccionario_no_suave"),
    ReportFigure("danca_frac_projections.pdf", "chua_fraccionario_no_suave"),
    ReportFigure("danca_frac_time_series.pdf", "chua_fraccionario_no_suave"),
    ReportFigure("danca_frac_spectrum_x.pdf", "chua_fraccionario_no_suave"),
    ReportFigure("chua_frac_ns_fig01a_transfer_real.pdf", "chua_fraccionario_no_suave"),
    ReportFigure("chua_frac_ns_fig01b_transfer_imag.pdf", "chua_fraccionario_no_suave"),
    ReportFigure("chua_frac_ns_fig02_continuation_story.pdf", "chua_fraccionario_no_suave"),
    ReportFigure("chua_frac_ns_fig03_final_attractor.pdf", "chua_fraccionario_no_suave"),
    ReportFigure("chua_frac_ns_fig03abc_final_attractor_projections.pdf", "chua_fraccionario_no_suave"),
    ReportFigure("chua_frac_ns_fig04_time_series.pdf", "chua_fraccionario_no_suave"),
    ReportFigure("chua_frac_ns_fig11a_fft_x.pdf", "chua_fraccionario_no_suave"),
    ReportFigure("chua_frac_ns_fig12_lyapunov_two_methods.pdf", "chua_fraccionario_no_suave"),
    ReportFigure("chua_frac_ns_fig13_E0_hiddenness_spherical_3d.pdf", "chua_fraccionario_no_suave"),
    ReportFigure("chua_frac_ns_fig13_Ep_hiddenness_spherical_3d.pdf", "chua_fraccionario_no_suave"),
    ReportFigure("chua_frac_ns_fig13_Em_hiddenness_spherical_3d.pdf", "chua_fraccionario_no_suave"),
    ReportFigure("chua_frac_ns_fig14_hiddenness_contact_heatmap.pdf", "chua_fraccionario_no_suave"),
    ReportFigure("chua_frac_ns_biased_fig01a_transfer_real.pdf", "chua_fraccionario_no_suave_sesgado"),
    ReportFigure("chua_frac_ns_biased_fig01b_transfer_imag.pdf", "chua_fraccionario_no_suave_sesgado"),
    ReportFigure("chua_frac_ns_biased_fig02_continuation_story.pdf", "chua_fraccionario_no_suave_sesgado"),
    ReportFigure("chua_frac_ns_biased_fig03_attractor.pdf", "chua_fraccionario_no_suave_sesgado"),
    ReportFigure("chua_frac_ns_biased_fig04_projections.pdf", "chua_fraccionario_no_suave_sesgado"),
    ReportFigure("chua_frac_ns_biased_fig04_timeseries.pdf", "chua_fraccionario_no_suave_sesgado"),
    ReportFigure("chua_frac_ns_biased_fig11_fft.pdf", "chua_fraccionario_no_suave_sesgado"),
    ReportFigure("chua_frac_ns_biased_fig12_lyapunov.pdf", "chua_fraccionario_no_suave_sesgado"),
    ReportFigure("chua_frac_ns_biased_fig13_E0_hiddenness_spherical_3d.pdf", "chua_fraccionario_no_suave_sesgado"),
    ReportFigure("chua_frac_ns_biased_fig13_Ep_hiddenness_spherical_3d.pdf", "chua_fraccionario_no_suave_sesgado"),
    ReportFigure("chua_frac_ns_biased_fig13_Em_hiddenness_spherical_3d.pdf", "chua_fraccionario_no_suave_sesgado"),
    ReportFigure("chua_frac_ns_biased_fig14_hiddenness_contact_heatmap.pdf", "chua_fraccionario_no_suave_sesgado"),
    ReportFigure("chua_frac_ns_biased_fig15_extended_heatmap.pdf", "chua_fraccionario_no_suave_sesgado"),
    ReportFigure("wu2023_chua_fractional_arctan_q099_x0_plus_phase3d.pdf", "chua_arctan_wu2023"),
    ReportFigure("wu2023_chua_fractional_arctan_q099_x0_plus_projections.pdf", "chua_arctan_wu2023"),
    ReportFigure("wu2023_chua_fractional_arctan_q099_x0_plus_timeseries.pdf", "chua_arctan_wu2023"),
    ReportFigure("wu2023_chua_fractional_arctan_q099_x0_minus_phase3d.pdf", "chua_arctan_wu2023"),
    ReportFigure("wu2023_chua_fractional_arctan_q099_x0_minus_projections.pdf", "chua_arctan_wu2023"),
    ReportFigure("wu2023_chua_fractional_arctan_q099_x0_minus_timeseries.pdf", "chua_arctan_wu2023"),
    ReportFigure("chua_frac_arctan_c590_fig00_initial_seed.pdf", "chua_arctan_c590"),
    ReportFigure("chua_frac_arctan_c590_fig01_transfer_components.pdf", "chua_arctan_c590"),
    ReportFigure("chua_frac_arctan_c590_fig02a_linearized_vs_original.pdf", "chua_arctan_c590"),
    ReportFigure("chua_frac_arctan_c590_fig02b_continuation_path.pdf", "chua_arctan_c590"),
    ReportFigure("chua_frac_arctan_c590_fig03_attractor.pdf", "chua_arctan_c590"),
    ReportFigure("chua_frac_arctan_c590_fig04_timeseries.pdf", "chua_arctan_c590"),
    ReportFigure("chua_frac_arctan_c590_fig11_fft.pdf", "chua_arctan_c590"),
    ReportFigure("chua_frac_arctan_c590_fig12_lyapunov_two_methods.pdf", "chua_arctan_c590"),
    ReportFigure("chua_frac_arctan_c590_fig13_E0_hiddenness_spherical_3d.pdf", "chua_arctan_c590"),
    ReportFigure("chua_frac_arctan_c590_fig13_Ep_hiddenness_spherical_3d.pdf", "chua_arctan_c590"),
    ReportFigure("chua_frac_arctan_c590_fig13_Em_hiddenness_spherical_3d.pdf", "chua_arctan_c590"),
    ReportFigure("chua_frac_arctan_c590_fig14_hiddenness_contact_heatmap.pdf", "chua_arctan_c590"),
)


def _target_subdir(suffix: str) -> Path:
    return TARGET_REPORT_DIR / suffix.lstrip(".")


def _source_candidates(filename: str) -> Iterable[Path]:
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    yield SOURCE_REPORT_DIR / suffix.lstrip(".") / filename
    yield TARGET_REPORT_DIR / suffix.lstrip(".") / filename
    yield VERSION2_ROOT / "library_figures" / "current" / suffix.lstrip(".") / filename
    for run_dir in (VERSION2_ROOT / "library_figures" / "by_run").glob("*"):
        yield run_dir / suffix.lstrip(".") / filename
    if suffix == ".pdf":
        yield SOURCE_REPORT_DIR / "png" / f"{stem}.png"


def _write_nonlinearity_comparison() -> None:
    TARGET_REPORT_DIR.joinpath("pdf").mkdir(parents=True, exist_ok=True)
    TARGET_REPORT_DIR.joinpath("png").mkdir(parents=True, exist_ok=True)
    SOURCE_REPORT_DIR.joinpath("pdf").mkdir(parents=True, exist_ok=True)
    SOURCE_REPORT_DIR.joinpath("png").mkdir(parents=True, exist_ok=True)

    x = np.linspace(-3.0, 3.0, 1200)
    m0, m1 = -0.1768, -1.1468
    a1, a2, rho = 0.4, -1.5585, 1.0
    nonsmooth = m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1.0) - np.abs(x - 1.0))
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

    export_figure(
        fig,
        "chua_nonlinearity_piecewise_vs_arctan",
        "model_reference",
        {
            "caption_key": "chua_nonlinearity_piecewise_vs_arctan",
            "source_script": "hidden_attractors/plotting/generate_unified_report_figures.py",
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
        report_targets=["unified_chua_fractional", "df_nc_chua"],
    )
    plt.close(fig)


def _ensure_c590_publication_figures() -> None:
    expected = SOURCE_REPORT_DIR / "pdf" / "chua_frac_arctan_c590_fig03_attractor.pdf"
    if expected.exists():
        return
    from .generate_publication_figures import generate_candidate_publication_figures

    generate_candidate_publication_figures(C590_CANDIDATE_DIR)


def _sync_one(figure: ReportFigure) -> dict[str, str]:
    destination = _target_subdir(Path(figure.filename).suffix) / figure.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    for source in _source_candidates(figure.filename):
        if source.exists():
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            return {
                "filename": figure.filename,
                "section": figure.section,
                "source": str(source.relative_to(VERSION2_ROOT)).replace("\\", "/"),
                "destination": str(destination.relative_to(VERSION2_ROOT)).replace("\\", "/"),
            }
    raise FileNotFoundError(f"Missing report figure source: {figure.filename}")


def _latex_figure_references() -> set[str]:
    if not REPORT_TEX.exists():
        return set()
    text = REPORT_TEX.read_text(encoding="utf-8")
    pattern = re.compile(r"\\(?:reportinclude|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}")
    refs: set[str] = set()
    for match in pattern.finditer(text):
        value = match.group(1).strip()
        if value.startswith("#"):
            continue
        if value.endswith(".pdf") or value.endswith(".png"):
            refs.add(Path(value).name)
    return refs


def generate_unified_report_figures(*, verify_latex: bool = True) -> Path:
    _write_nonlinearity_comparison()
    _ensure_c590_publication_figures()

    manifest_rows = [_sync_one(figure) for figure in REPORT_FIGURES]

    if verify_latex:
        references = _latex_figure_references()
        expected = {figure.filename for figure in REPORT_FIGURES}
        undeclared = sorted(references - expected)
        missing_in_report = sorted(expected - references)
        missing_files = [
            name
            for name in references
            if not (TARGET_REPORT_DIR / Path(name).suffix.lstrip(".") / name).exists()
        ]
        if undeclared or missing_in_report or missing_files:
            raise RuntimeError(
                "Unified report figure mismatch:\n"
                f"undeclared_refs={undeclared}\n"
                f"declared_but_not_cited={missing_in_report}\n"
                f"missing_files={missing_files}"
            )

    manifest = {
        "report": "docs/reporte_unificado_chua_fraccionario.tex",
        "target_report": "library_figures/by_report/unified_chua_fractional",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "figures": manifest_rows,
    }
    manifest_path = TARGET_REPORT_DIR / "unified_report_figure_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate and verify figures for the unified Chua report.")
    parser.add_argument("--no-verify-latex", action="store_true", help="Do not compare the bundle against LaTeX references.")
    args = parser.parse_args(argv)
    manifest_path = generate_unified_report_figures(verify_latex=not args.no_verify_latex)
    print(f"[Unified Report Figures] manifest={manifest_path}")


if __name__ == "__main__":
    main()


