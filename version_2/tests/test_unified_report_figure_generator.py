from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import tools.unified_report_assets as report_figures
from tools.unified_report_assets import (
    REPORT_TEX,
    _collect_tex_documents,
    _latex_figure_references,
)


def test_recursive_latex_discovery_follows_inputs(tmp_path: Path) -> None:
    root = tmp_path / "root.tex"
    child = tmp_path / "section.tex"
    root.write_text(r"\input{section}", encoding="utf-8")
    child.write_text(
        r"\reportinclude[width=1cm]{nested_figure.pdf}", encoding="utf-8"
    )
    documents = _collect_tex_documents(root)
    assert [path.name for path, _ in documents] == ["root.tex", "section.tex"]
    references = _latex_figure_references(root)
    assert set(references) == {"nested_figure.pdf"}
    assert Path(references["nested_figure.pdf"]).name == "section.tex"


def test_canonical_report_figure_references_are_discovered() -> None:
    references = _latex_figure_references(REPORT_TEX)
    assert len(references) >= 50
    assert "matignon_complex_plane.pdf" in references
    assert "chua_nonlinearity_piecewise_vs_arctan.pdf" in references


def test_nonlinearity_writer_uses_current_export_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    exported_pdf = tmp_path / "exported" / "figure.pdf"
    exported_png = tmp_path / "exported" / "figure.png"
    exported_pdf.parent.mkdir(parents=True)
    exported_pdf.write_bytes(b"pdf")
    exported_png.write_bytes(b"png")

    def fake_export(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return exported_pdf, exported_png

    monkeypatch.setattr(report_figures, "export_figure", fake_export)
    monkeypatch.setattr(report_figures, "TARGET_REPORT_DIR", tmp_path / "target")
    monkeypatch.setattr(report_figures, "SOURCE_REPORT_DIR", tmp_path / "source")

    report_figures._write_nonlinearity_comparison()

    assert captured["kwargs"]["export_targets"] == [
        "unified_chua_fractional",
        "df_nc_chua",
    ]
    assert "report_targets" not in captured["kwargs"]
    assert (tmp_path / "target" / "pdf" / "figure.pdf").is_file()
    assert (tmp_path / "source" / "png" / "figure.png").is_file()
    plt.close("all")
