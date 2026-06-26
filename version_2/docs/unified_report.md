# Unified Report

The synchronized manual metadata are defined in [docs/manual_manifest.yaml](manual_manifest.yaml); scientific claims remain governed by `THESIS_CLAIMS.md`.

For a complete user-facing description of installation, CLI usage, examples, outputs, evidence labels and limitations, see `USER_MANUAL.md`.

## Synchronization Targets

As defined in [docs/manual_manifest.yaml](manual_manifest.yaml), the project's documentation is synchronized across three manual targets:
1. **Markdown User Manual**: `USER_MANUAL.md`
2. **Unified LaTeX/PDF Report**: `docs/reporte_unificado_chua_fraccionario.tex`
3. **Web Docs & Home Page**: Located in the external repository `Xerkkun/hidden-attractors` under `src/content/docs/` and `src/pages/hidden-attractors/index.astro`.

> [!NOTE]
> **LaTeX report status**: The LaTeX source `docs/reporte_unificado_chua_fraccionario.tex` is the canonical unified report. Keep CLI commands, figures, arctan radius limits, and test-count summaries synchronized with `docs/manual_manifest.yaml` and `validation/freeze_audit/`.

The canonical LaTeX source is:

```text
docs/reporte_unificado_chua_fraccionario.tex
```

It consolidates the previous `.tex` files into a thematic structure:

- mathematical theory of the fractional Chua system;
- reference titles for each method;
- public library calls;
- real-trajectory examples;
- generated report figures promoted through `library_figures/`;
- Nyquist, continuation, bifurcation, spectral, hiddenness, and Lyapunov diagnostic sections with conservative evidence labels;
- the official fixed stage order and conservative interpretation of hiddenness.

The older standalone LaTeX reports were retired to avoid repeated, stale
results. Keep result updates in `reporte_unificado_chua_fraccionario.tex`.

## Build

From `version_2/docs`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error reporte_unificado_chua_fraccionario.tex
```

If `latexmk` is not available:

```bash
pdflatex -interaction=nonstopmode -halt-on-error reporte_unificado_chua_fraccionario.tex
```

## Figures

The active report figures live under:

```text
library_figures/by_report/df_nc_chua/
library_figures/by_report/unified_chua_fractional/
```

The `df_nc_chua` target is the canonical destination for the current
three-example report. The `unified_chua_fractional` target is kept only for
figures still referenced by `docs/reporte_unificado_chua_fraccionario.tex`.

Older copied figures under `docs/assets/figures/chua_fractional_report/` were
retired as active report assets. New or regenerated report figures should be
promoted through the library figure export path described in
[Figure Gallery](figure_gallery.md) and [Figure Export Policy](figure_export_policy.md),
not copied manually from ad-hoc `outputs/` directories.

## Reference Policy

Use [Code Reference Map](code_reference_map.md) before adding new calculation
code. New methods must state whether they come from a paper, a local numerical
contract, or an external library adapter.

The Chua arctan c590 result referenced by the unified report is radius-limited: the canonical validation package `validation/chua_fractional_arctan/` supports local radii `r <= 0.3` with 8400 finite probes and zero contacts, while macro-radius contacts remain extended audit evidence rather than a global basin proof.

