# Figure Gallery

Figures for the active reports are managed under the canonical library figure
tree, not as copied run images inside `docs/assets/figures/`.

```text
library_figures/by_report/df_nc_chua/
library_figures/by_report/unified_chua_fractional/
```

The `df_nc_chua` target supports the current three-example report: integer
Chua, nonsmooth fractional Chua, and fractional arctan/Wu2023-c590. The
`unified_chua_fractional` target is retained only for figures still referenced
by `docs/reporte_unificado_chua_fraccionario.tex`.

Ordinary run folders under `outputs/` and dated `library_figures/by_run/`
directories are provenance or regeneration inputs. They should not be cited as
report assets unless they are promoted into `library_figures/by_report/`.

## Active Report Families

- Integer Chua reference figures: transfer closure, continuation, final
  attractor, hiddenness controls, basin cuts, FFT, PSD, and Lyapunov
  convergence.
- Nonsmooth fractional Chua figures: Danca reference reproduction, centered
  control route, biased DF candidate, hiddenness spheres, and extended-radius
  heatmaps.
- Fractional arctan figures: Wu2023 non-promoted reproduction and the c590
  candidate audit, including spherical hiddenness probes and contact heatmaps.

## Retired Local Copies

Older copied figures formerly stored under
`docs/assets/figures/chua_fractional_report/` were local documentation assets.
They are not the active source for the report figures after the migration to
`library_figures/by_report/`.
