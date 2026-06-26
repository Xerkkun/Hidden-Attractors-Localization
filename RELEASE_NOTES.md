# Release Notes

## 1.0.0

This release prepares `hidden-attractors-fo` as the active Python library
distribution under `version_2/`. It does not change numerical parameters,
seeds, tolerances, classifiers, or promoted scientific conclusions.

### Release contents

- Unified public CLI: `hidden-attractors`.
- Three official report examples: integer Chua reference, non-smooth fractional
  BDF methodology, and arctan Wu2023/c590 audit lane.
- Exhaustive API inventory in `version_2/docs/api_reference.md` generated from
  the active `hidden_attractors` package.
- Updated README, quick start, getting started guide, examples index, user
  manual, release manifest, and reproducibility metadata.
- Release readiness metadata under `version_2/release_package/`.

### Evidence boundary

The integer `q=1` Chua route is the reproduced software reference. The Danca
2017 non-smooth fractional case is a partial reference implementation because
key published numerical details are not reported. The official nearby fractional
candidate is rejected/self-excited under the current neighborhood contract. The Wu2023 arctan remains a bibliographic ADM lane, while the canonical `validation/chua_fractional_arctan/` package promotes the c590 Caputo lane as radius-limited hiddenness evidence for local radii `r <= 0.3`.

DF/Nyquist, continuation, plots, Lyapunov estimates, FFT/PSD, Poincare sections,
and 0-1 tests are diagnostics or seed tools; they do not certify hiddenness by
themselves.

### Citation and license

The package license remains MIT. Archived DOI: `10.17605/OSF.IO/ZGK74`.
