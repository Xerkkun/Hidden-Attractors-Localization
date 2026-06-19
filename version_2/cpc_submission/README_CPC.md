# CPC submission package

`hidden-attractors-fo` provides reproducible workflows for theoretical-numerical search, localization, reproduction, audit, and conservative classification of hidden-attractor candidates in integer- and commensurate fractional-order Chua/Lur'e systems.

This directory is CPC preparation material. It is not evidence of CPC acceptance and does not introduce new scientific claims.

## Install

```bash
cd version_2
python -m pip install -e ".[dev,analysis,legacy]"
```

## Minimal checks

```bash
hidden-attractors --help
hidden-attractors validate cpc-readiness
hidden-attractors validate cpc-readiness --strict
hidden-attractors validate contract --allow-pending
python -m pytest -q -m "cpc_readiness"
```

`hidden-attractors validate cpc-readiness --strict` is expected to pass when only the declared final-submission items remain. `hidden-attractors validate cpc-readiness --submission-strict` is reserved for the final submission package and may fail while the manuscript, arctan validation, final scientific freeze audit, or executed sample outputs remain pending.

## CI and freeze audit boundary

The GitHub Actions CI matrix for the CPC cleanup has passed. This confirms package hygiene and cross-platform test execution for the current repository state. It does not replace the full scientific freeze audit, which remains a separate artifact to regenerate once final promoted validation cases are fixed for submission.

CI status: passed for current CPC cleanup. Freeze audit: last full scientific freeze audit corresponds to commit `2bcea3430c50d3fb4e5eb70c8621cb3550dcc59a` and must be regenerated only when the final scientific evidence set is frozen.

## Evidence included

Promoted evidence is under `validation/`. Promoted scientific figures belong in `library_figures/` and are generated through `hidden_attractors.plotting.export.export_figure`.

Local/regenerable outputs belong under `outputs/`, `validation_outputs/`, `runs*/`, or `figures/` and remain outside Git.

Editorial drafts, the official Elsevier/CPC template, and the final manuscript are prepared locally under ignored `paper/`. They are intentionally not tracked as part of the software repository readiness contract. The repository tracks the software package, promoted validation evidence, citation metadata, reproducibility notes, and CPC submission scaffolding under `version_2/cpc_submission/`.

## What is not claimed

The package does not certify global mathematical hiddenness. It records finite-time numerical evidence under explicit solver, memory, horizon, and tested-neighborhood contracts. The arctan route is implemented algebraically, pending full validation, and is not promoted as a validated hidden attractor.

## Authorship, supervision, and code provenance

Maria Fernanda Moreno Lopez is the principal author and maintainer. Dr. Esteban Tlelo Cuautle is acknowledged as doctoral thesis director and research guide. Dr. Oscar Martinez-Fuentes is acknowledged for reviewing the theoretical fractional-calculus component. Dr. Luis Gerardo de la Fraga is acknowledged for code provenance related to EFORK and the integer-order Lyapunov algorithm. Formal CPC article authorship should be confirmed before submission.
