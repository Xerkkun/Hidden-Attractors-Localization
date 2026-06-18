# CPC submission package

`hidden-attractors-fo` provides reproducible workflows for localization, reproduction, audit, and conservative classification of hidden-attractor candidates in integer- and commensurate fractional-order Chua/Lur'e systems.

## Install

```bash
cd version_2
python -m pip install -e ".[dev,analysis,legacy]"
```

## Minimal run

```bash
hidden-attractors --help
hidden-attractors init -e chua_fractional
hidden-attractors inspect-config -p chua_fractional
hidden-attractors validate contract --allow-pending
hidden-attractors validate cpc-readiness
```

## Evidence included

Promoted evidence is under `validation/`. Promoted scientific figures belong in `library_figures/` and are generated through `hidden_attractors.plotting.export.export_figure`.

## What is not claimed

The package does not certify global mathematical hiddenness. It records finite-time numerical evidence under explicit solver, memory, horizon, and neighborhood contracts. The arctan route is implemented algebraically but is not promoted as a validated hidden attractor.

## Authorship, supervision, and code provenance

Maria Fernanda Moreno Lopez is the principal author and maintainer. Dr. Esteban Tlelo Cuautle is acknowledged as doctoral thesis director and research guide. Dr. Oscar Martinez-Fuentes is acknowledged for reviewing the theoretical fractional-calculus component. Dr. Luis Gerardo de la Fraga is acknowledged for code provenance related to EFORK and the integer-order Lyapunov algorithm. Formal CPC article authorship should be confirmed before submission.
