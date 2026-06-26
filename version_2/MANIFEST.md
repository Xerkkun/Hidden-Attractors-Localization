# Version 2 manifest

`version_2/` is the active library-style distribution of `hidden-attractors-fo`.
It should run from this directory without depending on historical project-root
scripts.

## Public package

- `hidden_attractors/models/`: Chua parameters, vector fields, nonlinearities,
  equilibria, and Jacobians.
- `hidden_attractors/systems/`: `ChaoticSystem`, `LureSystem`, registry,
  aliases, and workflow-capability requirements.
- `hidden_attractors/seed_generation/` and `hidden_attractors/lure/`: DF,
  Nyquist, Lur'e, and seed reconstruction helpers.
- `hidden_attractors/integrations/`, `solvers/`, `native/`: integer and Caputo
  solver contracts, Python references, and C backends.
- `hidden_attractors/analysis/`, `diagnostics/`: finite-time trajectory,
  spectral, Poincare, boundedness, zero-one, Lyapunov, and method-comparison
  diagnostics.
- `hidden_attractors/verification/`, `basins/`: stability, neighborhoods,
  hiddenness contracts, candidate gates, and basin labels.
- `hidden_attractors/workflows/`: official protocol, continuation, robustness,
  basins, hiddenness, report generation, and reusable workflow specs.
- `hidden_attractors/plotting/`, `io.py`: canonical figure and CSV/JSON helpers.

The exhaustive symbol inventory is `docs/api_reference.md`; it lists every
function, class, and method defined under `hidden_attractors`.

## Public CLI

The package exposes one public console script:

```text
hidden-attractors
```

Maintained workflows are subcommands, including `run`, `init`,
`inspect-config`, `inspect`, `validate`, `protocol`, `seed`, `continuation`,
`robustness`, `hiddenness`, `basin`, `bifurcation`, `lyapunov`, `chaos-test`,
`published`, and `report`.

Historical standalone command names are legacy/deprecated and are not the public
release API.

## Official examples

- `examples/chua_integer_lure_reference/`: reproduced integer `q=1` Lur'e
  software reference.
- `examples/chua_nonsmooth_biased_hidden_attractor/`: proposed BDF methodology
  for the non-smooth fractional Chua case.
- `examples/chua_arctan_wu2023/`: Wu2023 bibliographic lane plus promoted c590
  Caputo lane; c590 is radius-limited to local radii `r <= 0.3` and is not
  a global basin proof.

Small API examples remain in `examples/*.py` and should import from
`hidden_attractors`.

## Evidence and outputs

- Promoted validation evidence: `validation/`
- Promoted scientific figures: `library_figures/`
- Ordinary/local outputs: `outputs/`, `validation_outputs/`, `runs*/`, `figures/`
- Release packaging metadata: `release_package/`

DF/Nyquist, continuation, plots, Lyapunov, FFT/PSD, Poincare, and zero-one tests
are diagnostics or seed tools. Hiddenness labels require sampled equilibrium
neighborhood or basin evidence under a recorded numerical contract.

## Legacy material

`tools/legacy/` preserves historical scripts for traceability only. Reusable
logic should live under `hidden_attractors/`, with CLI access routed through the
unified `hidden-attractors` command.
