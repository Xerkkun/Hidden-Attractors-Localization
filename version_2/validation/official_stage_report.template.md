# Official Stage: `<stage>`

## Contract

Record `q`, `h`, time horizons, backend, memory policy, tolerances, decision
thresholds, random seed policy, and the matching JSON summary path.

## Inputs And Outputs

List stage inputs, machine-readable outputs, plots, and provenance. Use
`lambda` for continuation. Seed-generation stages must state that describing
functions and Machado/FDF generate seeds only.

## Decision

Use only official verdicts. Periodic-looking direct integration before
continuation is recorded as `pre_continuation_periodic` and remains admissible.
Hiddenness requires a robust dynamic reference, interior ball tests around
every equilibrium, and all required basin slices.
