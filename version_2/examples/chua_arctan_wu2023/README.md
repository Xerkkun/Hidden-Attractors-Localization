# Fractional Chua Arctan Wu2023 and c590 Example

This example keeps two lanes separate:

1. a Wu2023 bibliographic reproduction lane; and
2. a proposed Caputo full-history dynamic lane for the c590 candidate.

Neither lane is promoted as a validated hidden-attractor result in this release.

## Status

| Lane | Contract | Current status |
| --- | --- | --- |
| Wu2023 bibliographic lane | arctan algebra, equilibria, Lur'e split, reported initial conditions, local ADM recurrence | Partial/non-promoted; reported initial conditions classify as periodic/nonchaotic under the local ADM contract |
| Proposed c590 lane | Caputo ABM full-history search and neighborhood sampling | Candidate under review; macro-radius contacts require conservative status |

The local ADM recurrence used to mirror the article is not the same as a
full-memory Caputo ABM or EFORK validation. The c590 lane is a methodology lane
for testing a new arctan candidate, not a final hiddenness claim.

## Run

From `version_2`:

```bash
python examples/chua_arctan_wu2023/run_example.py --quick
python examples/chua_arctan_wu2023/run_example.py
python examples/chua_arctan_wu2023/run_example.py --all
```

Selected stages:

```bash
python examples/chua_arctan_wu2023/run_example.py --steps published
python examples/chua_arctan_wu2023/run_example.py --steps search continuation figures --quick
python examples/chua_arctan_wu2023/run_example.py --steps verification --all
```

`--run-published-trajectories` additionally calls the trajectory reproduction
script for the Wu2023 lane.

## Evidence files

- Configuration: `configs/chua_arctan_wu2023_caputo.json`
- Bibliographic validation package: `validation/reference_cases/fractional_chua_arctan_wu2023/`
- c590 candidate output: `outputs/arctan_hidden_candidate_search/c590_q09999_seed9_candidate_20260623/`
- Reproducibility contract: `reproducibility.yaml`

## Lur'e split

For the arctan nonlinearity, the documented split is:

```text
P = [[-alpha*(1+a1), alpha, 0], [1, -1, 1], [0, -beta, -gamma]]
b = [-alpha, 0, 0]^T
r = [1, 0, 0]^T
psi(sigma) = a2 * atan(rho * sigma)
```

The Wu2023 bibliographic seed lane uses the published integer Laplace transfer
mode. Fractional spectral seed generation is a separate configurable extension
and must be labeled as experimental unless promoted by a future validation
contract.

## Hiddenness boundary

No output from this example should be labeled as a confirmed hidden attractor
without sampled neighborhoods of `E0`, `E+`, and `E-`, a robust dynamic
reference, and a complete Caputo-compatible contract. Periodic or regular
post-transient behavior is retained as diagnostic evidence, not chaos evidence.
