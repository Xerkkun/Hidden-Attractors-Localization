# Fractional Chua Arctan Wu2023 and c590 Example

This example keeps two lanes separate:

1. a Wu2023 bibliographic reproduction lane; and
2. a Caputo full-history dynamic lane for the c590 candidate, limited to local radii `r <= 0.3`.

The Wu2023 bibliographic lane remains partial. The c590 lane is one smooth-nonlinearity validation example with finite radius-limited hiddenness evidence for local radii `r <= 0.3` with 8400 probes and zero contacts; macro radii `1.0` and `2.0` remain extended audit evidence.

## Status

| Lane | Contract | Current status |
| --- | --- | --- |
| Wu2023 bibliographic lane | arctan algebra, equilibria, Lur'e split, reported initial conditions, local ADM recurrence | Bibliographic ADM reproduction; reported initial conditions classify as periodic/nonchaotic under the local ADM contract |
| c590 lane | Caputo ABM full-history search and neighborhood sampling | Reported for local radii `r <= 0.3` with 8400 probes and zero contacts; macro radii `1.0` and `2.0` retained as extended audit |

The local ADM recurrence used to mirror the article is not the same as a
full-memory Caputo ABM or EFORK validation. The c590 lane is a smooth-nonlinearity
methodology example, but only under the recorded local-radius contract and not
as a global basin proof.

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
python examples/chua_arctan_wu2023/run_example.py --steps search figures --quick
python examples/chua_arctan_wu2023/run_example.py --steps verification
python tools/rerun_c590_discovery.py --stage all --execute \
  --output-dir outputs/reproductions/c590_discovery
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
mode. The c590 lane follows its own recorded route: global and local integer
screens, a sweep of independent Caputo initial-value problems in \(q\), biased
restarts from interior states, and cross-step auditing. Every fractional run
accumulates its complete Caputo history from its declared initial time; history
is not transferred between independent values of \(q\) or between restarts.

## Hiddenness boundary

The c590 outputs may be labeled only as `hiddenness_supported_under_tested_local_radii`
for `r <= 0.3`. A contact detected on a sphere of large radius around an
equilibrium is not, by itself, evidence that the attractor is self-excited;
large-radius contacts are extended basin-geometry audits. The c590 lane should
not be described as a globally proved hidden attractor. Periodic or regular
post-transient behavior in the Wu2023 bibliographic lane is retained as
diagnostic evidence, not chaos evidence.
