# Geometric-topological initial pilot

Campaign: `tg_pll_mavpd_wu_b0_b2_20260812`<br>
Execution date: 2026-08-12<br>
Protocol: `geometric_topological_initial_pilot_v1`

This directory is the canonical finite-time B0-B2 pilot bundle for the integer
lead-lag PLL, integer MAVPD at ξ=3.1, and Caputo Wu arctangent Chua system at
(q=0.99). The run completed its declared orchestration; it is not a global
hiddenness proof, a complete dynamic edge-tracking result, or a topological
certificate.

## Inventory and bounded results

- 24 seed-bank rows;
- 90 trajectory rows, all with integration status `ok`;
- 12 initial edge-bracket rows;
- 3 frozen case contracts;
- 90 trajectory CSV files and 90 companion metadata records;
- 5 campaign figures in PNG/PDF pairs.

| Case | Evidence decision | Precise interpretation |
| --- | --- | --- |
| PLL (q=1) | EV-TG3 | Only `pll_pp_plus` passed the two-solver B1/B2 destination and observable gates, reaching `equilibrium:E_focus`. The case level is the maximum attained by one route, not an all-seeds certificate. |
| MAVPD (q=1), ξ=3.1 | EV-TG2 | PP/FDF seeds reproduced finite inner/outer reference destinations, but the B1/B2 observable gate did not close for the PP routes. |
| Wu Caputo (q=0.99) | EV-TG2 | Native ABM PECE and EFORK3 used causal full memory. The FPP-A± trajectories remained unresolved transients at the tested B0/B2 horizons. |

Paired central-inversion diagnostics gave maximum scaled odd residual 0.0 for
MAVPD PP± at B2 with DOP853/RK4 and for Wu FPP-A± at B0/B2 with both
full-memory solvers. These are stored-trajectory covariance checks, not global
symmetry theorems.

The PLL bracket was rejected because its endpoints were unresolved at the
required confirmation levels (width unchanged at
(8.928571428571065\times10^{-4})). The MAVPD segment stopped on a third
destination after two iterations, reducing its scaled width from
(1.4506875921190598) to (0.7253437960595299). Initial-data boundary
bisection is recorded as EV-TG2; it cannot establish EV-TG4.

## Artifact map

- `campaign_manifest.json`: frozen campaign and budget contract;
- `contracts/`: complete mathematical/numerical contract for each case;
- `seed_bank.csv`: seed provenance and Caputo history semantics;
- `trajectory_metrics.csv`: classifier, solver, budget, artifact, and hash rows;
- `trajectories/` and `metadata/`: raw finite trajectories and companion records;
- `evidence_decisions.json`: per-route gates, symmetry diagnostics, and edge outcomes;
- `edge_brackets.csv`: endpoint confirmations, probes, iterations, and summaries;
- `run_summary.json`: compact completion counts and scientific limit;
- `figures_manifest.json` and `figures/`: traceable finite-time visualizations;
- `outer_enclosures.json`: explicit `pending_TG7_not_executed` record.

To reproduce from the repository root, run the maintained module with this run
identifier and a fresh output directory:

```text
python -m hidden_attractors.workflows.geometric_topological_pilot --run-id tg_pll_mavpd_wu_b0_b2_20260812 --output-root validation/06_geometric_topological_campaign/runs/tg_pll_mavpd_wu_b0_b2_20260812_reproduction
```

Do not interpret the four projected near-periodic Wu labels as exact
nonconstant periodic solutions of the Caputo IVP. TG7 outer enclosures, Conley
indices, EV-TG4 evolved-edge tracking, and equilibrium-neighborhood hiddenness
surveys remain pending.
