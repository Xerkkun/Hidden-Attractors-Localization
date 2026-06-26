# Hiddenness tests

Official hiddenness run for `m1_m1p2000_m0_m0p2000_branch_0` under the current fractional nonsmooth Chua candidate contract.

Parameters: `alpha=8.4562`, `beta=12.0732`, `gamma=0.0052`, `m0=-0.2`, `m1=-1.2`, `q=0.9998`, `h=0.01`.

The run used spherical probes around `E+`, `E-`, and `E0` with radii `1e-5`, `3e-5`, `1e-4`, `3e-4`, `1e-3`, and `1e-2`. The sample counts per radius were `100`, `150`, `200`, `250`, `300`, and `350`, for `4050` trajectories total. The run was executed with `4` workers.

Result summary:

| Equilibrium | Target hits | Decision |
| --- | ---: | --- |
| `E+` | 649 | `self_excited_contact_detected` |
| `E-` | 656 | `self_excited_contact_detected` |
| `E0` | 0 | `no_contact_detected` |

Global labels: `TARGET=1305`, `EQ=1232`, `DIV=1395`, `OTHER=118`.

Veredict: `chaotic_self_excited_candidate_not_hidden_under_tested_equilibrium_neighborhoods`.

The candidate is therefore not promoted as hidden under the tested equilibrium-neighborhood contract. It is reported as a chaotic self-excited candidate because contacts with the target candidate were detected from neighborhoods of the external equilibria.

Primary artifacts:

- `outputs/candidate_chaos_hiddenness/danca2017_chua_fractional_saturation_candidate/report/candidate_chaos_hiddenness_summary.json`
- `outputs/candidate_chaos_hiddenness/danca2017_chua_fractional_saturation_candidate/report/hiddenness_decisions.csv`
- `outputs/candidate_chaos_hiddenness/danca2017_chua_fractional_saturation_candidate/report/ball_sampling_results.csv`
