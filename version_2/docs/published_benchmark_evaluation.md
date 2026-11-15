# Published cloned-dynamics verdicts, version 2

The evaluator used for new runs is
`published_tolerance_and_required_sign_v2`. A quantitative pass requires all
absolute component errors to be strictly below `tolerance_abs_target` and,
when the YAML sets `sign_pattern_required: true`, exact component signs to
match as well. A sign-pattern pass requires matching signs and first-exponent
error strictly below `tolerance_abs_initial`; it is not full quantitative
validation. An unsuccessful solver or a nonfinite spectrum is inconclusive.
Reduced smoke runs never count as published quantitative validation.

The historical evaluator accepted the quantitative tolerance before checking
the sign condition. It remains available explicitly as
`published_tolerance_first_v1` for reproducibility. Select a version through
`run_benchmark_row(..., criteria_version=...)`, YAML
`expected.evaluation_criteria_version`, or the runner's
`--evaluation-criteria-version` argument. The explicit API/CLI choice overrides
YAML. With neither choice, V2 applies. Existing sealed results are unchanged.

Every new record preserves the established fields and also records the
criteria version, tolerances, whether the required sign gate was met, and the
`legacy_status` obtained from the same spectrum. The near-zero diagnostic
threshold defaults to 0.02, consistent with the earlier discrepancy
diagnostic. It is a reporting threshold only: values are never rounded to
zero and this diagnostic never changes the verdict. Future criteria must be
declared before their numerical campaign rather than tuned after observation.

## Evaluating archived spectra

`evaluate_benchmark_spectrum(computed, published, expected, ...)` is a pure
evaluator. Running it on archived spectra produces a derived classification,
not a new integration. Record the original result and YAML hashes along with
the evaluator hash, keep the original status, and write to a new directory.
The 2026-11-09 thesis correction does this for all 24 newly audited Fischer
rows: V1 verdicts are reproduced for 24/24; V2 gives 7 quantitative passes,
8 sign-pattern passes, and 9 failures. One financial commensurate q=0.9 row
changes from quantitative pass to failure because its second-exponent sign
differs. This operation performs zero integrations. It does not resolve the
remaining spectral discrepancies or validate the complete method.

## Reference-rail correction and preserved legacy protocol

Fischer et al. (2020), DOI `10.1016/j.apnum.2020.03.027`, section 2.2,
equations (10), (20), and (21), describe a separately generated reference
trajectory, with each short fiducial/clone block anchored to the reference
at that physical time. The legacy `published_block_restart` implementation
instead advances its short fiducial endpoint into the next block. These
protocols are distinct for fractional memory. In the financial system they
also use different reference and clone steps (0.05 and 0.01), so equivalence
cannot be assumed at integer order either. The former runner used the
clone step only and did not generate the separate rail.

The equations, parameters, initial states, and printed table spectra match
the primary local paper on this audit. Preserve current outputs as
block-restarted local indicators. The new `fischer_reference_rail_v1` protocol
integrates the reference from a fixed zero lower limit, with complete history
and one PECE correction per step. Each short clone/fiducial integration still
restarts its own memory, then the next block is anchored to the separately
sampled reference. `reference_h` and clone `h` remain distinct. Exact physical
grid sampling is required; interpolation is available only through explicit
`reference_sampling="linear"`, recorded in metadata. It is not used by the
new published comparisons. A financial block at t=10 uses reference index
200 when h_ref=0.05, rather than clone index 1000.

The full reference uses `integrate_reference_abm`: O(N*d) storage, complete
causal PECE weights, direct sums inside the active convolution block, and
FFT contributions from completed blocks to every remaining time step. FFT
blocks do not restart memory. This avoids the N-by-N weight tables of the
short clone integrator. Controls compare all trajectory samples against
direct quadrature across nondividing block sizes, including heterogeneous
orders; constant-forcing Caputo solutions and a half-order Mittag--Leffler
solution supply analytical checks. The clone exponent estimator remains a
finite-time restarted-memory procedure, even though its reference retains
the full initial history.

The public `compute_cloned_dynamics_spectrum` default preserves
`legacy_block_restart_v1`. The benchmark helper and CLI select the reference
rail when h, N, and T_ref are declared, and record the selected protocol;
explicit legacy remains available. Result objects expose the reference
arrays, and metadata records their canonical float64 hashes, step, memory
policy, sampling rule, and convergence histories. New CLI runs require a new
`--output-dir` and reject one already containing recorded benchmark results.

There is a second source-backed correction for Jerk: section 3.1, printed
page 196, identifies T_clone=10 s as the more accurate choice (table 2,
condition III: N_clone=1000 and K=100). The original YAML used T_clone=1 s.
The audit preserves its 24-row reference-rail diagnostic and runs all eight
Jerk orders again under a separately frozen T_clone=10 contract. Those runs
are new integrations, not reclasifications. The contract records any
preceding numerical results already observed when this source-based choice
was fixed; no derivative-order row is selected by whether it passes.

For the corrected article configuration use the six derived YAML files in
`validation/lyapunov_benchmarks/fischer_reference_rail_v1`. They retain the
original YAML hashes and full original settings, but explicitly select rail
V1, evaluator V2, and the source-backed Jerk interval. The original benchmark
directory plus the new automatic rail isolates the anchoring change while
retaining its original Jerk T_clone=1; it is a diagnostic contrast rather
than the recommended article contract.

```powershell
python validation/python/run_cloned_dynamics_benchmarks.py `
  --benchmarks-dir validation/lyapunov_benchmarks/fischer_reference_rail_v1 `
  --output-dir local_reports/fischer_article_contract_new_run
```

To reproduce the historical execution and evaluator into a fresh directory:

```powershell
python validation/python/run_cloned_dynamics_benchmarks.py `
  --execution-protocol legacy_block_restart_v1 `
  --evaluation-criteria-version published_tolerance_first_v1 `
  --output-dir local_reports/fischer_legacy_new_run
```

In the new full comparison, the original-YAML rail has 11 quantitative
passes, 6 sign-pattern passes and 7 failures. Substituting all eight
source-motivated Jerk T_clone=10 runs yields 18 quantitative passes, 4
sign-pattern passes and 2 failures. Financial (0.9,1,1) still crosses the
second-exponent sign; Jerk (0.7,1,1) still misses signs and tolerances. This
is partial article reproduction, not validation of every published row.

Do not replace historical values or widen tolerances to force a pass. Full
article reproduction remains subject to the actual component errors and
strict signs recorded for each row; it is separate from a software contract
passing its own numerical controls.

The Danca--Kuznetsov (2018) Rabinovich--Fabrikant equations, Jacobian,
parameters, initial state, step, horizon, and orthonormalization interval
match the local primary article and its appendices. The new audit still
misses the third published exponent by about 0.2525353 (tolerance 0.05).
That numerical discrepancy remains open; no equation change is justified
by this comparison.
