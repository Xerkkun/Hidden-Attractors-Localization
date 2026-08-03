# Integer hidden-chaos search

The maintained integer workflow separates three questions that must not be
collapsed into one label:

1. **Localization:** can the declared equations, Jacobian, exact Lur'e form,
   transfer condition, and describing function generate a seed directly?
2. **Dynamics:** does a bounded post-transient trajectory retain a robustly
   positive largest Lyapunov exponent, with compatible return-map evidence?
3. **Hiddenness:** do finite probes around every declared equilibrium avoid the
   candidate's calibrated reference cloud?

Only a candidate that passes the dynamics and hiddenness gates receives the
finite label `chaotic_hidden_under_tested_neighborhoods`.

## MAVPD route

For the modified autonomous Van der Pol--Duffing system,

\[
\dot y_1=\delta\gamma y_1+\delta y_2-\delta y_1^3,\qquad
\dot y_2=y_1-\xi y_2-y_3,\qquad
\dot y_3=\rho y_2,
\]

the exact scalar Lur'e split uses `sigma=y1`, `psi(sigma)=sigma^3`,
`b=(-delta,0,0)`, and `c=(1,0,0)`. The primary route calls
`integer_lure_seed(...)` and `continue_integer_lure_seed`. The direct seed API
has no `nscan`, `search_route`, or scan-fallback argument. It does not use a
frequency grid or values copied from a report. The declared bounds
`1e-5 <= omega <= 50` are only an admissibility interval for exact polynomial
roots.

The maintained code writes the transfer as
`G(s)=c.T@(P-sI)^(-1)@b`, whereas the Wolfram comparison also records the
standard resolvent `W(s)=c.T@(sI-P)^(-1)@b=-G(s)`. Thus the code equation
`k=-1/Re(G(i*omega))` and the Wolfram equation `k=1/Re(W(i*omega))` are
identical conventions, not different seed formulas.

The alternative path is permitted only after neither direct base branch passes
the declared finite-time LLE chaos screen. Among successful base branches, it
recomputes which one has the lowest direct harmonic frequency; that branch's
lambda-continuation endpoint supplies the alternative path. In the maintained
run this rule selects branch 0 (`omega0=10.5975230`). Branch 1 is screened at
the base point but is not transported through A1, a declared coverage limit.
`continue_integer_parameter_path` takes a
system factory and a complete parameter dictionary at every node. It calls
`mavpd_2023_system(parameters)` at each step, so the equations and the Lur'e
matrix cannot silently retain stale parameters.

At the nonzero equilibria the characteristic polynomial is

\[
p(\lambda)=\lambda^3+a_1\lambda^2+a_2\lambda+a_3,
\]

with

\[
a_1=\xi+2\delta\gamma,\quad
a_2=\rho-\delta+2\delta\gamma\xi,\quad
a_3=2\delta\gamma\rho.
\]

Writing `g=2 delta gamma`, the Hopf equality `a1*a2=a3` becomes

\[
\xi g^2+(\xi^2-\delta)g+\xi(\rho-\delta)=0.
\]

`mavpd_hopf_gamma_boundaries` solves this polynomial from the active
parameters. This is a Routh--Hurwitz imaginary-pair crossing and is treated as
a candidate Hopf boundary: transversality and Hopf nondegeneracy are not
asserted. The example continues through declared offsets from this boundary;
it does not perform a blind multiparameter grid.

The cited article supplies the differential-equation family. `xi=2.85` is a
declared local continuation endpoint, not a coordinate selected by the
candidate screen. At that endpoint, the screen selects
`gamma=gamma_H+0.010` without reading an expected offset. Neither value is
copied from or attributed as a published parameter tuple in that article.

## Executed pipeline

1. `mavpd_2023_system` declares the vector field, analytic Jacobian,
   equilibria, and exact Lur'e form.
2. `integer_lure_seed` solves the direct integer transfer polynomial and
   evaluates the cubic describing function. Both positive branches produce
   `omega0`, `k`, `a0`, and the harmonic seed anew.
3. `continue_integer_lure_seed` performs the ordinary lambda continuation for
   each branch. Short variational-QR runs show that neither base branch passes
   the declared screen; this finite negative screen is the recorded trigger for
   the alternative and is not a proof of regularity.
4. The alternative-source rule selects the successful direct branch with the
   lowest direct harmonic frequency. `mavpd_hopf_gamma_boundaries` derives
   the two nonzero-equilibrium Routh--Hurwitz crossing candidates from the
   active equations. `continue_integer_parameter_path` first transports the
   state to the declared `xi` endpoint, then in `gamma`, rebuilding the
   complete system at every node.
5. Declared Hopf offsets are screened jointly by the largest finite-time
   Lyapunov exponent and probes around `E0`. This is a structured local
   refinement, not a frequency or arbitrary initial-condition sweep.
6. The selected node is rerun with stricter tolerances and a longer variational
   window. A combined Lyapunov control perturbs the transported candidate
   initial state (not the original harmonic seed) and simultaneously changes
   tolerances, maximum step, burn-in, and accumulation horizon. A fixed-step
   EFORK trajectory supplies a separate integrator control.
7. `calibrate_attractor_reference` constructs a cloud-distance classifier from
   disjoint temporal windows of the same candidate trajectory. Equilibrium
   negative controls test separation but do not set its threshold.
8. `run_integer_hidden_chaos_controls` probes all equilibria on three radii;
   a separate test follows both signs of the unstable eigenvector of `E0`.
9. `evaluate_candidate_gate` combines, but does not conflate, boundedness,
   chaos diagnostics, robustness, and finite sampled hiddenness.

`validation/wolfram/cases/mavpd_integer.wl` independently starts from the
source vector field and derives the Lur'e matrices, Jacobian, equilibria,
transfer polynomial, describing function, both direct seeds,
`det(lambda I-J)` at `E+/-`, and the Routh--Hurwitz Hopf equation. It does not
read the Python artifacts or a report equation. Its `Step 2b` block declares
`xi=57/20` (the YAML's local continuation endpoint) and `1/100` (the
Python-selected offset) only for a posterior stability check. Wolfram selects
neither value. It derives the crossing boundary and imaginary-pair frequency,
but does not run or certify the chaotic or hiddenness search.

## Numerical functions

- `dop853_q1_integrate`: adaptive integer ODE trajectory with explicit
  tolerances and output sampling.
- `integer_system_dop853_variational_qr`: dimension-independent DOP853
  integration of state and variational equations with segmentwise QR
  normalization, returning `AdaptiveLyapunovResult`.
- `compute_boundedness_metrics`: finite-state and post-transient boundedness
  checks under the declared divergence radius.
- `detect_poincare_crossings` and `summarize_poincare_points`: sampled
  sign-change interpolation and nonduplicate section geometry.
- `zero_one_test`: deterministic-seed 0--1 diagnostics for flow strides and
  Poincare return coordinates.
- `spectral_diagnostics_multicoordinate`: multicoordinate normalized FFT-power
  support diagnostic; it is not a Welch PSD.
- `efork_q1_integrate`: fixed-step integer EFORK trajectory used only as an
  integrator/cloud cross-check of the strict DOP853 candidate.
- `calibrate_attractor_reference` and `classify_cloud_against_reference`: set
  the normalized cloud-distance threshold from independent candidate windows
  and classify probe or control clouds; negative controls check separation but
  do not set the threshold.
- `run_integer_hidden_chaos_controls`: regenerates probe centers from the
  active equilibria and integrates deterministic sphere or ball probes.
- `evaluate_candidate_gate`: keeps hiddenness promotion separate from the
  stricter joint hidden-chaos promotion.
- `save_figure_pair_local`, `promote_local_figure_pair`, and the runner's
  `_finalize_figure_manifest`: create local title-free PNG/PDF pairs and delay
  global promotion until the full gate passes.
- `_new_run_status`, `_record_run_phase`, `_verify_recorded_artifacts`, and
  `_config_sha256`: bind run ID, resolved configuration, completed phases, and
  artifact hashes in the resumable ledger.

The lambda continuation uses fixed-step EFORK. Parameter continuation,
candidate trajectories, variational exponents, and neighborhood probes use
adaptive DOP853. EFORK is then reused as a fixed-step cross-check of the final
candidate cloud; it is not the solver that produces the strict exponent.

For the strict variational calculation, each QR segment starts from the
orthonormal basis produced by the previous segment:

\[
\dot Y_m=J(y(t))Y_m,\quad Y_m(t_{m-1})=Q_{m-1},\quad
Y_m(t_m)=Q_mR_m,\quad Q_0=I,
\]

\[
\widehat\lambda_i(T)=T^{-1}\sum_m\log|R_{m,ii}|.
\]

The divergence consistency control uses
`div f = delta*gamma - xi - 3*delta*y1^2`. It compares the exponent sum from
the independent variational accumulation (`T_LE=1200`, after its own burn-in)
with the sampled divergence average over the retained state trajectory
`[300,900]`. Because these are different integrations and windows, the
reported residual is a finite-time consistency check, not the same-window
trace identity evaluated tautologically.

For candidate windows `R_j`, the cloud scale and probe statistic are

\[
S=\|\operatorname{ptp}(\operatorname{vstack}(R_j),\mathrm{axis}=0)\|_2,
\qquad D(A)=\operatorname{median}_j d_N(A,R_j),
\]

where `d_N` is the symmetric median nearest-neighbor distance divided by `S`.
If `b` is the 0.95 quantile of between-window distances, the acceptance
threshold is `tau=3b` and the ambiguity margin is `max(0.25*tau,b)`. Negative
controls must remain beyond `tau+margin`; otherwise calibration reports
`overlapping_controls`.

The 0--1 statistic is evaluated across flow sampling strides and on the
Poincare return sequence. Three of the 10 recorded flow strides receive the
local chaotic-candidate label, one receives the regular-candidate label, and
six are inconclusive. The gate uses the return sequence, not a selected flow
stride. Poincare crossings are obtained by linear interpolation of sampled
sign changes, not by an exact event map. The normalized FFT result is retained
as `spectral_inconclusive` and is supporting-only; it cannot veto or certify
the variational result.

## Reproduction and outputs

```powershell
& '..\.venv\Scripts\python.exe' examples\modified_van_der_pol_duffing_integer_hidden_chaos_search\run_example.py --output-dir tmp\mavpd-integer-hidden-chaos-full-staging
```

The run writes the direct derivation, parameter path, candidate-screening
contract, every preliminary probe, trajectory, Lyapunov convergence, 0--1
sensitivity, Poincare section, FFT power, equilibrium stability, every final
hiddenness probe, robustness controls, joint gate, figures, and phase timings
under the explicit staging directory. After validation, the complete bundle
can be copied to `validation/reference_cases/mavpd_integer_hidden_chaos/`. Every
maintained figure is exported as both PNG and vector PDF without an internal
title.

The runner freezes a SHA-256 bundle of every maintained scientific source at
startup and checks it after each phase. A source change aborts promotion rather
than combining artifacts from different implementations. It also maintains a
run ID, resolved-configuration digest, phase ledger, and artifact hashes.
Resume mode accepts only an in-progress full run whose contract, search,
trajectory, diagnostics, timings, selected endpoint, and source bundle all
match. Quick mode is isolated under `tmp`, is explicitly non-promotable, and
cannot update the global figure manifest. Full figure promotion occurs only
after the joint gate passes.

Before copying a full staging bundle into the canonical reference directory,
verify that it is a completed non-quick run, that the joint gate passed, and
that every ledger hash matches its local artifact:

```powershell
$stage = Resolve-Path 'tmp\mavpd-integer-hidden-chaos-full-staging'
$status = Get-Content -Raw (Join-Path $stage 'run_status.json') | ConvertFrom-Json
if ($status.status -ne 'complete' -or $status.quick_mode -ne $false) {
    throw 'staging is incomplete or quick'
}
$gate = Get-Content -Raw (Join-Path $stage '09_candidate_gate.json') | ConvertFrom-Json
if ($gate.gate.chaotic_hidden_promotion_allowed -ne $true) {
    throw 'joint scientific gate did not pass'
}
foreach ($item in $status.artifacts.PSObject.Properties) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $stage $item.Name)).Hash.ToLowerInvariant()
    if ($actual -ne ([string]$item.Value).ToLowerInvariant()) {
        throw "artifact hash mismatch: $($item.Name)"
    }
}
$canonical = Join-Path (Resolve-Path '.').Path 'validation\reference_cases\mavpd_integer_hidden_chaos'
if (Test-Path -LiteralPath $canonical) {
    throw 'archive the existing canonical bundle first'
}
Copy-Item -LiteralPath $stage -Destination $canonical -Recurse
```

The final lines promote only after the checks. If the canonical directory
already exists, archive it to an explicit recoverable path before rerunning the
copy; never merge or overwrite it silently.

For an interrupted full run, the following command is allowed only when the
completed contract, selection, trajectory, diagnostics, and stability
artifacts are already present. It recomputes the reference calibration before
rerunning the hiddenness probes and the joint gate:

```powershell
& '..\.venv\Scripts\python.exe' examples\modified_van_der_pol_duffing_integer_hidden_chaos_search\run_example.py --resume-validated-candidate --output-dir tmp\mavpd-integer-hidden-chaos-full-staging
```

## Recorded result

The maintained full run used the declared endpoint and selected

\[
(\xi,\gamma,\delta,\rho)
=(2.85,\;0.15380379839949113,\;100,\;200),
\qquad \gamma_H=0.14380379839949112.
\]

Only `gamma` is selected by this screen; `xi` is not jointly optimized. The
selection additionally requires Lyapunov status `ok`, zero ambiguous probes,
and zero numerical failures. The offset screen records only a finite-time
largest exponent and sampled
contacts from `E0`; it does not locate a bifurcation boundary or classify every
node globally:

| Hopf offset | Largest exponent | Contacts from 12 `E0` probes | Screen interpretation |
| ---: | ---: | ---: | --- |
| 0.002 | 0.92911 | 6 | positive LLE; sampled `E0` contact |
| 0.003 | 0.89340 | 6 | positive LLE; sampled `E0` contact |
| 0.005 | 0.18832 | 6 | positive LLE; sampled `E0` contact |
| 0.008 | 0.81221 | 0 | passes the declared finite screen |
| 0.010 | 0.72394 | 0 | selected by the declared rule |
| 0.012 | 0.31512 | 0 | positive LLE below the screening threshold |
| 0.015 | -0.00887 | 0 | does not pass the positive-LLE screen |

The strict variational run yielded

\[
(\lambda_1,\lambda_2,\lambda_3)
=(0.71335040,\;-0.00090588,\;-20.92362496),
\qquad D_{KY}=2.03405.
\]

The combined initial-condition, tolerance, maximum-step, and horizon control
retained a positive largest exponent (`0.68303381`). The signed residual
between the exponent sum and the mean vector-field divergence sampled on a
distinct trajectory window was `-0.00669698`; it is a consistency residual,
not a same-window identity. The
Poincare section retained 1,174 nonduplicate crossings; its two
return-coordinate 0--1 statistics were `0.998516` and `0.998432`. EFORK
reached the same calibrated reference cloud.

The main hiddenness contract contains 108 probes: three equilibria, three
radii (`1e-5`, `1e-3`, `1e-2`), and 12 deterministic directions. All 108 ended
at an equilibrium; none contacted the candidate cloud and none failed. Four
additional probes along both signs of the unstable `E0` eigenvector also had
zero contacts. The resulting finite label is
`chaotic_hidden_under_tested_neighborhoods`.

## Recorded time

`phase_timings.csv` is written after every completed phase using
`time.perf_counter`. A resumed run preserves those recorded timers and never
reconstructs them from mutable file timestamps. The timing table below is
updated only from a complete canonical run; it measures this workstation and
configuration and is not a general performance benchmark. The
`candidate_gate` timing includes both evaluation of the joint gate and figure
manifest finalization/promotion.

The focused software contract is rerun separately with:

```powershell
& '..\.venv\Scripts\python.exe' -m pytest `
  tests\test_candidate_gate.py `
  tests\test_figure_export_contract.py `
  tests\test_no_direct_savefig_outside_export.py `
  tests\test_integer_lure_workflow.py `
  tests\test_modified_van_der_pol_duffing.py `
  tests\test_integer_hidden_chaos_workflow.py `
  tests\test_mavpd_integer_hidden_chaos_example.py `
  tests\test_integer_nonchua_wolfram.py -q
```

`PENDING_FINAL_FOCUSED_SUITE_RESULT` is intentionally left for replacement by
the frozen final run; no earlier test count is promoted here.

## Evidence boundary

A positive finite-time exponent is not, by itself, a proof of chaos. Likewise,
finite zero-contact probes do not prove global hiddenness. The maintained claim
is restricted to the declared solver tolerances, observation windows,
directions, radii, reference calibration, and robustness controls distributed
across `reproducibility.yaml`, `05_chaos_diagnostics.json`,
`07_hiddenness_summary.json`, `09_candidate_gate.json`, and
`run_manifest.json`. The candidate has not yet received an independent Julia
reproduction; the earlier Python--Julia MAVPD comparison covers the periodic
`gamma=0.1` audit, not this locally derived chaotic point.

The maintained Kalman--Fitts reference currently supports a hidden periodic
cycle, not a promoted hidden-chaos result. Any future chaotic search for that
family requires its own source, executable route, long-run diagnostics, and
formal neighborhood probes; exploratory states are not evidence for this
MAVPD example.
