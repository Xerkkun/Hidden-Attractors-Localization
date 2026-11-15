# Finite Chua equilibrium classification

The current policy is `terminal_v2`. This is independent of the general
`destination_classifier_v1_1` used by geometric/topological workflows.
Neither policy establishes asymptotic convergence, chaos, or hiddenness.

## Numerical contract

An equilibrium label requires the final `CAP_WIN` consecutive samples to stay
within `EPS_EQ` of the same equilibrium. The native basin integrator counts
eligible samples at or after `t_burn`; the standalone hiddenness executable
checks its complete saved trajectory and uses its separate burn threshold for
Poincare contacts. All samples are checked for threshold crossing and numerical
failure. A qualified earlier equilibrium visit does not end integration or
override later escape. An insufficient terminal window is inconclusive.

The native basin API uses these finite class IDs:

| ID | `terminal_v2` interpretation |
|---|---|
| 0 | terminal equilibrium proximity |
| 1, 2 | bounded positive/negative mean heuristic |
| 3 | observed norm threshold crossing |
| 4 | inconclusive finite record |
| 5 | numerical failure |
| 6 | qualified earlier equilibrium visit, without terminal proximity |

The norm threshold is not a proof of divergence to infinity. Classes 1/2 do
not compare a target reference and do not establish attraction. Exceeding the
mean-radius bound without crossing the norm threshold produces inconclusive
class 4. A final equilibrium window also remains a finite observation, including
when the equilibrium itself is unstable.

```python
from hidden_attractors.native import BasinBackend
from hidden_attractors.basins import class_label

backend = BasinBackend.build()
policy = "terminal_v2"
code = backend.classify_point(
    [0.1, 0., 0.], q=1., h=.01, Lm=20., t_final=20., t_burn=5.,
    equilibrium_tol=.03, cap_win=90, divergence_norm=80.,
    classifier_policy=policy,
)
label = class_label(code, classifier_policy=policy)
```

Always save the policy, parameters, time grid, memory mode, equilibrium tolerance,
window length and norm thresholds with the result. Window lengths count samples:
`CAP_WIN=90` at `h=.01` spans `.89` units from its first to last observation.
The default API returns an integer for compatibility; the integer alone is not
a complete result contract. Existing `CLASS_LABELS` and `class_label` without a
policy retain historical text, so use the policy argument for current figures.

## Historical replay

`classifier_policy="legacy_v1"` explicitly selects the historical early-capture
rule. The original C symbol `classify_basin_point` retains its argument list and
legacy semantics. The new symbol `classify_basin_point_with_policy` takes one
additional integer (`0` legacy, `1` terminal); the library ABI is now 4 so stale
binaries are rejected and rebuilt. The integrator and fractional-memory formulas
are unchanged by this correction.

The standalone `chua_hidden_backend.c` executable defaults to
`--classifier-policy terminal_v2`; `--classifier-policy legacy_v1` replays its old
rule. New CSVs explicitly name the classifier. In the terminal policy they use
`EQ_TERMINAL`, `THRESHOLD_CROSSED`, `TARGET_CONTACT`, `OTHER_FINITE`,
`INCONCLUSIVE`, `EQ_VISIT` and `NUMERICAL_FAILURE`, and carry visit and first-event
indices. A visit can coexist with a later target-contact result: the contact
class remains the destination observation and the visit is a separate field.
An integration failure is a numerical failure; its event index is `-1` when
the integrator did not return the failing trajectory. The standalone and native
basin class-number schemes are distinct; use their respective labels.

Historical campaigns, CSVs and figures are not rewritten. In particular, the
504 archived C01 probes lack the complete initial-condition records and cannot
be relabelled from their aggregate counts. A future campaign must freeze new
initial conditions and its contract, save all trajectories and declare that it
is a new experiment. Pin frozen source files when reproducing the old audit;
do not infer an original-source hash from the current working tree.

## Validation

`tests/test_chua_terminal_classification.py` tests stable analytic decay, unstable
late escape, earlier visits, insufficient horizons and nonfinite tails. It also
integrates the C01 unstable-eigenvector perturbation with the native basin
backend: the old policy reports an early capture, while the terminal policy
observes the later norm threshold. Historical reference-grid tests explicitly
select `legacy_v1` and retain their original expected data.
