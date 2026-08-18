# Geometric-Topological Localization Campaign

This page documents the experimental geometric-topological implementation
described in the master report. Repository validation records contain a bounded
B0--B2 pilot; initialization templates remain distinguishable from executed
evidence. An unexecuted record never means “no attractor”.

The implementation keeps four questions separate:

1. Which local geometric objects can generate or rank initial data?
2. Which seed records are genuinely different after scales, symmetries, and
   fractional-history contracts are taken into account?
3. What finite-time destination is supported by a trajectory?
4. Does a confirmed pair of destinations delimit a reproducible finite-resolution
   basin edge?

None of these questions alone proves attraction, chaos, global hiddenness,
Wada structure, or a Conley index.

## Mathematical objects implemented

For an autonomous integer-order flow

\[
\dot x=f(x),
\]

the geometric engine evaluates the three distinct objects

\[
f(x),\qquad J_f(x),\qquad a(x)=J_f(x)f(x).
\]

It then exposes:

- velocity surfaces \(f_i(x)=0\);
- acceleration surfaces \(a_i(x)=0\);
- the singular-Jacobian surface \(\det J_f(x)=0\), when the state dimension is
  square and small enough for a determinant;
- perpetual-point residuals \(J_f(p)f(p)\), with equilibria explicitly excluded;
- connecting-set residuals given by the minors
  \(f_i a_j-f_j a_i\), equivalently the component of \(a\) orthogonal to \(f\);
- declared affine symmetries and their numerical equivariance residuals; and
- exact regional geometry for piecewise-affine Chua, with no arbitrary
  classical Jacobian assigned on a switching surface.

The PP gate stores the raw norm and makes its decision with the scaled
residual used by the master protocol,

\[
r_{\mathrm{PP}}(x)=
\frac{\lVert J_f(x)f(x)\rVert_2}
{1+\lVert J_f(x)\rVert_2\lVert f(x)\rVert_2},
\]

where the matrix norm is spectral. The raw norm remains available for
diagnostics; `is_candidate` uses the normalized value. The FPP-A record also
keeps its Gamma-dependent startup-acceleration norm separate from this
geometric, (q)-independent zero-set test.

For the Caputo reset problem

\[
{}^{\mathrm C}D_{t_0+}^{q}x=f(x),\qquad 0<q\leq 1,
\]

the implemented FPP-A diagnostic uses the local Picard expansion

\[
x(t_0+h)=x_0+
\frac{h^q}{\Gamma(q+1)}f_0+
\frac{h^{2q}}{\Gamma(2q+1)}J_0f_0+O(h^{3q}).
\]

Thus \(J_f(p)f(p)=0\), with \(f(p)\ne0\), is treated as a vanishing
second Picard-Caputo startup coefficient. It is not called a global
fractional acceleration and it does not use the false chain rule
\({}^{\mathrm C}D^q f(x(t))=J_f(x){}^{\mathrm C}D^q x(t)\).

```python
import numpy as np

from hidden_attractors.geometry import (
    evaluate_differential_geometry,
    fractional_perpetual_startup_residual,
    perpetual_point_residual,
)
from hidden_attractors.systems import get_system

system = get_system("chua-nonsmooth")
state = np.array([0.2, 0.1, -0.3])

local = evaluate_differential_geometry(system, state)
pp = perpetual_point_residual(system, state)
fpp = fractional_perpetual_startup_residual(system, state, q=0.9998)

print(local.field, local.jacobian, local.jacobian_field)
print(pp.is_candidate, fpp.second_coefficient)
```

Use `chua_nonsmooth_partition(...)` for explicit regional PWL evaluation.
At \(x=\pm1\), the field is continuous but the classical Jacobian is not
unique; the engine reports that fact instead of choosing one side silently.

## Unified seed bank

`hidden_attractors.seed_bank` stores seeds from describing functions, the
theory-only Machado auxiliary transform, continuation, PP/FPP-A, critical surfaces,
connecting curves, KCC rankings, eigendirections, edge tracking, and imported
records. A record includes:

- system and parameter-set identifiers;
- integer or Caputo order and the value of \(q\);
- point, sampled-history, analytic-history, or continued-history semantics;
- lower terminal and history reference for Caputo data;
- source artifact, source record, residual, score, parent, and transform; and
- provenance metadata restricted to values that can be reconstructed from
  JSON without silently converting local objects or paths to strings.

Deduplication occurs only inside an identical semantic partition. Distances
are nondimensionalized by a declared coordinate scale. A caller that enables
symmetry reduction must supply a finite list and explicitly assert that it is
the complete group; this assertion is recorded but is not, by itself, a proof
of closure, inverses, or equivariance. History-based Caputo records are not
deduplicated by a point-only symmetry action. Periodic coordinates such as PLL
phase are handled by a wrapped cylindrical distance, not by pretending that
one translation generates a finite group.

```python
import numpy as np

from hidden_attractors.seed_bank import SeedRecord, build_seed_bank

records = [
    SeedRecord(
        seed_id="pll_pp_1",
        system_id="pll_lead_lag",
        route="perpetual_point",
        state=(0.0224, 1.5707963267948966),
    ),
    SeedRecord(
        seed_id="pll_pp_1_shifted",
        system_id="pll_lead_lag",
        route="imported",
        state=(0.0224, 1.5707963267948966 + 2.0 * np.pi),
    ),
]

bank = build_seed_bank(
    records,
    coordinate_scale=(0.05, 1.0),
    periodic_coordinates={1: 2.0 * np.pi},
)
assert len(bank.representatives) == 1
```

A point-valued Caputo record is never silently promoted to an inherited
history. Every Caputo record declares its lower terminal and initial time.
History-based records additionally require a history reference whose coverage
contains the complete interval from the lower terminal through the initial
time.

## Conservative destination classifier

`hidden_attractors.verification.destination_classifier` assigns one of six
finite-time labels:

| Label | Meaning under the declared window |
| --- | --- |
| `equilibrium` | The final window converges to a supplied equilibrium. |
| `periodic` | Spectral, window-drift, and return tests support near-periodic motion. |
| `recurrent` | A bounded, non-collapsed tail passes stationarity and recurrence gates. |
| `escape` | A declared scaled radius is exceeded persistently. |
| `transient` | The trajectory is finite and bounded but has not settled. |
| `ambiguous` | Data, solver status, reference separation, or threshold margins are insufficient. |

Recurrence requires more than a nearby pair of samples: the classifier combines
a Theiler window, a minimum physical time lag, a prior excursion of declared
radius, and repeated returns. This rejects the common false positive produced
by a slowly drifting monotone trajectory. Campaign runs require an explicit
`coordinate_scale` by default; inferred scales are available only when the
contract deliberately relaxes that gate and are tagged as diagnostic-only.
Escape can instead use an explicitly declared absolute radius. For Caputo
systems, `periodic` is automatically qualified as a projected near-periodic
finite-time label, not an exact nonconstant periodic solution of the hereditary
IVP.

Destination-classifier schema `1.1` separates return incidence from return
multiplicity. `excursion_return_pair_count` counts all qualifying forward-time
pairs and retains the historical alias `excursion_return_count`.
`excursion_return_anchor_count` counts distinct starting samples with at least
one qualifying return, while `excursion_return_fraction` is that anchor count
divided by `recurrence_sample_count` and therefore lies in `[0, 1]`.
`excursion_return_pairs_per_sample` preserves the former unbounded quantity,
and `excursion_return_mean_multiplicity` records the mean number of qualifying
return pairs per returning anchor. The recurrence count gate continues to use
the pair count; the return-fraction gate uses the bounded anchor fraction.

```python
from hidden_attractors.verification.destination_classifier import (
    DestinationClassifierContract,
    classify_destination,
)

contract = DestinationClassifierContract(
    burn_time=100.0,
    order_kind="integer",
    coordinate_scale=(1.0, 1.0, 1.0),
    recurrence_theiler_samples=10,
)

decision = classify_destination(
    times,
    states,
    contract=contract,
    equilibria=equilibria,
    references=reference_clouds,
    integration_status="ok",
)
```

`transient` and `ambiguous` are unresolved outcomes for edge tracking. A
specific edge destination uses `equilibrium:<id>` or `reference:<id>` whenever
the supplied references support that distinction.

## Initial edge-bracket refinement

`hidden_attractors.verification.edge_tracking` implements the guarded initial
bracket-refinement layer of edge tracking, not an unrestricted bisection and
not yet the complete time-dependent edge-trajectory algorithm:

1. classify both endpoints independently at B1 and B2;
2. require one stable terminal label at each endpoint and require the labels
   to differ;
3. evaluate the midpoint at the tracking budget;
4. if it is ambiguous, inspect two quarter points;
5. stop on evaluator failure, a third destination, multiple transitions, an
   ambiguous limit, maximum iterations, or the declared metric tolerance; and
6. retain every endpoint confirmation, midpoint, probe, label, and bracket
   width.

`ScaledEuclideanGeometry` handles ordinary states. `ScaledCylindricalGeometry`
uses the shortest local phase chart and rejects an antipodal midpoint whose
chart is not unique.

For Caputo experiments, choose one of two semantics:

- `caputo_reset_initial_state`: every coordinate starts a new causal IVP at the
  declared lower terminal;
- `admissible_history_family_parameter`: the coordinate parametrizes a named
  admissible family of complete histories.

The edge module never linearly interpolates raw history arrays as though they
were Markov states.

This first version covers endpoint confirmation and items 1-4 of the six-step
protocol in the master report. It does not yet co-integrate the final pair,
periodically rebracket their evolved states, record returns of a long edge
trajectory, or require three non-collinear brackets. Those operations remain a
separate dynamic runner milestone and must not be inferred from a converged
initial bracket. Consistently, `EdgeRunContext` defaults to EV-TG2 and the
pilot writes EV-TG2 explicitly: initial-data boundary bisection cannot assign
EV-TG4, even when its finite-resolution width converges.

## Campaign artifact contract

`hidden_attractors.workflows.geometric_topological_campaign` creates six files
before any run:

- `campaign_manifest.json`;
- `seed_bank.csv`;
- `trajectory_metrics.csv`;
- `edge_brackets.csv`;
- `outer_enclosures.json`; and
- `evidence_decisions.json`.

Initialization writes schemas and explicit pending statuses. `outer_enclosures`
remains pending until an actual outer-approximation or isolating-block method is
run; no placeholder is interpreted as a Conley computation.

```python
from hidden_attractors.workflows.geometric_topological_campaign import (
    CampaignManifest,
    initialize_campaign_artifacts,
)

paths = initialize_campaign_artifacts(
    CampaignManifest(
        campaign_id="tg_pilot_001",
        cases=("pll_lead_lag", "mavpd_xi_3p1"),
    ),
    root="validation/06_geometric_topological_campaign/pilot_001",
)
```

## Finite pilot evidence

The recorded pilot exercises seed-bank, trajectory, initial-bracket, solver
comparison, and symmetry-covariance contracts for three cases. Completion of a
declared orchestration does not mean that global hiddenness or a topological
certificate was established.

| Case | Solvers and memory contract | Finite evidence | Bounded result |
| --- | --- | --- | --- |
| integer lead-lag PLL, (q=1) | DOP853 and fixed-step RK4 | EV-TG3 for one route; six route records total | `pll_pp_plus` reached `equilibrium:E_focus` with both solvers at B0, B1, and B2 and passed the B1/B2 destination and observable gates. The other five routes remained EV-TG2; `pll_pp_minus` was `transient:unsettled` through B2. |
| integer MAVPD, ξ=3.1 | DOP853 and fixed-step RK4 | EV-TG2 | PP and validation-only FDF routes reproduced the two inner reference destinations and the outer cycle where tested. The PP routes kept their B1/B2 destination, but failed the declared horizon-observable gate, so no EV-TG3 promotion occurred. |
| Caputo Wu arctangent Chua, (q=0.99) | native full-memory ABM PECE and EFORK3, reset at the declared lower terminal | EV-TG2 | Finite solver-comparison evidence only, not hiddenness evidence. All FPP-A (+) and (-) runs used causal full memory and remained `transient:unsettled` at their tested B0/B2 horizons. Other bank routes produced finite recurrent and projected near-periodic labels, not exact periodic solutions of the Caputo IVP. |

The aggregate case level is deliberately the maximum attained by any seed
route, not an all-seeds certificate. Thus an EV-TG3 route label records only the
declared two-solver finite gate for that route; it is not a statement about all
seeds or the global basin.

Central-inversion covariance is checked from paired numerical trajectories,
not inferred from algebra alone. Initial brackets may legitimately return an
unresolved endpoint or a third destination. The former rejects an inadmissible
binary bracket; the latter records a basin intersection that binary bisection
must not silently cross. Both remain finite initial-data evidence.

## Wolfram algebra audit

The independent repository case
`validation/wolfram/cases/geometric_topological_engine.wl` covers
Picard--Caputo coefficients, coordinate-change identities, affine equivariance,
critical surfaces, PP/FPP-A representatives, connecting minors, and the
declared PLL KCC fixture. These checks establish algebraic consistency only.

## Current implementation boundary

The public implementation stops at guarded initial-bracket refinement. It does
not provide a time-dependent EV-TG4 edge trajectory, admissible-family Caputo
history survey, interval outer map, validated isolating block, homology, or
Conley-index certification. Equilibrium-neighborhood and basin-connectivity
evidence also remain separate requirements for any finite hiddenness statement;
none is inferred from a chaos diagnostic or floating-point bracket.
