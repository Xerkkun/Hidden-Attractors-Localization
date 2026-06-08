# Phase E: Published Continuation Comparison

## Overview

This directory contains configurations and outputs for **Phase E** of the
validation pipeline: `validation/published_continuation_comparison`.

This phase compares the continuation or paper-style strategy of the library
against what the published articles actually report.

---

## What this phase does NOT do

- Does **not** verify hiddenness (`hidden_verified` is never set).
- Does **not** certify chaos (`chaos_verified` is never set).
- Does **not** invent data that is not in the articles (omega0, k, a0, seed,
  attractor ranges, or continuation paths are never fabricated).

---

## Distinction from other validation phases

| Phase | Purpose |
|---|---|
| `published_cases` | Stores bibliographic metadata and known published values per article. |
| `fractional_memory_validation` | Validates full vs. windowed Caputo memory sensitivity for a given IC. |
| `continuation_memory_validation` | Validates eta-continuation (deformed Lure path) with history transport. |
| `integrator_method_validation` | Cross-validates integrator methods (ABM, EFORK, RK4) against each other. |
| `integrator_crosscheck` | Checks integrator consistency across step sizes and configurations. |
| **`published_continuation_comparison`** | Compares paper-style integration strategy vs. library history-transport strategy, using only data actually reported by the article. |

---

## What "paper-style" means

If a published article does not specify how memory is transported in a
numerical continuation, the **paper-style** strategy is:

```
last_point_restart
```

i.e., only the final state X(t_final) is carried to the next segment.
No discrete history is transported.

This reproduces the *inferred* practice of the article when no explicit
memory-transport protocol is reported.

---

## What "Caputo-aware" means

The **Caputo-aware** strategy is:

```
history_window_transport
```

The discrete history window `H_k = {X(t_{k-M}), ..., X(t_k)}` is transported
between segments, and the RHS is recomputed under the new field. This respects
the non-local character of the Caputo derivative across segment boundaries.

---

## What happens when the article does not report continuation

If `paper.reports_continuation == false`, the pipeline:

1. Does **not** write `paper_continuation_reproduced`.
2. Writes instead:
   ```
   paper_does_not_report_continuation
   paper_style_comparison_performed
   ```
3. Performs a comparison between paper-style strategy (inferred from
   published data) and Caputo-aware strategy (if applicable).

---

## What happens when data is missing

If published data (omega0, k, a0, seed, attractor ranges, initial conditions)
is not reported in the article, those fields are marked `null` or `missing`
in the YAML and in the output summary:

```
seed_source: "missing"
```

The pipeline will **not** invent values to fill missing data.

---

## Case descriptions

### 1. Kuznetsov et al. 2017

- **System**: Chua integer-order with saturation nonlinearity.
- **q = 1.0** — no Caputo memory, no history transport.
- **Available data**: IC from paper `[5.8576, 0.3694, -8.3686]`, DF seed,
  omega0, k, a0, parameters.
- **Strategy**: `paper_style_initial_condition_integration` from published IC
  and seed. No deformed Lure continuation (not reported by paper). No
  history transport (q=1 requires none).
- **Status field**: `paper_does_not_report_continuation` (article uses DF
  method for seed location, not numerical continuation).

### 2. Danca 2017

- **System**: Chua fractional saturation (non-smooth), q=0.9998, ABM.
- **Available data**: parameters, q, h, integrator. All DF seed and IC data
  are missing from the article.
- **Strategy**: No paper-style integration possible (missing IC/seed). The
  dynamic contract is still explicit: `integrator=ABM`, backend
  `python_abm_full_history`, `memory_mode=full`,
  `memory_policy=full_history`, `caputo_history_accumulated=true`. All
  comparison modes are disabled. Outputs `published_data_missing`.
- **Status field**: `paper_does_not_report_continuation`, `published_data_missing`.

#### Candidate-selection traceability derived from Danca 2017

The nearby candidate used by the current report is not stored as a top-level
`published_continuation_comparison/*.yaml` case, because the runner executes all
top-level YAML files as published article comparisons. Its traceability record is
kept separately at:

```
validation/candidate_selection/danca2017_nearby_saturation_candidate.yaml
```

That record preserves the exact Danca parameters as the published reference and
documents the derived sweep candidate
`m1_m1p2000_m0_m0p2000_branch_0`, with `(m0,m1)=(-0.2,-1.2)`. It also records
the proposed fractional DF plus ABM-Caputo continuation route and the integer
DF plus integer-continuation control route used only because the article does
not specify a continuation algorithm. It must not be read as a closed
reproduction of the Danca paper, and it makes no hiddenness or Lyapunov-based
chaos claim.

### 3. Wu et al. 2023

- **System**: Chua fractional arctan, q=0.99.
- **Available data**: parameters, q, published ICs `x0_plus`, `x0_minus`.
  DF parameters (k, a0, omega0) not reported.
- **Strategy**: `paper_style_initial_condition_integration` from published ICs
  using `ADM_WU2023`, backend `adm_local_reproduction`, `adm_order=4`,
  `memory_mode=none`, `memory_policy=none_local_adm` and
  `caputo_history_accumulated=false`. No deformed Lure continuation is run
  because `k` is null and is not invented. The restart-vs-history comparison
  is disabled for this published Wu reproduction because it would compare ABM
  memory policies that the ADM reproduction does not use.
- **Status field**: `paper_does_not_report_continuation`,
  `continuation_auxiliary_unavailable`,
  `published_initial_condition_reintegrated`.

---

## Allowed `overall_status` values

| Status | Condition |
|---|---|
| `published_continuation_reproduced` | `paper.reports_continuation == true` AND the reported path was reproduced. |
| `published_initial_condition_reintegrated` | Paper reports IC, integration gives bounded non-trivial dynamics. |
| `published_seed_reintegrated` | Paper reports seed, integration gives bounded non-trivial dynamics. |
| `published_paper_style_comparison_performed` | Paper does not report continuation; paper-style vs. history comparison was run. |
| `published_comparison_partial_original_only` | k=null, deformed Lure unavailable; original system comparison only. |
| `published_comparison_inconclusive` | Cannot conclude due to diverged/collapsed/NaN trajectories. |
| `published_data_missing` | No usable published IC or seed available. |
| `published_continuation_not_reported` | Paper does not report continuation, no data available to run comparison. |

> [!IMPORTANT]
> `published_continuation_reproduced` is only valid when
> `paper.reports_continuation == true`. It must never appear for
> Danca 2017 or Wu 2023.

---

## No claims

```
hiddenness_certified_by_this_pipeline: false
chaos_certified_by_this_pipeline: false
no_hidden_verified_claim: true
```

`chaotic_dynamics_candidate_detected: true` means the trajectory is classified
as `chaotic_candidate_by_geometry` by the classifier heuristics. It does **not**
mean chaos is proven.
