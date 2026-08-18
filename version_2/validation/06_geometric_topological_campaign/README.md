# Geometric-Topological Campaign Evidence

This directory is the initialized evidence surface for the TG0-TG8 campaign.
It was created on 2026-08-11 by the repository workflow, before any new dynamic
run.

## Current status

- Software primitives: implemented and covered by focused tests.
- Wolfram algebra audit: 50/50 checks passed.
- New trajectory integrations: not executed.
- Initial edge brackets: not executed.
- Time-dependent edge trajectories and rebracketing: not implemented.
- Outer enclosures, isolating blocks, homology, and Conley indices: not executed.
- New hiddenness or chaos decisions: none.

Header-only CSV files and empty `records` arrays are intentional pending states.
They are not negative results.

## Files

| File | Role |
| --- | --- |
| `campaign_manifest.json` | Frozen cases, B0-B2 budgets, protocol, and claim boundary. |
| `seed_bank.csv` | Unified seed records after provenance-preserving deduplication. |
| `trajectory_metrics.csv` | Finite-time classifier inputs and outcomes. |
| `edge_brackets.csv` | Endpoint confirmations and initial-data bracket refinements. |
| `outer_enclosures.json` | Reserved TG7 surface; explicitly pending. |
| `evidence_decisions.json` | Promotion, retain, reformulate, or stop decisions. |
| `wolfram/` | Algebraic validation summary and text evidence. |

The current edge implementation covers only the initial-data boundary
bisection (steps 1-4 of the master-report protocol). It does not yet integrate
the narrow pair forward, rebracket evolved states, collect edge-trajectory
returns, or establish reproducibility from three non-collinear brackets.

## Regeneration

From the `version_2` directory, initialize a new campaign in a new run folder
with `CampaignManifest` and `initialize_campaign_artifacts`. Existing files are
protected unless replacement is explicitly requested.

Run the independent algebra audit with:

```text
python validation/python/run_wolfram_validations.py \
  --case validation/wolfram/cases/geometric_topological_engine.wl \
  --out validation/outputs/wolfram
```

The Wolfram PASS result certifies only the identities and finite residuals
listed in the summary. It does not certify an attractor, a basin boundary,
chaos, hiddenness, Wada structure, or a Conley index.
