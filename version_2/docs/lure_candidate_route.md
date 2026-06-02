# Seed Generation Families

This page replaces the former separate Lur'e and Machado route description.
The official methodology has a single `seed_generation` stage configured in:

```text
configs/unified_caputo_protocol.json
```

## Uniform Seed Record

Every reconstructed seed writes:

```json
{
  "family": "lure_classical_centered | lure_classical_biased | machado_centered | machado_biased",
  "centered_or_biased": "centered | biased",
  "A": 0.0,
  "sigma0": 0.0,
  "omega": 0.0,
  "mu": 1.0,
  "theta": 0.0,
  "q": 0.9998,
  "harmonic_residual": 0.0,
  "rho_H": 0.0,
  "x0": [0.0, 0.0, 0.0],
  "reconstruction_metadata": {},
  "source_config": "configs/unified_caputo_protocol.json"
}
```

`lure_classical_centered` is the former centered classical DF or centered
Lur'e construction. It is one conceptual route, not two. Describing functions predict harmonic seed candidates in the Chua circuit by searching for frequency balance points [ref:kuznetsov_2017_chua_df]. `machado_centered`
and `machado_biased` extend the searchable seed family through `mu` and
`theta` [ref:machado_2015_fractional_describing_functions]; they do not prove that the resulting target is hidden.

## Soft Precheck

Direct integration before continuation is diagnostic. Its possible useful
labels include:

```text
pre_continuation_periodic
pre_continuation_quasiperiodic
pre_continuation_chaotic_looking
pre_continuation_equilibrium_collapse
```

A seed with `pre_continuation_periodic` is still admitted to
`ContinuationPlan(lambda_values=...)`. Only invalid parameters, NaN/Inf,
catastrophic numerical failure or exact duplication may reject a seed here.

## Decision Boundary

Hard exclusion of periodic or collapsed target dynamics occurs only in
`post_continuation_filter`, after the path reaches `lambda=1` and is
integrated on the original target system. Hiddenness is assessed later still:
only robust survivors receive ball-neighborhood and basin-slice tests around
all equilibria.

## Historical Reproduction

The files `configs/lure_biased_multiparam_q09998.yaml` and
`configs/machado_candidate_route.yaml` remain only to interpret or reproduce
older artifacts while their adapters are being removed. New results must use
the unified configuration and official JSON envelopes.
