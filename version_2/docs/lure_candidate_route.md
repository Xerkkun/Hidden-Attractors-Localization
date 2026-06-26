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
  "family": "lure_classical_centered | lure_classical_biased",
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

*Nota sobre Machado/FDF*: Las familias `machado_centered` y `machado_biased` se conservan como soporte teorico/interno planificado. No forman parte de la superficie publica del CLI de esta entrega; `hidden-attractors seed --help` solo lista `lure-centered` y `lure-biased`.

`lure_classical_centered` es la antigua ruta clásica centrada. Las funciones descriptivas predicen candidatos armónicos en el Chua buscando puntos de balance de frecuencia ($W_q(j\omega)N(A) = -1$).
*Advertencia Científica*: La función descriptiva es una herramienta de aproximación armónica de primer armónico para ubicar posibles semillas localizadas. **No constituye de ninguna manera una prueba de existencia de atractor ni de su ocultedad**. La ocultedad se determina posteriormente y de forma rigurosa en el protocolo evaluando vecindades locales de todos los equilibrios (Stage 52 `hiddenness_tests`).

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
