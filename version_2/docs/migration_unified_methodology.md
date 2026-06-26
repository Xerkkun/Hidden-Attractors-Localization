# Migration To The Unified Methodology

## Table of Contents / Índice de Contenidos

- [English Version](#english-version)
- [Versión en Español](#versión-en-español)

---

## English Version

### Migration Overview

`version_2` now has one official Caputo hidden-attractor protocol. Old route
names must not appear in new promoted outputs.

## Renamed Concepts

| Previous wording | Official representation |
| --- | --- |
| classical centered describing function | `seed_generation.family=lure_classical_centered` |
| centered Lur'e route | `seed_generation.family=lure_classical_centered` |
| biased Lur'e route | `seed_generation.family=lure_classical_biased` |
| Machado/FDF route | `machado_centered` or `machado_biased` (Marked as planned, unsupported in this release) |
| epsilon or eta continuation | `ContinuationPlan(lambda_values=...)` with internal mapping in provenance |
| early periodicity filter | `soft_precheck`; periodic seeds use `pre_continuation_periodic` and continue |
| sphere-only controls | interior ball sampling in `hiddenness_tests` |

1. Describing functions and reconstructed Lur'e points are seed sources only. They are not hiddenness evidence.

## Removed And Retained Material

`version_1/` is removed from the maintained repository after its relevant
integer reference artifacts were promoted to
`validation/reference_cases/chua_integer_q1/`.

`tools/legacy/` is temporarily retained for a narrow reason: current
maintained wrappers still import the full fractional Chua driver and Danca ABM
adapter, and existing transition tests exercise the precheck/continuation
adapter. It is no longer a documented route for new results. Its remaining
dependencies must be ported into `hidden_attractors/` before that directory
can be deleted safely.

Older generated validation stage trees under former directory names were
removed. The surviving `validation/reference_cases/` subtree is retained
only for independently motivated benchmark evidence. New promotions conform
to `configs/validation_contract.json` version 2 and use the official stage
envelope.

## EFORK Correction Invariant

Every executable EFORK implementation still reachable from `version_2`, in
Python or C and including retained compatibility engines, is checked against
the published third-stage ordering:

```text
K3 = F(... + a31*K1 + a32*K2)
```

`tests/test_efork_published_validation.py` tests the Python implementations,
the package-native C backends, and the C/Python engines temporarily retained
under `tools/legacy/`. No superseded executable EFORK tree is kept as a second
workflow surface.

---

## Versión en Español

`version_2` ahora tiene un único protocolo oficial de atractores ocultos de Caputo. Los nombres de las rutas antiguas no deben aparecer en las nuevas salidas promovidas.

## Conceptos Renombrados

| Terminología Anterior | Representación Oficial |
| --- | --- |
| classical centered describing function | `seed_generation.family=lure_classical_centered` |
| centered Lur'e route | `seed_generation.family=lure_classical_centered` |
| biased Lur'e route | `seed_generation.family=lure_classical_biased` |
| Machado/FDF route | `machado_centered` o `machado_biased` (Marcado como planificado, no soportado en este release) |
| epsilon o eta continuation | `ContinuationPlan(lambda_values=...)` con mapeo interno en procedencia |
| early periodicity filter | `soft_precheck`; las semillas periódicas usan `pre_continuation_periodic` y continúan |
| sphere-only controls | muestreo de bolas interiores en `hiddenness_tests` |

1. Las funciones descriptivas y los puntos Lur'e reconstruidos son únicamente fuentes de semillas. No constituyen evidencia de ocultedad.

## Material Eliminado y Retenido

`version_1/` se elimina del repositorio mantenido después de que sus artefactos de referencia enteros relevantes fueran promovidos a `validation/reference_cases/chua_integer_q1/`.

`tools/legacy/` se retiene temporalmente por una razón estrecha: los wrappers mantenidos actuales aún importan el driver completo de Chua fraccionario y el adaptador Danca ABM, y las pruebas de transición existentes ejercitan el adaptador precheck/continuation. Ya no es una ruta documentada para nuevos resultados. Sus dependencias restantes deben ser portadas a `hidden_attractors/` antes de que ese directorio pueda ser eliminado de manera segura.

Los árboles de etapas de validación generados anteriormente bajo nombres de directorio antiguos fueron eliminados. El subárbol subyacente `validation/reference_cases/` se retiene solo para evidencia de benchmark motivada independientemente. Las nuevas promociones cumplen con `configs/validation_contract.json` versión 2 y usan el envelope de etapa oficial.

## Invariante de Corrección EFORK

Cada implementación ejecutable de EFORK que aún es accesible desde `version_2`, en Python o C e incluyendo motores de compatibilidad retenidos, se verifica contra el orden de tercera etapa publicado:

```text
K3 = F(... + a31*K1 + a32*K2)
```

`tests/test_efork_published_validation.py` prueba las implementaciones de Python, los backends C nativos del paquete y los motores C/Python retenidos temporalmente bajo `tools/legacy/`. No se mantiene ningún árbol ejecutable de EFORK superado como segunda superficie de flujo de trabajo.

*Nota Científica e Invariante de Caputo*:
2. En sistemas de Caputo, la continuación numérica debe propagar la memoria e historia. Las continuaciones que utilicen únicamente el último estado (warm-start numérico) no se consideran continuación Caputo estricta y emitirán una advertencia.
