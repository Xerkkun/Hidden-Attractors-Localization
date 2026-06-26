# Unified Protocol Pipeline Guide / Guía del Pipeline de Protocolo Unificado

## Table of Contents / Índice de Contenidos

- [English Version](#english-version)
- [Versión en Español](#versión-en-español)

---

## English Version

### Hidden Attractors Fractional Order - version_2 (EN)

**Last updated:** Complete migration to the canonical unified Caputo protocol.  
All search, continuation, and validation of hidden attractors follows a strict order of 10 stages.

### 1. Main Rule

The library is used from `hidden_attractors/`. Obsolete or "legacy" scripts must not be used or extended.

**Operational calculation rule:**

- **Lightweight preparation, algebraic seeds, state reading, and metrics reporting:** Python API.
- **Large sweep integration, basins, hiddenness tests, and Lyapunov exponents:** C backends (`FractionalChuaBackend`, `BasinBackend`).
- **No seed is rejected** for being periodic prior to continuation (`pre_continuation_periodic`).

### 2. Canonical Unified Command (`hidden-attractors protocol`)

The only promoted entrypoint for validation is the unified subcommand `hidden-attractors protocol`, which exposes the mandatory subcommands corresponding to the official stage order. Legacy wrappers are not public release entry points:

```bash
hidden-attractors protocol generate-seeds --help
hidden-attractors protocol soft-precheck --help
hidden-attractors protocol continue --help
hidden-attractors protocol filter-survivors --help
hidden-attractors protocol build-reference --help
hidden-attractors protocol robustness --help
hidden-attractors protocol hiddenness --help
hidden-attractors protocol diagnostics --help
```

Each command generates or processes a machine-readable `summary.json` with the same common metadata structure (`schema_version`, `protocol_version`, `stage`, etc.).

### 3. Complete Protocol Sequence

Example of sequential execution using a contract (e.g., `configs/unified_caputo_protocol.json`):

#### a) Seed Search and Generation

```bash
hidden-attractors protocol generate-seeds --contract configs/unified_caputo_protocol.json --output outputs/run/03_seed_generation/summary.json
```

#### b) Soft Precheck (Diagnostic, does not reject for periodicity)

```bash
hidden-attractors protocol soft-precheck --contract configs/unified_caputo_protocol.json --payload outputs/run/03_seed_generation/summary.json --output outputs/run/04_soft_precheck/summary.json
```

#### c) Homotopy Continuation by Parameter $\lambda$

```bash
hidden-attractors protocol continue --contract configs/unified_caputo_protocol.json --candidate-id candidate_001 --lambda-values 0,0.25,0.5,0.75,1 --output outputs/run/05_continuation/summary.json
```

#### d) Post-Continuation Filtering (Periodic orbits discarded here)

```bash
hidden-attractors protocol filter-survivors --contract configs/unified_caputo_protocol.json --payload outputs/run/05_continuation/summary.json --output outputs/run/06_post_continuation_filter/summary.json
```

#### e) Dynamic Reference Creation

```bash
hidden-attractors protocol build-reference --contract configs/unified_caputo_protocol.json --payload outputs/run/06_post_continuation_filter/summary.json --output outputs/run/07_dynamic_reference/summary.json
```

#### f) Robustness Tests (Step $h$ changes, $L_m$, full vs truncated memory)

```bash
hidden-attractors protocol robustness --contract configs/unified_caputo_protocol.json --payload outputs/run/07_dynamic_reference/summary.json --output outputs/run/08_robustness/summary.json
```

#### g) Hiddenness Tests (Inner ball sampling and basin slices)

```bash
hidden-attractors protocol hiddenness --contract configs/unified_caputo_protocol.json --payload outputs/run/08_robustness/summary.json --output outputs/run/09_hiddenness_tests/summary.json
```

#### h) Complementary Diagnostics (FFT, PSD, Lyapunov)

```bash
hidden-attractors protocol diagnostics --contract configs/unified_caputo_protocol.json --payload outputs/run/09_hiddenness_tests/summary.json --output outputs/run/10_diagnostics/summary.json
```

### 4. Migration State and Cleanup

Obsolete historical paths have been removed or unified:

- All old standalone entrypoints like `hidden-attractors-unified-chua` and `hidden-attractors-machado-targeted` were removed.
- Directed Machado sweep configurations under `configs/machado_targeted_verification.yaml` were retired. All configurations now route through `configs/unified_caputo_protocol.json`.
- Danca ABM replicas remain under `tools/legacy/danca2017_chua_abm_replication.py` only for historical benchmark and comparison validation.
- Pre-continuation periodicity is no longer a cause for rejection, guaranteeing search space preservation for Lur'e transport.

---

## Versión en Español

### Hidden Attractors Fractional Order - version_2 (ES)

**Última actualización:** Migración completa al protocolo canónico unificado de Caputo.  
Toda la búsqueda, continuación y validación de atractores ocultos sigue un orden estricto de 10 etapas.

### 1. Regla Principal

La librería se usa desde `hidden_attractors/`. Los scripts obsoletos o "legacy" no deben usarse ni ampliarse.

**Regla operativa de cálculo:**

- **Preparación ligera, semillas algebraicas, lectura de estados y reporte de métricas:** Python API.
- **Integración de sweeps grandes, cuencas, pruebas de ocultedad y exponentes de Lyapunov:** C backends (`FractionalChuaBackend`, `BasinBackend`).
- **No se rechaza ninguna semilla** por ser periódica antes de la continuación (`pre_continuation_periodic`).

### 2. Comando Canónico Unificado (`hidden-attractors protocol`)

El único entrypoint promocionado para la validación es `hidden-attractors protocol`, el cual expone 8 subcomandos obligatorios correspondientes al orden de etapas oficiales:

```bash
hidden-attractors protocol generate-seeds --help
hidden-attractors protocol soft-precheck --help
hidden-attractors protocol continue --help
hidden-attractors protocol filter-survivors --help
hidden-attractors protocol build-reference --help
hidden-attractors protocol robustness --help
hidden-attractors protocol hiddenness --help
hidden-attractors protocol diagnostics --help
```

Cada comando genera o procesa un archivo `summary.json` compatible con lectura automática (machine-readable) que cuenta con la misma estructura común de metadatos (`schema_version`, `protocol_version`, `stage`, etc.).

### 3. Secuencia Completa del Protocolo

Ejemplo de ejecución secuencial usando un contrato (ej: `configs/unified_caputo_protocol.json`):

#### a) Búsqueda y Generación de Semillas

```bash
hidden-attractors protocol generate-seeds --contract configs/unified_caputo_protocol.json --output outputs/run/03_seed_generation/summary.json
```

#### b) Precheck Suave (Diagnóstico, no rechaza por periodicidad)

```bash
hidden-attractors protocol soft-precheck --contract configs/unified_caputo_protocol.json --payload outputs/run/03_seed_generation/summary.json --output outputs/run/04_soft_precheck/summary.json
```

#### c) Continuación por Parámetro de Homotopía $\lambda$

```bash
hidden-attractors protocol continue --contract configs/unified_caputo_protocol.json --candidate-id candidate_001 --lambda-values 0,0.25,0.5,0.75,1 --output outputs/run/05_continuation/summary.json
```

#### d) Filtrado Post-Continuación (Aquí se descartan órbitas periódicas)

```bash
hidden-attractors protocol filter-survivors --contract configs/unified_caputo_protocol.json --payload outputs/run/05_continuation/summary.json --output outputs/run/06_post_continuation_filter/summary.json
```

#### e) Creación de Referencia Dinámica

```bash
hidden-attractors protocol build-reference --contract configs/unified_caputo_protocol.json --payload outputs/run/06_post_continuation_filter/summary.json --output outputs/run/07_dynamic_reference/summary.json
```

#### f) Pruebas de Robustez (Cambios de paso $h$, $L_m$, memoria completa vs. truncada)

```bash
hidden-attractors protocol robustness --contract configs/unified_caputo_protocol.json --payload outputs/run/07_dynamic_reference/summary.json --output outputs/run/08_robustness/summary.json
```

#### g) Pruebas de Ocultedad (Muestreo interior en bolas y cortes de cuenca)

```bash
hidden-attractors protocol hiddenness --contract configs/unified_caputo_protocol.json --payload outputs/run/08_robustness/summary.json --output outputs/run/09_hiddenness_tests/summary.json
```

#### h) Diagnósticos Complementarios (FFT, PSD, Lyapunov)

```bash
hidden-attractors protocol diagnostics --contract configs/unified_caputo_protocol.json --payload outputs/run/09_hiddenness_tests/summary.json --output outputs/run/10_diagnostics/summary.json
```

### 4. Estado de Migración y Limpieza

Las rutas históricas obsoletas han sido eliminadas o unificadas:

- Se eliminaron todos los entrypoints antiguos como `hidden-attractors-unified-chua` y `hidden-attractors-machado-targeted`.
- Las configuraciones de barrido Machado dirigidas bajo `configs/machado_targeted_verification.yaml` se retiraron. Toda la configuración pasa ahora por `configs/unified_caputo_protocol.json`.
- Las réplicas de Danca ABM permanecen bajo `tools/legacy/danca2017_chua_abm_replication.py` únicamente para validación histórica de benchmark y comparación.
- La periodicidad antes de la continuación ya no es causa de descarte, garantizando la preservación del espacio de búsqueda para transporte mediante Lur'e.
