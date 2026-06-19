# Testing / Pruebas

## Table of Contents / Índice de Contenidos
- [English Version](#english-version)
- [Versión en Español](#versión-en-español)

---

## English Version

# Testing

## Smoke Checks

Quick commands from `version_2/`:

```bash
python -m compileall hidden_attractors examples tests tools/cli
python -m hidden_attractors.cli.main --help
python -m hidden_attractors.cli.main inspect --help
python -m hidden_attractors.cli.main validate --help
python -m hidden_attractors.cli.main seed --help
python -m hidden_attractors.cli.main continuation --help
```

If the package is installed in editable mode:

```bash
hidden-attractors --help
hidden-attractors inspect systems
hidden-attractors inspect candidates
hidden-attractors validate contract --allow-pending
```

---

## Pytest

```bash
python -m pip install -e ".[dev,analysis,legacy]"
python -m pytest -q
```

### Hygiene and CPC Readiness Tests

The project keeps a small hygiene/readiness test suite because numerical tests do not protect repository publication boundaries. These tests guard against retracking local outputs, local manuscripts, absolute paths, legacy CLI entry points, unpromoted validation outputs, and overclaimed CPC metadata.

To run these tests specifically:

```bash
python -m pytest -q -m "hygiene"
python -m pytest -q -m "cpc_readiness"
python -m pytest -q -m "not hygiene and not cpc_readiness"
```

At the current thesis-freeze audit, `validation/freeze_audit/` reports 797 passed and 34 skipped.

Current tests verify:

- Chua equilibria are zeros of the vector field;
- final candidate loading returns the expected reference records;
- the official ten-stage order and uniform JSON envelope;
- periodic pre-continuation seeds remain eligible for continuation;
- hiddenness cannot receive its strongest label without ball sampling, robust reproduction, all equilibria, and all basin planes;
- every executable Python and C EFORK source uses `K3 = a31*K1 + a32*K2`.

---

## Native Backends

Native backend tests should be added cautiously. Keep them short, write to a temporary output directory, and record whether OpenMP was active.

---

## Validation Evidence

Tests answer whether the package still behaves as expected. Validation evidence answers whether a scientific claim is backed by traceable numerical artifacts.

Use `outputs/` for ordinary generated run products. Promote only selected evidence into `validation/`, following `configs/validation_contract.json`.

Each validation stage should include:

- one short `*_validation.md` interpretation;
- one `*_validation_summary.json` or equivalent summary JSON;
- CSV tables for numerical checks;
- PNG/PDF figures for visual evidence when relevant.

The final report should cite the stage summaries and selected artifacts instead of embedding all raw data.

Run the contract checker from `version_2/` after evidence has been promoted:

```bash
hidden-attractors validate contract
```

And, when pending stages are allowed:

```bash
hidden-attractors validate contract --allow-pending
```

If alternative root is needed:

```bash
hidden-attractors validate contract --validation-root path/to/validation
```

### Legacy commands no longer installed

These names may appear in old notes only; use the grouped `hidden-attractors` CLI:
- `hidden-attractors-check-validation` (deprecated; use `hidden-attractors validate contract` instead)
- `hidden-attractors-protocol` (deprecated; use `hidden-attractors protocol <stage>` instead)

---

## Versión en Español

# Pruebas

## Chequeos de Humo

Comandos rápidos desde `version_2/`:

```bash
python -m compileall hidden_attractors examples tests tools/cli
python -m hidden_attractors.cli.main --help
python -m hidden_attractors.cli.main inspect --help
python -m hidden_attractors.cli.main validate --help
python -m hidden_attractors.cli.main seed --help
python -m hidden_attractors.cli.main continuation --help
```

Si el paquete está instalado en modo editable:

```bash
hidden-attractors --help
hidden-attractors inspect systems
hidden-attractors inspect candidates
hidden-attractors validate contract --allow-pending
```

---

## Pytest

```bash
python -m pip install -e ".[dev,analysis,legacy]"
python -m pytest -q
```

### Pruebas de Higiene y Preparación para el CPC

El proyecto mantiene un pequeño conjunto de pruebas de higiene/preparación porque las pruebas numéricas no protegen los límites de publicación del repositorio. Estas pruebas protegen contra el rastreo de salidas locales, manuscritos locales, rutas absolutas, puntos de entrada de CLI heredados, salidas de validación no promocionadas y metadatos de CPC exagerados.

Para ejecutar estas pruebas específicamente:

```bash
python -m pytest -q -m "hygiene"
python -m pytest -q -m "cpc_readiness"
python -m pytest -q -m "not hygiene and not cpc_readiness"
```

En la auditoría de congelación de tesis actual, `validation/freeze_audit/` reporta 797 aprobadas y 34 omitidas.

Las pruebas actuales verifican:

- los equilibrios de Chua son ceros del campo vectorial;
- la carga final de candidatos devuelve los registros de referencia esperados;
- el orden oficial de diez etapas y el sobre (envelope) JSON uniforme;
- las semillas periódicas previas a la continuación siguen siendo elegibles para la continuación;
- la ocultedad no puede recibir su etiqueta más fuerte sin muestreo de bolas, reproducción robusta, todos los equilibrios y todos los planos de la cuenca;
- cada fuente ejecutable de Python y C EFORK utiliza `K3 = a31*K1 + a32*K2`.

---

## Backends Nativos

Las pruebas de backend nativas deben agregarse con precaución. Manténgalas cortas, escriba en un directorio de salida temporal y registre si OpenMP estaba activo.

---

## Evidencia de Validación

Las pruebas responden si el paquete todavía se comporta como se espera. La evidencia de validación responde si un reclamo científico está respaldado por artefactos numéricos trazables.

Use `outputs/` para los productos de ejecución generados ordinariamente. Promueva solo la evidencia seleccionada a `validation/`, siguiendo `configs/validation_contract.json`.

Each validation stage should include:
Cada etapa de validación debe incluir:

- one short `*_validation.md` interpretation;
- una interpretación corta en `*_validation.md`;
- un `*_validation_summary.json` o JSON de resumen equivalente;
- tablas CSV para comprobaciones numéricas;
- figuras PNG/PDF para evidencia visual cuando sea relevante.

El informe final debe citar los resúmenes de etapa y los artefactos seleccionados en lugar de incrustar todos los datos sin procesar.

Ejecute el comprobador de contratos desde `version_2/` después de que la evidencia haya sido promocionada:

```bash
hidden-attractors validate contract
```

Y, cuando se permitan etapas pendientes:

```bash
hidden-attractors validate contract --allow-pending
```

Si se necesita una raíz alternativa:

```bash
hidden-attractors validate contract --validation-root path/to/validation
```

### Comandos Legacy que ya no se Instalan

Estos nombres pueden aparecer únicamente en notas antiguas; use la CLI agrupada de `hidden-attractors`: