# Contributing / Contribución

## Table of Contents / Índice de Contenidos

- [English Version](#english-version)
- [Versión en Español](#versión-en-español)

---

## English Version

### Contributing

## Development Setup

To configure your development environment:

```bash
python -m pip install -e ".[dev,analysis,docs,legacy]"
python -m compileall hidden_attractors examples tests tools/cli
python -m pytest -q
```

## Adding Library Code

- Put reusable model code in `hidden_attractors/models/`.
- Put reusable diagnostics in `hidden_attractors/analysis/`.
- Put reusable workflow orchestration in `hidden_attractors/workflows/`.
- Put C backend sources in `hidden_attractors/native/csrc/`.
- To expose new command-line interfaces:
  1. Implement arguments parsing and dispatch logic in `hidden_attractors/cli/`.
  2. Register the group and subcommand dispatcher in `hidden_attractors/cli/main.py`.

## Adding Examples

Examples belong in `examples/`. They should be short and runnable from the repository root.

## Migrating Legacy Scripts

When a legacy script from `tools/legacy/` becomes important:

1. Extract reusable functions into `hidden_attractors/`;
2. Add tests or a smoke example;
3. Integrate the subcommand inside `hidden_attractors/cli/main.py`;
4. Document the numerical contract and outputs.

## Scientific Standards

Do not report hiddenness without the equilibrium-neighborhood and basin evidence required by the workflow. Always report the numerical contract.

---

## Versión en Español

### Contribución

## Configuración de Desarrollo

Para configurar su entorno de desarrollo:

```bash
python -m pip install -e ".[dev,analysis,docs,legacy]"
python -m compileall hidden_attractors examples tests tools/cli
python -m pytest -q
```

## Adición de Código de la Biblioteca

- Colocar el código de modelo reutilizable en `hidden_attractors/models/`.
- Colocar los diagnósticos reutilizables en `hidden_attractors/analysis/`.
- Colocar la orquestación del flujo de trabajo reutilizable en `hidden_attractors/workflows/`.
- Colocar las fuentes del backend de C en `hidden_attractors/native/csrc/`.
- Para exponer nuevas interfaces de línea de comandos:
  1. Implementar el parser de argumentos y lógica en `hidden_attractors/cli/`.
  2. Registrar el despachador de grupo y subcomando en `hidden_attractors/cli/main.py`.

## Adición de Ejemplos

Los ejemplos pertenecen a `examples/`. Deben ser cortos y ejecutables desde la raíz del repositorio.

## Migración de Scripts Heredados (Legacy)

Cuando un script heredado de `tools/legacy/` se vuelve importante:

1. Extraer funciones reutilizables en `hidden_attractors/`;
2. Añadir pruebas o un ejemplo de humo;
3. Integrar el subcomando dentro del despachador unificado `hidden_attractors/cli/main.py`;
4. Documentar el contrato numérico y las salidas.

## Estándares Científicos

No informe de ocultedad sin la evidencia de vecindad de equilibrios y cuencas requerida por el flujo de trabajo. Informe siempre del contrato numérico.
