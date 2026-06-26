# Contributing / Contribución

## Table of Contents / Índice de Contenidos

- [English Version](#english-version)
- [Versión en Español](#versión-en-español)

---

## English Version

### Contributing

## Development Setup

```bash
python -m pip install -e ".[dev]"
python -m compileall hidden_attractors examples tests tools/cli
python -m pytest -q
```

## Adding Library Code

- Put reusable model code in `hidden_attractors/models/`.
- Put reusable diagnostics in `hidden_attractors/analysis/`.
- Put reusable workflow orchestration in `hidden_attractors/workflows/`.
- Put C backend sources in `hidden_attractors/native/csrc/`.
- Keep command wrappers thin and place them in `tools/cli/`.

## Adding Examples

Examples belong in `examples/`. They should be short and runnable from the repository root.

## Migrating Legacy Scripts

When a legacy script becomes important:

1. extract reusable functions into `hidden_attractors/`;
1. add tests or a smoke example;
1. leave a thin wrapper in `tools/cli/` if the command name matters;
1. document the numerical contract and outputs.

## Scientific Standards

Do not report hiddenness without the equilibrium-neighborhood and basin evidence required by the workflow. Always report the numerical contract.

---

## Versión en Español

### Contribución

## Configuración de Desarrollo

```bash
python -m pip install -e ".[dev]"
python -m compileall hidden_attractors examples tests tools/cli
python -m pytest -q
```

## Adición de Código de la Biblioteca

- Colocar el código de modelo reutilizable en `hidden_attractors/models/`.
- Colocar los diagnósticos reutilizables en `hidden_attractors/analysis/`.
- Colocar la orquestación del flujo de trabajo reutilizable en `hidden_attractors/workflows/`.
- Colocar las fuentes del backend de C en `hidden_attractors/native/csrc/`.
- Mantener los comandos envoltorios delgados y colocarlos en `tools/cli/`.

## Adición de Ejemplos

Los ejemplos pertenecen a `examples/`. Deben ser cortos y ejecutables desde la raíz del repositorio.

## Migración de Scripts Heredados (Legacy)

Cuando un script heredado se vuelve importante:

1. extraer funciones reutilizables en `hidden_attractors/`;
1. añadir pruebas o un ejemplo de humo;
1. dejar un envoltorio delgado en `tools/cli/` si el nombre del comando importa;
1. documentar el contrato numérico y las salidas.

## Estándares Científicos

No informe de ocultedad sin la evidencia de vecindad de equilibrios y cuencas requerida por el flujo de trabajo. Informe siempre del contrato numérico.
