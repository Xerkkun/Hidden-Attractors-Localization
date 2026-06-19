# Contributing / Contribución

## Development Setup / Configuración de Desarrollo

```bash
python -m pip install -e ".[dev]"
python -m compileall hidden_attractors examples tests tools/cli
python -m pytest -q
```

## Adding Library Code / Adición de Código de la Biblioteca

- Put reusable model code in `hidden_attractors/models/`.
- Put reusable diagnostics in `hidden_attractors/analysis/`.
- Put reusable workflow orchestration in `hidden_attractors/workflows/`.
- Put C backend sources in `hidden_attractors/native/csrc/`.
- Keep command wrappers thin and place them in `tools/cli/`.
- 
- Colocar el código de modelo reutilizable en `hidden_attractors/models/`.
- Colocar los diagnósticos reutilizables en `hidden_attractors/analysis/`.
- Colocar la orquestación del flujo de trabajo reutilizable en `hidden_attractors/workflows/`.
- Colocar las fuentes del backend de C en `hidden_attractors/native/csrc/`.
- Mantener los comandos envoltorios delgados y colocarlos en `tools/cli/`.

## Adding Examples / Adición de Ejemplos

Examples belong in `examples/`. They should be short and runnable from the repository root.
Los ejemplos pertenecen a `examples/`. Deben ser cortos y ejecutables desde la raíz del repositorio.

## Migrating Legacy Scripts / Migración de Scripts Heredados (Legacy)

When a legacy script becomes important:
Cuando un script heredado se vuelve importante:

1. extract reusable functions into `hidden_attractors/`;
2. add tests or a smoke example;
3. leave a thin wrapper in `tools/cli/` if the command name matters;
4. document the numerical contract and outputs.
5. 
1. extraer funciones reutilizables en `hidden_attractors/`;
2. añadir pruebas o un ejemplo de humo;
3. dejar un envoltorio delgado en `tools/cli/` si el nombre del comando importa;
4. documentar el contrato numérico y las salidas.

## Scientific Standards / Estándares Científicos

Do not report hiddenness without the equilibrium-neighborhood and basin evidence required by the workflow. Always report the numerical contract.
No informe de ocultedad sin la evidencia de vecindad de equilibrios y cuencas requerida por el flujo de trabajo. Informe siempre del contrato numérico.
