# Installation / Instalación

## Editable Install / Instalación Editable

Depending on your current working directory, run one of the following commands:
Dependiendo de su directorio de trabajo actual, ejecute uno de los siguientes comandos:

### A. From the Repository Root Directory / Desde el Directorio Raíz del Repositorio
```bash
python -m pip install -e version_2
```

### B. From the `version_2/` Subdirectory / Desde el Subdirectorio `version_2/`
```bash
python -m pip install -e .
```

---

## Recommended Install for Validation and Development / Instalación Recomendada para Validación y Desarrollo

To run unit tests, validation contracts, and chaos analysis:
Para ejecutar pruebas unitarias, contratos de validación y análisis de caos:

### A. From the Repository Root Directory / Desde el Directorio Raíz del Repositorio
```bash
python -m pip install -e "version_2[dev,analysis,legacy]"
```

### B. From the `version_2/` Subdirectory / Desde el Subdirectorio `version_2/`
```bash
python -m pip install -e ".[dev,analysis,legacy]"
```

---

## Supported Environments / Entornos Soportados

- **Python Version**: Requires `Python >= 3.11`.
- **CI Matrix**: Versions `3.11`, `3.12`, and `3.13` are fully tested in the automatic CI pipeline.
- For detailed package support boundaries and rolling support guidelines, refer to the [Dependency Policy](dependency_policy.md).
- 
- **Versión de Python**: Requiere `Python >= 3.11`.
- **Matriz de CI**: Las versiones `3.11`, `3.12` y `3.13` se prueban completamente en la canalización automática de CI.
- Para obtener detalles sobre los límites de soporte de paquetes y las pautas de soporte continuo, consulte la [Política de Dependencias](dependency_policy.md).

---

## Verification and Smoke Checks / Verificaciones y Pruebas de Humo

After installation, verify that the unified CLI command is registered and running correctly:
Después de la instalación, verifique que el comando CLI unificado esté registrado y se ejecute correctamente:

```bash
hidden-attractors --help
hidden-attractors inspect systems
hidden-attractors validate contract --allow-pending
```

---

## Native C-solvers Compilation / Compilación de Solvers Nativos en C

The library contains native high-performance C solvers under `hidden_attractors/native/csrc/`. These compile dynamically into `.runtime_native/` on demand.
La biblioteca contiene solvers nativos de C de alto rendimiento bajo `hidden_attractors/native/csrc/`. Estos se compilan dinámicamente bajo demanda en `.runtime_native/`.

- **Windows**: Requires a GCC compiler (e.g. MinGW) on your system `PATH`.
- **Linux**: Requires `build-essential`.
- **macOS**: Requires Xcode Command Line Tools. OpenMP parallelism can be supported by running `brew install libomp`. If OpenMP is missing or disabled, compile by passing the environment variable `ALLOW_NO_OPENMP=1`.
- 
- **Windows**: Requiere un compilador GCC (por ejemplo, MinGW) en la variable de entorno `PATH` de su sistema.
- **Linux**: Requiere `build-essential`.
- **macOS**: Requiere Xcode Command Line Tools. El paralelismo OpenMP se puede soportar ejecutando `brew install libomp`. Si falta OpenMP o está desactivado, compile pasando la variable de entorno `ALLOW_NO_OPENMP=1`.
