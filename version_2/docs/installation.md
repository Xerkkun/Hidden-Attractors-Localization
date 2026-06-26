# Installation / Instalación

## Table of Contents / Índice de Contenidos

- [English Version](#english-version)
- [Versión en Español](#versión-en-español)

---

## English Version

### 1. PyPI Installation (For End Users)

To install the latest stable version of the package directly from PyPI, run:

```bash
python -m pip install hidden-attractors-fo
```

Verify that the unified public CLI is installed and runs:

```bash
hidden-attractors --help
hidden-attractors inspect systems
```

### 2. TestPyPI Installation (For Release Testing)

To test the package distribution, install from TestPyPI. Note that TestPyPI does not always resolve dependencies automatically, so they should be installed first or separately if needed:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps hidden-attractors-fo
```

### 3. Development Installation from Repository

To install the library in editable mode for development, running tests, or building documentation:

From the workspace root directory:

```bash
python -m pip install -e "version_2[dev,analysis,docs,legacy]"
```

From the `version_2/` directory:

```bash
python -m pip install -e ".[dev,analysis,docs,legacy]"
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\version_2[dev,analysis,docs,legacy]"
```

---

## Supported Environments

- **Python Version**: Requires `Python >= 3.11`.
- **CI Matrix**: Versions `3.11`, `3.12`, and `3.13` are fully tested in the automatic CI pipeline.
- For detailed package support boundaries and rolling support guidelines, refer to the [Dependency Policy](dependency_policy.md).

---

## Verification and Smoke Checks

After installation, verify that the unified CLI command is registered and running correctly:

```bash
hidden-attractors --help
hidden-attractors inspect systems
hidden-attractors validate contract --allow-pending
```

---

## Native C-solvers Compilation

The library contains native high-performance C solvers under `hidden_attractors/native/csrc/`. These compile dynamically into `.runtime_native/` on demand.

- **Windows**: Requires a GCC compiler (e.g. MinGW) on your system `PATH`.
- **Linux**: Requires `build-essential`.
- **macOS**: Requires Xcode Command Line Tools. OpenMP parallelism can be supported by running `brew install libomp`. If OpenMP is missing or disabled, compile by passing the environment variable `ALLOW_NO_OPENMP=1`.

---

## Versión en Español

### 1. Instalación desde PyPI (Para Usuarios Finales)

Para instalar la última versión estable directamente desde PyPI, ejecute:

```bash
python -m pip install hidden-attractors-fo
```

Verifique la interfaz de línea de comandos pública unificada:

```bash
hidden-attractors --help
hidden-attractors inspect systems
```

### 2. Instalación desde TestPyPI (Prueba de Release)

Para probar la distribución del paquete, instálelo desde TestPyPI. Tenga en cuenta que TestPyPI no siempre resuelve las dependencias automáticamente, por lo que es posible que deban instalarse de forma previa o independiente si es necesario:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps hidden-attractors-fo
```

### 3. Instalación de Desarrollo desde Repositorio

Para instalar la biblioteca en modo editable para desarrollo, ejecución de pruebas o compilación de documentación:

Desde el directorio raíz del espacio de trabajo:

```bash
python -m pip install -e "version_2[dev,analysis,docs,legacy]"
```

Desde el subdirectorio `version_2/`:

```bash
python -m pip install -e ".[dev,analysis,docs,legacy]"
```

En Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".\version_2[dev,analysis,docs,legacy]"
```

---

## Entornos Soportados

- **Versión de Python**: Requiere `Python >= 3.11`.
- **Matriz de CI**: Las versiones `3.11`, `3.12` y `3.13` se prueban completamente en la canalización automática de CI.
- Para obtener detalles sobre los límites de soporte de paquetes y las pautas de soporte continuo, consulte la [Política de Dependencias](dependency_policy.md).

---

## Verificaciones y Pruebas de Humo

Después de la instalación, verifique que el comando CLI unificado esté registrado y se ejecute correctamente:

```bash
hidden-attractors --help
hidden-attractors inspect systems
hidden-attractors validate contract --allow-pending
```

---

## Compilación de Solvers Nativos en C

La biblioteca contiene solvers nativos de C de alto rendimiento bajo `hidden_attractors/native/csrc/`. Estos se compilan dinámicamente bajo demanda en `.runtime_native/`.

- **Windows**: Requiere un compilador GCC (por ejemplo, MinGW) en la variable de entorno `PATH` de su sistema.
- **Linux**: Requiere `build-essential`.
- **macOS**: Requiere Xcode Command Line Tools. El paralelismo OpenMP se puede soportar ejecutando `brew install libomp`. Si falta OpenMP o está desactivado, compile pasando la variable de entorno `ALLOW_NO_OPENMP=1`.
