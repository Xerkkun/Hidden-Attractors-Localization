# Hidden Attractors in Fractional-Order Systems

A scientific research library for reproducing, auditing, and extending numerical workflows for hidden-attractor candidates in integer- and fractional-order Chua and Lur'e systems.

---

## 🗺️ Navigation / Navegación

* 🚀 **[Quick Start Guide / Guía de Inicio Rápido](version_2/docs/quick_start.md)** — **Recommended entry point for new users / Ruta recomendada para usuarias nuevas.**
* 🇬🇧 [English version of the Guide](#-english-installation-and-usage-guide)
* 🇪🇸 [Versión en Español de la Guía](#-guía-de-instalación-y-uso-en-español)
* 📖 For a complete user-facing description of installation, CLI usage, examples, outputs, evidence labels and limitations, see [USER_MANUAL.md](version_2/USER_MANUAL.md). / Para una descripción completa de instalación, CLI, ejemplos, salidas, etiquetas de evidencia y limitaciones, véase [USER_MANUAL.md](version_2/USER_MANUAL.md).

---

## 🇬🇧 English Installation and Usage Guide

### 📂 Repository Structure

The project has been refactored to use `/version_2` as the formal, unified, and maintained library package.
* **`version_2/`**: Contains the core library (`hidden_attractors`), configurations, test suite, and examples.
* **`src/`**: Legacy folder. The entry-point script `src/cli/run_workflow.py` acts as an adapter, redirecting to the new package and displaying a deprecation warning.
* **`DF y NC Chua entero y fraccionario/` / `DF y NC Chua entero y fraccionario copy/`**: Local-only LaTeX report folders. They are untracked and excluded from Git, remaining purely local to the workspace and outside the active Python package or testing flows.

---

### ⚙️ Prerequisites
* **Python**: Version `3.11` or newer.
* **Pip**: Python package installer.

---

### 💻 Installation Instructions

#### Step 1: Install the Python Library
To install the library, run the following command from the repository root:

```bash
pip install -e version_2
```
*Note: The `-e` flag installs the package in **editable mode**, meaning changes to the source code take effect immediately without requiring a reinstall.*

If you plan to run unit tests or do development work, install the development dependencies:
```bash
pip install -e "version_2[dev]"
```

#### Step 2: C-Compiler Toolchain Setup (By Platform)
The library uses high-performance numerical integrators written in C (such as ABM and EFORK solvers), which compile dynamically on demand. A C compiler must be available in your system's environment.

##### 🪟 Windows Setup
1. **Download MSYS2**: Go to [msys2.org](https://www.msys2.org/) and download the installer.
2. **Install GCC and Make**: Open the **MSYS2 UCRT64** terminal and run:
   ```bash
   pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make
   ```
3. **Update PATH Environment Variable**:
   Add the binary directory to your system environment variable `PATH` (typically `C:\msys64\ucrt64\bin`).
4. **Verify**: Open a new PowerShell/Command Prompt window and verify:
   ```cmd
   gcc --version
   ```

##### 🐧 Linux Setup
Install development tools using your system's package manager:
* **Ubuntu / Debian**:
  ```bash
  sudo apt update
  sudo apt install build-essential
  ```
* **Fedora / RHEL**:
  ```bash
  sudo dnf groupinstall "Development Tools"
  ```
* **Arch Linux**:
  ```bash
  sudo pacman -S base-devel
  ```

##### 🍎 macOS Setup
1. **Install Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```
2. **Install OpenMP support** (recommended for parallel processing of basins/bifurcations):
   ```bash
   brew install libomp
   ```
3. *(Optional)* If you wish to disable OpenMP, set the environment variable:
   ```bash
   export ALLOW_NO_OPENMP=1
   ```

---

### 🧪 Verification and Testing

1. **Run Unit Tests**: Ensure everything works correctly. Run the suite from `version_2/`. At the current thesis-freeze audit, the artifacts under `version_2/validation/freeze_audit/` report 797 passed tests and 34 skipped tests. Future runs may change the count; the freeze-audit artifact is the official source:
   ```bash
   cd version_2
   pytest
   ```
2. **Verify CLI Registration**: Check that the CLI tool is registered in your environment:
   ```bash
   hidden-attractors --help
   ```

---

### 🚀 Running Experiments

You can launch simulations and analysis pipelines directly using YAML configuration files:

```bash
hidden-attractors run -c version_2/configs/examples/chua_arctan_attractor_only_fractional.yaml
```
*(Note: Chua arctan is implemented algebraically, pending full hiddenness validation.)*

The outputs (csv trajectories, figures, and logs) will be saved in the `outputs/` folder.

For details on configuration keys and workflow options, refer to the [Reference Guide (REFERENCE_GUIDE.md)](version_2/REFERENCE_GUIDE.md).

---
---

## 🇪🇸 Guía de Instalación y Uso en Español

### 📂 Estructura del Repositorio

El proyecto ha sido reestructurado para utilizar `/version_2` como la librería formal única, unificada y mantenida.
* **`version_2/`**: Contiene la librería principal (`hidden_attractors`), configuraciones, conjunto de pruebas unitarias y ejemplos.
* **`src/`**: Carpeta heredada (legacy). El script `src/cli/run_workflow.py` actúa como adaptador redirigiendo al nuevo paquete y emitiendo una advertencia de depreciación.
* **`DF y NC Chua entero y fraccionario/` / `DF y NC Chua entero y fraccionario copy/`**: Carpetas de reportes LaTeX de uso estrictamente local. Están excluidas de Git (untracked) y no forman parte del flujo activo de la librería ni del conjunto de pruebas automatizadas.

---

### ⚙️ Requisitos Previos
* **Python**: Versión `3.11` o superior.
* **Pip**: Instalador de paquetes de Python.

---

### 💻 Instrucciones de Instalación

#### Paso 1: Instalar la Librería de Python
Para instalar la librería, ejecuta el siguiente comando desde la raíz del repositorio:

```bash
pip install -e version_2
```
*Nota: La bandera `-e` instala el paquete en **modo editable**, lo que permite que los cambios en el código fuente tengan efecto inmediato sin necesidad de reinstalar.*

Si planeas ejecutar pruebas unitarias o realizar tareas de desarrollo, instala las dependencias de desarrollo:
```bash
pip install -e "version_2[dev]"
```

#### Paso 2: Configuración del Compilador de C (Por Plataforma)
La librería integra integradores numéricos de alto rendimiento escritos en C (solver ABM y EFORK), los cuales se compilan dinámicamente según sea necesario. Es necesario contar con un compilador de C en el sistema.

##### 🪟 Configuración en Windows
1. **Descargar MSYS2**: Ve a [msys2.org](https://www.msys2.org/) y descarga el instalador.
2. **Instalar GCC y Make**: Abre la terminal de **MSYS2 UCRT64** y ejecuta:
   ```bash
   pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make
   ```
3. **Actualizar la Variable PATH**:
   Agrega la ruta de binarios a tu variable de entorno `PATH` del sistema (usualmente `C:\msys64\ucrt64\bin`).
4. **Verificar**: Abre una nueva consola de PowerShell o CMD y comprueba con:
   ```cmd
   gcc --version
   ```

##### 🐧 Configuración en Linux
Instala las herramientas de desarrollo utilizando el gestor de paquetes de tu distribución:
* **Ubuntu / Debian**:
  ```bash
  sudo apt update
  sudo apt install build-essential
  ```
* **Fedora / RHEL**:
  ```bash
  sudo dnf groupinstall "Development Tools"
  ```
* **Arch Linux**:
  ```bash
  sudo pacman -S base-devel
  ```

##### 🍎 Configuración en macOS
1. **Instalar Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```
2. **Instalar soporte para OpenMP** (recomendado para procesamiento en paralelo de cuencas/bifurcaciones):
   ```bash
   brew install libomp
   ```
3. *(Opcional)* Si deseas compilar sin soporte OpenMP, define la variable de entorno:
   ```bash
   export ALLOW_NO_OPENMP=1
   ```

---

### 🧪 Verificación y Pruebas

1. **Ejecutar Pruebas Unitarias**: Asegúrate de que todo funcione correctamente. Ejecuta la suite desde `version_2/`. En el estado actual de auditoría tesis-freeze, los artefactos bajo `version_2/validation/freeze_audit/` reportan 797 pruebas aprobadas y 34 omitidas. El conteo puede cambiar en ejecuciones futuras; la fuente oficial es el artefacto de freeze audit:
   ```bash
   cd version_2
   pytest
   ```
2. **Verificar Comando CLI**: Comprueba que la herramienta CLI esté registrada en tu sistema:
   ```bash
   hidden-attractors --help
   ```

---

### 🚀 Ejecución de Experimentos

Puedes lanzar simulaciones y flujos de análisis directamente utilizando archivos de configuración YAML:

```bash
hidden-attractors run -c version_2/configs/examples/chua_arctan_attractor_only_fractional.yaml
```
*(Nota: Chua arctan está implementado algebraicamente, pendiente de validación completa de ocultedad.)*

Los resultados (trayectorias en csv, imágenes y bitácoras) se guardarán en la carpeta `outputs/`.

Para más detalles sobre las opciones de los archivos de configuración y fases de análisis, consulta la [Guía de Referencia (REFERENCE_GUIDE.md)](version_2/REFERENCE_GUIDE.md).

---

### CPC readiness and citation

The active distribution is `version_2/`. Root-level metadata files (`CITATION.cff`, `.zenodo.json`, `codemeta.json`, `AUTHORS.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`, and `REPRODUCIBILITY.md`) describe the software package and the current CPC-preparation state. The archived DOI recorded for this package is:

```text
10.17605/OSF.IO/ZGK74
```

Editorial drafts live in `paper/`. Reproducible submission packaging lives in `version_2/cpc_submission/`. These folders do not promote new scientific claims; they organize code, metadata, and evidence boundaries for a future CPC submission.

### Freeze audit status for CPC preparation

The current CPC-preparation files are not yet covered by a regenerated freeze audit. The last recorded freeze audit corresponds to commit `2bcea3430c50d3fb4e5eb70c8621cb3550dcc59a`; regenerate it before treating the CPC package as final submission material.
