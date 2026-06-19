# Official Index of Examples and Workflows / Índice Oficial de Ejemplos y Workflows

## Table of Contents / Índice de Contenidos
- [English Version](#english-version)
- [Versión en Español](#versión-en-español)

---

## English Version

# Official Index of Examples and Workflows

This document classifies all available execution paths in the repository in a rigorous and official manner.

---

## A. Reproducible Official Examples

These are the only complete, official examples that a new user should execute directly to understand the methodological flow.

### Example 1 — Non-Smooth Fractional Chua with Biased Describing Function

* **Execution path**: [run_example.py](../examples/chua_nonsmooth_biased_hidden_attractor/run_example.py)
* **Purpose**:Run the localization and verification pipeline for hidden attractors using the Biased Describing Function (BDF).
* **Official commands**:

```bash
  cd version_2/examples/chua_nonsmooth_biased_hidden_attractor
  # Quick smoke check (~1-2 minutes) / Ejecución de prueba rápida (~1-2 minutos)
  python run_example.py --quick
  # Standard execution (~10-15 minutes) / Ejecución estándar (~10-15 minutos)
  python run_example.py
  # Full massive run (~hours) / Ejecución masiva completa (~horas)
  python run_example.py --all
  ```

---

## B. CLI Presets

CLI presets are pre-packaged configurations designed to run analysis workflows using the unified command `hidden-attractors`. They are configuration profiles for the command-line interface, not standalone examples.

### 1. `chua_integer`
* **Purpose**:Localize hidden attractors in the integer-order Chua system.
* **System**: `chua_integer_saturation` (piecewise linear, q = 1.0).
* **Analysis type**:Lur'e DF seed search, parameter continuation, and attractor simulation.
* **Status**:Stable (Recommended).
* **Command**: `hidden-attractors run -p chua_integer`
* **Expected output**:Parameter continuation log and simulated Chua attractor.
* **Generates figures**:Yes, under `library_figures/` (attractor, Nyquist, time-series, continuation).
* **Recommended for new users**:Yes.

### 2. `chua_fractional`
* **Purpose**:Localize hidden attractors in the fractional-order Chua system.
* **System**: `chua_fractional_saturation` (piecewise linear, q = 0.998).
* **Analysis type**:Fractional seed search, homotopy continuation, and fractional-solver simulation.
* **Status**:Stable (Recommended).
* **Command**: `hidden-attractors run -p chua_fractional`
* **Expected output**:Localized trajectories and fractional continuation data.
* **Generates figures**:Yes, under `library_figures/`.
* **Recommended for new users**:Yes.

### 3. `chua_arctan`
* **Purpose**:Localize hidden attractors with smooth nonlinearity (arctan type).
* **System**: `chua-arctan` (fractional order, q = 0.95).
* **Analysis type**:Lur'e DF with arctan nonlinearity and verification.
* **Status**:Stable (Recommended).
* **Command**: `hidden-attractors run -p chua_arctan`
* **Expected output**:Analytical approximation and continuation data for smooth nonlinearity.
* **Generates figures**:Yes, under `library_figures/`.
* **Recommended for new users**:Yes.

### 4. `chua_bifurcation`
* **Purpose**:Generate fractional-order bifurcation diagrams by varying key system parameters.
* **System**: `chua_fractional_saturation` (q = 0.998).
* **Analysis type**:Dynamic bifurcation analysis via parameter sweeping.
* **Status**:Stable.
* **Command**: `hidden-attractors bifurcation run -p chua_bifurcation` (or with `-c configs/examples/chua_fractional_bifurcation.yaml`)
* **Expected output**:CSV of detected local extrema and bifurcation diagram.
* **Generates figures**:Yes.
* **Recommended for new users**:No (requires more computing time and specialized knowledge).

### 5. `chua_lyapunov`
* **Purpose**:Compute Lyapunov exponent spectra and finite-time local estimates for fractional Chua.
* **System**: `chua_fractional_saturation` (q = 0.98).
* **Analysis type**:Variational/Jacobian analysis and temporal convergence.
* **Status**:Stable.
* **Command**: `hidden-attractors lyapunov compute -c configs/examples/chua_fractional_lyapunov.yaml`
* **Expected output**:CSV spectrum and convergence, PNG/PDF convergence curves, metadata reports.
* **Generates figures**:Yes.
* **Recommended for new users**:Yes.

### 6. `chua_zero_one`
* **Purpose**:Execute the 0-1 test for chaos as a complementary diagnostic on Chua time-series.
* **System**: `chua_fractional_saturation` (q = 0.98).
* **Analysis type**:K parameter calculation and displacement phase plane (p-q).
* **Status**:Stable.
* **Command**: `hidden-attractors chaos-test zero-one -c configs/examples/chua_fractional_zero_one.yaml`
* **Expected output**:CSV of c-values, displacement trajectory (p-q), and chaotic/regular classification report.
* **Generates figures**:Yes.
* **Recommended for new users**:Yes.

### 7. `chua_basin`
* **Purpose**:Compute and classify local basins of attraction for detected attractors.
* **System**: `chua_fractional_saturation` (q = 0.998).
* **Analysis type**:2D mapping of basin slices.
* **Status**:Stable
* **Command**: `hidden-attractors run -p chua_basin`
* **Expected output**:Classification matrix of initial conditions (convergence to equilibrium, attractor, or divergence).
* **Generates figures**:Yes.
* **Recommended for new users**:No (computationally heavy).

---

## C. Specialized Workflows

Specialized workflows are not standalone examples or alternative methodologies. They are low-level interfaces used by the official pipeline, validation tests, or advanced analysis.

> [!WARNING]
> Advanced usage. Not recommended as a first entry point.

* **`hidden-attractors protocol`**:CLI interface for step-by-step execution of formal protocol stages (seed generation, precheck, continuation, filtering, reference, and diagnostics).
* **`hidden-attractors robustness overlay`**:Workflow to run numerical robustness analysis under varying step sizes and solver conditions.
* **`hidden-attractors basin refined`**:Advanced pipeline for refining local basin of attraction boundaries.
* **`hidden-attractors hiddenness sphere-controls`**:Local hiddenness checks sampling spherical neighborhoods around equilibrium points.
* **`hidden-attractors basin strict-target-refinement`**:Fine search and refinement of target attractor.
* **`hidden-attractors report fractional-run`**:Advanced orchestrator for automatic compilation of execution reports and figure galleries.
* **`hidden-attractors validate contract`**:Validation drivers for internal consistency.

---

## D. Internal Utilities

These utility commands are not intended for new users. They are used internally for development, debugging, and consistency checking.

* **`hidden-attractors inspect candidates`**:Utility to inspect metadata of registered hidden attractor candidates.
* **`hidden-attractors inspect systems`**:Inspection tool to list available systems in the central registry.
* **`hidden-attractors inspect workflow-requirements`**:Diagnostic script to validate local dependencies and numerical capabilities.
* **`hidden-attractors validate contract`**:Runs validation assertions against the data model.

---

## E. Legacy

These directories contain frozen or historical code removed from the active development path to guarantee reproducibility. They are not active alternatives or valid solvers for simulation.

Historical migration scripts are intentionally excluded from the active repository.
The active implementation lives in `version_2/hidden_attractors/`.

* **`version_2/tools/legacy/`**:Frozen historical tools and sources, exposed only as legacy compatibility commands if needed.

---

## Versión en Español

# Índice Oficial de Ejemplos y Workflows

Este documento clasifica de forma rigurosa y oficial todas las formas de ejecución disponibles en el repositorio.

---

## Ejemplos Oficiales Reproducibles

Estos son los únicos ejemplos completos y oficiales que una usuaria nueva debe ejecutar directamente para comprender el flujo metodológico.

### Ejemplo 1 — Chua fraccionario no suave con función descriptiva sesgada

* **Ruta de ejecución**: [run_example.py](../examples/chua_nonsmooth_biased_hidden_attractor/run_example.py)
* **Propósito**:Ejecutar el pipeline de localización y verificación de atractores ocultos usando la Función Descriptiva Sesgada (BDF).
* **Comandos oficiales**:

```bash
  cd version_2/examples/chua_nonsmooth_biased_hidden_attractor
  # Quick smoke check (~1-2 minutes) / Ejecución de prueba rápida (~1-2 minutos)
  python run_example.py --quick
  # Standard execution (~10-15 minutes) / Ejecución estándar (~10-15 minutos)
  python run_example.py
  # Full massive run (~hours) / Ejecución masiva completa (~horas)
  python run_example.py --all
  ```

---

## Presets de CLI

Los presets de CLI son configuraciones rápidas predefinidas para ejecutar flujos de análisis de la librería mediante el comando unificado `hidden-attractors`. No son ejemplos independientes, sino perfiles de configuración para la interfaz de comandos.

### 1. `chua_integer`
* **Propósito**:Localizar atractores ocultos en el sistema Chua de orden entero.
* **Sistema**: `chua_integer_saturation` (piecewise linear, q = 1.0).
* **Tipo de análisis**:Búsqueda de semillas Lure DF, continuación y simulación de atractores.
* **Estado**:Estable (Recomendado).
* **Comando**: `hidden-attractors run -p chua_integer`
* **Salida esperada**:Bitácora de continuación de parámetros y el atractor de Chua simulado.
* **Genera figuras**:Sí, bajo `library_figures/` (atractor, Nyquist, series temporales, continuación).
* **Recomendado para nuevas usuarias**:Sí.

### 2. `chua_fractional`
* **Propósito**:Localizar atractores ocultos en el sistema Chua de orden fraccionario.
* **Sistema**: `chua_fractional_saturation` (piecewise linear, q = 0.998).
* **Tipo de análisis**:Búsqueda de semillas fraccionarias, continuación homotópica y simulación con integrador fraccionario.
* **Estado**:Estable (Recomendado).
* **Comando**: `hidden-attractors run -p chua_fractional`
* **Salida esperada**:Trayectorias localizadas y datos de continuación fraccionaria.
* **Genera figuras**:Sí, bajo `library_figures/`.
* **Recomendado para nuevas usuarias**:Sí.

### 3. `chua_arctan`
* **Propósito**:Localizar atractores ocultos con no-linealidad suave (tipo arcotangente).
* **Sistema**: `chua-arctan` (fractional order, q = 0.95).
* **Tipo de análisis**:Lure DF con no-linealidad arcotangente y verificación.
* **Estado**:Estable (Recomendado).
* **Comando**: `hidden-attractors run -p chua_arctan`
* **Salida esperada**:Datos de aproximación analítica y continuación en no-linealidad suave.
* **Genera figuras**:Sí, bajo `library_figures/`.
* **Recomendado para nuevas usuarias**:Sí.

### 4. `chua_bifurcation`
* **Propósito**:Generar diagramas de bifurcación de orden fraccionario variando parámetros clave del sistema.
* **Sistema**: `chua_fractional_saturation` (q = 0.998).
* **Tipo de análisis**:Análisis dinámico de bifurcaciones mediante barrido de parámetros.
* **Estado**:Estable.
* **Comando**: `hidden-attractors bifurcation run -p chua_bifurcation` (or with `-c configs/examples/chua_fractional_bifurcation.yaml`)
* **Salida esperada**:Archivo CSV de extremos locales detectados y diagrama de bifurcación.
* **Genera figuras**:Sí.
* **Recomendado para nuevas usuarias**:No (requiere más tiempo de cómputo y conocimientos especializados).

### 5. `chua_lyapunov`
* **Propósito**:Computar el espectro de exponentes de Lyapunov y estimaciones locales de tiempo finito para Chua fraccionario.
* **Sistema**: `chua_fractional_saturation` (q = 0.98).
* **Tipo de análisis**:Análisis variacional/Jacobiano y convergencia temporal.
* **Estado**:Estable.
* **Comando**: `hidden-attractors lyapunov compute -c configs/examples/chua_fractional_lyapunov.yaml`
* **Salida esperada**:CSV de espectro y convergencia, curvas de convergencia PNG/PDF, reportes de metadatos.
* **Genera figuras**:Sí.
* **Recomendado para nuevas usuarias**:Sí.

### 6. `chua_zero_one`
* **Propósito**:Ejecutar la prueba 0-1 de caos como diagnóstico computacional complementario sobre series temporales del sistema Chua.
* **Sistema**: `chua_fractional_saturation` (q = 0.98).
* **Tipo de análisis**:Cálculo de parámetro K y plano de fase de desplazamiento.
* **Estado**:Estable.
* **Comando**: `hidden-attractors chaos-test zero-one -c configs/examples/chua_fractional_zero_one.yaml`
* **Salida esperada**:CSV de valores c, trayectoria de desplazamiento p-q y reporte de clasificación caótica/regular.
* **Genera figuras**:Sí.
* **Recomendado para nuevas usuarias**:Sí.

### 7. `chua_basin`
* **Propósito**:Computar y clasificar las cuencas de atracción locales de los atractores detectados.
* **Sistema**: `chua_fractional_saturation` (q = 0.998).
* **Tipo de análisis**:Mapeo 2D de secciones de cuencas de atracción.
* **Estado**:Advanced.
* **Comando**: `hidden-attractors run -p chua_basin`
* **Salida esperada**:Matriz de clasificaciones de condiciones iniciales (convergencia a equilibrio, atractor o divergencia).
* **Genera figuras**:Sí.
* **Recomendado para nuevas usuarias**:No (cómputo masivo muy demandante).

---

## Workflows Especializados

Los workflows especializados no son ejemplos independientes ni metodologías alternativas. Son interfaces de bajo nivel usadas por el pipeline oficial, pruebas de validación o análisis avanzados.

> [!WARNING]
> Uso avanzado. No es el punto de entrada recomendado para una primera ejecución.

* **`hidden-attractors protocol`**:Interfaz CLI para la ejecución paso a paso de las etapas formales del protocolo (generación de semillas, precheck, continuación, filtrado, referencia y diagnóstico).
* **`hidden-attractors robustness overlay`**:Workflow para ejecutar análisis de robustez numérica variando tamaños de paso y condiciones del resolvedor.
* **`hidden-attractors basin refined`**:Pipeline avanzado para el refinamiento de bordes de cuencas de atracción locales.
* **`hidden-attractors published danca-abm-sphere-controls`**:Pruebas locales de ocultedad muestreando vecindades esféricas alrededor de los puntos de equilibrio (mecanismo estándar y modo de comparación Danca).
* **`hidden-attractors hiddenness strict-target-refinement`**:Rutina de búsqueda y aproximación fina del atractor objetivo.
* **`hidden-attractors report fractional-run`**:Generador automático de reportes científicos unificados.
* **`hidden-attractors validate contract`**:Controladores de validación de la consistencia interna.

---

## Auxiliares Internos

Estas herramientas y comandos de utilidad no deben aparecer como comandos recomendados para usuarias nuevas. Son utilizados internamente para labores de desarrollo, depuración y consistencia.

* **`hidden-attractors inspect candidates`**:Utilidad para inspeccionar los metadatos de los candidatos a atractores ocultos registrados.
* **`hidden-attractors inspect systems`**:Herramienta de inspección para listar los sistemas dinámicos disponibles en el registro central.
* **`hidden-attractors inspect workflow-requirements`**:Script de diagnóstico para validar dependencias y capacidades numéricas del entorno local.
* **`hidden-attractors validate contract`**:Ejecución de contratos de validación contra el modelo de datos.

---

## Archived

Estos directorios contienen código congelado o histórico que ha sido retirado de la ruta activa de desarrollo para garantizar la reproducibilidad y evitar confusión metodológica. No son alternativas activas ni ejecutables válidos para simulación.

* **`version_2/tools/legacy/`**:Fuentes y herramientas históricas congeladas, expuestas solo como comandos de compatibilidad legacy si es necesario.