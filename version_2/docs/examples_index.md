# Official Index of Examples and Workflows / Índice Oficial de Ejemplos y Workflows

This document classifies all available execution paths in the repository in a rigorous and official manner.
Este documento clasifica de forma rigurosa y oficial todas las formas de ejecución disponibles en el repositorio.

---

## A. Reproducible Official Examples / Ejemplos Oficiales Reproducibles

These are the only complete, official examples that a new user should execute directly to understand the methodological flow.
Estos son los únicos ejemplos completos y oficiales que una usuaria nueva debe ejecutar directamente para comprender el flujo metodológico.

### Example 1 — Non-Smooth Fractional Chua with Biased Describing Function / Ejemplo 1 — Chua fraccionario no suave con función descriptiva sesgada

* **Execution path / Ruta de ejecución**: [run_example.py](../examples/chua_nonsmooth_biased_hidden_attractor/run_example.py)
* **Purpose / Propósito**: Run the localization and verification pipeline for hidden attractors using the Biased Describing Function (BDF). / Ejecutar el pipeline de localización y verificación de atractores ocultos usando la Función Descriptiva Sesgada (BDF).
* **Official commands / Comandos oficiales**:
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

## B. CLI Presets / Presets de CLI

CLI presets are pre-packaged configurations designed to run analysis workflows using the unified command `hidden-attractors`. They are configuration profiles for the command-line interface, not standalone examples.
Los presets de CLI son configuraciones rápidas predefinidas para ejecutar flujos de análisis de la librería mediante el comando unificado `hidden-attractors`. No son ejemplos independientes, sino perfiles de configuración para la interfaz de comandos.

### 1. `chua_integer`
* **Purpose / Propósito**: Localize hidden attractors in the integer-order Chua system. / Localizar atractores ocultos en el sistema Chua de orden entero.
* **System / Sistema**: `chua_integer_saturation` (piecewise linear, q = 1.0).
* **Analysis type / Tipo de análisis**: Lur'e DF seed search, parameter continuation, and attractor simulation. / Búsqueda de semillas Lure DF, continuación y simulación de atractores.
* **Status / Estado**: Stable (Recommended). / Estable (Recomendado).
* **Command / Comando**: `hidden-attractors run -p chua_integer`
* **Expected output / Salida esperada**: Parameter continuation log and simulated Chua attractor. / Bitácora de continuación de parámetros y el atractor de Chua simulado.
* **Generates figures / Genera figuras**: Yes, under `library_figures/` (attractor, Nyquist, time-series, continuation). / Sí, bajo `library_figures/` (atractor, Nyquist, series temporales, continuación).
* **Recommended for new users / Recomendado para nuevas usuarias**: Yes. / Sí.

### 2. `chua_fractional`
* **Purpose / Propósito**: Localize hidden attractors in the fractional-order Chua system. / Localizar atractores ocultos en el sistema Chua de orden fraccionario.
* **System / Sistema**: `chua_fractional_saturation` (piecewise linear, q = 0.998).
* **Analysis type / Tipo de análisis**: Fractional seed search, homotopy continuation, and fractional-solver simulation. / Búsqueda de semillas fraccionarias, continuación homotópica y simulación con integrador fraccionario.
* **Status / Estado**: Stable (Recommended). / Estable (Recomendado).
* **Command / Comando**: `hidden-attractors run -p chua_fractional`
* **Expected output / Salida esperada**: Localized trajectories and fractional continuation data. / Trayectorias localizadas y datos de continuación fraccionaria.
* **Generates figures / Genera figuras**: Yes, under `library_figures/`. / Sí, bajo `library_figures/`.
* **Recommended for new users / Recomendado para nuevas usuarias**: Yes. / Sí.

### 3. `chua_arctan`
* **Purpose / Propósito**: Localize hidden attractors with smooth nonlinearity (arctan type). / Localizar atractores ocultos con no-linealidad suave (tipo arcotangente).
* **System / Sistema**: `chua-arctan` (fractional order, q = 0.95).
* **Analysis type / Tipo de análisis**: Lur'e DF with arctan nonlinearity and verification. / Lure DF con no-linealidad arcotangente y verificación.
* **Status / Estado**: Stable (Recommended). / Estable (Recomendado).
* **Command / Comando**: `hidden-attractors run -p chua_arctan`
* **Expected output / Salida esperada**: Analytical approximation and continuation data for smooth nonlinearity. / Datos de aproximación analítica y continuación en no-linealidad suave.
* **Generates figures / Genera figuras**: Yes, under `library_figures/`. / Sí, bajo `library_figures/`.
* **Recommended for new users / Recomendado para nuevas usuarias**: Yes. / Sí.

### 4. `chua_bifurcation`
* **Purpose / Propósito**: Generate fractional-order bifurcation diagrams by varying key system parameters. / Generar diagramas de bifurcación de orden fraccionario variando parámetros clave del sistema.
* **System / Sistema**: `chua_fractional_saturation` (q = 0.998).
* **Analysis type / Tipo de análisis**: Dynamic bifurcation analysis via parameter sweeping. / Análisis dinámico de bifurcaciones mediante barrido de parámetros.
* **Status / Estado**: Stable. / Estable.
* **Command / Comando**: `hidden-attractors bifurcation run -p chua_bifurcation` (or with `-c configs/examples/chua_fractional_bifurcation.yaml`)
* **Expected output / Salida esperada**: CSV of detected local extrema and bifurcation diagram. / Archivo CSV de extremos locales detectados y diagrama de bifurcación.
* **Generates figures / Genera figuras**: Yes. / Sí.
* **Recommended for new users / Recomendado para nuevas usuarias**: No (requires more computing time and specialized knowledge). / No (requiere más tiempo de cómputo y conocimientos especializados).

### 5. `chua_lyapunov`
* **Purpose / Propósito**: Compute Lyapunov exponent spectra and finite-time local estimates for fractional Chua. / Computar el espectro de exponentes de Lyapunov y estimaciones locales de tiempo finito para Chua fraccionario.
* **System / Sistema**: `chua_fractional_saturation` (q = 0.98).
* **Analysis type / Tipo de análisis**: Variational/Jacobian analysis and temporal convergence. / Análisis variacional/Jacobiano y convergencia temporal.
* **Status / Estado**: Stable. / Estable.
* **Command / Comando**: `hidden-attractors lyapunov compute -c configs/examples/chua_fractional_lyapunov.yaml`
* **Expected output / Salida esperada**: CSV spectrum and convergence, PNG/PDF convergence curves, metadata reports. / CSV de espectro y convergencia, curvas de convergencia PNG/PDF, reportes de metadatos.
* **Generates figures / Genera figuras**: Yes. / Sí.
* **Recommended for new users / Recomendado para nuevas usuarias**: Yes. / Sí.

### 6. `chua_zero_one`
* **Purpose / Propósito**: Execute the 0-1 test for chaos as a complementary diagnostic on Chua time-series. / Ejecutar la prueba 0-1 de caos como diagnóstico computacional complementario sobre series temporales del sistema Chua.
* **System / Sistema**: `chua_fractional_saturation` (q = 0.98).
* **Analysis type / Tipo de análisis**: K parameter calculation and displacement phase plane (p-q). / Cálculo de parámetro K y plano de fase de desplazamiento.
* **Status / Estado**: Stable. / Estable.
* **Command / Comando**: `hidden-attractors chaos-test zero-one -c configs/examples/chua_fractional_zero_one.yaml`
* **Expected output / Salida esperada**: CSV of c-values, displacement trajectory (p-q), and chaotic/regular classification report. / CSV de valores c, trayectoria de desplazamiento p-q y reporte de clasificación caótica/regular.
* **Generates figures / Genera figuras**: Yes. / Sí.
* **Recommended for new users / Recomendado para nuevas usuarias**: Yes. / Sí.

### 7. `chua_basin`
* **Purpose / Propósito**: Compute and classify local basins of attraction for detected attractors. / Computar y clasificar las cuencas de atracción locales de los atractores detectados.
* **System / Sistema**: `chua_fractional_saturation` (q = 0.998).
* **Analysis type / Tipo de análisis**: 2D mapping of basin slices. / Mapeo 2D de secciones de cuencas de atracción.
* **Status / Estado**: Stable / Advanced. / Estable / Avanzado.
* **Command / Comando**: `hidden-attractors run -p chua_basin`
* **Expected output / Salida esperada**: Classification matrix of initial conditions (convergence to equilibrium, attractor, or divergence). / Matriz de clasificaciones de condiciones iniciales (convergencia a equilibrio, atractor o divergencia).
* **Generates figures / Genera figuras**: Yes. / Sí.
* **Recommended for new users / Recomendado para nuevas usuarias**: No (computationally heavy). / No (cómputo masivo muy demandante).

---

## C. Specialized Workflows / Workflows Especializados

Specialized workflows are not standalone examples or alternative methodologies. They are low-level interfaces used by the official pipeline, validation tests, or advanced analysis.
Los workflows especializados no son ejemplos independientes ni metodologías alternativas. Son interfaces de bajo nivel usadas por el pipeline oficial, pruebas de validación o análisis avanzados.

> [!WARNING]
> Advanced usage. Not recommended as a first entry point.
> Uso avanzado. No es el punto de entrada recomendado para una primera ejecución.

* **`hidden-attractors protocol`**: CLI interface for step-by-step execution of formal protocol stages (seed generation, precheck, continuation, filtering, reference, and diagnostics). / Interfaz CLI para la ejecución paso a paso de las etapas formales del protocolo (generación de semillas, precheck, continuación, filtrado, referencia y diagnóstico).
* **`hidden-attractors robustness overlay`**: Workflow to run numerical robustness analysis under varying step sizes and solver conditions. / Workflow para ejecutar análisis de robustez numérica variando tamaños de paso y condiciones del resolvedor.
* **`hidden-attractors basin refined`**: Advanced pipeline for refining local basin of attraction boundaries. / Pipeline avanzado para el refinamiento de bordes de cuencas de atracción locales.
* **`hidden-attractors hiddenness sphere-controls` / `hidden-attractors published danca-abm-sphere-controls`**: Local hiddenness checks sampling spherical neighborhoods around equilibrium points. / Pruebas locales de ocultedad muestreando vecindades esféricas alrededor de los puntos de equilibrio (mecanismo estándar y modo de comparación Danca).
* **`hidden-attractors basin strict-target-refinement` / `hidden-attractors hiddenness strict-target-refinement`**: Fine search and refinement of target attractor. / Rutina de búsqueda y aproximación fina del atractor objetivo.
* **`hidden-attractors report fractional-run`**: Advanced orchestrator for automatic compilation of execution reports and figure galleries. / Generador automático de reportes científicos unificados.
* **`hidden-attractors validate contract`**: Validation drivers for internal consistency. / Controladores de validación de la consistencia interna.

---

## D. Internal Utilities / Auxiliares Internos

These utility commands are not intended for new users. They are used internally for development, debugging, and consistency checking.
Estas herramientas y comandos de utilidad no deben aparecer como comandos recomendados para usuarias nuevas. Son utilizados internamente para labores de desarrollo, depuración y consistencia.

* **`hidden-attractors inspect candidates`**: Utility to inspect metadata of registered hidden attractor candidates. / Utilidad para inspeccionar los metadatos de los candidatos a atractores ocultos registrados.
* **`hidden-attractors inspect systems`**: Inspection tool to list available systems in the central registry. / Herramienta de inspección para listar los sistemas dinámicos disponibles en el registro central.
* **`hidden-attractors inspect workflow-requirements`**: Diagnostic script to validate local dependencies and numerical capabilities. / Script de diagnóstico para validar dependencias y capacidades numéricas del entorno local.
* **`hidden-attractors validate contract`**: Runs validation assertions against the data model. / Ejecución de contratos de validación contra el modelo de datos.

---

## E. Legacy / Archived / Legacy / Archivado

These directories contain frozen or historical code removed from the active development path to guarantee reproducibility. They are not active alternatives or valid solvers for simulation.
Estos directorios contienen código congelado o histórico que ha sido retirado de la ruta activa de desarrollo para garantizar la reproducibilidad y evitar confusión metodológica. No son alternativas activas ni ejecutables válidos para simulación.

Historical migration scripts are intentionally excluded from the active repository.
The active implementation lives in `version_2/hidden_attractors/`.

* **`version_2/tools/legacy/`**: Frozen historical tools and sources, exposed only as legacy compatibility commands if needed. / Fuentes y herramientas históricas congeladas, expuestas solo como comandos de compatibilidad legacy si es necesario.
