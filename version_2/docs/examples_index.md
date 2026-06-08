# Índice Oficial de Ejemplos y Workflows

Este documento clasifica de forma rigurosa y oficial todas las formas de ejecución disponibles en el repositorio.

---

## A. Ejemplos Oficiales Reproducibles

Estos son los únicos ejemplos completos y oficiales que una usuaria nueva debe ejecutar directamente para comprender el flujo metodológico.

### Ejemplo 1 — Chua fraccionario no suave con función descriptiva sesgada

* **Ruta de ejecución**: [run_example.py](../examples/chua_nonsmooth_biased_hidden_attractor/run_example.py)
* **Propósito**: Ejecutar el pipeline de localización y verificación de atractores ocultos usando la Función Descriptiva Sesgada (BDF).
* **Comandos oficiales**:
  ```bash
  cd version_2/examples/chua_nonsmooth_biased_hidden_attractor
  # Ejecución de prueba rápida (~1-2 minutos)
  python run_example.py --quick
  # Ejecución estándar (~10-15 minutos)
  python run_example.py
  # Ejecución masiva completa (~horas)
  python run_example.py --all
  ```

---

## B. Presets de CLI

Los presets de CLI son configuraciones rápidas predefinidas para ejecutar flujos de análisis de la librería mediante el comando unificado `hidden-attractors`. No son ejemplos independientes, sino perfiles de configuración para la interfaz de comandos.

### 1. `chua_integer`
* **Propósito**: Localizar atractores ocultos en el sistema Chua de orden entero.
* **Sistema**: `chua_integer_saturation` (piecewise lineal, q = 1.0).
* **Tipo de análisis**: Búsqueda de semillas Lure DF, continuación y simulación de atractores.
* **Estado**: Estable (Recomendado).
* **Comando**: `hidden-attractors run -p chua_integer`
* **Salida esperada**: Bitácora de continuación de parámetros y el atractor de Chua simulado.
* **Genera figuras**: Sí, bajo `library_figures/` (atractor, Nyquist, series temporales, continuación).
* **Recomendado para nuevas usuarias**: Sí.

### 2. `chua_fractional`
* **Propósito**: Localizar atractores ocultos en el sistema Chua de orden fraccionario.
* **Sistema**: `chua_fractional_saturation` (piecewise lineal, q = 0.998).
* **Tipo de análisis**: Búsqueda de semillas fraccionarias, continuación homotópica y simulación con integrador fraccionario.
* **Estado**: Estable (Recomendado).
* **Comando**: `hidden-attractors run -p chua_fractional`
* **Salida esperada**: Trayectorias localizadas y datos de continuación fraccionaria.
* **Genera figuras**: Sí, bajo `library_figures/`.
* **Recomendado para nuevas usuarias**: Sí.

### 3. `chua_arctan`
* **Propósito**: Localizar atractores ocultos con no-linealidad suave (tipo arcotangente).
* **Sistema**: `chua-arctan` (de orden fraccionario, q = 0.95).
* **Tipo de análisis**: Lure DF con no-linealidad arcotangente y verificación.
* **Estado**: Estable (Recomendado).
* **Comando**: `hidden-attractors run -p chua_arctan`
* **Salida esperada**: Datos de aproximación analítica y continuación en no-linealidad suave.
* **Genera figuras**: Sí, bajo `library_figures/`.
* **Recomendado para nuevas usuarias**: Sí.

### 4. `chua_bifurcation`
* **Propósito**: Generar diagramas de bifurcación de orden fraccionario variando parámetros clave del sistema.
* **Sistema**: `chua_fractional_saturation` (q = 0.998).
* **Tipo de análisis**: Análisis dinámico de bifurcaciones mediante barrido de parámetros.
* **Estado**: Estable.
* **Comando**: `hidden-attractors run -p chua_bifurcation`
* **Salida esperada**: Archivo CSV de extremos locales detectados y diagrama de bifurcación.
* **Genera figuras**: Sí.
* **Recomendado para nuevas usuarias**: No (requiere más tiempo de cómputo y conocimientos especializados).

### 5. `chua_basin`
* **Propósito**: Computar y clasificar las cuencas de atracción locales de los atractores detectados.
* **Sistema**: `chua_fractional_saturation` (q = 0.998).
* **Tipo de análisis**: Mapeo 2D de secciones de cuencas de atracción.
* **Estado**: Estable / Avanzado.
* **Comando**: `hidden-attractors run -p chua_basin`
* **Salida esperada**: Matriz de clasificaciones de condiciones iniciales (convergencia a equilibrio, atractor o divergencia).
* **Genera figuras**: Sí.
* **Recomendado para nuevas usuarias**: No (cómputo masivo muy demandante).

---

## C. Workflows Especializados

Los workflows especializados no son ejemplos independientes ni metodologías alternativas. Son interfaces de bajo nivel usadas por el pipeline oficial, pruebas de validación o análisis avanzados.

> [!WARNING]
> Uso avanzado. No es el punto de entrada recomendado para una primera ejecución.

* **`hidden-attractors-protocol` (Protocolo)**: Interfaz CLI para la ejecución paso a paso de las etapas formales del protocolo (generación de semillas, precheck, continuación, filtrado, referencia y diagnóstico).
* **`hidden-attractors-robustness-overlay` (Robustez)**: Workflow para ejecutar análisis de robustez numérica variando parámetros de discretización.
* **`hidden-attractors-refined-basin` (Cuencas)**: Pipeline avanzado para el refinamiento de bordes de cuencas de atracción.
* **`hidden-attractors-sphere-controls` y `hidden-attractors-danca-abm-sphere-controls` (Esferas alrededor de equilibrios)**: Pruebas locales de ocultedad muestreando vecindades esféricas alrededor de los equilibrios (mecanismo estándar y modo de comparación Danca).
* **`hidden-attractors-strict-target-refinement` (Refinamiento)**: Rutina de búsqueda y aproximación fina del atractor objetivo.
* **`hidden-attractors-fractional-report-run` (Reporte)**: Orquestador avanzado para la compilación automática de reportes de ejecución y galería de figuras.
* **`hidden-attractors-check-validation` (Validación)**: CLI para ejecutar aserciones de consistencia matemática y consistencia física de la librería.

---

## D. Auxiliares Internos

Estas herramientas y comandos de utilidad no deben aparecer como comandos recomendados para usuarias nuevas. Son utilizados internamente para labores de desarrollo, depuración y consistencia.

* **`hidden-attractors-list-candidates`**: Utilidad para inspeccionar los metadatos de los candidatos a atractores ocultos registrados.
* **`hidden-attractors-systems`**: Herramienta de inspección para listar los sistemas dinámicos disponibles en el registro central.
* **`hidden-attractors-workflow-requirements`**: Script de diagnóstico para validar dependencias y capacidades numéricas del entorno local.
* **`hidden-attractors-check-validation`**: Ejecución de contratos de validación contra el modelo de datos.

---

## E. Legacy / Archivado

Estos directorios contienen código congelado o histórico que ha sido retirado de la ruta activa de desarrollo para garantizar la reproducibilidad y evitar confusión metodológica. No son alternativas activas ni ejecutables válidos para simulación.

* **`_archived_figure_scripts/`**: Repositorio central de scripts de figuras legacy y código obsoleto modularizado.
* **`_archived_figure_scripts/reference_scripts/`**: Scripts históricos anteriormente ubicados en la raíz `_reference_scripts/` (eliminada).
* **`version_2/tools/legacy/`**: Fuentes y herramientas históricas congeladas, expuestas solo como comandos de compatibilidad legacy si es necesario.
