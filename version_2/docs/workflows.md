# Workflows — Official Guide / Flujos de Trabajo (Workflows) — Guía Oficial

## Table of Contents / Índice de Contenidos

- [English Version](#english-version)
- [Versión en Español](#versión-en-español)

---

## English Version

Consult the `THESIS_CLAIMS.md` to see the current classification of results and defensible claims.

This document details the library's workflows for `hidden-attractors-fo`.

---

## 1. Recommended Route for New Users

If this is your first time interacting with this repository, the recommended learning and execution path is as follows:

1. **Installation**: Install the library in editable mode with `pip install -e version_2`.
2. **Official Example**: Run the quick smoke check of Example 1 to verify the pipeline:

   ```bash
   cd version_2/examples/chua_nonsmooth_biased_hidden_attractor
   python run_example.py --quick
   ```

3. **Explore Presets**: Run the unified CLI command to see a stable preset:

   ```bash
   hidden-attractors run -p chua_fractional
   ```

4. **Read Guides**: Consult the [Quick Start Guide](quick_start.md) to understand where outputs are saved and what rules to follow.

---

## 2. Official Example

**Example 1 — Non-Smooth Fractional Chua with Biased Describing Function** is the reference case for searching for candidates compatible with hiddenness. Its official execution entry point is:

- **File**: [run_example.py](../examples/chua_nonsmooth_biased_hidden_attractor/run_example.py)
- **Internal logic**: Implemented cleanly in the core library (`hidden_attractors/workflows/biased_chua.py`).

> [!NOTE]
> **Scientific and Reproducibility Warning:** This example **is not a reproduction of Danca's (2017) system**.
> Danca's original system **was not reproducible due to lack of published details** (such as exact coordinates of the hidden attractor initial conditions, DF solver parameters, and numerical continuation details). Consequently, this example performs a systematic candidate search in a parameter sweep to identify neighborhoods compatible with hiddenness.
>

This example sequentially executes the following phases:

1. **Step 1**:Centered reference search (centered DF, $c=0$).
2. **Step 2**:Affine homotopy search with biased describing function (BDF, $c \neq 0$).
3. **Step 3**:Standard hiddenness verification via local sphere sweeping.
4. **Step 4**:Parallel extended search for volumetric hiddenness.
5. **Step 5**:Summary and figure export to the centralized gallery `library_figures/` according to the [Figure Export Policy](figure_export_policy.md).

---

## 3. CLI Presets and Unified Commands

CLI presets are pre-packaged configurations ready to run with the `hidden-attractors` command. You can consult the complete list and details of each (system, stability, purpose, etc.) in the [Index of Examples and Workflows](examples_index.md).

Usage examples:

```bash
# Find seeds using centered Lur'e DF
hidden-attractors seed lure-centered -p chua_fractional

# Find seeds using biased Lur'e DF
hidden-attractors seed lure-biased -p chua_fractional

# Run scalar continuation
hidden-attractors continuation run -c path/to/config.yaml -s outputs/seeds.csv

# Run multiparameter continuation
hidden-attractors continuation multiparameter -c path/to/config.yaml

# Run standard Fractional Chua preset
hidden-attractors run -p chua_fractional

# Run Chua with arctan nonlinearity preset
hidden-attractors run -p chua_arctan

# Run a bifurcation sweep
hidden-attractors bifurcation run -p chua_bifurcation

# Run Lyapunov exponent estimates
hidden-attractors lyapunov compute -c configs/examples/chua_fractional_lyapunov.yaml

# Run the 0-1 test for chaos
hidden-attractors chaos-test zero-one -c configs/examples/chua_fractional_zero_one.yaml
```

---

## 4. Specialized Workflows

Specialized workflows are not standalone examples or alternative methodologies. They are low-level interfaces used by the official pipeline, validation tests, or advanced analysis.

> [!WARNING]
> Specialized workflows are not standalone examples or alternative methodologies. They are low-level interfaces used by the official pipeline, validation tests, or advanced analysis.

### Specialized Workflows Commands

- **`hidden-attractors protocol`**: Step-by-step sequential execution of the official protocol (seed generation, continuation, validation, etc.).
- **`hidden-attractors robustness overlay`**: Numerical robustness analysis varying step sizes and solver conditions.
- **`hidden-attractors basin refined`**: Fine refinement of local basin of attraction boundaries.
- **`hidden-attractors hiddenness sphere-controls`**: Hiddenness checking via spherical neighborhood sampling around equilibrium points.
- **`hidden-attractors basin strict-target-refinement`**: Numerical refinement of the localized attractor.
- **`hidden-attractors report fractional-run`**: Advanced orchestrator for automatic compilation of execution reports and figure galleries.
- **`hidden-attractors validate contract`**: Validation checkers for internal consistency.

---

## 5. Programmatic API (Python)

To build custom workflows, the library exposes clear modules:

```python
from hidden_attractors.workflows.config_loader import load_config
from hidden_attractors.systems import get_system
from hidden_attractors.integrations.selector import integrate

# Load configuration from a YAML file
config = load_config("configs/examples/chua_fractional_centered_lure_df.yaml")

# Retrieve a system definition
system = get_system("chua-nonsmooth")

# Numerically integrate using the unified solver selector
times, states, status = integrate(
    rhs=system.rhs,
    x0=[0.1, 0.0, 0.0],
    q=0.998,
    h=0.001,
    t_final=100.0,
    integrator="efork3",
    system=system
)
```

---

## 6. Legacy and Historical Archive

Historical migration scripts are intentionally excluded from the active repository.
The active implementation lives in `version_2/hidden_attractors/`.

- **Legacy Tools**:Retained under `version_2/tools/legacy/` only for backward compatibility in specific C solvers.

---

## Versión en Español

Consulta la `THESIS_CLAIMS.md` para ver la clasificación actual de resultados y claims defendibles.

Este documento detalla los flujos de trabajo de la biblioteca `hidden-attractors-fo`.

---

## Ruta Recomendada para Usuarias Nuevas

Si es la primera vez que interactúas con este repositorio, la ruta recomendada de aprendizaje y ejecución es la siguiente:

1. **Instalación**: Instala la librería en modo editable con `pip install -e version_2`.
2. **Ejemplo Oficial**: Ejecuta la prueba rápida del Ejemplo 1 para verificar el pipeline:

   ```bash
   cd version_2/examples/chua_nonsmooth_biased_hidden_attractor
   python run_example.py --quick
   ```

3. **Exploración de Presets**: Ejecuta el comando de CLI unificado para ver un preset estable:

   ```bash
   hidden-attractors run -p chua_fractional
   ```

4. **Lectura de Guías**: Consulta la [Guía de Inicio Rápido](quick_start.md) para comprender dónde se guardan las salidas y qué reglas seguir.

---

## Ejemplo Oficial

El **Ejemplo 1 — Chua fraccionario no suave con función descriptiva sesgada** es el caso de referencia para la búsqueda de candidatos compatibles con ocultedad. Su entrada de ejecución oficial es:

- **Archivo**: [run_example.py](../examples/chua_nonsmooth_biased_hidden_attractor/run_example.py)
- **Lógica interna**: Implementada de forma limpia en el núcleo de la librería (`hidden_attractors/workflows/biased_chua.py`).

> [!NOTE]
>
> **Advertencia Científica y de Reproducibilidad:** Este ejemplo **no es una reproducción del sistema de Danca (2017)**.
> El sistema original de Danca **no fue reproducible debido a la falta de información publicada** (como las coordenadas exactas de las condiciones iniciales del atractor oculto, parámetros del resolvedor DF, y el método de continuación numérica). Por consiguiente, este ejemplo realiza una búsqueda sistemática de candidatos en un sweep de parámetros para identificar vecindades compatibles con ocultedad.

Este ejemplo ejecuta de forma secuencial las siguientes fases:

1. **Paso 1**:Búsqueda centrada de referencia (DF centrada, $c=0$).
2. **Paso 2**:Búsqueda homotópica afín con función descriptiva sesgada (BDF, $c \neq 0$).
3. **Paso 3**:Verificación de ocultedad estándar mediante barrido de esferas local.
4. Paso 4** (Optional:Búsqueda extendida en paralelo de ocultedad volumétrica.
5. **Paso 5**:Resumen y exportación de figuras a la galería centralizada `library_figures/` según la [Política de Exportación de Figuras](figure_export_policy.md).

---

## Presets de CLI y Comandos Unificados

Los presets de CLI son configuraciones empaquetadas listas para ejecutar con el comando `hidden-attractors`. Puedes consultar la lista completa y detalles de cada uno (sistema, estabilidad, propósito, etc.) en el [Índice de Ejemplos y Workflows](examples_index.md).

Ejemplos de uso:

```bash
# Find seeds using centered Lur'e DF
hidden-attractors seed lure-centered -p chua_fractional

# Find seeds using biased Lur'e DF
hidden-attractors seed lure-biased -p chua_fractional

# Run scalar continuation
hidden-attractors continuation run -c path/to/config.yaml -s outputs/seeds.csv

# Run multiparameter continuation
hidden-attractors continuation multiparameter -c path/to/config.yaml

# Run standard Fractional Chua preset
hidden-attractors run -p chua_fractional

# Run Chua with arctan nonlinearity preset
hidden-attractors run -p chua_arctan

# Run a bifurcation sweep
hidden-attractors bifurcation run -p chua_bifurcation

# Run Lyapunov exponent estimates
hidden-attractors lyapunov compute -c configs/examples/chua_fractional_lyapunov.yaml

# Run the 0-1 test for chaos
hidden-attractors chaos-test zero-one -c configs/examples/chua_fractional_zero_one.yaml
```

---

## Workflows Especializados

Los workflows especializados no son ejemplos independientes ni metodologías alternativas. Son interfaces de bajo nivel usadas por el pipeline oficial, pruebas de validación o análisis avanzados.

> [!WARNING]
> Los workflows especializados no son ejemplos independientes ni metodologías alternativas. Son interfaces de bajo nivel usadas por el pipeline oficial, pruebas de validación o análisis avanzados.

### Comandos de Workflows Especializados

- **`hidden-attractors protocol`**: Ejecución secuencial y detallada del protocolo oficial (generación de semillas, continuación, validación, etc.).
- **`hidden-attractors robustness overlay`**: Análisis de robustez numérica variando tamaños de paso y condiciones del resolvedor.
- **`hidden-attractors basin refined`**: Refinamiento fino de las fronteras de cuencas de atracción locales.
- **`hidden-attractors published danca-abm-sphere-controls`**: Pruebas de ocultedad en vecindades esféricas alrededor de los puntos de equilibrio.
- **`hidden-attractors hiddenness strict-target-refinement`**: Refinamiento numérico del atractor localizado.
- **`hidden-attractors report fractional-run`**: Generador automático de reportes científicos unificados.
- **`hidden-attractors validate contract`**: Controladores de validación de la consistencia interna.

---

## API Programática (Python)

Para realizar desarrollos personalizados, la biblioteca expone módulos claros:

```python
from hidden_attractors.workflows.config_loader import load_config
from hidden_attractors.systems import get_system
from hidden_attractors.integrations.selector import integrate

# Load configuration from a YAML file
config = load_config("configs/examples/chua_fractional_centered_lure_df.yaml")

# Retrieve a system definition
system = get_system("chua-nonsmooth")

# Numerically integrate using the unified solver selector
times, states, status = integrate(
    rhs=system.rhs,
    x0=[0.1, 0.0, 0.0],
    q=0.998,
    h=0.001,
    t_final=100.0,
    integrator="efork3",
    system=system
)
```

---

## Legacy y Archivo Histórico

- **Herramientas Legacy**: Conservadas bajo `version_2/tools/legacy/` solo para compatibilidad hacia atrás en resolvedores específicos de C.
