# Getting Started with Hidden Attractors Localization / Primeros Pasos con la Localización de Atractores Ocultos

`hidden-attractors-fo` provides two main interfaces: a high-level command-line interface (CLI) for running standard pre-packaged workflows, and a low-level Python API for custom scripting and custom integrations.

`hidden-attractors-fo` ofrece dos interfaces principales: una interfaz de línea de comandos (CLI) de alto nivel para ejecutar flujos de trabajo predefinidos estándar y una API de Python de bajo nivel para secuencias de comandos personalizadas e integraciones personalizadas.

---

## 1. High-Level Interface / Interfaz de Alto Nivel

The CLI offers a simple entry point to run presets, initialize custom folders, and inspect configuration trees.

La CLI ofrece un punto de entrada sencillo para ejecutar presets, inicializar carpetas personalizadas e inspeccionar árboles de configuración.

### Running Presets / Ejecución de Presets

To run a built-in workflow preset:
Para ejecutar un preset de flujo de trabajo integrado:

```bash
hidden-attractors run -p chua_integer
hidden-attractors run -p chua_fractional
hidden-attractors run -p chua_arctan_only_fractional
```

### Initializing Examples / Inicialización de Ejemplos

To copy a pre-packaged template config to your local working directory:
Para copiar una configuración de plantilla preempaquetada en su directorio de trabajo local:

```bash
hidden-attractors init -e chua_fractional
```

### Inspecting Configurations / Inspección de Configuraciones

To preview the fully normalized effective configuration mapping (including defaults) before execution:
Para previsualizar el mapeo de configuración efectivo completamente normalizado (incluidos los valores predeterminados) antes de la ejecución:

```bash
hidden-attractors inspect-config -p chua_fractional
```

To run a custom YAML file directly:
Para ejecutar un archivo YAML personalizado directamente:

```bash
hidden-attractors run -c path/to/config.yaml
```

The primary stable user-facing CLI is `hidden-attractors`. Specialized workflows (such as protocol stages, robustness overlays, basin mappings, and report generation) are accessible as subcommands under this unified interface. For a detailed list and migration guide from legacy standalone commands, see [CLI Migration Guide](cli_migration_legacy_entrypoints.md).

La CLI principal y estable orientada al usuario es `hidden-attractors`. Los flujos de trabajo especializados (como las etapas del protocolo, las capas de robustez, los mapeos de cuencas y la generación de reportes) son accesibles como subcomandos bajo esta interfaz unificada. Para obtener una lista detallada y una guía de migración desde los comandos independientes heredados, consulte la [Guía de Migración de CLI](cli_migration_legacy_entrypoints.md).

---

## 2. Low-Level Python API / API de Python de Bajo Nivel

For custom scripting, you can load configurations, fetch systems, and run integrations programmatically.

Para secuencias de comandos personalizadas, puede cargar configuraciones, recuperar sistemas y ejecutar integraciones mediante programación.

### Loading Configuration / Carga de Configuración

```python
from hidden_attractors.workflows.config_loader import load_config

# Load, normalize, and validate a configuration YAML
config = load_config("configs/examples/chua_fractional_centered_lure_df.yaml")
```

### Retrieving Systems / Recuperación de Sistemas

```python
from hidden_attractors.systems import get_system

# Get a system definition from the registry (e.g., Chua's system with arctan nonlinearity)
system = get_system("chua-arctan")
```

### Performing Integrations / Realización de Integraciones

```python
from hidden_attractors.integrations.selector import integrate

# Run a numerical integration using the unified selector
times, states, status = integrate(
    rhs=system.rhs,
    x0=[0.1, 0.0, 0.0],
    q=0.99,
    h=0.01,
    t_final=50.0,
    integrator="efork3",
    system=system
)
```

---

## 3. Configuration Guide / Guía de Configuración

### Activating/Deactivating Stages / Activación y Desactivación de Etapas

You can control which parts of the workflow run using flags in the `stages` section of your configuration YAML:
Puede controlar qué partes del flujo de trabajo se ejecutan mediante indicadores en la sección `stages` de su YAML de configuración:

```yaml
stages:
  seed_search: true
  continuation: true
  final_simulation: true
  hiddenness_tests: false
  basin_slices: false
  bifurcation: false
  attractor_only: false
```

### Changing the Integrator / Cambio del Integrador

To change the numerical integrator, edit the `integrator.name` parameter. The selector validates compatibility with the fractional order `q` automatically:
Para cambiar el integrador numérico, edite el parámetro `integrator.name`. El selector valida automáticamente la compatibilidad con el orden fraccionario `q`:

```yaml
integrator:
  name: rk4  # allowed for q = 1.0; use 'abm' or 'efork3' for q < 1.0
  h: 0.01
```

### Activating Memory Window / Activación de la Ventana de Memoria

Windowed memory is required for long fractional-order simulations where full Caputo history is computationally prohibitive. Configure it under the `integrator` section:
Se requiere memoria con ventana para simulaciones largas de orden fraccionario donde el historial completo de Caputo es computacionalmente prohibitivo. Configúrelo bajo la sección `integrator`:

```yaml
integrator:
  memory_mode: window        # Options: full, window, none
  memory_policy: finite_window
  memory_window_steps: 4000  # Number of history steps to keep
```

You can also specify `memory_window_time` (e.g. `2.0` seconds), which will be converted to steps using `memory_window_steps = round(memory_window_time / h)`.
También puede especificar `memory_window_time` (por ejemplo, `2.0` segundos), que se convertirá a pasos usando `memory_window_steps = round(memory_window_time / h)`.

### Running Specific Workflows / Ejecución de Flujos de Trabajo Específicos

#### Running Attractor Only / Ejecución de Solo Atractor
Set `run_attractor_only` (or `stages.attractor_only`) to `true` to integrate the system from an initial condition without running seed search or continuation.
Establezca `run_attractor_only` (o `stages.attractor_only`) en `true` para integrar el sistema desde una condición inicial sin ejecutar la búsqueda de semillas o la continuación.

#### Running Bifurcation Sweeps / Ejecución de Barridos de Bifurcación
Configure the `bifurcation` section in the YAML, then run the bifurcation workflow. You can sweep parameters like `beta` or the fractional order `q`:
Configure la sección `bifurcation` en el YAML, luego ejecute el flujo de trabajo de bifurcación. Puede realizar barridos de parámetros como `beta` o el orden fraccionario `q`:

```yaml
stages:
  bifurcation: true
bifurcation:
  parameter: q
  values:
    min: 0.95
    max: 1.0
    n: 50
```

#### Running Basin Slices / Ejecución de Cortes de Cuencas
Generate 2D slices of the basins of attraction by enabling `basin_slices` (or `stages.basin_slices`):
Genere cortes en 2D de las cuencas de atracción habilitando `basin_slices` (o `stages.basin_slices`):

```yaml
stages:
  basin_slices: true
basin:
  planes: ["xy", "xz"]
  grid_n: 100
  around_equilibria: true
```

---

## 4. Methodological Scope & Directories / Alcance Metodológico y Directorios

> [!IMPORTANT]
> **Heuristics vs Proof**: Describing Function (DF), Lur'e decomposition, and Machado-family analysis only construct **candidate seeds** for starting continuation. They are heuristics and **do not prove** that an attractor is mathematically hidden. Proof of localization requires exhaustive neighborhood probing and basin slice tests.
>
> **Heurística frente a Prueba**: La función descriptiva (DF), la descomposición de Lur'e y el análisis de la familia de Machado solo construyen **semillas candidatas** para comenzar la continuación. Son heurísticas y **no prueban** que un atractor esté matemáticamente oculto. La prueba de localización requiere un sondeo exhaustivo de vecindades y pruebas de cortes de cuencas.

> [!NOTE]
> **Outputs Directory**: Ordinary execution results, plots, summaries, and data files are saved under `outputs/`.
> The `validation/` directory is reserved exclusively for promoted validation evidence and final benchmark records, never for routine output files.
>
> **Directorio de Salidas**: Los resultados ordinarios de la ejecución, los gráficos, los resúmenes y los archivos de datos se guardan bajo `outputs/`. El directorio `validation/` está reservado exclusivamente para la evidencia de validación promocionada y los registros de referencia finales, nunca para archivos de salida rutinarios.
