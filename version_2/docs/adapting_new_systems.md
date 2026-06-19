# Adapting New Systems / Adaptación de Nuevos Sistemas

## Table of Contents / Índice de Contenidos
- [English Version](#english-version)
- [Versión en Español](#versión-en-español)

---

## English Version

# Adapting New Systems

The installable library should treat Chua and Danca as examples, not as the only possible systems. New systems enter the package through two explicit layers:

1. a registered `ChaoticSystem`, which defines the mathematical model;
2. a `WorkflowInputSpec`, which defines the numerical experiment.

This separation is intentional. A vector field is not enough to claim that equilibrium-ball controls, basin cuts, strict refinement, continuation, Lyapunov estimates, or hiddenness checks are meaningful. Each workflow must record the extra ingredients it needs.

---

## Where To Add A System

For built-in systems, add the system definition in:

```text
hidden_attractors/systems/builtins.py
```

For external or project-specific systems, register the system from a user package or script:

```python
from hidden_attractors.systems import ChaoticSystem, register_system

register_system(ChaoticSystem(...), replace=True)
```

Every new system should provide:

- `name`: stable lowercase identifier, for example `lorenz63` or `fractional-rossler`;
- `dimension`: state dimension;
- `rhs(state, parameters)`: vector field for the integer system or the right side of the Caputo equation;
- `parameters`: numerical parameter set used by default;
- `equilibria(parameters)`: named equilibria when they exist;
- `jacobian(state, parameters)`: analytic Jacobian when stability, Matignon, or fast Lyapunov diagnostics are needed;
- `lure`: manual Lur'e split only when Nyquist/describing-function workflows are requested.
- `equilibria(parameters)`: equilibrios nombrados cuando existen;

The library must not infer equilibria, Lur'e form, fractional order, memory policy, or basin targets silently.

---

## Workflow Input Spec

Reusable CLI commands and migrated legacy scripts should accept or build a `WorkflowInputSpec` from:

```text
hidden_attractors/workflows/specs.py
```

The spec records the experiment-level inputs:

- `IntegratorSpec`: solver implementation, order kind, `q`, step size `h`, horizon, burn-in, memory policy, and output columns;
- `DestinationClassifierSpec`: finite-time labels and thresholds for target, infinity, equilibrium, and unknown outcomes;
- `TargetReferenceSpec`: candidate attractor seed, recorded reference trajectory, symmetry rule, and target definition;
- `SphereControlSpec`: compatibility type name for equilibrium-centered ball samples, radii, and sample growth per radius;
- `BasinSliceSpec`: plane/grid definition and fixed coordinates;
- `StrictRefinementSpec`: trajectory-similarity thresholds and negative control policy;
- `TrajectoryDiagnosticsSpec`: retained tail window, observables, spectra, sections, and metric policy;
- `ParameterSweepSpec`: bifurcation/sweep parameter, values or range, seed policy, and plotted observable;
- `RobustnessCaseSpec`: allowed numerical or parameter perturbations.

The spec is a reproducibility contract, not a theorem. Passing validation means that the numerical run is auditable; it does not prove hiddenness.

---

## Requirements By Workflow

Inspect requirements from the command line:

```bash
hidden-attractors inspect workflow-requirements
hidden-attractors inspect workflow-requirements --workflow sphere-controls
hidden-attractors inspect workflow-requirements --workflow strict-refinement --system chua-nonsmooth
hidden-attractors inspect workflow-requirements --example-spec
```

The same information is available in Python:

```python
from hidden_attractors.systems import get_system, requirements_for, check_system_capability

system = get_system("chua-nonsmooth")
for item in requirements_for("sphere-controls"):
    print(item.key, item.add_where)
print(check_system_capability(system, "sphere-controls").as_lines())
```

Package-level capability checks only inspect hooks on `ChaoticSystem`. Integrator, target-reference, basin-slice, and refinement thresholds live in `WorkflowInputSpec`, so they are reported as missing until a workflow spec is provided.

---

## CLI Policy

New maintained CLI commands should follow this pattern:

1. accept `--system` for registered systems when the workflow is generic;
2. accept `--spec path/to/workflow_spec.json` when the run needs explicit solver, basin, target, or threshold configuration;
3. write the effective spec next to outputs as JSON;
4. write a stage summary using only the official envelope and verdict labels from `hidden_attractors.workflows.protocol`;
5. use the same spec pattern for robustness, bifurcation, Lyapunov, continuation, and diagnostics, not only basins/refinement/spheres;
6. avoid environment-only configuration except as a compatibility layer.
7. 

Chua/Danca-specific CLIs may remain while their numerical details are being migrated, but new reusable logic should live under `hidden_attractors/`, not inside `tools/legacy/`.

---

## Legacy Policy

Legacy adapters may keep installed command names for reproducibility. They may not publish new runs with historical methodology labels. When adding behavior to a retained adapter:

1. move reusable mathematical or numerical logic into `hidden_attractors/`;
2. make the legacy script a thin wrapper around that package function;
3. build or load a `WorkflowInputSpec` and save it in the output directory;
4. declare any fixed-system assumption in the help text and output metadata;
5. route new summaries through the official protocol contract.
6. 

This keeps old artifacts reproducible while preventing the library API from becoming a collection of one-off scripts.

---

## Minimum Inputs For Hiddenness Evidence

For an integer-order system:

- vector field and parameter set;
- equilibria and Jacobian;
- integrator contract with `order_kind="integer"`;
- candidate reference trajectory or seed;
- equilibrium-neighborhood controls;
- basin or alternative destination classifier;
- strict refinement thresholds if unresolved cells are revisited.

For a fractional Caputo system:

- all integer-system inputs above;
- fractional order `q`;
- solver type and memory policy: full history, finite memory, or external;
- memory length if finite memory is used;
- burn-in and post-transient sampling windows;
- documentation of any Weyl/Liouville-Weyl seed approximation before Caputo validation.
- orden fraccionario `q`;

For describing-function or Nyquist routes:

- manual Lur'e split;
- branch convention for `(j omega)^q`;
- scalar nonlinearity and describing function;
- seed interpretation as heuristic or Weyl-asymptotic, followed by Caputo or integer-system validation.

---

## Existing System-Specific Workflows

The current strict refinement and Danca ABM command names remain available because they encode published or recorded numerical comparisons. Official hiddenness evidence nevertheless uses interior ball samples:

```bash
hidden-attractors hiddenness strict-target-refinement --help
hidden-attractors published danca-abm-sphere-controls --help
```

They are compatibility adapters, not competing methodologies. New runs should enter through `hidden-attractors protocol` and dispatch to the registered system, solver backend, classifier, and dynamic reference.

---

## Versión en Español

# Adaptación de Nuevos Sistemas

La biblioteca instalable debe tratar a Chua y Danca como ejemplos, no como los únicos sistemas posibles. Los nuevos sistemas entran en el paquete a través de dos capas explícitas:

1. un `ChaoticSystem` registrado, que define el modelo matemático;
2. una `WorkflowInputSpec`, que define el experimento numérico.

Esta separación es intencional. Un campo vectorial no es suficiente para afirmar que los controles de bolas de equilibrio, los cortes de cuencas, el refinamiento estricto, la continuación, las estimaciones de Lyapunov o los controles de ocultedad son significativos. Cada flujo de trabajo debe registrar los ingredientes adicionales que necesita.

---

## Dónde Agregar un Sistema

Para sistemas integrados, agregue la definición del sistema en:

```text
hidden_attractors/systems/builtins.py
```

Para sistemas externos o específicos del proyecto, registre el sistema desde un paquete o script de usuario:

```python
from hidden_attractors.systems import ChaoticSystem, register_system

register_system(ChaoticSystem(...), replace=True)
```

Cada nuevo sistema debe proporcionar:

- `dimension`: state dimension;
- `equilibria(parameters)`: named equilibria when they exist;
- `name`: identificador estable en minúsculas, por ejemplo `lorenz63` o `fractional-rossler`;
- `dimension`: dimensión de estado;
- `rhs(state, parameters)`: campo vectorial para el sistema entero o el lado derecho de la ecuación de Caputo;
- `parameters`: conjunto de parámetros numéricos utilizados por defecto;
- `equilibria(parameters)`: equilibrios nombrados cuando existen;
- `jacobian(state, parameters)`: Jacobiano analítico cuando se necesitan diagnósticos de estabilidad, Matignon o Lyapunov rápidos;
- `lure`: división manual de Lur'e solo cuando se solicitan flujos de trabajo de Nyquist/función descriptiva.

La biblioteca no debe inferir silenciosamente equilibrios, forma de Lur'e, orden fraccionario, política de memoria u objetivos de cuenca.

---

## Especificación de Entrada de Flujo de Trabajo

Los comandos CLI reutilizables y los scripts heredados migrados deben aceptar o compilar una `WorkflowInputSpec` desde:

```text
hidden_attractors/workflows/specs.py
```

La especificación registra las entradas a nivel de experimento:

- `IntegratorSpec`: implementación del resolvedor, tipo de orden, `q`, tamaño de paso `h`, horizonte, tiempo de transitorio (burn-in), política de memoria y columnas de salida;
- `DestinationClassifierSpec`: etiquetas y umbrales de tiempo finito para resultados de objetivo (target), infinito, equilibrio y desconocido;
- `TargetReferenceSpec`: semilla del candidato a atractor, trayectoria de referencia registrada, regla de simetría y definición del objetivo;
- `SphereControlSpec`: nombre de tipo de compatibilidad para muestras de bolas centradas en el equilibrio, radios y crecimiento de muestras por radio;
- `BasinSliceSpec`: definición de plano/rejilla y coordenadas fijas;
- `StrictRefinementSpec`: umbrales de similitud de trayectoria y política de control negativo;
- `TrajectoryDiagnosticsSpec`: ventana de cola retenida, observables, espectros, secciones y política métrica;
- `ParameterSweepSpec`: parámetro de barrido/bifurcación, valores o rango, política de semilla y observable graficado;
- `RobustnessCaseSpec`: perturbaciones numéricas o de parámetros permitidas.

La especificación es un contrato de reproducibilidad, no un teorema. Aprobar la validación significa que la ejecución numérica es auditable; no prueba la ocultedad.

---

## Requisitos por Flujo de Trabajo

Inspeccione los requisitos desde la línea de comandos:

```bash
hidden-attractors inspect workflow-requirements
hidden-attractors inspect workflow-requirements --workflow sphere-controls
hidden-attractors inspect workflow-requirements --workflow strict-refinement --system chua-nonsmooth
hidden-attractors inspect workflow-requirements --example-spec
```

La misma información está disponible en Python:

```python
from hidden_attractors.systems import get_system, requirements_for, check_system_capability

system = get_system("chua-nonsmooth")
for item in requirements_for("sphere-controls"):
    print(item.key, item.add_where)
print(check_system_capability(system, "sphere-controls").as_lines())
```

Los controles de capacidad a nivel de paquete solo inspeccionan los ganchos (hooks) en `ChaoticSystem`. Los umbrales de integrador, referencia de objetivo, corte de cuenca y refinamiento viven en `WorkflowInputSpec`, por lo que se reportan como faltantes hasta que se proporcione una especificación de flujo de trabajo.

---

## Política de CLI

Los nuevos comandos CLI mantenidos deben seguir este patrón:

7. 
1. aceptar `--system` para sistemas registrados cuando el flujo de trabajo es genérico;
2. aceptar `--spec ruta/al/workflow_spec.json` cuando la ejecución necesita una configuración explícita de resolvedor, cuenca, objetivo o umbrales;
3. escribir la especificación efectiva junto a las salidas como JSON;
4. escribir un resumen de etapa utilizando únicamente las etiquetas de veredicto y envoltura oficiales de `hidden_attractors.workflows.protocol`;
5. usar el mismo patrón de especificación para robustez, bifurcación, Lyapunov, continuación y diagnósticos, no solo para cuencas/refinamiento/esferas;
6. evitar la configuración solo a nivel de entorno, excepto como capa de compatibilidad.

Las CLI específicas de Chua/Danca pueden permanecer mientras se migran sus detalles numéricos, pero la nueva lógica reutilizable debe vivir bajo `hidden_attractors/`, no dentro de `tools/legacy/`.

---

## Política de Legacy

Los adaptadores de legacy pueden mantener los nombres de comando instalados para fines de reproducibilidad. No pueden publicar nuevas ejecuciones con etiquetas metodológicas históricas. Al agregar comportamiento a un adaptador retenido:

6. 
1. trasladar la lógica matemática o numérica reutilizable a `hidden_attractors/`;
2. hacer que el script heredado sea un envoltorio delgado del de esa función del paquete;
3. compilar o cargar una `WorkflowInputSpec` y guardarla en el directorio de salida;
4. declarar cualquier suposición de sistema fijo en el texto de ayuda y en los metadatos de salida;
5. enrutar los nuevos resúmenes a través del contrato oficial del protocolo.

Esto mantiene reproducibles los artefactos antiguos y evita que la API de la biblioteca se convierta en una colección de scripts únicos.

---

## Entradas Mínimas para Evidencia de Ocultedad

Para un sistema de orden entero:

- campo vectorial y conjunto de parámetros;
- equilibrios y Jacobiano;
- contrato de integrador con `order_kind="integer"`;
- trayectoria de referencia candidata o semilla;
- controles de vecindad del equilibrio;
- cuenca o clasificador de destino alternativo;
- umbrales de refinamiento estricto si se vuelven a visitar celdas no resueltas.

Para un sistema fraccionario de Caputo:

- fractional order `q`;
- todas las entradas del sistema entero mencionadas anteriormente;
- orden fraccionario `q`;
- tipo de resolvedor y política de memoria: historial completo (full history), memoria finita (finite memory) o externo;
- longitud de memoria si se utiliza memoria finita;
- ventanas de muestreo post-transitorio y burn-in;
- documentación de cualquier aproximación de semilla de Weyl/Liouville-Weyl antes de la validación de Caputo.

Para rutas de Nyquist o función descriptiva:

- manual Lur'e split;
- división manual de Lur'e;
- convención de rama para `(j omega)^q`;
- no-linealidad escalar y función descriptiva;
- interpretación de semillas como heurísticas o asintóticas de Weyl, seguidas de validación de Caputo o de sistema entero.

---

## Flujos de Trabajo Existentes Específicos del Sistema

Los nombres actuales de los comandos de refinamiento estricto y Danca ABM siguen estando disponibles porque codifican comparaciones numéricas publicadas o registradas. Sin embargo, la evidencia oficial de ocultedad utiliza muestras de bolas interiores:

```bash
hidden-attractors hiddenness strict-target-refinement --help
hidden-attractors published danca-abm-sphere-controls --help
```

Son adaptadores de compatibilidad, no metodologías competidoras. Las nuevas ejecuciones deben ingresar a través de `hidden-attractors protocol` y enviarse al sistema registrado, resolvedor backend, clasificador y referencia dinámica.