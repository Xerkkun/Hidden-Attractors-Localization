# Examples / Ejemplos

Examples are intentionally small and import from `hidden_attractors`.
Los ejemplos son intencionalmente pequeños e importan desde `hidden_attractors`.

## Quick Equilibria Check / Verificación Rápida de Equilibrios

```bash
python examples/quickstart_equilibria.py
```

Purpose: verify that the non-smooth Chua equilibria are zeros of the vector field.
Propósito: verificar que los equilibrios no suaves de Chua sean ceros del campo vectorial.

## List Final Candidates / Listado de Candidatos Finales

```bash
python examples/list_final_candidates.py
```

Purpose: load the reference candidates from `outputs/` using the public candidate API.
Propósito: cargar los candidatos de referencia desde `outputs/` usando la API pública de candidatos.

## Minimal Chua Protocol / Protocolo Chua Mínimo

```bash
python examples/minimal_chua_protocol.py
python examples/minimal_chua_protocol.py --run
```

Purpose: create a small, explicit contract for the unified fractional Chua workflow, keeping heavy numerical stages delegated to the packaged C-backed entry point. The default command only writes the JSON contract and shell command; `--run` launches the workflow.
Propósito: crear un contrato pequeño y explícito para el flujo de trabajo fraccionario unificado de Chua, manteniendo las etapas numéricas pesadas delegadas en el punto de entrada empaquetado respaldado por C. El comando predeterminado solo escribe el contrato JSON y el comando de shell; `--run` inicia el flujo de trabajo.

## Custom System Definition / Definición de Sistema Personalizado

```bash
python examples/custom_system_definition.py
hidden-attractors inspect systems
```

Purpose: show how a user registers a new chaotic system through the public `ChaoticSystem` contract.
Propósito: mostrar cómo un usuario registra un nuevo sistema caótico a través del contrato público `ChaoticSystem`.

## Create a Robustness Overlay Config / Creación de una Configuración de Capa de Robustez

```bash
python examples/create_robustness_overlay_config.py
```

Purpose: write a workflow configuration without launching long simulations.
Propósito: escribir una configuración de flujo de trabajo sin iniciar simulaciones largas.

## Aggregate Existing Robustness Output / Agregación de Resultados de Robustez Existentes

```bash
python examples/aggregate_existing_robustness_overlay.py outputs/robustness_overlay_c_trajectories_20260517
```

Purpose: regenerate summary tables and plots from an existing output folder.
Propósito: regenerar tablas de resumen y gráficos a partir de una carpeta de salida existente.

## Dynamical Analysis Gallery / Galería de Análisis Dinámico

```bash
python examples/dynamical_analysis_gallery.py
```

Purpose: generate phase-space, phase-projection, time-series, and post-processed bifurcation figures using the public API.
Propósito: generar figuras de espacio de fase, proyección de fase, series temporales y bifurcación posprocesada usando la API pública.

With an existing project trajectory:
Con una trayectoria de proyecto existente:

```bash
python examples/dynamical_analysis_gallery.py --trajectory-csv <trajectory_csv>
```

The example writes figures and tabular bifurcation points under `outputs/examples/dynamical_analysis_gallery/`.
El ejemplo escribe figuras y puntos de bifurcación tabulares bajo `outputs/examples/dynamical_analysis_gallery/`.

## Adding Examples / Adición de Ejemplos

New examples should:
Los nuevos ejemplos deben:

- import from `hidden_attractors`;
- register new systems through `hidden_attractors.systems` when they introduce a model not already built in;
- avoid duplicating workflow internals;
- write to a new folder under `outputs/` or require `--output-dir`;
- document whether they launch long numerical jobs.
- 
- importar desde `hidden_attractors`;
- registrar nuevos sistemas a través de `hidden_attractors.systems` cuando introduzcan un modelo que no esté integrado de forma nativa;
- evitar duplicar aspectos internos de los flujos de trabajo;
- escribir en una nueva carpeta bajo `outputs/` o requerir `--output-dir`;
- documentar si inician trabajos numéricos largos.
