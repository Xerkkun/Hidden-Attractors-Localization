# Experimental Non-Smooth Chua Matrix / Matriz Experimental Chua No Suave

## Table of Contents / Índice de Contenidos
- [English Version](#english-version)
- [Versión en Español](#versión-en-español)

---

## English Version

This folder is an external runner for the matrix created in `outputs/chua_nonsmooth_fractional_memory_matrix/`. It does not modify the main workflow or the closed validations; it consumes the models, seeds, integrators, diagnostics, and plotters already available in `hidden_attractors`.

### Mathematical Foundation

- `integer_like_seed` uses the classical describing function closure with $W(s) = r^T (sI-P)^{-1} b$, equivalent to $q=1$.
- `fractional_seed` uses the same classical DF with $W_q(i\omega) = r^T ((i\omega)^q I-P)^{-1}b$, principal branch $(i\omega)^q = \omega^q \exp(i q \pi/2)$.
- These seeds are asymptotic harmonic oscillations of Weyl type used to initialize numerical tests; they are not exact periodic Caputo cycles, nor do they prove hiddenness.
- The continuation `integer_like_q1` propagates only the last point.
- The continuation `fractional_caputo` transports history. In ABM, `full` mode preserves the chronological Caputo history of the entire chain in `eta` and `truncated` preserves exactly the restarted window `Lm`. In EFORK, `truncated` uses `Lm` and `full` uses a sufficiently long horizon so as not to truncate the integrated chain.
- The EFORK wrapper reports history counts but does not expose the values of its entire internal prehistory in `full` mode. For this reason, `final_history_window.npz` identifies in that case the observed available tail and marks `exact_transported_window=false`; in `truncated` mode, `stage_keep >= Lm` is required and the saved window is indeed exact.
- The external ABM backend to this experiment exposes Lur'e homotopy continuation: the final saved history is exact for both `full` and `truncated` ABM routes. For `integer_like_q1`, both ABM and EFORK label and apply `last_point_only`.

### Execution

From `version_2/`:

```bash
python experiments/chua_nonsmooth_memory_matrix/run_shared_cache_tasks.py --workers 4
python experiments/chua_nonsmooth_memory_matrix/run_continuation_tasks.py --workers 4
python experiments/chua_nonsmooth_memory_matrix/run_hiddenness_tasks.py --workers 4
python figure_scripts/chua_nonsmooth_memory_matrix_run_figure_tasks.py --workers 2
python experiments/chua_nonsmooth_memory_matrix/aggregate_results.py
```

On macOS, the portable launcher can also be run:

```bash
WORKERS=2 bash experiments/chua_nonsmooth_memory_matrix/run_matrix_mac.sh
```

`WORKERS=1` is prudent for ABM `full`: retaining the full history can consume a lot of memory and its Volterra sum grows with the horizon.

Each process forces `OMP_NUM_THREADS=1`, `OMP_THREAD_LIMIT=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`. Task tables are validated to prevent two workers from writing to the same `cache_key`.

Continuations use by default, at each value of `eta`, a transient segment and an observed segment of length `t_burn`; they can be explicitly declared with `--stage-transient` and `--stage-keep`, and both values are recorded in the metadata.

For short wiring smoke tests, `run_hiddenness_tasks.py` accepts `--max-probes-per-cloud N`. This reduction never promotes hiddenness: if there are no contacts, the result remains `inconclusive`.

### Results

All results are written to the root of the matrix outputs and include `status.json` and metadata for commit, date, numerical contract, `q`, `h`, `t_final`, `t_burn`, `Lm`, integrator, memory policy, `exp_id`, and `cache_key`. A task with `status="ok"` is reused when relaunched; `--force` orders it to be recalculated.

The only admissible hiddenness labels are:

- `compatible_with_hiddenness_under_tested_radii`
- `not_hidden_under_tested_radii`
- `inconclusive`
- `numerical_failure`

An absence of contacts from the tested balls is limited evidence under the documented numerical contract, not a global proof of a hidden attractor. If the candidate trajectory is classified as periodic in the post-transient control, the hiddenness verdict remains `inconclusive`.

In addition to `phase3d_candidate` and `time_series` aggregated per experiment, each candidate configuration integrated after continuation is saved under `figures/<exp_id>/candidate_attractors/`, with names identifying the continuation solver, target integrator, memory policy, and sign.

---

## Versión en Español

Esta carpeta es un ejecutor externo para la matriz creada en `outputs/chua_nonsmooth_fractional_memory_matrix/`. No modifica el workflow principal ni las validaciones cerradas; consume los modelos, semillas, integradores, diagnósticos y graficadores ya disponibles en `hidden_attractors`.

### Fundamento Matemático

- `integer_like_seed` usa la clausura de función descriptiva clásica con $W(s) = r^T (sI-P)^{-1} b$, equivalente a $q=1$.
- `fractional_seed` usa la misma DF clásica con $W_q(i\omega) = r^T ((i\omega)^q I-P)^{-1}b$, rama principal $(i\omega)^q = \omega^q \exp(i q \pi/2)$.
- Estas semillas son oscilaciones armónicas asintóticas tipo Weyl usadas para inicializar pruebas numéricas; no son ciclos periódicos exactos de Caputo ni demuestran ocultedad.
- La continuación `integer_like_q1` propaga solo el último punto.
- La continuación `fractional_caputo` transporta historia. En ABM, el modo `full` conserva la historia Caputo cronológica de toda la cadena en `eta` y `truncated` conserva exactamente la ventana reiniciada `Lm`. En EFORK, `truncated` usa `Lm` y `full` usa un horizonte suficientemente largo para no truncar la cadena integrada.
- El wrapper EFORK informa conteos de historia pero no expone los valores de toda su prehistoria interna en modo `full`. Por eso `final_history_window.npz` identifica en ese caso la cola observada disponible y marca `exact_transported_window=false`; en modo `truncated` se exige `stage_keep >= Lm` y la ventana guardada sí es exacta.
- El backend ABM externo a este experimento expone continuación de la homotopía Lur'e: la historia final guardada es exacta para las rutas ABM `full` y `truncated`. Para `integer_like_q1`, tanto ABM como EFORK etiquetan y aplican `last_point_only`.

### Ejecución

Desde `version_2/`:

```bash
python experiments/chua_nonsmooth_memory_matrix/run_shared_cache_tasks.py --workers 4
python experiments/chua_nonsmooth_memory_matrix/run_continuation_tasks.py --workers 4
python experiments/chua_nonsmooth_memory_matrix/run_hiddenness_tasks.py --workers 4
python figure_scripts/chua_nonsmooth_memory_matrix_run_figure_tasks.py --workers 2
python experiments/chua_nonsmooth_memory_matrix/aggregate_results.py
```

En macOS también puede ejecutarse el lanzador portátil:

```bash
WORKERS=2 bash experiments/chua_nonsmooth_memory_matrix/run_matrix_mac.sh
```

`WORKERS=1` es prudente para ABM `full`: retener la historia completa puede consumir mucha memoria y su suma de Volterra crece con el horizonte.

Cada proceso fuerza `OMP_NUM_THREADS=1`, `OMP_THREAD_LIMIT=1`, `MKL_NUM_THREADS=1` y `OPENBLAS_NUM_THREADS=1`. Las tablas de tareas se validan para impedir que dos workers escriban el mismo `cache_key`.

Las continuaciones usan por defecto, en cada valor de `eta`, un segmento transitorio y uno observado de longitud `t_burn`; pueden declararse explícitamente con `--stage-transient` y `--stage-keep`, y ambos valores quedan en metadatos.

Para pruebas cortas de cableado, `run_hiddenness_tasks.py` acepta `--max-probes-per-cloud N`. Esa reducción nunca promueve ocultedad: si no hay impactos, el resultado queda como `inconclusive`.

### Resultados

Todos los resultados se escriben en la raíz de outputs de la matriz y llevan `status.json` y metadatos de commit, fecha, contrato numérico, `q`, `h`, `t_final`, `t_burn`, `Lm`, integrador, política de memoria, `exp_id` y `cache_key`. Una tarea con `status="ok"` se reutiliza al relanzar; `--force` ordena recalcularla.

Las etiquetas de ocultedad admisibles son solamente:

- `compatible_with_hiddenness_under_tested_radii`
- `not_hidden_under_tested_radii`
- `inconclusive`
- `numerical_failure`

Una ausencia de impactos desde las bolas probadas es evidencia limitada bajo el contrato numérico documentado, no una prueba global de atractor oculto. Si la trayectoria candidata queda clasificada como periódica en el control post-transitorio, el veredicto de ocultedad se conserva como `inconclusive`.

Además de `phase3d_candidate` y `time_series` agregadas por experimento, cada configuración candidata integrada después de la continuación se guarda bajo `figures/<exp_id>/candidate_attractors/`, con nombres que identifican solver de continuación, integrador objetivo, política de memoria y signo.
