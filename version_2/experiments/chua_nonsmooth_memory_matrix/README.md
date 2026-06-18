# Matriz experimental Chua no suave

Esta carpeta es un ejecutor externo para la matriz creada en
`outputs/chua_nonsmooth_fractional_memory_matrix/`. No modifica el workflow
principal ni las validaciones cerradas; consume los modelos, semillas,
integradores, diagnosticos y graficadores ya disponibles en
`hidden_attractors`.

## Fundamento matematico

- `integer_like_seed` usa la clausura de funcion descriptiva clasica con
  `W(s) = r^T (sI-P)^(-1) b`, equivalente a `q=1`.
- `fractional_seed` usa la misma DF clasica con
  `W_q(i omega) = r^T ((i omega)^q I-P)^(-1)b`, rama principal
  `(i omega)^q = omega^q exp(i q pi/2)`.
- Estas semillas son oscilaciones armonicas asintoticas tipo Weyl usadas para
  inicializar pruebas numericas; no son ciclos periodicos exactos de Caputo ni
  demuestran ocultedad.
- La continuacion `integer_like_q1` propaga solo el ultimo punto.
- La continuacion `fractional_caputo` transporta historia. En ABM, el modo
  `full` conserva la historia Caputo cronologica de toda la cadena en `eta` y
  `truncated` conserva exactamente la ventana reiniciada `Lm`. En EFORK,
  `truncated` usa `Lm` y `full` usa un horizonte suficientemente largo para
  no truncar la cadena integrada.
- El wrapper EFORK informa conteos de historia pero no expone los valores de
  toda su prehistoria interna en modo `full`. Por eso
  `final_history_window.npz` identifica en ese caso la cola observada
  disponible y marca `exact_transported_window=false`; en modo `truncated`
  se exige `stage_keep >= Lm` y la ventana guardada si es exacta.
- El backend ABM externo a este experimento expone continuacion de la
  homotopia Lur'e: la historia final guardada es exacta para las rutas ABM
  `full` y `truncated`. Para `integer_like_q1`, tanto ABM como EFORK etiquetan
  y aplican `last_point_only`.

## Ejecucion

Desde `version_2/`:

```bash
python experiments/chua_nonsmooth_memory_matrix/run_shared_cache_tasks.py --workers 4
python experiments/chua_nonsmooth_memory_matrix/run_continuation_tasks.py --workers 4
python experiments/chua_nonsmooth_memory_matrix/run_hiddenness_tasks.py --workers 4
python figure_scripts/chua_nonsmooth_memory_matrix_run_figure_tasks.py --workers 2
python experiments/chua_nonsmooth_memory_matrix/aggregate_results.py
```

En macOS tambien puede ejecutarse el lanzador portatil:

```bash
WORKERS=2 bash experiments/chua_nonsmooth_memory_matrix/run_matrix_mac.sh
```

`WORKERS=1` es prudente para ABM `full`: retener la historia completa puede
consumir mucha memoria y su suma de Volterra crece con el horizonte.

Cada proceso fuerza `OMP_NUM_THREADS=1`, `OMP_THREAD_LIMIT=1`,
`MKL_NUM_THREADS=1` y `OPENBLAS_NUM_THREADS=1`. Las tablas de tareas se
validan para impedir que dos workers escriban el mismo `cache_key`.

Las continuaciones usan por defecto, en cada valor de `eta`, un segmento
transitorio y uno observado de longitud `t_burn`; pueden declararse
explicitamente con `--stage-transient` y `--stage-keep`, y ambos valores
quedan en metadatos.

Para pruebas cortas de cableado, `run_hiddenness_tasks.py` acepta
`--max-probes-per-cloud N`. Esa reduccion nunca promueve ocultedad: si no hay
impactos, el resultado queda como `inconclusive`.

## Resultados

Todos los resultados se escriben en la raiz de outputs de la matriz y llevan
`status.json` y metadatos de commit, fecha, contrato numerico, `q`, `h`,
`t_final`, `t_burn`, `Lm`, integrador, politica de memoria, `exp_id` y
`cache_key`. Una tarea con `status="ok"` se reutiliza al relanzar; `--force`
ordena recalcularla.

Las etiquetas de ocultedad admisibles son solamente:

- `compatible_with_hiddenness_under_tested_radii`
- `not_hidden_under_tested_radii`
- `inconclusive`
- `numerical_failure`

Una ausencia de impactos desde las bolas probadas es evidencia limitada bajo
el contrato numerico documentado, no una prueba global de atractor oculto.
Si la trayectoria candidata queda clasificada como periodica en el control
post-transitorio, el veredicto de ocultedad se conserva como `inconclusive`.

Ademas de `phase3d_candidate` y `time_series` agregadas por experimento, cada
configuracion candidata integrada despues de la continuacion se guarda bajo
`figures/<exp_id>/candidate_attractors/`, con nombres que identifican solver
de continuacion, integrador objetivo, politica de memoria y signo.
