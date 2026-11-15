# Mapas finitos Wu2023 con despacho ADM real

Estos mapas no contienen una afirmación de ocultedad (no hiddenness claim).
Las restricciones de radios locales del contrato de validación de Chua arctan
permanecen vigentes: extender el dominio de una malla no extiende ese contrato.

El programa `validation/reference_cases/fractional_chua_arctan_wu2023/scripts/chua_arctan_wu2023_plot_basins.py` acepta `--config`, `--output-dir` y `--planes xy xz yz`. El contrato original selecciona `adm_wu2023`; ahora cada PVI llama directamente a ese integrador con sus parámetros, `q`, `h`, `N`, orden cuatro y umbral declarados. No convierte la orden fraccionaria ni sustituye ADM por EFORK.

```powershell
python validation/reference_cases/fractional_chua_arctan_wu2023/scripts/chua_arctan_wu2023_plot_basins.py --config validation/reference_cases/fractional_chua_arctan_wu2023/config.json --output-dir outputs/wu_adm_nueva --points 9 --planes xy xz yz
```

La carpeta debe ser nueva o estar vacía. Se reserva antes de integrar mediante `_run_output`; una corrida interrumpida conserva su contenido y obliga a elegir otro destino. `--output antigua.png` se conserva como alias de nombre de carpeta `antigua`; los archivos nuevos se llaman `map_xy.png`, etc. `--t-final` sólo acepta el horizonte ya declarado: para cambiarlo hay que entregar un contrato separado que mantenga compatibles `h`, `N` y `t_final`.

Cada plano guarda matriz NPZ, tabla CSV y PNG. Cada PVI guarda los tiempos y estados completos en un NPZ y un JSON con estado del integrador, clasificación y SHA-256. `manifest.json` enlaza entradas, contrato efectivo, fuentes y salidas. La semilla de la malla es determinista; no hay muestreo aleatorio.

Antes de integrar se guardan copias y hashes de las fuentes numéricas y de clasificación correspondientes a la ruta elegida. EFORK incluye su selector, fachada, integrador, validación de historia y enlace del campo; ADM incluye su implementación especializada. Si una de esas fuentes cambia durante la ejecución, no se cierra el manifiesto y se conserva la corrida parcial. Los datos archivados antes de esta ampliación conservan sus manifiestos originales y su cobertura de fuentes original.

`basin_map` permite `points`, `limit`, `planes`, `fixed_coordinate`, `t_burn`, `terminal_window_points`, `equilibrium_tolerance`, `equilibrium_span_tolerance` y `mean_zero_tolerance`. Los valores efectivos siempre quedan guardados. Para la clase `equilibrium_terminal` se exige horizonte completo, ventana después del transitorio, controles compartidos de distancia final/percentil 95 y extensión de la cola, y que **todos** los puntos de la ventana estén dentro de la misma bola de equilibrio. Una visita temprana o sólo el último punto no bastan. Las clases no finitas, cruce del umbral e incompletas se separan.

`bounded_positive_mean`, `bounded_negative_mean` y `bounded_near_zero_mean` sólo describen el signo de la media de x en esa ventana finita. Pueden depender de la fase y de la ventana: no identifican atractores ni contacto con una referencia. `is_reference_target` y `hidden_verified` permanecen en falso.

La clase `threshold_crossed` exige un estado guardado cuya norma sea estrictamente mayor que el umbral. Un estado textual `diverged` sin cruce observado se marca `inconclusive`, con el motivo `reported_divergence_without_observed_crossing`; no basta el nombre del estado del integrador para afirmar el evento.

El ADM implementado reproduce la actualización local de Wu; no acumula historia Caputo. Para un diagnóstico EFORK separado, el contrato debe seleccionar explícitamente `integrator: efork3`, `backend: caputo_diagnostic` (o `python`), `memory_policy: full`, `memory_length >= t_final` y `caputo_history_accumulated: true`. Esta ruta usa EFORK de Python con historia completa y sin captura temprana de equilibrio. Rechaza contratos de memoria truncada para evitar una ambigüedad adicional. Las dos rutas conservan su método en todas las salidas.

Las pruebas incluyen despacho especializado, parámetros distintos a los predeterminados, orden próxima a uno sin redondearla, contratos incompatibles sin creación de salidas, protección de carpetas, captura temprana/último punto, estados no finitos y una malla ADM real 3×3 cuyos nueve PVI guardados coinciden bit a bit con llamadas directas al integrador.
