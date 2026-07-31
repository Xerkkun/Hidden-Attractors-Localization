# Catálogo reproducible de gráficas

Este catálogo cubre las 33 funciones públicas cuyo nombre comienza con
`plot_` o `render_` en `hidden_attractors.plotting`. Sus entradas no son curvas
pedagógicas ni datos sintéticos construidos a mano: todas proceden de cálculos
numéricos reproducibles ejecutados con el sistema dinámico registrado
`chua-nonsmooth`. El caso se identifica como
`chua_nonsmooth_real_system_catalog_20260729` y su procedencia es una
reintegración con la implementación canónica de la biblioteca.

El vector de parámetros es
`alpha=8.4562`, `beta=12.0732`, `gamma=0.0052`, `m0=-0.1768`,
`m1=-1.1468`, `a1=0.4`, `a2=-1.5585` y `rho=1.0`. Según la figura, el
generador usa una trayectoria `q=1` con `efork_q1`, una continuación entera
en `lambda`, un barrido real de `beta` con RK4, exponentes finito-temporales
con QR--Benettin, controles integrados desde vecindades de los equilibrios o
una clasificación de cuenca por tiempo finito. Los valores exactos del paso,
intervalo temporal, condición inicial, mallas, umbrales, semillas y hashes
SHA-256 quedan registrados en
`docs/assets/generated_plot_catalog/catalog_results.json`.

Que los datos sean resultados numéricos reales de un sistema definido no los
convierte en mediciones experimentales ni en una nueva validación. Cada
imagen demuestra una salida de la API y no certifica por sí sola caos,
estabilidad asintótica, ocultamiento, convergencia del integrador ni
desempeño científico.

## Regeneración

Desde la raíz de `version_2`:

```powershell
python figure_scripts/generate_plot_catalog_examples.py
```

Para regenerar un único ejemplo:

```powershell
python figure_scripts/generate_plot_catalog_examples.py --only plot_phase_space
```

El listado verificable de funciones y resultados queda en
`docs/assets/generated_plot_catalog/catalog_results.json`. El generador
compara su inventario con `hidden_attractors.plotting.__all__` y falla si la
API pública añade, elimina o reordena una función sin actualizar el catálogo.
El manifiesto también registra las 62 imágenes producidas: una representativa
por función y las 29 salidas adicionales de las funciones multisalida.

## Convenciones de los ejemplos

Los comandos de las tablas suponen:

```python
from hidden_attractors import get_system
from hidden_attractors.plotting import *

# trajectory y trajectory_alt: arreglos (N, 4) con columnas t, x, y, z
# output_path: archivo .png; output_dir: directorio de salida
# config: diccionario con system_id, h y opciones específicas del gráfico
```

En el catálogo, esos objetos se rellenan con resultados calculados para
`get_system("chua-nonsmooth")`. El campo `input_bundle` de cada fila del
manifiesto identifica qué cálculo alimentó la llamada, y `produced_outputs`
enumera todas sus imágenes junto con el hash correspondiente.

Los tipos especializados (`BifurcationPoint`, `SpectrumResult`,
`LyapunovResult`, `IntegerLureContinuationStep` e
`IntegerHiddennessProbe`) se construyen con las clases públicas indicadas por
la firma. El script de regeneración contiene un conjunto mínimo completo y
ejecutable para cada llamada.

## Trayectorias, series y espectros

| Función | Datos mínimos | Llamada directa | Ejemplo |
|---|---|---|---|
| `plot_phase_space` | `trajectory` `(N,4)` | `plot_phase_space(trajectory, output_path, dims=("x", "y", "z"))` | [PNG](assets/generated_plot_catalog/examples/08_plot_phase_space.png) |
| `plot_phase_projections` | `trajectory` `(N,4)` | `plot_phase_projections(trajectory, output_path)` | [PNG](assets/generated_plot_catalog/examples/07_plot_phase_projections.png) |
| `plot_time_series` | `trajectory` `(N,4)` | `plot_time_series(trajectory, output_path, columns=("x", "y", "z"))` | [PNG](assets/generated_plot_catalog/examples/10_plot_time_series.png) |
| `plot_trajectory_overlay` | lista de trayectorias y etiquetas | `plot_trajectory_overlay([trajectory, trajectory_alt], ["A", "B"], title="Comparación", output_path=output_path)` | [PNG](assets/generated_plot_catalog/examples/12_plot_trajectory_overlay.png) |
| `plot_spectrum` | un `SpectrumResult` | `plot_spectrum(spectrum, output_path, x_units="Hz")` | [PNG](assets/generated_plot_catalog/examples/09_plot_spectrum.png) |
| `plot_trajectory_spectra` | `trajectory` `(N,4)` con muestreo temporal | `plot_trajectory_spectra(trajectory, output_dir, method="fft", prefix="case")` | [PNG](assets/generated_plot_catalog/examples/11_plot_trajectory_spectra.png) |
| `plot_lyapunov_convergence` | un `LyapunovResult` | `plot_lyapunov_convergence(result, output_path)` | [PNG](assets/generated_plot_catalog/examples/03_plot_lyapunov_convergence.png) |
| `plot_attractor_trajectories` | trayectoria, equilibrios y `config` | `plot_attractor_trajectories(trajectory, equilibria, config, output_dir)` | [PNG](assets/generated_plot_catalog/examples/24_plot_attractor_trajectories.png) |
| `plot_flexible_attractor_and_projections` | trayectoria, equilibrios y `config` | `plot_flexible_attractor_and_projections(trajectory, equilibria, config, output_dir, "case")` | [PNG](assets/generated_plot_catalog/examples/25_plot_flexible_attractor_and_projections.png) |
| `plot_timeseries_data` | trayectoria y `config` | `plot_timeseries_data(trajectory, config, output_dir, "case")` | [PNG](assets/generated_plot_catalog/examples/26_plot_timeseries_data.png) |

`plot_trajectory_spectra` acepta `method="fft"` o los alias
`"psd"`, `"welch"` y `"psd_welch"`. `plot_timeseries_data` también escribe un
CSV con las columnas `t,x,y,z`.

## Bifurcación, cuencas y estabilidad

| Función | Datos mínimos | Llamada directa | Ejemplo |
|---|---|---|---|
| `plot_bifurcation_diagram` | secuencia de `BifurcationPoint` | `plot_bifurcation_diagram(points, output_path, parameter_label="p", observable_label="x")` | [PNG](assets/generated_plot_catalog/examples/06_plot_bifurcation_diagram.png) |
| `plot_basin_slices` | diccionario `plane -> (u, v, classes)` | `plot_basin_slices({"xy": (u, v, classes)}, "system_id", output_dir)` | [PNG](assets/generated_plot_catalog/examples/13_plot_basin_slices.png) |
| `plot_basin_slice_file` | plano, dos ejes y matriz de clases | `plot_basin_slice_file("xy", u, v, classes, "E0", "system_id", output_dir)` | [PNG](assets/generated_plot_catalog/examples/14_plot_basin_slice_file.png) |
| `plot_matignon_equilibria` | `ChaoticSystem`, equilibrios y orden `q` | `plot_matignon_equilibria(system, equilibria, 0.9, output_dir)` | [PNG](assets/generated_plot_catalog/examples/15_plot_matignon_equilibria.png) |

Las matrices de cuenca se obtienen integrando una malla real \(31\times31\)
del plano \(xy\), con \(z=0\), `q=1`, `efork_q1`, `h=0.02` y horizonte
`t_final=35`. Las clases usan umbrales operativos finito-temporales declarados
en el manifiesto; sus colores no demuestran una partición asintótica o global
de la cuenca.

## Transferencia de Lur'e y función descriptiva

| Función | Datos mínimos | Llamada directa | Ejemplo |
|---|---|---|---|
| `plot_lure_nyquist_describing_function` | `LureSystem` y `HarmonicSeed` | `plot_lure_nyquist_describing_function(system.lure, seed, output_path, q=1.0)` | [PNG](assets/generated_plot_catalog/examples/04_plot_lure_nyquist_describing_function.png) |
| `plot_lure_transfer_components` | `LureSystem` y `HarmonicSeed` | `plot_lure_transfer_components(system.lure, seed, output_path, q=1.0)` | [PNG](assets/generated_plot_catalog/examples/05_plot_lure_transfer_components.png) |
| `plot_nyquist_transfer` | malla de frecuencias, valores complejos y candidatos `(A,omega,k)` | `plot_nyquist_transfer(omega, W, candidates, config, output_dir)` | [PNG](assets/generated_plot_catalog/examples/16_plot_nyquist_transfer.png) |
| `plot_describing_function` | sistema con contrato `lure`, candidatos y límites en `config` | `plot_describing_function(system, candidates, config, output_dir)` | [PNG](assets/generated_plot_catalog/examples/17_plot_describing_function.png) |
| `plot_harmonic_residual_map` | mismo contrato más rango de frecuencias | `plot_harmonic_residual_map(system, candidates, config, output_dir)` | [PNG](assets/generated_plot_catalog/examples/18_plot_harmonic_residual_map.png) |

En estas llamadas un candidato tiene el orden `(A, omega, k)`. El ejemplo
real del catálogo declara `transfer_convention="opposite_sign"` y
`harmonic_condition="1_plus_WN"`; por tanto usa
`W_code(s)=c.T@(P-sI)^(-1)@b`, representa
`|1 + N(A)W_code(i omega)|` y marca el cierre `W_code=-1/k`. La API también
admite la pareja normalizada `standard + 1_minus_WN`, para la cual
`W_report=-W_code`, el residuo es `|1-N(A)W_report|` y el cierre es
`W_report=+1/k`. Las funciones infieren la clave complementaria cuando sólo
se declara una y rechazan parejas de signo incoherentes, salvo habilitación
explícita. Un mínimo visual no sustituye la verificación numérica de cierre
ni valida una trayectoria.

## Continuación y controles de vecindad

| Función | Datos mínimos | Llamada directa | Ejemplo |
|---|---|---|---|
| `plot_integer_lure_continuation` | secuencia de `IntegerLureContinuationStep` | `plot_integer_lure_continuation(steps, output_path)` | [PNG](assets/generated_plot_catalog/examples/02_plot_integer_lure_continuation.png) |
| `plot_integer_hiddenness_controls` | trayectoria objetivo y `IntegerHiddennessProbe` | `plot_integer_hiddenness_controls(trajectory, probes, output_path)` | [PNG](assets/generated_plot_catalog/examples/01_plot_integer_hiddenness_controls.png) |
| `plot_continuation_eta` | lista de pasos con `lambda_value`, `x_out`, `status` y `trajectory` | `plot_continuation_eta(cont_steps, config, output_dir)` | [PNG](assets/generated_plot_catalog/examples/19_plot_continuation_eta.png) |
| `plot_continuation_first_last_comparison` | al menos dos pasos con trayectoria | `plot_continuation_first_last_comparison(cont_steps, config, output_dir)` | [PNG](assets/generated_plot_catalog/examples/20_plot_continuation_first_last_comparison.png) |
| `plot_continuation_timeseries_comparison` | al menos dos trayectorias `(t,x,y,z)` | `plot_continuation_timeseries_comparison(cont_steps, config, output_dir)` | [PNG](assets/generated_plot_catalog/examples/21_plot_continuation_timeseries_comparison.png) |
| `plot_continuation_progression` | pasos con `lambda_value`, `x_in`, `x_out` y trayectoria | `plot_continuation_progression(cont_steps, config, output_dir)` | [PNG](assets/generated_plot_catalog/examples/22_plot_continuation_progression.png) |
| `plot_continuation_tracking` | pasos con `lambda_value`, `x_out` y `status` | `plot_continuation_tracking(cont_steps, config, output_dir)` | [PNG](assets/generated_plot_catalog/examples/23_plot_continuation_tracking.png) |
| `plot_neighborhood_control_spheres` | trayectoria, sondas, equilibrios y `config` | `plot_neighborhood_control_spheres(trajectory, probe_results, equilibria, config, output_dir)` | [PNG](assets/generated_plot_catalog/examples/27_plot_neighborhood_control_spheres.png) |
| `plot_sphere_test_results` | equilibrio, radio y registros de sondas | `plot_sphere_test_results("E0", eq, radius, probe_runs, output_dir)` | [PNG](assets/generated_plot_catalog/examples/28_plot_sphere_test_results.png) |

Un control de vecindad sólo visualiza la procedencia y clasificación de las
sondas suministradas. La figura por sí misma no demuestra que un atractor sea
oculto.

El nombre público `plot_continuation_eta` y los nombres de archivo que
contienen `_eta` se conservan por compatibilidad. Los ejes y las etiquetas del
catálogo usan el parámetro de continuación `lambda`, que es el nombre del dato
`lambda_value`.

## Renderizadores unificados

| Función | Datos mínimos | Llamada directa | Ejemplo |
|---|---|---|---|
| `render_attractor` | trayectoria, equilibrios y `config` | `render_attractor(trajectory, equilibria, config, run_id="case")` | [PNG](assets/generated_plot_catalog/examples/29_render_attractor.png) |
| `render_basin` | ejes, matriz de clases y `config` | `render_basin(grid_x, grid_y, basin_grid, config, run_id="case")` | [PNG](assets/generated_plot_catalog/examples/30_render_basin.png) |
| `render_nyquist` | frecuencias, `W`, `N`, candidatos y `config` | `render_nyquist(freqs, W, N, candidates, config, run_id="case")` | [PNG](assets/generated_plot_catalog/examples/31_render_nyquist.png) |
| `render_matignon` | arreglo de autovalores, `q` y `config` | `render_matignon(eigenvalues, q, config, run_id="case")` | [PNG](assets/generated_plot_catalog/examples/32_render_matignon.png) |
| `render_all_plots` | cualquier subconjunto compatible de los cuatro bloques anteriores | `render_all_plots(trajectory=trajectory, equilibria=equilibria, config=config, run_id="case")` | [PNG](assets/generated_plot_catalog/examples/33_render_all_plots.png) |

Los cuatro renderizadores escriben PNG, PDF y metadatos JSON en el almacén
configurado por `HIDDEN_ATTRACTORS_OUTPUT_DIR`. `render_all_plots` es un
orquestador: no define un quinto tipo matemático de gráfica.

## Registro visual de todas las salidas adicionales

Las tablas anteriores muestran una imagen representativa por cada una de las
33 funciones. Las siguientes 29 imágenes completan el inventario de las diez
funciones multisalida. El comando indicado regenera tanto la imagen principal
como todas las adicionales de su fila.

| Función | Comando exacto | Salidas adicionales |
|---|---|---|
| `plot_trajectory_spectra` | `python figure_scripts/generate_plot_catalog_examples.py --only plot_trajectory_spectra` | [componente 1](assets/generated_plot_catalog/examples/11_plot_trajectory_spectra__02_catalog_fft_component_1.png)<br>[componente 2](assets/generated_plot_catalog/examples/11_plot_trajectory_spectra__03_catalog_fft_component_2.png) |
| `plot_nyquist_transfer` | `python figure_scripts/generate_plot_catalog_examples.py --only plot_nyquist_transfer` | [ampliación del cierre](assets/generated_plot_catalog/examples/16_plot_nyquist_transfer__02_fig01b_nyquist_zoom_x.png)<br>[partes real e imaginaria](assets/generated_plot_catalog/examples/16_plot_nyquist_transfer__03_transfer_real_imag.png) |
| `plot_continuation_eta` | `python figure_scripts/generate_plot_catalog_examples.py --only plot_continuation_eta` | [amplitud contra lambda](assets/generated_plot_catalog/examples/19_plot_continuation_eta__02_continuation_amplitude_vs_eta.png)<br>[primer y último paso](assets/generated_plot_catalog/examples/19_plot_continuation_eta__03_continuation_first_last_comparison.png)<br>[proyecciones del primer y último paso](assets/generated_plot_catalog/examples/19_plot_continuation_eta__04_continuation_first_last_projections.png)<br>[series temporales](assets/generated_plot_catalog/examples/19_plot_continuation_eta__05_continuation_timeseries_comparison_x.png)<br>[progresión](assets/generated_plot_catalog/examples/19_plot_continuation_eta__06_continuation_progression.png) |
| `plot_continuation_first_last_comparison` | `python figure_scripts/generate_plot_catalog_examples.py --only plot_continuation_first_last_comparison` | [proyecciones del primer y último paso](assets/generated_plot_catalog/examples/20_plot_continuation_first_last_comparison__02_continuation_first_last_projections.png) |
| `plot_continuation_tracking` | `python figure_scripts/generate_plot_catalog_examples.py --only plot_continuation_tracking` | [estado por paso](assets/generated_plot_catalog/examples/23_plot_continuation_tracking__02_continuation_tracking_status.png) |
| `plot_attractor_trajectories` | `python figure_scripts/generate_plot_catalog_examples.py --only plot_attractor_trajectories` | [proyección xy](assets/generated_plot_catalog/examples/24_plot_attractor_trajectories__02_attractor_xy.png)<br>[proyección xz](assets/generated_plot_catalog/examples/24_plot_attractor_trajectories__03_attractor_xz.png)<br>[proyección yz](assets/generated_plot_catalog/examples/24_plot_attractor_trajectories__04_attractor_yz.png) |
| `plot_flexible_attractor_and_projections` | `python figure_scripts/generate_plot_catalog_examples.py --only plot_flexible_attractor_and_projections` | [proyección xy](assets/generated_plot_catalog/examples/25_plot_flexible_attractor_and_projections__02_catalog_flexible_xy.png)<br>[proyección xz](assets/generated_plot_catalog/examples/25_plot_flexible_attractor_and_projections__03_catalog_flexible_xz.png)<br>[proyección yz](assets/generated_plot_catalog/examples/25_plot_flexible_attractor_and_projections__04_catalog_flexible_yz.png) |
| `plot_timeseries_data` | `python figure_scripts/generate_plot_catalog_examples.py --only plot_timeseries_data` | [serie y](assets/generated_plot_catalog/examples/26_plot_timeseries_data__02_catalog_series_timeseries_y.png)<br>[serie z](assets/generated_plot_catalog/examples/26_plot_timeseries_data__03_catalog_series_timeseries_z.png)<br>[series xyz](assets/generated_plot_catalog/examples/26_plot_timeseries_data__04_catalog_series_timeseries_xyz.png) |
| `render_attractor` | `python figure_scripts/generate_plot_catalog_examples.py --only render_attractor` | [proyección xy](assets/generated_plot_catalog/examples/29_render_attractor__02_chua-nonsmooth_attractor_xy.png)<br>[proyección xz](assets/generated_plot_catalog/examples/29_render_attractor__03_chua-nonsmooth_attractor_xz.png)<br>[proyección yz](assets/generated_plot_catalog/examples/29_render_attractor__04_chua-nonsmooth_attractor_yz.png) |
| `render_all_plots` | `python figure_scripts/generate_plot_catalog_examples.py --only render_all_plots` | [atractor xy](assets/generated_plot_catalog/examples/33_render_all_plots__02_chua-nonsmooth_attractor_xy.png)<br>[atractor xz](assets/generated_plot_catalog/examples/33_render_all_plots__03_chua-nonsmooth_attractor_xz.png)<br>[atractor yz](assets/generated_plot_catalog/examples/33_render_all_plots__04_chua-nonsmooth_attractor_yz.png)<br>[cuenca](assets/generated_plot_catalog/examples/33_render_all_plots__05_chua-nonsmooth_basin.png)<br>[Nyquist](assets/generated_plot_catalog/examples/33_render_all_plots__06_chua-nonsmooth_nyquist.png)<br>[Matignon](assets/generated_plot_catalog/examples/33_render_all_plots__07_chua-nonsmooth_matignon.png) |

Así, el catálogo conserva una relación verificable de 33 funciones, 33
comandos individuales y 62 PNG. Los sufijos `_eta` de algunas rutas se
mantienen únicamente para no romper compatibilidad; dentro de las gráficas el
parámetro se etiqueta como `lambda`.
