# Ruta Lur'e Clásica

Esta ruta separa las semillas obtenidas con función descriptiva clásica por transformación tipo Lur'e. No usa puntos perpetuos y no mezcla la ruta Machado, salvo para una comparación de invariantes cuando esos archivos ya existen.

La convencion de transferencia es la interna del repositorio:

```text
W_code(lambda) = r^T (P - lambda I)^(-1) b
lambda = (j omega)^q = omega^q exp(j q pi/2)
```

Si se usa la convencion matematica alternativa `r^T (lambda I - P)^(-1) b`, cambia el signo: `W_alt = -W_code`.

## Problema

Los candidatos Lure clasicos pueden estar repartidos entre corridas `df_compare`, barridos de `q`, verificaciones ocultas y carpetas de salida del pipeline. Ademas, `corrida1_refined_verification.py` esta orientado a candidatos Machado con campos `mu`, `theta`, `branch` y `candidate_id` de esa familia. Esta ruta crea un manifest limpio para Lure clasico y luego ejecuta diagnosticos especificos.

## Manifest

Si ya existen resultados:

```powershell
python lure_candidate_manifest.py --config configs/lure_candidate_route.yaml --source auto
```

El modo `auto` busca primero resultados existentes en `df_seed_comparison/`, `q_order_sweep/`, `hidden_verify/`, `final_pdf_figs/`, `runs*/`, `chua_piecewise/` y `chua_arctan/`. Solo incluye candidatos `classic`, `lure` o `nyquist_df`; excluye Machado.

Si no hay resultados suficientes y la configuracion permite reproducir:

```powershell
python lure_candidate_manifest.py --config configs/lure_candidate_route.yaml --source reproduce
```

La reproduccion llama funciones del pipeline oficial para Nyquist/DF, semilla armonica, continuacion en epsilon y verificacion de ocultedad. No ejecuta cuencas ni bifurcaciones.

Salidas:

```text
outputs/lure_route/lure_candidates_manifest.csv
outputs/lure_route/lure_candidates_manifest.json
```

## rho_H

```powershell
python lure_rhoH_diagnostics.py --config configs/lure_candidate_route.yaml
```

Calcula:

```text
rho_H = sum_{k=2..K} |W_q(k omega)| |Y_k(A,sigma0)| / (|W_q(omega)| |Y_1(A,sigma0)| + eps)
```

Para Lure clasico, si `sigma0` falta se usa `0`. La clase `rhoH_class` es diagnostica: `good`, `marginal`, `poor` o `missing`. No descarta candidatos automaticamente.

Salidas:

```text
outputs/lure_route/lure_rhoH_diagnostics.csv
outputs/lure_route/plots/lure_rhoH_vs_q.png
outputs/lure_route/plots/lure_rhoH_vs_A.png
outputs/lure_route/plots/lure_harmonic_spectrum_<candidate_id>.png
```

Si hay archivos Machado, tambien escribe:

```text
outputs/lure_route/lure_vs_machado_comparison.csv
```

## Semillas y Filtro Temprano

La función descriptiva clásica, sesgada o Machado/FDF es una aproximación de primer armónico: propone semillas, pero no prueba ciclos periódicos exactos de Caputo ni atractores ocultos. La ruta ampliada de Lur'e sesgada está en:

```text
configs/lure_biased_multiparam_q09998.yaml
tools/legacy/lure_biased_multiparam_search.py
```

La ruta Machado/FDF se mantiene separada para no inferir ausencia de semillas Machado a partir de un manifiesto que las excluya:

```text
configs/machado_candidate_route.yaml
outputs/machado_lure_route/machado_candidates_manifest.csv
outputs/machado_lure_route/machado_candidates_manifest.json
```

El cribado armónico duro acepta un punto si

```text
residual_abs < residual_keep,  rho_H < rhoH_keep,  A > 1e-4,  omega > 0.
```

Si no acepta ninguno, no se pierde la información de búsqueda. Se retienen semillas exploratorias por

```text
score = residual_abs + lambda_rho * max(0, rho_H - rhoH_priority)^2
```

con etiqueta `best_available_seed_not_accepted`. Ésta no significa evidencia de ocultedad.

Antes de la continuación larga, cada semilla reconstruida se integra bajo una matriz de cuatro contratos numéricos:

| Caso | Integrador | Historia | Papel en la decisión |
|---|---|---|---|
| `efork_full_history` | EFORK-3 C corregido | Sin truncamiento durante el horizonte simulado, usando `Lm = t_transient + observation_time` | Celda primaria: habilita o rechaza continuación |
| `efork_truncated_Lm8` | EFORK-3 C corregido | Ventana finita `Lm = 8` | Comparación de sensibilidad a memoria |
| `abm_full_history` | ABM C | Historia completa de Caputo | Comparación metodológica y referencia tipo Danca |
| `abm_truncated_Lm8` | ABM C | Ventana deslizante reiniciada `Lm = 8` | Comparación de sensibilidad a memoria |

La integración de cada celda tiene dos intervalos distintos:

```text
[0, t_transient]                              transitorio, no se clasifica
[t_transient, t_transient + observation_time] observación para periodicidad
```

Las configuraciones actuales establecen `t_transient = 120` y `observation_time = 120`. Por tanto, las FFT, entropías, derivas de frecuencia y secciones de Poincaré no usan el tramo inicial que nace directamente de la construcción armónica. La ejecución priorizada aplica la matriz a los 30 candidatos mejor ordenados; los aceptados duros restantes se registran como diferidos y no se interpretan como rechazados. Para cada componente centrada del intervalo de observación \(x_j(t_n)\), el filtro calcula

```text
P_j(omega_k) = |FFT[x_j](omega_k)|^2
H_j = - sum_k p_jk log(p_jk) / log(N_f)
C_j = max_k P_j(omega_k) / sum_l P_j(omega_l)
p_jk = P_j(omega_k) / sum_l P_j(omega_l)
```

Una semilla se marca `rejected_periodic_post_transient` sólo si la celda primaria `efork_full_history` presenta evidencia combinada: baja entropía y concentración alta, o concentración alta con frecuencia dominante estable entre ventanas, en al menos dos componentes cuando así se configura. Un pico FFT aislado no basta para descartar una trayectoria. Una discrepancia observada sólo en una celda truncada se conserva como sensibilidad numérica y no elimina la semilla principal.

Las semillas con `early_periodicity_status=nonperiodic_post_transient` en la celda primaria son las únicas habilitadas para continuación. Los artefactos de esta decisión son:

```text
biased_lure_all_evaluations.csv
top_ranked_all_evaluations.csv
biased_lure_candidates.csv
biased_lure_seed_bank.csv
post_transient_periodicity_matrix.csv
rejected_periodic_post_transient.csv
early_periodicity_summary.json
```

## Verificacion Refinada

Etapa A:

```powershell
python lure_refined_route.py --config configs/lure_candidate_route.yaml --stage A
```

Etapa B:

```powershell
python lure_refined_route.py --config configs/lure_candidate_route.yaml --stage B --resume
```

Etapa C:

```powershell
python lure_refined_route.py --config configs/lure_candidate_route.yaml --stage C --resume
```

Mapas locales:

```powershell
python lure_refined_route.py --config configs/lure_candidate_route.yaml --local-basin --resume
```

La verificacion recalcula los equilibrios de:

```text
P X + b psi(r^T X) = 0
```

Para el caso por tramos, se resuelven las regiones y se verifica consistencia regional. La estabilidad usa Matignon con:

```text
estable si |arg(lambda_i)| > q*pi/2
```

Salidas principales:

```text
outputs/lure_route/refined/equilibria_lure_summary.csv
outputs/lure_route/refined/lure_refined_raw.csv
outputs/lure_route/refined/lure_refined_summary.csv
outputs/lure_route/refined/lure_refined_decision.csv
outputs/lure_route/refined/lure_refined_report.md
outputs/lure_route/refined/lure_refined_summary.json
```

## Interpretacion

Etiquetas conservadoras del flujo nuevo:

```text
hard_candidate_accepted
best_available_seed_not_accepted
rejected_periodic_post_transient
nonperiodic_post_transient
continuation_survivor
rejected_by_equilibrium_neighborhood
compatible_with_hiddenness_under_tested_radii
hidden_verified
```

`hidden_verified` queda reservado para una verificación completa: cálculo y estabilidad de todos los equilibrios, simulación desde vecindades pequeñas de todos ellos, comparación con el atractor candidato y evidencia de que la cuenca no intersecta ninguna vecindad abierta bajo los radios probados. La ausencia de contactos en una muestra finita sólo produce `compatible_with_hiddenness_under_tested_radii`. Si una vecindad de cualquier equilibrio alcanza el candidato, se registra `rejected_by_equilibrium_neighborhood`.

## Lure vs Machado

La comparacion Lure vs Machado contrasta `q`, estado final, rangos, estadisticas de cola, pico FFT, entropia PSD y etiqueta de objetivo si existen. Si los invariantes quedan cercanos, `likely_same_attractor=true`. En ese caso, una prueba de no ocultedad para uno afecta la interpretacion del otro.

## Notas Cientificas

La función descriptiva propone; la dinámica causal temprana filtra; la continuación transporta; las cuencas deciden ocultedad.
