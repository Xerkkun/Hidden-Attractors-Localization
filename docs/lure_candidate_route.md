# Ruta Lure Clasica

Esta ruta separa los candidatos obtenidos con funcion descriptiva clasica por transformacion tipo Lure. No usa puntos perpetuos y no mezcla la ruta Machado, salvo para una comparacion de invariantes cuando esos archivos ya existen.

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

Etiquetas conservadoras:

```text
compatible_with_hiddenness_under_tested_radii
not_supported_by_refined_neighborhood_test
inconclusive_isolated_hit
invalid_or_divergent
```

La ausencia de contactos en una muestra finita no declara `hidden_verified`. Solo indica compatibilidad bajo los radios y muestras probados. Si una vecindad abierta de un equilibrio alcanza el candidato, el candidato no es oculto bajo la definicion operativa.

## Lure vs Machado

La comparacion Lure vs Machado contrasta `q`, estado final, rangos, estadisticas de cola, pico FFT, entropia PSD y etiqueta de objetivo si existen. Si los invariantes quedan cercanos, `likely_same_attractor=true`. En ese caso, una prueba de no ocultedad para uno afecta la interpretacion del otro.

## Notas Cientificas

La funcion descriptiva y el balance armonico son heuristicas para generar semillas. No prueban un ciclo limite exacto de Caputo. La validacion debe hacerse con el sistema fraccionario causal de Caputo y memoria.
