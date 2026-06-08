# Resumen Chua Fraccionario: semillas, funcion descriptiva, continuacion e integradores

Fecha de consolidacion: 2026-06-07.

Este archivo se enfoca solamente en el sistema de Chua fraccionario usado en el repositorio. Se separan cuatro cosas que no deben mezclarse:

- reproduccion publicada;
- semillas obtenidas con funcion descriptiva;
- continuacion numerica fraccionaria;
- integradores usados para la simulacion final.

Estado de evidencia: las etiquetas `chaotic_candidate_pending_robustness` son candidatos de caos por diagnostico finito. No son todavia certificacion final de caos ni prueba de ocultedad.

---

## 1. Sistemas Chua fraccionarios tratados

La forma comun usada por el codigo es la descomposicion de Lure

```text
D_t^q x = P x + b psi(r^T x),   0 < q <= 1
```

con

```text
b = [-alpha, 0, 0]^T
r = [1, 0, 0]^T
sigma = r^T x = x
```

La matriz lineal se arma con una pendiente base `s0`:

```text
P = [
  [-alpha (1 + s0),  alpha,       0],
  [ 1,              -1,           1],
  [ 0,              -beta,       -gamma]
]
```

Familias usadas:

| Familia | `system_id` | `s0` | `psi(sigma)` | Orden dinamico |
| :--- | :--- | :---: | :--- | :---: |
| Chua no suave / saturacion | `chua_fractional_saturation` | `m1` | `(m0 - m1) clip(sigma, -1, 1)` | `q=0.9998` en los barridos principales |
| Chua suave arctan | `chua_fractional_arctan` | `a1` | `a2 atan(rho sigma)` | `q=0.99` en los barridos principales |

Parametros lineales comunes en los barridos:

| Parametro | Valor |
| :--- | :---: |
| `alpha` | `8.4562` |
| `beta` | `12.0732` |
| `gamma` | `0.0052` |

---

## 2. Formas en que se obtuvieron las semillas iniciales

### 2.1 Semilla publicada entera: `published_integer_laplace`

Uso: reproducir o comparar casos publicados, aunque despues la dinamica sea fraccionaria.

Transferencia:

```text
W_pub(j omega) = r^T (j omega I - P)^(-1) b
```

Contrato:

| Campo | Valor |
| :--- | :--- |
| `seed_transfer_mode` | `published_integer_laplace` |
| `q_seed` | `1.0` |
| Dependencia de `q` dinamico | No |
| Construccion | formula cerrada entera / semilla DF publicada |
| Uso correcto | reproduccion publicada, comparacion, exploracion etiquetada como tal |

Casos fraccionarios publicados:

| Caso | Sistema | Dinamica final publicada | Politica de memoria |
| :--- | :--- | :--- | :--- |
| `danca2017_chua_fractional_saturation` | saturacion fraccionaria | `ABM`, `q=0.9998` | memoria completa |
| `wu2023_chua_fractional_arctan` | arctan fraccionario | `ADM_WU2023`, `q=0.99` | ADM local sin historia Caputo acumulada |

Nota metodologica: este modo no es la propuesta fraccionaria nueva. En particular, no debe mezclarse con `fractional_spectral` en una misma conclusion.

### 2.2 Semilla espectral fraccionaria: `fractional_spectral`

Uso: incorporar el orden fraccionario en la fase de la transferencia antes de construir la semilla.

Transferencia conceptual:

```text
W_q(j omega) = r^T ((j omega)^q I - P)^(-1) b
```

En el codigo hay dos convenciones equivalentes de signo segun el modulo:

| Familia | Funcion | Convencion usada |
| :--- | :--- | :--- |
| Saturacion | `find_omega_gain_candidates` | busca `Im(W_q)=0`, usa `k = -1 / Re(W_q)` |
| Arctan Wu2023 | `find_centered_arctan_wu2023_branches(..., transfer_mode="fractional_spectral")` | usa `1 - W_q N(A) = 0`, con `k = 1 / Re(W_q)` |

Contrato:

| Campo | Valor |
| :--- | :--- |
| `seed_transfer_mode` | `fractional_spectral` |
| `q_seed` | igual al `q` usado en la semilla |
| Dependencia de `q` dinamico | Si |
| Uso correcto | busqueda propuesta de candidatos fraccionarios |

### 2.3 Semillas no suaves por saturacion

Script: `search_saturation_candidates.py`.

Flujos usados:

| Flujo | `seed_q` | `q_dynamics` | Salida principal |
| :--- | :---: | :---: | :--- |
| semilla entera + dinamica fraccionaria | `1.0` | `0.9998` | `outputs/saturation_search_seed1_mem_full_sweep/summary.csv` |
| semilla fraccionaria + memoria completa | `0.9998` | `0.9998` | `outputs/saturation_search_seed0p9998_mem_full_sweep/summary.csv` |
| semilla fraccionaria + memoria truncada | `0.9998` | `0.9998` | `outputs/saturation_search_seed0p9998_mem_window_sweep/summary.csv` |

Procedimiento:

1. Escanear `omega` y buscar cruces `Im(W_q(j omega)) = 0`.
2. Calcular la ganancia `k`.
3. Resolver `N(A0) = k`.
4. Construir la semilla modal `x0` con `build_fractional_seed`.
5. Aplicar continuacion homotopica hasta `eta=1`.

### 2.4 Semillas arctan fraccionarias

Script: `search_arctan_fractional.py`.

Flujos disponibles:

| Flujo | `transfer_mode` | `q_dynamics` | Memoria | Estado en este archivo |
| :--- | :--- | :---: | :--- | :--- |
| arctan propuesto, memoria completa | `fractional_spectral` | `0.99` | completa | no analizado aqui |
| arctan propuesto, memoria truncada | `fractional_spectral` | `0.99` | ventana `Lm=10 s` | no analizado aqui |

Procedimiento:

1. Construir `ChuaParameters(model="arctan", alpha, beta, gamma, a1, a2, rho)`.
2. Buscar ramas con `find_centered_arctan_wu2023_branches(q=0.99, transfer_mode="fractional_spectral")`.
3. Resolver `omega0`, `k`, `A0` y semilla `x0`.
4. Continuar con ABM Caputo hasta `eta=1`.
5. Simular la dinamica final y clasificar la trayectoria post-transiente.

### 2.5 Exploraciones `version_2` con semilla publicada

Herramienta: `version_2/tools/search_arctan_full_memory_candidates.py`.

Modos disponibles:

| Modo | Memoria | Uso |
| :--- | :--- | :--- |
| `abm_full` | Caputo completa | exploracion arctan con memoria completa |
| `abm_restart` | reinicio por ultimo punto | control exploratorio, no equivalente a historia Caputo completa |
| `adm_restart` | ADM local tipo Wu | exploratorio, restringido a `rho=1` |

Estos modos quedan documentados aqui solo como rutas metodologicas; sus resultados arctan no se analizan en este resumen.

---

## 3. Funcion descriptiva usada

### 3.1 Condicion armonica general

La funcion descriptiva se usa para encontrar una oscilacion armonica compatible con el sistema lineal:

```text
Im(W(j omega0)) = 0
N(A0) = k
```

Despues se construye una semilla de estado usando el modo propio asociado a `(j omega0)^q`.

### 3.2 Saturacion no suave

Ganancia residual:

```text
g = m0 - m1
```

Funcion descriptiva:

```text
N(A) = g,                                           si A <= 1
N(A) = (2g/pi) [ asin(1/A) + sqrt(A^2 - 1)/A^2 ],  si A > 1
```

Compatibilidad:

```text
sign(k) = sign(g)
0 < |k| <= |g|
```

### 3.3 Arctan suave

No linealidad residual:

```text
psi(sigma) = a2 atan(rho sigma)
```

Funcion descriptiva:

```text
N(A) = a2 * 2 * (sqrt(1 + (rho A)^2) - 1) / (rho A^2)
```

Compatibilidad:

```text
sign(k) = sign(a2)
0 < |k| < |a2| rho
```

Amplitud cerrada usada para arctan:

```text
A0^2 = 4 a2 (a2 rho - k) / (k^2 rho)
```

---

## 4. Continuacion numerica

La continuacion usa una homotopia entre el sistema lineal corregido y el sistema no lineal completo.

Matriz lineal estabilizada:

```text
P0 = P + k b r^T
```

Campo deformado:

```text
f_eta(x) = P0 x + eta b [psi(r^T x) - k r^T x]
```

Contrato usado en los barridos de la raiz:

| Campo | Valor |
| :--- | :--- |
| `eta_values` | `0.0, 0.1, ..., 1.0` |
| Integrador de continuacion | `ABM` Caputo |
| `h` | `0.01` |
| Tiempo transiente por paso | `30 s` |
| Tiempo retenido por paso | `30 s` |
| `memory_mode=full` | conserva historia Caputo completa |
| `memory_mode=window` | conserva ventana; en estos barridos `Lm=10 s` |

Las corridas arctan de `version_2` quedan fuera del analisis de resultados de este archivo.

---

## 5. Integradores finales

| Integrador | Donde se uso | Papel | Memoria |
| :--- | :--- | :--- | :--- |
| `ABM` | `search_arctan_fractional.py`, `search_saturation_candidates.py` | integrador principal de dinamica fraccionaria final | completa o ventana |
| `EFORK3` | `compare_solvers_saturation.py`, preset `version_2` | reintegracion/comparacion de robustez | completa o ventana |
| `ADM_WU2023` | reproduccion publicada de Wu 2023 | reproduccion local del articulo | sin historia Caputo acumulada |
| `abm_restart` | `version_2/tools/search_arctan_full_memory_candidates.py` | control exploratorio | reinicio por punto final |
| `adm_restart` | `version_2/tools/search_arctan_full_memory_candidates.py` | control exploratorio tipo Wu | reinicio local |

Simulacion final registrada para la parte analizada aqui:

| Familia | `q` | `h` | `t_final` | `t_transient` | Integrador |
| :--- | :---: | :---: | :---: | :---: | :--- |
| saturacion | `0.9998` | `0.01` | `300 s` | `100 s` | `ABM` |
| saturacion comparativa | `0.9998` | `0.01` | `300 s` | `100 s` | `ABM`, `EFORK3 full`, `EFORK3 window` |

---

## 6. Resultados actuales de saturacion fraccionaria

Salidas:

- `outputs/saturation_search_seed1_mem_full_sweep/summary.csv`
- `outputs/saturation_search_seed0p9998_mem_full_sweep/summary.csv`
- `outputs/saturation_search_seed0p9998_mem_window_sweep/summary.csv`

Conteos por corrida:

| Corrida | `chaotic_candidate_pending_robustness` | `nonperiodic_candidate` | `regular_periodic_rejected` | `continuation_failed` |
| :--- | ---: | ---: | ---: | ---: |
| semilla `q=1`, memoria completa | 1 | 6 | 5 | 9 |
| semilla `q=0.9998`, memoria completa | 1 | 1 | 5 | 14 |
| semilla `q=0.9998`, memoria ventana | 1 | 1 | 5 | 14 |

Candidato robusto en saturacion:

| `case_id` | `m1` | `m0` | `omega0` | `k` | `A0` | Estado |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `m1_m1p2000_m0_m0p2000_branch_0` | `-1.2` | `-0.2` | `2.040286` con `seed_q=0.9998`; `2.039187` con `seed_q=1` | `0.263223` aprox. | `4.801924` aprox. | candidato fuerte; persiste en memoria completa y ventana |

---

## 7. Graficas de candidatos de saturacion

![Resumen saturacion memoria completa](outputs/chaotic_candidates_plots/summary_sat_full.png)

![Resumen saturacion memoria ventana](outputs/chaotic_candidates_plots/summary_sat_win.png)

![Saturacion fuerte, memoria completa](outputs/chaotic_candidates_plots/sat_full/m1_m1p2000_m0_m0p2000_branch_0_detailed.png)

![Saturacion fuerte, memoria ventana](outputs/chaotic_candidates_plots/sat_win/m1_m1p2000_m0_m0p2000_branch_0_window_detailed.png)

![Comparacion ABM y EFORK del candidato de saturacion](outputs/saturation_comparison/m1_m1p2000_m0_m0p2000_branch_0_four_solvers_phase3d.png)

---

## 8. Siguiente filtro antes de ocultedad

Antes de llamar oculto a cualquiera de estos candidatos:

1. Reintegrar desde la semilla continuada y desde vecindades de los equilibrios.
2. Ejecutar diagnosticos 0-1, espectro/FFT, Poincare y Lyapunov bajo un contrato numerico registrado.
3. Para Caputo `q<1`, no usar `integer_qr_benettin` como cierre.
4. Mantener separado el resultado `published_integer_laplace` de los candidatos `fractional_spectral`.
5. Reportar si la memoria fue `full`, `window`, `restart` o `none_local_adm`.
