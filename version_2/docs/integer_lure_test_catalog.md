# Catálogo de pruebas Lur'e de orden entero

Este archivo es la **única cola canónica** de sistemas candidatos para las
pruebas de orden entero (`q = 1`). La inclusión de un sistema sólo certifica
que su estructura puede escribirse como Lur'e escalar de forma directa o
mediante una transformación compatible; no certifica que ya se haya localizado
un atractor oculto.

Los resultados fraccionarios no pertenecen a esta lista. Se conservarán en una
cola separada para una etapa posterior.

## Alcance estructural

Se admiten en la fase actual:

- **D0 -- Lur'e escalar directa:**
  \(\dot x=Ax+b\psi(c^{\mathsf T}x)\).
- **T1 -- transformación compatible:** un desplazamiento de equilibrio, una
  transformación de estado invertible, una restricción documentada a una
  variedad invariante o una representación PWA con saltos de rango uno lleva
  al sistema escalar anterior.
- **A1 -- ruta teórica alternativa:** la representación escalar existe, pero
  la ruta directa de transferencia/función descriptiva se intenta y se rechaza
  por una razón matemática registrada. Sólo entonces se usa una construcción
  publicada, por ejemplo un mapa de conmutación o una homotopía entre
  nonlinearidades.

Los sistemas que necesitan varios canales no lineales independientes se marcan
como **M1** y no entran todavía en el contrato escalar de `LureSystem`.

## Lote principal de pruebas

| Prioridad | Sistema | Clase | Papel científico | Estado reproducible |
|---:|---|---|---|---|
| Base | Chua no suave entero | D0 | candidato oculto bajo contrato finito | validado en Python y Julia |
| Base no-Chua | Kalman--Fitts de cuarto orden | D0 estructural + A1 | ciclo límite oculto; valida continuación alternativa | validado en Python y Julia |
| 1 | Van der Pol--Duffing autónomo modificado | D0 | candidato periódico para `xi=3.1`, control negativo para `xi=3.5` y búsqueda caótica separada en un punto local | auditorías periódica y caótica validadas en Python y Julia bajo contratos finitos |
| 2 | PLL bifásico con filtro lead--lag | T1 + A1 | oscilación periódica oculta en una variable angular | comparaciones Python--Julia `quick` y `full` aprobadas, ambas con 96 sondeos finitos |
| 3 | Leonov--Kuznetsov, sistema (162) | D0 estructural + A1 | ciclo límite oculto mediante saturación y continuación | ecuaciones y semilla disponibles; falta congelar el manifiesto |
| 4 | Jerk exponencial de Fischer, `q=1` | D0 con balance DC | control negativo: atractor autoexcitado, no oculto | RHS y benchmark locales; conserva discrepancias publicadas/locales |

Las prioridades 1--4 conservan el orden de trabajo. Los dos casos base, las
auditorías MAVPD periódica y caótica y PLL `quick/full` ya tienen comparación
cruzada. En PLL no se mezclaron contratos: `quick` se comparó con `quick` y
`full` con `full`. Leonov--Kuznetsov y Fischer permanecen en cola.

## Fichas de los casos priorizados

### 1. Van der Pol--Duffing autónomo modificado

El sistema entero de tercer orden es

\[
\begin{aligned}
 \dot y_1&=\delta\gamma y_1+\delta y_2-\delta y_1^3,\\
 \dot y_2&=y_1-\xi y_2-y_3,\\
 \dot y_3&=\rho y_2.
\end{aligned}
\]

Es D0 con

\[
A=\begin{pmatrix}
\delta\gamma&\delta&0\\
1&-\xi&-1\\
0&\rho&0
\end{pmatrix},\qquad
b=\begin{pmatrix}-\delta\\0\\0\end{pmatrix},\qquad
c=\begin{pmatrix}1\\0\\0\end{pmatrix},\qquad
\psi(\sigma)=\sigma^3.
\]

La fuente publica \(\gamma=0.1\), \(\delta=100\), \(\rho=200\) y casos con
\(\xi=3.5\) y \(\xi=3.1\). También proporciona resultados de
\(\omega_0\), ganancia y semilla. Esos números se usarán únicamente como
regresiones posteriores: la ejecución deberá recalcularlos a partir de las
ecuaciones, el Jacobiano, la transformación Lur'e y la transferencia.

La reproducción completa ya distingue los dos casos. Para \(\xi=3.1\),
Python y Julia obtienen el mismo candidato periódico y ninguno de los 108
sondeos comunes alrededor de los tres equilibrios lo alcanza. Para
\(\xi=3.5\), 24 sondeos alcanzan cada ciclo simétrico en ambas
implementaciones; por ello no se conserva la etiqueta oculta bajo ese
contrato. La dinámica local de \(\xi=3.1\) resultó periódica, no
cuasiperiódica como declara la fuente, y la discrepancia queda registrada.

Fuente primaria: [Mathematics 11, 591](https://doi.org/10.3390/math11030591).

#### Ruta caótica MAVPD separada

`examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/run_example.py`
recalcula primero las dos semillas directas en el caso base
`(xi,gamma)=(3.1,0.1)`. Después de la continuación ordinaria, ninguna rama
supera el cribado de caos de Lyapunov de tiempo finito declarado. Ésta es la
única conclusión negativa usada para activar la alternativa: no se promueve
otra etiqueta; sólo se registra que cada rama no supera el cribado finito, sin
afirmar una clasificación asintótica.

La alternativa transporta el estado primero hasta `xi=2.85` y después en
`gamma`, con el sistema y su descomposición Lur'e reconstruidos en cada nodo.
La frontera superior de Hopf se deriva de la ecuación de Routh--Hurwitz activa
y sólo se prueban siete desplazamientos declarados. El punto mantenido,
`gamma=0.1538037983994911`, corresponde a `gamma_H+0.010`; es un resultado
local de esta continuación y del cribado numérico, no una combinación de
parámetros publicada.

La implementación etapa por etapa es:

| Etapa | Script o función principal | Integrador o cálculo |
|---|---|---|
| Ecuaciones, Lur'e, transferencia, función descriptiva y dos semillas | `mavpd_2023_system`, `integer_lure_seed` | álgebra y raíces del polinomio racional; sin integración ni malla de frecuencias |
| Continuación ordinaria en `lambda` de cada rama | `continue_integer_lure_seed` | `efork_q1_integrate`, paso fijo y sin memoria fraccionaria |
| Cribado de las ramas base | `integer_system_dop853_variational_qr` | DOP853 adaptativo para estado y ecuaciones variacionales con QR |
| Continuación de parámetros en `xi` y `gamma` | `continue_integer_parameter_path` | `dop853_q1_integrate`; reconstrucción completa del sistema en cada nodo |
| Cribado de desplazamientos de Hopf | `integer_system_dop853_variational_qr`, `calibrate_attractor_reference`, `run_integer_hidden_chaos_controls` | DOP853 adaptativo para exponente, nube candidata y sondeos finitos de `E0` |
| Trayectoria y diagnóstico estricto del punto seleccionado | `dop853_q1_integrate`, `integer_system_dop853_variational_qr` | DOP853 adaptativo; 0--1, Poincaré y FFT son posprocesos, no integradores |
| Control independiente del integrador | `efork_q1_integrate` | EFORK entero de paso fijo, comparado contra la nube calibrada |
| Ocultedad y decisión conjunta | `run_integer_hidden_chaos_controls`, `evaluate_candidate_gate` | DOP853 para 108 sondeos principales y cuatro dirigidos; el gate no integra |

El gate separa acotación, evidencia de caos, robustez y ocultedad muestreada.
La etiqueta máxima es `chaotic_hidden_under_tested_neighborhoods`: no prueba
una separación global de cuencas. Julia reprodujo independientemente la ruta
caótica en `src/mavpd_hidden_chaos_demo.jl`: recalculó las dos ramas, activó la
alternativa tras su fallo finito, volvió a seleccionar `xi=2.85`,
`gamma=0.15380379839949113` y obtuvo 112/112 decisiones coincidentes. El
comparador `output/comparison/mavpd_integer_hidden_chaos_comparison_full.json`
se ejecuta sólo después y no suministra semilla, estado terminal ni parámetros
a ninguna localización. Los tiempos son descriptivos, no un benchmark general.

### 2. PLL bifásico con filtro lead--lag

La forma publicada es

\[
\dot x=-\frac{x}{\tau_1+\tau_2}
+\left(1-\frac{\tau_2}{\tau_1+\tau_2}\right)\frac{\sin\theta}{2},
\]

\[
\dot\theta=\omega_\Delta-\frac{Lx}{\tau_1+\tau_2}
-\frac{\tau_2L}{\tau_1+\tau_2}\frac{\sin\theta}{2}.
\]

Se clasifica T1 porque se desplaza un equilibrio bloqueado para eliminar el
término afín. La salida no lineal es \(\sin\theta\), pero \(\theta\) vive en
un cilindro; la recurrencia, la distancia entre trayectorias y los sondeos de
ocultedad deben usar distancia angular módulo \(2\pi\).

La ruta directa también tiene un rechazo analítico reproducible. Para la
transferencia normalizada \(G(s)=c^{\mathsf T}(sI-A)^{-1}b\),

\[
 G(s)=-\frac{L}{2}\frac{1+\tau_2s}{s(1+(\tau_1+\tau_2)s)},
\]

se cumple \(\operatorname{Im}G(i\omega)>0\) para toda \(\omega>0\). No hay
cruce de Nyquist compatible y un barrido de frecuencias no puede crearlo. La
ruta A1 promovible es la transformación exacta de Andronov

\[
(\tau_1+\tau_2)\dot y=\omega_\Delta-\frac{L}{2}\sin\theta
-\left(1+\frac{\tau_2L}{2}\cos\theta\right)y,
\qquad y=\dot\theta,
\]

seguida por un mapa de retorno en \(\theta=0\pmod{2\pi}\) y continuación
desde \(L=0\) hasta \(L=500\). Esta ruta deriva tanto el ciclo estable como
la órbita inestable separatriz sin usar las condiciones iniciales publicadas.

Datos publicados para la regresión posterior:
\(\tau_1=0.0448\), \(\tau_2=0.0185\), \(L=500\),
\(\omega_\Delta=178.9\), horizonte de 5 s. Las condiciones iniciales
publicadas se conservarán como controles posteriores y no como sustituto de la
ruta de localización.

Fuente primaria: [ICUMT 2015](https://doi.org/10.1109/ICUMT.2015.7382409).

### 3. Leonov--Kuznetsov, sistema (162)

\[
\begin{aligned}
 \dot x_1&=-x_2-10\phi(\sigma),\\
 \dot x_2&=x_1-10.1\phi(\sigma),\\
 \dot x_3&=x_4,\\
 \dot x_4&=-x_3-x_4+\phi(\sigma),\\
 \sigma&=x_1-10.1x_3-0.1x_4.
\end{aligned}
\]

Es una representación D0 exacta. La fuente construye primero una solución con
una familia de saturaciones \(\varepsilon_j=0.1,0.2,\ldots,1\), transporta el
estado terminal entre etapas y después continúa la no linealidad hasta
`tanh`. La semilla analítica publicada se guarda como regresión; deberá
recalcularse en cada ejecución.

Antes de implementarlo deben verificarse visualmente en la fuente la fórmula
por tramos completa de \(\phi_j\) y el criterio de parada de cada etapa.

Fuente primaria: [International Journal of Bifurcation and Chaos 23,
1330002](https://doi.org/10.1142/S0218127413300024).

### 4. Jerk exponencial de Fischer como control negativo

\[
 \dot x=y,\qquad \dot y=z,\qquad
 \dot z=-az-I_c\left(e^{y/(nV_T)}-1\right)-x.
\]

Es D0 con salida \(c^{\mathsf T}x=y\) y una no linealidad exponencial que
requiere balance DC. El código y los parámetros de benchmark ya existen en
`hidden_attractors/systems/fischer_benchmarks.py`.

Este sistema se usará como **prueba negativa de ocultedad**: el equilibrio
entero es inestable y el atractor se clasifica como autoexcitado. La
discrepancia local ya documentada en el tercer exponente de Lyapunov debe
mantenerse visible; no se reinterpretará como una reproducción completa.

Fuente primaria: [Applied Numerical Mathematics 154,
187--204](https://doi.org/10.1016/j.apnum.2020.03.027).

## Referencias ya reproducidas

| Sistema | Ruta ejecutada | Evidencia almacenada |
|---|---|---|
| Chua no suave entero | transferencia y función descriptiva directas, semilla recalculada, continuación ordinaria y 504 sondeos comunes | Python: `outputs/examples/chua_integer_lure_reference_clean_full/`; Julia: `output/chua_integer_nonsmooth/full/`; comparación: `output/comparison/chua_integer_comparison.json` en el proyecto Julia |
| Kalman--Fitts | diagnóstico directo incompatible, mapa de conmutación `sign` y continuación `sign -> tanh`; la semilla publicada no se usa como entrada | Python: `validation/reference_cases/kalman_fitts_integer_q1/`; Julia: `output/kalman_fitts_integer/full/`; comparación: `output/comparison/kalman_fitts_integer_comparison.json` |
| Van der Pol--Duffing autónomo modificado | dos ramas directas recalculadas y auditoría periódica; la alternativa caótica se activa sólo porque ninguna rama base supera el cribado finito, sin asignarle por ello una clasificación asintótica | Python periódico: `examples/modified_van_der_pol_duffing_integer_lure_audit/`; Python caos oculto: `examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/` y `validation/reference_cases/mavpd_integer_hidden_chaos/`; Julia: `output/mavpd_integer/` y `output/mavpd_integer_hidden_chaos/full/`; comparador caótico: `output/comparison/mavpd_integer_hidden_chaos_comparison_full.json` |
| PLL bifásico lead--lag | rechazo analítico de la ruta directa, ciclo de Andronov desde `L=0`, continuación a `L=500`, separatriz y 96 sondeos cilíndricos | Python `quick`: `validation/reference_cases/pll_lead_lag_integer_q1/`; Python `full`: `validation/reference_cases/pll_lead_lag_integer_q1_full/`; Julia `quick/full`: `output/pll_lead_lag_integer/`; comparadores: `output/comparison/pll_lead_lag_integer_comparison_{quick,full}.json` |

## Reservas pendientes

| Sistema | Compatibilidad | Motivo para no promoverlo todavía |
|---|---|---|
| Genesio--Tesi | D0 exacta, \(\psi(\sigma)=\sigma^2\) | necesita balance DC y segundo armónico; faltan parámetros e IC enteros congelados desde fuente primaria |
| Oscilador memristivo cúbico | T1 sobre una variedad invariante | faltan parámetros, constante de la variedad e IC verificados |
| Jerk hiperbólico de Joshi--Ranjan | D0 estructural | persiste la ambigüedad documental entre `sinh` y la función hiperbólica inversa, además de parámetros/IC incompletos |
| Jerk PWL de Barajas--Ramírez y Ponce--Pacheco | D0 PWA | el caso exitoso usa una característica explícitamente tipo diodo de Chua y hay dos juegos de IC que deben resolverse; queda como control, no como nuevo no-Chua |
| Jerks de Rech y Rasul--Salih | por certificar | falta congelar desde las fuentes las ecuaciones exactas y verificar si hay uno o varios canales no lineales |

No deben reutilizarse parámetros de publicaciones fraccionarias imponiendo
simplemente `q = 1`; cada caso entero necesita su propio contrato publicado o
un experimento nuevo declarado como tal.

## Sistemas fuera del contrato escalar actual

Lorenz, Rabinovich--Fabrikant, el sistema financiero, el sistema four-wing y
los jerks con productos independientes como \(xy\), \(x^2\) o
\(z\operatorname{erf}(z)\) requieren una representación Lur'e multicanal M1.
Pueden estudiarse más adelante, pero no deben mezclarse con esta batería
escalar ni presentarse como fallos de la ruta D0/T1.

## Reglas de promoción y cierre cruzado

Para registrar una **promoción científica provisional en Python**, el ejemplo
reproducible debe declarar y ejecutar:

1. ecuaciones, parámetros, todos los equilibrios y Jacobiano;
2. transformación Lur'e y función de transferencia;
3. ruta directa sin barrido de frecuencias como primer intento;
4. cálculo nuevo de \(\omega_0\), ganancia, amplitud y semilla;
5. continuación y candidato final;
6. diagnóstico de periodicidad o caos;
7. sondeos de ocultedad alrededor de todos los equilibrios bajo un contrato
   Python explícito y finito;
8. tiempos por etapa y tiempo de pared;
9. YAML, comandos, versiones, figuras y resultados regenerables.

Ese nivel permite conservar un candidato Python con una etiqueta finita y
condicionada, como `chaotic_hidden_under_tested_neighborhoods`; no autoriza
atribuirle comparación cruzada ni reproducción independiente.

El **cierre/comparación cruzada Python--Julia** es un nivel posterior. Requiere
que Julia repita la construcción algebraica, la localización, la continuación
y los sondeos de ocultedad con el mismo contrato de ecuaciones, parámetros,
radios, direcciones, horizontes y criterio de clasificación, y que se archive
la comparación de resultados. La ausencia de ese cierre no impide registrar
la promoción provisional Python, pero debe permanecer explícita en el estado
del caso.

Los valores publicados o calculados previamente con Mathematica/MATLAB son
controles de regresión posteriores, nunca entradas operativas. Los barridos de
frecuencia, condiciones iniciales o semillas, las transferencias sesgadas y la
continuación multiparámetro son rutas alternativas explícitas y sólo se activan
después de documentar por qué falló la ruta principal.

La ocultedad siempre se reporta como una conclusión numérica finita ligada a
radios, muestreo, horizonte, integrador y tolerancias. No constituye una prueba
global de separación de cuencas.
