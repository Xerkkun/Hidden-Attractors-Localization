# Reconstrucción por retardos y recurrencia avanzada

Estado: `experimental`. Este bloque trabaja sobre trayectorias muestreadas y es
común a sistemas enteros y fraccionarios. Sus resultados son diagnósticos
empíricos de muestra finita; no demuestran caos, hiddenness ni que el estado
hereditario de una FDE posea un embedding difeomorfo finito.

## Embedding generalizado

`generalized_delay_embedding` recibe pares explícitos `(columna, lag)`. Para un
índice ancla `i`, la coordenada correspondiente es

\[
y_k(i)=x_{c_k}[i-\tau_k].
\]

Un `lag` positivo mira al pasado y uno negativo al futuro. La función calcula la
intersección exacta de índices válidos y devuelve tanto `anchor_indices` como
`aligned_times`; esto evita perder la alineación al comparar una reconstrucción
con eventos u otra trayectoria. Se admiten tiempos irregulares, pero en ese caso
no se inventa una única duración física para un retardo expresado en muestras.

```python
from hidden_attractors.analysis.delay_embedding import generalized_delay_embedding

embedding = generalized_delay_embedding(
    trajectory,
    ((0, 0), (1, 4), (0, 8)),
    dt=0.01,
    time_unit="s",
)
vectors = embedding.vectors
```

Esta API cubre el núcleo de `genembed`/`GeneralizedEmbedding` sin depender de
Julia. PECUZAL y selectores multivariados unificados siguen reservados para una
extensión opcional porque combinan más estimadores y decisiones.

## Selección de retardo

### Autocorrelación

`estimate_delay_autocorrelation` calcula mediante FFT zero-padded

\[
\rho(k)=\frac{\sum_{i=0}^{N-k-1}(x_i-\bar x)(x_{i+k}-\bar x)}
{\sum_{i=0}^{N-1}(x_i-\bar x)^2}.
\]

La normalización global —no la versión “unbiased” dividida por `N-k`— queda en
los metadatos. Puede seleccionarse el primer cruce positivo/no positivo o el
primer mínimo local. Si el criterio no aparece dentro de `max_lag`, se devuelve
`criterion_not_found`, no un retardo fabricado.

Con tiempos irregulares ACF, MI y FNN siguen usando retardos de índice; el
resultado conserva `time_source` y advierte que el lag no es un tiempo físico
constante.

### Información mutua

`estimate_delay_mutual_information` evalúa el estimador histográfico

\[
I(k)=\sum_{a,b}p_{ab}(k)\log\frac{p_{ab}(k)}{p_a(k)p_b(k)}
\]

en *nats*. Es un estimador *plug-in* de bins fijos, no la partición adaptativa
recursiva del artículo original. Los bordes se calculan una vez y se reutilizan
en todos los retardos. `MI(0)` se incluye sólo al seleccionar para que `lag=1`
pueda ser un mínimo; la salida conserva retardos positivos. `minimum_pairs=16`
evita colas con solapamiento ínfimo. El primer mínimo usa el criterio de
Fraser--Swinney; el mínimo global sólo aparece mediante fallback explícito.

## Falsos vecinos cercanos

`false_nearest_neighbors` construye, para dimensión `m`, retardos
`0,tau,...,(m-1)tau` y comparte los mismos anclajes con la extensión `m+1`. Un
vecino pasa a falso si se activa alguna prueba Kennel--Brown--Abarbanel:

\[
\frac{|\Delta x_{m+1}|}{R_m}>R_{tol},
\qquad
\frac{R_{m+1}}{\sigma_x}>A_{tol}.
\]

La búsqueda usa `scipy.spatial.cKDTree`, una ventana de Theiler sobre índices
fuente y desempate determinista. La norma euclídea queda etiquetada KBA; Manhattan
y Chebyshev son `generalized_lp_fnn_using_kba_threshold_form`. Cada dimensión
reporta muestras, vecinos y estado. Por defecto se exigen 20 comparaciones
válidas antes de que una dimensión pueda seleccionarse.

La FNN puede dar respuestas engañosas en ruido o procesos aleatorios. Deben
inspeccionarse la curva completa, sensibilidad a `tau`, Theiler, umbrales,
longitud y ruido; el primer cruce de un umbral no es una prueba topológica.

## Matrices de recurrencia

`recurrence_advanced` implementa tres contratos:

- auto-recurrencia `R_ij = 1[||x_i-x_j|| <= epsilon]`;
- cross-recurrencia rectangular entre dos trayectorias de igual dimensión;
- recurrencia conjunta como AND de auto-recurrencias sincronizadas.

Las normas euclídea, Manhattan y Chebyshev tienen rutas Numba y NumPy. El cálculo
se procesa por bloques para evitar un tensor `N×M×d`, aunque la matriz booleana
densa final sigue requiriendo `O(NM)` memoria y se protege con `max_bytes`.

El umbral puede ser un radio fijo o una tasa global objetivo. En el segundo caso
se selecciona el menor estadístico de orden que alcanza el conteo solicitado y
se incluyen todos los empates; la tasa lograda puede por ello superar la meta.
La selección guarda su propio límite `max_threshold_bytes` porque actualmente
materializa los `NM` valores elegibles. No se anuncia todavía una matriz sparse.

En recurrencia conjunta, `target_rate` selecciona actualmente **un radio común**
sobre la máxima distancia entre componentes y ajusta la tasa de la matriz AND
final. Esto exige que las trayectorias componentes tengan escalas comparables o
se normalicen de forma declarada. No equivale a la política habitual de
RecurrenceAnalysis.jl que puede fijar un radio por componente antes del AND;
esa variante `componentwise` continúa pendiente.

```python
from hidden_attractors.analysis.recurrence_advanced import (
    auto_recurrence_matrix,
    recurrence_quantification_advanced,
)

rp = auto_recurrence_matrix(
    vectors,
    target_rate=0.03,
    metric="euclidean",
    theiler_window=20,
)
rqa = recurrence_quantification_advanced(rp, min_diagonal=2, min_vertical=2)
```

`theiler_window=w` excluye `|i-j|<=w`; `None` conserva todas las entradas. Por
tanto HAFO `w=0` excluye la LOI, mientras RecurrenceAnalysis.jl `theiler=0` la
incluye y `theiler=1` la excluye. La convención queda en metadatos. Para una
cross-recurrencia rectangular sólo tiene sentido tras alinear tiempos físicos.
Si no queda ninguna entrada elegible, la construcción se rechaza. La matriz del
resultado es no escribible para que sus conteos y su RQA no diverjan por mutación.

## RQA y normalizaciones

La salida avanzada conserva la matriz y calcula:

- tasa de recurrencia `RR` sobre entradas elegibles;
- determinismo `DET`, longitud diagonal media, `Lmax`, `DIV=1/Lmax` y entropía
  de longitudes diagonales;
- laminaridad `LAM`, tiempo de atrapamiento `TT`, `Vmax` y entropía vertical;
- `TREND` clásico como pendiente de densidad diagonal por 1000 índices, con
  `trend_border=10` y recorte a la mayor matriz cuadrada desde la LOI;
- `normalized_absolute_trend` como la anterior variante HAFO, ahora nombrada por
  separado, contra separación absoluta normalizada y sin factor `1000`.

Las entropías usan logaritmo natural y sólo líneas que cumplen `lmin`/`vmin`;
una distribución vacía produce `NaN`, no cero. Sin diagonal válida,
`DIV=inf`. En matrices demasiado pequeñas para Theiler y `trend_border`, TREND
es `NaN`. Ambas convenciones de tendencia quedan documentadas porque las
implementaciones históricas difieren en exclusiones, normalización y escala.

## Uso con orden fraccionario

La reconstrucción y RQA pueden aplicarse a una señal generada por Caputo, RL,
GL, templada u otra formulación sólo después de obtener una trayectoria válida.
Los metadatos del experimento deben conservar derivada, orden, terminal,
prehistoria, método y política de memoria. La proximidad de dos vectores
proyectados no implica proximidad de sus estados históricos completos.

## Pruebas y referencias

`tests/test_delay_embedding_advanced.py` cubre alineación exacta, tiempos
irregulares, ACF, información mutua, FNN, Theiler, métricas y falta de vecinos.
`tests/test_recurrence_advanced.py` fija matrices pequeñas exactas, cross/joint,
empates de tasa, paridad Numba--NumPy, métricas RQA y guardas de memoria.

- F. Takens, “Detecting strange attractors in turbulence”, LNM 898 (1981),
  [DOI 10.1007/BFb0091924](https://doi.org/10.1007/BFb0091924).
- A. M. Fraser y H. L. Swinney, “Independent coordinates for strange
  attractors from mutual information”, *Physical Review A* 33 (1986),
  [DOI 10.1103/PhysRevA.33.1134](https://doi.org/10.1103/PhysRevA.33.1134).
- M. B. Kennel, R. Brown y H. D. I. Abarbanel, “Determining embedding
  dimension for phase-space reconstruction using a geometrical construction”,
  *Physical Review A* 45 (1992),
  [DOI 10.1103/PhysRevA.45.3403](https://doi.org/10.1103/PhysRevA.45.3403).
- J.-P. Eckmann, S. O. Kamphorst y D. Ruelle, “Recurrence Plots of Dynamical
  Systems”, *Europhysics Letters* 4 (1987),
  [DOI 10.1209/0295-5075/4/9/004](https://doi.org/10.1209/0295-5075/4/9/004).
- N. Marwan, M. C. Romano, M. Thiel y J. Kurths, “Recurrence plots for the
  analysis of complex systems”, *Physics Reports* 438 (2007),
  [DOI 10.1016/j.physrep.2006.11.001](https://doi.org/10.1016/j.physrep.2006.11.001).
- [Documentación oficial de RecurrenceAnalysis.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/recurrenceanalysis/stable/).
