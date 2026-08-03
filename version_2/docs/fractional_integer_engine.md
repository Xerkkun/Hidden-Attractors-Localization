# Motor unificado entero y fraccionario de HAFO

Estado: núcleo experimental ejecutable. No constituye por sí mismo evidencia de
caos, atracción u ocultedad.

## Decisión de arquitectura

HAFO mantiene el contrato matemático y los algoritmos fraccionarios dentro del
proyecto. No importa ni copia `pynamicalsys`: su API es una referencia funcional,
pero su licencia GPL-3.0 no es adecuada para incorporarla al núcleo permisivo de
HAFO. El ecosistema `DynamicalSystems.jl` se reserva para adaptadores opcionales
por lotes y validación cruzada.

La frontera numérica es:

| Trabajo | Backend preferido | Razón |
| --- | --- | --- |
| RHS/Jacobiano de usuario, mapas y prototipos | Numba | Mantiene callbacks genéricos en modo compilado y un fallback Python verificable. |
| Convolución histórica, memoria y ensambles | C/OpenMP | Evita el callback Python, vectoriza y paraleliza bucles estables. |
| Convolución offline larga | FFT de SciPy/NumPy | Reduce el costo asintótico cuando toda la historia ya está disponible. |
| Matrices pequeñas SALI/GALI/LDI | NumPy/LAPACK de referencia y Numba/Householder caliente | C queda diferido hasta que un benchmark de ensambles demuestre ventaja; Julia sólo compara trayectorias completas. |
| Catálogos avanzados de análisis | JuliaCall opcional | Una llamada gruesa por trayectoria; nunca una llamada Julia por paso del solver. |

El documento [matriz upstream](upstream_function_matrix.md) contiene la auditoría
función por función y las fronteras de licencia.

## Contrato matemático

Un escalar `q` no define un problema fraccionario. HAFO separa:

1. definición de derivada y kernel;
2. orden conmensurado, por componente, variable o distribuido;
3. terminal inferior y, cuando corresponda, prehistoria;
4. semántica de condición inicial;
5. discretización;
6. política de memoria;
7. backend y nivel de evidencia.

`FractionalProblem` implementa este contrato para solvers de trayectoria. El
registro distingue además `execution_kind="sampled_operator"`, de modo que una
derivada numérica válida no pueda ejecutarse accidentalmente como solver FDE.

```python
from hidden_attractors.fractional import FractionalProblem, solve_fractional_problem

problem = FractionalProblem(
    derivative="caputo",
    method="caputo_abm_pece",
    orders=[0.92, 0.92, 0.92],
    initial_state=[0.1, 0.0, 0.0],
    step=0.001,
    t_span=(0.0, 10.0),
    memory_policy="full_history",
)
result = solve_fractional_problem(problem, rhs, parameters)
```

La duración debe contener un número entero de pasos; el contrato no excede
silenciosamente `t_span` ni modifica el paso solicitado.

## Definiciones y métodos actuales

| Definición | Ruta numérica | Estado | Alcance |
| --- | --- | --- | --- |
| Caputo | ABM/PECE | implementado | Solver de trayectoria; historia completa o ventana declarada. |
| Caputo | EFORK-3 | implementado | Solver conmensurado; memoria completa o finita. |
| Caputo/GL | recurrencia GL explícita | experimental | Solver discreto con inicialización `caputo_shifted` o `discrete_gl`. |
| GL / Riemann–Liouville | convolución directa | implementado/experimental | Operador muestreado; no convierte `x(t0)` en dato inicial RL. |
| Riemann–Liouville templada | GL templada | experimental | Conjugación exponencial sin corrección oculta; no es Caputo templada. |
| Caputo templada | conjugación exponencial + ABM/PECE amortiguado | experimental | Solver conmensurado en estado físico; historia completa o ventana declarada, con ruta C/Python. |
| GL de orden variable | GL directa con `q(t_n)` | experimental | Convención concreta; otras definiciones de orden variable no son equivalentes. |
| Caputo de orden variable tipo III | L1 implícito + corrector Picard | experimental | Solver con orden temporal prescrito, historia completa `O(N²)`, suma Numba/Python y fallo estructurado. |
| Orden distribuido | cuadratura en orden + GL en tiempo | experimental | Operador con doble discretización, pesos y densidad explícitos. |
| Caputo multitérmino | fachada semántica + kernel L1 combinado | experimental | Suma finita atómica con coeficientes no negativos sin normalización; reutiliza el solver distribuido y conserva el límite backward Euler. |
| Hadamard / Caputo--Hadamard | CQ BDF1/BDF2 en `log(t/a)` | experimental | Operador en malla física exponencial; terminal `a>0`, sin solver FDE. |
| Caputo--Hadamard | ABM/PECE en `log(t/a)` | experimental | Solver conmensurado de historia completa; malla física no uniforme. |
| Conformable de Khalil | reescalamiento de `f'(t)` | experimental | Operador local, sin memoria hereditaria. |
| Conformable de Khalil | RK4 en `tau=(t-a)^q/q` | experimental | Solver local conmensurado; paso uniforme en reloj conformable y malla física no uniforme. |
| Caputo–Fabrizio | recurrencia exponencial exacta por intervalo | `research_required` | Operador de datos; normalización explícita y sin solver FDE promovido. |
| Atangana–Baleanu–Caputo | convolución por intervalos lineales | experimental | Operador de datos para `0<q<=1/2`; normalización explícita, Python/Numba/FFT y validación Wolfram finita. |
| Atangana–Baleanu–Caputo | predictor--corrector de Lee--Kim--Jang | experimental | Solver conmensurado para `0<q<1`, compatibilidad inicial e historia completa `O(N²)`; no SOE. |
| RL / Caputo templada | CQ BDF1/BDF2 directa o FFT | experimental | Operador muestral por conjugación, no solver; conserva ancla y pesos explícitos. |
| RL / Caputo templada | Fast Method II FBDF1/GNGF2 | experimental | Ventana exacta y cola recurrente real con tolerancia de compresión finita; GNGF2 no es BDF2 fraccionario. |
| RL / Caputo templada | CQ por símbolo desplazado y correcciones de arranque | planificado | Ruta adicional; no sustituye al ABM templado ni comparte silenciosamente los pesos de conjugación. |

El [catálogo de métodos](fractional_method_catalog.md) documenta fórmulas,
complejidad, pruebas y referencias de cada ruta.

## Operadores muestreados

### GL/RL y backend C

`grunwald_letnikov_derivative` ofrece la referencia Numba. Las funciones
`native_grunwald_letnikov_*` compilan bajo demanda una ABI C v1 con órdenes por
componente, historia completa o ventana, OpenMP y fallback estructurado a Numba.

```python
from hidden_attractors.fractional import native_grunwald_letnikov_derivative

derivative = native_grunwald_letnikov_derivative(
    samples,
    step=0.001,
    orders=[0.7, 0.8, 0.9],
    definition="riemann_liouville_gl",
    fallback=True,
)
```

La ventana finita cambia el operador; por ello queda registrada en el resultado
y nunca se presenta como una optimización matemáticamente neutra.

### Templada, variable y conformable

`riemann_liouville_gl_derivative` exige el token
`OPERATOR_ONLY_INITIAL_CONDITION`. `tempered_grunwald_letnikov_derivative`
implementa la definición RL templada por conjugación exponencial y se reduce
exactamente a GL cuando `tempering=0`.

`tempered_convolution_quadrature` amplía ese operador a BDF1/BDF2 y añade la
Caputo templada conjugada con el ancla discreta
`x0*exp(-lambda*n*h)*sum(omega)`. Admite `q_i` y `lambda_i` por componente y
backends Python/Numba directos o FFT *batch*. No materializa exponentes
positivos, no resta `lambda**q*x`, no implementa correcciones de arranque y no
es un solver. `lambda=0` delega exactamente a CQ ordinaria; BDF1--RL coincide
con GL templada.

`tempered_fast_multistep_history` es la ruta recurrente separada ya
ejecutable. Conserva los retardos recientes con pesos exactos y aproxima sólo
la cola mediante nodos reales, con estado
`y <- exp(-lambda*h)/(1+r) * (y+u)`. Admite FBDF1 y GNGF2, refina la cuadratura
hasta satisfacer el error L1 de todos los pesos comprimidos de la malla finita
y mantiene exacta la corrección Caputo. El costo es
`O(d*(Q+n0)*N)` y la memoria histórica activa `O(d*(Q+n0))`; el error reportado
es de compresión, no de CQ/FDE. GNGF2 sólo coincide con BDF2 en `q=1`.
La [derivación completa](tempered_fast_multistep_history.md) registra además la
validación Wolfram, el ejemplo Chua de postprocesamiento y los límites de
evidencia. La CQ por símbolo `[delta/h+lambda]**q` continúa como método futuro
independiente.

`integrate_tempered_caputo_abm` cubre una definición distinta: la identidad
`v=exp(tempering*(t-a))*x` produce factores históricos
`exp(-tempering*(t_n-t_j))`, que HAFO evalúa directamente sobre `x`. Admite orden
conmensurado, historia completa o una ventana de reinicio explícita. Devuelve
`states` físicos y sólo reconstruye `transformed_states` cuando esa representación
cabe en `float64`. La ruta acelerada ejecuta pesos e historia física en C con un
callback de RHS; la referencia usa NumPy. `FractionalProblem` exige
`derivative="tempered_caputo"`,
`method="tempered_caputo_abm_pece_transform"` y
`kernel_parameters={"tempering": lambda}`. La transformación está anclada en
[Li, Deng y Zhao (2019)](https://doi.org/10.3934/dcdsb.2019026), pero el método
discreto de HAFO es ABM de historia exponencialmente amortiguada, algebraicamente
equivalente a la transformación, no el predictor--corrector de Jacobi de esa
publicación. El umbral de divergencia se aplica durante la recurrencia en el
estado físico y los metadatos declaran la semántica exacta de la ventana.

`variable_order_grunwald_letnikov_derivative` usa el orden de la salida
`q(t_n)` para todos los pesos de esa suma histórica. Esta convención se conserva
en metadatos porque no es intercambiable con otras derivadas de orden variable.

`integrate_variable_order_caputo_type3_l1` es un solver distinto: fija la
definición Caputo tipo III de Tavares--Almeida--Torres y usa `alpha(t_n)` en el
kernel L1 completo de cada salida. Acepta programas temporales con firma
`alpha(t)`, `alpha(t, initial_state)` o
`alpha(t, initial_state, parameters)`; el contexto de estado es siempre una
copia fija del estado inicial, por lo que no se introduce un orden dependiente
del estado. La historia directa usa Numba sin `fastmath` o Python, cuesta
`O(N²*d)` de trabajo total (`O(N²)` por componente) y se conserva en
`O(N*d)`. El estado
implícito se resuelve por Picard y reporta tolerancias, iteraciones y
`corrector_nonconvergence`. Una declaración `smooth` activa el chequeo
`f(a,x0)=0`; `nonsmooth` no hereda la tasa L1 suave. El método se ejecuta también
mediante `FractionalProblem(derivative="caputo_variable_type3",
method="vo_caputo_type3_l1", ..., allow_experimental=True)`.

`conformable_khalil_derivative` calcula
`(t-a)**(1-q) * f'(t)` sobre una derivada ordinaria conocida. Exige una política
explícita en el terminal cuando `q<1` y no se etiqueta como operador con memoria.

`integrate_conformable_rk4` resuelve el modelo local mediante
`tau=(t-a)**q/q` y RK4 clásico. Devuelve `times` físicos y `clock_times`
uniformes; `FractionalProblem` conserva esa distinción mediante
`grid_coordinate="conformable_clock"`. El backend Numba requiere el ABI
`rhs(t, state, parameter_vector)`; el fallback Python acepta callables
declarativos. Esta reutilización de RK4 no convierte la definición conformable
en un operador con historia.

### Orden distribuido

`distributed_order_gl_derivative` combina nodos de orden, pesos de cuadratura y,
opcionalmente, una densidad. Los pesos firmados requieren una semántica firmada;
la normalización a masa unitaria rechaza cancelación algebraica. La ruta Numba
acumula un orden a la vez y evita un tensor `orden × tiempo × dimensión`.

`integrate_distributed_order_caputo_l1` es un carril distinto y ejecutable para
una medida Caputo discreta no negativa. Agrega antes de integrar un solo kernel
`K[k] = sum_r Omega[r] * h**(-alpha[r]) / Gamma(2-alpha[r]) * b[r,k]`, de modo
que el costo estructural es `O(R*N + N²*d)` y no `O(R*N²*d)`. El estado
corriente se obtiene con un corrector Picard auditado; una masa en `alpha=1` se
trata como backward Euler exacto. En `FractionalProblem`, `orders` son los nodos
independientes de la dimensión y `order_mode="distributed"`; los pesos,
densidad, normalización y nombre de regla quedan en `kernel_parameters`. Sólo
`full_history` está disponible y los errores temporal y de cuadratura no se
estiman automáticamente.

La validación focal comprende 86 casos directos y 6 pruebas Wolfram. El oráculo
independiente integra el kernel por intervalos, construye el kernel multinodo y
resuelve una recurrencia afín; la diferencia máxima cruzada fue
`1.5543122344752192e-15`. Este resultado no certifica convergencia o estabilidad
de un flujo no lineal.

### Caputo multitérmino

`integrate_multi_term_caputo_l1` especializa la medida discreta como la ecuación
finita `sum_j c_j C D^(alpha_j) x = f(t,x)`. A diferencia de la ruta distribuida,
no expone densidad, semántica de cuadratura ni normalización: los `coefficients`
son parámetros de la ecuación. La fachada agrupa sólo órdenes `float64`
exactamente iguales mediante `math.fsum`, registra ceros y términos originales,
y llama una sola vez a `integrate_distributed_order_caputo_l1`.

La canonización cuesta `O(R log R)` y no reconstruye los bucles `O(RN+N²d)`.
La [derivación, API y evidencia SciSpace/Wolfram](multi_term_caputo_l1.md)
documentan la frontera entre suma atómica y orden distribuido continuo.

### Caputo–Fabrizio

`caputo_fabrizio_derivative` integra exactamente el kernel exponencial en cada
intervalo con interpolación lineal y usa una recurrencia `O(Nd)`. El valor de
normalización `M(alpha)` siempre se registra. También existe una suma directa
`O(N²d)` únicamente como oráculo de prueba.

Esta implementación permanece `research_required`: las derivadas de kernel no
singular tienen problemas conocidos de compatibilidad inicial y de teorema
fundamental. No se reutilizan pesos Caputo ni se presenta el operador como solver.

### Atangana–Baleanu–Caputo: operador y solver separados

`atangana_baleanu_caputo_derivative` sigue siendo un operador sobre una historia
ya muestreada. Su construcción por intervalos lineales está auditada para
`0<q<=1/2` y puede evaluarse de forma directa o por FFT *batch*. En cambio,
`integrate_abc_predictor_corrector` resuelve un IVP ABC conmensurado para
`0<q<1` mediante la recurrencia convencional de historia completa de
Lee--Kim--Jang. No intercambia pesos con el operador muestral y no usa FFT ni la
aproximación rápida por sumas de exponenciales.

El solver trata `B(q)` como un parámetro explícito y conserva su valor en el
resultado; si no se suministra otra convención, registra `B(q)=1`. Para una
condición inicial clásica comprueba `f(a,x0)=0`, restricción que resulta al
evaluar la ecuación integral en el terminal. El artículo presupone un primer
valor de error `O(h**2)` sin proporcionar su cálculo; HAFO obtiene ese valor por
iteración de punto fijo de una ecuación implícita de producto--trapecio, registra
las iteraciones y hace explícita la falta de convergencia. Esta regla de arranque
y la aplicación de la recurrencia a estados vectoriales con un orden común son
extensiones de HAFO; la evidencia publicada de Lee--Kim--Jang es escalar.

La ruta cuesta `O(d*N**2)` y almacena `O(d*N)` valores históricos. Un test de
refinamiento observa orden dos en un problema escalar cuadrático suave; las
pruebas manufacturadas y la paridad Numba--Python comprueban casos finitos, pero
no establecen ese orden para Chua no suave, estabilidad global, caos, atracción
ni ocultedad. La definición permanece experimental y se documenta junto con la
crítica de Diethelm--Garrappa--Giusti--Stynes a kernels no singulares.

### Hadamard y Caputo--Hadamard

`hadamard_convolution_quadrature` transforma `u=log(t/a)` y reutiliza la CQ
BDF1/BDF2 sobre una malla uniforme en `u`. En tiempo físico la malla es
exponencial y el operador entero límite es `t*d/dt`, no `d/dt`. La ruta cruda y
la desplazada Caputo--Hadamard tienen tokens iniciales distintos. Python, Numba
y FFT comparten los pesos canónicos; faltan correcciones de arranque, por lo que
esta API permanece como operador `experimental`.

`integrate_caputo_hadamard_abm` proporciona una vía distinta de solver:
transforma el IVP a Caputo en `u` y usa ABM/PECE de historia completa. Exige
`0<q<1`, orden conmensurado y `log_step`; el RHS recibe el tiempo físico. En
`FractionalProblem`, `grid_coordinate="log_t_over_lower_terminal"` hace
explícita la semántica del paso. No están implementadas aún las mallas graduadas
ni una ruta pura Numba/C para el RHS transformado.

## Orden entero y análisis compartido

`hidden_attractors.integrations.numba_general` proporciona RK4 y mapas genéricos
para funciones `numba.njit`. Conserva las rutas enteras preexistentes y sirve de
ABI común para Toolbox Chaos.

`hidden_attractors.analysis.recurrence` aplica embedding, matriz de recurrencia y
RQA a trayectorias muestreadas enteras o fraccionarias. Esta reutilización es de
datos: no convierte una FDE no local en un sistema Markoviano de dimensión finita.

`hidden_attractors.analysis.correlation_dimension` aplica el mismo principio a
la suma de correlación \(q=2\). El contrato local usa pares no ordenados, ventana
de Theiler, comparación estricta `distancia < radio` y un denominador que cuenta
sólo los pares temporalmente elegibles. Sus backends Python, Numba y C nativo
consumen la trayectoria ya calculada; no alteran el integrador entero o
fraccionario ni cruzan lenguajes durante la marcha temporal.

El ajuste obliga a declarar `fit_radius_range` y no selecciona una región de
escala automáticamente. En trayectorias fraccionarias, la pendiente describe
únicamente la proyección o reconstrucción suministrada: el historial retenido
por la derivada puede hacer que esas coordenadas no constituyan el estado
Markoviano completo. El contrato, los metadatos y los límites de evidencia se
detallan en [Dimensión de correlación](correlation_dimension.md).

`hidden_attractors.analysis.contracts` introduce además el sobre común
`TrajectoryInput/PrehistorySpec/AnalysisResult`. Las muestras quedan
inmutables y alineadas con la coordenada temporal; la huella SHA-256 incluye
proyección, derivada, orden, terminal, prehistoria, política de memoria, solver
y tolerancias. El adaptador de `SimulationResult` conserva el tiempo físico y,
cuando difieren, las muestras de la coordenada de integración. Este contrato
describe la historia: no habilita una prehistoria en un solver que todavía no
la consuma. Véase [Contrato común de análisis](analysis_contracts.md).

Sobre ese contrato,
`hidden_attractors.analysis.permutation_entropy` implementa Bandt--Pompe para
una serie escalar o una componente declarada de una trayectoria. Construye
ventanas forward de dimensión (m) y retardo \(\tau\), codifica la permutación
por rango Lehmer lexicográfico y conserva las (m!) celdas, incluidas las de
conteo cero. Las políticas `stable_index`, `omit` y `raise` hacen visible el
tratamiento de empates. La entropía plugin se normaliza por
\(\log_b(m!)\), no por el número de patrones observados.

Python, Numba y C/OpenMP comparten conteos exactos `uint64`; el ABI nativo no
cruza lenguajes por ventana y limita a 64 MiB los histogramas privados antes
de pasar a incrementos atómicos. El límite (m\leq10) hace explícita la
explosión factorial. En datos fraccionarios, la entropía describe únicamente
el observable bajo la historia registrada y no certifica entropía KS, caos u
ocultedad. El desarrollo completo está en
[Entropía de permutación](permutation_entropy.md).

El despacho `auto` es dimensionado: C se activa desde 131 072 ventanas para
\(m=2,8\), desde 32 768 para \(3\leq m\leq7\), y no se activa automáticamente
para \(m=9,10\). Esta última guarda evita la ruta atómica de histogramas
factoriales grandes; la selección C explícita permanece disponible. El
benchmark local separa el kernel ordinal del coste común de contrato, huella y
metadatos.

### SALI, GALI y LDI de orden entero

`hidden_attractors.analysis.alignment_indices` acepta matrices instantáneas
con forma `(dimension, n_vectors)` e historiales tangentes
`(n_samples, n_vectors, dimension)`. SALI, GALI y su forma LDI se exponen
junto con fachadas para flujos y mapas que propagan ecuaciones variacionales o
partículas vecinas. Todas las rutas ejecutables exigen `q=1`.
NumPy/SVD-LAPACK es la referencia y Numba usa QR de Householder por lotes sobre
copias, sin reortogonalizar las direcciones físicas.

No se añadió un kernel C: para estas matrices densas pequeñas LAPACK ya ejecuta
el álgebra nativa y la ruta Numba elimina el bucle Python. Julia queda como
comparación opcional por trayectoria. SALI/GALI/LDI fraccionario conserva el
estado `research_required`, porque cada operador no local necesita una
ecuación variacional y una norma/renormalización en espacio de historia. Véase
[SALI, GALI y LDI enteros](sali_gali_integer.md).

La cuadratura de convolución de Lubich BDF1/BDF2 ya está disponible como operador
muestreado en `hidden_attractors.fractional.convolution_quadrature`, con rutas
directas Python/Numba y FFT. No incluye correcciones de arranque ni resuelve una
FDE; por ello permanece `experimental`.

La misma infraestructura se reutiliza para Hadamard/Caputo--Hadamard únicamente
después de transformar a tiempo logarítmico. Este reúso elimina duplicación
numérica, pero no hace equivalentes los kernels ni permite mezclar resultados
obtenidos en mallas físicas uniformes y exponenciales.

## Validación y rendimiento

El caso Wolfram independiente `sali_gali_integer.wl` aprobó 12/12
comprobaciones a 80 dígitos. El comparador recorre las seis fachadas públicas
contra rotación, mapa hiperbólico y flujo diagonal exactos; la diferencia
global máxima fue `1.7763568394002505e-15` y la de flujo
`8.881784197001252e-16`. El resumen autoritativo está en
`validation/outputs/wolfram/sali_gali_integer_verified/` y no usa como
oráculo un resultado fallido conservado en la ruta no verificada.

El benchmark `bench_alignment_indices.py` separa compilación y medición
caliente. En este host, para 64, 512 y 4096 muestras, obtuvo razones medianas
NumPy/Numba de 3.141, 2.122 y 2.018, con peor diferencia GALI
`1.5543122344752192e-15`. El artefacto retenido es
`validation/outputs/benchmarks/alignment_indices_numpy_numba_20260803.json`,
SHA-256
`FF4AEB083FDDB7594857F70DC841413CDDDBD36533CE31F6936CBD29BF2B13D8`.
Es evidencia de ingeniería dependiente del host, no una regla universal ni
evidencia dinámica.

La suma de correlación tiene además un oráculo Wolfram independiente sobre seis
puntos 2D exactos, seis radios y ventana de Theiler `w=1`. Reproduce el
denominador `10`, los conteos estrictos `[0, 4, 6, 6, 8, 10]` y, para el rango
de ajuste explícito `[1.1, 2.1]`, la pendiente `0.8664716421373693`. La peor
diferencia Python--Wolfram retenida es `3.552713678800501e-15` frente a
`5e-13`. Esta comparación valida sólo conteo, normalización y regresión finitos;
no certifica una región de escala, dimensión fractal, caos, atracción u
ocultedad.

El caso Wolfram independiente `permutation_entropy.wl` construye cuatro
fixtures exactos: (m=3,\tau=1), (m=3,\tau=2) y empates bajo
`stable_index`/`omit`. Verifica ventanas, rangos Lehmer, conteos,
probabilidades, entropía base dos y normalización. Esta concordancia finita no
es una validación de comportamiento asintótico, selección de (m,\tau),
entropía KS, caos, atracción u ocultedad.

El benchmark `benchmarks/bench_correlation_sum.py` conserva calentamiento y
medición por separado y comprueba igualdad de conteos. En este host Numba fue
ligeramente más rápido a 31 626 pares y C/OpenMP con cuatro hilos a 124 750 pares; esos cruces
son diagnósticos de software dependientes de carga y plataforma, no una regla
universal ni evidencia dinámica.

La validación independiente en
`validation/wolfram/cases/gl_fractional_operator_validation.wl` cubre pesos GL,
identidades Beta/Gamma, polinomios, constante RL, límite `q -> 1`, diferencia
hacia atrás y recurrencias escalares. Una ejecución local previa con Wolfram
14.3 produjo 17/17 casos y una peor discrepancia Python de `9.77e-15` bajo
tolerancia `5e-12`; el JSON exacto de esa ejecución no está retenido en el árbol
actual, por lo que estas cifras no se consideran evidencia de release hasta
regenerarlo con el protocolo incluido.

El caso independiente
`validation/wolfram/cases/hadamard_fractional_operator.wl` sí fue ejecutado y
retenido el 2 de agosto de 2026. Verifica transformación logarítmica,
identidades Gamma/Beta, CQ BDF1/BDF2 generada por expansión formal, constantes,
límite entero y un IVP Caputo--Hadamard manufacturado. El resumen y la
comparación están en
`validation/outputs/wolfram/hadamard_fractional_operator/`; ambos aprobaron,
con peor discrepancia Python--Wolfram `3.019806626980426e-14` y discrepancia de
estado ABM `8.881784197001252e-16`. El alcance sigue siendo identidad simbólica
y consistencia finita, no estabilidad general ni evidencia dinámica.

El artefacto independiente
`validation/outputs/wolfram/atangana_baleanu_operator/` reconstruye para
`alpha=1/2` el kernel de Mittag--Leffler, los pesos por intervalo, la constante,
una rampa y una señal no polinómica sin leer HAFO. El resumen retenido marca
`passed=true`; `tests/test_atangana_baleanu_wolfram.py` compara después las
rutas Python, Numba y FFT. Su frontera dice explícitamente: consistencia de un
operador en malla finita, sin afirmar convergencia del solver ABC, compatibilidad
inicial, estabilidad, caos u ocultedad.

El caso separado
`validation/outputs/wolfram/abc_predictor_corrector/` verifica el solver ABC
convencional: Wolfram integra simbólicamente las bases lineales y reconstruye una
recurrencia manufacturada independiente. El residuo simbólico fue cero; la peor
diferencia de pesos Wolfram--Python fue `3.5388358909926865e-16` y la de estado,
`2.220446049250313e-16`. El error finito `4.952013688854784e-4` frente a la
solución Volterra exacta es sólo diagnóstico y no demuestra una tasa global,
estabilidad ni propiedades dinámicas.

El benchmark `benchmarks/bench_fractional_gl_kernels.py` separa compilación,
calentamiento y medición. La ejecución retenida del 2 de agosto de 2026, con 11
repeticiones calientes en Windows 11/AMD64 (Python 3.14.3, NumPy 2.4.5, GCC y
OpenMP activos), obtuvo para C frente a Numba `3.309x` en `N=1800,d=3` con
historia completa y `1.997x` en `N=6000,d=3,window=128`; la FFT fue `12.304x`
más rápida que Numba para la primera carga. En el barrido, Numba ganó hasta
`N=512` y FFT desde `N=1024` (`2.837x`) hasta `N=4096` (`38.313x`), lo que
respalda en ese host el umbral conservador `FFT_AUTO_THRESHOLD=1024`. El registro
completo, incluidas dispersión, hardware, compilador, hashes y calentamiento, está
en `validation/outputs/benchmarks/fractional_gl_kernels_2026-08-02.json`. Sigue
siendo evidencia de ingeniería dependiente del host, no una superioridad
universal ni una caracterización publicable de rendimiento multiplataforma.

```bash
python benchmarks/bench_fractional_gl_kernels.py --repeats 11 \
  --output validation/outputs/benchmarks/fractional_gl_kernels_2026-08-02.json
python -m pytest tests/test_fractional_contracts_and_gl.py \
  tests/test_native_grunwald_letnikov.py \
  tests/test_gl_fractional_wolfram.py \
  tests/test_hadamard_wolfram.py \
  tests/test_atangana_baleanu_wolfram.py \
  tests/test_abc_predictor_corrector.py \
  tests/test_abc_predictor_corrector_wolfram.py \
  tests/test_tempered_caputo_solver.py \
  tests/test_tempered_caputo_regressions.py \
  tests/test_tempered_convolution_quadrature.py \
  tests/test_tempered_convolution_quadrature_wolfram.py \
  tests/test_tempered_convolution_quadrature_benchmark.py \
  tests/test_variable_order_caputo_type3_solver.py \
  tests/test_distributed_order_caputo_solver.py \
  tests/test_distributed_order_caputo_wolfram.py \
  tests/test_conformable_solver.py -q
```

## Siguiente orden de implementación

1. memoria rápida por sumas de exponenciales o recurrencias controladas, y
   completar las CQ BDF ordinaria/templada con correcciones de arranque y
   solvers implícitos por familia;
2. solver RL con datos iniciales propios y métodos Caputo templados de mayor
   orden; los carriles Caputo templado ABM y CQ templada muestral ya están
   implementados, pero no son intercambiables;
3. historia rápida para el L1 tipo III ya implementado, definiciones de orden
   variable tipo I/II y CQ/mallas graduadas para orden distribuido; el solver
   Caputo L1 de medida discreta ya está implementado;
4. backend `abc_fast_soe_predictor_corrector` por suma de exponenciales,
   separado del predictor--corrector convencional de historia completa ya
   implementado;
5. ecuaciones diferenciales con retardo y ecuaciones en diferencias
   fraccionarias como tipos de problema separados; la fachada Caputo
   multitérmino sobre el kernel L1 combinado ya está implementada sin
   reconstruir el solver distribuido;
6. extender CLV y SALI/GALI/LDI desde sus fachadas enteras `q=1` ya
   implementadas sólo después de formular el cociclo de historia de cada
   operador fraccionario; ampliar además periodicidad, dimensiones
   generalizadas, fixed-mass/boxed, sustitutos y búsqueda de atractores. Todas
   esas variantes tangentes fraccionarias permanecen `research_required`; la
   dimensión de
   correlación directa \(q=2\) ya dispone de contrato local, sin trasladar
   automáticamente teoría ODE al espacio de historia fraccionario.

## Referencias de diseño

- [Lubich, 1986](https://doi.org/10.1137/0517050), cuadratura de convolución.
- [Diethelm, Ford y Freed, 2004](https://doi.org/10.1023/B:NUMA.0000027736.85078.be), análisis de error Adams.
- [Sabzikar, Meerschaert y Chen, 2015](https://doi.org/10.1016/j.jcp.2014.04.024), cálculo fraccionario templado.
- [Li, Deng y Zhao, 2019](https://doi.org/10.3934/dcdsb.2019026), buena formulación y predictor--corrector para FDE templadas.
- [Samko y Ross, 1993](https://doi.org/10.1080/10652469308819027), orden variable.
- [Tavares, Almeida y Torres, 2016](https://doi.org/10.1016/j.cnsns.2015.10.027), definiciones Caputo de orden variable y aproximaciones.
- [Fang, Sun y Wang, 2020](https://doi.org/10.1016/j.camwa.2020.07.009), L1 y aceleración de historia para Caputo de orden variable.
- [Diethelm y Ford, 2009](https://doi.org/10.1016/j.cam.2008.07.018), orden distribuido.
- [Hu, Liu, Anh y Turner, 2014](https://doi.org/10.21914/ANZIAMJ.V55I0.7888), cuadratura de orden y L1 implícito.
- [Lin y Xu, 2007](https://doi.org/10.1016/j.jcp.2007.02.001), análisis de L1 bajo regularidad suficiente.
- [Caputo y Fabrizio, 2015](https://doi.org/10.12785/pfda/010201), kernel exponencial.
- [Diethelm et al., 2020](https://doi.org/10.1515/fca-2020-0032), crítica de kernels no singulares.
- [Grassberger y Procaccia, 1983](https://doi.org/10.1016/0167-2789(83)90298-1), suma y dimensión de correlación.
- [Theiler, 1986](https://doi.org/10.1103/PhysRevA.34.2427), exclusión de correlaciones temporales espurias.
- [Deshmukh et al., 2021](https://doi.org/10.1063/5.0069365), extracción y caracterización de regiones de escala.
- [Bandt y Pompe, 2002](https://doi.org/10.1103/PhysRevLett.88.174102), patrones ordinales y entropía de permutación.
- [Unakafova y Keller, 2013](https://doi.org/10.3390/e15104392), codificación eficiente de patrones solapados.
- [Traversaro et al., 2018](https://doi.org/10.1063/1.5022021), tratamiento y riesgos de valores empatados.
- [Rey et al., 2023](https://doi.org/10.1063/5.0171508), distribución asintótica con dependencia entre patrones.
- [Skokos, 2001](https://doi.org/10.1088/0305-4470/34/47/309), índices de alineamiento.
- [Skokos et al., 2004](https://doi.org/10.1088/0305-4470/37/24/006), comportamiento de SALI.
- [Skokos, Bountis y Antonopoulos, 2007](https://doi.org/10.1016/j.physd.2007.04.004), definición de GALI.
- [Manda, Hillebrand y Skokos, 2025](https://doi.org/10.1016/j.cnsns.2025.108635), método multiparticle.
- [Rolim Sales, Leonel y Antonopoulos, 2026](https://doi.org/10.1016/j.chaos.2026.117884), forma SVD/LDI para mapas y flujos.
- [Ma, Long y Zhu, 2016](https://doi.org/10.1142/S0218127416501820), cautelas para sistemas disipativos.

La localización bibliográfica se apoyó en SciSpace y la bitácora reproducible
está en `docs/scispace_fractional_method_evidence.md`. Las
definiciones fraccionarias ejecutables se anclan a artículos primarios mediante
`hidden_attractors.fractional.references`; las referencias de análisis aún se
conservan en sus módulos/documentos y deberán converger a un registro común.
