# Catálogo verificable de definiciones y métodos fraccionarios

Este documento describe el contrato actual del núcleo fraccionario de HAFO.
Separa cuatro objetos que no deben confundirse: la definición del
operador, su discretización, el contrato de condiciones iniciales y memoria, y
el backend que ejecuta los cálculos. Una función que evalúa una derivada sobre
muestras conocidas no es, por ese hecho, un solver de una ecuación diferencial
fraccionaria (FDE).

El alcance de las verificaciones aquí citadas es algebraico y numérico sobre
mallas finitas. Ninguna de ellas demuestra estabilidad global de un método para
un sistema arbitrario, existencia de un atractor, caos, atracción ni
*hiddenness*.

## Criterio bibliográfico y de verificación

Las fórmulas y decisiones de implementación se apoyan en los artículos
primarios, las páginas de los editores y los libros enumerados en la
[bibliografía](#bibliografia-verificada). Los resúmenes de herramientas de
descubrimiento bibliográfico no son fuentes normativas del código.

Wolfram se usa como oráculo independiente sólo para casos finitos y formulaciones
específicas, no como validador general de cálculo fraccionario. El caso
`validation/wolfram/cases/gl_fractional_operator_validation.wl`
reconstruye los pesos sin leer el código HAFO, comprueba identidades simbólicas
para monomios y constantes, estudia convergencia en varias resoluciones y
verifica el límite entero `q=1`. El comparador
`validation/python/gl_fractional_compare_wolfram.py`
y `tests/test_gl_fractional_wolfram.py`
contrastan el artefacto Wolfram con Python; la prueba en vivo se omite si
`wolframscript` o su licencia local no están disponibles. La documentación
oficial de Wolfram confirma que `FractionalD` representa RL y que
`NFractionalD` ofrece los métodos `"RiemannLiouville"` y
`"GrunwaldLetnikov"` [R21]. Hay casos independientes adicionales, documentados
en sus secciones, para Hadamard/Caputo--Hadamard, el operador muestral ABC de
orden `alpha=1/2`, Caputo variable tipo III y Caputo de orden distribuido. Cada
caso reconstruye sus fórmulas sin leer HAFO. Ninguno valida por sí solo
estabilidad, convergencia general, dinámica caótica, atracción u ocultedad.

## Estados y contrato común

Los estados proceden del registro
`hidden_attractors/fractional/contracts.py`:

- `implemented`: existe una ruta ejecutable que forma parte del contrato actual.
- `experimental`: existe código ejecutable, pero su alcance o estabilidad exige
  validación específica del problema y, para un `FractionalProblem`, aceptación
  explícita cuando corresponda.
- `research_required`: el operador puede tener una ruta diagnóstica, pero sus
  dificultades matemáticas impiden habilitarlo como solver FDE general.
- `planned`: hay un contrato y una fuente bibliográfica, pero no una ruta
  ejecutable completa.
- `theoretical_only`: se registra una equivalencia matemática, no un algoritmo
  numérico público.

El contrato estructurado
`FractionalProblem` en `hidden_attractors/fractional/problem.py` fija por
separado `derivative`, `method`, orden escalar o por componente, estado inicial,
`step`, `t_span`, terminal inferior, semántica de condiciones iniciales, política
de memoria y parámetros del kernel. En la actualidad despacha los solvers
`caputo_abm_pece`, `efork3`, `gl_explicit_discrete`,
`tempered_caputo_abm_pece_transform`, `vo_caputo_type3_l1`,
`distributed_order_caputo_l1`, `abc_predictor_corrector`,
`caputo_hadamard_abm_pece` y `conformable_rk4_clock`. Los objetos públicos que
el catálogo identifica explícitamente como operadores sobre muestras no deben
interpretarse como solvers FDE por el solo hecho de compartir una definición.

En las estimaciones siguientes, `N` es el número de pasos o muestras, `d` la
dimensión del estado, `L` la ventana de historia, `J` el número de nodos en el
espacio de órdenes y `U` el número de órdenes por componente distintos. La
memoria indicada excluye, salvo que se diga lo contrario, las matrices de entrada
y salida de tamaño `O(Nd)`.

## Matriz de disponibilidad

| Definición o método | Naturaleza | Estado | Backend actual | Ruta pública principal |
|---|---|---:|---|---|
| Caputo | operador no local de ley de potencia | `implemented` | C o Python según solver | `FractionalProblem(derivative="caputo", ...)` |
| ABM/PECE de Caputo | solver FDE | `implemented` | C para sistemas registrados; Python/NumPy de referencia | `solve_fractional_problem`, `caputo_abm_integrate` |
| EFORK-3 de Caputo | solver FDE | `implemented` | C cuando la ABI admite el sistema; Python en otro caso | `solve_fractional_problem`, `efork_integrate` |
| GL directo | operador sobre muestras | `implemented` | Numba o C/OpenMP; referencia Python en APIs especializadas | `grunwald_letnikov_derivative` |
| GL rápido por FFT | operador *batch* de historia completa | `implemented` | `scipy.fft`/pocketfft o selector a Numba directo | `fast_grunwald_letnikov_derivative` |
| GL desplazado para Caputo | operador sobre muestras | `implemented` | Numba o C/OpenMP | `definition="caputo_shifted"` |
| Cuadratura de convolución Lubich BDF1/BDF2 | operador sobre muestras | `implemented` | Python, Numba directo o FFT *batch* | `lubich_convolution_quadrature` |
| Hadamard/Caputo--Hadamard CQ BDF1/BDF2 | operador sobre malla exponencial | `implemented` | Python, Numba directo o FFT *batch* | `hadamard_convolution_quadrature` |
| Caputo--Hadamard ABM/PECE | solver FDE en tiempo logarítmico uniforme | `implemented` | C con callback transformado o Python/NumPy | `integrate_caputo_hadamard_abm`, `solve_fractional_problem` |
| Solver explícito GL | solver FDE discreto | `implemented` | Numba para RHS compilado; Python/NumPy para RHS genérico | `integrate_gl_explicit` |
| RL aproximada por GL | operador sobre muestras | `implemented` | Numba o Python; también C con etiqueta RL | `riemann_liouville_gl_derivative` |
| RL templada por GL | operador sobre muestras | `implemented` | Numba o Python | `tempered_grunwald_letnikov_derivative` |
| CQ templada BDF1/BDF2 por conjugación | operador RL templado o Caputo conjugado sobre muestras | `implemented` | Python/Numba directo o FFT *batch* | `tempered_convolution_quadrature` |
| Caputo templada por conjugación | solver FDE conmensurado en tiempo físico | `implemented` | historia física amortiguada en C con callback RHS o Python/NumPy | `integrate_tempered_caputo_abm`, `solve_fractional_problem` |
| GL de orden variable | operador sobre muestras | `implemented` | Numba o Python | `variable_order_grunwald_letnikov_derivative` |
| Caputo de orden variable tipo III | solver FDE L1 implícito en tiempo físico | `implemented` | historia L1 directa en Numba o Python; corrector Picard Python | `integrate_variable_order_caputo_type3_l1`, `solve_fractional_problem` |
| Conformable de Khalil | operador local y solver RK4 en reloj conformable | `implemented` | Numba o Python | `conformable_khalil_derivative`, `integrate_conformable_rk4` |
| Caputo--Fabrizio recurrente | operador sobre muestras | `research_required` en definición; método `implemented` | Numba o Python; suma directa Python como oráculo | `caputo_fabrizio_derivative` |
| Orden distribuido por GL | operador con doble discretización | `implemented` | Numba o Python | `distributed_order_gl_derivative` |
| Caputo de orden distribuido | solver FDE L1 implícito con medida discreta positiva | `implemented` | kernel combinado e historia Numba o Python; Picard Python | `integrate_distributed_order_caputo_l1`, `solve_fractional_problem` |
| Caputo multitérmino | fachada de suma finita sobre el solver L1 distribuido | `implemented` | canonización Python `O(R log R)`; kernel e historia Numba o Python; Picard Python | `canonicalize_multi_term_caputo_terms`, `integrate_multi_term_caputo_l1` |
| Atangana--Baleanu--Caputo | operador sobre muestras con kernel Mittag--Leffler | `implemented` con crítica metodológica explícita | pesos Numba/Python; convolución directa Numba/Python o FFT *batch* | `atangana_baleanu_caputo_derivative` |
| Predictor--corrector ABC convencional | solver FDE conmensurado de historia completa | `implemented` | Numba para RHS compilado o Python/NumPy | `integrate_abc_predictor_corrector`, `solve_fractional_problem` |

## Caputo con Adams--Bashforth--Moulton PECE

### Definición y convención

Para `0 < q < 1` y terminal inferior `a`, HAFO usa la derivada izquierda de
Caputo

$$
{}^{C}_{a}D_t^q x(t)=\frac{1}{\Gamma(1-q)}
\int_a^t (t-\tau)^{-q}x'(\tau)\,d\tau,
$$

con el problema de valor inicial `x(a)=x0`. ABM/PECE discretiza la ecuación
integral de Volterra equivalente: un paso de Adams--Bashforth predice y un paso
de Adams--Moulton corrige. Los pesos implementados están documentados en
`hidden_attractors/integrations/abm.py`. La ruta monolítica exige
orden conmensurable `q<1`; la ruta separada
`hidden_attractors/integrations/abm_fractional.py`
acepta órdenes por componente dentro de bloques reiniciados.

### Terminal, malla, memoria y costo

- Condición inicial: valores clásicos en `a`. No se reutilizan como datos RL.
- Malla: uniforme, `t_n=a+n h`.
- Horizonte: la duración debe contener un número entero de pasos; la fachada
  rechaza el dato si requeriría sobrepasar el terminal.
- Prehistoria opcional: tiempos y estados deben tener la misma longitud, forma
  `(H,d)`, incremento uniforme `h` y un último estado coherente con la condición
  inicial. Datos no conformes no pertenecen a este contrato.
- `full_history`: Caputo completo; costo `O(N^2 d)` y almacenamiento de
  trayectoria/RHS `O(Nd)`.
- `finite_window`: conserva a lo sumo `L` valores de RHS; costo `O(NLd)`. Es un
  operador de memoria truncada y no equivale a Caputo completo.
- `block_restart`: cada bloque vuelve a usar su primer estado como nuevo
  terminal. El bloque cuesta `O(B^2 d)`; con órdenes por componente se
  precalculan matrices de pesos de tamaño `O(U B^2)`. Se pierde la memoria de
  bloques anteriores por definición.

### Backend, rutas y pruebas

`caputo_abm_integrate` entra en el despachador nativo
`hidden_attractors/integrations/fractional_c.py` cuando el
sistema está registrado y el backend C está disponible; si no, utiliza la ruta
Python/NumPy. `solve_fractional_problem` expone el mismo contrato estructurado.

Las pruebas relevantes son
`tests/test_efork_published_validation.py`,
que compara ABM C con la referencia Diethelm y comprueba transporte de historia,
`tests/test_abm_fractional_diagnostics.py`,
que cubre orden conmensurable y por componente, y
`tests/test_fractional_problem.py`, que prueba
el despacho y los metadatos. No hay todavía un oráculo Wolfram general para el
ABM Caputo en tiempo físico; la ruta Caputo--Hadamard sí posee, más adelante,
un IVP manufacturado independiente en coordenada logarítmica.

### Límites y referencias

La convergencia publicada presupone regularidad suficiente; las singularidades
débiles cerca de `a`, el truncamiento de memoria y los reinicios pueden cambiar
el orden observado. El método actual no implementa la construcción de orden
superior de Yan--Pal--Ford [R5]. Referencias que definen o analizan la ruta:
[R1], [R2], [R3] y [R4].

## Caputo con EFORK-3

### Definición y convención

EFORK-3 resuelve el mismo IVP de Caputo conmensurable `0<q<1`. Para cada paso
forma tres etapas

$$
K_1=h^q f(t_n,x_n),\quad
K_2=h^q f(t_n+c_2h,x_n+a_{21}K_1),
$$

$$
K_3=h^q f(t_n+c_3h,x_n+a_{31}K_1+a_{32}K_2),\qquad
x_{n+1}=x_n+w_1K_1+w_2K_2+w_3K_3,
$$

con coeficientes Gamma dependientes de `q` y la corrección de memoria publicada
por Ghoreishi, Ghaffari y Saad [R6]. El orden de `a31*K1+a32*K2` es parte del
contrato verificado.

### Terminal, malla, memoria y costo

- Condición inicial clásica de Caputo en un terminal fijo; puede recibir una
  prehistoria explícita.
- Malla uniforme; la duración debe contener un número entero de pasos y nunca
  se redondea hacia un terminal posterior.
- Una prehistoria explícita debe usar tiempos uniformes separados por `h`,
  estados con forma `(H,d)` y un último estado coherente con `x0`.
- Historia completa: el cálculo directo del término de memoria cuesta
  `O(N^2 d)` y almacena `O(Nd)`.
- Ventana de `L` pasos: `O(NLd)`; altera el contrato Caputo.
- Solo orden conmensurable en la fachada `FractionalProblem`.

### Backend, rutas y pruebas

La implementación está en
`hidden_attractors/integrations/efork.py`. Usa el backend C del
despachador para sistemas admitidos y cae a la implementación Python publicada
si se autoriza. `q=1` tiene una rama entera EFORK_Q1 específica y no debe
presentarse como EFORK fraccionario.

`tests/test_efork_published_validation.py`
reproduce errores terminales publicados, comprueba coeficientes, orden de las
etapas, C frente a la referencia y transporte de historia. No existe aún un
oráculo Wolfram independiente de EFORK. Fuente primaria: [R6].

### Límites

La presencia de tres etapas no elimina el costo histórico ni demuestra
estabilidad para un sistema no lineal dado. La ventana, la continuación y el
límite entero son protocolos numéricos diferentes y deben etiquetarse.

## Grünwald--Letnikov directo y desplazado

### Definición y convención

En una malla uniforme `t_n=a+n h`, los pesos se generan sin evaluar funciones
Gamma:

$$
g_0(q)=1,\qquad
g_k(q)=g_{k-1}(q)\left(1-\frac{q+1}{k}\right)
=(-1)^k\binom{q}{k}.
$$

El operador discreto directo es

$$
D_{h,a}^{q}x_n=h^{-q}\sum_{k=0}^{n}g_k(q)x_{n-k}.
$$

`definition="grunwald_letnikov"` conserva la historia cruda;
`"riemann_liouville_gl"` usa la misma suma con la etiqueta explícita de
aproximación RL; y `"caputo_shifted"` aplica la suma a `x_n-x_0`. Para
`0<q<1`, este último es la aproximación GL/desplazada compatible con Caputo,
pero no cambia la definición cruda en una identidad universal.

### Terminal, malla, costo y backend

El primer valor es la muestra en el terminal inferior; no se reconstruye
prehistoria. La malla debe ser uniforme. La historia completa cuesta
`O(N^2d)`; una ventana de `L` muestras cuesta `O(NLd)` y define otro operador.
La ruta Numba está en
`hidden_attractors/fractional/grunwald_letnikov.py`.
El backend C/OpenMP se describe por separado más abajo.

### Pruebas y oráculo

`tests/test_fractional_contracts_and_gl.py`
comprueba la recurrencia binomial, un monomio, la constante RL, órdenes por
componente y la ventana. El caso Wolfram y su comparador verifican la identidad
de los pesos, las fórmulas continuas
`D_C^q t^m=Gamma(m+1)t^(m-q)/Gamma(m+1-q)` y
`D_RL^q c=c t^(-q)/Gamma(1-q)`, convergencia al refinar `h` y el límite de
diferencia hacia atrás para `q=1`. Es evidencia finita, no un teorema de
convergencia para toda entrada. Referencias: [R2], [R7] y [R21].

## GL rápido por convolución FFT

`fast_grunwald_letnikov_derivative` en
`hidden_attractors/fractional/fast_grunwald_letnikov.py`
evalúa exactamente la misma convolución discreta causal de historia completa
que el operador GL directo, salvo el redondeo de punto flotante. No cambia la
definición matemática: admite GL crudo, la etiqueta de aproximación RL y el
desplazamiento `x-x0` compatible con Caputo.

Para impedir el *aliasing* circular, la ruta rápida elige una longitud de FFT
`M=next_fast_len(2*N-1)`, rellena con ceros, multiplica los espectros reales y
recorta las primeras `N` muestras de la convolución lineal. Utiliza
`scipy.fft.rfft/irfft`, cuyo backend reportado es pocketfft. El costo aritmético
es `O(d N log N)` y el espacio de trabajo principal `O(dN)`; los metadatos
incluyen longitud de FFT, versiones, backend solicitado y una estimación de
bytes que excluye buffers opacos internos de SciPy y del asignador.

`backend="auto"` selecciona FFT cuando `N>=1024` y Numba directo por debajo de
ese umbral. `auto_threshold` puede configurarse y siempre queda registrado. El
valor 1024 es una política conservadora derivada del sondeo público con `d=3`,
no una afirmación de que FFT sea más rápida en todo host: el cruce depende de
CPU, biblioteca, dimensión y calentamiento, y para historias cortas la
preparación y las asignaciones pueden dominar.

Esta ruta está `implemented`, es *batch/offline* y exclusivamente de historia
completa. No implementa ventana finita, actualización *streaming*, solver FDE ni
orden variable; para esos contratos se necesita otro algoritmo. El trabajo de
Matusiak [R22] es un precedente relacionado de convolución FFT fraccionaria, no
la fuente exacta de esta ruta: la implementación HAFO actual usa
`scipy.fft`/pocketfft, es de orden fijo por componente y no afirma reproducir su
algoritmo de orden variable.

`tests/test_fast_grunwald_letnikov.py`
compara FFT con el operador Numba para longitudes pares e impares, órdenes por
componente y las tres etiquetas; comprueba `q=1`, diferencia entre constante RL
y constante desplazada, umbral automático, relleno suficiente contra aliasing,
validación de entradas, metadatos y exportación pública. Wolfram valida el
operador GL/RL finito que constituye el objetivo de la convolución, pero no
valida el rendimiento ni la implementación pocketfft. Referencias: [R7] y
[R22].

## Solver explícito GL

### Recurrencia

`hidden_attractors/fractional/gl_solver.py` despeja el valor
nuevo de la suma GL y evalúa el RHS de forma rezagada:

$$
x_n=a_0+h^q f(t_{n-1},x_{n-1})-
\sum_{k=1}^{n}g_k(q)(x_{n-k}-a_0),
$$

donde `a0=x0` para `initialization="caputo_shifted"` y `a0=0` para
`"discrete_gl"`. La segunda opción es un problema discreto GL con `x0` como
primera muestra; no es un IVP RL clásico.

La recurrencia es una construcción discreta de HAFO obtenida al despejar la suma
GL finita y evaluar el RHS con un paso de rezago. [R2] respalda la definición GL
de base, pero no constituye un análisis publicado de estabilidad, convergencia
o memoria truncada para esta fórmula exacta.

### Contrato computacional

- Malla uniforme, orden escalar o por componente y terminal `t0` explícito.
- Historia completa `O(N^2d)`; ventana `O(NLd)`; pesos y trayectoria
  `O(Nd)`.
- `integrate_gl_explicit_numba` exige un RHS `numba.njit` y un vector numérico
  de parámetros. `integrate_gl_explicit` selecciona esa ruta o un fallback
  Python/NumPy para sistemas declarativos y la GUI.
- Estado de implementación `implemented`: no se atribuye al método una región
  de estabilidad ni un orden general más fuerte que lo verificado.

Las pruebas comparan Numba con Python, la solución de `D_C^q x=1` y el límite
Euler para `q=1`; Wolfram reconstruye de manera independiente esos dos casos
escalares. Rutas: `tests/test_fractional_problem.py`,
`tests/test_fractional_contracts_and_gl.py`
y el caso Wolfram anterior. [R2] es referencia del operador GL subyacente, no
validación bibliográfica de la recurrencia HAFO.

## Backend C/OpenMP para GL

El kernel original HAFO está en
`hidden_attractors/native/csrc/grunwald_letnikov_lib.c`
y su ABI `ctypes` en
`hidden_attractors/fractional/native_grunwald_letnikov.py`.
Ofrece pesos, convolución sin escala y derivada escalada. Los buffers son
`float64`, contiguos y *row-major* con forma `(N,d)`; el wrapper valida forma,
finitud, orden, paso, aliasing y desbordamientos antes o dentro de la ABI.

El C almacena pesos en orden `lag-major` para acceder de forma contigua a los
componentes. Cuando el compilador activa OpenMP, paraleliza tiempos con
`schedule(guided,16)` para `N>=256` y usa `omp simd` en el eje de componentes.
La compilación solicita OpenMP, pero el resultado registra por separado
`openmp_requested` y `openmp_active`: no debe afirmarse que OpenMP quedó activo
sin leer esos metadatos. La biblioteca compartida queda identificada por SHA-256
del fuente, versión de ABI y `kernel_id`.

Con historia completa, el trabajo sigue siendo `O(N^2d)` y los pesos ocupan
`O(Nd)`; con ventana `L`, el trabajo es `O(NLd)` y los pesos `O(Ld)`, además de
entrada/salida. OpenMP reduce tiempo de pared en casos apropiados, no la
complejidad asintótica. Si no se puede compilar o cargar la biblioteca, el
wrapper usa Numba solo cuando `fallback=True`.

`tests/test_native_grunwald_letnikov.py`
compara pesos y resultados multicomponente C--Numba, historia completa y
ventana, GL crudo y desplazado, forma escalar, metadatos de compilación y
fallback. Sus tolerancias prueban paridad numérica; no constituyen un benchmark
de rendimiento. Referencias matemáticas: [R2] y [R7].

## Cuadratura de convolución de Lubich BDF1/BDF2

`hidden_attractors.fractional.convolution_quadrature` construye pesos a partir de
la función generadora BDF

$$
\delta(\zeta)^q=\sum_{k=0}^{\infty}\omega_k\zeta^k,
\qquad
D_h^q x_n=h^{-q}\sum_{k=0}^{n}\omega_k x_{n-k}.
$$

Actualmente se admiten

$$
\delta_1(\zeta)=1-\zeta,
\qquad
\delta_2(\zeta)=\frac32-2\zeta+\frac12\zeta^2,
$$

para `0<q<=1`, historias reales `float64`, malla uniforme representable, historia
completa y órdenes por componente. Las entradas complejas se rechazan en vez de
perder silenciosamente su parte imaginaria. Tanto `times` como la malla construida
con `step` deben ser finitas y estrictamente crecientes; la uniformidad se compara
en la escala del incremento, no en la magnitud absoluta del terminal. BDF1
coincide exactamente con los pesos GL canónicos. Los pesos BDF2 se generan en
`O(N)` mediante la recurrencia obtenida de
`delta W' = q delta' W`; se contrastan contra una expansión factorial
independiente de 60 dígitos.

La convención `riemann_liouville` aplica la convolución cruda y exige el token
`operator_only_no_ivp`. La convención `caputo_shifted` aplica el mismo operador a
`x-x(a)` y exige `point_value_shift_x_minus_x0`. Ninguna de las dos rutas resuelve
una FDE. Los backends `python` y `numba` cuestan `O(d*N^2)`; `fft` usa *zero
padding* para una convolución lineal `O(d*N*log N)` y es sólo *batch*, no online.

El arranque BDF2 es deliberadamente visible: usa el historial truncado en el
terminal, no inventa prehistoria ni sustituye los primeros pasos por BDF1. Aún no
hay correcciones iniciales. Éstas pueden ser necesarias ante baja regularidad o
incompatibilidad, como analiza Jin--Li--Zhou [R24]. El resultado estructurado
registra `startup_convention="terminal_truncated_history_no_prehistory_extrapolation"`
y `starting_corrections="none_implemented"`.

`tests/test_lubich_convolution_quadrature.py` verifica BDF1=GL, expansión BDF2,
límite `q=1`, convergencia sobre monomios, orden por componente, malla y paridad
Python--Numba--FFT. Las bases teóricas son Lubich [R7, R23]. La ruta está
`implemented` bajo ese contrato limitado; no incluye correcciones de arranque,
BDF de mayor orden, CQ-RK ni un solver FDE.

## Hadamard y Caputo--Hadamard por CQ en tiempo logarítmico

Para un terminal estrictamente positivo `a`, el integral de Hadamard usa un
kernel logarítmico y la medida `ds/s`,

$$
({}^{H}_{a}I_t^\alpha x)(t)=\frac{1}{\Gamma(\alpha)}
\int_a^t\left(\log\frac{t}{s}\right)^{\alpha-1}x(s)\frac{ds}{s}.
$$

Con `0<q<=1`, la derivada Hadamard tipo Riemann--Liouville aplica
`delta=t*d/dt` después de `I_H^(1-q)`; la variante Caputo--Hadamard aplica
`I_H^(1-q)` a `delta x`. HAFO no identifica estos operadores con RL o Caputo en
tiempo físico. Usa la transformación exacta de variable

$$
u=\log(t/a),\qquad \delta x(t)=\frac{d}{du}x(ae^u),
$$

y ejecuta la CQ canónica en una malla uniforme `u_n=n*bar_tau`, que corresponde
a la malla física exponencial `t_n=a*exp(n*bar_tau)`. La ruta cruda exige
`hadamard_operator_only_no_ivp`; la ruta Caputo--Hadamard aplica la CQ a
`x(t)-x(a)` y exige `caputo_hadamard_point_value_shift`.

`hidden_attractors.fractional.hadamard_convolution_quadrature` admite BDF1 o
BDF2, órdenes escalares o por componente y backends `python`, `numba` o `fft`.
Las rutas directas cuestan `O(d*N^2)`; la convolución *batch* con relleno cero,
`O(d*N*log N)`. La función devuelve tanto los tiempos físicos como
`log_times`, pesos, transformación, contrato inicial, complejidad y fuentes.
Rechaza terminales no positivos y mallas que no sean uniformes en `log(t/a)`.

Jarad--Abdeljawad--Baleanu [R25] fija la modificación Caputo--Hadamard. Yin et
al. [R26] extienden CQ BDF de órdenes 1 a 6 a cálculo de Hadamard sobre mallas
exponenciales y analizan correcciones para datos singulares. HAFO implementa
solamente BDF1/BDF2 **sin** esas correcciones; por ello no hereda sus órdenes
óptimos para soluciones no suaves ni afirma resolver la ecuación de
subdifusión del artículo.

`tests/test_hadamard_convolution_quadrature.py` contrasta la transformación con
la CQ canónica, las tres rutas de ejecución, constantes, potencias de
`log(t/a)`, el límite entero `delta=t*d/dt`, órdenes por componente, contrato de
malla y rechazo de ejecución como solver. Es evidencia de operador sobre malla
finita, no de caos, atractor u ocultamiento.

La validación Wolfram retenida en
`validation/outputs/wolfram/hadamard_fractional_operator/` deriva de forma
independiente el cambio de variable, identidades Gamma/Beta y los pesos BDF1/2
por expansión formal. Su comparador Python aprobó con discrepancia numérica
máxima `3.019806626980426e-14`; el IVP ABM de forzamiento constante coincidió a
`8.881784197001252e-16`. `tests/test_hadamard_wolfram.py` aprobó siete pruebas
con el artefacto presente. Esta evidencia certifica esas identidades y casos
finitos, no estabilidad general, convergencia de todo RHS, caos ni hiddenness.

### Solver ABM/PECE Caputo--Hadamard

Para el IVP

$$
{}^{CH}_{a}D_t^q x(t)=f(t,x(t)),\qquad x(a)=x_0,
$$

la transformación de Zheng [R27] da exactamente

$$
{}^{C}_{0}D_u^q y(u)=f(ae^u,y(u)),\qquad
y(u)=x(ae^u),\quad u=\log(t/a).
$$

`integrate_caputo_hadamard_abm` reutiliza el ABM/PECE Caputo canónico en una
malla uniforme en `u`. Admite por ahora `0<q<1` conmensurado, condición puntual,
historia completa y paso explícitamente llamado `log_step`. El RHS siempre
recibe tiempo físico `a*exp(u)`. El resultado devuelve `times`, `log_times`,
estado, backend, coordenadas de malla, memoria y alcance de evidencia.

`FractionalProblem` exige
`grid_coordinate="log_t_over_lower_terminal"`; así `step` no puede confundirse
silenciosamente con un paso físico. La salida es uniforme en tiempo logarítmico
y no uniforme en tiempo físico. En consecuencia, FFT/Welch deben trabajar en
`u` con unidades declaradas o sobre un remuestreo físico explícito.

El backend acelerado reutiliza el historial ABM en C, pero el cambio
`u -> a*exp(u)` y un RHS genérico cruzan un callback Python; esto queda en
`backend="c_abm_with_python_time_transform"`. La referencia Python/NumPy se
conserva y ambas rutas se comparan. No se anuncia todavía un RHS transformado
puramente C/Numba.

En `solver_info`, `n_steps` y `n_steps_completed` cuentan incrementos realmente
completados; `n_samples` cuenta puntos almacenados, incluido el inicial, y
`n_samples_returned` cuenta los puntos entregados después de aplicar la política
de historia. Esta distinción evita llamar "pasos" a las `N+1` muestras de una
trayectoria completa.

Green--Yan [R31] estudian específicamente el Adams uniforme sobre la malla
logarítmica y muestran que el error depende de la regularidad; esa referencia no
autoriza una promesa uniforme de orden dos para todo `0<q<1`. Green--Liu--Yan
[R28] analizan además PECE en mallas logarítmicas graduadas y recuperación de
orden bajo regularidad débil. HAFO implementa únicamente el caso uniforme
`r=1`; no contiene sus pesos graduados ni hereda las tasas óptimas publicadas.
`tests/test_caputo_hadamard_solver.py` verifica forzamiento constante, potencia
manufacturada de `log(t/a)`, equivalencia con Caputo transformado, tiempo físico
del RHS, paridad de backends, dispatcher y contratos degenerados. Es una
trayectoria numérica finita, no una certificación dinámica.

## Riemann--Liouville por GL

Para `0<q<1`, la derivada RL izquierda es

$$
{}^{RL}_{a}D_t^q x(t)=\frac{d}{dt}\left[
\frac{1}{\Gamma(1-q)}\int_a^t(t-\tau)^{-q}x(\tau)\,d\tau
\right].
$$

`riemann_liouville_gl_derivative` en
`hidden_attractors/fractional/sampled_operators.py`
la aproxima mediante la suma GL cruda. Exige que `times[0]==lower_terminal`,
malla uniforme y el token
`initial_condition_semantics="operator_only_no_ivp"`. Este token evita
interpretar `x(a)` como condición inicial RL: un IVP RL clásico requiere datos
de integral fraccionaria, por ejemplo `(I_a^(1-q)x)(a+)`.

La API especializada usa historia completa, con costo `O(N^2d)`, y dispone de
Numba y referencia Python. La API GL genérica también permite etiquetar la suma
como `riemann_liouville_gl` y truncar la memoria. En `n=0`, el valor discreto es
`h^(-q)x(a)`; puede divergir al refinar y no debe confundirse con un valor
puntual finito en el terminal.

`tests/test_sampled_fractional_operators.py`
compara constantes y potencias con fórmulas Gamma, verifica `q=1`, paridad
Numba--Python y rechazo de semánticas IVP implícitas. Wolfram cubre constante,
monomio y límite entero en malla finita. Referencias: [R2], [R7] y [R21].

## Riemann--Liouville templada por GL

La convención implementada es la conjugación exponencial no corregida

$$
{}^{RL}_{a}D_t^{q,\lambda}x(t)=e^{-\lambda(t-a)}
{}^{RL}_{a}D_t^q\left(e^{\lambda(\cdot-a)}x(\cdot)\right)(t),
\qquad \lambda\ge 0,
$$

que en la malla produce los pesos
`h^(-q) g_k(q) exp(-lambda*k*h)`. No se resta silenciosamente
`lambda^q x`, y no es Caputo templada.

La función
`tempered_grunwald_letnikov_derivative` en
`hidden_attractors/fractional/sampled_operators.py`
es un operador, no un solver. Usa terminal explícito, token
`operator_only_no_ivp`, malla uniforme, historia completa, orden escalar o por
componente y `lambda>=0`. Su costo es `O(N^2d)` y el backend puede ser Numba o
Python.

Las pruebas verifican reducción exacta a RL para `lambda=0`, la identidad de
conjugación sobre una potencia, el caso `q=1`, validación de parámetros y
paridad Numba--Python. No hay oráculo Wolfram templado. La definición actual se
basa en [R8]. El método de Bibi y ur Rehman [R9], que usa integración de producto
y Newton--Cotes para problemas templados, está verificado bibliográficamente
pero corresponde a una extensión no implementada; no describe el kernel GL actual.

## CQ templada BDF1/BDF2 por conjugación exponencial

`tempered_convolution_quadrature` implementa el operador muestral

$$
\delta_p(e^{-\lambda h}\zeta)^q
=\sum_{k\ge0}e^{-\lambda kh}\omega_k^{(q,p)}\zeta^k,
\qquad p\in\{1,2\}.
$$

La ruta RL aplica directamente los pesos amortiguados. La ruta Caputo resta el
ancla correcta en la coordenada conjugada:

$$
h^{-q}\sum_{k=0}^{n}e^{-\lambda kh}\omega_kx_{n-k}
-h^{-q}x_0e^{-\lambda nh}\sum_{k=0}^{n}\omega_k.
$$

No materializa `exp(+lambda*t)`, no resta `lambda**q*x`, no implementa
correcciones numéricas de arranque y no es un solver FDE. Admite órdenes y
temperings por componente. Python y Numba son convoluciones directas
`O(d*N**2)`; FFT es convolución lineal *batch* `O(d*N*log(N))`, no historia
rápida ni *streaming*.

El contrato conserva `tempered_symbol_shift_cq` como ruta todavía `planned`
para `[delta(z)/h+lambda]**q`, que no comparte los pesos ni el ancla discreta.
La historia recurrente de Guo et al. ya es un operador implementado separado,
con API pública experimental, descrito en la sección siguiente. La validación Wolfram independiente de esta
CQ directa pasó 18/18 pruebas y la diferencia máxima contra el núcleo público
fue `4.44089209850063e-15`. Consulte
[Tempered BDF Convolution Quadrature](tempered_convolution_quadrature.md) para
la derivación completa, comandos, benchmark y límites de evidencia.

## Historia multistep templada recurrente: Fast Method II

`tempered_fast_multistep_history` ejecuta FBDF1 y GNGF2 sobre historias
muestreadas RL templadas o Caputo templadas conjugadas. Sus generadores son

$$
\Omega_{\mathrm{FBDF1}}(z)=(1-z)^q,
\qquad
\Omega_{\mathrm{GNGF2}}(z)=(1-z)^q\left(1+\frac q2(1-z)\right).
$$

GNGF2 no se renombra BDF2 fraccionario: el factor real de BDF2 atraviesa una
rama fraccionaria negativa. Sólo en `q=1` GNGF2 se reduce exactamente al
polinomio BDF2 ordinario. La CQ BDF2 directa/FFT continúa siendo una ruta
separada.

Para `0<q<1`, la cola usa nodos reales adimensionales `r_j=h*lambda_j` y

$$
\omega_\ell\approx\sum_{j=0}^{Q-1}a_j(1+r_j)^{-\ell-1}.
$$

El signo `-sin(pi*q)/pi` se fija mediante la identidad beta/reflexión que
reproduce exactamente los coeficientes FBDF1 negativos para `ell>=1`. Tras
reescalar el estado auxiliar de la publicación, HAFO aplica

$$
y_m^{(j)}=\frac{e^{-\sigma h}}{1+r_j}
\left(y_{m-1}^{(j)}+u_{m-1}\right).
$$

Los retardos `0..n0` se calculan con pesos exactos; sólo la cola es comprimida.
La corrección Caputo usa la suma parcial exacta, no la cuadratura. El trabajo
es `O(d*(Q+n0)*N)` y la memoria histórica activa es `O(d*(Q+n0))`, sin contar
entrada y salida.

Cuando `quadrature_points=None`, HAFO refina `Q=65,129,257,...` hasta que el
error relativo L1 de **todos** los pesos comprimidos de la malla finita cumple
`relative_tolerance`. El límite de operador reportado controla esa compresión,
no el error CQ/FDE, las correcciones de arranque ni el teorema analítico del
trapecio infinito.

La validación independiente Wolfram pasó 13/13 aserciones; el mayor residuo
interno fue `1.72700092683619e-26`. La reconstrucción Python independiente y
el núcleo público difirieron del artefacto a lo sumo
`8.881784197001252e-15` y `1.2434497875801753e-14`, respectivamente. El ejemplo
`tempered_fast_history_chua.py` sólo postprocesa una historia Chua entera y
compara con convolución directa; no resuelve una FDE ni clasifica dinámica.
Consulte [Fast Recurrent Tempered Multistep History](tempered_fast_multistep_history.md)
para el álgebra, API, evidencia y límites completos.

## Solver Caputo templado por conjugación exponencial

Para `0<q<1` y `lambda>=0`, HAFO declara la definición Caputo templada

$$
{}^C_aD_t^{q,\lambda}x(t)=e^{-\lambda(t-a)}
{}^C_aD_t^q\!\left(e^{\lambda(\cdot-a)}x(\cdot)\right)(t).
$$

Con `v(t)=exp(lambda*(t-a))*x(t)`, el IVP se reduce exactamente a

$$
{}^C_aD_t^qv(t)=e^{\lambda(t-a)}
f\!\left(t,e^{-\lambda(t-a)}v(t)\right),\qquad v(a)=x(a).
$$

`integrate_tempered_caputo_abm` usa esta transformación para derivar los factores
`exp(-lambda*(t_n-t_j))` y los aplica directamente a cada contribución histórica
ABM/PECE. Los kernels Python y C almacenan así el estado físico `x`, no el estado
potencialmente enorme `v`. La fuente primaria [R34] establece el operador, su
ecuación de Volterra y un predictor--corrector de Jacobi; **HAFO no afirma
implementar ese algoritmo de Jacobi**. La adaptación hereda el contrato de error
de ABM, no las tasas de otro método.

La malla es uniforme en tiempo físico, el orden debe ser común y `tempering`
forma parte obligatoria de `kernel_parameters`. `full_history` conserva la
historia; `finite_window` queda etiquetada como reinicio deslizante con ancla
`exp(-lambda*(t-t_s))*x(t_s)`, no como truncamiento neutro del problema con
terminal fijo. El backend acelerado ejecuta los pesos y la historia física en C,
pero un RHS arbitrario atraviesa un callback Python; los metadatos no lo presentan
como un campo vectorial C puro. La divergencia se evalúa en cada estado físico
aceptado antes de pedir la siguiente derivada. Los exponentes del kernel son no
positivos, por lo que un `lambda*T` grande no fuerza a almacenar `v`; el campo
opcional `transformed_states` queda en `None` si su reconstrucción no cabe en
`float64`.

La reducción `lambda=0` debe coincidir muestra a muestra con Caputo. La prueba
manufacturada usa

$$
x_*(t)=e^{-\lambda\tau}(x_0+\tau^p),\qquad
f_*(t)=e^{-\lambda\tau}
\frac{\Gamma(p+1)}{\Gamma(p+1-q)}\tau^{p-q},
\quad \tau=t-a.
$$

Estas comprobaciones validan trayectorias finitas y la transformación; no
prueban estabilidad no lineal, caos, atracción ni ocultedad.

## Grünwald--Letnikov de orden variable

Las derivadas de orden variable no son únicas. HAFO fija una convención
reproducible: en cada tiempo de salida se congela `q_n=q(t_n)` en todos los
pesos de esa suma,

$$
D^{q(\cdot)}_{h,a}x_n=h^{-q_n}\sum_{k=0}^{n}g_k(q_n)x_{n-k}.
$$

No se agregan términos con derivadas de `q(t)` y la salida no debe renombrarse
como otra convención tipo I/II. La función
`variable_order_grunwald_letnikov_derivative` en
`hidden_attractors/fractional/sampled_operators.py`
requiere historia muestreada desde un terminal explícito, malla uniforme y
semántica `operator_only_no_ivp`. Admite un orden por tiempo y componente; una
lista ambigua de solo `d` valores se rechaza.

Como los pesos cambian en cada salida, la ruta directa cuesta `O(N^2d)` y no
reutiliza un vector estacionario de coeficientes. Hay backends Numba y Python.
Las pruebas comprueban reducción exacta al GL de orden constante, límite `q=1`,
paridad de backends y validación de forma. No hay oráculo Wolfram para esta
convención. Samko--Ross [R10] es una referencia fundacional para el orden
variable, pero no valida esta suma concreta con el orden congelado en el tiempo
de salida; esa elección es una convención explícita de HAFO. El método espectral
de Ahmed--Izadi--Cattani [R11] es un candidato reciente para FDE de orden
variable, no una validación ni una implementación de esta suma GL.

### Solver Caputo de orden variable tipo III

HAFO implementa por separado la convención tipo III de
Tavares--Almeida--Torres [R35]:

$$
{}^{C,III}_{a}D_t^{\alpha(t)}x(t)
=\frac{1}{\Gamma(1-\alpha(t))}
\int_a^t(t-s)^{-\alpha(t)}x'(s)\,ds,
\qquad 0<\alpha(t)<1.
$$

El valor corriente $\alpha_n=\alpha(t_n)$ se usa en toda la fila histórica.
Sobre una malla uniforme, `integrate_variable_order_caputo_type3_l1` aplica

$$
\frac{h^{-\alpha_n}}{\Gamma(2-\alpha_n)}
\sum_{j=0}^{n-1}a_{n-j-1}^{(n)}(X_{j+1}-X_j),\qquad
a_k^{(n)}=(k+1)^{1-\alpha_n}-k^{1-\alpha_n}.
$$

Separando el último incremento, el estado satisface la ecuación implícita

$$
X_n=X_{n-1}-H_n+\Gamma(2-\alpha_n)h^{\alpha_n}f(t_n,X_n),
$$

que HAFO resuelve por Picard con tolerancias, máximo de iteraciones y política
de no convergencia explícitos. Este corrector para sistemas vectoriales es una
adaptación HAFO; no se atribuye como teorema del artículo. La suma histórica
directa cuesta `O(N^2)` filas y `O(N^2*d)` operaciones por componentes, almacena
`O(N*d)` y puede usar Numba sin `fastmath`. La referencia rápida de Fang--Sun--Wang
[R36] sustenta la hoja de ruta, no permite llamar rápida a esta implementación.

La función de orden es un programa temporal prescrito. Puede exponerse como
`alpha(t)`, `alpha(t, initial_state)` o
`alpha(t, initial_state, parameters)`; en las dos últimas firmas HAFO entrega
siempre una copia fija del estado inicial, nunca el estado iterado. Una solución
$C^1$ exige $f(a,X_0)=0$; el solver avisa sólo si se declara
`initial_regularity="smooth"`, mientras `nonsmooth` (y el alias `weak`) conserva
la posibilidad de un arranque débilmente singular sin heredar la tasa suave.
La solución manufacturada, la reducción a orden constante, la paridad
Python--Numba y el fallo estructurado del corrector verifican ejecuciones finitas;
no demuestran estabilidad general, convergencia sobre campos no suaves, caos,
atracción ni ocultedad. El contrato completo está en
[`variable_order_caputo_type3.md`](variable_order_caputo_type3.md).

## Derivada conformable de Khalil

Para una función diferenciable y `t>a`, la identidad implementada es

$$
T_q^a x(t)=(t-a)^{1-q}x'(t),\qquad 0<q\le 1.
$$

`conformable_khalil_derivative` en
`hidden_attractors/fractional/sampled_operators.py`
recibe valores o un callable de la derivada ordinaria; no aproxima `x'` desde
las muestras. Los tiempos solo deben ser estrictamente crecientes, no uniformes.
En `t=a` y `q<1`, el usuario debe seleccionar `terminal_policy`: rechazar,
asumir derivada acotada y valor cero, o proporcionar el límite.

Es un operador local sin historia: costo `O(Nd)`, memoria auxiliar `O(d)` y
backend Numba o Python. Para un orden conmensurado, el solver
`integrate_conformable_rk4` usa el cambio de reloj

$$
\tau=\frac{(t-a)^q}{q},\qquad
\frac{dx}{d\tau}=f\!\left(a+(q\tau)^{1/q},x\right),
$$

y aplica RK4 clásico con paso uniforme en `tau`. La malla física es en general
no uniforme. `FractionalProblem` exige
`grid_coordinate="conformable_clock"`, `memory_policy="none"` y el método
`conformable_rk4_clock`; órdenes distintos por componente se rechazan porque no
existe un solo reloj transformado. RHS con ABI Numba usan un kernel `nopython`;
RHS genéricos usan el camino Python y `allow_python_fallback=False` permite
exigir aceleración.

Las pruebas cubren potencias, `q=1`, política del terminal, transformación
manufacturada, paridad Python--Numba, RHS callable y despacho por problema. No
hay oráculo Wolfram. Este solver sigue siendo una ODE local reparametrizada: no
modela memoria hereditaria y no debe agregarse a resultados de Caputo/RL como
si fuese la misma familia. Referencia primaria: [R12].

## Caputo--Fabrizio con recurrencia exponencial

### Definición y normalización

Para `0<alpha<1`, HAFO implementa el operador de kernel exponencial

$$
{}^{CF}_{a}D_t^\alpha x(t)=\frac{M(\alpha)}{1-\alpha}
\int_a^t x'(\tau)
\exp\left[-\frac{\alpha}{1-\alpha}(t-\tau)\right]d\tau.
$$

`M(alpha)` siempre se evalúa y registra. El valor por defecto documentado es
`M(alpha)=1`; cualquier otra convención debe pasarse explícitamente. La
recurrencia siguiente es una derivación algebraica propia de HAFO: sobre una
malla uniforme se interpola linealmente cada intervalo y se integra exactamente
el kernel exponencial, lo que da

$$
S_n=\rho S_{n-1}+x_n-x_{n-1},\qquad
D_n=\frac{M(\alpha)(1-\rho)}{\alpha h}S_n,
\quad \rho=e^{-\alpha h/(1-\alpha)}.
$$

### Contrato, costo y pruebas

`caputo_fabrizio_derivative` en
`hidden_attractors/fractional/caputo_fabrizio.py`
es solo un operador sobre muestras. El primer valor es `x(a)`; no hay
prehistoria. La recurrencia Numba/Python cuesta `O(Nd)` y usa `O(d)` de estado
auxiliar. `caputo_fabrizio_derivative_reference` evalúa la misma interpolación
mediante suma directa `O(N^2d)` y sirve de oráculo. `alpha=0` y `alpha=1` son
políticas discretas explícitas de la API, respectivamente
`M(0)(x(t)-x(a))` y una diferencia hacia atrás con `M(1)=1`; en particular, el
segundo caso no se presenta como identidad exacta del operador CF en el extremo.

`tests/test_caputo_fabrizio_operator.py`
comprueba constante, rampa analítica con normalización configurable, linealidad,
extremos, recurrencia frente a suma directa y paridad Numba--Python. No hay un
solver CF FDE habilitado ni un oráculo Wolfram.

### Estado `research_required`

La definición original es [R13]. La recurrencia lineal anterior no se copia ni
se atribuye a Cao--Wang--Xu [R14] o a Liu--Fan--Yin--Li [R32]. Estos últimos
describen una fórmula rápida de segundo orden evaluada en puntos medios, con
coste `O(N)` y memoria `O(1)`, que se conserva solo como método relacionado hasta
comprobar una equivalencia ecuación por ecuación. Cao--Wang--Xu proponen un
solver FDE de alto orden mediante interpolación cuadrática; ese solver tampoco
está implementado por la recurrencia anterior. Diethelm et al. [R15] muestran
problemas con el teorema fundamental y restricciones de compatibilidad inicial
para kernels no singulares. Por ello, la definición permanece
`research_required`, aunque el operador muestral y su oráculo directo sean
ejecutables. El `cf_predictor_corrector` del registro sigue `planned`.

## Orden distribuido: operador GL y solver Caputo L1

HAFO aproxima un operador de orden distribuido

$$
\mathcal{D}_{\mu}x(t)=\int_{q_{min}}^{q_{max}}
D^q x(t)\,d\mu(q)
\approx\sum_{j=1}^{J}\Omega_j D^{q_j}_{h,a}x(t).
$$

`distributed_order_gl_derivative` en
`hidden_attractors/fractional/distributed_order.py`
recibe nodos `q_j`, pesos de cuadratura o masas y, si corresponde, densidades
separadas. Los modos `nonnegative_*` y `signed_*` hacen explícito si se aceptan
factores negativos; la normalización opcional `unit_mass` registra masa
algebraica y norma L1. Cada derivada base puede ser GL/RL cruda o GL desplazada
compatible con Caputo.

El alcance bibliográfico es más estrecho que ese contrato computacional: [R16]
y [R17] respaldan el marco de orden distribuido tipo Caputo con pesos no
negativos. No justifican por sí solos pesos firmados, bases GL/RL crudas ni
ventanas de memoria. HAFO conserva esas opciones como combinaciones algebraicas
diagnósticas y no les atribuye interpretación física ni convergencia general.

Hay dos discretizaciones: cuadratura en el orden y convolución GL en el tiempo.
Con historia completa el costo es `O(JdN^2)`; con ventana, `O(JdNL)`. El kernel
Numba acumula un orden a la vez y conserva solo un vector de pesos, de modo que
la memoria auxiliar es `O(L)` además de la salida, no `O(JNd)`. Existe una
referencia Python equivalente.

El operador muestral está `implemented`; el solver FDE general
`distributed_order_quadrature` continúa `planned`. Las pruebas en
`tests/test_distributed_order_operator.py`
verifican que una masa delta reproduce GL, que una combinación coincide con la
suma de operadores públicos, constante desplazada, `q=1`, ventana, pesos
firmados, normalización, metadatos y paridad Numba--Python. No hay oráculo
Wolfram para las variantes GL/RL firmadas o truncadas. Referencias: Caputo
[R16] y el análisis numérico de Diethelm--Ford
[R17] para el marco Caputo no negativo, junto con [R2] y [R7] para los
operadores GL discretos. La CQ de Yin et al. [R33] es una fuente específica para
un futuro solver de orden distribuido con correcciones; no valida el operador
GL de doble cuadratura ni sus variantes firmadas actuales.

La ruta específica `caputo_distributed_order` no reutiliza ese operador. Para
nodos \(0<q_r\le1\) y masas no negativas forma los coeficientes L1

$$
\rho_r=\frac{\Omega_r h^{-q_r}}{\Gamma(2-q_r)},\qquad
K_k=\sum_r\rho_r\bigl[(k+1)^{1-q_r}-k^{1-q_r}\bigr].
$$

Precomputar \(K_k\) reduce el costo ingenuo `O(R*N^2*d)` a
`O(R*N+N^2*d)`, con almacenamiento `O(N*d+N+R)`. El estado implícito satisface

$$
x_n=x_{n-1}-K_0^{-1}\sum_{k=1}^{n-1}K_k\Delta x_{n-k}
    +K_0^{-1}f(t_n,x_n),
$$

y se resuelve mediante Picard con residuo, tolerancias y no convergencia
estructurada. Una masa en \(q=1\) usa exactamente el límite backward Euler y
desactiva el chequeo \(f(a,x_0)=0\), porque aparece el término clásico
\(x'(a)\). Los pesos firmados siguen siendo sólo diagnósticos del operador GL.

En `FractionalProblem`, `orders` contiene los nodos de la medida y produce
`order_mode="distributed"`; no se replica por componente ni se conserva un
orden nominal ignorado. La cuadratura, pesos efectivos, normalización,
coeficiente corriente, backend real y fronteras de error quedan en metadatos.
La derivación, API y ejemplo están en
[`distributed_order_caputo_l1.md`](distributed_order_caputo_l1.md). La
cuadratura en orden se apoya en [R16], [R17] y [R37]; L1 en [R37], [R38]. El
kernel agregado y el corrector vectorial son adaptaciones HAFO y no heredan una
prueba general de estabilidad para sistemas caóticos.

El caso independiente
`validation/wolfram/cases/distributed_order_caputo_l1.wl` deriva cada peso
multinodo mediante `Integrate` y resuelve una recurrencia lineal sin importar
fórmulas de HAFO. La corrida focal aprobó 6 pruebas; el residuo simbólico, el
residuo de recurrencia y el error de la solución afín fueron cero en el artefacto
Wolfram, y la diferencia máxima Wolfram--HAFO fue
`1.5543122344752192e-15` frente a tolerancia `8e-12`. La solución afín exacta es
una comprobación finita y no una tasa de convergencia.

### Especialización Caputo multitérmino

Para la medida atómica

$$
\mu=\sum_{j=1}^{R}c_j\delta_{\alpha_j},
$$

la integral en el orden es exactamente la suma finita

$$
\sum_{j=1}^{R}c_j\,{}_a^CD_t^{\alpha_j}x(t).
$$

`integrate_multi_term_caputo_l1` fija `weight_semantics="nonnegative_mass"`,
`density_values=None` y `normalization="none"`; por tanto, no acepta opciones
de cuadratura continua que carecen de sentido para esta fachada. Antes de
delegar, agrupa sólo órdenes `float64` exactamente iguales y usa `math.fsum`
para sus coeficientes. Los términos originales, grupos, ceros eliminados y la
suma no normalizada se preservan en `MultiTermCaputoResult`.

La literatura de ecuaciones multiorden y multitérmino [R39], L1 multitérmino
[R40], tratamientos de la capa inicial [R41] y la representación mediante
deltas de Dirac [R42] fijan la frontera matemática. La fachada y su trazabilidad
son adaptaciones HAFO. La implementación no crea un backend adicional: la capa
`O(R log R)` es pequeña y todo el trabajo histórico permanece en el kernel
Numba/Python ya descrito. Véase
[`multi_term_caputo_l1.md`](multi_term_caputo_l1.md).

## Atangana--Baleanu en sentido de Caputo

La definición registrada, para `0<alpha<1`, tiene kernel de Mittag--Leffler:

$$
{}^{ABC}_{a}D_t^\alpha x(t)=\frac{B(\alpha)}{1-\alpha}
\int_a^t x'(\tau)
E_\alpha\left[-\frac{\alpha}{1-\alpha}(t-\tau)^\alpha\right]d\tau.
$$

La normalización `B(alpha)`, el terminal y la compatibilidad inicial deben formar
parte explícita del problema. No es válido reutilizar pesos ABM de Caputo ni la
recurrencia exponencial CF: el kernel es distinto.

`atangana_baleanu_caputo_derivative` implementa sólo un operador sobre muestras
uniformes, no un solver FDE. Para una interpolación lineal de la historia usa

$$
D_n=\frac{B(\alpha)}{1-\alpha}\sum_{j=1}^{n}
(x_j-x_{j-1})w_{n-j},\qquad
w_k=\frac{F((k+1)h)-F(kh)}{h},
$$

con

$$
F(s)=sE_{\alpha,2}\!\left(-\frac{\alpha}{1-\alpha}s^\alpha\right).
$$

Los pesos se calculan desde la serie definitoria de (E_{\alpha,2}), con
comprobaciones explícitas de convergencia, cancelación, positividad y
monotonía. `abc_piecewise_linear_weights` ofrece la construcción Numba/Python;
la convolución completa puede ser directa `O(dN^2)` o FFT fuera de línea
`O(dN log N)`, además del costo `O(NK)` de los (K) términos de serie usados.
No se fija un cruce automático FFT sin benchmark específico.

La ruta conserva la restricción analizada por Yadav--Pandey--Shukla
`0 < alpha <= 1/2` [R29], aunque la definición abstracta existe en
`0 < alpha < 1`. `B(alpha)` se evalúa y registra; el valor por defecto es uno y
`atangana_baleanu_normalization` proporciona explícitamente la convención
`1-alpha+alpha/Gamma(alpha)`. La prueba independiente para `alpha=1/2` usa la
identidad cerrada con `erfcx`; otras pruebas contrastan pesos, linealidad y
paridad Numba--Python--FFT. Solicitudes cuya serie sale del dominio numérico
comprobado se rechazan.

La definición y sus rutas ejecutables están `implemented` bajo su dominio
numérico declarado, y la crítica de
[R15] forma parte del contrato de evidencia. Esto permite investigar una
definición solicitada sin presentar los kernels no singulares como equivalentes
a Caputo ni como libres de restricciones iniciales.

### Predictor--corrector convencional de Lee--Kim--Jang

`integrate_abc_predictor_corrector` implementa la recurrencia convencional de
historia completa de las ecuaciones (9)--(14) de [R30], no el algoritmo rápido
por suma de exponenciales. Para `0 < alpha < 1` resuelve la formulación de
Volterra

$$
x(t)=x_0+c_\alpha f(t,x(t))+
\frac{d_\alpha}{\Gamma(\alpha)}
\int_a^t(t-s)^{\alpha-1}f(s,x(s))\,ds,
$$

con `c_alpha=(1-alpha)/B(alpha)` y `d_alpha=alpha/B(alpha)`. La normalización
`B(alpha)` es un parámetro explícito: acepta un escalar positivo o un callable y
su convención predeterminada registrada es `B(alpha)=1`. El valor evaluado y su
descripción se conservan en el resultado. Para un dato inicial clásico regular,
la ecuación en `t=a` impone `f(a,x0)=0`: HAFO calcula el residuo y rechaza el
problema cuando excede `compatibility_tolerance`.

El artículo presupone un primer valor con error `O(h^2)`, pero no prescribe cómo
obtenerlo. HAFO añade un arranque propio: resuelve por punto fijo la ecuación
implícita de producto--trapecio del primer intervalo, registra iteraciones y
devuelve `startup_no_convergence` si no satisface la tolerancia. Esta decisión de
arranque es de HAFO, no una fórmula atribuida a [R30]. Después aplica los pesos
lineales publicados, expuestos también por `abc_linear_product_weights`, y
conserva toda la historia, con costo `O(d*N^2)` y memoria `O(d*N)`;
`solver_info["fast_soe_used"]` es falso.

La publicación y sus experimentos numéricos justifican la recurrencia escalar.
HAFO aplica el mismo balance por coordenada a un vector con un único orden
conmensurado; esa extensión vectorial es implementación de HAFO y no evidencia
publicada adicional. Una prueba de refinamiento observa orden dos en un problema
escalar cuadrático y suave; junto con la paridad Python--Numba, esto es evidencia
de implementación finita. No garantiza orden dos para un campo no suave como
Chua, ni estabilidad no lineal, caos, atracción u ocultedad.

El caso independiente
`validation/wolfram/cases/abc_predictor_corrector.wl` integra simbólicamente las
bases lineales que generan los pesos y reconstruye una trayectoria manufacturada
sin importar HAFO. El artefacto retenido reporta residuo simbólico cero,
diferencia máxima Wolfram--Python de pesos `3.5388358909926865e-16` y diferencia
de trayectoria `2.220446049250313e-16`. El error finito frente a la solución
Volterra exacta, `4.952013688854784e-4`, se conserva como diagnóstico y no como
prueba de una tasa de convergencia, estabilidad o dinámica caótica.

## Métodos registrados que aún no son rutas completas

| Contrato | Estado | Límite actual y fuente |
|---|---:|---|
| `tempered_symbol_shift_cq` | `planned` | Implementar y validar `[delta(z)/h+lambda]**q` como método separado; no es backend de la CQ por conjugación ya ejecutable. |
| `variable_order_caputo` + `variable_order_pece` | `planned` | Ruta genérica tipo I/II aún no implementada; no sustituye al solver tipo III L1 ya ejecutable. Samko--Ross [R10]. |
| `cf_predictor_corrector` | `planned` bajo definición `research_required` | Implementación separada del operador recurrente; Cao--Wang--Xu [R14] es candidato, sujeto a [R15]. |
| `abc_fast_soe_predictor_corrector` | `planned` | Construir y validar una ruta distinta basada en la aproximación por suma de exponenciales de [R30]; el solver convencional no acepta `fast_history`. |
| `distributed_order_quadrature` | `planned` | Ruta genérica/CQ con correcciones; no sustituye al solver Caputo L1 específico ya ejecutable. Análisis y CQ: [R17], [R33]. |
| `local_ode_transform` | `theoretical_only` | Transformación local conformable; debe permanecer fuera de afirmaciones de memoria hereditaria. |

## Límites y extensiones no implementadas

1. **Historia rápida de Caputo por suma de exponenciales.** Jiang, J. Zhang,
   Q. Zhang y Z. Zhang [R19] reducen el historial directo a
   `O(N*Nexp*d)` de trabajo y `O(Nexp*d)` de memoria auxiliar, con `Nexp`
   dependiente de tolerancia y horizonte. El DOI de la publicación fue
   verificado. Es una referencia primaria para esta posible extensión de
   `fast_history`, que no está implementada en HAFO.
2. **Caputo de orden superior.** Yan--Pal--Ford [R5] describe métodos directos
   y Adams de orden superior bajo hipótesis de suavidad. Esa familia no está
   implementada y requeriría validación contra soluciones con singularidad
   inicial; no forma parte del contrato del ABM actual.
3. **FDE templadas de mayor orden.** Bibi--ur Rehman [R9] cubre IVP/TVP
   templados mediante Newton--Cotes e interpolación generalizada. Es una familia
   distinta del solver Caputo templado por conjugación+ABM ya implementado y no
   debe reemplazarlo sin una comparación ecuación por ecuación.
4. **Orden variable rápido, tipo I/II y métodos globales.** El solver tipo III
   L1 directo ya es ejecutable, pero sigue siendo cuadrático. Fang--Sun--Wang
   [R36] es la referencia para historia rápida tipo III; Ahmed--Izadi--Cattani
   [R11] es candidato para problemas suaves y globales. Ninguno reemplaza
   automáticamente la suma GL congelada ni hereda sus pruebas.
5. **CF de alto orden.** Cao--Wang--Xu [R14] permanece únicamente bajo el estado
   `research_required` y sujeto a las restricciones de [R15].
6. **Historia ABC rápida por SOE.** La variante rápida de [R30] requiere una
   implementación y validación separadas. No debe etiquetarse el
   predictor--corrector convencional `O(N^2)` como rápido ni reutilizar el
   backend FFT del operador muestral como si fuese el algoritmo online.

Wang y Huang, *High order fast algorithm for the Caputo fractional derivative*,
está localizado como preprint `arXiv:1705.06101`; no se encontró un DOI de
publicación revisada por pares que pudiera afirmarse con seguridad en este
corte. Se conserva como referencia contextual de una extensión no implementada,
no como referencia normativa ni como implementación presente.

## Bibliografía verificada

- **[R1]** M. Caputo, “Linear Models of Dissipation whose Q is almost
  Frequency Independent—II”, *Geophysical Journal International* 13(5),
  529–539 (1967). DOI:
  [10.1111/j.1365-246X.1967.tb02303.x](https://doi.org/10.1111/j.1365-246X.1967.tb02303.x).
- **[R2]** I. Podlubny, *Fractional Differential Equations*, Academic Press,
  vol. 198 (1999). ISBN 978-0-12-558840-9.
  [Página del editor](https://shop.elsevier.com/books/fractional-differential-equations/podlubny/978-0-12-558840-9).
- **[R3]** K. Diethelm, N. J. Ford y A. D. Freed, “Detailed Error Analysis for
  a Fractional Adams Method”, *Numerical Algorithms* 36, 31–52 (2004). DOI:
  [10.1023/B:NUMA.0000027736.85078.be](https://doi.org/10.1023/B:NUMA.0000027736.85078.be).
- **[R4]** C. Li y C. Tao, “On the Fractional Adams Method”, *Computers &
  Mathematics with Applications* 58(8), 1573–1588 (2009). DOI:
  [10.1016/j.camwa.2009.07.050](https://doi.org/10.1016/j.camwa.2009.07.050).
- **[R5]** Y. Yan, K. Pal y N. J. Ford, “Higher Order Numerical Methods for
  Solving Fractional Differential Equations”, *BIT Numerical Mathematics* 54,
  555–584 (2014). DOI:
  [10.1007/s10543-013-0443-3](https://doi.org/10.1007/s10543-013-0443-3).
- **[R6]** F. Ghoreishi, R. Ghaffari y N. Saad, “Fractional Order Runge–Kutta
  Methods”, *Fractal and Fractional* 7(3), 245 (2023). DOI:
  [10.3390/fractalfract7030245](https://doi.org/10.3390/fractalfract7030245).
- **[R7]** Ch. Lubich, “Discretized Fractional Calculus”, *SIAM Journal on
  Mathematical Analysis* 17(3), 704–719 (1986). DOI:
  [10.1137/0517050](https://doi.org/10.1137/0517050).
- **[R8]** F. Sabzikar, M. M. Meerschaert y J. Chen, “Tempered Fractional
  Calculus”, *Journal of Computational Physics* 293, 14–28 (2015). DOI:
  [10.1016/j.jcp.2014.04.024](https://doi.org/10.1016/j.jcp.2014.04.024).
- **[R9]** A. Bibi y M. ur Rehman, “A Numerical Method for Solutions of
  Tempered Fractional Differential Equations”, *Journal of Computational and
  Applied Mathematics* 443, 115772 (2024). DOI:
  [10.1016/j.cam.2024.115772](https://doi.org/10.1016/j.cam.2024.115772).
- **[R10]** S. G. Samko y B. Ross, “Integration and Differentiation to a
  Variable Fractional Order”, *Integral Transforms and Special Functions* 1(4),
  277–300 (1993). DOI:
  [10.1080/10652469308819027](https://doi.org/10.1080/10652469308819027).
- **[R11]** H. M. Ahmed, M. Izadi y C. Cattani, “A Spectral Approach to
  Variable-Order Fractional Differential Equations: Improved Operational
  Matrices for Fractional Jacobi Functions”, *Mathematics* 13(16), 2544
  (2025). DOI:
  [10.3390/math13162544](https://doi.org/10.3390/math13162544).
- **[R12]** R. Khalil, M. Al Horani, A. Yousef y M. Sababheh, “A New
  Definition of Fractional Derivative”, *Journal of Computational and Applied
  Mathematics* 264, 65–70 (2014). DOI:
  [10.1016/j.cam.2014.01.002](https://doi.org/10.1016/j.cam.2014.01.002).
- **[R13]** M. Caputo y M. Fabrizio, “A New Definition of Fractional
  Derivative without Singular Kernel”, *Progress in Fractional Differentiation
  and Applications* 1(2), 73–85 (2015). DOI:
  [10.12785/pfda/010201](https://doi.org/10.12785/pfda/010201).
- **[R14]** J. Cao, Z. Wang y C. Xu, “A High-Order Scheme for Fractional
  Ordinary Differential Equations with the Caputo–Fabrizio Derivative”,
  *Communications on Applied Mathematics and Computation* 2, 179–199 (2020).
  DOI:
  [10.1007/s42967-019-00043-8](https://doi.org/10.1007/s42967-019-00043-8).
- **[R15]** K. Diethelm, R. Garrappa, A. Giusti y M. Stynes, “Why Fractional
  Derivatives with Nonsingular Kernels Should Not Be Used”, *Fractional
  Calculus and Applied Analysis* 23, 610–634 (2020). DOI:
  [10.1515/fca-2020-0032](https://doi.org/10.1515/fca-2020-0032).
- **[R16]** M. Caputo, “Distributed Order Differential Equations Modelling
  Dielectric Induction and Diffusion”, *Fractional Calculus and Applied
  Analysis* 4, 421–442 (2001). No se afirma DOI;
  [índice oficial del volumen](https://www.math.bas.bg/complan/fcaa/volume4/index.html).
- **[R17]** K. Diethelm y N. J. Ford, “Numerical Analysis for
  Distributed-Order Differential Equations”, *Journal of Computational and
  Applied Mathematics* 225, 96–104 (2009). DOI:
  [10.1016/j.cam.2008.07.018](https://doi.org/10.1016/j.cam.2008.07.018).
- **[R18]** A. Atangana y D. Baleanu, “New Fractional Derivatives with
  Nonlocal and Non-Singular Kernel: Theory and Application to Heat Transfer
  Model”, *Thermal Science* 20(2), 763–769 (2016). DOI:
  [10.2298/TSCI160111018A](https://doi.org/10.2298/TSCI160111018A).
- **[R19]** S. Jiang, J. Zhang, Q. Zhang y Z. Zhang, “Fast Evaluation of the
  Caputo Fractional Derivative and Its Applications to Fractional Diffusion
  Equations”, *Communications in Computational Physics* 21(3), 650–678
  (2017). DOI:
  [10.4208/cicp.OA-2016-0136](https://doi.org/10.4208/cicp.OA-2016-0136).
- **[R20]** K. Wang y J. Huang, “High Order Fast Algorithm for the Caputo
  Fractional Derivative” (preprint, 2017),
  [arXiv:1705.06101](https://arxiv.org/abs/1705.06101). No se atribuye DOI de
  revista.
- **[R21]** Wolfram Research, “Fractional Calculus” y `NFractionalD`,
  [documentación oficial](https://reference.wolfram.com/language/tutorial/FractionalCalculus.html).
- **[R22]** M. Matusiak, “Fast Evaluation of Grünwald–Letnikov Variable
  Fractional-Order Differentiation and Integration Based on the FFT
  Convolution”, en *Advanced, Contemporary Control*, AISC 1196, 879–890,
  Springer (2020). DOI:
  [10.1007/978-3-030-50936-1_74](https://doi.org/10.1007/978-3-030-50936-1_74).
- **[R23]** Ch. Lubich, “Convolution Quadrature Revisited”, *BIT Numerical
  Mathematics* 44, 503–514 (2004). DOI:
  [10.1023/B:BITN.0000046813.23911.2D](https://doi.org/10.1023/B:BITN.0000046813.23911.2D).
- **[R24]** B. Jin, B. Li y Z. Zhou, “Correction of High-Order BDF Convolution
  Quadrature for Fractional Evolution Equations”, *SIAM Journal on Scientific
  Computing* 39(6), A3129–A3152 (2017). DOI:
  [10.1137/17M1118816](https://doi.org/10.1137/17M1118816).
- **[R25]** F. Jarad, T. Abdeljawad y D. Baleanu, “Caputo-Type Modification of
  the Hadamard Fractional Derivatives”, *Advances in Difference Equations*
  2012, 142 (2012). DOI:
  [10.1186/1687-1847-2012-142](https://doi.org/10.1186/1687-1847-2012-142).
- **[R26]** B. Yin, G. Zhang, Y. Liu y H. Li, “Convolution Quadrature for
  Hadamard Fractional Calculus and Correction Methods for the Subdiffusion
  with Singular Source Terms”, *Communications in Nonlinear Science and
  Numerical Simulation* 138, 108221 (2024). DOI:
  [10.1016/j.cnsns.2024.108221](https://doi.org/10.1016/j.cnsns.2024.108221).
- **[R27]** X. Zheng, “Logarithmic Transformation Between (Variable-Order)
  Caputo and Caputo--Hadamard Fractional Problems and Applications”, *Applied
  Mathematics Letters* 121, 107366 (2021). DOI:
  [10.1016/j.aml.2021.107366](https://doi.org/10.1016/j.aml.2021.107366).
- **[R28]** C. W. H. Green, Y. Liu y Y. Yan, “Numerical Methods for
  Caputo--Hadamard Fractional Differential Equations with Graded and
  Non-Uniform Meshes”, *Mathematics* 9(21), 2728 (2021). DOI:
  [10.3390/math9212728](https://doi.org/10.3390/math9212728).
- **[R29]** S. Yadav, R. K. Pandey y A. K. Shukla, “Numerical
  Approximations of Atangana--Baleanu Caputo Derivative and Its Application”,
  *Chaos, Solitons & Fractals* 118, 58--64 (2019). DOI:
  [10.1016/j.chaos.2018.11.009](https://doi.org/10.1016/j.chaos.2018.11.009).
- **[R30]** S. Lee, H. Kim y B. Jang, “A Novel Numerical Method for Solving
  Nonlinear Fractional-Order Differential Equations and Its Applications”,
  *Fractal and Fractional* 8(1), 65 (2024). DOI:
  [10.3390/fractalfract8010065](https://doi.org/10.3390/fractalfract8010065).
- **[R31]** C. W. H. Green y Y. Yan, “Detailed Error Analysis for a Fractional
  Adams Method on Caputo--Hadamard Fractional Differential Equations”,
  *Foundations* 2(4), 839--861 (2022). DOI:
  [10.3390/foundations2040057](https://doi.org/10.3390/foundations2040057).
- **[R32]** Y. Liu, E. Fan, B. Yin y H. Li, “Fast Algorithm Based on the Novel
  Approximation Formula for the Caputo--Fabrizio Fractional Derivative”, *AIMS
  Mathematics* 5(3), 1729--1744 (2020). DOI:
  [10.3934/math.2020117](https://doi.org/10.3934/math.2020117).
- **[R33]** B. Yin, Y. Liu, H. Li y Z. Zhang, “Approximation Methods for the
  Distributed Order Calculus Using the Convolution Quadrature”, *Discrete and
  Continuous Dynamical Systems - B* 26(3), 1447--1468 (2021). DOI:
  [10.3934/dcdsb.2020168](https://doi.org/10.3934/dcdsb.2020168).
- **[R34]** C. Li, W. Deng y L. Zhao, “Well-Posedness and Numerical Algorithm
  for the Tempered Fractional Differential Equations”, *Discrete and Continuous
  Dynamical Systems - B* 24(4), 1989--2015 (2019). DOI:
  [10.3934/dcdsb.2019026](https://doi.org/10.3934/dcdsb.2019026).
- **[R35]** D. Tavares, R. Almeida y D. F. M. Torres, “Caputo Derivatives of
  Fractional Variable Order: Numerical Approximations”, *Communications in
  Nonlinear Science and Numerical Simulation* 35, 69--87 (2016). DOI:
  [10.1016/j.cnsns.2015.10.027](https://doi.org/10.1016/j.cnsns.2015.10.027).
- **[R36]** Z. W. Fang, H. W. Sun y H. Wang, “A Fast Method for Variable-Order
  Caputo Fractional Derivative with Applications to Time-Fractional Diffusion
  Equations”, *Computers & Mathematics with Applications* 80, 1443--1458
  (2020). DOI:
  [10.1016/j.camwa.2020.07.009](https://doi.org/10.1016/j.camwa.2020.07.009).
- **[R37]** Z. Hu, F. Liu, V. Anh e I. Turner, “Numerical Methods for the Time
  Distributed-Order Superdiffusion Equation”, *ANZIAM Journal* 55,
  C464--C478 (2014). DOI:
  [10.21914/ANZIAMJ.V55I0.7888](https://doi.org/10.21914/ANZIAMJ.V55I0.7888).
- **[R38]** Y. Lin y C. Xu, “Finite Difference/Spectral Approximations for the
  Time-Fractional Diffusion Equation”, *Journal of Computational Physics* 225,
  1533--1552 (2007). DOI:
  [10.1016/j.jcp.2007.02.001](https://doi.org/10.1016/j.jcp.2007.02.001).
- **[R39]** K. Diethelm y N. J. Ford, “Multi-Order Fractional Differential
  Equations and Their Numerical Solution”, *Applied Mathematics and
  Computation* 154, 621--640 (2004). DOI:
  [10.1016/S0096-3003(03)00739-2](https://doi.org/10.1016/S0096-3003(03)00739-2).
- **[R40]** J. Ren y Z.-Z. Sun, “Efficient and Stable Numerical Methods for
  Multi-Term Time Fractional Sub-Diffusion Equations”, *East Asian Journal on
  Applied Mathematics* 4, 242--266 (2014). DOI:
  [10.4208/EAJAM.181113.280514A](https://doi.org/10.4208/EAJAM.181113.280514A).
- **[R41]** M. She, D. Li y H.-W. Sun, “A Transformed L1 Method for Solving the
  Multi-Term Time-Fractional Diffusion Problem”, *Mathematics and Computers in
  Simulation* 193, 584--606 (2022). DOI:
  [10.1016/j.matcom.2021.11.005](https://doi.org/10.1016/j.matcom.2021.11.005).
- **[R42]** M. A. Zaky y J. A. Tenreiro Machado, “Multi-Dimensional Spectral
  Tau Methods for Distributed-Order Fractional Diffusion Equations”,
  *Computers & Mathematics with Applications* 79, 476--488 (2020). DOI:
  [10.1016/j.camwa.2019.07.008](https://doi.org/10.1016/j.camwa.2019.07.008).

## Regla de uso científico

Todo resultado debe conservar en sus metadatos la definición, método, terminal,
tipo de condición inicial, orden u órdenes, paso, política de memoria, ventana,
backend y referencias. Comparar dos trayectorias solo es científicamente válido
si esos contratos coinciden o si la diferencia es precisamente el objeto del
experimento. Una trayectoria finita y la concordancia entre dos backends prueban
consistencia de implementación dentro de tolerancia; no prueban por sí solas
dinámica caótica, existencia de atractor ni ocultamiento de su cuenca.
