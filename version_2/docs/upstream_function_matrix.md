# Matriz de funciones upstream para HAFO

**Auditoría técnica y de interoperabilidad — revisión 2026-08-03**

Este documento compara el catálogo observable de HAFO `version_2` con las API
oficiales de **pynamicalsys** (el proyecto al que a veces se alude como
“pydinamicalsys”) y del ecosistema **DynamicalSystems.jl**. Su objetivo es decidir
qué conviene implementar de forma nativa, acelerar con Numba o C, o exponer como
adaptador opcional de Julia.

No es una afirmación de equivalencia científica ni un benchmark. Que una función
acepte una trayectoria fraccionaria no demuestra que el integrador, el atractor,
el exponente de Lyapunov o la hiddenness sean correctos. Las versiones `stable`
de la documentación son móviles; cualquier integración reproducible deberá fijar
versiones, hashes y un `Manifest.toml`.

## Decisión ejecutiva

La arquitectura más eficiente y mantenible es híbrida:

1. **HAFO conserva el núcleo matemático y los contratos fraccionarios.** Ni
   pynamicalsys ni los paquetes principales auditados de DynamicalSystems.jl
   sustituyen un solucionador de ecuaciones diferenciales fraccionarias con
   historial. El operador, la inicialización, la política de memoria y el método
   numérico deben permanecer explícitos en HAFO.
2. **C es el backend de producción para los bucles de mayor coste:** convolución
   histórica, memoria rápida, grandes ensambles, distancias/recurrencia y, cuando
   la teoría sea válida, propagación tangente y residuos de shooting.
3. **Numba es el backend general para funciones de usuario y algoritmos medianos:**
   RHS/Jacobianos/mapas, prototipos verificables, extracción de eventos y métricas
   que deben funcionar sin una instalación de Julia.
4. **Julia debe ser una extensión opcional por lotes**, no una dependencia del
   bucle interno: ComplexityMeasures.jl, DelayEmbeddings.jl, RecurrenceAnalysis.jl,
   FractalDimensions.jl, Attractors.jl y PeriodicOrbits.jl ofrecen mucha amplitud
   y evolucionan con rapidez. El intercambio óptimo es `Trajectory -> ndarray /
   StateSpaceSet -> resultado estructurado`, con una llamada gruesa mediante
   [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/).
5. **pynamicalsys se usa como referencia funcional, no como dependencia ni fuente
   para copiar.** Su repositorio declara GPL-3.0, mientras HAFO se distribuye con
   otra licencia permisiva. Importar o traducir su código al núcleo introduciría
   obligaciones de licencia. Los algoritmos necesarios deben reimplementarse de
   manera independiente desde artículos primarios, conservando atribución.

En particular, una función Julia no debe invocarse desde C/Numba en cada paso del
integrador. Esa frontera destruye la posibilidad de compilación `nopython`, añade
coste FFI y crea dos semánticas de memoria. Se debe ejecutar toda la simulación en
un solo backend o enviar una trayectoria completa a una rutina de análisis.

## Convenciones de esta matriz

| Código | Alcance | Regla para HAFO |
|---|---|---|
| **I** | Sistema entero clásico | Se puede implementar o adaptar con la teoría usual de ODE/mapas. |
| **T** | Transformación de datos o trayectoria | Se aplica a muestras enteras o fraccionarias sin cambiar el algoritmo, siempre que se registren muestreo, transitorio, proyección y prehistoria. |
| **F** | Requiere teoría fraccionaria | No se puede portar literalmente: depende del operador, el historial, la inicialización o una ecuación variacional no local. |
| **D** | Ecuación en diferencias fraccionaria | Es distinta de un mapa iterado entero y necesita contrato propio. |

“T” significa reutilización computacional, no equivalencia dinámica. Por ejemplo,
RQA puede calcularse sobre `x(t)` de un sistema Caputo, pero describe esa proyección
finita y no necesariamente el estado hereditario completo.

### Estado HAFO observado durante la auditoría

El catálogo y el código actuales ya contienen:

- flujos enteros con RK4 y rutas adaptativas, mapas discretos y ABI genérica
  Numba;
- Caputo ABM/PECE y EFORK; ABM/PECE Caputo--Hadamard en tiempo logarítmico;
  Caputo templado por conjugación exponencial más ABM; Caputo de orden variable
  tipo III mediante L1/Picard; Caputo de orden distribuido mediante kernel L1
  combinado/Picard; predictor--corrector ABC
  convencional; y solver conformable local;
- operadores muestreados GL, Riemann--Liouville, RL templado,
  Caputo--Fabrizio, ABC, orden variable y orden distribuido, junto con contratos
  diferenciados para las rutas todavía planificadas;
- Poincaré, máximos, barridos de bifurcación, cuencas, continuación, Welch/FFT,
  prueba 0–1 y Lyapunov entero; las rutas Lyapunov fraccionarias están etiquetadas
  como experimentales;
- embedding generalizado con retardos firmados, ACF/MI y FNN; auto/cross/joint
  recurrence densa con L1/L2/Linf, radio o tasa global, Theiler y RQA ampliada;
- SALI/GALI/LDI para historiales tangentes, flujos y mapas enteros `q=1`,
  con propagación variacional o multiparticle y backends NumPy/Numba;
- CLV de Ginelli para historias QR, flujos y mapas enteros `q=1`, con QR
  NumPy/LAPACK, recursión backward NumPy/Numba y ángulos de pares/subespacios;
- adaptadores opcionales limitados para entropías, dimensión de correlación,
  Rosenstein, Hurst, DFA e Higuchi.

No se observaron todavía implementaciones fraccionarias de CLV, búsqueda
robusta de órbitas periódicas, RQA sparse/por ventanas/multiumbral, redes de
recurrencia, PECUZAL, el marco completo de complejidad o la familia completa de
dimensiones. CLV y SALI/GALI/LDI fraccionarios permanecen
`research_required` por operador y semántica de memoria.

## Versiones y licencias observadas

| Proyecto | Evidencia oficial consultada | Licencia observada | Decisión |
|---|---|---|---|
| pynamicalsys | [API continua](https://pynamicalsys.readthedocs.io/en/stable/api/cds.html), [API discreta](https://pynamicalsys.readthedocs.io/en/stable/api/dds.html), [repositorio](https://github.com/mrolims/pynamicalsys) | GPL-3.0 en el repositorio; la API continua mostró 1.6.0, mientras páginas cacheadas bajo `stable` mostraron números anteriores | No importar, enlazar ni copiar al núcleo MIT/permisivo. Usar especificación funcional y artículos primarios. |
| DynamicalSystems.jl | [contenido oficial actual](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/dynamicalsystems/stable/contents/), [repositorio](https://github.com/JuliaDynamics/DynamicalSystems.jl) | Metapaquete; verificar la licencia y el árbol de dependencias del commit fijado | Adaptador opcional; no convertirlo en requisito del solver HAFO. |
| ChaosTools.jl | [documentación](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/chaostools/stable/), [repositorio](https://github.com/JuliaDynamics/ChaosTools.jl) | MIT | Referencia/oráculo opcional; reimplementar localmente las funciones P0/P1. |
| Attractors.jl | [documentación](https://juliadynamics.github.io/Attractors.jl/dev/), [repositorio](https://github.com/JuliaDynamics/Attractors.jl) | MIT | Adaptador por lotes útil para mapeadores, continuación global y estabilidad no local. |
| RecurrenceAnalysis.jl | [documentación](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/recurrenceanalysis/stable/), [repositorio](https://github.com/JuliaDynamics/RecurrenceAnalysis.jl) | MIT; el repositorio mostraba v2.1.4 como release más reciente observada | Llevar el subconjunto P0 a C/Numba; Julia conserva funciones avanzadas opcionales. |
| ComplexityMeasures.jl | [documentación](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/), [repositorio](https://github.com/JuliaDynamics/ComplexityMeasures.jl) | MIT | Preferir JuliaCall opcional para la amplitud total; núcleo local reducido y estable. |
| DelayEmbeddings.jl | [documentación](https://juliadynamics.github.io/DelayEmbeddings.jl/stable/), [repositorio](https://github.com/JuliaDynamics/DelayEmbeddings.jl) | MIT | `embed/genembed` local; métodos de selección avanzada como extensión Julia. |
| PeriodicOrbits.jl | [documentación](https://juliadynamics.github.io/PeriodicOrbits.jl/dev/), [repositorio](https://github.com/JuliaDynamics/PeriodicOrbits.jl) | El repositorio contiene `LICENSE.md`, pero la vista consultada no identificó su SPDX | No distribuir el adaptador sin comprobar y registrar la licencia del commit fijado. |
| FractalDimensions.jl | [documentación](https://juliadynamics.github.io/FractalDimensions.jl/stable/), [repositorio](https://github.com/JuliaDynamics/FractalDimensions.jl) | MIT | Núcleo C/Numba para sumas de correlación; Julia para la familia completa y como comparación. |

Esta tabla no sustituye una revisión legal. Para cualquier binario o entorno
Julia distribuido se debe generar un SBOM con versión, `git-tree-sha1`, licencia
y licencia transitiva.

## Matriz pynamicalsys frente a HAFO

La clase continua oficial representa exclusivamente
`du/dt = f(t,u,p)` y ofrece RK4/RK45. Por tanto, no es un motor fraccionario. Su
valor para HAFO está en la forma de la API y en su catálogo de diagnóstico.

| Grupo oficial | Funciones representativas | Estado HAFO | Clase | Acción recomendada |
|---|---|---|---|---|
| Definición y evolución de flujos | `ContinuousDynamicalSystem`, `integrator`, `evolve_system`, `trajectory`; RHS y Jacobiano de usuario | ABI y solvers enteros presentes; contratos fraccionarios presentes | I/F | Adoptar una interfaz uniforme `Problem + Solver + Trajectory`, no el objeto GPL. El RHS de usuario `(t,u,p)` puede adaptarse trivialmente a HAFO si es código del usuario. El método fraccionario recibe además operador, orden, terminal inferior, prehistoria y memoria. |
| RK4/RK45 | selector `rk4`/`rk45` | RK4 y ruta adaptativa ya existen | I | Mantener Numba para RHS arbitrario y C para modelos registrados/ensambles; verificar contra problemas ODE conocidos. No reutilizar RK45 para una FDE no local. |
| Poincaré, estroboscópico y máximos | `poincare_section`, `stroboscopic_map`, `maxima_map` | Sección y máximos disponibles | T/F | Completar eventos con interpolación y orientación. En fraccionario son observables de trayectoria; el retorno sobre `x` no es un mapa Markoviano salvo que el estado incluya memoria suficiente. |
| Cuencas continuas | `basin_of_attraction` con reducción PS/SM y DBSCAN | Clasificación y sondeos de cuenca presentes | T/F | C/Numba para ensambles y clustering separado. En FDE, dos puntos `x(t0)` sin la misma prehistoria no son necesariamente la misma condición inicial. Registrar política de historia y etiqueta “muestreo finito”. |
| Lyapunov | `lyapunov` | Entero implementado; fraccionario experimental | I/F | C/Numba Benettin/QR para entero. Para cada derivada no local validar la ecuación variacional y la renormalización del historial; no promocionar el resultado clonado a espectro canónico. |
| Vectores covariantes y alineamiento | `CLV`, `CLV_angles`, `SALI`, `LDI`, `GALI` | CLV/ángulos y SALI/GALI/LDI enteros implementados; variantes F pendientes | I/F | CLV usa QR NumPy/LAPACK y recursión backward NumPy/Numba; alineamiento conserva NumPy/SVD y Numba/Householder. Las variantes fraccionarias son líneas de investigación por operador y memoria, no un cambio de integrador. |
| Recurrence-time entropy y Hurst | `recurrence_time_entropy`, `hurst_exponent` | Hurst opcional; RQA parcial, no RTE equivalente completa | T | Implementar una API de datos común con Theiler window, método de umbral y estimador explícitos. Comparar sobre trayectorias congeladas; no usar como prueba única de caos. |
| Mapas discretos | `DiscreteDynamicalSystem`, `step`, `trajectory`, `bifurcation_diagram` | Mapas y barridos disponibles | I | Mantener kernel Numba genérico y C para modelos/ensambles. Separar totalmente de ecuaciones en diferencias fraccionarias (D). |
| Puntos y órbitas periódicas de mapas | `period`, `find_periodic_orbit`, estabilidad, autovalores/autovectores, variedades | Parcial en equilibrios; órbitas periódicas planeadas | I/D | Numba/C para mapas enteros. Una ecuación en diferencias fraccionaria requiere estado de historia y contrato D; no pasarla a esta API por conveniencia. |
| Lyapunov finito y CLV para mapas | `lyapunov`, variantes de tiempo finito, `CLV`, ángulos | CLV y ángulos para mapas enteros implementados; espectro/variantes D conservan su evidencia separada | I/D | Compartir QR y recursión Ginelli con el núcleo entero. Extensión D sólo con ecuación variacional discreta fraccionaria publicada/validada. |
| Alineamiento de mapas | `SALI`, `LDI`, `GALI` | Implementado para mapas enteros por Jacobiano o partículas | I/D | NumPy/Numba validados con fixtures exactos; D queda bloqueado por teoría y por semántica de memoria. |
| Escape, supervivencia y transporte | escape/survival, difusión, MSD y promedios | No hay familia completa | T/F | Métricas de trayectoria P2 en Numba/C. Declarar región, censoring, ensemble y prehistoria; “escape” proyectado no equivale a abandonar el estado histórico. |
| Rotación y variedades | rotation number, stable/unstable manifolds | No hay paridad completa | I/F | Rotación desde datos puede ser T; variedades invariantes son I y requieren una teoría de espacio de historia para F. Prioridad baja para HAFO. |
| Métricas de cuenca | `basin_entropy`, `uncertainty_fraction` | Implementadas como diagnósticos finitos; faltan barridos/protocolos multiescala | T/F | Consolidar tamaño de malla, escala/desplazamiento de cajas, muestra, clasificador, tolerancias e historia. |
| Métricas de series | `recurrence_matrix`, recurrence-time entropy, Hurst R/S | Matriz/RQA parcial y adaptadores | T | Ampliar localmente sin depender de pynamicalsys. Usar sus resultados sólo como comparación externa reproducible, nunca como oráculo único. |
| Sistemas Hamiltonianos | API Hamiltoniana separable e integradores asociados | Fuera del núcleo actual | I/F | P2 entero si Toolbox Chaos lo exige. “Hamiltoniano fraccionario” necesita formulación específica y no debe inferirse por reemplazar `d/dt`. |

### Frontera de licencia de pynamicalsys

Son aceptables:

- leer su API y los artículos citados para construir requisitos;
- escribir pruebas de comportamiento con datos propios;
- ejecutar una comparación externa en un entorno independiente y registrar la
  versión GPL;
- adaptar una función RHS/Jacobiano que pertenece al usuario, sin importar
  pynamicalsys.

No se recomienda para el núcleo distribuido de HAFO:

- copiar o traducir sus kernels Numba;
- importar pynamicalsys como dependencia obligatoria o “opcional” silenciosa;
- envolver sus clases dentro de la API pública sin una decisión explícita de
  licencia;
- publicar sus afirmaciones de velocidad como rendimiento de HAFO. Los tiempos
  deben medirse en HAFO, con warm-up JIT separado, mismas tolerancias y hardware.

## Matriz DynamicalSystems.jl por paquete

### DynamicalSystemsBase / StateSpaceSets

| Función upstream | Estado o equivalente HAFO | Clase | Integración eficiente |
|---|---|---|---|
| Sistemas continuos/discretos, `trajectory`, `step!`, parámetros y estado | Contratos repartidos entre sistemas, integradores y resultados | I/F/D | Inspirar una API común, pero conservar `FractionalProblem` separado. Un `CoupledODEs` no representa la memoria de una FDE. |
| `StateSpaceSet` y operaciones sobre conjuntos | HAFO usa `ndarray` | T | Definir frontera canónica `float64`, forma `(n_samples,n_features)`, tiempo aparte y metadatos. Convertir una vez al entrar/salir de Julia. |
| Tangent dynamical systems | Lyapunov entero y rutas fraccionarias experimentales | I/F | Entero en C/Numba. Para F, construir el sistema tangente del operador completo y comprobar sensibilidad a ventana/historial. |
| Parallel/ensemble dynamical systems | Ensambles disponibles en varios workflows | I/F | C para integración masiva y Python para planificación; prehistoria individual o compartida debe ser parte del lote. |

No debe pasarse una closure Julia como callback de paso a un kernel Numba/C. Si el
modelo sólo existe en Julia, la simulación completa debe ejecutarse allí y HAFO
recibe la trayectoria. Si se desea el solver HAFO, se implementa el RHS matemático
en su ABI con pruebas cruzadas.

### ChaosTools.jl

Documentación oficial: [diagramas orbitales](https://juliadynamics.github.io/ChaosTools.jl/stable/orbitdiagram/),
[Lyapunov](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/chaostools/stable/lyapunovs/),
[detección de caos](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/chaostools/dev/chaos_detection/),
[periodicidad](https://juliadynamics.github.io/ChaosTools.jl/stable/periodicity/) y
[eventos raros](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/chaostools/dev/rareevents/).

| Grupo | API upstream | HAFO | Clase | Prioridad/backend |
|---|---|---|---|---|
| Diagramas y mapas reducidos | `orbitdiagram`, `PoincareMap`, `StroboscopicMap` | Barridos, Poincaré y máximos | T/F | **P0**, Python+Numba; conservar operador, orden, memoria y política de continuación por punto. |
| Espectro y máximo Lyapunov | `lyapunovspectrum` por QR, `lyapunov` por trayectorias, crecimiento local, `lyapunov_from_data` | Entero sí; F experimental; adaptador Rosenstein | I/T/F | **P0/P1**, C/Numba. `lyapunov_from_data` es T; espectro tangente es F para sistemas no locales. |
| GALI y prueba 0–1 | `gali`, `testchaos01`, `predictability` | 0–1 y SALI/GALI/LDI enteros implementados | T/I/F | GALI entero usa NumPy/SVD y Numba/Householder; C se difiere hasta que un benchmark justifique un kernel. GALI fraccionario requiere teoría. |
| Puntos fijos y periodicidad | `fixedpoints`, `periodicorbits`, `davidchacklai`, `estimate_period` | Equilibrios sí; PO no | I/T/F | Estimación de periodo T en Numba; roots/PO enteros C/Numba. No aplicar shooting ODE al estado físico fraccionario. |
| Tiempos de retorno/salida/transición | `mean_return_times`, `exit_entry_times`, transit/return times | Parcial mediante series/workflows | T/F | **P2**, Numba; región y prehistoria explícitas. |
| Reducción dimensional | Broomhead–King, DyCA | Delay embedding simple | T | **P2**, Julia opcional primero; integrar local sólo con demanda y tests. |

### Attractors.jl

La API oficial incluye mapeadores por proximidad, recurrencia y características,
extracción de atractores, fracciones/cuencas, continuación global, estabilidad no
local, fronteras de cuenca, edge states y tipping.

| Grupo | API upstream representativa | HAFO | Clase | Prioridad/backend |
|---|---|---|---|---|
| Mapeadores | `AttractorsViaProximity`, `AttractorsViaRecurrences`, `AttractorsViaFeaturizing`, `map_to_basin` | Clasificadores y sondeos específicos | T/F | **P0/P1**, Python para estrategia + C/Numba para trayectorias/distancias. JuliaCall opcional para mapeadores avanzados. |
| Extracción y cuencas | `extract_attractors`, `basins_fractions`, `basins_of_attraction` | Cuencas presentes | T/F | Ensambles C. Contrato de terminación/clasificador y prehistoria obligatorio. |
| Continuación global | `global_continuation`, seed/continue/match, agregación | Continuación HAFO presente | F | **P1**. Orquestación Python; Julia opcional para comparar matching. Cada paso debe declarar si continúa historia, reinicia o usa ventana. |
| Estabilidad no local | acumuladores, medidas y cuantificadores de estabilidad | Métricas parciales | T/F | **P1/P2**, Julia opcional. Una estimación finita de cuenca/resiliencia no es prueba global. |
| Fronteras, edge states y tipping | herramientas de boundary/edge/tipping | Hueco | I/F | **P2**. En F el estado es histórico; cualquier extensión requiere formular la frontera en ese espacio o declarar que sólo se estudia una proyección. |

Para investigación exploratoria, Attractors.jl es el candidato Julia de mayor
valor. Para una API estable de Toolbox Chaos, HAFO debe poseer el contrato, el
muestreo y los resultados; Julia queda detrás de una feature opcional.

### RecurrenceAnalysis.jl

Fuentes oficiales: [recurrence plots](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/recurrenceanalysis/stable/rplots/),
[RQA](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/recurrenceanalysis/stable/quantification/) y
[redes de recurrencia](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/recurrenceanalysis/stable/networks/).

| Función upstream | HAFO actual | Clase | Acción |
|---|---|---|---|
| `RecurrenceMatrix`, `CrossRecurrenceMatrix`, `JointRecurrenceMatrix` | Auto/cross/joint densas implementadas | T | Consolidar ABI; C sólo si benchmarks muestran ventaja sobre bloques Numba. |
| Radio fijo, radio escalado, tasa global/local fija; métricas configurables | Radio fijo o tasa global; L1/L2/Linf | T | Falta radio escalado y tasa local fija; preservar política y empates. |
| Sparse/threaded recurrence | Bloques Numba/NumPy pero salida densa | T | **P0/P1**: sparse y estimación sin materializar `N²`; medir memoria y crossover. |
| Theiler window, `lmin`, `vmin`, bordes | Theiler general y mínimos de línea explícitos | T | Falta catálogo de reglas de borde alternativas; no afirmar paridad total. |
| RR, DET, L, Lmax, DIV, ENTR, TREND | Implementados con normalizaciones declaradas | T | Comparar convenciones contra fixtures Julia y literatura antes de estabilizar. |
| LAM, TT, Vmax, entropía vertical | Implementados | T | Añadir RQA por ventanas y sensibilidad de umbral. |
| MRT, RTE, NMPRT | No completos | T | **P1**: histogramas/definiciones explícitas y casos censurados. |
| Windowed RQA | No | T | **P1**: ventanas sin copias y tiempo central explícito. |
| `rna`: densidad, transitividad, camino medio, diámetro | No | T | **P2**: Julia opcional; C/graph backend sólo si el uso justifica la dependencia. |

Todas estas funciones son T: admiten trayectorias válidas de cualquier operador.
Se deben guardar `dt`, irregularidad de muestreo, normalización, embedding, variables
proyectadas, Theiler window, umbral y política de memoria del solver. Dos RQA con
distintas proyecciones del estado histórico no son intercambiables.

### ComplexityMeasures.jl

El paquete ofrece marcos extensibles para espacios de resultados, estimadores de
probabilidad con corrección de sesgo, medidas de información discretas y
diferenciales, entropías y complejidades. Su [API de complejidad](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/complexity/)
incluye, entre otras, ApproximateEntropy, SampleEntropy, Lempel–Ziv 76,
Missing/Reverse Dispersion, BubbleEntropy y StatisticalComplexity; también ofrece
análisis [multiescala](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/multiscale/).

| Grupo | HAFO | Clase | Decisión |
|---|---|---|---|
| Espacios de resultados y probabilidades | `TrajectoryInput`/`PrehistorySpec`/`AnalysisResult` ya congelan datos y procedencia; no hay aún un marco general de estimadores de probabilidad | T | Mantener el sobre HAFO local y usar **P1 JuliaCall** para el catálogo combinatorio amplio, sin duplicarlo de inmediato. |
| Shannon, Rényi, Tsallis y estimadores diferenciales | Adaptadores puntuales | T | API HAFO pequeña con metadatos; delegación Julia por lotes para catálogo amplio. |
| Permutation/sample/spectral entropy | Bandt--Pompe ya es local en Python/Numba/C con despacho dimensionado por coste factorial; sample/spectral conservan rutas existentes | T | Histograma denso (m!), Lehmer, retardo, base y empates explícitos; Wolfram independiente y benchmark m=2--10. **P1 siguiente**: sesgo/intervalos y sample entropy local. |
| Lempel–Ziv, dispersion, bubble, statistical complexity | Ausentes | T | **P1/P2 JuliaCall**; promover a local sólo si Toolbox Chaos exige uso frecuente/offline. |
| Multiescala y curvas entropía–complejidad | Ausente | T | **P2 JuliaCall**, devolviendo escalas, estimador, unidades y sesgo; no sólo un escalar. |

Estas métricas no requieren modificar la derivada, pero sí controlar longitud,
estacionariedad, frecuencia de muestreo, coarse graining, cuantización y sesgo. Una
entropía elevada no prueba caos ni hiddenness.

### DelayEmbeddings.jl

Fuentes oficiales: [embedding](https://juliadynamics.github.io/DelayEmbeddings.jl/stable/embed/),
[selección separada](https://juliadynamics.github.io/DelayEmbeddings.jl/stable/separated/) y
[selección unificada/PECUZAL](https://juliadynamics.github.io/DelayEmbeddings.jl/stable/unified/).

| Función upstream | HAFO | Clase | Acción |
|---|---|---|---|
| `embed`, `DelayEmbedding` | Embedding uniforme escalar/multivariable | T | **P0**, conservar implementación NumPy/Numba y estandarizar forma/slices. |
| `genembed`, `GeneralizedEmbedding`, `τrange` | Pares columna/lag arbitrarios y alineación exacta | T | Falta constructor de rangos y vistas sin copia; la API matemática básica está implementada. |
| `estimate_delay`: autocorrelación, información mutua y otros criterios | ACF-FFT e MI histográfica separadas | T | Añadir estimadores alternativos y comparación cruzada Julia, sin selección opaca. |
| Cao/AFNN, FNN y variantes | FNN KBA con cKDTree, Theiler y tres métricas | T | Cao/AFNN siguen pendientes; validar sensibilidad y ruido antes de estabilizar. |
| `optimal_separated_de` | No | T | **P1**, orquestación Python con estimadores intercambiables. |
| `pecuzal_embedding` y selección unificada multivariable | No | T/F | **P1/P2 JuliaCall** primero. En F, usar como reconstrucción empírica y no afirmar automáticamente un embedding difeomorfo finito del estado hereditario. |

El vector `x(t), x(t-τ), ...` es directamente procesable, pero una FDE no local
puede tener dimensión efectiva de historia infinita. HAFO debe distinguir
“reconstrucción útil para datos” de “reconstrucción demostrada del espacio de
estados”.

### PeriodicOrbits.jl

La [API/tutorial oficial](https://juliadynamics.github.io/PeriodicOrbits.jl/dev/tutorial/)
expone `periodic_orbit`, `periodic_orbits`, `InitialGuess`, resultados de órbita y
algoritmos para mapas (Schmelcher–Diakonos, Davidchack–Lai) y flujos
(`OptimizedShooting`). También incluye utilidades como periodo mínimo, igualdad,
distancia y estabilidad.

| Función | HAFO | Clase | Acción |
|---|---|---|---|
| Periodic points en mapas | No robusto | I/D | **P2 C/Numba** para mapas enteros. D necesita historia discreta y teoría propia. |
| Shooting optimizado de flujos | No | I/F | **P2** entero en C/Numba o extensión Julia. Para F no imponer `x(T)=x(0)` como condición suficiente. |
| Periodo mínimo/deduplicación/distancia | Parcial ad hoc | T/I/F | Implementar utilidades de datos, pero etiquetar la norma y si compara sólo estado físico o también historia. |
| Estabilidad de PO | No | I/F | Entero mediante monodromía/Floquet; en F requiere operador de evolución en el espacio de historia y literatura específica. |

Una respuesta periódica fraccionaria puede estudiarse con historia periódica,
forzamiento asintótico o formulación funcional, pero no con el shooting ODE sobre
`x` sin más. Hasta existir validación primaria, HAFO debe marcar esta capacidad
fraccionaria como `research_required`.

### FractalDimensions.jl

La documentación oficial incluye `generalized_dim`,
`grassberger_proccacia_dim`, sumas de correlación directa/boxed/fixed-mass,
`takens_best_estimate_dim`, dimensiones puntuales/locales, `kaplanyorke_dim`,
`higuchi_dim`, estimadores EVT y ajuste explícito de regiones de escala mediante
`slopefit`.

| Grupo | HAFO | Clase | Acción |
|---|---|---|---|
| Sumas y dimensión de correlación \(q=2\) | Implementación local Python/Numba/C | T | **Disponible**: pares no ordenados, ventana de Theiler, tres métricas, conteos y curva normalizada, pendiente/local slopes y rango de ajuste explícito. Mantener Julia sólo como comparación opcional. |
| Dimensión generalizada/Rényi | No completa | T | **P1 JuliaCall**, ligada a ComplexityMeasures; implementación local posterior. |
| Fixed-mass, Takens best estimate, local/pointwise | No | T | **P1/P2 JuliaCall**, con intervalos y sensibilidad a escala. |
| Kaplan–Yorke | Fácil de calcular; no contrato completo | I/F | Fórmula local, pero su validez hereda la del espectro. Un espectro F experimental produce una dimensión F experimental. |
| Higuchi | Adaptador opcional | T | Mantener local con muestreo uniforme explícito; mide rugosidad de la gráfica, no dimensión del atractor. |
| EVT: dimensión, extremal index, persistencia | No | T | **P2 JuliaCall**; conservar umbrales, ajuste, p-values y diagnóstico de cola. |
| Detección de región de escala | No automática | T | La API exige un intervalo inclusivo declarado por quien llama y entrega curva, puntos usados y método de ajuste. **P1**: añadir sensibilidad e incertidumbre antes de considerar heurísticas trazables. |

La propia documentación advierte que las funciones de conveniencia automatizan
heurísticas de escala. HAFO no convierte una pendiente automática en evidencia
de dimensión: el contrato actual exige curva, rango explícito, ventana de
Theiler y procedencia del muestreo/proyección. La incertidumbre estadística y la
selección automática siguen fuera del alcance actual; véase
[`correlation_dimension.md`](correlation_dimension.md).

## Matriz priorizada C vs Numba vs Julia

| Prioridad | Capacidad HAFO | Entero | Fraccionario | C | Numba | Julia opcional | Criterio de aceptación |
|---|---|---:|---:|---|---|---|---|
| **P0** | Contrato `Problem/Solver/Trajectory` y metadatos | I | F/D | ABI de buffers | Validación y adaptadores | Conversión por lote | Round-trip sin perder `t`, orden, derivada, terminal, prehistoria, memoria, solver y tolerancias. |
| **P0** | Convolución/memoria FDE | — | F | **Principal**: pesos, historial, FFT/SOE cuando proceda | Referencia pequeña y RHS de usuario | No en inner loop | Soluciones manufacturadas, convergencia y comparación independiente por familia de derivada. |
| **P0** | Flujos/mapas genéricos enteros | I | —/D | Modelos registrados y ensambles | **Principal** para funciones de usuario | Simulación completa alternativa | Paridad de trayectoria bajo mismas tolerancias; separar JIT warm-up. |
| **P0** | Recurrence matrix y RQA completa | T | T | **Principal** para distancias, bloques, sparse | Referencia y líneas/histogramas | Oráculo/extensión | Matrices exactas pequeñas, paridad de definiciones, memoria subcuadrática opcional. |
| **P0** | Poincaré/stroboscopic/máximos | T | T/F | Eventos masivos opcionales | **Principal** | Comparación | Interpolación y orientación verificadas; advertencia no-Markov para proyección F. |
| **P0** | Ensambles, cuencas y clasificación | I/T | T/F | **Principal** para integración masiva | Features/clustering y fallback | Mapeadores avanzados | Repetibilidad, convergencia de malla/muestra, tasa no clasificada e historia explícita. |
| **P0** | Basin entropy/uncertainty | T | T/F | Conteos masivos | **Principal** | Comparación | Sensibilidad a escala, intervalos y clasificador documentados. |
| **P1** | Delay embedding general, delay/FNN | T | T/F | Vecinos/distancias | **Principal** | PECUZAL/unified | Retardos, proyección y Theiler explícitos; sin claim automático de embedding F. |
| **P1** | Entropías/complejidad | T | T | Bandt--Pompe C/Numba ya local; kernels calientes selectos restantes | Subconjunto estable | **Principal para catálogo amplio** | Bandt--Pompe conserva histograma, empates, retardo, base y muestra; faltan sesgo/CI y catálogo adicional. |
| **P1** | Dimensiones fractales | T/I | T/F | Correlation sums/boxing | Glue y estimadores simples | **Principal para catálogo amplio** | Curvas/rangos/CI; Kaplan–Yorke hereda estado del Lyapunov. |
| **P1** | Lyapunov entero, CLV, SALI/GALI/LDI | I | F | No justificado para la pasada CLV backward en el benchmark fechado: la fase Numba fue 1.60 % del tiempo end-to-end grande y su techo ideal de sustitución fue 1.016x; reevaluar sólo con un cuello productivo medido | **Principal** para modelos de usuario, recursión Ginelli y Householder por lotes | Oráculo por lote | CLV y SALI/GALI/LDI enteros cumplen fixtures analíticos, Wolfram y paridad; todas las variantes F permanecen `research_required`. |
| **P1** | Continuación global/matching | I/T | F | Ensambles | Features y orquestación | Attractors.jl | Historia continua/reiniciada/ventana registrada y branch matching auditable. |
| **P2** | Órbitas periódicas | I | F/D | Residuos/Jacobianos | Prototipo | PeriodicOrbits.jl | Entero primero; F sólo tras definir periodicidad e historia y validar con paper primario. |
| **P2** | Redes de recurrencia | T | T | Graph kernels si se justifica | Métricas pequeñas | **Primera opción** | Definición de red, conectividad y tratamiento diagonal explícitos. |
| **P2** | Escape, retorno, transporte | T/I | T/F | Ensambles | **Principal** | Comparación | Regiones, censoring, proyección e historia declarados. |
| **P2** | Hamiltonianos/variedades | I | F | Según demanda | Prototipo | Ecosistema Julia | No llamar “fraccionario” a una sustitución formal sin estructura publicada. |

## Superposición por definición fraccionaria

El catálogo upstream es principalmente entero. La portabilidad real se decide con
esta segunda capa:

| Familia de derivada HAFO | Métodos numéricos que conviene priorizar | Funciones T reutilizables | Funciones que permanecen F |
|---|---|---|---|
| Caputo power-law | ABM/PECE y EFORK actuales; convolution quadrature y memoria rápida como expansión | RQA, entropías, delay, dimensiones, eventos, espectros de datos, cuencas muestreadas | estabilidad local, variacional/Lyapunov canónico, CLV/SALI/GALI/LDI, PO/Floquet, continuación de historial |
| Grünwald–Letnikov | convolución directa actual; recurrencias/FFT con tests de convergencia | Las mismas T | inicialización GL, estabilidad, variacional y ecuaciones en diferencias D no se heredan de Caputo |
| Riemann–Liouville | convolution quadrature y datos iniciales fraccionarios explícitos | Las mismas T después de obtener trayectoria válida | equilibrio/inicialización, sensibilidad y periodicidad dependen de la formulación RL |
| Hadamard / Caputo--Hadamard | CQ BDF como operador y ABM/PECE Caputo--Hadamard sobre malla uniforme en `log(t/a)` | Las mismas T si conservan los tiempos físicos no uniformes | terminal positivo, condición inicial logarítmica, estabilidad y variacional son específicos; faltan malla graduada y memoria rápida |
| Tempered Caputo/RL | Caputo por conjugación + ABM/PECE, RL--GL templada, CQ BDF1/BDF2 directa/FFT y Fast Method II FBDF1/GNGF2 con ventana exacta y cola recurrente ya ejecutables; SOE de solver, símbolo desplazado y correcciones como expansión | Las mismas T | criterios de estabilidad y variacional con tempering; ningún operador muestral es solver, GNGF2 no es BDF2 fraccionario y la conjugación no equivale en malla finita a `[delta/h+lambda]**q` |
| Caputo–Fabrizio | recurrencia exponencial dedicada; no pesos Caputo | Las mismas T | normalización, estabilidad y tangentes del kernel no singular |
| Atangana–Baleanu–Caputo | kernel Mittag–Leffler dedicado y fast history validada | Las mismas T | normalización, estabilidad, tangentes y PO específicos |
| Orden variable | GL congelado sobre muestras y solver Caputo tipo III L1 ya ejecutables; no reutilizar caches invariantes | Las mismas T con perfil temporal `q(t)` y definición registrados | tipo I/II, historia rápida, estabilidad, variacional, continuación y periodicidad requieren teoría específica |
| Orden distribuido | operador GL de doble cuadratura y solver Caputo L1 de medida discreta positiva ya ejecutables; kernel combinado `O(RN+N²d)` | Las mismas T con nodos, pesos, densidad/cuadratura y normalización registrados | errores temporal/de orden separados, malla graduada, CQ corregida, estabilidad, variacional y sensibilidad dependen de la distribución |
| Caputo multitérmino | fachada atómica con coeficientes no negativos, canonización exacta y reutilización del kernel L1 distribuido | Las mismas T con órdenes, coeficientes y grupos de términos registrados | no hay error de cuadratura en orden; malla graduada, correcciones de arranque, historia rápida, coeficientes firmados/matriciales y estabilidad siguen pendientes |
| Conformable/local | transformación ODE sólo cuando sea matemáticamente equivalente | I/T | Debe quedar en lane local separado: no aporta memoria hereditaria ni valida otras derivadas |

La diversificación actual ya cubre rutas ejecutables Caputo, GL/RL, Hadamard,
Caputo templada ABM y CQ templada RL/Caputo, Caputo variable tipo III, Caputo distribuida L1, Caputo
multitérmino L1, ABC y
conformable sin tratarlas como
kernels equivalentes. La
prioridad siguiente es: (1) memoria rápida con error control y CQ con correcciones
de arranque; (2) Caputo de orden variable tipo I/II y aceleración del L1 tipo III;
(3) CQ/malla graduada y estimadores separados para el solver distribuido L1 ya
ejecutable; (4) SOE rápida para ABC como método distinto; y (5)
Caputo--Fabrizio sólo bajo su compatibilidad inicial y crítica bibliográfica.

## ABI de interoperabilidad propuesta

Un adaptador externo sólo debe aceptar y devolver estructuras serializables:

```text
TrajectoryInput
  t: float64[n]
  x: float64[n, d]
  sampled_uniformly: bool
  projection: list[str]
  transient_interval: (t0, t1)
  system_kind: integer_flow | integer_map | fractional_continuous | fractional_difference
  derivative_definition: optional string
  order: optional scalar/vector/specification
  lower_terminal_and_prehistory: optional structured metadata
  memory_policy: optional full | finite_window | restart | fast_approximation
  solver_and_tolerances: structured metadata

AnalysisResult
  method: canonical string
  values: arrays/scalars
  parameters: structured metadata
  backend: hafo_c | hafo_numba | julia
  package_versions_and_hashes: structured metadata
  status: finite_numerical_diagnostic | validated_reference | experimental
  warnings: list[str]
```

Reglas de ejecución:

- una sola llamada Julia por trayectoria/lote o por análisis compuesto;
- arrays contiguos y sin copias innecesarias; no callback Julia en cada paso;
- entorno Julia fijado y precompilado; primer arranque medido aparte;
- timeout, cancelación y errores convertidos a excepciones HAFO con versiones;
- el resultado local y el Julia comparten esquema, pero nunca se declaran
  idénticos sin pruebas numéricas;
- el backend elegido forma parte del resultado reproducible.

## Fases de implementación accionables

### Fase A — paridad de datos y kernels P0

1. **Completado:** esquema `TrajectoryInput/PrehistorySpec/AnalysisResult`
   inmutable, serializable, con huella y tabla de capacidades por operador.
2. Consolidar recurrence/RQA local ya auto/cross/joint: salida sparse, ventanas,
   umbrales locales/multiumbral, medidas faltantes y fixtures Julia.
3. Consolidar cuenca entropy/uncertainty ya implementadas con barridos de escala,
   desplazamientos de caja y protocolos de convergencia de muestra.
4. Consolidar ensambles C y ABI Numba de RHS/mapas sin cruzar lenguajes por paso.
5. Construir fixtures pequeños deterministas y trayectorias congeladas; comparar
   resultados por definición, no sólo con tolerancia escalar global.

### Fase B — extensiones de análisis P1

1. Consolidar `genembed`, delay/FNN locales ya implementados y añadir extensión
   Julia opcional para PECUZAL/Cao/AFNN.
2. Crear extras independientes, por ejemplo `hafo[julia-analysis]`, con entorno
   fijado para ComplexityMeasures/RecurrenceAnalysis/DelayEmbeddings/
   FractalDimensions.
3. Consolidar la suma de correlación \(q=2\) ya disponible en Python/Numba/C:
   añadir benchmarks multiplataforma, sensibilidad e intervalos, y después
   evaluar algoritmos boxed/vecinos y selección trazable de escala.
4. **Completado para `q=1`:** CLV y SALI/GALI/LDI para entero, con rutas
   variacional/multiparticle o forward-QR/backward-Ginelli, NumPy/Numba y
   oráculos Wolfram. Mantener todas las variantes F/D en
   `research_required` hasta formular el cociclo de historia.
5. Ampliar las CQ BDF1/BDF2 ordinaria, Hadamard y templada ya implementadas y
   verificadas en casos finitos con correcciones de arranque y solvers por
   familia antes de barridos masivos. La historia rápida templada FBDF1/GNGF2
   ya dispone de tolerancia de compresión finita; aún no sustituye una historia
   rápida SOE dentro de un solver Caputo.

### Fase C — investigación P2

1. PeriodicOrbits entero con Julia como comparación externa.
2. Formular PO, Floquet, CLV y variedades para cada operador fraccionario en el
   espacio de historia antes de escribir el kernel.
3. Añadir recurrence networks, rare events, transport y herramientas Hamiltonianas
   sólo cuando existan casos de uso en Toolbox Chaos/HAFO.

## Evidencia mínima antes de declarar una capacidad

| Tipo de capacidad | Evidencia mínima |
|---|---|
| Kernel entero | caso analítico o referencia primaria, convergencia/tolerancias, comparación de backends y tests de dtype/contigüidad. |
| Solver fraccionario | definición e inicialización inequívocas, solución manufacturada o publicada, estudio de convergencia, memoria completa vs aproximada y sensibilidad de paso. |
| Métrica de trayectoria | fixture exacto pequeño, parámetros del estimador, muestreo/transitorio/proyección, comparación independiente y caso degenerado. |
| Cuenca/hiddenness | dominio, malla/muestra, prehistoria, clasificador, tasa no resuelta, convergencia y lenguaje “evidencia finita”; nunca prueba global por defecto. |
| Lyapunov/CLV/SALI/GALI/LDI F | ecuación variacional específica del operador, tratamiento de historia, validación primaria y sensibilidad a memoria; una trayectoria visual no basta. |
| Periodicidad F | definición de periodicidad del estado histórico, condición de frontera correcta, residuo, estabilidad y comparación publicada. |
| Benchmark | mismo problema/tolerancia/output, warm-up separado, hardware/software/hashes, varias repeticiones e intervalos; una cifra upstream no se transfiere a HAFO. |

## Fuentes oficiales y artículos primarios

### pynamicalsys

- [Documentación oficial estable](https://pynamicalsys.readthedocs.io/en/stable/)
- [API de sistemas continuos](https://pynamicalsys.readthedocs.io/en/stable/api/cds.html)
- [API de sistemas discretos](https://pynamicalsys.readthedocs.io/en/stable/api/dds.html)
- [Métricas de series temporales](https://pynamicalsys.readthedocs.io/en/stable/api/time_series_metrics.html)
- [Métricas de cuenca](https://pynamicalsys.readthedocs.io/en/stable/api/basin_metrics.html)
- [Changelog](https://pynamicalsys.readthedocs.io/en/stable/changelog.html)
- [Repositorio y licencia GPL-3.0](https://github.com/mrolims/pynamicalsys)
- [Artículo primario en *Chaos, Solitons & Fractals*](https://doi.org/10.1016/j.chaos.2025.117269)

### JuliaDynamics

- [Índice oficial de DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/dynamicalsystems/stable/contents/)
- [DynamicalSystems.jl: artículo JOSS](https://doi.org/10.21105/joss.00598)
- [ChaosTools.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/chaostools/stable/)
- [Attractors.jl](https://juliadynamics.github.io/Attractors.jl/dev/)
- [RecurrenceAnalysis.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/recurrenceanalysis/stable/)
- [ComplexityMeasures.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/)
- [ComplexityMeasures.jl: artículo primario](https://doi.org/10.1371/journal.pone.0324431)
- [DelayEmbeddings.jl](https://juliadynamics.github.io/DelayEmbeddings.jl/stable/)
- [PeriodicOrbits.jl](https://juliadynamics.github.io/PeriodicOrbits.jl/dev/)
- [Referencias primarias de los algoritmos de periodicidad](https://juliadynamics.github.io/PeriodicOrbits.jl/dev/references/)
- [FractalDimensions.jl](https://juliadynamics.github.io/FractalDimensions.jl/stable/)
- [Revisión primaria de estimadores de dimensión fractal](https://doi.org/10.1063/5.0160394)
- [JuliaCall/PythonCall: interoperabilidad oficial](https://juliapy.github.io/PythonCall.jl/stable/juliacall/)

## Conclusión

HAFO no debe convertirse en un wrapper de dos librerías enteras. Debe tomar su
catálogo como especificación de producto, poseer los kernels P0/P1 que determinan
reproducibilidad y rendimiento, y usar Julia como extensión de análisis de alto
nivel. La única transferencia realmente directa entre todos los backends es una
trayectoria y sus metadatos; los solucionadores, tangentes, estabilidad y
periodicidad fraccionarios deben permanecer ligados a la definición concreta de
derivada y a su estado de memoria.
