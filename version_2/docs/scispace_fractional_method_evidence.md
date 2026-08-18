# Evidencia SciSpace para el núcleo fraccionario

Fecha de consulta inicial: 2 de agosto de 2026. Estado: **bitácora bibliográfica
de diseño**; no es por sí sola validación del software, del método numérico ni de
la dinámica. Registra preguntas, identificadores y decisiones de lectura, pero
no es una exportación congelada del índice ni una correspondencia
ecuación-a-ecuación entre artículos y código. Las consultas se realizaron con
búsqueda semántica de SciSpace y se contrastaron con DOI/editoriales cuando el
índice no devolvió el registro final. Ningún resumen del índice autoriza copiar
un algoritmo o atribuirle a HAFO un orden de convergencia sin revisar la fuente
primaria y la fórmula implementada.

## Preguntas de búsqueda

Se formularon preguntas completas sobre: (1) la definición fundacional de la
modificación Caputo--Hadamard; (2) cuadratura de convolución para cálculo de
Hadamard; (3) transformación logarítmica a tiempo de Caputo; y (4) métodos
predictor--corrector, mallas graduadas y memoria rápida para ecuaciones
Caputo--Hadamard. SciSpace también generó la columna de conclusiones para los
trabajos de 2020, 2021 y 2022 citados abajo. La columna de metodología no estuvo
disponible para esos identificadores, por lo que no se usa como evidencia.

## Resultados que gobiernan la implementación

| Fuente | Resultado recuperado | Decisión en HAFO |
|---|---|---|
| Jarad, Abdeljawad y Baleanu (2012), [DOI 10.1186/1687-1847-2012-142](https://doi.org/10.1186/1687-1847-2012-142) | Introduce la modificación de tipo Caputo de la derivada de Hadamard. | Contrato `caputo_hadamard` separado de la derivada Hadamard--Riemann--Liouville cruda. |
| Gambo et al. (2014), [DOI 10.1186/1687-1847-2014-10](https://doi.org/10.1186/1687-1847-2014-10) | Extiende el teorema fundamental del cálculo fraccionario en el marco Caputo--Hadamard. | Referencia complementaria; no se confunde con el artículo fundador de 2012. |
| Gohar, Li y Li (2020), [DOI 10.1007/s00009-020-01605-4](https://doi.org/10.1007/s00009-020-01605-4) | Propone reglas rectangular, logarítmica L1 y predictor--corrector modificado, con análisis de estabilidad y error. | Candidato prioritario para una futura familia L1/logarítmica; no se declara implementado todavía. |
| Green, Liu y Yan (2021), [DOI 10.3390/math9212728](https://doi.org/10.3390/math9212728) | Predictor--corrector sobre mallas uniformes, graduadas y no uniformes en la coordenada logarítmica; la razón de gradación recupera órdenes óptimos bajo baja regularidad. | El solver actual implementa únicamente el caso uniforme `r=1`; las mallas graduadas quedan como extensión explícita. |
| Green y Yan, *Detailed Error Analysis for a Fractional Adams Method on Caputo--Hadamard Fractional Differential Equations* (2022), [DOI 10.3390/foundations2040057](https://doi.org/10.3390/foundations2040057) | Analiza el predictor--corrector en malla uniforme logarítmica y la dependencia del error respecto a la regularidad; no proporciona una promesa uniforme de orden 2 para todo `0<q<1`. | Es la fuente más próxima al ABM/PECE uniforme actual, pero las tasas publicadas solo se invocan bajo sus hipótesis y después de una validación específica. |
| Yin et al., prepublicación indexada por SciSpace como [arXiv:2311.06869](https://doi.org/10.48550/arXiv.2311.06869), versión editorial [DOI 10.1016/j.cnsns.2024.108221](https://doi.org/10.1016/j.cnsns.2024.108221) | Extiende CQ de Lubich a cálculo de Hadamard mediante mallas exponenciales y desarrolla correcciones BDF de alto orden para fuentes singulares. | HAFO implementa CQ BDF1/BDF2 sin correcciones de arranque; BDF3--6 y correcciones permanecen en la hoja de ruta. |
| Lubich (1986), [DOI 10.1137/0517050](https://doi.org/10.1137/0517050) | Establece CQ fraccionaria a partir de potencias de métodos lineales multipaso. | Base de los pesos BDF1/BDF2 directos y FFT en coordenada uniforme. |

## Métodos recientes identificados, aún no trasladados

- El esquema Caputo--Hadamard de alto orden de 2024,
  [DOI 10.4208/jcm.2312-m2023-0098](https://doi.org/10.4208/jcm.2312-m2023-0098),
  reporta análisis de estabilidad y una precisión de orden alto para un problema
  de difusión. Requiere una implementación y validación propias antes de entrar
  al núcleo ODE.
- Las aproximaciones por suma de exponenciales reducen memoria y coste en varias
  familias Caputo o de núcleo no singular. SciSpace recuperó ejemplos de 2018,
  2022 y 2024, pero no justifican copiar el algoritmo sin rederivar el núcleo
  específico de Hadamard/Caputo--Hadamard.
- Los resultados para Atangana--Baleanu--Caputo y Caputo--Fabrizio no son
  intercambiables: sus núcleos y condiciones iniciales son distintos.

Una segunda ronda de SciSpace amplió la hoja de ruta más allá de Caputo:

- Para Atangana--Baleanu--Caputo, Lee, Kim y Jang (2024),
  [DOI 10.3390/fractalfract8010065](https://doi.org/10.3390/fractalfract8010065),
  presentan un predictor--corrector con aproximación por suma de exponenciales,
  coste lineal declarado y análisis de error global de segundo orden. Es el
  fundamento primario del `abc_predictor_corrector` convencional ya ejecutable.
  HAFO fijó normalización explícita, compatibilidad inicial y un arranque propio;
  la variante SOE permanece como método planificado separado.
- Para Caputo--Fabrizio, Liu--Fan--Yin--Li (2020),
  [DOI 10.3934/math.2020117](https://doi.org/10.3934/math.2020117), proponen una
  fórmula de segundo orden evaluada en puntos medios y un algoritmo rápido con
  coste `O(N)` y memoria `O(1)`. La recurrencia HAFO procede de integrar el
  kernel con interpolación lineal por intervalos y **no** se declara idéntica a
  esa fórmula; el artículo se conserva como método relacionado hasta realizar
  una comparación ecuación por ecuación.
- Para Caputo templada, el método de diferencias/elementos finitos de 2020,
  [DOI 10.1007/s10915-020-01238-5](https://doi.org/10.1007/s10915-020-01238-5),
  ofrece evaluación rápida con análisis de estabilidad y convergencia. El
  operador actual de HAFO es RL templado por conjugación exponencial; no debe
  renombrarse como Caputo templada.
- Para orden variable, SciSpace recuperó una aproximación rápida L2--1-sigma
  con suma de exponenciales,
  [arXiv:2102.02960](https://doi.org/10.48550/arXiv.2102.02960), y trabajos que
  distinguen explícitamente Caputo tipo III o Riemann--Liouville. Esto refuerza
  la decisión de conservar el operador GL actual como convención de orden
  congelado en el tiempo de evaluación, sin equipararlo a esas definiciones.
- Para orden distribuido, el CQ de 2021,
  [DOI 10.3934/dcdsb.2020168](https://doi.org/10.3934/dcdsb.2020168), incorpora
  correcciones para la estructura no suave propia del problema y recupera el
  orden óptimo. Es la extensión natural del operador de doble cuadratura actual
  hacia un solver, después de fijar correcciones de arranque y error en orden;
  no valida los modos GL/RL crudos, pesos firmados o memoria truncada del
  operador muestral actual.

## Límite del índice

La búsqueda exacta del DOI `10.1016/j.aml.2021.107366` no devolvió en SciSpace
el registro solicitado. HAFO conserva esa referencia sólo tras contraste con la
fuente editorial/registro DOI. Esta ausencia se documenta para no presentar el
resultado del índice como exhaustivo.

## Ronda SciSpace del 3 de agosto de 2026: ABC y kernels no singulares

Para decidir si el contrato ABC debía seguir siendo solamente teórico se hizo
una nueva búsqueda con la pregunta completa:

> What peer-reviewed papers compare numerical methods for fractional
> derivatives with nonsingular kernels, especially Caputo-Fabrizio and
> Atangana-Baleanu, and discuss consistency, stability, or limitations?

SciSpace devolvió, entre otros, estos identificadores persistentes dentro del
índice: `1jmdyai06v` para Yadav--Pandey--Shukla, `pnqs2t3w36bz` para
Diethelm--Garrappa--Giusti--Stynes y `gkq70xfeh4e5` para el método de punto
medio sucesivo de 2025. Se solicitaron además las columnas `methods_used` y
`limitations` para esos tres registros. Las columnas son resúmenes generados
por el índice, por lo que se usaron para orientar la lectura y no como sustituto
de las ecuaciones del artículo.

La fuente editorial confirmó que Yadav et al.,
[DOI 10.1016/j.chaos.2018.11.009](https://doi.org/10.1016/j.chaos.2018.11.009),
construyen aproximaciones numéricas del operador ABC y restringen el análisis
presentado a `0 < alpha <= 1/2`. HAFO conserva esa restricción en
`abc_sampled_convolution`; no la extiende silenciosamente a `alpha > 1/2`.
SciSpace resumió correctamente que el trabajo presenta dos aproximaciones y
una aplicación de advección--difusión, pero el resumen no justifica atribuirle
un predictor--corrector ODE que el artículo no implementa.

La columna de limitaciones del registro crítico coincidió con la publicación
final de Diethelm et al.,
[DOI 10.1515/fca-2020-0032](https://doi.org/10.1515/fca-2020-0032): valor cero
forzado en el instante inicial, objeciones al teorema fundamental y reducción a
operadores clásicos bajo restricciones. Por eso la función ejecutable conserva
`implementation_status="research_required"`, expone la normalización y no se
presenta como sustituto de Caputo o RL.

Una consulta separada sobre métodos rápidos identificó y después se verificó
en la editorial el trabajo de Lee--Kim--Jang,
[DOI 10.3390/fractalfract8010065](https://doi.org/10.3390/fractalfract8010065).
Ese artículo presenta un predictor--corrector ABC convencional cuadrático y
una aceleración por suma de exponenciales con costo lineal declarado, análisis
de error global de orden dos y un ejemplo de Rössler. Se registra como candidato
normativo del PCM convencional: HAFO implementa sus ecuaciones (9)--(14), añade
un arranque implícito auditado y mantiene el SOE como ruta planificada distinta.
El operador muestral conserva pesos y alcance propios; no se presenta como PCM.

Las otras preguntas de esta ronda cubrieron: métodos estables Caputo/RL/GL/CQ
con memoria rápida; predictor--corrector y CQ Hadamard/Caputo--Hadamard; y
métodos de orden variable o distribuido. Sus resultados refuerzan el backlog de
SOE, mallas graduadas y solvers de orden variable, pero no autorizan a copiar
pesos entre definiciones con kernels distintos.

## Ronda SciSpace del 3 de agosto de 2026: prioridad posterior a ABC

Se formularon dos preguntas completas. La primera pidió artículos primarios con
solvers reproducibles y análisis de convergencia para Caputo--Fabrizio, Caputo
de orden variable, orden distribuido y Caputo templada. La segunda pidió
específicamente una recurrencia temporal estable para FDE ordinarias
Caputo--Fabrizio, incluyendo compatibilidad inicial y prueba manufacturada.

SciSpace localizó, entre otros:

- `44q916bzy4`, Li--Deng--Zhao, DOI editorial correcto
  [10.3934/dcdsb.2019026](https://doi.org/10.3934/dcdsb.2019026);
- `3d2cge4l79`, predictor--corrector templado rápido,
  [10.1007/s11075-016-0169-9](https://doi.org/10.1007/s11075-016-0169-9);
- `58dknsgi0c`, Cao--Wang--Xu para Caputo--Fabrizio,
  [10.1007/s42967-019-00043-8](https://doi.org/10.1007/s42967-019-00043-8);
- `hqcx7g1ci4j4`, Tavares--Almeida--Torres sobre las tres definiciones Caputo
  de orden variable,
  [10.1016/j.cnsns.2015.10.027](https://doi.org/10.1016/j.cnsns.2015.10.027);
- `133hbdyh9t`, la crítica de compatibilidad de kernels no singulares,
  [10.1515/fca-2020-0032](https://doi.org/10.1515/fca-2020-0032).

Se pidió la columna `methodology` para `58dknsgi0c`, `5g33kfb8n6` y
`2qqsok5bc0`; SciSpace devolvió `0/3` registros con datos. Esa ausencia queda
registrada y no se rellenó mediante inferencias del índice.

La ficha arXiv/SciSpace del trabajo templado mostró metadatos DOI conflictivos.
La editorial AIMS y el manuscrito primario
[arXiv:1501.00376](https://arxiv.org/abs/1501.00376) confirmaron
`10.3934/dcdsb.2019026`. La fuente demuestra la conjugación exponencial, la
formulación de Volterra, buena formulación y un predictor--corrector de Jacobi.
HAFO implementa una adaptación distinta y explícita:
`v=exp(lambda*(t-a))*x` se usa para derivar factores
`exp(-lambda*(t_n-t_j))`, aplicados por ABM/PECE directamente al estado físico.
Por eso el método se llama `tempered_caputo_abm_pece_transform` y no se atribuye
como algoritmo de Jacobi. La reducción `lambda=0`, una solución manufacturada,
la paridad C--Python y las regresiones contra overflow artificial del estado
transformado son las comprobaciones locales.

La misma ronda confirmó por qué Caputo--Fabrizio no debe preceder a la ruta
templada en un toolbox de atractores ocultos: una solución clásica regular
impone `f(a,x0)=0`, condición que una semilla arbitraria de un barrido de cuenca
normalmente incumple. Sus operadores y futuros solvers se conservan en el
catálogo, pero bajo una puerta científica explícita. Para orden variable se
seleccionó y declaró expresamente Caputo tipo III: HAFO implementa ahora el L1
directo con `alpha(t_n)` sobre toda la historia y un corrector Picard identificado
como adaptación propia. Las rutas tipo I/II y la aceleración por bloques de Fang
siguen pendientes. Para orden distribuido debe separarse el error de cuadratura
en orden del error temporal.

El cierre de implementación como `implemented` y la conservación de la API como
`experimental` se apoyaron en las fuentes primarias localizadas y en pruebas
locales, no en el resumen de SciSpace. El oráculo independiente
`variable_order_caputo_type3_l1.wl` derivó los pesos por integración simbólica y
reconstruyó una recurrencia sin leer HAFO: residuo simbólico cero, diferencia
máxima Wolfram--HAFO `1.5543122344752192e-15` y diferencia de trayectoria
`2.220446049250313e-16`. Los errores finitos `2.0245592e-2` (operador) y
`1.4766213e-2` (recurrencia) se registran como diagnósticos, no como prueba de
convergencia, estabilidad, caos, atracción u ocultedad.

## Ronda SciSpace: solver Caputo de orden distribuido

Para decidir el siguiente ejecutor se formuló literalmente la pregunta:

> Which peer-reviewed papers derive and analyze numerical time-stepping
> methods, especially L1 or quadrature-in-order schemes, for nonlinear Caputo
> distributed-order differential equations with initial-value conditions?

Entre los diez resultados, SciSpace localizó:

- `s8anc31w2n4z`, Huang--Chen--An,
  [DOI 10.1007/s10915-021-01726-2](https://doi.org/10.1007/s10915-021-01726-2),
  con L1 sobre malla graduada para una ecuación de difusión de orden
  distribuido;
- `3ued1yt7yp`, Durastante,
  [DOI 10.1007/s10092-019-0329-0](https://doi.org/10.1007/s10092-019-0329-0),
  con cuadratura de Gauss adaptativa y fórmulas de integración producto;
- `2kx8ah4m4a`, Derakhshan--Rezaei--Marasi,
  [DOI 10.1016/j.matcom.2023.07.017](https://doi.org/10.1016/j.matcom.2023.07.017),
  con Gauss--Legendre en el orden y L2-1 temporal.

Se solicitó la columna `methodology` para esos tres identificadores; SciSpace
devolvió `0/3` registros con datos. No se rellenó esa columna mediante
inferencias. Los resúmenes orientaron la búsqueda, pero la decisión matemática
se contrastó con las fuentes primarias de Caputo, Diethelm--Ford,
Hu--Liu--Anh--Turner y Lin--Xu.

La consecuencia para HAFO fue implementar una ruta específica
`caputo_distributed_order` en lugar de promover el operador GL genérico. La
cuadratura explícita produce un problema multitérmino; cada término usa L1 y se
agrega en un kernel único antes de la marcha temporal. El Picard vectorial, la
optimización `O(R*N + N²*d)` y el átomo `alpha=1` como backward Euler son
adaptaciones declaradas. Los teoremas publicados para problemas lineales de
difusión no se extienden a Chua ni a ocultedad. Los errores temporal y de
cuadratura permanecen separados y no se estiman automáticamente.

El cierre local como `implemented`, con API todavía `experimental`, se completó
con 86 casos directos y 6 pruebas Wolfram. El oráculo deriva los pesos mediante
`Integrate`, no lee HAFO y resuelve
una recurrencia lineal afín: residuo simbólico, residuo discreto y error afín
`0`; diferencia máxima Wolfram--HAFO `1.5543122344752192e-15` frente a tolerancia
`8e-12`. Es consistencia finita de implementación, no una extensión de los
teoremas lineales a sistemas caóticos.

## Consulta específica: Caputo multitérmino como medida atómica

Para la fachada multitérmino se ejecutaron dos preguntas nuevas:

1. “Which peer-reviewed papers define multi-term Caputo fractional differential
   equations and derive stable convergent numerical methods, especially L1 or
   predictor-corrector schemes, with explicit coefficients, initial conditions,
   consistency, and convergence orders?”
2. “Which peer-reviewed papers show that a finite sum of Caputo derivatives is
   a distributed-order derivative with a discrete atomic measure, and what
   numerical analysis distinguishes multi-term Caputo equations from genuinely
   continuous distributed-order equations?”

En la primera búsqueda se conservaron `3tp7pod1yv` (Ren--Sun) y `30zlukzu`;
SciSpace devolvió la columna `methods_used` para `2/2`. En la segunda se
conservaron `1o5cvoa44t` (Kochubei), `3hf80rfehx` y `3ued1yt7yp`; la misma
columna quedó disponible para `3/3`. Los resultados sustentan dos decisiones:
una suma finita es una medida de órdenes atómica, mientras una densidad continua
requiere cuadratura y un error de orden separado; L1 se aplica término a término
antes de combinar el kernel.

HAFO declara como adaptaciones propias la canonización exacta, `math.fsum`, la
política de ceros, los metadatos de grupos y la fachada estructurada. No se
heredan teoremas de difusión lineal para sistemas dinámicos no lineales.

Para trazabilidad reproducible, cualquier promoción desde esta bitácora a
referencia normativa debe registrar al menos: función pública, definición,
ecuación de discretización, DOI y ubicación de la ecuación en la fuente,
desviaciones de HAFO y artefacto de validación. La presencia de un DOI en esta
bitácora solo demuestra que la fuente fue localizada y contrastada, no que su
algoritmo ya esté implementado o validado.

## Ronda SciSpace del 3 de agosto de 2026: dimensión de correlación q=2

Para fijar el contrato de la suma de correlación y separar la definición del
problema de elegir una escala se formuló literalmente la pregunta:

> Which peer-reviewed papers derive or critically analyze the
> Grassberger-Procaccia correlation-sum estimator of correlation dimension for
> finite time series, including temporal-correlation exclusion with a Theiler
> window and explicit selection of a scaling region?

Entre los diez resultados se retuvieron para revisión metodológica:

- `2i5yczlsfi`, Wang y Chen (2001), DOI
  `10.1177/107754630100700705`;
- `4fff2yrm1l`, Frank, Keller y Sporer (1996), DOI
  `10.1103/PhysRevE.53.5831`;
- `kay9cvojrdw6`, Deshmukh et al. (2021), DOI `10.1063/5.0069365`.

Se solicitó la columna `methodology` para los tres identificadores y SciSpace
devolvió `0/3`; esa ausencia no se completó mediante inferencias ni se presentó
como extracción del método.

Las decisiones matemáticas se contrastaron con las fuentes primarias:

- Grassberger--Procaccia,
  [DOI 10.1016/0167-2789(83)90298-1](https://doi.org/10.1016/0167-2789(83)90298-1),
  ancla la suma directa de correlación para \(q=2\);
- Theiler,
  [DOI 10.1103/PhysRevA.34.2427](https://doi.org/10.1103/PhysRevA.34.2427),
  motiva excluir pares temporalmente próximos mediante una ventana explícita;
- Deshmukh--Bradley--Garland--Meiss,
  [DOI 10.1063/5.0069365](https://doi.org/10.1063/5.0069365), analiza la
  extracción y caracterización de regiones de escala.

HAFO implementa una elección deliberadamente más conservadora que una
selección automatizada: devuelve la curva completa y exige al llamador un
`fit_radius_range` inclusivo; no infiere una región de escala. El resumen
indexado en SciSpace orientó esta decisión bibliográfica, pero no valida la
implementación.

La comprobación numérica es independiente de SciSpace. Un caso Wolfram con seis
puntos exactos verifica el conteo estricto, la normalización con ventana de
Theiler y el ajuste log--log declarado; la peor diferencia Python--Wolfram
retenida es `3.552713678800501e-15` frente a `5e-13`. Es evidencia finita de
consistencia de implementación, no prueba de una región de escala válida,
consistencia estadística del estimador, dimensión fractal, caos, atracción u
ocultedad.

## Ronda SciSpace del 3 de agosto de 2026: SALI y GALI enteros

Para localizar las definiciones primarias, los algoritmos explícitos y sus
límites se formularon tres preguntas literales:

1. “Which peer-reviewed papers introduce the Smaller Alignment Index (SALI)
   and Generalized Alignment Index (GALI) for distinguishing regular and
   chaotic dynamics, and give explicit numerical algorithms, normalization
   rules, asymptotic behavior, and benchmark dynamical systems for continuous
   flows and discrete maps?”
2. “Which peer-reviewed papers analyze SALI or GALI for dissipative nonlinear
   flows and maps, state their relation to Lyapunov exponents, and describe
   numerically stable computation with repeated normalization, QR, singular
   values, or logarithmic volumes?”
3. “How does the 2025 peer-reviewed multi-particle method compute GALI without
   variational equations, and what deviation size, renormalization interval,
   precision, error estimates, and benchmark systems are recommended for a
   reliable implementation?”

La primera consulta retuvo `560r32rzdl`, `h3e4j7k9sf83`, `rj04gyes8gwz` y
`1rppdfqkxr`; SciSpace devolvió la columna `methods_used` para `4/4`. La
segunda retuvo `4j7btyt6zp` y `jh9zyvd2k6y0`, con `methods_used` para `2/2`.
La tercera retuvo `oubc9gre7wfb` y `h3e4j7k9sf83`, con `methods_used` para
`2/2`.

Los resultados orientaron la selección, que después se contrastó con los
artículos primarios. HAFO conserva las definiciones de Skokos y colaboradores,
el volumen por producto SVD y la normalización independiente. El trabajo de
Manda--Hillebrand--Skokos de 2025 sustenta una ruta multiparticle separada y
sus parámetros quedan expuestos, no fijados como universales. El artículo de
Ma--Long--Zhu recuerda que un indicador de alineamiento no debe interpretarse
en sistemas disipativos copiando sin más las reglas de sistemas conservativos.

Una búsqueda primaria complementaria localizó además el artículo publicado en
2026 de Rolim Sales--Leonel--Antonopoulos,
[DOI 10.1016/j.chaos.2026.117884](https://doi.org/10.1016/j.chaos.2026.117884),
que deriva mediante SVD las tasas de LDI/GALI para sistemas continuos y
discretos y la tasa SALI de mapas considerando siempre los dos mayores
exponentes, incluso cuando el segundo es negativo. Esa fuente reciente
respalda la rama de mapas y el cálculo SVD, pero tampoco convierte un valor o
umbral finito en una clasificación automática.

La futura variante fraccionaria no se promovió. Ninguno de estos resultados
formula por sí mismo la perturbación completa en el espacio de historia de una
derivada no local; reutilizar el sistema tangente ODE y cambiar únicamente el
integrador sería una extrapolación no demostrada.

## Ronda SciSpace del 3 de agosto de 2026: vectores covariantes de Lyapunov

Para el siguiente bloque del catálogo se formularon preguntas completas sobre
el algoritmo Ginelli, las alternativas Wolfe--Samelson/Kuptsov--Parlitz, la
estabilidad numérica y la posible extensión a sistemas fraccionarios no
locales. Entre los resultados retenidos están:

- `26md6352yj`, Ginelli et al. (2007), DOI
  `10.1103/PhysRevLett.99.130601`, algoritmo CLV para mapas y flujos;
- `uu14cijq8j`, Kuptsov--Parlitz (2012), DOI
  `10.1007/s00332-012-9126-5`, teoría, algoritmos y vectores adjuntos;
- `3kb6l1sifw`, Ginelli et al. (2013), DOI
  `10.1088/1751-8113/46/25/254005`, revisión del algoritmo dinámico;
- `kjplihmcd925`, Froyland et al., comparación de Ginelli,
  Wolfe--Samelson, SVD y dicotomías; el artículo publicado corresponde al DOI
  `10.1016/j.physd.2012.12.005`;
- `3ivkqfdt9h`, Noethen (2021), DOI
  `10.3934/jcd.2021014`, generalización de Ginelli a espacios de Hilbert.

SciSpace devolvió la columna `conclusions` para `5/5` trabajos y
`methodology` para `0/5`. La columna ausente no se reconstruyó por
inferencia. Las decisiones algorítmicas se contrastan con los artículos
primarios: QR hacia adelante, factores triangulares almacenados, recursión
hacia atrás, normalización por columna y covariancia bajo el propagador
tangente.

La búsqueda fraccionaria literal preguntó si existe una formulación primaria
de CLV para Caputo, Riemann--Liouville, Grünwald--Letnikov,
Caputo--Fabrizio o Atangana--Baleanu en espacio de historia, en vez de reutilizar
el Jacobiano ODE instantáneo. Los diez resultados devueltos trataron estabilidad
de Lyapunov, ecuaciones fraccionarias en espacios de Hilbert o aplicaciones
heurísticas, pero no proporcionaron directamente el contrato CLV histórico
solicitado. Esto no demuestra inexistencia bibliográfica; sí impide promover
una fachada fraccionaria con la evidencia localizada. HAFO implementará primero
CLV `q=1` y conservará cada variante no local como
`research_required` hasta disponer de cociclo tangente, norma y
renormalización específicos del operador y su memoria.

Una segunda consulta SciSpace preguntó literalmente por criterios adaptativos
o monitorizados para detener los transitorios forward/backward de Ginelli y por
la pérdida de precisión en subespacios centrales. Retuvo la tesis de
convergencia de Noethen (`n133623uyu`), Kuptsov--Parlitz (`uu14cijq8j`) y la
comparación de Froyland et al. (`kjplihmcd925`). La columna `conclusions` estuvo
disponible para `3/3` y `methodology` para `0/3`; de nuevo, el contenido ausente
no se reconstruyó. SciSpace no devolvió en esa consulta el trabajo más reciente
de du Plessis, Hillebrand y Skokos, localizado después en la página primaria de
la revista (DOI `10.1016/j.physd.2026.135237`). Ese artículo de 2026 propone
vigilar la convergencia de los transitorios y advierte que una evolución
backward excesivamente larga puede degradar subespacios centrales degenerados.
Por ello, la primera API HAFO conserva horizontes explícitos, registra que no
aplica todavía un criterio automático de parada y no interpreta columnas
individuales como únicas cuando los exponentes son repetidos o casi
degenerados.

### Consulta complementaria sobre convergencia y brecha espectral

Una consulta SciSpace complementaria retuvo de nuevo la comparación de
Froyland--Hüls--Morriss (DOI `10.1016/j.physd.2012.12.005`; el resultado
duplicado como arXiv `1204.0871` corresponde al mismo trabajo),
Kuptsov--Parlitz (DOI `10.1007/s00332-012-9126-5`) y el análisis de
convergencia mediante proyectores de Noethen, *Computing Covariant Lyapunov
Vectors in Hilbert Spaces* (DOI `10.3934/jcd.2021014`). Esta última línea liga
la velocidad de convergencia a la separación espectral: una brecha pequeña
justifica conservar un warning y diagnosticar subespacios casi degenerados,
pero no proporciona un criterio automático universal de parada. En
subespacios centrales o degenerados deben compararse subespacios/proyectores y
tratar con cautela las columnas individuales. Esta conclusión se aplica aquí
al cociclo entero `q=1`; no se cita como teoría CLV fraccionaria.

## Ronda SciSpace del 3 de agosto de 2026: CQ RL/Caputo templada

La consulta se formuló como una pregunta completa sobre artículos primarios que
derivan cuadratura de convolución o métodos lineales multipaso para derivadas
RL y Caputo templadas, conjugación exponencial, BDF1/BDF2, correcciones de
arranque, estabilidad y mejoras recientes hasta 2026. SciSpace retuvo, entre
otros:

- `50fevts7oq`, Chen--Deng, *Discretized fractional substantial calculus*,
  DOI `10.1051/m2an/2014037`;
- Guo--Zeng--Turner--Burrage--Karniadakis, *Efficient Multistep Methods for
  Tempered Fractional Calculus*, DOI `10.1137/18M1230153`;
- `fgad6jg64m`, Jin--Li--Zhou, correcciones de arranque BDF, DOI
  `10.1137/17M1118816`;
- `29n3nmss`, Qiao et al., BDF2 rápido para un problema integro-diferencial
  templado, DOI `10.1016/j.camwa.2022.08.014`.

La columna `conclusions` estuvo disponible para `3/3` artículos solicitados y
`methodology` para `0/3`; la metodología ausente no se reconstruyó. Los
resúmenes orientaron la selección, pero las fórmulas se contrastaron con las
fuentes primarias. Chen--Deng y Guo et al. sustentan

$$
\omega_k^{(q,\lambda)}=e^{-\lambda kh}\omega_k^{(q,0)},
\qquad
\delta_p(e^{-\lambda h}\zeta)^q.
$$

La revisión reveló una distinción decisiva. La Caputo templada conjugada exige
el ancla `x-exp(-lambda*(t-a))*x(a)` para `0<q<=1`; no autoriza sustituirla por
`x-x(a)`. Tampoco deben identificarse la conjugación discreta
`delta(exp(-lambda*h)*z)**q`, la CQ del símbolo
`(delta(z)/h+lambda)**q`, la SCQ con desplazamiento temporal `theta` ni la
variante normalizada que resta `lambda**q*x`. HAFO implementa sólo la primera y
registra las restantes como contratos separados o pendientes.

Los resultados 2025--2026 localizados para L1 corregido, mallas no uniformes y
evaluación rápida fortalecen el backlog de singularidad inicial e historia
comprimida, pero no reemplazan el núcleo BDF verificado. En particular, un
método antiguo sigue retenido cuando su álgebra continúa siendo la base del
algoritmo actual. El cierre como `implemented`, con API aún `experimental`, se
apoya además en reducción exacta para `lambda=0`, solapamiento BDF1--GL, refinamiento manufacturado,
paridad Python/Numba/FFT y un oráculo Wolfram independiente de 80 dígitos; no
en el resumen de SciSpace por sí solo.

### Seguimiento: Fast Method II templado

La misma búsqueda se amplió a algoritmos de historia rápida, aproximaciones
reales por trapecio, recurrencias estables y sumas de exponenciales. SciSpace
identificó como fuente decisiva a Guo--Zeng--Turner--Burrage--Karniadakis,
[DOI 10.1137/18M1230153](https://doi.org/10.1137/18M1230153). Las ecuaciones no
se reconstruyeron desde un resumen: se verificaron contra el manuscrito
primario completo, incluido su generador GNGF de segundo orden, la separación
local/historia y la recurrencia real.

Esa lectura cambió dos decisiones de implementación. Primero, el generador

$$
(1-z)^q\left(1+\frac q2(1-z)\right)
$$

se expone como GNGF2 y no como BDF2 fraccionario: sólo coincide con BDF2 en el
límite entero `q=1`. Segundo, para `0<q<1` la identidad beta/reflexión exige el
factor `-sin(pi*q)/pi` para reproducir los coeficientes FBDF1 negativos. Un
signo contrario en una ecuación intermedia del manuscrito es incompatible con
esa identidad, mientras que la definición posterior de su integrando sí usa
el signo negativo.

HAFO implementa así `tempered_fast_multistep_history` con ventana local exacta,
cola recurrente FBDF1/GNGF2 y calibración L1 sobre todos los pesos comprimidos
de la malla finita. El teorema de trapecio exponencial de Trefethen--Weideman,
[DOI 10.1137/130932132](https://doi.org/10.1137/130932132), fundamenta la
aproximación real, pero HAFO no inventa sus constantes de franja analítica:
reporta una verificación a posteriori de compresión, no una cota CQ/FDE.

El cierre como `implemented`, conservando la API `experimental`, depende de
pruebas directas Python/Numba, reducciones enteras, ejemplo Chua sólo como postprocesamiento y un oráculo
Wolfram independiente de 80 dígitos que pasó 13/13 aserciones. El detalle
matemático y los límites de evidencia están en
[Fast Recurrent Tempered Multistep History](tempered_fast_multistep_history.md).
