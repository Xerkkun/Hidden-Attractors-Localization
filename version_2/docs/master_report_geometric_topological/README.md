# Monografía y guía de estudio geométrico--topológicas

Esta carpeta conserva las fuentes y los apoyos de una edición anterior,
necesarios para la regresión algebraica y el generador de figuras. Las dos
ediciones vigentes para lectura, revisión de tesis y compilación están en
[reportes_revision_tesis](../reportes_revision_tesis/README.md).

Los PDF anteriores de 308 y 236 páginas se retiraron del árbol activo después
de verificar su respaldo íntegro en
`Codes/Respaldos_limpieza/Tesis_HAFO_rastreados_20261109/archivos_rastreados_retirados.zip`.
El manifiesto de limpieza conserva rutas y SHA-256; los PDF actuales tienen
338 y 274 páginas. Las fuentes y los artefactos científicos de esta carpeta
se mantienen con sus bytes anteriores.

## Ediciones históricas archivadas

- `reporte_maestro_atractores_ocultos.pdf`: monografía científica digital.
- `reporte_maestro_lectura_estudio.pdf`: guía secuencial destinada a estudio e
  impresión.

Ambas ediciones proceden de `reporte_maestro_atractores_ocultos.tex`. La segunda
activa el interruptor editorial mediante
`reporte_maestro_lectura_estudio.tex`.

Ambas salidas comparten íntegramente el núcleo científico: hipótesis, teoría,
derivaciones algebraicas, métodos propuestos, contratos numéricos, casos,
tablas, resultados positivos y negativos, discusión y problemas abiertos. La
guía añade objetivos, prerrequisitos, ejercicios, pistas y autocontroles. La
monografía digital añade apéndices de identidades algebraicas y el atlas
completo de resultados gráficos.

El núcleo común incluye además una teoría explícita de puntos perpetuos de
orden entero y su extensión metodológica a Caputo (FPP--A/FPP--H), junto con
predictibilidad, shadowing, exponentes de Lyapunov de tiempo finito, caos
transitorio, sillas caóticas, crisis y una ruta de Conley. Las afirmaciones
publicadas se separan de las adaptaciones del proyecto y de los programas de
prueba todavía abiertos.

La ruta de aprendizaje de la guía avanza por dependencias internas: EDO y
álgebra lineal, sistemas dinámicos, topología y geometría local, caos y cuencas,
cálculo fraccionario y dinámica con memoria; sólo después aplica los métodos de
localización. Las lecturas externas son ampliaciones opcionales y no son
necesarias para seguir las demostraciones, reproducir las tablas ni interpretar
los resultados del volumen impreso.

Los diagramas vectoriales explican las construcciones matemáticas, incluido
un ciclo oculto con cuencas determinadas exactamente. La guía incluye además
cinco figuras seleccionadas del piloto: la ruta conceptual y cuatro
resultados representativos de PLL, MAVPD y Wu. Las figuras
numéricas restantes se declaran individualmente como digitales; el atlas
extenso no se imprime.

## Compilación de la edición histórica

Para la edición vigente, seguir el README de `../reportes_revision_tesis/`.
Sólo si se necesita reconstruir la edición histórica, desde esta carpeta:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error reporte_maestro_atractores_ocultos.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error reporte_maestro_lectura_estudio.tex
```

La bibliografía se encuentra en `references.bib`, y el desarrollo está dividido
en `report_sections/`. Los archivos `integer_*.tex` y
`mavpd_chaos_screening_rows.tex` contienen tablas numéricas generadas que se
incluyen desde la fuente principal.

## Verificación algebraica cruzada

`wolfram/topologia_geometria_hidden_validation.wl` contiene la comprobación
simbólica y numérica de la extensión geométrico--topológica. Su salida registrada
está en `wolfram/topologia_geometria_hidden_validation.txt`. La contraparte
integrada se encuentra en
`version_2/validation/wolfram/cases/geometric_topological_engine.wl` y ejecuta
57 decisiones algebraicas PASS/FAIL. Incluye por separado la covariancia del
desarrollo de arranque transportado, el defecto de recalcularlo desde un campo
transformado no linealmente y las identidades cuadráticas de acotación de Lorenz.

La verificación algebraica no sustituye la campaña dinámica. El piloto
reproducible PLL--MAVPD--Wu se documenta en la sección de resultados
geométrico--topológicos: alcanza como máximo EV--TG3 para una ruta del PLL y
EV--TG2 para MAVPD y Wu. El seguimiento dinámico completo de borde, las
imágenes exteriores y el índice de Conley permanecen abiertos; una bisección
inicial no constituye por sí sola certificación topológica ni prueba global de
ocultedad.
