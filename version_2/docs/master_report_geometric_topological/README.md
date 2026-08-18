# Monografía y guía de estudio geométrico--topológicas

Esta carpeta contiene una fuente científica común y dos salidas editoriales:
una monografía digital y una guía autónoma para lectura e impresión.

## Ediciones

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
monografía digital añade trazabilidad reproducible y el atlas completo de
figuras promovidas.

La ruta de aprendizaje de la guía avanza por dependencias internas: EDO y
álgebra lineal, sistemas dinámicos, topología y geometría local, caos y cuencas,
cálculo fraccionario y dinámica con memoria; sólo después aplica los métodos de
localización. Las lecturas externas son ampliaciones opcionales y no son
necesarias para seguir las demostraciones, reproducir las tablas ni interpretar
los resultados del volumen impreso.

Las dos ediciones comparten los diagramas vectoriales que forman parte de una
explicación matemática y cinco figuras seleccionadas del piloto: la ruta
conceptual y cuatro resultados representativos de PLL, MAVPD y Wu. Las figuras
numéricas restantes se declaran individualmente como digitales; el atlas
extenso no se imprime.

## Compilación

Desde esta carpeta:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error reporte_maestro_atractores_ocultos.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error reporte_maestro_lectura_estudio.tex
```

La bibliografía se encuentra en `references.bib`, y el desarrollo está dividido
en `report_sections/`. Los archivos `integer_*.tex` y
`mavpd_chaos_screening_rows.tex` contienen tablas numéricas generadas que se
incluyen desde la fuente principal.

## Verificación algebraica cruzada

`wolfram/topologia_geometria_hidden_validation.wl` contiene la auditoría
simbólica y numérica de la extensión geométrico--topológica. Su salida registrada
está en `wolfram/topologia_geometria_hidden_validation.txt`. La contraparte
integrada se encuentra en
`version_2/validation/wolfram/cases/geometric_topological_engine.wl` y ejecuta
50 decisiones algebraicas PASS/FAIL.

La verificación algebraica no sustituye la campaña dinámica. El piloto
reproducible PLL--MAVPD--Wu se documenta en la sección de resultados
geométrico--topológicos: alcanza como máximo EV--TG3 para una ruta del PLL y
EV--TG2 para MAVPD y Wu. El seguimiento dinámico completo de borde, las
imágenes exteriores y el índice de Conley permanecen abiertos; una bisección
inicial no constituye por sí sola certificación topológica ni prueba global de
ocultedad.
