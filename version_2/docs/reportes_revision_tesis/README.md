# Localización de atractores ocultos: monografía y guía de estudio

Los dos documentos comparten teoría, derivaciones, métodos, casos y resultados:

- `reporte_maestro_atractores_ocultos.pdf`: monografía científica con el atlas
  de figuras y los apéndices algebraicos.
- `reporte_maestro_lectura_estudio.pdf`: guía de estudio con ejercicios y
  lecturas vinculados al desarrollo matemático.

La fuente común es `reporte_maestro_atractores_ocultos.tex`. El archivo
`reporte_maestro_lectura_estudio.tex` activa la organización de la guía.
Las secciones se encuentran en `report_sections/`, la bibliografía en
`references.bib` y las figuras fuente en `figures/`, `integer_assets/` y
`report_diagrams/`. Las tablas numéricas generadas conservan sus fuentes.

La copia de tesis está en `08_tesis_doctorado/reportes_revision/`, respecto
a la raíz de Articulos. Su duplicado está en
`version_2/docs/reportes_revision_tesis/`, respecto a la raíz del software.
Las dos carpetas deben contener los mismos archivos fuente y los mismos PDF.

## Compilación

Desde esta carpeta, en PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\compilar_reportes.ps1
```

Se requieren LaTeX, BibTeX, `latexmk` y Perl. El compilador usa `build/` para
los intermedios y copia los dos PDF terminados a esta carpeta. Las rutas de
inclusión son relativas; las figuras PDF y PNG son insumos de compilación.
La bibliografía contiene 147 entradas y conserva las claves de cita previas.
El texto introductorio y el estado del arte se encuentran en
`report_sections/01_introduccion_antecedentes.tex`; incluyen el planteamiento,
la hipótesis, los objetivos y el alcance del primer capítulo de tesis.

La figura de reducción escalar de los equilibrios se regenera con
`python generar_figura_equilibrios.py` (NumPy, SciPy y Matplotlib).
El programa lee los parámetros de la fuente LaTeX, calcula las raíces por
Brent y guarda el PDF vectorial y los residuos numéricos en
`figures/fundamentos/`. No integra trayectorias ni sustituye datos de campañas.

## Convenciones de escritura

El texto científico expone el problema, los supuestos, el cálculo y su
interpretación. Las derivaciones incluyen sustituciones y pasos intermedios,
con los dominios necesarios para dividir, invertir una matriz o aplicar un
teorema. La voz académica usa la primera persona del plural cuando ayuda a
seguir el razonamiento. Los ejercicios de la guía forman parte del estudio
matemático; las notas editoriales y las listas de revisión se mantienen fuera
de los PDF.

Los parámetros, datos, contratos numéricos y figuras se modifican sólo cuando
existe una razón científica documentada. Una mejora de redacción no equivale
a una integración nueva. Las conclusiones se apoyan en los resultados y en
las condiciones del experimento correspondiente.

## Registros externos

El respaldo previo a esta edición de redacción, las comprobaciones
algebraicas y la validación de los PDF están en
`Codes/Respaldos_limpieza/Revision_redaccion_reportes_20261109/`.
`reportes_antes.zip` conserva los 359 archivos originales y permite recuperar
ambas copias, que eran idénticas al comenzar. El informe externo identifica
las modificaciones, la compilación y la revisión visual.

Los archivos de `auditoria/` corresponden a los cotejos matemáticos,
bibliográficos y de resultados anteriores. Sus fechas y hashes describen
aquellos cálculos; las comprobaciones visuales de PDF anteriores no autentican
automáticamente los PDF de esta edición. La trazabilidad completa de las
campañas se conserva en los expedientes de tesis y de software, con su archivo
recuperable documentado en `ARCHIVO_HISTORICO.md`.
