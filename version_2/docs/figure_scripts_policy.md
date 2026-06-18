# PolÃ­tica de Scripts de Figuras (Figure Scripts Policy)

Esta polÃ­tica establece las directrices para organizar, desarrollar y mantener los scripts que generan figuras y grÃ¡ficos en el ecosistema de `hidden_attractors`.

## 1. CentralizaciÃ³n en `version_2/figure_scripts/`

Todos los scripts ejecutables cuyo propÃ³sito principal sea la generaciÃ³n de figuras y visualizaciones interpretativas deben ubicarse exclusivamente en:
`version_2/figure_scripts/`

- **Nombres Descriptivos y Prefijados**: Los scripts deben llevar nombres que indiquen claramente su sistema de origen o su experimento para evitar colisiones (ej. `chua_arctan_wu2023_plot_basins.py`).
- **No Scripts Sueltos**: NingÃºn script de ploteo activo debe residir en la raÃ­z del paquete, en la carpeta de herramientas genÃ©ricas (`tools/`) o dentro de carpetas de experimentos/ejemplos concretos.

## 2. Contrato de ExportaciÃ³n de Figuras

Para garantizar la higiene y la predictibilidad del repositorio, los scripts de figuras deben adherirse a las siguientes reglas:

- **Evitar `savefig` Directo en LibrerÃ­as**: Las funciones internas de visualizaciÃ³n en la librerÃ­a (`hidden_attractors/plotting/`) no deben decidir rutas locales rÃ­gidas ni llamar directamente a `plt.savefig` sin un mecanismo de intercepciÃ³n.
- **Uso de Interceptores**: Se debe preferir el uso de funciones auxiliares del paquete que envuelvan las exportaciones (como `intercept_and_export_path`) para redirigir dinÃ¡micamente las figuras generadas a la ruta de salida correspondiente.
- **EstandarizaciÃ³n de Formatos**:
  - Las figuras destinadas a la documentaciÃ³n oficial o reportes cientÃ­ficos deben exportarse preferentemente en formato **PDF** (grÃ¡ficos vectoriales sin pÃ©rdida).
  - Las cuencas de atracciÃ³n complejas o representaciones de gran volumen pueden usar **PNG** optimizado para evitar archivos vectoriales excesivamente pesados.

## 3. Politica de Almacenamiento de Salidas Graficas

- **Figuras cientificas promovidas / canonicas**:
  - Las figuras que sirven como evidencia cientifica oficial deben colocarse en `version_2/library_figures/`.
  - Deben generarse mediante `hidden_attractors.plotting.export.export_figure`.
  - Deben tener manifiesto reproducible y no pueden sustituir las tablas JSON/CSV ni los reportes de validacion.
- **Assets decorativos o web-only**:
  - `version_2/docs/assets/` queda reservado para imagenes de documentacion, sitio o material explicativo.
  - Estas imagenes no son evidencia cientifica promovida por si mismas.
- **Evidencia numerica promovida**:
  - `version_2/validation/` contiene JSON, CSV, MD, TEX, manifiestos y reportes.
  - Puede referenciar figuras canonicas en `library_figures/`, pero no debe convertirse en una carpeta de salida grafica general.
- **Outputs exploratorios / regenerables**:
  - Los archivos temporales, diagnosticos interactivos o corridas locales deben guardarse en `version_2/outputs/` o `version_2/figures/`.
  - Estas rutas estan excluidas del repositorio mediante `.gitignore`.

## 4. AutomatizaciÃ³n e IntegraciÃ³n

- Las pruebas automÃ¡ticas de higiene (como `test_no_loose_figure_scripts.py`) escanean periÃ³dicamente el repositorio para asegurar que no existan scripts de figuras sueltos fuera de las rutas autorizadas.
- Cualquier adiciÃ³n de un nuevo script de figuras debe acompaÃ±arse de su correspondiente registro en `version_2/docs/figure_scripts_inventory.md`.

