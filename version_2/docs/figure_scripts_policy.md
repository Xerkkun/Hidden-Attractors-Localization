# Política de Scripts de Figuras (Figure Scripts Policy)

Esta política establece las directrices para organizar, desarrollar y mantener los scripts que generan figuras y gráficos en el ecosistema de `hidden_attractors`.

## 1. Centralización en `version_2/figure_scripts/`

Todos los scripts ejecutables cuyo propósito principal sea la generación de figuras y visualizaciones interpretativas deben ubicarse exclusivamente en:
`version_2/figure_scripts/`

- **Nombres Descriptivos y Prefijados**: Los scripts deben llevar nombres que indiquen claramente su sistema de origen o su experimento para evitar colisiones (ej. `chua_arctan_wu2023_plot_basins.py`).
- **No Scripts Sueltos**: Ningún script de ploteo activo debe residir en la raíz del paquete, en la carpeta de herramientas genéricas (`tools/`) o dentro de carpetas de experimentos/ejemplos concretos.

## 2. Contrato de Exportación de Figuras

Para garantizar la higiene y la predictibilidad del repositorio, los scripts de figuras deben adherirse a las siguientes reglas:

- **Evitar `savefig` Directo en Librerías**: Las funciones internas de visualización en la librería (`hidden_attractors/plotting/`) no deben decidir rutas locales rígidas ni llamar directamente a `plt.savefig` sin un mecanismo de intercepción.
- **Uso de Interceptores**: Se debe preferir el uso de funciones auxiliares del paquete que envuelvan las exportaciones (como `intercept_and_export_path`) para redirigir dinámicamente las figuras generadas a la ruta de salida correspondiente.
- **Estandarización de Formatos**:
  - Las figuras destinadas a la documentación oficial o reportes científicos deben exportarse preferentemente en formato **PDF** (gráficos vectoriales sin pérdida).
  - Las cuencas de atracción complejas o representaciones de gran volumen pueden usar **PNG** optimizado para evitar archivos vectoriales excesivamente pesados.

## 3. Política de Almacenamiento de Salidas gráficas

- **Figuras Promovidas / Canónicas**:
  - Las figuras que sirven como evidencia científica oficial o ilustran la documentación del repositorio deben colocarse en `version_2/docs/assets/` o en subcarpetas específicas bajo `version_2/validation/` (ej. `validation/reference_cases/`).
  - Estas figuras están bajo seguimiento de Git (no se ignoran) y deben mantenerse actualizadas.
- **Figuras Locales / Regenerables**:
  - Los archivos temporales, gráficos de diagnóstico interactivos o figuras generadas en ejecuciones locales/de prueba deben guardarse en `version_2/outputs/` o `version_2/figures/`.
  - Estas rutas están explícitamente excluidas del repositorio mediante `.gitignore` para mantener la higiene y evitar inflar el tamaño de la base de datos de Git.

## 4. Automatización e Integración

- Las pruebas automáticas de higiene (como `test_no_loose_figure_scripts.py`) escanean periódicamente el repositorio para asegurar que no existan scripts de figuras sueltos fuera de las rutas autorizadas.
- Cualquier adición de un nuevo script de figuras debe acompañarse de su correspondiente registro en `version_2/docs/figure_scripts_inventory.md`.
