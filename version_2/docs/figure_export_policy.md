# Política de Exportación de Figuras y Reproducibilidad

Este documento define la política oficial para la generación, exportación y almacenamiento de figuras en el repositorio. El objetivo principal es garantizar la higiene del código, la reproducibilidad de los resultados científicos y la protección contra la contaminación de datos promovidos por ejecuciones de pruebas.

## 1. Principio General

Toda figura utilizada como evidencia promovida en validaciones o informes oficiales debe exportarse de manera única a través de la API centralizada:

```python
hidden_attractors.plotting.export.export_figure
```

Esto garantiza que las figuras se guarden en la ubicación canónica estructurada bajo:

```
version_2/library_figures/
```

y que se generen automáticamente los archivos de metadatos JSON y los manifiestos correspondientes de forma consistente.

---

## 1.1. Reglas de Rutas y Canonicidad

Para garantizar la portabilidad y limpieza de la evidencia de validación:
* **Ubicación Canónica**: Todas las figuras promovidas deben referenciarse utilizando rutas relativas al repositorio dentro de la carpeta `version_2/library_figures/`.
* **Prohibición de Rutas Personales**: Está estrictamente prohibido incluir rutas absolutas locales personales (por ejemplo, `C:/[Usuarios]/...`, `/[Usuarios]/...`, `Desktop/[Codigos]/...`) en el código, pruebas, manifiestos o archivos de validación promovidos.

* **Manejo de Rutas Externas Legacy**: Rutas relativas que apunten fuera de la carpeta canónica (como `.._.._.._/[Nombre_Directorio]/Figs/`) se consideran rutas no canónicas. Solo se permiten dentro de campos o secciones marcados explícitamente como legacy (por ejemplo, el campo `legacy_external_figures_not_promoted` en archivos JSON, o bajo un encabezado Markdown que contenga la palabra "Legacy"). No forman parte de la evidencia científica promovida y se conservan únicamente por procedencia histórica.


---


## 2. Diferencia entre Figura Promovida y Figura Legacy

| Tipo de Figura | Descripción | Metadatos y Manifiesto | Ubicación Canónica | Permitido en Evidencia Promovida |
| --- | --- | --- | --- | --- |
| **Figura Promovida** | Aparece en reportes, manifiestos o validación oficial; es parte del cuerpo de evidencia numérica del proyecto. | Sí (JSON/CSV) | `library_figures/` | **Sí** |
| **Figura Legacy / Exploratoria** | Proviene de scripts históricos o de análisis transitorios. | No | Fuera de `library_figures/` | **No** |

---

## 3. Regla de `savefig`

* Queda estrictamente prohibido llamar a `savefig` directamente en flujos promovidos de la librería (`hidden_attractors/plotting/`, `hidden_attractors/workflows/`, `examples/`, y `tools/cli/`).
* Únicamente `hidden_attractors/plotting/export.py` tiene permitido invocar directamente el método `fig.savefig` de matplotlib.
* Cualquier otro script que requiera guardar figuras debe utilizar `export_figure` o `intercept_and_export_path`.
* Las excepciones legacy deben estar documentadas en la tabla a continuación.

### Inventario de Excepciones Legacy

| Archivo | Motivo | Estado | Permitido en evidencia promovida |
| --- | --- | --- | --- |
| `hidden_attractors/workflows/danca_abm_sphere_controls.py` | script histórico/exploratorio | legacy | no |
| `hidden_attractors/workflows/fractional_report_run.py` | reporte transitorio | legacy | no |
| `hidden_attractors/workflows/refined_basin.py` | exploración de cuencas | legacy | no |
| `hidden_attractors/plotting/generate_publication_figures.py` | generador histórico de figuras | legacy | no |
| `hidden_attractors/plotting/matignon.py` | diagnóstico gráfico auxiliar | legacy | no |
| `hidden_attractors/plotting/dynamics.py` | diagnóstico gráfico auxiliar | legacy | no |
| `hidden_attractors/plotting/basin.py` | diagnóstico gráfico auxiliar | legacy | no |
| `hidden_attractors/plotting/overlays.py` | diagnóstico gráfico auxiliar | legacy | no |

---

## 4. Relación con Reproducibilidad y Ejecución de Pruebas

Para evitar que las pruebas de integración o unitarias contaminen la carpeta de figuras oficial (`version_2/library_figures`):
1. Ninguna prueba debe escribir en el directorio real de figuras.
2. Todas las pruebas que invoquen funciones de ploteo o exportación deben redirigir temporalmente las rutas utilizando `tmp_path` y `monkeypatch` en pytest.
3. Se verificará en las pruebas de higiene que no ocurran escrituras no autorizadas.

---

## 5. Advertencia Científica

> [!WARNING]
> **Una figura no constituye una prueba de ocultedad.**
> Las figuras son meras representaciones visuales complementarias. La clasificación de ocultedad o auto-excitación de un atractor depende estrictamente de la evaluación numérica rigurosa de las vecindades de todos los equilibrios del sistema bajo el contrato numérico establecido, y no de análisis cualitativos de gráficos tridimensionales.
