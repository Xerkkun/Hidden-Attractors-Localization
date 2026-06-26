# Manual Manifest

El archivo `docs/manual_manifest.yaml` actúa como la fuente canónica para sincronizar los metadatos y configuraciones documentales del proyecto a través de los siguientes manuales y objetivos:

* `USER_MANUAL.md`;
* `docs/reporte_unificado_chua_fraccionario.tex` (Reporte unificado LaTeX/PDF);
* Documentación web en el repositorio `Xerkkun/hidden-attractors` (Astro/Starlight).

## Reglas y Clarificaciones Importantes

1. **Sin resultados científicos nuevos**: El manifiesto no contiene ni genera nuevos resultados numéricos ni científicos. Es un archivo estrictamente documental y de consistencia de metadatos.
2. **Relación con las afirmaciones científicas**: El manifiesto no reemplaza a `THESIS_CLAIMS.md`. Este último sigue siendo la máxima y única autoridad científica para las afirmaciones defendibles.
3. **Rol limitado a metadatos**: El manifiesto solo fija metadatos documentales, versiones oficiales, rutas del CLI y rutas canónicas de exportación.
4. **Consistencia obligatoria**: Los manuales y reportes del proyecto no deben declarar estados, versiones ni CLI commands que sean incompatibles con los valores definidos en `manual_manifest.yaml`.
5. **Base científica de la ocultedad**: La ocultedad de los atractores sigue dependiendo estrictamente de la evidencia numérica bajo el contrato de vecindades de prueba en tiempo finito, no de figuras de simulación, funciones descriptivas (DF), diagramas de Nyquist o continuación numérica de órbitas.
6. **Conteo de pruebas unitarias**: El conteo oficial de pruebas unitarias y de integración debe tomarse del directorio `validation/freeze_audit/` (específicamente de los artefactos de freeze audit). No debe escribirse manualmente de manera rígida en múltiples lugares del código, salvo como un snapshot explícito que remita a dicha ruta oficial.

## Significado de los Campos del Manifiesto

| Campo | Significado |
| :--- | :--- |
| `manual_version` | Versión documental del paquete de manuales y guías del proyecto. |
| `package_version` | Versión de la librería de Python (`hidden_attractors`). |
| `public_cli` | Único comando público de CLI instalado (`hidden-attractors`). |
| `entry_point` | Entry point real en el archivo `pyproject.toml`. |
| `freeze_audit` | Metadatos y ruta de la fuente oficial del conteo de pruebas (`validation/freeze_audit/`). |
| `claims_source` | Nombre de la fuente oficial de claims científicos defendibles (`THESIS_CLAIMS.md`). |
| `canonical_figures` | Directorio raíz canónico para almacenar figuras promovidas (`library_figures/`). |
| `manual_targets` | Rutas y repositorios de los manuales que deben sincronizarse. |
| `scientific_scope` | Límites y definiciones del alcance científico permitido en los manuales. |
| `claim_status_summary` | Estado actualizado de cada candidato o familia de sistemas analizados. |
| `forbidden_public_claims` | Afirmaciones prohibidas y sobreafirmaciones científicas no permitidas en los manuales públicos. |
| `public_evidence_labels` | Etiquetas estandarizadas para clasificar la evidencia y resultados científicos. |
| `documentation_policy` | Reglas y políticas a seguir para la inclusión de comandos CLI legacy, figuras, claims y conteo de pruebas. |

