# Notebooks / Notebooks (Cuadernos)

Notebook-style examples are useful for teaching, inspection, and report preparation. Keep them light enough to run without launching long numerical jobs.
Los ejemplos estilo notebook (cuaderno) son útiles para la enseñanza, la inspección y la preparación de reportes. Manténgalos lo suficientemente ligeros como para ejecutarse sin iniciar trabajos numéricos largos.

## Available Notebooks / Notebooks Disponibles

- `examples/notebooks/hidden_attractors_quickstart.ipynb`

This notebook covers:
Este notebook cubre:

- importing the package; / importar el paquete;
- checking Chua equilibria; / comprobar los equilibrios de Chua;
- loading final candidate records; / cargar los registros finales de candidatos;
- computing a lightweight trajectory diagnostic on synthetic sample data. / calcular un diagnóstico de trayectoria ligero sobre datos de muestra sintéticos.

For phase-space plots, bifurcation post-processing, and optional complexity metrics, mirror the documented workflow in [Dynamical Analysis](dynamical_analysis.md).
Para gráficos de espacio de fase, posprocesamiento de bifurcación y métricas de complejidad opcionales, refleje el flujo de trabajo documentado en [Análisis Dinámico (Dynamical Analysis)](dynamical_analysis.md).

## Notebook Rules / Reglas de los Notebooks

- Keep the first cells self-contained and explanatory.
- Avoid long C/EFORK runs unless the notebook name clearly says so.
- Store generated figures under `outputs/notebooks/<notebook-name>/`.
- Mirror important notebook examples as `.py` scripts when possible.
- 
- Mantener las primeras celdas autónomas y explicativas.
- Evitar ejecuciones largas de C/EFORK a menos que el nombre del notebook lo indique claramente.
- Almacenar las figuras generadas bajo `outputs/notebooks/<notebook-name>/`.
- Reflejar ejemplos de notebooks importantes como scripts `.py` cuando sea posible.
