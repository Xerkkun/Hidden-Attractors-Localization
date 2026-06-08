import os
import re

# List of files requested for explicit review
requested_files = [
    "scratch_plot_saturation_attractor.py",
    "scratch_extend_user.py",
    "scratch_four_plots.py",
    "generate_all_plots_and_summary.py",
    "plot_chaotic_candidates.py",
    "search_saturation_candidates.py",
    "search_arctan_fractional.py",
    "compare_solvers_saturation.py",
    "tools/generate_saturation_candidate_report_assets.py",
    "tools/run_saturation_candidate_chaos_hiddenness_tests.py",
    "tools/run_danca_exact_chaos_hiddenness_tests.py",
    "version_2/tools/plot_candidate_story_figures.py",
    "version_2/tools/plot_sphere_closeup.py",
    "version_2/tools/plot_validation_outputs.py",
    "version_2/tools/plot_fractional_reference_attractors.py",
    "version_2/tools/search_arctan_full_memory_candidates.py",
    "version_2/tools/validation/generate_chua_nonsmooth_efork_figure.py",
    "version_2/tools/validation/validate_chua_fractional_nonsmooth_algebra.py",
    "version_2/tools/validation/validate_efork3_ghoreishi_ghaffari.py",
    "version_2/tools/legacy/chua_initial_cond.py",
    "version_2/tools/legacy/harmonic_diagnostics.py",
    "version_2/tools/legacy/danca2017_chua_abm_replication.py",
    "version_2/docs/scripts/generate_chua_model_seed_continuation_figures.py",
    "version_2/docs/scripts/generate_chua_nonlinearity_comparison.py",
    "version_2/docs/scripts/generate_lorenz_chaos_figures.py",
    "version_2/docs/scripts/generate_matignon_complex_plane.py",
    "version_2/examples/chua_arctan_wu2023/plot_basins.py",
    "version_2/experiments/chua_nonsmooth_memory_matrix/run_figure_tasks.py",
    "version_2/hidden_attractors/plotting/plot_trajectories.py",
    "version_2/hidden_attractors/plotting/plot_basins.py",
    "version_2/hidden_attractors/plotting/plot_df.py",
    "version_2/hidden_attractors/plotting/plot_transfer.py",
    "version_2/hidden_attractors/plotting/plot_continuation.py",
    "version_2/hidden_attractors/plotting/plot_matignon.py",
    "version_2/hidden_attractors/plotting/plot_sphere_tests.py",
    "version_2/hidden_attractors/plotting/dynamics.py",
    "version_2/hidden_attractors/plotting/basin.py",
    "version_2/hidden_attractors/plotting/overlays.py",
    "version_2/hidden_attractors/plotting/generate_publication_figures.py"
]

root_dir = r"c:\Users\moren\Desktop\Codes\Hidden Attractors Fractional Order"

# Map to search files and determine classification, inputs, outputs, etc.
inventory = []

for file_rel in requested_files:
    # Look for the file in the workspace
    filepath = os.path.join(root_dir, file_rel.replace('/', os.sep))
    exists = os.path.exists(filepath)
    
    if not exists:
        # Check if it exists under version_2/ if it was specified under root, or vice versa
        if file_rel.startswith("version_2/"):
            alt_path = os.path.join(root_dir, file_rel[10:].replace('/', os.sep))
            if os.path.exists(alt_path):
                filepath = alt_path
                exists = True
        else:
            alt_path = os.path.join(root_dir, "version_2", file_rel.replace('/', os.sep))
            if os.path.exists(alt_path):
                filepath = alt_path
                exists = True
                
    rel_found = os.path.relpath(filepath, root_dir).replace('\\', '/') if exists else file_rel
    
    if not exists:
        inventory.append({
            "path": rel_found,
            "exists": False,
            "type": "N/A",
            "figures": "N/A",
            "data": "N/A",
            "outputs": "N/A",
            "action": "Archivar (No existe en el disco)",
            "deficiencies": "No existe"
        })
        continue

    # Analyze existing file
    content = ""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        content = ""

    # Basic analysis
    lines = content.split('\n')
    has_savefig = "savefig" in content
    has_title = "set_title" in content or "suptitle" in content or "plt.title" in content
    has_dark = any(x in content for x in ["facecolor", "BG", "PANEL", "dark", "#0d0d1a"])
    
    # Classification logic based on path
    if "plotting" in filepath:
        classification = "Módulo de librería"
        action = "Conservar / Estandarizar"
    elif "example" in filepath:
        classification = "Ejemplo de flujo"
        action = "Conservar / Integrar"
    elif "validation" in filepath or "validate" in filepath:
        classification = "Script de validación"
        action = "Migrar a tests/validación y archivar"
    elif "legacy" in filepath or "chua_initial_cond.py" in filepath or "harmonic_diagnostics" in filepath or "danca2017" in filepath:
        classification = "Legacy"
        action = "Archivar"
    elif "scratch" in file_rel or "scratch_" in file_rel:
        classification = "Scratch / Temporal"
        action = "Archivar"
    elif "experiments" in filepath:
        classification = "Experimento"
        action = "Archivar"
    elif "tools" in filepath:
        classification = "Herramienta / Script de reporte"
        action = "Migrar o Archivar"
    else:
        classification = "Script de reporte / Workflow"
        action = "Migrar / Archivar"

    # Try to extract generated figures and consumed data
    fig_matches = re.findall(r"savefig\(([^)]+)\)", content)
    figures_list = []
    for m in fig_matches:
        # clean up the filename string
        m_clean = m.split(',')[0].strip().replace("'", "").replace('"', '')
        figures_list.append(m_clean)
    
    figures_desc = ", ".join(figures_list) if figures_list else "Figuras de análisis (traectorias, espectro, etc.)"
    if "attractor" in file_rel:
        figures_desc = "Trayectorias del atractor"
    elif "basin" in file_rel:
        figures_desc = "Cuencas de atracción"
    elif "nyquist" in file_rel or "df" in file_rel:
        figures_desc = "Función Descriptiva / Nyquist"
    elif "matignon" in file_rel:
        figures_desc = "Sector de Matignon / Eigenvalores"

    # Detect data consumed
    data_consumed = "Configuración/parámetros en el código"
    if "load" in content or "read_" in content or "open(" in content:
        data_consumed = "Archivos JSON/CSV de corridas u outputs"
    if "states" in content or "solve" in content or "integrate" in content:
        data_consumed += " + Simulación en tiempo real (integrador)"

    # Deficiencies check
    deficiencies = []
    if has_savefig:
        deficiencies.append("Llamada directa a savefig")
    if has_title:
        deficiencies.append("Uso de títulos internos (title/suptitle)")
    if has_dark:
        deficiencies.append("Fondos no blancos / Estilos customizados")
    if "pdf" in content and "png" not in content:
        deficiencies.append("Exporta solo PDF")
    elif "png" in content and "pdf" not in content:
        deficiencies.append("Exporta solo PNG")
    elif not has_savefig:
        deficiencies.append("No exporta (solo plt.show o módulo interno)")

    def_desc = ", ".join(deficiencies) if deficiencies else "Ninguna detectada"

    inventory.append({
        "path": rel_found,
        "exists": True,
        "type": classification,
        "figures": figures_desc,
        "data": data_consumed,
        "outputs": "Figuras exportadas" if has_savefig else "Ninguna (módulo)",
        "action": action,
        "deficiencies": def_desc
    })

# Output markdown inventory
md_filepath = os.path.join(root_dir, "version_2", "docs", "figure_scripts_inventory.md")
os.makedirs(os.path.dirname(md_filepath), exist_ok=True)

with open(md_filepath, "w", encoding="utf-8") as f:
    f.write("# Inventario de Scripts de Figuras\n\n")
    f.write("Este inventario recopila los scripts de generación de figuras en el repositorio, su clasificación, entradas/salidas y el plan de migración/archivo correspondiente.\n\n")
    
    f.write("| Ruta del Archivo | Clasificación | Figuras que genera | Datos que consume | Salidas | Acción propuesta | Deficiencias de Estilo / API |\n")
    f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
    for item in inventory:
        f.write(f"| `{item['path']}` | {item['type']} | {item['figures']} | {item['data']} | {item['outputs']} | **{item['action']}** | {item['deficiencies']} |\n")

print(f"Inventory generated at {md_filepath}")
