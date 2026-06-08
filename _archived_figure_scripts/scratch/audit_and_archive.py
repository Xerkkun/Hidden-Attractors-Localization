import os
import shutil
import datetime
from pathlib import Path

# Paths
root_dir = Path("c:/Users/moren/Desktop/Codes/Hidden Attractors Fractional Order")
archive_dir = root_dir / "_archived_figure_scripts"

# Candidate files to check and archive if not imported
candidates = [
    # Root scripts
    ("scratch_plot_saturation_attractor.py", "root"),
    ("scratch_extend_user.py", "root"),
    ("scratch_four_plots.py", "root"),
    ("generate_all_plots_and_summary.py", "root"),
    ("plot_chaotic_candidates.py", "root"),
    ("search_saturation_candidates.py", "root"),
    ("search_arctan_fractional.py", "root"),
    ("compare_solvers_saturation.py", "root"),
    
    # Root tools
    ("tools/generate_saturation_candidate_report_assets.py", "tools"),
    ("tools/run_saturation_candidate_chaos_hiddenness_tests.py", "tools"),
    ("tools/run_danca_exact_chaos_hiddenness_tests.py", "tools"),
    
    # version_2 tools
    ("version_2/tools/plot_candidate_story_figures.py", "version_2_tools"),
    ("version_2/tools/plot_sphere_closeup.py", "version_2_tools"),
    ("version_2/tools/plot_validation_outputs.py", "version_2_tools"),
    ("version_2/tools/plot_fractional_reference_attractors.py", "version_2_tools"),
    ("version_2/tools/search_arctan_full_memory_candidates.py", "version_2_tools"),
    
    # version_2 validation
    ("version_2/tools/validation/generate_chua_nonsmooth_efork_figure.py", "validation"),
    ("version_2/tools/validation/validate_chua_fractional_nonsmooth_algebra.py", "validation"),
    ("version_2/tools/validation/validate_efork3_ghoreishi_ghaffari.py", "validation"),
    
    # version_2 legacy
    ("version_2/tools/legacy/chua_initial_cond.py", "legacy"),
    ("version_2/tools/legacy/harmonic_diagnostics.py", "legacy"),
    ("version_2/tools/legacy/danca2017_chua_abm_replication.py", "legacy"),
    
    # docs/scripts
    ("version_2/docs/scripts/generate_chua_model_seed_continuation_figures.py", "docs_scripts"),
    ("version_2/docs/scripts/generate_chua_nonlinearity_comparison.py", "docs_scripts"),
    ("version_2/docs/scripts/generate_lorenz_chaos_figures.py", "docs_scripts"),
    ("version_2/docs/scripts/generate_matignon_complex_plane.py", "docs_scripts"),
    
    # examples and experiments
    ("version_2/examples/chua_arctan_wu2023/plot_basins.py", "examples"),
    ("version_2/experiments/chua_nonsmooth_memory_matrix/run_figure_tasks.py", "experiments")
]

# Find active files in the workspace to audit references
active_files = []
ignore_dirs = {".git", ".venv", "_archived_figure_scripts", "scratch", ".pytest_cache", ".pytest_tmp", "__pycache__", "build", "dist"}

for r, ds, fs in os.walk(str(root_dir)):
    ds[:] = [d for d in ds if d not in ignore_dirs]
    for f in fs:
        if f.endswith(".py") or f.endswith(".yaml") or f.endswith(".yml"):
            active_files.append(Path(r) / f)

def file_is_referenced(filename_stem, filepath):
    # Check if the filename stem is referenced in any active file excluding itself and scratch files
    for af in active_files:
        if af.name == filepath.name or "scratch" in str(af):
            continue
        try:
            content = af.read_text(encoding="utf-8", errors="ignore")
            # Look for import or script execution reference
            if filename_stem in content:
                return af
        except Exception:
            pass
    return None

archived_entries = []
kept_entries = []

for rel_path_str, category in candidates:
    filepath = root_dir / rel_path_str.replace('/', os.sep)
    if not filepath.exists():
        # Check if it is in version_2/ or root if misplaced
        if rel_path_str.startswith("version_2/"):
            alt_path = root_dir / rel_path_str[10:].replace('/', os.sep)
            if alt_path.exists():
                filepath = alt_path
        else:
            alt_path = root_dir / "version_2" / rel_path_str.replace('/', os.sep)
            if alt_path.exists():
                filepath = alt_path

    if not filepath.exists():
        print(f"File {rel_path_str} does not exist. Skipping.")
        continue

    stem = filepath.stem
    ref_file = file_is_referenced(stem, filepath)
    
    # Special safety check: do not archive validate_chua_fractional_nonsmooth_algebra or plot_candidate_story_figures
    # as they are actively called by fractional_report_run.py
    if stem in ["validate_chua_fractional_nonsmooth_algebra", "plot_candidate_story_figures"]:
        ref_file = "fractional_report_run.py (known runtime dependency)"
        
    if ref_file:
        print(f"Keeping {filepath.name} because it is referenced in {ref_file}")
        kept_entries.append((filepath, ref_file))
        
        # Update savefig in kept files to use intercept_and_export_path
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            if "savefig" in content and "intercept_and_export_path" not in content:
                # Add import and replacement
                # We can do a simple replacement for savefig calls
                # Find all fig.savefig(...) and replace with intercept_and_export_path
                lines = content.split('\n')
                updated = False
                for idx, line in enumerate(lines):
                    if "savefig(" in line and "intercept_and_export_path" not in line:
                        # Find the figure variable name, usually fig or fig_3d or fig_2d
                        fig_var = "fig"
                        if "fig_3d" in line:
                            fig_var = "fig_3d"
                        elif "fig_2d" in line:
                            fig_var = "fig_2d"
                        elif "fig_zoom" in line:
                            fig_var = "fig_zoom"
                        elif "fig_comp" in line:
                            fig_var = "fig_comp"
                            
                        # Extract the path argument
                        # e.g., fig.savefig(path, dpi=220)
                        # We will replace this line with a call to intercept_and_export_path
                        # and import it
                        indent = len(line) - len(line.lstrip())
                        line_indent = " " * indent
                        
                        kind = "attractor"
                        if "nyquist" in line.lower() or "df" in line.lower() or "transfer" in line.lower():
                            kind = "nyquist"
                        elif "matignon" in line.lower() or "complex" in line.lower():
                            kind = "matignon"
                        elif "basin" in line.lower():
                            kind = "basin"
                        elif "sphere" in line.lower() or "hiddenness" in line.lower():
                            kind = "sphere_test"
                            
                        # Extract path expression
                        # split by ( and )
                        args = line.split("savefig(")[1].split(")")[0]
                        path_expr = args.split(",")[0].strip()
                        
                        replacement_line = f"{line_indent}from version_2.hidden_attractors.plotting.export import intercept_and_export_path\n{line_indent}intercept_and_export_path({fig_var}, {path_expr}, '{kind}')"
                        lines[idx] = replacement_line
                        updated = True
                if updated:
                    filepath.write_text("\n".join(lines), encoding="utf-8")
                    print(f"Updated savefig calls in active file: {filepath.name}")
        except Exception as e:
            print(f"Error updating active file {filepath.name}: {e}")
    else:
        # Move to archive
        dest_subfolder = archive_dir / category
        dest_subfolder.mkdir(parents=True, exist_ok=True)
        dest_filepath = dest_subfolder / filepath.name
        
        # Move file
        shutil.move(str(filepath), str(dest_filepath))
        print(f"Archived {filepath.name} -> {dest_filepath.relative_to(root_dir)}")
        
        archived_entries.append({
            "original_path": str(filepath.relative_to(root_dir)).replace('\\', '/'),
            "new_path": str(dest_filepath.relative_to(root_dir)).replace('\\', '/'),
            "reason": f"Obsolete figure script. Replaced by central plotting library.",
            "equivalent": "version_2/hidden_attractors/plotting/render_all.py",
            "date": datetime.date.today().isoformat()
        })

# Write ARCHIVE_INDEX.md
index_path = archive_dir / "ARCHIVE_INDEX.md"
with open(index_path, "w", encoding="utf-8") as f:
    f.write("# Indice de Scripts de Figuras Archivados\n\n")
    f.write("Este directorio contiene scripts de generación de figuras históricos y obsoletos que han sido desactivados en favor del sistema unificado de plotting de la librería.\n\n")
    
    f.write("| Ruta Original | Ruta Archivada | Razón de Archivo | Equivalente en Plotting Library | Fecha |\n")
    f.write("| --- | --- | --- | --- | --- |\n")
    for entry in archived_entries:
        f.write(f"| `{entry['original_path']}` | `{entry['new_path']}` | {entry['reason']} | `{entry['equivalent']}` | {entry['date']} |\n")

print(f"Archiving complete. ARCHIVE_INDEX.md written at {index_path}")
