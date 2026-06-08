import os

root_dir = "version_2/hidden_attractors/plotting"

def replace_in_file(filename, replacements_list):
    filepath = os.path.join(root_dir, filename)
    if not os.path.exists(filepath):
        print(f"Skipping {filename}: not found.")
        return
        
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    original_content = content
    for target, replacement in replacements_list:
        content = content.replace(target, replacement)
        
    if content != original_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated {filename}")
    else:
        print(f"No changes made to {filename}")

# 1. basin.py
replace_in_file("basin.py", [
    ("fig.savefig(fig_path, dpi=300)", """from .export import intercept_and_export_path
    intercept_and_export_path(fig, fig_path, "basin")""")
])

# 2. plot_basins.py
replace_in_file("plot_basins.py", [
    ("fig.savefig(fig_path, dpi=300)", """from .export import intercept_and_export_path
    intercept_and_export_path(fig, fig_path, "basin")""")
])

# 3. matignon.py
replace_in_file("matignon.py", [
    ("fig.savefig(fig_path, dpi=300)", """from .export import intercept_and_export_path
    intercept_and_export_path(fig, fig_path, "matignon")""")
])

# 4. plot_matignon.py
replace_in_file("plot_matignon.py", [
    ("fig.savefig(fig_path, dpi=300)", """from .export import intercept_and_export_path
    intercept_and_export_path(fig, fig_path, "matignon")""")
])

# 5. overlays.py
replace_in_file("overlays.py", [
    ("fig.savefig(path, dpi=220)", """from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "attractor")""")
])

# 6. plot_df.py
replace_in_file("plot_df.py", [
    ('fig.savefig(os.path.join(fig_dir, "describing_function.png"), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig, os.path.join(fig_dir, "describing_function.png"), "nyquist")"""),
    ('fig.savefig(os.path.join(fig_dir, "harmonic_residual_map.png"), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig, os.path.join(fig_dir, "harmonic_residual_map.png"), "nyquist")""")
])

# 7. plot_sphere_tests.py
replace_in_file("plot_sphere_tests.py", [
    ("fig.savefig(fig_path_png, dpi=300)", """from .export import intercept_and_export_path
    intercept_and_export_path(fig, fig_path_png, "sphere_test")"""),
    ("fig.savefig(fig_path_pdf)", "pass")
])

# 8. plot_continuation.py
replace_in_file("plot_continuation.py", [
    ('fig_3d.savefig(os.path.join(fig_dir, "continuation_first_last_comparison.png"), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig_3d, os.path.join(fig_dir, "continuation_first_last_comparison.png"), "continuation")"""),
    ('fig_2d.savefig(os.path.join(fig_dir, "continuation_first_last_projections.png"), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig_2d, os.path.join(fig_dir, "continuation_first_last_projections.png"), "continuation")"""),
    ('fig.savefig(os.path.join(fig_dir, "continuation_timeseries_comparison_x.png"), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig, os.path.join(fig_dir, "continuation_timeseries_comparison_x.png"), "continuation")"""),
    ('fig.savefig(os.path.join(fig_dir, "continuation_progression.png"), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig, os.path.join(fig_dir, "continuation_progression.png"), "continuation")"""),
    ('fig.savefig(os.path.join(fig_dir, "continuation_tracking_status.png"), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig, os.path.join(fig_dir, "continuation_tracking_status.png"), "continuation")""")
])

# 9. plot_trajectories.py
replace_in_file("plot_trajectories.py", [
    ('fig_3d.savefig(os.path.join(fig_dir, filename_3d), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig_3d, os.path.join(fig_dir, filename_3d), "attractor")"""),
    ('fig_2d.savefig(os.path.join(fig_dir, f"{file_prefix}_{proj_name}.png"), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig_2d, os.path.join(fig_dir, f"{file_prefix}_{proj_name}.png"), "attractor")"""),
    ('fig.savefig(os.path.join(fig_dir, f"{file_prefix}_timeseries_{var_name}.png"), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig, os.path.join(fig_dir, f"{file_prefix}_timeseries_{var_name}.png"), "time_series")"""),
    ('fig.savefig(os.path.join(fig_dir, f"{file_prefix}_timeseries_xyz.png"), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig, os.path.join(fig_dir, f"{file_prefix}_timeseries_xyz.png"), "time_series")"""),
    ('fig.savefig(os.path.join(fig_dir, "fig05b_hiddenness_overview.png"), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig, os.path.join(fig_dir, "fig05b_hiddenness_overview.png"), "sphere_test")"""),
    ('fig.savefig(os.path.join(fig_dir, "fig05b_hiddenness_overview.pdf"))', "pass")
])

# 10. plot_transfer.py
replace_in_file("plot_transfer.py", [
    ('fig.savefig(os.path.join(fig_dir, "transfer_nyquist.png"), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig, os.path.join(fig_dir, "transfer_nyquist.png"), "nyquist")"""),
    ('fig.savefig(os.path.join(fig_dir, "fig01_nyquist_df.png"), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig, os.path.join(fig_dir, "fig01_nyquist_df.png"), "nyquist")"""),
    ('fig.savefig(os.path.join(fig_dir, "fig01_nyquist_df.pdf"))', "pass"),
    ('fig_zoom.savefig(os.path.join(fig_dir, "fig01b_nyquist_zoom_x.png"), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig_zoom, os.path.join(fig_dir, "fig01b_nyquist_zoom_x.png"), "nyquist")"""),
    ('fig_zoom.savefig(os.path.join(fig_dir, "fig01b_nyquist_zoom_x.pdf"))', "pass"),
    ('fig_comp.savefig(os.path.join(fig_dir, "transfer_real_imag.png"), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig_comp, os.path.join(fig_dir, "transfer_real_imag.png"), "transfer")"""),
    ('fig_comp.savefig(os.path.join(fig_dir, "fig01c_transfer_real_imag.png"), dpi=300)', """from .export import intercept_and_export_path
    intercept_and_export_path(fig_comp, os.path.join(fig_dir, "fig01c_transfer_real_imag.png"), "transfer")"""),
    ('fig_comp.savefig(os.path.join(fig_dir, "fig01c_transfer_real_imag.pdf"))', "pass")
])

# 11. generate_publication_figures.py
replace_in_file("generate_publication_figures.py", [
    ("fig.savefig(str(path_png), dpi=300, bbox_inches='tight')", """from .export import intercept_and_export_path
    intercept_and_export_path(fig, str(path_png), "publication")"""),
    ("fig.savefig(str(path_pdf), bbox_inches='tight')", "pass")
])

print("All other plotting modules updated!")
