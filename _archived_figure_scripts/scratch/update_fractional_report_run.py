filepath = "version_2/hidden_attractors/workflows/fractional_report_run.py"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Replace savefig in _plot_biased_nyquist_df
target1 = """    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)"""
replacement1 = """    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    from version_2.hidden_attractors.plotting.export import intercept_and_export_path
    intercept_and_export_path(fig, output, "nyquist")
    plt.close(fig)"""

# Replace savefig in _plot_spectrum
target2 = """    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)"""
# Note: this might appear in multiple functions (like _plot_spectrum and _plot_robustness_overlay).
# Let's check how many times it appears.
# We will do a generic replacement for this block since they both use "output" as the variable and plt.close(fig).
# Let's specify:
replacement2 = """    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    from version_2.hidden_attractors.plotting.export import intercept_and_export_path
    intercept_and_export_path(fig, output, "diagnostics")
    plt.close(fig)"""

# Let's check how many times target2 is in the file.
# If we count, it should be in _plot_spectrum and _plot_robustness_overlay.
# Since one is "spectrum" (or "fft") and the other is "robustness", let's customize them uniquely:
target_spectrum = """    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)""" # but wait, let's verify if _plot_spectrum has:
#    ax.legend(loc="best", fontsize=8)
#    fig.tight_layout()
#    fig.savefig(output, dpi=220)
#    plt.close(fig)
# Let's verify the exact lines. In the view:
# 724:def _plot_spectrum(traj: np.ndarray, h: float, output: Path, *, omega0: float | None = None) -> None:
# ...
# 741:    fig.savefig(output, dpi=220)
# 742:    plt.close(fig)
# Let's see what is immediately before fig.savefig(output, dpi=220) in _plot_spectrum.
# Let's just find and replace using unique targets.

# In _plot_spectrum:
target_spec = """    ax.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
    ax.plot(f_rad, values, lw=0.9, color="#1e293b")
    if omega0 is not None:
        ax.axvline(omega0, color="#ef4444", lw=1.0, ls="--", label=f"omega0={omega0:.4f}")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)"""

replacement_spec = """    ax.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
    ax.plot(f_rad, values, lw=0.9, color="#1e293b")
    if omega0 is not None:
        ax.axvline(omega0, color="#ef4444", lw=1.0, ls="--", label=f"omega0={omega0:.4f}")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    from version_2.hidden_attractors.plotting.export import intercept_and_export_path
    intercept_and_export_path(fig, output, "fft")
    plt.close(fig)"""

# In _plot_robustness_overlay:
target_rob = """    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)"""

replacement_rob = """    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    from version_2.hidden_attractors.plotting.export import intercept_and_export_path
    intercept_and_export_path(fig, output, "robustness")
    plt.close(fig)"""

# In plot_hiddenness_ball_figures:
target_ball = """        path = plot_dir / f"{safe_name(candidate_id)}_equilibrium_ball_samples.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)"""

replacement_ball = """        path = plot_dir / f"{safe_name(candidate_id)}_equilibrium_ball_samples.png"
        from version_2.hidden_attractors.plotting.export import intercept_and_export_path
        intercept_and_export_path(fig, path, "sphere_test")
        plt.close(fig)"""

new_content = content.replace(target1, replacement1)
new_content = new_content.replace(target_spec, replacement_spec)
new_content = new_content.replace(target_rob, replacement_rob)
new_content = new_content.replace(target_ball, replacement_ball)

if new_content != content:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("fractional_report_run.py updated successfully!")
else:
    print("No changes made to fractional_report_run.py.")
