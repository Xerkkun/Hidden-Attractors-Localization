import re

filepath = "version_2/hidden_attractors/plotting/dynamics.py"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# We need to replace the savefig calls in each function:
# 1. plot_phase_space
# 2. plot_phase_projections
# 3. plot_time_series
# 4. plot_bifurcation_diagram
# 5. plot_lure_nyquist_describing_function
# 6. plot_lure_transfer_components
# 7. plot_integer_lure_continuation
# 8. plot_fractional_continuation_phase_story
# 9. plot_phase_space_with_reference_points
# 10. plot_integer_hiddenness_controls
# 11. plot_spectrum
# 12. plot_lyapunov_convergence

replacements = [
    (r"def plot_phase_space\(.*?\):.*?fig\.savefig\(path, dpi=220\)", 
     "attractor"),
    (r"def plot_phase_projections\(.*?\):.*?fig\.savefig\(path, dpi=220\)", 
     "attractor"),
    (r"def plot_time_series\(.*?\):.*?fig\.savefig\(path, dpi=220\)", 
     "time_series"),
    (r"def plot_bifurcation_diagram\(.*?\):.*?fig\.savefig\(path, dpi=220\)", 
     "bifurcation"),
    (r"def plot_lure_nyquist_describing_function\(.*?\):.*?fig\.savefig\(path, dpi=220\)", 
     "nyquist"),
    (r"def plot_lure_transfer_components\(.*?\):.*?fig\.savefig\(path, dpi=220\)", 
     "transfer"),
    (r"def plot_integer_lure_continuation\(.*?\):.*?fig\.savefig\(path, dpi=220\)", 
     "continuation"),
    (r"def plot_fractional_continuation_phase_story\(.*?\):.*?fig\.savefig\(path, dpi=220\)", 
     "continuation"),
    (r"def plot_phase_space_with_reference_points\(.*?\):.*?fig\.savefig\(path, dpi=220\)", 
     "attractor"),
    (r"def plot_integer_hiddenness_controls\(.*?\):.*?fig\.savefig\(path, dpi=220\)", 
     "sphere_test"),
    (r"def plot_spectrum\(.*?\):.*?fig\.savefig\(path, dpi=220\)", 
     "fft"),
    (r"def plot_lyapunov_convergence\(.*?\):.*?fig\.savefig\(path, dpi=220\)", 
     "lyapunov")
]

# Let's split content by functions or do a find-and-replace
# Since re.sub with dotall can match across functions, we must be careful.
# Instead of wild regex, let's replace the exact lines by their context.

# Let's check each target context in dynamics.py
new_content = content

# 1. plot_phase_space
target1 = """    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)"""
replacement1 = """    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "attractor")
    plt.close(fig)"""

# In plot_phase_space, it's the first occurrence.
# Let's replace only the first occurrence for plot_phase_space
new_content = new_content.replace(target1, replacement1, 1)

# Now, let's look at each savefig block in order.
# Let's replace each target by its surrounding lines so it's unique:

# 2. plot_phase_projections
target2 = """    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)"""
replacement2 = """    if title:
        fig.suptitle(title)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "attractor")
    plt.close(fig)"""
new_content = new_content.replace(target2, replacement2, 1)

# 3. plot_time_series
target3 = """    if title:
        ax.set_title(title)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)"""
replacement3 = """    if title:
        ax.set_title(title)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "time_series")
    plt.close(fig)"""
new_content = new_content.replace(target3, replacement3, 1)

# 4. plot_bifurcation_diagram
target4 = """    ax.set_xlabel(parameter_label)
    ax.set_ylabel(observable_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)"""
replacement4 = """    ax.set_xlabel(parameter_label)
    ax.set_ylabel(observable_label)
    ax.set_title(title)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "bifurcation")
    plt.close(fig)"""
new_content = new_content.replace(target4, replacement4, 1)

# 5. plot_lure_nyquist_describing_function
target5 = """    ax.set_xlabel(r"Re$(W_q(i\omega))$")
    ax.set_ylabel(r"Im$(W_q(i\omega))$")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)"""
replacement5 = """    ax.set_xlabel(r"Re$(W_q(i\omega))$")
    ax.set_ylabel(r"Im$(W_q(i\omega))$")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "nyquist")
    plt.close(fig)"""
new_content = new_content.replace(target5, replacement5, 1)

# 6. plot_lure_transfer_components
target6 = """    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)"""
# Note: both plot_lure_transfer_components and plot_integer_lure_continuation and plot_fractional_continuation_phase_story end like this.
# Let's replace the first one for plot_lure_transfer_components
replacement6 = """    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "transfer")
    plt.close(fig)
    return str(path)"""
new_content = new_content.replace(target6, replacement6, 1)

# 7. plot_integer_lure_continuation
target7 = """    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)"""
replacement7 = """    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "continuation")
    plt.close(fig)
    return str(path)"""
new_content = new_content.replace(target7, replacement7, 1)

# 8. plot_fractional_continuation_phase_story
target8 = """    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)"""
replacement8 = """    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "continuation")
    plt.close(fig)"""
new_content = new_content.replace(target8, replacement8, 1)

# 9. plot_phase_space_with_reference_points
target9 = """    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if seed_effective is not None or continuation_final is not None:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)"""
replacement9 = """    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if seed_effective is not None or continuation_final is not None:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "attractor")
    plt.close(fig)"""
new_content = new_content.replace(target9, replacement9, 1)

# 10. plot_integer_hiddenness_controls
target10 = """    ax.set_xlabel(f"x{dims[0]}")
    ax.set_ylabel(f"x{dims[1]}")
    ax.set_zlabel(f"x{dims[2]}")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)"""
replacement10 = """    ax.set_xlabel(f"x{dims[0]}")
    ax.set_ylabel(f"x{dims[1]}")
    ax.set_zlabel(f"x{dims[2]}")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "sphere_test")
    plt.close(fig)"""
new_content = new_content.replace(target10, replacement10, 1)

# 11. plot_spectrum
target11 = """    if omega_marker is not None and x_units in {"rad/s", "omega"}:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)"""
replacement11 = """    if omega_marker is not None and x_units in {"rad/s", "omega"}:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "fft")
    plt.close(fig)"""
new_content = new_content.replace(target11, replacement11, 1)

# 12. plot_lyapunov_convergence
target12 = """    ax.set_xlabel("time")
    ax.set_ylabel("exponent")
    ax.set_title("Lyapunov convergence")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)"""
replacement12 = """    ax.set_xlabel("time")
    ax.set_ylabel("exponent")
    ax.set_title("Lyapunov convergence")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, path, "lyapunov")
    plt.close(fig)"""
new_content = new_content.replace(target12, replacement12, 1)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(new_content)

print("dynamics.py updated successfully!")
