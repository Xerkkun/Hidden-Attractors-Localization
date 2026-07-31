import os
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, List, Tuple


_VALID_TRANSFER_CONVENTIONS = {"standard", "opposite_sign"}
_VALID_HARMONIC_CONDITIONS = {"1_minus_WN", "1_plus_WN"}


def _resolve_transfer_plot_contract(config: dict) -> tuple[str, str]:
    """Resolve a coherent transfer/sign pair for standalone plotting calls.

    The historical plotting default is the normalized/report convention
    ``W_report = c^T(sI-P)^(-1)b`` with ``1-NW_report=0``.  If callers provide
    only one of the two contract keys, the matching counterpart is inferred.
    """
    transfer_convention = config.get("transfer_convention")
    harmonic_condition = config.get("harmonic_condition")

    if transfer_convention is None and harmonic_condition is None:
        transfer_convention = "standard"
        harmonic_condition = "1_minus_WN"
    elif transfer_convention is None:
        transfer_convention = (
            "standard"
            if harmonic_condition == "1_minus_WN"
            else "opposite_sign"
        )
    elif harmonic_condition is None:
        harmonic_condition = (
            "1_minus_WN"
            if transfer_convention == "standard"
            else "1_plus_WN"
        )

    if transfer_convention not in _VALID_TRANSFER_CONVENTIONS:
        raise ValueError(
            "Invalid transfer_convention for plotting: "
            f"{transfer_convention!r}. Must be one of "
            f"{sorted(_VALID_TRANSFER_CONVENTIONS)}."
        )
    if harmonic_condition not in _VALID_HARMONIC_CONDITIONS:
        raise ValueError(
            "Invalid harmonic_condition for plotting: "
            f"{harmonic_condition!r}. Must be one of "
            f"{sorted(_VALID_HARMONIC_CONDITIONS)}."
        )

    is_standard_pair = (
        transfer_convention == "standard"
        and harmonic_condition == "1_minus_WN"
    )
    is_code_pair = (
        transfer_convention == "opposite_sign"
        and harmonic_condition == "1_plus_WN"
    )
    if (
        not (is_standard_pair or is_code_pair)
        and not config.get("allow_nonstandard_sign_pairing", False)
    ):
        raise ValueError(
            "Incoherent transfer/sign pair for plotting: use "
            "('standard', '1_minus_WN') for W_report or "
            "('opposite_sign', '1_plus_WN') for W_code. Set "
            "allow_nonstandard_sign_pairing=True only for an intentional "
            "nonstandard comparison."
        )

    return transfer_convention, harmonic_condition


def _closure_plot_terms(
    harmonic_condition: str,
) -> tuple[float, str, str]:
    """Return target multiplier, LaTeX target, and residual expression."""
    if harmonic_condition == "1_minus_WN":
        return 1.0, r"1/k", r"1-N(A)W(i\omega)"
    return -1.0, r"-1/k", r"1+N(A)W(i\omega)"


def plot_nyquist_transfer(
    omega_grid: np.ndarray,
    w_vals: np.ndarray,
    candidates: List[Tuple[float, float, float]],
    config: dict,
    output_dir: str
) -> None:
    """Plot ``w_vals`` with the closure selected by the transfer contract.

    ``w_vals`` are assumed to have been evaluated using the
    ``transfer_convention`` declared in ``config``.  Calls that omit both sign
    keys keep the historical ``standard``/``1_minus_WN`` behavior.
    """
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    transfer_convention, harmonic_condition = _resolve_transfer_plot_contract(
        config
    )
    target_multiplier, target_latex, _ = _closure_plot_terms(
        harmonic_condition
    )
    transfer_symbol = (
        r"W_{\mathrm{report}}"
        if transfer_convention == "standard"
        else r"W_{\mathrm{code}}"
    )
    
    # 1. RENDER NYQUIST PLOT
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#cbd5e1')
    ax.axhline(0, color='#64748b', linewidth=1.0)
    ax.axvline(0, color='#64748b', linewidth=1.0)
    
    real_parts = [val.real for val in w_vals if not np.isnan(val.real)]
    imag_parts = [val.imag for val in w_vals if not np.isnan(val.imag)]
    
    if len(real_parts) > 0:
        ax.plot(
            real_parts,
            imag_parts,
            color='#0284c7',
            linewidth=1.8,
            label=rf'${transfer_symbol}(i\omega)$ trajectory',
        )
        
    for idx, (A, w0, k) in enumerate(candidates):
        target_pt = target_multiplier / k
        ax.scatter([target_pt], [0.0], color='#ef4444', s=60, zorder=5,
                   label=(
                       f'closure {idx+1}: $W={target_latex}$, '
                       f'$\\omega_0$={w0:.3f}, $k$={k:.3f}'
                       if idx == 0 else f'closure {idx+1}'
                   ))
                   
    ax.set_title(f"Nyquist Plot - {config['system_id']}", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel(r'$\mathrm{Re}(W)$', fontsize=10)
    ax.set_ylabel(r'$\mathrm{Im}(W)$', fontsize=10)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, os.path.join(fig_dir, "transfer_nyquist.png"), "nyquist")
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, os.path.join(fig_dir, "fig01_nyquist_df.png"), "nyquist")
    pass
    plt.close(fig)
    
    # 1.b. RENDER NYQUIST ZOOM PLOT (fig01b_nyquist_zoom_x)
    if len(candidates) > 0:
        fig_zoom, ax_zoom = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        ax_zoom.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
        ax_zoom.axhline(0, color="#64748b", linewidth=0.8, linestyle=":")
        ax_zoom.axvline(0, color="#64748b", linewidth=0.8, linestyle=":")
        
        if len(real_parts) > 0:
            ax_zoom.plot(real_parts, imag_parts, color="#0284c7", linewidth=1.8)
            
        target_pt = target_multiplier / candidates[0][2]
        ax_zoom.scatter([target_pt], [0.0], color='#ef4444', marker='x', s=60, zorder=5)
        
        span = 0.5
        ax_zoom.set_xlim(target_pt - span, target_pt + span)
        ax_zoom.set_ylim(-span, span)
        ax_zoom.set_title(
            rf"Nyquist Zoom around $W={target_latex}$",
            fontsize=11,
            fontweight="bold",
            pad=12,
        )
        ax_zoom.set_xlabel(r"$\mathrm{Re}(W)$")
        ax_zoom.set_ylabel(r"$\mathrm{Im}(W)$")
        plt.tight_layout()
        from .export import intercept_and_export_path
        intercept_and_export_path(fig_zoom, os.path.join(fig_dir, "fig01b_nyquist_zoom_x.png"), "nyquist")
        pass
        plt.close(fig_zoom)
    
    # 2. RENDER REAL & IMAG COMPONENTS PLOT
    fig_comp, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, dpi=300)
    
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='#cbd5e1')
    axes[0].plot(
        omega_grid,
        np.real(w_vals),
        color='#2563eb',
        linewidth=1.5,
        label=rf'$\mathrm{{Re}}({transfer_symbol}(i\omega))$',
    )
    if len(candidates) > 0:
        chosen_k = candidates[0][2]
        closure_value = target_multiplier / chosen_k
        axes[0].axhline(
            closure_value,
            color='#ef4444',
            linestyle='--',
            linewidth=1.1,
            label=rf'${target_latex}$ closure',
        )
        axes[0].scatter(
            [candidates[0][1]],
            [closure_value],
            color='#ef4444',
            s=45,
            zorder=5,
        )
        
    axes[0].set_ylabel(r'$\mathrm{Re}(W)$', fontsize=10)
    axes[0].set_title("Real Component vs Frequency", fontsize=11, fontweight='bold')
    axes[0].legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='#cbd5e1')
    axes[1].plot(
        omega_grid,
        np.imag(w_vals),
        color='#0891b2',
        linewidth=1.5,
        label=rf'$\mathrm{{Im}}({transfer_symbol}(i\omega))$',
    )
    axes[1].axhline(0.0, color='#64748b', linestyle='--', linewidth=1.0, label='Zero crossing')
    if len(candidates) > 0:
        axes[1].scatter([candidates[0][1]], [0.0], color='#ef4444', s=45, zorder=5)
        
    axes[1].set_xlabel(r'$\omega$ (rad/s)', fontsize=10)
    axes[1].set_ylabel(r'$\mathrm{Im}(W)$', fontsize=10)
    axes[1].set_title("Imaginary Component vs Frequency", fontsize=11, fontweight='bold')
    axes[1].legend(loc='best', fontsize=8, framealpha=0.9, facecolor='#f8fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    from .export import intercept_and_export_path
    intercept_and_export_path(fig_comp, os.path.join(fig_dir, "transfer_real_imag.png"), "transfer")
    from .export import intercept_and_export_path
    intercept_and_export_path(fig_comp, os.path.join(fig_dir, "fig01c_transfer_real_imag.png"), "transfer")
    pass
    plt.close(fig_comp)
