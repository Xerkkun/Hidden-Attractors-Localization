import matplotlib as mpl
import matplotlib.pyplot as plt

def apply_library_style():
    """
    Applies the library's unified visual style rules globally via rcParams.
    """
    # Pure white backgrounds
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['savefig.facecolor'] = 'white'
    
    # Grid configuration
    mpl.rcParams['axes.grid'] = False  # Turned on manually when needed
    mpl.rcParams['grid.color'] = '#cbd5e1'
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['grid.linestyle'] = '--'
    
    # Typography
    mpl.rcParams['font.size'] = 9
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 8
    
    # Ensure titles are turned off by default (style guide constraint)
    mpl.rcParams['axes.titleweight'] = 'normal'
    mpl.rcParams['axes.titlesize'] = 10

def apply_axes_style(ax, grid=False, is_3d=False):
    """
    Applies style adjustments to an individual axes object.
    Enforces white background, removes labels/ticks customization if needed,
    and formats the grid.
    """
    ax.set_facecolor('white')
    if grid:
        ax.grid(True, color='#cbd5e1', linestyle='--', linewidth=0.5)
    else:
        ax.grid(False)
        
    if is_3d:
        # 3D specific style
        for attr in ['xaxis', 'yaxis', 'zaxis']:
            getattr(ax, attr).pane.fill = False
            getattr(ax, attr).pane.set_edgecolor('#cbd5e1')
            getattr(ax, attr).pane.set_linewidth(0.5)
        ax.grid(True, color='#cbd5e1', linestyle='--', linewidth=0.3, alpha=0.5)
        
    # Remove titles to comply with rule "Sin titulos internos"
    ax.set_title("")

def get_figsize(kind):
    """
    Returns standard figure size for the given plot kind.
    - 2D: (5.2, 3.6)
    - 3D: (5.2, 4.2)
    - Basins: (5.2, 4.2)
    """
    if kind in ["3d", "attractor_3d", "basin", "basins", "matignon"]:
        return (5.2, 4.2)
    return (5.2, 3.6)
