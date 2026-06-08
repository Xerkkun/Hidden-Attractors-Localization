FIGURE_REGISTRY = {}

def register_figure(figure_id, renderer_fn, kind):
    """
    Registers a figure renderer function.
    """
    FIGURE_REGISTRY[figure_id] = {
        "renderer": renderer_fn,
        "kind": kind
    }

def get_registered_figures():
    """
    Returns the dictionary of all registered figures.
    """
    return FIGURE_REGISTRY

# Register our primary rendering methods
from .renderers import render_attractor, render_basin, render_nyquist, render_matignon

register_figure("attractor", render_attractor, "attractor")
register_figure("basin", render_basin, "basin")
register_figure("nyquist", render_nyquist, "nyquist")
register_figure("matignon", render_matignon, "matignon")
