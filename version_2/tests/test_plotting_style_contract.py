import matplotlib.pyplot as plt
import pytest
from hidden_attractors.plotting.style import apply_library_style, apply_axes_style, get_figsize

def test_apply_library_style():
    apply_library_style()
    fig, ax = plt.subplots()
    
    # Verify backgrounds are white
    # Matplotlib stores color as RGBA tuple, (1, 1, 1, 1) or 'white' or '#ffffff'
    fig_color = fig.get_facecolor()
    assert fig_color in [(1.0, 1.0, 1.0, 1.0), "white", "#ffffff"]
    
    # Verify no title
    assert ax.get_title() == ""
    assert fig._suptitle is None
    plt.close(fig)

def test_apply_axes_style():
    fig, ax = plt.subplots()
    apply_axes_style(ax, grid=True)
    
    assert ax.get_facecolor() in [(1.0, 1.0, 1.0, 1.0), "white", "#ffffff", (1.0, 1.0, 1.0, 0.0)]
    assert ax.get_title() == ""
    plt.close(fig)

def test_get_figsize():
    assert get_figsize("3d") == (5.2, 4.2)
    assert get_figsize("basin") == (5.2, 4.2)
    assert get_figsize("2d") == (5.2, 3.6)
