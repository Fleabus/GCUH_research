import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.widgets import Slider
figure = False

def multicolored_lines():
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """
    x = np.linspace(0, 3000, 3000)
    y = np.random.rand(len(x))
    plot_signal(x, y)

def plot_signal(x, y, z=None):
    figure, ax = plt.subplots()
    lc = colorline(x, y, z)
    plt.xlim(0, 1200)
    plt.ylim(y.min(), y.max())
    mng = plt.get_current_fig_manager()
    figure.set_size_inches(15, 6)

def colorline(
        x, y, z=None, cmap='RdYlGn', norm=plt.Normalize(0.0, 1.0),
        linewidth=2, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """


    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #print(segments)
    return segments
