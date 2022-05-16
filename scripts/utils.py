# %% imports
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import seaborn as sns
import os
import numpy as np
sr2degsq = 3282.806350011744  # (u.sr).to(u.deg**2) = np.rad2deg(1)**2

# pathes
# '/Users/sdbykov/work/clustering_forecast/'
#rep_path = os.path.dirname(os.path.abspath(__file__))+'/'
rep_path = '/Users/sdbykov/work/forecast_clustering/' #set path to the root folder

path2res_forecast = rep_path + 'results/data/'
path2plots = rep_path + 'results/plots/'


# %% set RC parameters
# see https://matplotlib.org/stable/tutorials/introductory/customizing.html
rc = {
    "figure.figsize": [10, 10],
    "figure.dpi": 100,
    "savefig.dpi": 300,
    # fonts and text sizes
    # 'font.family': 'sans-serif',
    'font.family': 'Calibri',
    'font.sans-serif': 'Lucida Grande',
    'font.style': 'normal',
    "font.size": 25,
    "axes.labelsize": 25,
    "axes.titlesize": 25,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 15,

    # lines
    "axes.linewidth": 1.25,
    "lines.linewidth": 1.75,
    "patch.linewidth": 1,

    # grid
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.linestyle": "--",
    "grid.linewidth": 0.75,
    "grid.alpha": 0.75,

    # ticks
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.width": 1.25,
    "ytick.major.width": 1.25,
    "xtick.minor.width": 1,
    "ytick.minor.width": 1,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,

    # 'lines.marker': 'o',
    'lines.markeredgewidth': 1.5,
    "lines.markersize": 10,
    "lines.markeredgecolor": "k",
    'axes.titlelocation': 'left',
    "axes.formatter.limits": [-2, 2],
    "axes.formatter.use_mathtext": True,
    "axes.formatter.min_exponent": 2,
    'axes.formatter.useoffset': False,
    "figure.autolayout": False,
    "hist.bins": "auto",
    "scatter.edgecolors": "k",
}


def set_mpl(palette_name='pastel'):
    sns.set_palette(palette_name, color_codes=True)
    print(f'set palette to {palette_name}')
    matplotlib.rcParams.update(rc)
    print("set matplotlib rc")

set_mpl('pastel')



def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor=None,
                     edgecolor='k', alpha=0.4, plot_errorbars=False, lw=2.5,
                     xscale='log', yscale='log',
                     label=''):
    """ a simple function to make error as boxes on a plot"""

    
    if ax is None:
        fig, ax = plt.subplots()
    if xerror.ndim == 1:  # assume symmetric errors
        yerror = np.vstack((yerror, yerror))
        xerror = np.vstack((xerror, xerror))
    'https://matplotlib.org/stable/gallery/statistics/errorbars_and_boxes.html'
    # Loop over data points; create box from errors at each point
    errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())  # type: ignore
                  for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]
    _ = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror, color='k',
                    fmt='None', ecolor=edgecolor, alpha=0.5*plot_errorbars)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if facecolor is None:
        ax.plot(xdata, ydata,  '-', alpha=alpha*1.2,
                label=label, lw=lw)  # cycle colors
        color = ax.get_lines()[-1].get_color()
        pc = PatchCollection(errorboxes, facecolor=color, alpha=alpha,
                             edgecolor=edgecolor, lw=0.5)
    else:
        ax.plot(xdata, ydata,  '-', alpha=alpha,
                label=label, color=facecolor)  # cycle colors
        pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                             edgecolor=edgecolor, lw=0.5)
    ax.add_collection(pc)

    return ax
