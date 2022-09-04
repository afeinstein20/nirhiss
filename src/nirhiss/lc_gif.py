import os, sys
import imageio
import matplotlib
import numpy as np
from pylab import *
from astropy import units
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

__all__ = ['binning', 'animate', 'create_gif']

def binning(time, flux, binsize=10):
    """ Bins the light curves if set to True in main function.
    """
    nw, nf = np.array([]), np.array([])

    for i in np.arange(int(binsize/2), int(len(flux)-binsize/2), binsize, dtype=int):
        idx = np.arange(int(i-binsize/2), int(i+binsize/2), 1, dtype=int)
        nw  = np.append(nw, time[i])
        nf  = np.append(nf, np.nanmedian(flux[idx]))

    return nw, nf

def setup_figure():
    """ Sets up the layout of the figure.
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16,4),
                                   gridspec_kw={'width_ratios':[1,2]})
    fig.set_facecolor('w')
    return fig, ax1, ax2

def setup_lims(ax1, ax2, xlim1, ylim1, xlim2, ylim2):
    """ Sets the subplots (x,y) limits.
    """
    ax1.set_xlim(xlim1)
    ax2.set_xlim(xlim2)
    ax2.set_ylim(ylim2)
    return

def setup_axes_labels(ax1, ax2):
    """ Sets up axes labels.
    """
    ax1.set_xlabel('wavelength [$\mu$m]')
    ax1.set_ylabel('transit depth [ppm]')
    ax2.set_xlabel('time [days]')
    ax2.set_ylabel('normalized flux')
    return

def setup_legends(ax1, ax2):
    """ Creates the legend for each subplot.
    """
    for ax in [ax1, ax2]:
        leg = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                        ncol=2, mode="expand", borderaxespad=0.,
                        fontsize=18)
    # Only for ax2
    for legobj in leg.legendHandles:
        legobj.set_linewidth(8.0)
    return

def animate(lightcurves, tspectra, colors, cmap_color, labels,
            bin_center, bin_width, xlim1=[0.5, 3], ylim1=[20000,23000],
            xlim2=[0,0.28], ylim2=[0.965, 1.015], outputdir='.'):
    """
    Creates each frame for the GIF. Plots the transmission spectrum on the left
    and the light curve on the right.

    Parameters
    ----------
    lightcurves : np.ndarray
        Array of (N x 2) arrays, where N is the number of pipelines/instruments
        to plot. `lightcurves` should be formatted as `[ [time, flux] ]`.
    tspectra : np.ndarray
       Array of (N x 4) arrays, where N is the number of pipelines/instruments
       to plot. `tspectra` should be formatted as `[ [wavelength, dppm, wave_err,
       dppm_err]` ].
    colors : np.array
       A list of colors to plot each pipeline/instrument. Should be length `N`.
    cmap_color : str
       A color to plot the vertical bar moving through wavelength space over the
       transmission spectrum.
    labels : np.array
       The label for each pipeline/instrument. Should be length `N`.
    bin_center : np.array
       The wavelength value of the bin center.
    bin_width : np.array
       The wavelength width around the bin center.
    xlim1 : np.array, optional
       x-limits for the transmission spectrum subplot. Default is `[0.5, 3]`.
    ylim1 : np.array, optional
       y-limits for the transmission spectrum subplot Default is `[20000,23000]`
       (in parts-per-million units).
    xlim2 : np.array, optional
       x-limits for the light curve subplot. Default is `[0,0.28]`.
    ylim2 : np.array, optional
       y-limits for the light curve subplot. Default is `[0.965, 1.015]`.
    outputdir : str, optional
       Where to save the current figures. Default is your current working
       directory.
    """
    fig, ax1, ax2 = setup_figure()

    for i in range(len(colors)):

        # Plots the transmission spectra
        ax1.errorbar(tspectra[i][0], tspectra[i][1],
                     xerr=tspectra[i][2], yerr=tspectra[i][3],
                     color=colors[i], linestyle='', marker='o',
                     animated=True)

        # Plots the light curves
        ax2.scatter(lightcurves[i][0],
                    lightcurves[i][1]/np.nanmedian(lightcurves[i][1]),
                    label=labels[i],
                    color=colors[i], animated=True)

    # Highlights the wavelength on the transmission spectrum plot
    ax1.axvspan(bin_center-bin_width, bin_center+bin_width,
                alpha=0.4, color=cmap_color,
                label='{} $\pm$ {} $\mu$m'.format(np.round(bin_center,3),
                                                  np.round(bin_width, 3)),
                animated=True)

    setup_legends(ax1, ax2)
    setup_axes_labels(ax1, ax2)
    setup_lims(ax1, ax2, xlim1, ylim1, xlim2, ylim2)
    plt.subplots_adjust(wspace=0.25)
    plt.savefig(os.path.join(outputdir, 'figure_{:.3f}.png'.format(bin_center)),
                dpi=150, rasterize=True, bbox_inches='tight')
    plt.close()
    return

def create_gif(outputdir, filename, fps=15):

    filenames = np.sort([os.path.join(outputdir, i) for i in os.listdir(outputdir)
                         if i.endswith('.png')])
    images = []
    i = 0
    for filename in filenames:
        images.append(imageio.imread(filename))
        i += 1
        os.remove(filename)
    imageio.mimsave(os.path.join(outputdir, 'lcs.gif'), images, fps=fps)
