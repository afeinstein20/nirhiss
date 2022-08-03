import time
import h5py
import os, sys
import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import astraeus.xarrayIO as xrio

from scipy.io import wavfile
from scipy.interpolate import interp2d, interp1d
from scipy.interpolate import NearestNDInterpolator

import ccdproc as ccdp
from astropy import units
from astropy.nddata import CCDData
from tqdm import tqdm_notebook
from scipy.interpolate import interp1d
from astropy.table import Table

from .utils import bin_at_resolution

from matplotlib.colors import ListedColormap
pmap = ListedColormap(np.load('/Users/belugawhale/parula_data.npy'))
colors = np.load('/Users/belugawhale/parula_colors.npy')

COLOR = 'k'
plt.rcParams['font.size'] = 20
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR

plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.major.size']  = 10 #12
plt.rcParams['ytick.major.size']  = 10 #12

plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['xtick.minor.size']  = 6
plt.rcParams['ytick.minor.size']  = 6

plt.rcParams['axes.linewidth'] = 3

plt.rcParams['font.size'] = 20
plt.rcParams['text.color'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['axes.edgecolor'] = COLOR
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['legend.facecolor'] = 'none'

__all__ = ['stacked_transits', 'transmission_spectrum']

def stacked_transits(time, wavelength, flux, variance,
                     centers=np.flip(np.linspace(0.85, 2.55, 15)),
                     offset=0.05, offset_delta=0.005, figsize=(8,14),
                     text=True, time_ind=230, linestyle=''):
    global colors
    wave_offset = 0.5

    x = 0

    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor('w')
    rms = np.zeros(len(centers))
    grid = np.zeros((len(centers), len(flux)))

    for i, center in enumerate(centers):
        q = np.where((wavelength>=center-wave_offset) &
                     (wavelength<=center+wave_offset) &
                     (np.nansum(variance,axis=0) < 1e7))[0]

        spec = np.nansum(flux[:,q],axis=1)/1e6
        yerr = np.sqrt(np.nansum(variance[:,q]**2,axis=1))/1e6

        rms[i] = np.sqrt(np.nansum(spec[:100]**2)/100)

        yerr /= np.nanmedian(spec)
        spec /= np.nanmedian(spec)

        ax.errorbar(time,
                    spec-offset,
                    yerr=yerr, linestyle=linestyle, c=colors[x],
                    marker='.', label=np.round(center,2))
        grid[i] = spec



        if text:
            ax.text(x=time[time_ind], y=np.nanmedian(spec[210:])-offset+0.001,
                    s='{} $\mu$m'.format(np.round(center,2)),
                    fontsize=16)

        offset -= offset_delta
        x+=int(256/len(centers)-1)
    return fig, rms, grid

def plot_type_scatter(w, d, we, de, ax, kwargs):
    """ Helper function for scatter-type plots. """
    ax.errorbar(w, d, xerr=we, yerr=de, marker='.', **kwargs)
    return

def plot_type_fill_between(w, d, de, ax, kwargs):
    """ Helper function for fill_between-type plots. """
    kwargs.pop('alpha')
    ax.plot(w, d, alpha=1, **kwargs)
    kwargs['lw'] = 0
    kwargs['label'] = ''
    ax.fill_between(w, d-de, d+de, alpha=0.4, **kwargs)
    return

def transmission_spectrum(wavelength, depth, wave_err, depth_err,
                          plot_type='scatter', ax=None,
                          **kwargs):
    """
    Plots a given transmission spectrum.

    Parameters
    ----------
    wavelength : np.ndarray
       Array of wavelength data.
    depth : np.ndarray
       Array of measured transit depths.
    wave_err : np.ndarray
       Errors or binsize of each wavelength the transit depth was evaluated
       over.
    depth_err : np.ndarray
       Errors on the measured transit depth.
    plot_type : str, optional
       The way to plot the transmission spectrum. Default is 'scatter' (will
       plot each measured depth as an individual point). Other option is
       'fill_between' (plots a line with shading for the depth error).
    ax : matplotlib.axes._subplots.AxesSubplot, optional
       Subplot to plot the transmission spectrum on. Default is None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14,4))

    if plot_type.lower() == 'scatter':
        plot_type_scatter(wavelength, depth, wave_err, depth_err,
                          ax, **kwargs)

    elif plot_type.lower() == 'fill_between':
        plot_type_fill_between(wavelength, depth, depth_err, ax,
                               **kwargs)
    else:
        return('plot_type not implemented. Please select between "scatter"\
                and "fill_between".')

    return fig


def ers_transmission_spectra(table, order, color, label, ax, alpha=0.4, lw=3,
                             upper_lim2=0.85, plot_type='fill_between',
                             binned=True, R=100):
    """
    Plotting all of the beautiful transmission spectra from the ERS program.

    Parameters
    ----------
    table : astropy.table.Table
       Table with the wavelength (`wave`), wavelength error (`wave_err`),
       transit depth (`dppm`), transit depth error (`dppm_err`), and order
       (`order`) as columns.
    order : int
       Which NIRISS order to plot. Options are 1 and 2.
    color : str
       Color to plot the transmission spectrum in.
    label : str
       Line label for the plot legend.
    ax : matplotlib.axes._subplots.AxesSubplot
       Subplot to plot the transmission spectrum on.
    alpha : float, optional
       How transparent to make the fill_between shading. Default is 0.4.
    lw : float, optional
       How thick to make the central line. Default is 3.
    upper_lim2 : float, optional
       The longest wavelength to evaluate for Order 2. Default is 0.85.
    plot_type : str, optional
       The way to plot the transmission spectrum. Default is 'fill_between'
       (plots a line with shading for the depth error). Other option is
       'scatter' (will plot each measured depth as an individual point).
    binned : bool, optional
       Whether or not to bin the transmission spectrum to a given resolution
       before plotting. Default is True.
    R : int, optional
       The resolution to bin the transmission spectrum to. Default is 100.
       Recommended R = 100 for the Order 1 and R = 50 for Order 2.
    """
    # Create masks for each order from the table (it's easier to plot the
    #   orders separately)
    if order==1:
        q = table['order'] == order
    elif order==2:
        q = (table['order']==order) & (table['wave']<upper_lim2)
    else:
        return('order must equal 1 or 2.')

    # Bins the spectrum if binned == True
    if binned:
        out = bin_at_resolution(table['wave'][q], table['dppm'][q], R=R)
    else:
        out = [table['wave'], table['dppm'], table['dppm_err']]

    # Plots the spectrum
    if plot_type == 'fill_between':
        plot_type_fill_between(out[0], out[1], out[2], ax,
                               kwargs={'lw':lw, 'alpha':alpha,
                                       'zorder':10, 'color':color,
                                       'label':label})

    elif plot_type == 'scatter':
        plot_type_scatter(out[0], out[1], table['wave_error'], out[2],
                          ax=ax,
                          kwargs={'alpha':1, 'label':label,
                                  'zorder':10, 'color':color,
                                  'linestyle':''})
    else:
        return('plot_type not implemented. Please select between "scatter"\
                and "fill_between".')

    return
