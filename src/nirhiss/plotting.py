import time
import h5py
import os, sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .utils import bin_at_resolution

__all__ = ['stacked_transits', 'transmission_spectrum', 'transit_residuals',
           'ers_transmission_spectra']

def stacked_transits(time, wavelength, flux, variance, colors,
                     centers=np.flip(np.linspace(0.85, 2.55, 15)),
                     offset=0.05, offset_delta=0.005, figsize=(8,14),
                     text=True, time_ind=230, linestyle=''):

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
                             binned=True, R=100, ms=10):
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

    print(label, len(out[0]), R)

    # Plots the spectrum
    if plot_type == 'fill_between':
        plot_type_fill_between(out[0], out[1], out[2], ax,
                               kwargs={'lw':lw, 'alpha':alpha,
                                       'zorder':10, 'color':color,
                                       'label':label})

    elif plot_type == 'scatter':
        plot_type_scatter(out[0], out[1], np.zeros(len(out[1])), out[2],
                          ax=ax,
                          kwargs={'alpha':1, 'label':label,
                                  'zorder':10, 'color':color,
                                  'linestyle':'', 'ms':ms})
    else:
        return('plot_type not implemented. Please select between "scatter"\
                and "fill_between".')

    return

def transit_residuals(time, flux, flux_err, residuals, residuals_err, color,
                      model=None, ax=None, size="20%", pad=0, index1=0,
                      index2=0, resid_lims=[-1000,1000], xlim=[-3,3],
                      lc_lims=[0.975, 1.001], labelx=False,
                      xlabel=None):
    """
    Helper function to create subplot with a transit and the residuals
    underneath.

    Parameters
    ----------
    time : np.ndarray
       Array of times.
    flux : np.ndarray
       Array of fluxes.
    flux_err : np.ndarray
       Array of flux errors.
    residuals : np.ndarray
       Array of residuals between the light curve and best-fit model.
    residuals_err : np.ndarray
       Array of errors on the residuals.
    color : str
       What color to plot the data in.
    model : np.ndarray, optional
       Best-fit transit model to overplot. Default is None (i.e. the model
       will not be overplotted).
    ax : matplotlib.axes._subplots.AxesSubplot, optional
       The subplot to plot the data on. Default is None (i.e. will create its
       own figure).
    size : str, optional
       The percent split between the main light curve and the residuals. Default
       is '20%'.
    pad : float, optional
       The amount of padding between the light curve and residuals subpanels.
       Default is 0.
    index1 : int, optional
       If looping through several columns, this will help with setting/removing
       axes labels and ticklabels. Default is 0. Would be "i" in the first loop.
    index2 : int, optional
       If looping through several rows, this will help with setting/removing
       axes labels and ticklabels. Default is 0. Would be "j" in the second loop.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))

    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size=size, pad=pad)
    ax.figure.add_axes(ax2)

    ax.errorbar(time, flux, yerr=flux_err, marker='.', linestyle='',
                color=color)

    if model is not None:
        ax.plot(time, model, 'k', zorder=3)

    ax2.errorbar(time, residuals, yerr=residuals_err,
                 marker='.', linestyle='', color=color)

    if index1==0:
        if labelx == False:
            ax2.set_xticklabels([])
        if index2==0:
            ax2.set_ylabel('Residuals', fontsize=16)
    if index2 > 0:
        ax2.set_yticklabels([])
    if index1==1 and index2==0:
        ax2.set_ylabel('Residuals', fontsize=16)
    if index1==1 and index2==2:
        ax2.set_xlabel('Time from Mid-Transit [hrs]')

    ax.set_xticks([])
    ax2.set_ylim(resid_lims)
    ax.set_ylim(lc_lims)
    ax.set_xlim(xlim)
    ax2.set_xlim(xlim)

    return
