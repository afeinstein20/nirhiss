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

__all__ = ['stacked_transits']

def stacked_transits(time, wavelength, flux, variance,
              centers=np.flip(np.linspace(0.85, 2.55, 15))):
    global colors
    wave_offset = 0.5

    x = 0

    fig, ax = plt.subplots(figsize=(8, 14))
    fig.set_facecolor('w')
    offset=0.05

    for center in centers:
        q = np.where((wavelength>=center-wave_offset) &
                     (wavelength<=center+wave_offset) &
                     (np.nansum(variance,axis=0) < 1e7))[0]

        spec = np.nansum(flux[:,q],axis=1)/1e6
        yerr = np.sqrt(np.nansum(variance[:,q]**2,axis=1))/1e6

        yerr /= np.nanmedian(spec[210:])
        spec /= np.nanmedian(spec[210:])

        ax.errorbar(time,
                    spec/np.nanmedian(spec[210:])-offset,
                    yerr=yerr, linestyle='', c=colors[x],
                    marker='.', label=np.round(center,2))

        ax.text(x=time[230], y=np.nanmedian(spec[210:])-offset+0.001,
                s='{} $\mu$m'.format(np.round(center,2)),
                fontsize=16)

        offset -= 0.005
        x+=15
    return fig
