import numpy as np
from tqdm import tqdm
import ccdproc as ccdp
from astropy import units
import matplotlib.pyplot as plt
from astropy.nddata import CCDData
from astropy.stats import SigmaClip
from scipy.signal import savgol_filter
from photutils import (MMMBackground, MedianBackground,
                       Background2D, MeanBackground)
import os

from . import clipping

__all__ = ['bkg_sub', 'simple_bkg']

def simple_bkg(data, mask, window_length=7, polyorder=2, mode='nearest'):
    """
    Completes a simple Savitsky Golay filter to the background to
    remove the 1/f noise.

    Parameters
    ----------
    data : np.ndarray
    mask : np.ndarray
    window_length : int, optional
       The length of the filter window (i.e. the number of coefficients).
       window_length must be an odd integer. Default is 7.
    polyorder : int, optional
       The order of the polynomial used to fit the samples. polyorder must be
       less than window_length. Default is 2.
    mode : str, optional
       Must be 'mirror', 'constant', 'nearest', 'wrap', or 'interp'. This
       determines the type of extension to use for the padded signal to which
       the filter is applied. Default is 'nearest'.

    Returns
    -------
    bkg : np.ndarray
       Array of background pixel values.
    """
    bkg = savgol_filter((data*mask), window_length=window_length,
                        polyorder=polyorder, axis=1,
                        mode=mode)
    return bkg

def bkg_sub(img, mask, sigma=5, bkg_estimator='median',
            box=(10, 2), filter_size=(1, 1)):
    """Completes a step for fitting a 2D background model.

    Parameters
    ----------
    img : np.ndarray
       Single exposure frame.
    mask : np.ndarray
       Mask to remove the orders.
    sigma : float; optional
       Sigma to remove above. Default is 5.
    bkg_estimator : str; optional
       Which type of 2D background model to use.
       Default is `median`.
    box : tuple; optional
       Box size by which to smooth over. Default
       is (10,2) --> prioritizes smoothing by
       column.
    filter_size : tuple; optional
       The window size of the 2D filter to apply to the
       low-resolution background map. Default is (1,1).

    Returns
    -------
    background : np.ndarray
       The modeled background image.
    background_error : np.ndarray
       Error estimation on the background fitting.
    """
    sigma_clip = SigmaClip(sigma=sigma)        # Sigma clipping mask

    # These are different ways to calculate the background noise
    if bkg_estimator.lower() == 'mmmbackground':  # 3*mean + 2*median
        bkg = MMMBackground()
    elif bkg_estimator.lower() == 'median':      # median background
        bkg = MedianBackground()
    elif bkg_estimator.lower() == 'mean':        # mean background
        bkg = MeanBackground()

    b = Background2D(img, box,
                     filter_size=filter_size,  # window size of the filter
                     bkg_estimator=bkg,       # how to calculate the bkg
                     sigma_clip=sigma_clip,   # performs sigma clipping
                     fill_value=0.0,          # used to fill masked pixels
                     mask=mask)               # masks the orders

    return b.background, np.sqrt(b.background_rms)
