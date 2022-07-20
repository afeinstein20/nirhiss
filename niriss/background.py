import numpy as np
from tqdm import tqdm
import ccdproc as ccdp
from astropy import units
import multiprocessing as mp
import matplotlib.pyplot as plt
from astropy.nddata import CCDData
from astropy.stats import SigmaClip
from scipy.signal import savgol_filter
from photutils import (MMMBackground, MedianBackground,
                       Background2D, MeanBackground)
import os

from . import clipping

__all__ = ['fitbg3', 'bkg_sub', 'simple_bkg']

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


def fitbg3(data, order_mask, readnoise=11,
           sigclip=[4, 4, 4], box=[(10, 2)],
           filter_size=(1, 1), sigma=5,
           bkg_estimator=['median'],
           isplots=0, testing=False, inclass=False):
    """
    Fit sky background with out-of-spectra data. Optimized to remove
    the 1/f noise in the NIRISS spectra (works in the y-direction).

    Parameters
    ----------
    data : Xarray Dataset, np.ndarray
        The Dataset object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    order_mask : np.ndarray
       Array masking where the three NIRISS orders are located.
    readnoise : float, optional
       An estimation of the readnoise of the detector.
       Default is 11.
    sigclip : list, array; optional
       A list or array of len(n_iiters) corresponding to the
       sigma-level which should be clipped in the cosmic
       ray removal routine. Default is [4, 4, 4].
    box : list, array; optional
       The box size along each axis. Box has two elements: (ny, nx). For best
       results, the box shape should be chosen such that the data are covered
       by an integer number of boxes in both dimensions. Default is (10, 2).
    filter_size : list, array; optional
       The window size of the 2D median filter to apply to the low-resolution
       background map. Filter_size has two elements: (ny, nx). A filter size of
       1 (or (1,1)) means no filtering. Default is (1, 1).
    sigma : float; optional
       Sigma to remove above. Default is 5.
    bkg_estimator : list, array; optional
       The value which to approximate the background values as. Options are
       "mean", "median", or "MMMBackground". Default is ['median', ].
    isplots : int, optional
       The level of output plots to display. Default is 0
       (no plots).
    testing : bool, optional
       Evaluates the background across fewer integrations to test and
       save computational time. Default is False.
    inclass : bool, optional
        If False (analyzing NIRISS data using s3_reduce_niriss), copy the
        data.flux array. Default is False.

    Returns
    -------
    bkg : np.ndarray
       The fitted background array.
    bkg_var : np.ndarray
       Errors on the fitted backgrouns.
    rm_crs : np.ndarray
       Array of masked bad pixels.
    """
    if inclass is False:
        data = np.copy(data)

    # Removes cosmic rays
    # Loops through niters cycles to make sure all pesky
    #    cosmic rays are trashed
    rm_crs = np.zeros(data.shape)
    bkg = np.zeros(data.shape)
    bkg_var = np.zeros(data.shape)

    # Does a first pass at CR removal in the time-direction
    #first_pass = clipping.time_removal(data, sigma=sigclip[0],
#                                       testing=testing)
    first_pass = np.copy(data)
    # Loops through and removes more cosimc rays
    for i in tqdm(range(len(data))):

        mask = np.array(first_pass[i], dtype=bool)
        ccd = CCDData((data[i]), unit=units.electron)

        # Second pass at removing cosmic rays, with ccdproc
        for n in range(len(sigclip)):
            m1 = ccdp.cosmicray_lacosmic(ccd, readnoise=readnoise,
                                         sigclip=sigclip[n])
            ccd = CCDData(m1.data * units.electron)
            mask[m1.mask] += True

        rm_crs[i] = m1.data
        rm_crs[i][mask >= 1] = np.nan

        v = np.zeros((len(bkg_estimator),
                      rm_crs[i].shape[0],
                      rm_crs[i].shape[1]))
        # Fits a 2D background (with the orders masked)
        for j in range(len(bkg_estimator)):
            b1, b1_err = bkg_sub(rm_crs[i],
                                 order_mask,
                                 bkg_estimator=bkg_estimator[j],
                                 sigma=sigma, box=box[j],
                                 filter_size=filter_size[j])
            bkg[i] += b1
            v[j] = b1_err

            if box[j][0] < 5 or box[j][1] < 5:
                b1 *= order_mask

        bkg_var[i] = np.nansum(v, axis=0)

    return bkg, bkg_var, rm_crs
