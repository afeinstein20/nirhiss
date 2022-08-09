import numpy as np
from tqdm import tqdm
from scipy.special import erf
from astropy.modeling.models import Gaussian1D, custom_model
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.convolution import Box1DKernel, convolve
from astropy.stats import sigma_clip

__all__ = ['gauss_removal', 'time_removal']

def skewed_gaussian(x, eta=0, omega=1, alpha=0, scale=1):
    """A skewed Gaussian model.

    Parameters
    ----------
    x : ndarray
        The values at which to evaluate the skewed Gaussian.
    eta : float; optional
        The Gaussian mean. Defaults to 0.
    omega : float, optional
        The skewed normal scale. Defaults to 1.
    alpha : float, optional
        The skewed normal shape. Defaults to 0.
    scale : float, optional
        A multiplier for the skewed normal. Defaults to 1.

    Returns
    -------
    ndarray
        The skewed Gaussian model evaluated at positions x.
    """
    t = alpha*(x-eta)/omega
    Psi = 0.5*(1+erf(t/np.sqrt(2)))
    psi = 2/(omega*np.sqrt(2*np.pi))*np.exp(-(x-eta)**2/(2*omega**2))
    return (psi * Psi)*scale


def gauss_removal(img, mask, linspace, where='bkg'):
    """An additional step to remove cosmic rays.

    This fits a Gaussian to the background (or a skewed Gaussian to the
    orders) and masks data points which are above a certain sigma.

    Parameters
    ----------
    img : np.ndarray
       Single exposure image.
    mask : np.ndarray
       An approximate mask for the orders.
    linspace : array
       Sets the lower and upper bin bounds for the
       pixel values. Should be of length = 2.
    where : str; optional
       Sets where the mask is covering. Default is `bkg`.
       Other option is `order`.

    Returns
    -------
    img : np.ndarray
       The same input image, now masked for newly identified
       outliers.
    """
    n, bins = np.histogram((img*mask).flatten(),
                           bins=np.linspace(linspace[0], linspace[1], 100))
    bincenters = (bins[1:]+bins[:-1])/2

    if where == 'bkg':
        g = Gaussian1D(mean=0, amplitude=100, stddev=10)
        rmv = np.where(np.abs(bincenters) <= 5)[0]
    elif where == 'order':
        GaussianSkewed = custom_model(skewed_gaussian)
        g = GaussianSkewed(eta=0, omega=20, alpha=4, scale=100)
        rmv = np.where(np.abs(bincenters) == 0)[0]

    # finds bin centers and removes bincenter = 0 (because this bin
    #   seems to be enormous and we don't want to skew the best-fit
    bincenters, n = np.delete(bincenters, rmv), np.delete(n, rmv)

    # fit the model to the histogram bins
    fitter = LevMarLSQFitter()
    gfit = fitter(g, bincenters, n)

    if where == 'bkg':
        xcr, ycr = np.where(np.abs(img * mask) >= gfit.mean + 2 * gfit.stddev)
    elif where == 'order':
        xcr, ycr = np.where(img * mask <= gfit.eta-1*gfit.omega)

    # returns an image that is nan-masked
    img[xcr, ycr] = np.nan
    return img


def time_removal(img, sigma=5, testing=False):
    """
    Removing cosmic rays in the time direction. This is meant as a
    first pass, not a final routine for cosmic ray removal.

    Parameters
    ----------
    img : np.ndarray
       The array of images (e.g. data from the DataClass()).
    sigma : float, optional
       The sigma outlier by which to mask pixels. Default=5.
    testing : bool
        If testing, only performs clipping.time_removal along 10 columns
        instead of the whole image.
    """
    cr_mask = np.zeros(img.shape)

    if not testing:
        y = img.shape[1]
    else:
        # If testing, only performs clipping.time_removal along 10 columns
        # instead of the whole image.
        y = 10

    for x in tqdm(range(y)):
        for y in range(img.shape[2]):
            dat = np.copy(img[:, x, y])
            ind = np.where((dat >= np.nanmedian(dat)+sigma*np.nanstd(dat)) |
                           (dat <= np.nanmedian(dat)-sigma*np.nanstd(dat)))
            ind = ind[0]
            if len(ind) > 0:
                cr_mask[ind, x, y] = 1.0

            # Checks for extreme differences in values
            diff = np.abs(np.diff(dat))
            ind = np.where(diff > sigma)[0]
            if len(ind) > 0:
                cr_mask[ind, x, y] = 1.0

    return cr_mask
