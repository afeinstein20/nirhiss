import os
import numpy as np
from astropy.io import fits

from .background_utils import bkg_sub
from .masking import data_quality_mask

__all__ = ['create_F277W_bkg']

def create_F277W_bkg(filename, path=None, save=False, output_path=None,
                     output_filename='f277w_background.npy'):
    """
    A wrapper class for creating a model of the 0th order contaminants
    which are present in the F277W filter images. The sole return of
    this class is the background model.

    Parameters
    ----------
    filename : str
       Name of the F277W filter file.
    path : str, optional
       The full path to where the F277W filter file is stored. Default is
       `None`. If `None`, will search the current working directory for the
       file.
    output_path : str, optional
       The full path to where the F277W background model should be saved.
       Default is `None`. If `None`, will save the background to the current
       working directory.
    output_filename : str, optional
       The file name to save the background to. Default is
       `f277w_background.npy`. If not the default filename, the filename
       should end in `.npy`.

    Returns
    -------
    bkg : np.ndarray
       Isolated 0th order sources from the F277W filter images.
    """
    self.filename = filename

    if path is None:
        path = '.'
    self.path = path

    hdu = fits.open(os.path.join(self.path, self.filename))
    data = np.copy(hdu[1].data)            # pulls the data from the FITS file
    dq = data_quality_mask(hdu[2].data)    # creates bad DQ masks
    dq = np.nanmedian(dq, axis=0)

    median = np.nanmedian(data, axis=0)    # creates median integration

    # creates rough mask around the overlapping region
    mask = np.zeros(med.shape, dtype=bool)
    mask[:150, :600] = 1
    med_bkg = median*~mask

    # masks everything that isn't a strong outlier (i.e. 0th order sources)
    mask2 = med_bkg < np.nanmedian(med_bkg)*1.5

    # Creates 2D background to remove extreneous background noise from the
    #   F277W frames
    b, _ = bkg_sub(3,2), (2,2), med_bkg, ~mask2)

    # The final background frame
    bkg = (med_bkg-b2)*dq

    if save:
        if output_path is None:
            output_path = '.'

        np.save(os.path.join(output_path, output_filename))

    hdu.close()

    return bkg
