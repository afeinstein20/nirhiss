import numpy as np


def chromatic_writer(filename, time, wavelength, flux, var):
    # Writes numpy files to read into chromatic
    np.save(filename+'.npy',
            [time, wavelength, flux, var])
    return
