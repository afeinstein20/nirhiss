import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.interpolate import NearestNDInterpolator

__all__ = ['interpolating_col', 'data_quality_mask',
           'interpolating_image']

def interpolating_image(data, mask):
    """Uses `scipy.interpolate.NearestNDInterpolator` to
    fill in bad pixels/cosmic rays/whichever mask you
    decide to pass in.

    Parameters
    ----------
    data : np.ndarray
       Image frame.
    mask : np.ndarray
       Mask of integers or boolean values, where values
       greater than 0/True are bad pixels.

    Returns
    -------
    cleaned : np.ndarray
       Array of shape `data` which now has interpolated
       values over the bad masked pixels.
    """
    def interpolate(d, m):
        if m.dtype == bool:
            m = ~m
        else:
            m = m > 0

        x, y = np.meshgrid(np.arange(d.shape[1]),
                           np.arange(d.shape[0]))
        xym = np.vstack((np.ravel(x[m]), np.ravel(y[m]))).T
        data = np.ravel(d[m])
        interp = NearestNDInterpolator(xym, data)
        return interp(np.ravel(x), np.ravel(y)).reshape(d.shape)

    cleaned = np.zeros(data.shape)

    if len(data.shape) == 3:
        for i in range(len(data)):
            cleaned[i] = interpolate(data[i], mask[i])
    else:
        cleaned = interpolate(data, mask)
    return cleaned

def interpolating_col(data, mask, reg=2):
    """
    Fills in masked pixel values with either a median value from
    surrounding pixels along the column.

    Parameters
    ----------
    data : np.ndarray
       Image frame.
    mask : np.ndarray
       Mask of integers, where values greater than 0 are bad
       pixels.
    reg : int, optional
       The number of pixels along the row to interpolate over.
       Default is 2.

    Returns
    -------
    interp : np.ndarray
       Image where the bad pixels are filled in with the
       appropriate values. Should return the same shape
       as `data`.
    """
    newdata = np.zeros(data.shape)
    for i in tqdm(range(data.shape[0])):
        for col in range(data.shape[2]):
            good = np.where(mask[i,:,col]==True)[0] # finds the good data points
            bad = np.where(mask[i,:,col]==False)[0] # finds the bad data points

            try: # doesn't handle the NaN cols/rows well
                interp = interp1d(np.arange(data.shape[1])[good],
                                  data[i,:,col][good])
                new = np.copy(data[i,:,col])

                for b in bad:
                    try:
                        new[b] = interp(b)
                    except ValueError: # doesn't handle the NaN cols/rows well
                        new[b] = 0.0
                newdata[i,:,col] = new

            except ValueError:
                newdata[i,:,col] = np.zeros(data.shape[1])
    return newdata


def data_quality_mask(dq):
    """
    Masks all pixels that are neither normal (value != 0)
    or reference pixels (value == 2147483648).

    Parameters
    ----------
    dq : np.array
       Array of data quality values.

    Returns
    -------
    dq_mask : np.array
       Boolean array masking where the bad pixels are.
    """
    dq_mask = np.ones(dq.shape, dtype=bool)

    if len(dq.shape) == 3:
        for i in range(len(dq)):
            x, y = np.where((dq[i] == 0) | (dq[i] == 2147483648))
            dq_mask[i, x, y] = False

    elif len(dq.shape) == 2:
        x, y = np.where((dq[i] == 0) | (dq[i] == 2147483648))
        dq_mask[x, y] = False
    else:
        return('Data quality array should be 2D.')

    return ~dq_mask
