import numpy as np
from astropy.table import Table

__all__ = ['bin_at_resolution', 'chromatic_writer',
           'lpc_to_format']

def chromatic_writer(filename, time, wavelength, flux, var):
    # Writes numpy files to read into chromatic
    np.save(filename+'.npy',
            [time, wavelength, flux, var])
    return


def lpc_to_format(filename1, filename2, filename):
    """
    Takes the outputs of exotep and puts it in the agreed upon format.

    Parameters
    ----------
    filename1 : str
       The filename (+ path) for the output csv for NIRISS order 1.
    filename2 : str
       The filename (+ path) for the output csv for NIRISS order 2.
    filename : str
       The output filename to save the new table to.

    Returns
    -------
    tab : astropy.table.Table
    """
    table1 = Table.read(filename1, format='csv')
    table2 = Table.read(filename2, format='csv')

    tab = Table(names=['wave', 'wave_err', 'dppm', 'dppm_err', 'order'],
                dtype=[np.float64, np.float64, np.float64, np.float64, int])

    for i in range(len(table1)):
        row = [table1['wave'][i], table1['waveMin'][i],
               table1['yval'][i], table1['yerrLow'][i], 1]
        tab.add_row(row)

    short = table2[table2['wave'] < 0.9]
    for i in range(len(short)):
        row = [short['wave'][i], short['waveMin'][i],
               short['yval'][i], short['yerrLow'][i], 2]
        tab.add_row(row)

    tab.write(filename, format='csv', overwrite=True)

    return tab

def bin_at_resolution(wavelengths, depths, R = 100, method = 'median'):
    """
    Function that bins input wavelengths and transit depths (or any other observable, like flux) to a given
    resolution `R`. Useful for binning transit depths down to a target resolution on a transit spectrum.
    Parameters
    ----------
    wavelengths : np.array
        Array of wavelengths

    depths : np.array
        Array of depths at each wavelength.
    R : int
        Target resolution at which to bin (default is 100)
    method : string
        'mean' will calculate resolution via the mean --- 'median' via the median resolution of all points
        in a bin.
    Returns
    -------
    wout : np.array
        Wavelength of the given bin at resolution R.
    dout : np.array
        Depth of the bin.
    derrout : np.array
        Error on depth of the bin.

    """

    # Sort wavelengths from lowest to highest:
    idx = np.argsort(wavelengths)

    ww = wavelengths[idx]
    dd = depths[idx]

    # Prepare output arrays:
    wout, dout, derrout = np.array([]), np.array([]), np.array([])

    oncall = False

    # Loop over all (ordered) wavelengths:
    for i in range(len(ww)):

        if not oncall:

            # If we are in a given bin, initialize it:
            current_wavs = np.array([ww[i]])
            current_depths = np.array(dd[i])
            oncall = True

        else:

            # On a given bin, append next wavelength/depth:
            current_wavs = np.append(current_wavs, ww[i])
            current_depths = np.append(current_depths, dd[i])

            # Calculate current mean R:
            current_R = np.mean(current_wavs) / np.abs(current_wavs[0] - current_wavs[-1])

            # If the current set of wavs/depths is below or at the target resolution, stop and move to next bin:
            if current_R <= R:

                wout = np.append(wout, np.mean(current_wavs))
                dout = np.append(dout, np.mean(current_depths))
                derrout = np.append(derrout, np.sqrt(np.var(current_depths)) / np.sqrt(len(current_depths)))

                oncall = False

    return wout, dout, derrout
