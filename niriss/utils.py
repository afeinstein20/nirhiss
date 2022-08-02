import numpy as np
from astropy.table import Table


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

    tab = Table(names=['wave', 'wave_err', 'dppm', 'dppm_error', 'order'],
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
